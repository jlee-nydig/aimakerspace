"""Stone Ridge Investment Assistant — consolidated LangGraph Platform agent.

Architecture:
    START → input_guardrail →[pass]→ agent →[tools?]→ action → agent (loop)
                             [fail]→ END        [no tools]→ output_guardrail →[pass]→ helpfulness →[Y]→ END
                                                                              [fail]→ agent (retry)   [N]→ agent
"""
from __future__ import annotations

import hashlib
import logging
import os
from functools import lru_cache
from operator import itemgetter
from typing import Annotated, Any, Dict, List, Optional
from uuid import uuid4

import requests
import tiktoken
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_tavily import TavilySearch
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.tools import tool
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Constants & System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the Stone Ridge Investment Assistant — a knowledgeable \
AI that helps investors, analysts, and portfolio managers understand Stone Ridge \
Asset Management's investment philosophy, fund performance, and market outlook.

You have access to the Stone Ridge 2025 Investor Letter and various research tools. \
When answering questions:
- Ground your answers in the investor letter and retrieved context when available
- Use web search for current market data or recent news
- Use academic search for research papers on investment strategies
- Use X/Twitter tools to gauge market sentiment when relevant
- Be precise with financial figures and cite sources when possible
- If you don't know something, say so clearly

Key Stone Ridge themes: Bayesian investing, reinsurance, energy assets, \
bitcoin allocation, longtail risk, and alternative asset management."""

DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-20250514-v1:0"
EVAL_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# Investment-domain guardrails topics
VALID_TOPICS = [
    "investments", "portfolio management", "investor letters",
    "market analysis", "financial markets", "Stone Ridge",
    "asset management", "market sentiment", "economic outlook",
    "reinsurance", "energy assets", "bitcoin", "risk management",
    "fund performance", "alternative investments", "Bayesian investing",
]

INVALID_TOPICS = [
    "medical advice", "legal advice", "gambling",
    "explicit content", "political campaigning",
]

# ---------------------------------------------------------------------------
# 2. State Schema
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """State schema for the investment assistant graph."""
    messages: Annotated[List[BaseMessage], add_messages]
    guardrail_passed: Optional[bool]


# ---------------------------------------------------------------------------
# 3. Model Factories
# ---------------------------------------------------------------------------


def get_chat_model(
    model_name: str | None = None,
    *,
    temperature: float = 0,
    max_tokens: int | None = None,
) -> ChatAnthropic:
    """Return a configured ChatAnthropic instance."""
    name = model_name or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
    kwargs: Dict[str, Any] = {"model": name, "temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatAnthropic(**kwargs)


# ---------------------------------------------------------------------------
# 4. Caching
# ---------------------------------------------------------------------------


class CacheBackedEmbeddings:
    """Production cache-backed embeddings using OpenAI."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        cache_dir: str = "./cache/embeddings",
        batch_size: int = 32,
    ):
        from langchain.embeddings import CacheBackedEmbeddings as _LCCached
        from langchain.storage import LocalFileStore

        self.base_embeddings = OpenAIEmbeddings(model=model)
        safe_namespace = hashlib.md5(model.encode()).hexdigest()
        store = LocalFileStore(cache_dir)
        self.cached_embeddings = _LCCached.from_bytes_store(
            self.base_embeddings, store, namespace=safe_namespace, batch_size=batch_size
        )

    def get_embeddings(self):
        return self.cached_embeddings


def setup_llm_cache(cache_type: str = "memory", cache_path: str | None = None):
    """Set up LangChain LLM caching."""
    from langchain_core.caches import InMemoryCache
    from langchain_core.globals import set_llm_cache

    if cache_type == "memory":
        set_llm_cache(InMemoryCache())
    elif cache_type == "sqlite":
        from langchain_community.cache import SQLiteCache
        db_path = cache_path or "./cache/llm_cache.db"
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        set_llm_cache(SQLiteCache(database_path=db_path))


# ---------------------------------------------------------------------------
# 5. Production RAG
# ---------------------------------------------------------------------------


def _tiktoken_len(text: str) -> int:
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(text)
    return len(tokens)


class _RAGState(TypedDict):
    question: str
    context: List[Document]
    response: str


def _build_rag_graph(data_dir: str):
    """Build a minimal RAG graph over PDFs in *data_dir*."""
    try:
        directory_loader = DirectoryLoader(
            data_dir, glob="**/*.pdf", loader_cls=PyMuPDFLoader
        )
        documents = directory_loader.load()
    except Exception:
        documents = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750, chunk_overlap=0, length_function=_tiktoken_len
    )
    chunks = text_splitter.split_documents(documents) if documents else []

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    from langchain_qdrant import QdrantVectorStore
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="rag_collection",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    qdrant_vectorstore = QdrantVectorStore(
        client=client,
        collection_name="rag_collection",
        embedding=embedding_model,
    )
    if chunks:
        qdrant_vectorstore.add_documents(chunks)
    retriever = qdrant_vectorstore.as_retriever()

    human_template = (
        "\n#CONTEXT:\n{context}\n\nQUERY:\n{query}\n\n"
        "Use the provided context to answer the provided user query. "
        "Only use the provided context to answer the query. "
        'If you do not know the answer, or it\'s not contained in the provided context respond with "I don\'t know"'
    )
    chat_prompt = ChatPromptTemplate.from_messages([("human", human_template)])
    generator_llm = get_chat_model()

    def retrieve(state: _RAGState) -> _RAGState:
        retrieved_docs = retriever.invoke(state["question"]) if retriever else []
        return {"context": retrieved_docs}

    def generate(state: _RAGState) -> _RAGState:
        generator_chain = chat_prompt | generator_llm | StrOutputParser()
        response_text = generator_chain.invoke(
            {"query": state["question"], "context": state.get("context", [])}
        )
        return {"response": response_text}

    graph_builder = StateGraph(_RAGState)
    graph_builder = graph_builder.add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()


@lru_cache(maxsize=1)
def _get_rag_graph():
    data_dir = os.environ.get("RAG_DATA_DIR", "data")
    return _build_rag_graph(data_dir)


@tool
def retrieve_information(
    query: Annotated[str, "query to ask the retrieve information tool"],
):
    """Use Retrieval Augmented Generation to retrieve information from the Stone Ridge 2025 Investor Letter."""
    graph = _get_rag_graph()
    result = graph.invoke({"question": query})
    if isinstance(result, dict) and "response" in result:
        return result["response"]
    return result


# ---------------------------------------------------------------------------
# 6. X/Twitter Tools
# ---------------------------------------------------------------------------


def _x_bearer_token() -> str | None:
    return os.environ.get("X_BEARER_TOKEN")


@tool
def search_recent_posts(query: str, max_results: int = 20) -> str:
    """Search recent X/Twitter posts using the v2 API.
    Returns posts from the last 7 days matching the query.

    Args:
        query: The search query (e.g., 'Stone Ridge', '#bitcoin', 'reinsurance')
        max_results: Number of results to return (10-100, default 20)
    """
    token = _x_bearer_token()
    if not token:
        return "X_BEARER_TOKEN not configured — skipping X/Twitter search."

    url = "https://api.x.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "query": query,
        "max_results": min(max(max_results, 10), 100),
        "tweet.fields": "created_at,public_metrics,author_id,text",
        "expansions": "author_id",
        "user.fields": "name,username",
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        return f"X API request failed: {exc}"

    data = response.json()
    tweets = data.get("data", [])
    if not tweets:
        return "No posts found for this query."

    result_lines = [f"Found {len(tweets)} posts:\n"]
    for t in tweets:
        metrics = t.get("public_metrics", {})
        result_lines.append(
            f"[{t.get('created_at', 'unknown')[:10]}] "
            f"{t['text'][:200]}\n"
            f"  Likes: {metrics.get('like_count', 0)} | "
            f"Retweets: {metrics.get('retweet_count', 0)}"
        )
    return "\n\n".join(result_lines)


@tool
def get_user_posts(username: str, max_results: int = 20) -> str:
    """Get recent original posts (no retweets) from a specific X/Twitter user.

    Args:
        username: The X/Twitter handle without the @ sign (e.g., 'StoneRidgeAM')
        max_results: Number of results to return (10-100, default 20)
    """
    query = f"from:{username} -is:retweet"
    return search_recent_posts.invoke({"query": query, "max_results": max_results})


@tool
def get_post_by_id(post_id: str) -> str:
    """Retrieve a single X/Twitter post by its ID.

    Args:
        post_id: The numeric post/tweet ID
    """
    token = _x_bearer_token()
    if not token:
        return "X_BEARER_TOKEN not configured — skipping X/Twitter lookup."

    url = f"https://api.x.com/2/tweets/{post_id}"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "tweet.fields": "created_at,public_metrics,author_id,text",
        "expansions": "author_id",
        "user.fields": "name,username",
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        return f"X API request failed: {exc}"

    data = response.json()
    tweet = data.get("data", {})
    if not tweet:
        return "Post not found."

    metrics = tweet.get("public_metrics", {})
    return (
        f"[{tweet.get('created_at', 'unknown')[:10]}] "
        f"{tweet.get('text', '')}\n"
        f"  Likes: {metrics.get('like_count', 0)} | "
        f"Retweets: {metrics.get('retweet_count', 0)}"
    )


# ---------------------------------------------------------------------------
# 7. Tool Belt
# ---------------------------------------------------------------------------


def get_tool_belt() -> List:
    """Return the list of tools available to the agent."""
    tools: List = []

    # Tavily web search (optional)
    if os.environ.get("TAVILY_API_KEY"):
        tools.append(TavilySearch(max_results=5))

    # Arxiv academic search
    tools.append(ArxivQueryRun())

    # RAG retriever
    tools.append(retrieve_information)

    # X/Twitter tools (always included; they degrade gracefully without token)
    tools.extend([search_recent_posts, get_user_posts, get_post_by_id])

    return tools


# ---------------------------------------------------------------------------
# 8. Guardrails (lazy init)
# ---------------------------------------------------------------------------

_input_guard = None
_output_guard = None


def _get_input_guard():
    global _input_guard
    if _input_guard is not None:
        return _input_guard
    try:
        from guardrails import Guard
        from guardrails.hub import (
            DetectJailbreak,
            GuardrailsPII,
            ProfanityFree,
            RestrictToTopic,
        )

        guard = Guard()
        guard = guard.use(
            RestrictToTopic(
                valid_topics=VALID_TOPICS,
                invalid_topics=INVALID_TOPICS,
                disable_classifier=True,
                disable_llm=False,
                on_fail="exception",
            )
        )
        guard = guard.use(DetectJailbreak())
        guard = guard.use(
            GuardrailsPII(
                entities=["CREDIT_CARD", "SSN", "PHONE_NUMBER", "EMAIL_ADDRESS"],
                on_fail="fix",
            )
        )
        guard = guard.use(
            ProfanityFree(threshold=0.8, validation_method="sentence", on_fail="exception")
        )
        _input_guard = guard
        logger.info("Input guardrails configured")
        return guard
    except Exception as exc:
        logger.warning("Guardrails not available — running without input guards: %s", exc)
        return None


def _get_output_guard():
    global _output_guard
    if _output_guard is not None:
        return _output_guard
    try:
        from guardrails import Guard
        from guardrails.hub import GuardrailsPII, ProfanityFree

        guard = Guard()
        guard = guard.use(
            GuardrailsPII(
                entities=["CREDIT_CARD", "SSN", "PHONE_NUMBER", "EMAIL_ADDRESS"],
                on_fail="fix",
            )
        )
        guard = guard.use(
            ProfanityFree(threshold=0.8, validation_method="sentence", on_fail="exception")
        )
        _output_guard = guard
        logger.info("Output guardrails configured")
        return guard
    except Exception as exc:
        logger.warning("Guardrails not available — running without output guards: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 9. Graph Nodes
# ---------------------------------------------------------------------------


def _build_model_with_tools():
    model = get_chat_model()
    return model.bind_tools(get_tool_belt())


def input_guardrail(state: AgentState) -> Dict[str, Any]:
    """Validate user input with guardrails. Sets guardrail_passed flag."""
    guard = _get_input_guard()
    if guard is None:
        return {"guardrail_passed": True}

    last_message = state["messages"][-1]
    if not isinstance(last_message, HumanMessage):
        return {"guardrail_passed": True}

    try:
        result = guard.validate(last_message.content)
        passed = getattr(result, "validation_passed", True)
        if not passed:
            return {
                "guardrail_passed": False,
                "messages": [
                    AIMessage(
                        content="I'm sorry, but I can only help with investment-related topics "
                        "such as portfolio management, market analysis, and Stone Ridge funds. "
                        "Please rephrase your question."
                    )
                ],
            }
        return {"guardrail_passed": True}
    except Exception as exc:
        logger.warning("Input guardrail raised: %s", exc)
        return {
            "guardrail_passed": False,
            "messages": [
                AIMessage(
                    content="I'm sorry, but I can only help with investment-related topics "
                    "such as portfolio management, market analysis, and Stone Ridge funds. "
                    "Please rephrase your question."
                )
            ],
        }


def agent(state: AgentState) -> Dict[str, Any]:
    """Invoke the LLM with the accumulated messages."""
    model = _build_model_with_tools()
    messages = [HumanMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def output_guardrail(state: AgentState) -> Dict[str, Any]:
    """Validate agent output with guardrails."""
    guard = _get_output_guard()
    if guard is None:
        return {"guardrail_passed": True}

    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return {"guardrail_passed": True}

    try:
        result = guard.validate(last_message.content)
        passed = getattr(result, "validation_passed", True)
        return {"guardrail_passed": passed}
    except Exception as exc:
        logger.warning("Output guardrail raised: %s", exc)
        return {"guardrail_passed": False}


def helpfulness(state: AgentState) -> Dict[str, Any]:
    """Evaluate helpfulness of the latest response."""
    if len(state["messages"]) > 10:
        return {"messages": [AIMessage(content="HELPFULNESS:END")]}

    initial_query = state["messages"][0]
    final_response = state["messages"][-1]

    prompt_template = """Given an initial query and a final response, determine if the final response \
is extremely helpful or not. Please indicate helpfulness with a 'Y' and unhelpfulness as an 'N'.

Initial Query:
{initial_query}

Final Response:
{final_response}"""

    helpfulness_prompt = PromptTemplate.from_template(prompt_template)
    eval_llm = get_chat_model(model_name=EVAL_MODEL)
    chain = helpfulness_prompt | eval_llm | StrOutputParser()

    result = chain.invoke({
        "initial_query": initial_query.content,
        "final_response": final_response.content,
    })

    decision = "Y" if "Y" in result else "N"
    return {"messages": [AIMessage(content=f"HELPFULNESS:{decision}")]}


# ---------------------------------------------------------------------------
# 10. Routing
# ---------------------------------------------------------------------------


def route_after_input_guardrail(state: AgentState):
    if state.get("guardrail_passed") is False:
        return END
    return "agent"


def route_after_agent(state: AgentState):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "action"
    return "output_guardrail"


def route_after_output_guardrail(state: AgentState):
    if state.get("guardrail_passed") is False:
        return "agent"  # retry
    return "helpfulness"


def helpfulness_decision(state: AgentState):
    last = state["messages"][-1]
    text = getattr(last, "content", "")
    if text == "HELPFULNESS:END":
        return END
    if "HELPFULNESS:Y" in text:
        return "end"
    return "continue"


# ---------------------------------------------------------------------------
# 11. Graph Build
# ---------------------------------------------------------------------------


def build_graph():
    """Build the investment assistant graph with guardrails and helpfulness."""
    tool_node = ToolNode(get_tool_belt())

    g = StateGraph(AgentState)
    g.add_node("input_guardrail", input_guardrail)
    g.add_node("agent", agent)
    g.add_node("action", tool_node)
    g.add_node("output_guardrail", output_guardrail)
    g.add_node("helpfulness", helpfulness)

    g.set_entry_point("input_guardrail")

    g.add_conditional_edges(
        "input_guardrail",
        route_after_input_guardrail,
        {"agent": "agent", END: END},
    )
    g.add_conditional_edges(
        "agent",
        route_after_agent,
        {"action": "action", "output_guardrail": "output_guardrail"},
    )
    g.add_edge("action", "agent")
    g.add_conditional_edges(
        "output_guardrail",
        route_after_output_guardrail,
        {"agent": "agent", "helpfulness": "helpfulness"},
    )
    g.add_conditional_edges(
        "helpfulness",
        helpfulness_decision,
        {"continue": "agent", "end": END, END: END},
    )
    return g


def _route_after_agent_simple(state: AgentState):
    """Route for simple graph: tools or END (no guardrails/helpfulness)."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "action"
    return END


def build_simple_graph():
    """Build a simplified graph with just agent + action (no guardrails or helpfulness)."""
    tool_node = ToolNode(get_tool_belt())

    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("action", tool_node)

    g.set_entry_point("agent")
    g.add_conditional_edges(
        "agent",
        _route_after_agent_simple,
        {"action": "action", END: END},
    )
    g.add_edge("action", "agent")
    return g


graph = build_graph().compile()
simple_graph = build_simple_graph().compile()
