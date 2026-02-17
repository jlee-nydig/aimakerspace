"""Memory store configurations for the investment agent.

This module provides factory functions for creating memory stores
used by the investment agent for different memory types.
"""

from typing import Optional
from langchain_openai import OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore


def create_checkpointer() -> MemorySaver:
    """Create a checkpointer for short-term memory.

    The checkpointer saves graph state at each step, enabling:
    - Conversation history within a thread
    - State inspection and debugging
    - Time-travel debugging in LangGraph Studio

    Returns:
        MemorySaver: An in-memory checkpointer for development.

    Note:
        For production, consider using PostgresSaver or other
        persistent checkpointers.
    """
    return MemorySaver()


def create_memory_store(
    with_embeddings: bool = True,
    embedding_model: Optional[str] = "text-embedding-3-small",
    embedding_dims: int = 1536,
) -> InMemoryStore:
    """Create a memory store for long-term, semantic, episodic, and procedural memory.

    Args:
        with_embeddings: Whether to enable semantic search with embeddings.
        embedding_model: The OpenAI embedding model to use.
        embedding_dims: The dimension of the embedding vectors.

    Returns:
        InMemoryStore: A store configured for the specified memory types.

    Note:
        For production, consider using PostgresStore or other
        persistent stores.
    """
    if with_embeddings:
        embeddings = OpenAIEmbeddings(model=embedding_model)
        return InMemoryStore(
            index={
                "embed": embeddings,
                "dims": embedding_dims,
            }
        )
    else:
        return InMemoryStore()


def initialize_investment_store(store: InMemoryStore) -> None:
    """Initialize the store with default investment data.

    This function sets up:
    - Default procedural instructions for the investment agent
    - Sample investment knowledge for semantic memory

    Args:
        store: The memory store to initialize.
    """
    # Initialize procedural memory with default instructions
    default_instructions = """You are an Investment Advisory Assistant. Follow these guidelines:

1. Be objective and data-driven in all interactions
2. Always consider the user's risk tolerance and investment profile when giving advice
3. Provide evidence-based investment information
4. Present balanced perspectives without being overly bullish or bearish
5. Refer users to licensed financial advisors for specific investment decisions

When responding:
- Use the user's name if known
- Reference their investment goals and constraints when relevant
- Build on previous conversations when possible
- Keep responses focused and actionable
- Note that past performance doesn't guarantee future results"""

    store.put(
        ("agent", "instructions"),
        "investment_assistant",
        {
            "instructions": default_instructions,
            "version": 1,
        }
    )

    # Initialize sample episodic memories
    sample_episodes = [
        {
            "situation": "User asked about diversifying a concentrated portfolio",
            "input": "I have most of my wealth in my company's stock. How should I think about diversifying?",
            "output": "Concentrated positions can carry significant risk. I'd recommend considering a systematic diversification plan: 1) Assess your overall financial picture and risk tolerance. 2) Explore tax-efficient strategies like exchange funds or charitable remainder trusts. 3) Consider gradually rebalancing into a diversified mix of asset classes including broad market equities, fixed income, and alternatives. 4) Work with a financial advisor to create a timeline that balances tax implications with risk reduction. Would you like to explore any of these strategies in more detail?",
            "feedback": "User found this helpful and appreciated the structured approach with tax considerations",
        },
        {
            "situation": "User concerned about market downturn impact on retirement",
            "input": "I'm worried about a market crash right before I retire. What should I do?",
            "output": "Sequence-of-returns risk is a legitimate concern for pre-retirees. Here are some strategies to consider: 1) Gradually shift to a more conservative asset allocation as you approach retirement - a common approach is the 'glide path' strategy. 2) Build a cash reserve covering 1-2 years of expenses to avoid selling during downturns. 3) Consider a bucket strategy that segments your portfolio by time horizon. 4) Diversify across asset classes that may respond differently to market stress. Which of these approaches would you like to discuss further?",
            "feedback": "User appreciated the practical steps and felt more confident about their retirement planning",
        },
    ]

    for i, episode in enumerate(sample_episodes):
        store.put(
            ("agent", "episodes"),
            f"episode_{i}",
            {
                "text": episode["situation"],  # Used for semantic search
                **episode,
            }
        )


# Namespace conventions for the investment agent
NAMESPACES = {
    # User-specific namespaces (parameterized by user_id)
    "profile": lambda user_id: (user_id, "profile"),
    "preferences": lambda user_id: (user_id, "preferences"),
    "facts": lambda user_id: (user_id, "facts"),
    "investment_history": lambda user_id: (user_id, "investment_history"),

    # Shared namespaces
    "knowledge": ("investment", "knowledge"),
    "instructions": ("agent", "instructions"),
    "episodes": ("agent", "episodes"),
}
