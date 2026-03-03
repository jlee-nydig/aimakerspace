# Module 14: 🛤️ Guardrails & Caching

🎯 Learn a few upgrades for performance and security/trustworthiness.

📚 **Learning Outcomes**

- Understand guardrails, including the key categories of guardrails
- Understand the importance of semantic caching
- How to use Prompt and Embedding caches

🧰 **New Tools**

- Guardrails: [Guardrails AI](https://github.com/guardrails-ai/guardrails)
- Caching: [CacheBackedEmbeddings](https://python.langchain.com/docs/how_to/caching_embeddings/) and [Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching)

## 📛 Required Tooling & Account Setup

In addition to the tools we've already learned, in this module you'll need:

1. 🔑 Get your [Guardrails AI](https://hub.guardrailsai.com/keys) API key

**Guardrails Hub Setup:**

```bash
# Configure Guardrails API
uv run guardrails configure

# Install required guards
uv run guardrails hub install hub://tryolabs/restricttotopic
uv run guardrails hub install hub://guardrails/detect_jailbreak
uv run guardrails hub install hub://guardrails/profanity_free
uv run guardrails hub install hub://guardrails/guardrails_pii
```

## 📜 Recommended Reading

- [The AI Guardrails Index](https://www.guardrailsai.com/blog/ai-guardrails-index) (Feb 2025)
- [Caching](https://python.langchain.com/docs/how_to/caching_embeddings/) (July 2025)
- [Semantic Caching for Low-Cost LLM Serving](https://qdrant.tech/articles/semantic-cache-ai-data-retrieval/) (Aug 2025)
- [Guardrails](https://python.langchain.com/docs/), by LangChain

## Overview

In Module 14, we'll cover a few nice-to-haves for our production LLM applications that will become more and more important as we scale: Caching and Guardrails

As they say, cache is king. If we can save some cash with caching, we should. We can leverage both prompt and embedding caches in general, depending on our use cases.

Guardrails are the runtime checks that keep your AI application's inputs and outputs on track — enforcing safety, policy, brand, correctness, and structure.

## 🛤️ Guardrails

Guardrails are the policies and runtime controls that keep an AI system's inputs and outputs on track. Practically, they're input/output modules that monitor and constrain behavior in real time — detecting issues (e.g., safety violations, hallucinations), enforcing structure/access rules, and triggering fallbacks.

In this framing:

- **Guardrails (runtime):** Live validators around your LLM workflow that check, block, fix, or route content at input and output.
- **Alignment (goal):** System behaves per org values and product intent.
- **Evaluation (offline):** Pre-deploy testing on datasets to pick guards, thresholds, and fallbacks.
- **Validation (online):** Inference-time checks — what most guard frameworks do.
- **What a "guard" is:** Usually a small, tuned classifier (fast), sometimes rules/heuristics, with optional LLM-as-judge fallback for edge cases. Guards can also enforce schemas (valid JSON/XML/YAML) for tool/API calls.

### Practical Implementation

Guardrails are typically other LLMs. We can put them on the input or the output side of our applications:

- **Input guards:** on-topic, jailbreak/prompt-injection, PII detection/redaction.
- **Output guards:** content moderation/profanity, restricted topics/brand policy, hallucination/faithfulness, competitor mentions, schema validation/repair.

### Enhanced Agent Architecture with Guardrails

```
User Input → Input Guards → Agent → Tools → Output Guards → Response
     ↓           ↓          ↓       ↓         ↓               ↓
  Jailbreak   Topic     Model    RAG/     Content            Safe
  Detection   Check   Decision  Search   Validation        Response
```

Guardrails validate both inputs (user queries) and outputs (agent responses) to ensure:
- Conversations stay on-topic (e.g., investment domain)
- No PII leakage (credit cards, SSNs, etc.)
- No adversarial prompt injection
- Professional, appropriate responses

### Core Guardrails in Practice

Using Guardrails AI, we can configure the following guards:

1. **Topic Restriction** — Constrains queries to valid topics (e.g., investments, portfolio management, market analysis) and blocks invalid topics (e.g., medical advice, legal advice, gambling, explicit content). Uses `RestrictToTopic` from the Guardrails Hub.

2. **Jailbreak Detection** — Catches prompt injection and role-bypass attempts using `DetectJailbreak`. For example, blocks queries like "Ignore all previous instructions."

3. **PII Protection** — Detects and redacts sensitive entities (credit cards, SSNs, phone numbers, email addresses) using `GuardrailsPII` with an `on_fail="fix"` strategy that automatically redacts detected PII.

4. **Content Moderation** — Filters profanity and inappropriate content using `ProfanityFree` with configurable thresholds and sentence-level validation.

5. **Factuality / Hallucination Guard** — Checks answer grounding vs. provided sources (RAG-aware) beyond simple relevance using `LlmRagEvaluator` with a hallucination judge prompt.

### Design Patterns

- **Rules vs. Models (Workflows vs. Agency):** Prefer simple rules/heuristics or very small models when possible for performance. Add more robust larger LLM guardrails in a fallback capacity if not getting high-confidence outputs.
- **Beyond Escalation:** On-fail strategies can go beyond escalating to a larger fallback model. Alternatives include putting a human in the loop, or simply logging undesirable information and blocking the response with a default error message.
- **Internal vs. External Applications:** Heavier guards on apps exposed directly to external users. Emphasize output policy/brand & schema correctness.
- **Guards as a Service:** Treat guards as any other LLM-powered service with monitoring of both relevant LLM-specific and application-specific metrics, including latency budgets, threshold tuning, A/B testing, and drift monitoring. Iterate.

### 🦺 AI Guardrails Index

The AI Guardrails Index, released by Guardrails AI, covers key areas important to enterprise today:

1. **Jailbreak Prevention** — Detects prompt-injection & role-bypass patterns.
2. **PII Detection** — Finds/redacts sensitive entities (credit cards, SSNs, emails, phones, etc.).
3. **Content Moderation** — Toxicity/harassment/sexual/violent content filters.
4. **Hallucination / Faithfulness** — Checks answer grounding vs. provided sources (RAG-aware) beyond simple relevance.
5. **Competitor Presence** — Blocks or flags mentions of disallowed brands/products.
6. **Restricted Topics** — Policy taxonomy for business-specific "do/don't discuss".

### Choose The Right Guard for The Job

```
├─> Is your app public-facing? ── Yes ──> Prioritize INPUT guards
│                                └─ No ──> Prioritize OUTPUT + SCHEMA guards
│
├─> Primary risk = Safety/Compliance? ── Yes
│                                        └─ No (Reliability/Brand) ──> Schema
│
├─> Tight latency budget? ── Yes ──> Small classifier / rules first
│                            └─ No ──> Allow LLM-judge fallback on low-confidence
│
└─> Pick on-fail: raise | fix | fallback | log+escalate
```

**Quick Map (Risk → Guard → I/O → Default On-Fail)**

| Risk | Guard Type | I/O | Default On-Fail |
|------|-----------|-----|-----------------|
| Policy/safety | Restricted topics | Output | raise |
| Privacy | PII detection/redaction | Input & Output | fix (redact) |
| Toxic/brand | Moderation/profanity | Output | raise (or fix if light) |
| Reliability | Schema validation/repair | Output | fix (repair JSON) |
| Factuality | Hallucination/faithfulness | Output | fallback (regenerate/safer route) |
| Abuse | Jailbreak/prompt-injection | Input | raise |
| Brand | Competitor presence | Output | raise (or route) |
| Misuse | On-topic classifier | Input | raise |

**Quick Start Checklist:**

1. Define policies & risks (scope per surface: chat, tools, actions).
2. Pick guards per surface and set latency budget.
3. Choose thresholds and on-fail behaviors.
4. Evaluate offline on real samples; collect FP/FN and adjust.
5. Validate online with metrics (block rate, redaction rate, fallback rate, latency). Add alerts.

### Conclusions

- Guardrails ≠ only "safety"; they're a broad runtime control layer for safety + policy + reliability + structure.
- Start small: on-topic + PII + moderation, then add schema and hallucination checks for RAG/tooling.
- Favor fast classifiers; use LLM fallback sparingly where accuracy matters most.
- Close the loop: evaluate offline, validate online, monitor & iterate.

## 🏦 Caching

Caching is one of the easiest things that we can do to save cash. The TL;DR on caching is that if we've seen it before, then we can just remember it. We don't need a fresh generation of new tokens — we can use ones that we've already saved from previous runs.

This works equally well for embeddings (e.g., during RAG retrieval processes), and for remembering responses to entire prompts, as long as similar ones have been used in the past.

### Production Caching Architecture

Our caching strategy operates at two levels:

**Embedding Caching (Disk-backed):**
- Check local cache for previously computed embeddings
- If found: return cached vector (instant, free)
- If not found: call OpenAI API, store result in cache

**LLM Response Caching (In-memory or SQLite):**
- Identical prompts return cached responses
- Eliminates redundant API calls
- In-memory for development/demo, SQLite for production

**Benefits:**
- ⚡ Faster response times (cache hits are instant)
- 💰 Reduced API costs (no duplicate calls)
- 🔄 Consistent results for identical inputs
- 📈 Better scalability

### Cache-Backed Embeddings

The process of embedding data is time-consuming. For every vector in our vector database, as well as for every single query, we:

1. Send the text to an API endpoint (self-hosted, OpenAI, etc)
2. Wait for processing
3. Receive response

This process costs time and money. The more data we have, the more time and money it costs!

Instead, we can set up a cache to hold all of the data chunks that we've already converted to embeddings, and check it every time we need to use them.

If we cache [query, context], we can avoid using the retrieval parts of our RAG applications. The process is as follows:

```
1. Set up cache
2. Send text to embedding endpoint
3. Check cache
   - If Y, return cached
   - If N, convert text
       - Store new embeddings
4. Return vector embeddings
```

There are very few reasons *not* to use caching for production LLM applications.

### Prompt Caching

If we've seen this prompt before, we just use the stored response. That is, if we cache [query, response], we can *avoid using our entire RAG chain*.

In this way, we can speed up repeated large language model (LLM) requests by storing the results of certain parts of a prompt so they don't have to be recomputed. Even OpenAI offers this as a service to help you get automatic discounts on inputs that the model has recently seen.

Just like embedding caching, we can decrease latency AND increase throughput with this approach, a truly magical combination 🪄.

### Caching Considerations

When evaluating a caching strategy, consider:

- **Memory vs. Disk caching trade-offs** — In-memory is faster but volatile; disk-backed persists across restarts
- **Cache invalidation strategies** — When should cached results be refreshed?
- **Concurrent access patterns** — How does caching behave under multiple simultaneous users?
- **Cold start scenarios** — First-time queries always incur full latency
