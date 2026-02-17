"""Investment Memory Agent Package.

This package implements a memory-enabled investment advisory assistant using LangGraph.
It demonstrates all 5 memory types from the CoALA framework:
- Short-term: Conversation history within a thread
- Long-term: User preferences stored across sessions
- Semantic: Facts retrieved by meaning
- Episodic: Learning from past experiences
- Procedural: Self-improving instructions
"""

from investment_memory.agents import investment_graph, create_investment_agent
from investment_memory.stores import create_memory_store, create_checkpointer
from investment_memory.memory_types import (
    ShortTermMemory,
    LongTermMemory,
    SemanticMemory,
    EpisodicMemory,
    ProceduralMemory,
)
from investment_memory.utils import trim_conversation, summarize_conversation

__all__ = [
    "investment_graph",
    "create_investment_agent",
    "create_memory_store",
    "create_checkpointer",
    "ShortTermMemory",
    "LongTermMemory",
    "SemanticMemory",
    "EpisodicMemory",
    "ProceduralMemory",
    "trim_conversation",
    "summarize_conversation",
]
