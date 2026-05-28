"""Long-term (semantic) memory: embed text, store it, and retrieve by similarity."""
from __future__ import annotations

import math
from typing import Any

from . import db
from .config import settings
from .ollama_client import client


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


async def remember(agent_id: str, content: str) -> dict[str, Any]:
    """Embed and persist a fact/preference/note for an agent."""
    embedding = await client.embed(content)
    return db.add_memory(agent_id, content, embedding)


async def recall(
    agent_id: str,
    query: str,
    top_k: int | None = None,
    min_score: float | None = None,
) -> list[dict[str, Any]]:
    """Return the most relevant stored memories for a query, best first."""
    top_k = top_k if top_k is not None else settings.MEMORY_TOP_K
    min_score = min_score if min_score is not None else settings.MEMORY_MIN_SCORE
    memories = db.get_memories(agent_id)
    if not memories:
        return []
    q = await client.embed(query)
    scored = []
    for m in memories:
        score = _cosine(q, m["embedding"])
        if score >= min_score:
            scored.append({"id": m["id"], "content": m["content"],
                           "score": round(score, 4), "created_at": m["created_at"]})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def format_memory_block(memories: list[dict[str, Any]]) -> str:
    """Render retrieved memories as a system-context block to prepend to a turn."""
    if not memories:
        return ""
    lines = [f"- {m['content']}" for m in memories]
    return (
        "Relevant information you remember about this user/context "
        "(use it when helpful, ignore if irrelevant):\n" + "\n".join(lines)
    )
