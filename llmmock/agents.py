"""Agent orchestration: combine directive + retrieved memory + history, then call the model."""
from __future__ import annotations

from typing import Any, AsyncIterator

from fastapi import HTTPException

from . import db, memory
from .config import settings
from .ollama_client import client


def _chat_options(agent: dict[str, Any], temperature: float | None, num_ctx: int | None) -> dict[str, Any]:
    return {
        "temperature": temperature if temperature is not None else agent["temperature"],
        "num_ctx": num_ctx if num_ctx is not None else agent["num_ctx"],
    }


async def _build_messages(
    agent: dict[str, Any],
    session_id: str,
    user_message: str,
    use_memory: bool,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Assemble the message list sent to the model. Returns (messages, retrieved_memories)."""
    system_parts: list[str] = []
    if agent["directive"]:
        system_parts.append(agent["directive"])

    retrieved: list[dict[str, Any]] = []
    if use_memory:
        retrieved = await memory.recall(agent["id"], user_message, top_k=agent["top_k"])
        block = memory.format_memory_block(retrieved)
        if block:
            system_parts.append(block)

    messages: list[dict[str, str]] = []
    if system_parts:
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

    for m in db.get_history(agent["id"], session_id, limit=agent["max_history"]):
        messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": user_message})
    return messages, retrieved


def require_agent(agent_id: str) -> dict[str, Any]:
    agent = db.get_agent(agent_id) or db.get_agent_by_name(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id!r} not found")
    return agent


async def chat(
    agent: dict[str, Any],
    *,
    message: str,
    session_id: str = "default",
    reasoning: bool | None = None,
    use_memory: bool = True,
    remember: bool = False,
    temperature: float | None = None,
    num_ctx: int | None = None,
) -> dict[str, Any]:
    think = agent["reasoning"] if reasoning is None else reasoning
    messages, retrieved = await _build_messages(agent, session_id, message, use_memory)

    resp = await client.chat(
        agent["model"], messages, think=think,
        options=_chat_options(agent, temperature, num_ctx),
    )
    msg = resp.get("message", {})
    content = msg.get("content", "") or ""
    thinking = msg.get("thinking")

    db.add_message(agent["id"], session_id, "user", message)
    db.add_message(agent["id"], session_id, "assistant", content, thinking)
    if remember:
        await memory.remember(agent["id"], message)

    return {
        "agent_id": agent["id"],
        "session_id": session_id,
        "content": content,
        "thinking": thinking,
        "reasoning_enabled": think,
        "memories_used": retrieved,
        "model": agent["model"],
        "usage": {
            "prompt_tokens": resp.get("prompt_eval_count"),
            "completion_tokens": resp.get("eval_count"),
        },
    }


async def chat_stream(
    agent: dict[str, Any],
    *,
    message: str,
    session_id: str = "default",
    reasoning: bool | None = None,
    use_memory: bool = True,
    remember: bool = False,
    temperature: float | None = None,
    num_ctx: int | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield incremental chunks {type: 'thinking'|'content', delta: str} then a final 'done'."""
    think = agent["reasoning"] if reasoning is None else reasoning
    messages, retrieved = await _build_messages(agent, session_id, message, use_memory)

    full_content: list[str] = []
    full_thinking: list[str] = []
    async for chunk in client.chat_stream(
        agent["model"], messages, think=think,
        options=_chat_options(agent, temperature, num_ctx),
    ):
        m = chunk.get("message", {})
        if m.get("thinking"):
            full_thinking.append(m["thinking"])
            yield {"type": "thinking", "delta": m["thinking"]}
        if m.get("content"):
            full_content.append(m["content"])
            yield {"type": "content", "delta": m["content"]}
        if chunk.get("done"):
            break

    content = "".join(full_content)
    thinking = "".join(full_thinking) or None
    db.add_message(agent["id"], session_id, "user", message)
    db.add_message(agent["id"], session_id, "assistant", content, thinking)
    if remember:
        await memory.remember(agent["id"], message)

    yield {"type": "done", "agent_id": agent["id"], "session_id": session_id,
           "memories_used": retrieved, "reasoning_enabled": think}


def create(payload: dict[str, Any]) -> dict[str, Any]:
    if db.get_agent_by_name(payload["name"]):
        raise HTTPException(status_code=409, detail=f"Agent named {payload['name']!r} already exists")
    return db.create_agent(
        name=payload["name"],
        model=payload.get("model") or settings.DEFAULT_MODEL,
        directive=payload.get("directive", ""),
        reasoning=payload.get("reasoning", False),
        temperature=payload.get("temperature", settings.DEFAULT_TEMPERATURE),
        num_ctx=payload.get("num_ctx", settings.DEFAULT_NUM_CTX),
        max_history=payload.get("max_history", 20),
        top_k=payload.get("top_k", settings.MEMORY_TOP_K),
    )
