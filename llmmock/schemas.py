"""Pydantic request/response models for the agent API and the OpenAI-compatible API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .config import settings


# --- Agents ---------------------------------------------------------------

class AgentCreate(BaseModel):
    name: str = Field(..., description="Unique, human-friendly agent name")
    directive: str = Field("", description="The agent's system prompt / persona / standing instructions")
    model: str | None = Field(None, description="ollama model tag; defaults to server default")
    reasoning: bool = Field(False, description="Enable the model's thinking mode by default")
    temperature: float = settings.DEFAULT_TEMPERATURE
    num_ctx: int = settings.DEFAULT_NUM_CTX
    max_history: int = Field(20, description="Max prior messages from a session to include")
    top_k: int = Field(settings.MEMORY_TOP_K, description="Semantic memories injected per turn")


class AgentUpdate(BaseModel):
    name: str | None = None
    directive: str | None = None
    model: str | None = None
    reasoning: bool | None = None
    temperature: float | None = None
    num_ctx: int | None = None
    max_history: int | None = None
    top_k: int | None = None


class AgentChatRequest(BaseModel):
    message: str = Field(..., description="The user's message")
    session_id: str = Field("default", description="Conversation thread id for this agent")
    reasoning: bool | None = Field(None, description="Override the agent's reasoning setting for this turn")
    use_memory: bool = Field(True, description="Retrieve & inject relevant long-term memories")
    remember: bool = Field(False, description="Also store this user message as a long-term memory")
    stream: bool = False
    temperature: float | None = None
    num_ctx: int | None = None


class MemoryCreate(BaseModel):
    content: str = Field(..., description="A fact/preference/note for the agent to remember")


class ModelRef(BaseModel):
    model: str = Field(
        ...,
        description=(
            "Model to pull/delete. Accepts an ollama tag (e.g. 'qwen3:4b'), a HuggingFace "
            "GGUF repo (e.g. 'hf.co/ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M'), or a full "
            "HuggingFace URL (e.g. 'https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF')."
        ),
    )
    quantization: str | None = Field(
        None,
        description="Optional quant tag for HuggingFace GGUF repos, e.g. 'Q4_K_M'. "
        "Ignored if the model already carries a ':tag'.",
    )


# --- OpenAI-compatible ----------------------------------------------------

class OAIMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = None


class OAIChatRequest(BaseModel):
    model: str | None = None
    messages: list[OAIMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    stream: bool = False
    # Reasoning controls (any of these enables thinking):
    reasoning_effort: str | None = None  # OpenAI-style: "low"|"medium"|"high"
    think: bool | None = None            # convenience flag
    # Allow vendor extensions like num_ctx without erroring.
    num_ctx: int | None = None

    class Config:
        extra = "allow"
