"""Thin async wrapper around the ollama HTTP API (chat, embeddings, model listing)."""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from .config import settings


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or settings.OLLAMA_URL).rstrip("/")
        # Generous read timeout: local generation can be slow on modest hardware.
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=600.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def list_models(self) -> list[dict[str, Any]]:
        r = await self._client.get("/api/tags")
        r.raise_for_status()
        return r.json().get("models", [])

    async def pull_stream(self, model: str) -> AsyncIterator[dict[str, Any]]:
        """Pull/download a model, yielding ollama's progress chunks.

        `model` may be an ollama registry tag (e.g. 'qwen3:4b') or a HuggingFace GGUF
        repo (e.g. 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M').
        """
        # Long timeout: downloads can take a while. Disable read timeout for this stream.
        timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=None)
        async with self._client.stream(
            "POST", "/api/pull", json={"model": model, "stream": True}, timeout=timeout
        ) as r:
            if r.status_code != 200:
                body = await r.aread()
                raise OllamaError(f"pull failed ({r.status_code}): {body.decode()}")
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    async def delete_model(self, model: str) -> None:
        r = await self._client.request("DELETE", "/api/delete", json={"model": model})
        if r.status_code not in (200, 404):
            raise OllamaError(f"delete failed ({r.status_code}): {r.text}")

    async def embed(self, text: str, model: str | None = None) -> list[float]:
        model = model or settings.EMBED_MODEL
        r = await self._client.post("/api/embed", json={"model": model, "input": text})
        if r.status_code != 200:
            raise OllamaError(f"embed failed ({r.status_code}): {r.text}")
        data = r.json()
        embs = data.get("embeddings") or []
        if not embs:
            raise OllamaError(f"embed returned no vectors for model {model!r}")
        return embs[0]

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        think: bool = False,
        options: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Non-streaming chat. Returns the parsed ollama response dict."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": think,
        }
        if options:
            payload["options"] = options
        if tools:
            payload["tools"] = tools
        r = await self._client.post("/api/chat", json=payload)
        if r.status_code != 200:
            raise OllamaError(f"chat failed ({r.status_code}): {r.text}")
        return r.json()

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        think: bool = False,
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming chat. Yields each ollama ndjson chunk as a dict."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": think,
        }
        if options:
            payload["options"] = options
        async with self._client.stream("POST", "/api/chat", json=payload) as r:
            if r.status_code != 200:
                body = await r.aread()
                raise OllamaError(f"chat stream failed ({r.status_code}): {body.decode()}")
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


# Shared singleton used by the app.
client = OllamaClient()
