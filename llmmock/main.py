"""FastAPI app: OpenAI-compatible endpoint + an agents API with memory and directives."""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from . import agents, db, memory
from .config import settings
from .ollama_client import OllamaError, client
from .schemas import (
    AgentChatRequest,
    AgentCreate,
    AgentUpdate,
    MemoryCreate,
    ModelRef,
    OAIChatRequest,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.get_conn()  # init schema on startup
    yield
    await client.aclose()


app = FastAPI(title="llmmock", version="0.1.0", lifespan=lifespan)

# Local dev tool: allow any origin so the web client works from file:// or another port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# Headers that keep SSE flowing and stop proxies/clients from buffering the stream.
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}


# --- auth (optional) ------------------------------------------------------

def require_auth(authorization: str | None = Header(default=None)) -> None:
    if not settings.API_KEY:
        return
    expected = f"Bearer {settings.API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# --- misc -----------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, Any]:
    try:
        models = await client.list_models()
        ok = True
    except Exception as e:  # noqa: BLE001
        models, ok = [], False
        return {"status": "degraded", "ollama": str(e)}
    return {
        "status": "ok" if ok else "degraded",
        "ollama_url": settings.OLLAMA_URL,
        "default_model": settings.DEFAULT_MODEL,
        "embed_model": settings.EMBED_MODEL,
        "models_available": [m.get("name") for m in models],
    }


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "name": "llmmock",
        "web_client": "/ui",
        "docs": "/docs",
        "openai_base_url": "set your client's base_url to <host>/v1",
        "endpoints": {
            "openai": ["/v1/models", "/v1/chat/completions"],
            "agents": ["/agents", "/agents/{id}", "/agents/{id}/chat",
                       "/agents/{id}/memory", "/agents/{id}/sessions"],
        },
    }


@app.get("/ui", include_in_schema=False)
async def web_client() -> FileResponse:
    index = WEB_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="web client not found")
    return FileResponse(index)


# --- OpenAI-compatible ----------------------------------------------------

def _oai_messages_to_ollama(messages: list[Any]) -> list[dict[str, str]]:
    out = []
    for m in messages:
        content = m.content
        if isinstance(content, list):  # multimodal content parts -> join text
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        out.append({"role": m.role, "content": content or ""})
    return out


def _wants_thinking(req: OAIChatRequest) -> bool:
    if req.think is not None:
        return req.think
    if req.reasoning_effort and req.reasoning_effort.lower() not in ("none", "minimal"):
        return True
    return False


@app.get("/v1/models")
async def list_models(_: None = Depends(require_auth)) -> dict[str, Any]:
    models = await client.list_models()
    return {
        "object": "list",
        "data": [
            {"id": m.get("name"), "object": "model", "created": 0, "owned_by": "ollama"}
            for m in models
        ],
    }


_HF_URL = re.compile(r"^https?://(?:www\.)?huggingface\.co/(.+)$", re.IGNORECASE)


def normalize_model_ref(model: str, quantization: str | None = None) -> str:
    """Turn user input into a tag ollama understands.

    Accepts a full HuggingFace URL, a bare 'huggingface.co/...' path, an 'hf.co/...'
    repo, or a plain ollama tag. Strips /tree, /blob, /resolve paths and query strings,
    and appends an optional quant tag for HuggingFace GGUF repos.
    """
    s = model.strip()
    m = _HF_URL.match(s)
    if m:
        path = m.group(1)
    elif s.lower().startswith("huggingface.co/"):
        path = s[len("huggingface.co/"):]
    else:
        path = None

    if path is not None:
        path = path.split("?")[0].split("#")[0]
        path = re.sub(r"/(tree|blob|resolve)/.*$", "", path).strip("/")
        s = "hf.co/" + path

    # Apply a separate quant only to HuggingFace refs that don't already carry a ':tag'.
    if quantization and s.lower().startswith("hf.co/") and ":" not in s.split("/")[-1]:
        s = f"{s}:{quantization}"
    return s


@app.get("/models")
async def models_detailed(_: None = Depends(require_auth)) -> dict[str, Any]:
    """Installed models with size/details (richer than the OpenAI /v1/models list)."""
    models = await client.list_models()
    return {
        "data": [
            {
                "name": m.get("name"),
                "size": m.get("size"),
                "modified_at": m.get("modified_at"),
                "family": (m.get("details") or {}).get("family"),
                "parameter_size": (m.get("details") or {}).get("parameter_size"),
                "quantization": (m.get("details") or {}).get("quantization_level"),
            }
            for m in models
        ]
    }


# Active downloads, keyed by resolved model name. A pull runs as a server-side
# background task that updates its entry here, so progress is visible to every client
# (and survives a page reload), not just whoever started it.
_pulls: dict[str, dict[str, Any]] = {}
_pull_tasks: set[asyncio.Task] = set()


async def _run_pull(target: str) -> None:
    _pulls[target] = {"model": target, "status": "starting", "percent": None,
                      "completed": None, "total": None, "done": False, "error": None}
    try:
        async for chunk in client.pull_stream(target):
            e = _pulls[target]
            e["status"] = chunk.get("status", e["status"])
            total, completed = chunk.get("total"), chunk.get("completed")
            if total:
                e["total"], e["completed"] = total, completed
                e["percent"] = round(100 * (completed or 0) / total, 1)
            if chunk.get("status") == "success":
                break
        _pulls[target].update({"done": True, "status": "success", "percent": 100})
    except Exception as exc:  # noqa: BLE001 - surface any failure via the registry
        _pulls[target].update({"done": True, "error": str(exc) or exc.__class__.__name__})
    # Keep the finished entry around briefly so pollers can see the final state.
    await asyncio.sleep(8)
    _pulls.pop(target, None)


@app.post("/models/pull")
async def pull_model(req: ModelRef, _: None = Depends(require_auth)) -> dict[str, Any]:
    """Start downloading a model (ollama registry or HuggingFace GGUF) in the background.

    Returns immediately. Poll GET /models/pulls for live progress. Accepts an ollama tag,
    an 'hf.co/...' repo, or a full HuggingFace URL (plus an optional `quantization`).
    """
    target = normalize_model_ref(req.model, req.quantization)
    already = _pulls.get(target)
    if not already or already.get("done"):
        task = asyncio.create_task(_run_pull(target))
        _pull_tasks.add(task)
        task.add_done_callback(_pull_tasks.discard)
    return {"model": target, "started": True}


@app.get("/models/pulls")
async def list_pulls(_: None = Depends(require_auth)) -> dict[str, Any]:
    """Currently active (and just-finished) downloads, with progress."""
    return {"data": list(_pulls.values())}


@app.post("/models/delete")
async def delete_model(req: ModelRef, _: None = Depends(require_auth)) -> dict[str, Any]:
    target = normalize_model_ref(req.model, req.quantization)
    await client.delete_model(target)
    return {"deleted": target}


@app.post("/models/unload")
async def unload_model(req: ModelRef, _: None = Depends(require_auth)) -> dict[str, Any]:
    """Evict a model from memory now (frees VRAM). Used when switching chat models."""
    target = normalize_model_ref(req.model, req.quantization)
    ok = await client.unload(target)
    return {"unloaded": target, "ok": ok}


@app.get("/models/loaded")
async def loaded_models(_: None = Depends(require_auth)) -> dict[str, Any]:
    """Models currently loaded in memory."""
    return {"data": await client.loaded_models()}


@app.post("/models/check")
async def check_model(req: ModelRef, _: None = Depends(require_auth)) -> dict[str, Any]:
    """Verify a model actually loads/runs on this backend by generating one token.

    Returns {ok: true} if it runs, or {ok: false, error, hint} if it can't (e.g. the
    GGUF architecture isn't supported by this ollama version, or it won't fit in memory).
    """
    target = normalize_model_ref(req.model, req.quantization)
    try:
        await client.chat(target, [{"role": "user", "content": "hi"}],
                           options={"num_predict": 1})
        return {"model": target, "ok": True}
    except Exception as e:  # noqa: BLE001
        err = str(e) or e.__class__.__name__
        hint = None
        if "unable to load model" in err or "architecture" in err:
            hint = ("This model failed to load — often the GGUF uses an architecture this "
                    "ollama version doesn't support, or it doesn't fit in memory. Try the "
                    "ollama-registry build of the model, or a smaller quant.")
        return {"model": target, "ok": False, "error": err, "hint": hint}


@app.post("/v1/chat/completions")
async def chat_completions(req: OAIChatRequest, _: None = Depends(require_auth)):
    model = req.model or settings.DEFAULT_MODEL
    messages = _oai_messages_to_ollama(req.messages)
    think = _wants_thinking(req)
    options: dict[str, Any] = {}
    if req.temperature is not None:
        options["temperature"] = req.temperature
    if req.top_p is not None:
        options["top_p"] = req.top_p
    if req.top_k is not None:
        options["top_k"] = req.top_k
    if req.max_tokens is not None:
        options["num_predict"] = req.max_tokens
    options["num_ctx"] = req.num_ctx or settings.DEFAULT_NUM_CTX

    cid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if req.stream:
        return StreamingResponse(
            _oai_stream(cid, created, model, messages, think, options),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    try:
        resp = await client.chat(model, messages, think=think, options=options)
    except OllamaError as e:
        raise HTTPException(status_code=502, detail=str(e))

    msg = resp.get("message", {})
    message_obj: dict[str, Any] = {"role": "assistant", "content": msg.get("content", "")}
    if msg.get("thinking"):
        message_obj["reasoning_content"] = msg["thinking"]

    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": message_obj, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": resp.get("prompt_eval_count", 0),
            "completion_tokens": resp.get("eval_count", 0),
            "total_tokens": (resp.get("prompt_eval_count", 0) or 0)
            + (resp.get("eval_count", 0) or 0),
        },
    }


async def _oai_stream(cid, created, model, messages, think, options):
    def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
        payload = {
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    yield chunk({"role": "assistant"})
    try:
        async for c in client.chat_stream(model, messages, think=think, options=options):
            m = c.get("message", {})
            if m.get("thinking"):
                yield chunk({"reasoning_content": m["thinking"]})
            if m.get("content"):
                yield chunk({"content": m["content"]})
            if c.get("done"):
                break
    except OllamaError as e:
        yield chunk({"content": f"\n[error: {e}]"})
    yield chunk({}, finish="stop")
    yield "data: [DONE]\n\n"


# --- Agents ---------------------------------------------------------------

@app.post("/agents", status_code=201)
async def create_agent(body: AgentCreate, _: None = Depends(require_auth)) -> dict[str, Any]:
    return agents.create(body.model_dump())


@app.get("/agents")
async def list_agents(_: None = Depends(require_auth)) -> dict[str, Any]:
    return {"data": db.list_agents()}


@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str, _: None = Depends(require_auth)) -> dict[str, Any]:
    return agents.require_agent(agent_id)


@app.patch("/agents/{agent_id}")
async def update_agent(agent_id: str, body: AgentUpdate, _: None = Depends(require_auth)) -> dict[str, Any]:
    agent = agents.require_agent(agent_id)
    return db.update_agent(agent["id"], body.model_dump(exclude_none=True))


@app.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str, _: None = Depends(require_auth)) -> dict[str, Any]:
    agent = agents.require_agent(agent_id)
    db.delete_agent(agent["id"])
    return {"deleted": agent["id"]}


@app.post("/agents/{agent_id}/chat")
async def agent_chat(agent_id: str, body: AgentChatRequest, _: None = Depends(require_auth)):
    agent = agents.require_agent(agent_id)
    kwargs = dict(
        message=body.message, session_id=body.session_id, reasoning=body.reasoning,
        use_memory=body.use_memory, remember=body.remember,
        temperature=body.temperature, num_ctx=body.num_ctx,
    )
    if body.stream:
        async def gen():
            async for ev in agents.chat_stream(agent, **kwargs):
                yield f"data: {json.dumps(ev)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)
    try:
        return await agents.chat(agent, **kwargs)
    except OllamaError as e:
        raise HTTPException(status_code=502, detail=str(e))


# --- Agent memory ---------------------------------------------------------

@app.post("/agents/{agent_id}/memory", status_code=201)
async def add_memory(agent_id: str, body: MemoryCreate, _: None = Depends(require_auth)) -> dict[str, Any]:
    agent = agents.require_agent(agent_id)
    return await memory.remember(agent["id"], body.content)


@app.get("/agents/{agent_id}/memory")
async def list_memory(agent_id: str, query: str | None = None, _: None = Depends(require_auth)) -> dict[str, Any]:
    agent = agents.require_agent(agent_id)
    if query:
        return {"data": await memory.recall(agent["id"], query, top_k=agent["top_k"])}
    items = [{"id": m["id"], "content": m["content"], "created_at": m["created_at"]}
             for m in db.get_memories(agent["id"])]
    return {"data": items}


@app.delete("/agents/{agent_id}/memory/{memory_id}")
async def delete_memory(agent_id: str, memory_id: str, _: None = Depends(require_auth)) -> dict[str, Any]:
    agent = agents.require_agent(agent_id)
    if not db.delete_memory(agent["id"], memory_id):
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"deleted": memory_id}


# --- Agent sessions / conversation history --------------------------------

@app.get("/agents/{agent_id}/sessions")
async def list_sessions(agent_id: str, _: None = Depends(require_auth)) -> dict[str, Any]:
    agent = agents.require_agent(agent_id)
    return {"data": db.list_sessions(agent["id"])}


@app.get("/agents/{agent_id}/sessions/{session_id}/messages")
async def session_messages(agent_id: str, session_id: str, _: None = Depends(require_auth)) -> dict[str, Any]:
    agent = agents.require_agent(agent_id)
    return {"data": db.get_history(agent["id"], session_id)}


@app.delete("/agents/{agent_id}/sessions/{session_id}")
async def clear_session(agent_id: str, session_id: str, _: None = Depends(require_auth)) -> dict[str, Any]:
    agent = agents.require_agent(agent_id)
    deleted = db.clear_session(agent["id"], session_id)
    return {"deleted_messages": deleted}
