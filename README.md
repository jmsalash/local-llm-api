# llmmock — local LLM API backend for development

A small FastAPI service that gives you a **free, local** LLM API so you can develop and
test applications without spending money on hosted APIs. It runs models through
[ollama](https://ollama.com) and adds the pieces app development actually needs:

- **OpenAI-compatible endpoint** (`/v1/chat/completions`, `/v1/models`) — point any
  existing app or the `openai` SDK at it by changing only the `base_url`.
- **Reasoning on/off** per request (maps to the model's native thinking mode).
- **Configurable context** (`num_ctx`), temperature, max tokens.
- **Agents**: named agents that each keep their **own directive (system prompt)**,
  **own config**, **conversation history**, and a **long-term semantic memory** so you
  can give them context in advance.
- **Memory**: per-agent facts are embedded and the relevant ones are retrieved and
  injected into each turn automatically.
- Everything persists to a local SQLite file. No external services, no API keys.

## Model choice

The default model is `gemma4:e2b-it-q4_K_M` (Gemma 4 E2B: ~7 GB, 128K context, supports
**thinking**, tools, vision, audio) — a good all-round default that runs on a single
consumer GPU. Override it per request (`model` field) or per agent, or change
`DEFAULT_MODEL`.

Some other models that run well on ~8 GB of VRAM:

| Model | Why | Reasoning toggle |
|-------|-----|------------------|
| `gemma4:e2b-it-q4_K_M` *(default)* | multimodal, 128K ctx | native (`think`) |
| `qwen3:4b` / `qwen3:8b` | strong, explicit `/think` `/no_think` switch | native, cleanest |
| `llama3.1:8b` | solid general 8B | prompt-layer only |

VRAM is usually the binding constraint: a quantized model needs roughly its file size in
VRAM plus headroom for the context window. Models in the ~27B+ class won't fit in 8 GB
and will offload to CPU (slow), so prefer ≤8–9B quantized models on smaller cards. You
can pull more models from the **Models** tab in the web client (see below).

## Setup & run

Requires a running ollama (`ollama serve`) with the model pulled
(`ollama pull gemma4:e2b-it-q4_K_M`).

```bash
./run.sh                      # creates .venv, installs deps, starts on :8080
# or manually:
uv venv --python 3.12 && uv pip install -e .
.venv/bin/uvicorn llmmock.main:app --host 0.0.0.0 --port 8080
```

Config is via env vars or a `.env` file (see `.env.example`). Interactive API docs at
`http://localhost:8080/docs`.

## Web test client

A zero-build, single-page client is served at **`http://localhost:8080/ui`**. Use it to
exercise the backend by hand:

- **Chat tab** — OpenAI-compatible chat: pick a model, set the system prompt,
  temperature / `num_ctx` / `max_tokens`, toggle **Reasoning** and **Stream**. When
  reasoning is on, the model's thinking shows in a collapsible "reasoning" section under
  the answer.
- **Agents tab** — create agents with their own directive/config, add and delete
  long-term memories, and chat per session. Replies show which memories were retrieved.
- **Models tab** — list installed models (with size), delete them, and **download new
  ones** with a live progress bar. Accepts an ollama tag (`qwen3:4b`) or a HuggingFace
  GGUF repo (`hf.co/<user>/<repo>:<quant>`, e.g.
  `hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M`).

CORS is enabled (any origin) so you can also open `web/index.html` directly or from a
different port and point the **Base URL** field at the server.

### Managing models via the API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/models` | Installed models with size/params/quantization |
| POST | `/models/pull` | Start a background download (body `{"model": "...", "quantization": "Q4_K_M"}`); returns immediately |
| GET | `/models/pulls` | Active/just-finished downloads with live progress (poll this) |
| POST | `/models/delete` | Remove a model (body `{"model": "..."}`) |
| POST | `/models/check` | Load the model and generate one token; returns `{ok}` (plus `error`/`hint` if it can't run here) |

**Where to find model names:** browse [ollama.com/library](https://ollama.com/library) and
copy the name + tag (e.g. `qwen3:4b`) — these are best supported. Or use a
[HuggingFace GGUF repo](https://huggingface.co/models?library=gguf) and paste its URL.

**Will it run here?** Not every GGUF loads on every ollama version (e.g. ollama 0.20.5's
llama.cpp can't load Gemma-4 GGUFs — use the `gemma4:*` ollama-registry build instead).
Use `POST /models/check` (or the **Test** button in the web client) to load a model and
confirm it actually runs before relying on it; the web client also auto-tests a model
right after it finishes downloading.

`POST /models/pull` accepts an ollama tag, an `hf.co/...` repo, or a full HuggingFace URL
(it normalizes the URL and appends the quant). The pull runs server-side, so it keeps
going even if the client disconnects, and any client can watch it via `GET /models/pulls`
— that's what the web client's progress bars poll.

Downloading from HuggingFace requires a repo that contains **GGUF** files; append the
quantization as a tag (`:Q4_K_M`). This is handled by ollama under the hood, so anything
`ollama pull` supports works here too.

## Use it like the OpenAI API

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

r = client.chat.completions.create(
    model="gemma4:e2b-it-q4_K_M",
    messages=[{"role": "user", "content": "Hello!"}],
    reasoning_effort="medium",   # omit to disable reasoning
)
print(r.choices[0].message.content)
# thinking (when enabled) is in r.choices[0].message.reasoning_content
```

Reasoning is enabled by any of: `reasoning_effort` in `{"low","medium","high"}`, or a
`think: true` field. Context size: pass `num_ctx`. See `examples/openai_example.py`.

## Agents

Agents wrap the model with a persistent identity + memory.

```bash
# Create an agent with its own directive and config
curl -X POST localhost:8080/agents -d '{
  "name":"alfred",
  "directive":"You are Alfred, a concise butler. Address the user as sir.",
  "reasoning":false, "temperature":0.3, "max_history":10, "top_k":5
}'

# Give it long-term memory (persists across sessions and restarts)
curl -X POST localhost:8080/agents/alfred/memory -d '{"content":"User prefers Earl Grey tea."}'

# Chat — relevant memories are retrieved and injected automatically
curl -X POST localhost:8080/agents/alfred/chat -d '{
  "message":"What should I drink?", "session_id":"s1"
}'
```

You can reference an agent by its `id` or its `name` in the URL.

### Agent endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/agents` | Create agent (name, directive, model, reasoning, temperature, num_ctx, max_history, top_k) |
| GET | `/agents` | List agents |
| GET/PATCH/DELETE | `/agents/{id}` | Get / update directive+config / delete |
| POST | `/agents/{id}/chat` | Chat. Body: `message`, `session_id`, `reasoning?`, `use_memory?`, `remember?`, `stream?`, `temperature?`, `num_ctx?` |
| POST | `/agents/{id}/memory` | Add a long-term memory |
| GET | `/agents/{id}/memory?query=...` | List all memories, or semantic-search with `query` |
| DELETE | `/agents/{id}/memory/{mem_id}` | Forget a memory |
| GET | `/agents/{id}/sessions` | List conversation threads |
| GET | `/agents/{id}/sessions/{sid}/messages` | Full history of a thread |
| DELETE | `/agents/{id}/sessions/{sid}` | Clear a thread |

### Two kinds of memory

- **Conversation memory** — every turn in a `session_id` is stored and replayed (up to
  `max_history` messages), so the agent remembers the current dialogue.
- **Long-term semantic memory** — facts you `POST` to `/memory` are embedded
  (`nomic-embed-text`). On each turn the agent embeds the user message, retrieves the
  `top_k` most similar facts above `MEMORY_MIN_SCORE`, and injects them into the system
  prompt. This is the "give it context in advance" mechanism. Set `use_memory:false` on
  a turn to skip it, or `remember:true` to also store the user message as a memory.

## Config (env / `.env`)

| Var | Default | Meaning |
|-----|---------|---------|
| `OLLAMA_URL` | `http://localhost:11434` | ollama endpoint |
| `DEFAULT_MODEL` | `gemma4:e2b-it-q4_K_M` | fallback model |
| `EMBED_MODEL` | `nomic-embed-text` | embeddings for memory |
| `DEFAULT_NUM_CTX` | `8192` | context window |
| `DEFAULT_TEMPERATURE` | `0.7` | sampling temp |
| `MEMORY_TOP_K` | `5` | memories injected per turn |
| `MEMORY_MIN_SCORE` | `0.35` | min cosine similarity to inject |
| `API_KEY` | *(empty)* | if set, require `Authorization: Bearer <key>` |
| `HOST` / `PORT` | `0.0.0.0` / `8080` | bind address |

## Layout

```
llmmock/
  config.py         settings + .env loader
  ollama_client.py  async ollama wrapper (chat / think / embed / stream)
  db.py             sqlite: agents, messages, memories
  memory.py         embed + cosine recall
  agents.py         orchestration: directive + memory + history -> model
  main.py           FastAPI routes (OpenAI-compat + agents) + /ui + CORS
web/index.html      single-page test client served at /ui
examples/           openai_example.py, agent_example.py, curl_examples.sh
```
