#!/usr/bin/env bash
# Start the local LLM API backend. Creates a venv on first run.
set -euo pipefail
cd "$(dirname "$0")"

# Find a usable uv (it may not be on PATH on this machine).
UV="$(command -v uv || true)"
if [ -z "$UV" ] && [ -x "$HOME/.pyenv/versions/localAI/bin/uv" ]; then
  UV="$HOME/.pyenv/versions/localAI/bin/uv"
fi

if [ ! -d ".venv" ]; then
  echo "Creating virtualenv..."
  if [ -n "$UV" ]; then
    "$UV" venv --python 3.12
    "$UV" pip install -e .
  else
    python3 -m venv .venv
    .venv/bin/python -m pip install -e .
  fi
fi

# shellcheck disable=SC1091
source .venv/bin/activate

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"

# Optional HTTPS: set SSL_CERTFILE and SSL_KEYFILE to serve over TLS.
SSL_ARGS=()
SCHEME="http"
if [ -n "${SSL_CERTFILE:-}" ] && [ -n "${SSL_KEYFILE:-}" ]; then
  SSL_ARGS=(--ssl-certfile "$SSL_CERTFILE" --ssl-keyfile "$SSL_KEYFILE")
  SCHEME="https"
fi

echo "Starting llmmock on ${SCHEME}://${HOST}:${PORT}  (web client at /ui, docs at /docs)"
exec uvicorn llmmock.main:app --host "$HOST" --port "$PORT" "${SSL_ARGS[@]}" "$@"
