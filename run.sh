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
echo "Starting llmmock on http://${HOST}:${PORT}  (docs at /docs)"
exec uvicorn llmmock.main:app --host "$HOST" --port "$PORT" "$@"
