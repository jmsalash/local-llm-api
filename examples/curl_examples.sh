#!/usr/bin/env bash
# Quick curl tour of the API. Assumes the server is on :8080.
set -euo pipefail
B=http://localhost:8080

echo "### health"
curl -s $B/health | python3 -m json.tool

echo "### OpenAI: list models"
curl -s $B/v1/models | python3 -m json.tool

echo "### OpenAI: chat completion (no reasoning)"
curl -s $B/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"gemma4:e2b-it-q4_K_M",
  "messages":[{"role":"user","content":"Hello in one word"}],
  "max_tokens":20
}' | python3 -m json.tool

echo "### OpenAI: chat completion WITH reasoning"
curl -s $B/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"gemma4:e2b-it-q4_K_M",
  "messages":[{"role":"user","content":"2+2*2 = ? number only"}],
  "reasoning_effort":"medium",
  "max_tokens":200
}' | python3 -m json.tool

echo "### Create an agent"
curl -s -X POST $B/agents -H 'Content-Type: application/json' -d '{
  "name":"demo-bot",
  "directive":"You are a helpful, very brief assistant.",
  "reasoning":false
}' | python3 -m json.tool

echo "### Add a memory to it (use the id printed above)"
echo "curl -X POST $B/agents/<AGENT_ID>/memory -d '{\"content\":\"User name is Mano.\"}'"

echo "### Chat with the agent"
echo "curl -X POST $B/agents/<AGENT_ID>/chat -d '{\"message\":\"What is my name?\",\"session_id\":\"s1\"}'"
