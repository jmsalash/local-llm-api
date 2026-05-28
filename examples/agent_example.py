"""Drive the agents API: create an agent, give it memories + a directive, then chat.

    python examples/agent_example.py
"""
import httpx

BASE = "http://localhost:8080"


def main() -> None:
    with httpx.Client(base_url=BASE, timeout=300) as c:
        # Clean up a previous run if present.
        existing = c.get("/agents").json()["data"]
        for a in existing:
            if a["name"] == "research-assistant":
                c.delete(f"/agents/{a['id']}")

        # 1. Create an agent with its own directive and config.
        agent = c.post("/agents", json={
            "name": "research-assistant",
            "directive": (
                "You are a meticulous research assistant. Cite assumptions explicitly "
                "and keep answers under 4 sentences."
            ),
            "reasoning": False,        # default thinking mode for this agent
            "temperature": 0.4,
            "num_ctx": 8192,
            "max_history": 12,         # how many past turns to feed back in
            "top_k": 4,                # semantic memories injected per turn
        }).json()
        aid = agent["id"]
        print("created agent:", aid)

        # 2. Teach it some long-term memories (persist across sessions/restarts).
        for fact in [
            "The project codename is 'Bluebird'.",
            "The team ships on Fridays and freezes merges on Thursdays.",
            "We use Postgres 16 and deploy on Kubernetes.",
        ]:
            c.post(f"/agents/{aid}/memory", json={"content": fact})
        print("stored memories")

        # 3. Chat — relevant memories are retrieved automatically.
        r = c.post(f"/agents/{aid}/chat", json={
            "message": "When can I merge a non-urgent PR this week?",
            "session_id": "demo",
            "max_tokens": 120,
        }).json()
        print("\nQ: When can I merge a non-urgent PR this week?")
        print("A:", r["content"])
        print("memories used:", [m["content"] for m in r["memories_used"]])

        # 4. Turn reasoning on for a single, harder turn.
        r = c.post(f"/agents/{aid}/chat", json={
            "message": "If the freeze is Thursday and today is Wednesday, how many days can I still merge?",
            "session_id": "demo",
            "reasoning": True,
            "max_tokens": 300,
        }).json()
        print("\nQ (with reasoning): days left to merge?")
        print("A:", r["content"])
        print("thinking:", (r["thinking"] or "")[:200])


if __name__ == "__main__":
    main()
