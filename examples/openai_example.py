"""Use the official OpenAI SDK against the local backend — no code changes beyond base_url.

    pip install openai
    python examples/openai_example.py
"""
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",  # any string works unless you set API_KEY on the server
)

MODEL = "gemma4:e2b-it-q4_K_M"

print("--- plain completion ---")
resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Give me one fun fact about octopuses."},
    ],
    max_tokens=80,
)
print(resp.choices[0].message.content)

print("\n--- with reasoning (reasoning_effort) ---")
resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "What is 17 * 23? Answer with the number only."}],
    reasoning_effort="medium",   # toggles the model's thinking mode
    max_tokens=300,
)
msg = resp.choices[0].message
print("answer:", msg.content)
# Thinking is returned in the non-standard 'reasoning_content' field.
print("thinking:", getattr(msg, "reasoning_content", None))

print("\n--- streaming ---")
stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Count from 1 to 5."}],
    stream=True,
    max_tokens=40,
)
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
print()
