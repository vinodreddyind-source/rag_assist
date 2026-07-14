"""
RUN ON YOUR LAPTOP -- needs an OPENROUTER_API_KEY (free, no card, from
https://openrouter.ai/keys).

OpenRouter is an OpenAI-compatible gateway in front of 300+ models,
including ~25 genuinely free ones (IDs ending in :free). Two things make
it worth having as a third backend alongside Ollama and direct Gemini:

1. One API shape for many providers -- switching models is a string
   change, not a new SDK.
2. Built-in provider fallback: you can pass a list of backup models, and
   OpenRouter retries against the next one if the first is rate-limited or
   down. This is the platform-level version of the retry.py module in
   this same folder -- that module retries the SAME model after a delay;
   OpenRouter's fallback list retries a DIFFERENT model immediately. Good
   to know the difference if asked "how would you make this more
   resilient."

Free tier limits (verified mid-2026): 20 requests/minute, 50/day (rises to
1000/day permanently after ever purchasing $10 in credits -- never
required). Free-model roster rotates; check https://openrouter.ai/models
filtered to $0 pricing before assuming a specific model ID still works.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Rotates -- verify at https://openrouter.ai/models before relying on this
DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
FALLBACK_MODELS = ["google/gemma-3-12b-it:free", "qwen/qwen-2.5-7b-instruct:free"]


def _call_openrouter(prompt: str, model: str = DEFAULT_MODEL,
                      fallback_models: list[str] | None = None) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY -- see .env.example")

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if fallback_models:
        body["models"] = [model] + fallback_models  # OpenRouter's own fallback list

    response = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def generate_openrouter(prompt: str) -> str:
    return _call_openrouter(prompt, fallback_models=FALLBACK_MODELS)


def rerank_prompt(query: str, document: str) -> str:
    return f"""Score how relevant this document chunk is to the query, on a scale of 0.0 to 1.0.
Output ONLY the number, nothing else.

Query: {query}

Document: {document}

Relevance score (0.0-1.0):"""


def rerank_with_openrouter(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    import re
    scored = []
    for c in candidates:
        response_text = _call_openrouter(rerank_prompt(query, c["text"]), fallback_models=FALLBACK_MODELS)
        match = re.search(r"[\d.]+", response_text.strip())
        score = float(match.group()) if match else 0.0
        scored.append((c, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _score in scored[:top_n]]


if __name__ == "__main__":
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Set OPENROUTER_API_KEY to test this -- get one free at https://openrouter.ai/keys")
    else:
        answer = generate_openrouter("Say 'OpenRouter is working' and nothing else.")
        print(f"Response: {answer}")
