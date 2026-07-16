"""
Answer grader: checks whether the GENERATED ANSWER is actually supported
by the retrieved chunks, not hallucinated. This is a genuinely different
failure mode from the retrieval grader -- retrieval can score perfectly
(good chunks, high relevance) and the generator can still hallucinate or
overstate what the context actually supports. Phase 1 never checked this
at all; Phase 1's RAGAS faithfulness score was measured after the fact,
offline, not caught and corrected in the live request path.

Same reasoning as your Guidewire doc's answer-grader node: if the answer
isn't grounded, loop back toward retrieval rather than just returning a
possibly-wrong answer to the user.
"""

import os
from dotenv import load_dotenv

load_dotenv()

FAITHFULNESS_BACKEND = os.environ.get("GEN_BACKEND", "gemini")

FAITHFULNESS_PROMPT = """You are checking whether an answer is actually supported by its source context,
or whether it contains claims not backed by that context (hallucination).

Score 0.0-1.0: how well is EVERY claim in the answer supported by the context?
1.0 = every claim is directly supported. 0.0 = the answer makes claims the
context doesn't support at all, or contradicts the context.

Output ONLY the number, nothing else.

Context:
{context}

Answer to check:
{answer}

Faithfulness score (0.0-1.0):"""


def grade_answer(answer: str, chunks: list[dict]) -> float:
    if not answer or not chunks:
        return 0.0

    context = "\n\n".join(c["text"] for c in chunks)
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    response_text = _call_faithfulness_backend(prompt)

    import re
    match = re.search(r"[\d.]+", response_text.strip())
    if not match:
        print(f"WARNING: could not parse faithfulness score from response: {response_text[:200]!r}")
        return 0.0
    try:
        return max(0.0, min(1.0, float(match.group())))  # clamp to [0,1] -- don't trust the model to stay in range
    except ValueError:
        return 0.0


def _call_faithfulness_backend(prompt: str) -> str:
    from retry import retry_with_backoff, is_retryable_gemini_error, is_retryable_openrouter_error

    if FAITHFULNESS_BACKEND == "openrouter":
        from llm_openrouter import _call_openrouter, FALLBACK_MODELS
        return retry_with_backoff(
            lambda: _call_openrouter(prompt, fallback_models=FALLBACK_MODELS),
            retryable_check=is_retryable_openrouter_error,
        )
    if FAITHFULNESS_BACKEND == "ollama":
        import requests
        from generate import OLLAMA_URL, OLLAMA_MODEL
        response = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False})
        response.raise_for_status()
        return response.json()["response"]

    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY -- see .env.example")
    client = genai.Client(api_key=api_key)
    model = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
    response = retry_with_backoff(
        lambda: client.models.generate_content(model=model, contents=prompt),
        retryable_check=is_retryable_gemini_error,
    )
    return response.text


if __name__ == "__main__":
    # Sandbox-safe: verify parsing logic without a real API call
    class FakeMatch:
        pass

    def parse_only(response_text: str) -> float:
        import re
        match = re.search(r"[\d.]+", response_text.strip())
        if not match:
            return 0.0
        try:
            return max(0.0, min(1.0, float(match.group())))
        except ValueError:
            return 0.0

    assert parse_only("0.9") == 0.9
    assert parse_only("Score: 0.85, grounded in context") == 0.85
    assert parse_only("1.5") == 1.0  # clamped -- model went out of range
    assert parse_only("nonsense, no number") == 0.0
    print("Faithfulness score parsing verified: clean, embedded, out-of-range, and unparseable cases all handled correctly.")
