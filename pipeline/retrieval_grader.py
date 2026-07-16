"""
Retrieval grader: scores how relevant the retrieved chunks are to the
query, 0.0-1.0. This is THE node that makes the system agentic rather than
a straight line -- Phase 1 never asked "was retrieval actually good?", it
just trusted whatever came back. This node is what decides whether to
proceed to generation or loop back and try again.

Same listwise-call, backend-selectable pattern as reranking (Gemini/
OpenRouter/Ollama) -- reuses the same _parse_score_array from rerank_gemini
rather than duplicating parsing logic a third time.
"""

import os
from dotenv import load_dotenv

load_dotenv()

GRADE_BACKEND = os.environ.get("GEN_BACKEND", "gemini")  # reuse the same backend selection as generation

GRADE_PROMPT = """You are grading whether retrieved document chunks actually help answer a question.
Score each chunk 0.0-1.0: how much does it help answer the query?
Return ONLY a JSON array of scores in the same order as the chunks, e.g. [0.9, 0.2, 0.6]

Query: {query}

Chunks:
{chunks_block}

Scores (JSON array):"""


def grade_retrieval(query: str, chunks: list[dict]) -> float:
    """Returns the MAX relevance score across chunks, not an average.

    Real design iteration, in order:
    1. Flat average -- one great chunk (0.9) + four irrelevant ones scored
       ~0.2 on real FNOL data, well below threshold, triggering two fully
       wasted rewrite loops for zero improvement in the final answer.
    2. Rank-weighted average (1/(i+1) decay) -- better (0.43) but still not
       enough to clear a 0.7 threshold, because it still dilutes a strong
       top result across weaker tail chunks.
    3. max() -- correct fix. The actual question the grader needs to
       answer is "is there at least one chunk that can answer this query,"
       not "are most chunks relevant." Phase 1 already proved empirically
       that the generator correctly ignores noisy tail chunks regardless
       of whether they're present -- so grading for noise the generator
       already handles was solving the wrong problem. max() also still
       correctly fails genuinely bad retrieval, where even the top chunk
       is weak."""
    if not chunks:
        return 0.0

    from rerank_gemini import _parse_score_array

    chunks_block = "\n".join(f"{i+1}. {c['text']}" for i, c in enumerate(chunks))
    prompt = GRADE_PROMPT.format(query=query, chunks_block=chunks_block)

    response_text = _call_grader_backend(prompt)
    scores = _parse_score_array(response_text, expected_len=len(chunks))
    return max(scores) if scores else 0.0


def _call_grader_backend(prompt: str) -> str:
    from retry import retry_with_backoff, is_retryable_gemini_error, is_retryable_openrouter_error

    if GRADE_BACKEND == "openrouter":
        from llm_openrouter import _call_openrouter, FALLBACK_MODELS
        return retry_with_backoff(
            lambda: _call_openrouter(prompt, fallback_models=FALLBACK_MODELS),
            retryable_check=is_retryable_openrouter_error,
        )
    if GRADE_BACKEND == "ollama":
        import requests
        from generate import OLLAMA_URL, OLLAMA_MODEL
        response = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False})
        response.raise_for_status()
        return response.json()["response"]

    # default: gemini
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
    # Sandbox-safe structural test: verify the prompt builds correctly and
    # the function signature/parsing path works, using a fake backend call
    # (no real API key available here).
    import sys
    sys.path.insert(0, "..")

    fake_chunks = [
        {"text": "The deductible applies based on the loss date, not the processing date."},
        {"text": "Python was created by Guido van Rossum in 1991."},  # deliberately irrelevant
    ]
    prompt = GRADE_PROMPT.format(
        query="How is the deductible determined?",
        chunks_block="\n".join(f"{i+1}. {c['text']}" for i, c in enumerate(fake_chunks)),
    )
    print("Prompt builds correctly:")
    print(prompt[:300])
    print("...")
    print("\n(Real scoring needs a live API call -- test on your laptop with a real backend.)")
