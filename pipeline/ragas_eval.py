"""
RUN ON YOUR LAPTOP.
CI-gate style evaluation: run every golden question through the pipeline,
score faithfulness + answer relevancy with RAGAS, fail the "build" if either
drops below threshold. This is the real version of section 9 in your
interview doc -- same gate, actually executed against your own pipeline.

RAGAS needs a judge LLM -- defaults to Gemini (matching GEN_BACKEND's
default), since Ollama isn't what you're actually running. Set
RAGAS_JUDGE_BACKEND=ollama in .env if you do have Ollama and want a fully
local judge instead.

This now uses the REAL pipeline (Qdrant hybrid retriever with real
embeddings, backend-selectable reranker) instead of the earlier BM25-only
retrieval.py stub -- running the eval against a different pipeline than
what's actually deployed would make the gate meaningless.
"""

import json
import os
import sys
import asyncio
from dotenv import load_dotenv

load_dotenv()

# WORKAROUND ATTEMPT 1: Windows defaults to ProactorEventLoop, which has
# known incompatibilities with some asyncio timeout/task-context patterns
# that SelectorEventLoop doesn't share. This is a real, common fix for
# exactly this class of Windows-specific asyncio error -- try it before
# assuming the problem is Python 3.14 itself.
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
        # Newer Python may have removed/renamed this -- don't let the
        # workaround attempt itself crash the script if so.
        print("NOTE: WindowsSelectorEventLoopPolicy not available on this "
              "Python version -- skipping that workaround attempt.")

# WORKAROUND ATTEMPT 2: nest_asyncio patches event-loop nesting issues.
# Different failure mode than attempt 1 above -- keep both, they don't
# conflict, and between them cover more of the possible causes.
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass  # not installed -- `pip install nest_asyncio` if needed

sys.path.insert(0, os.path.dirname(__file__))
from retrieval_qdrant import QdrantHybridRetriever
from query_processing import expand_acronyms, route_product
from generate import generate_answer

RERANK_BACKEND = os.environ.get("RERANK_BACKEND", "gemini")
RERANK_FALLBACK_TO_OLLAMA = os.environ.get("GEN_FALLBACK_TO_OLLAMA", "true").lower() == "true"

if RERANK_BACKEND == "gemini":
    from rerank_gemini import rerank_with_gemini as _primary_rerank
elif RERANK_BACKEND == "openrouter":
    from llm_openrouter import rerank_with_openrouter as _primary_rerank
elif RERANK_BACKEND == "ollama":
    from rerank_ollama import rerank_with_ollama as _primary_rerank
else:
    from rerank import rerank as _primary_rerank


def rerank(query: str, candidates: list, top_n: int = 5):
    """Same fallback pattern as app/main.py -- if the primary reranker hits
    a real quota wall mid-eval-run, drop to local Ollama instead of killing
    the entire RAGAS run over one question's rerank call."""
    if RERANK_BACKEND == "ollama":
        return _primary_rerank(query, candidates, top_n=top_n)
    try:
        return _primary_rerank(query, candidates, top_n=top_n)
    except Exception as e:
        if not RERANK_FALLBACK_TO_OLLAMA:
            raise
        print(f"  WARNING: {RERANK_BACKEND} reranker failed ({type(e).__name__}: {e}). "
              f"Falling back to local Ollama for this question...")
        from rerank_ollama import rerank_with_ollama
        try:
            return rerank_with_ollama(query, candidates, top_n=top_n)
        except Exception:
            raise e

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

FAITHFULNESS_THRESHOLD = 0.85
RELEVANCY_THRESHOLD = 0.80


def run_pipeline_on_golden_set():
    with open(os.path.join(DATA_DIR, "golden_qa.json")) as f:
        golden = json.load(f)

    retriever = QdrantHybridRetriever()  # loads real embeddings, same as app/main.py

    records = []
    for item in golden:
        query = item["question"]
        expanded = expand_acronyms(query)
        product = route_product(expanded)
        candidates = [c for c, _ in retriever.hybrid_search(expanded, k=10, product_filter=product)]
        top_chunks = rerank(query, candidates, top_n=5)
        answer = generate_answer(query, top_chunks)

        print(f"  Processed: {query[:60]}...")

        records.append({
            "question": query,
            "answer": answer,
            "contexts": [c["text"] for c in top_chunks],
            "expected_source": item["expected_source"],
            "expected_answer": item.get("expected_answer", ""),
            "type": item["type"],
        })
    return records


def _get_judge():
    judge_backend = os.environ.get("RAGAS_JUDGE_BACKEND", "gemini")
    from ragas.llms import LangchainLLMWrapper

    if judge_backend == "ollama":
        from langchain_community.chat_models import ChatOllama
        return LangchainLLMWrapper(ChatOllama(model=os.environ.get("OLLAMA_MODEL", "llama3.2:3b")))

    # Default: Gemini. Uses a stronger-than-generation model as judge is the
    # ideal (your Guidewire doc's pattern: GPT-4.1 judge vs GPT-4.1-mini
    # grader), but gemini-flash-latest for both is a reasonable simplification
    # at this project's scale -- evaluation quality matters more than cost
    # since it runs on a small golden set, not every production query.
    from langchain_google_genai import ChatGoogleGenerativeAI
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY, or set RAGAS_JUDGE_BACKEND=ollama if you have Ollama running.")
    model = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
    return LangchainLLMWrapper(ChatGoogleGenerativeAI(model=model, google_api_key=api_key))


def _get_embeddings():
    """RAGAS's evaluate() needs an embeddings model internally (answer_relevancy
    specifically uses it to compare semantic similarity between the generated
    question and the original). Without this, it silently defaults to OpenAI
    embeddings and fails on a missing OPENAI_API_KEY -- we don't use OpenAI
    anywhere in this project. Reuse the SAME local all-MiniLM-L6-v2 model
    already used for retrieval instead: zero extra API calls, no additional
    quota risk, and consistent with the rest of the project's embedding
    choice."""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))


def evaluate_with_ragas(records: list[dict]):
    from datasets import Dataset
    from ragas import evaluate
    from ragas.run_config import RunConfig
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

    judge = _get_judge()
    embeddings = _get_embeddings()

    # context_precision and context_recall both need a "ground truth" answer
    # to compare retrieved context against -- that's why golden_qa.json needs
    # an expected-answer field, not just an expected-source.
    dataset = Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r.get("expected_answer", ""),
        }
        for r in records
    ])

    metrics = [faithfulness, answer_relevancy]
    if all(r.get("expected_answer") for r in records):
        metrics += [context_precision, context_recall]
    else:
        print("NOTE: skipping context_precision/context_recall -- some golden_qa.json "
              "items are missing 'expected_answer'.")

    results = evaluate(dataset=dataset, metrics=metrics, llm=judge, embeddings=embeddings,
                        run_config=RunConfig(max_workers=1))
    return results


def _mean_score(results, metric_name: str) -> float:
    """results[metric_name] is a LIST of per-question scores (ragas's
    EvaluationResult.__getitem__ returns t.List[float]), not an aggregate --
    this is the actual bug that crashed the .3f formatting, independent of
    whatever caused the underlying scores to be NaN. Use nanmean so a few
    failed jobs (NaN) don't silently zero out or crash the whole average."""
    import numpy as np
    scores = results[metric_name]
    return float(np.nanmean(scores))


def ci_gate():
    print("Running golden set through the REAL pipeline (Qdrant + real embeddings)...")
    records = run_pipeline_on_golden_set()

    print("\nScoring with RAGAS (judge model call per metric per question -- be patient)...")
    results = evaluate_with_ragas(records)

    faithfulness_score = _mean_score(results, "faithfulness")
    relevancy_score = _mean_score(results, "answer_relevancy")

    print(f"\nFaithfulness:      {faithfulness_score:.3f}  (threshold {FAITHFULNESS_THRESHOLD})")
    print(f"Answer relevancy:  {relevancy_score:.3f}  (threshold {RELEVANCY_THRESHOLD})")
    if "context_precision" in results._scores_dict:
        print(f"Context precision: {_mean_score(results, 'context_precision'):.3f}")
    if "context_recall" in results._scores_dict:
        print(f"Context recall:    {_mean_score(results, 'context_recall'):.3f}")

    import math
    if math.isnan(faithfulness_score) or math.isnan(relevancy_score):
        print("\nWARNING: NaN scores mean the underlying per-question evaluation jobs "
              "failed (check the 'Exception raised in Job[N]' lines above) -- this is "
              "NOT a passing or failing quality result, it's a broken run. Fix the "
              "underlying error before trusting any PASS/FAIL verdict below.")

    passed = (not math.isnan(faithfulness_score) and not math.isnan(relevancy_score) and
              faithfulness_score >= FAITHFULNESS_THRESHOLD and
              relevancy_score >= RELEVANCY_THRESHOLD)

    print("\nCI GATE:", "PASS" if passed else "FAIL - deployment would be blocked")
    return passed


if __name__ == "__main__":
    ci_gate()
