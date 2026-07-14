"""
RUN ON YOUR LAPTOP.
CI-gate style evaluation: run every golden question through the pipeline,
score faithfulness + answer relevancy with RAGAS, fail the "build" if either
drops below threshold. This is the real version of section 9 in your
interview doc -- same gate, actually executed against your own pipeline.

RAGAS needs a judge LLM. Ollama works via ragas' LangchainLLMWrapper +
langchain_community.chat_models.ChatOllama -- no OpenAI key needed, fully
local, just slower than GPT-4.1 would be as judge.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from retrieval import load_children, HybridRetriever
from query_processing import expand_acronyms
from rerank import rerank
from generate import generate_answer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

FAITHFULNESS_THRESHOLD = 0.85
RELEVANCY_THRESHOLD = 0.80


def run_pipeline_on_golden_set():
    with open(os.path.join(DATA_DIR, "golden_qa.json")) as f:
        golden = json.load(f)

    children = load_children()
    retriever = HybridRetriever(children)

    records = []
    for item in golden:
        query = item["question"]
        expanded = expand_acronyms(query)
        candidates = [c for c, _ in retriever.hybrid_search(expanded, k=10)]
        top_chunks = rerank(query, candidates, top_n=5)
        answer = generate_answer(query, top_chunks)

        records.append({
            "question": query,
            "answer": answer,
            "contexts": [c["text"] for c in top_chunks],
            "expected_source": item["expected_source"],
            "type": item["type"],
        })
    return records


def evaluate_with_ragas(records: list[dict]):
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas.llms import LangchainLLMWrapper
    from langchain_community.chat_models import ChatOllama

    judge = LangchainLLMWrapper(ChatOllama(model="llama3.1:8b"))

    # context_precision and context_recall both need a "ground truth" answer
    # to compare retrieved context against -- that's why golden_qa.json needs
    # an expected-answer field, not just an expected-source. If you only have
    # expected_source (as our starter golden set does), context_recall can't
    # be computed meaningfully -- add an `expected_answer` string per item to
    # unlock it. This is a real, explainable gap, not a bug to hide.
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
        print("NOTE: skipping context_precision/context_recall -- golden_qa.json "
              "needs an 'expected_answer' field per item to compute these. "
              "faithfulness/answer_relevancy don't need it.")

    results = evaluate(dataset=dataset, metrics=metrics, llm=judge)
    return results


def ci_gate():
    print("Running golden set through the pipeline...")
    records = run_pipeline_on_golden_set()

    print("Scoring with RAGAS (this calls the local Ollama judge model, be patient)...")
    results = evaluate_with_ragas(records)

    print(f"\nFaithfulness:     {results['faithfulness']:.3f}  (threshold {FAITHFULNESS_THRESHOLD})")
    print(f"Answer relevancy: {results['answer_relevancy']:.3f}  (threshold {RELEVANCY_THRESHOLD})")

    passed = (results["faithfulness"] >= FAITHFULNESS_THRESHOLD and
              results["answer_relevancy"] >= RELEVANCY_THRESHOLD)

    print("\nCI GATE:", "PASS" if passed else "FAIL - deployment would be blocked")
    return passed


if __name__ == "__main__":
    ci_gate()
