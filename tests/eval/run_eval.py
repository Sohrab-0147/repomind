"""RAGAS evaluation harness for RepoMind."""
import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from repomind.agent.orchestrator import handle_query
from repomind.context.retrievers.semantic_qdrant import retrieve
from repomind.observability.logger import get_logger

logger = get_logger(__name__)

DATASET = Path(__file__).parent / "dataset.json"


def run() -> dict:
    with open(DATASET) as f:
        cases = json.load(f)

    questions, answers, contexts, ground_truths = [], [], [], []

    for i, case in enumerate(cases):
        q = case["question"]
        logger.info(f"[{i+1}/{len(cases)}] {q}")
        try:
            chunks = retrieve(q, k=20, top_n=5)
            contexts.append([c["content"] for c in chunks])
            answers.append(handle_query(q, thread_id=f"eval-{i}"))
            questions.append(q)
            ground_truths.append(case["ground_truth"])
        except Exception as e:
            logger.error(f"Failed on {q}: {e}")

    ds = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    print("\n" + "=" * 60)
    print("RepoMind Eval Results")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k:24s}  {v:.4f}")
    print("=" * 60)
    return result


if __name__ == "__main__":
    run()
