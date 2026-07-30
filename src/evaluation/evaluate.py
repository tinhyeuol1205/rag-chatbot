from __future__ import annotations

"""
Evaluation Entry Point — Chạy đánh giá RAG pipeline.

Chạy bằng: make evaluate
Hoặc:      cd src && python -m evaluation.evaluate

Luồng xử lý:
  1. Load test dataset (10 câu hỏi + ground truth)
  2. Chạy từng câu hỏi qua RAG pipeline → thu answer + contexts
  3. Tính RAG Triad metrics (RAGAS hoặc simple fallback)
  4. In report
"""

from core import get_logger
from evaluation.dataset import EVAL_DATASET
from evaluation.metrics import EvalResult, evaluate_with_ragas
from retrieval.retriever import RAGRetriever

logger = get_logger(__name__)


def main():
    logger.info("Starting RAG evaluation", num_questions=len(EVAL_DATASET))

    retriever = RAGRetriever()
    results: list[EvalResult] = []

    for i, sample in enumerate(EVAL_DATASET, 1):
        question = sample["question"]
        ground_truth = sample["ground_truth"]

        logger.info(f"[{i}/{len(EVAL_DATASET)}] Evaluating", question=question[:60])

        try:
            # Chạy qua RAG pipeline
            answer = retriever.query(question, stream=False)

            # Thu thập contexts (từ retriever internals)
            # Simplified: chạy search riêng để lấy contexts
            expanded = retriever.expander.expand(question)
            search_results = retriever.searcher.search(question)
            contexts = [r["content"] for r in search_results[:5]]

            results.append(EvalResult(
                question=question,
                answer=answer,
                ground_truth=ground_truth,
                contexts=contexts,
            ))

            logger.info(
                f"[{i}/{len(EVAL_DATASET)}] Done",
                answer_preview=answer[:80],
            )

        except Exception as e:
            logger.error(f"[{i}/{len(EVAL_DATASET)}] Failed", error=str(e))
            results.append(EvalResult(
                question=question,
                answer=f"ERROR: {e}",
                ground_truth=ground_truth,
                contexts=[],
            ))

    # Tính metrics
    scores = evaluate_with_ragas(results)

    # In report
    _print_report(results, scores)


def _print_report(results: list[EvalResult], scores: dict):
    """In báo cáo đánh giá."""
    print("\n" + "=" * 70)
    print("📊 RAG EVALUATION REPORT")
    print("=" * 70)

    print(f"\nTotal questions: {len(results)}")
    print("\n--- Aggregate Scores ---")
    for metric, value in scores.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

    print("\n--- Per-Question Results ---")
    for i, r in enumerate(results, 1):
        status = "✅" if not r.answer.startswith("ERROR") else "❌"
        print(f"\n  {status} Q{i}: {r.question[:60]}")
        print(f"     Answer: {r.answer[:100]}...")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
