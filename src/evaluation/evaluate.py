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

from __future__ import annotations

import json
from pathlib import Path

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
        question, ground_truth = sample["question"], sample["ground_truth"]
        logger.info(f"[{i}/{len(EVAL_DATASET)}] Evaluating", question=question[:60])
        try:
            # ★ Chỉ chạy pipeline 1 LẦN (bug P0-4): contexts là context THẬT đã đưa vào LLM
            result = retriever.query_with_context(question)
            results.append(EvalResult(
                question=question,
                answer=result.answer,
                ground_truth=ground_truth,
                contexts=result.contexts,
            ))
            logger.info(f"[{i}/{len(EVAL_DATASET)}] Done",
                        answer_preview=result.answer[:80],
                        num_contexts=len(result.contexts))
        except Exception as e:
            logger.exception(f"[{i}/{len(EVAL_DATASET)}] Failed")
            results.append(EvalResult(
                question=question, answer=f"ERROR: {e}",
                ground_truth=ground_truth, contexts=[],
            ))

    # Tính metrics
    scores = evaluate_with_ragas(results)

    # In report + lưu kết quả để so sánh giữa các lần tune
    _print_report(results, scores)
    _save_results(results, scores)


def _save_results(results: list[EvalResult], scores: dict,
                  out_dir: str = "data/eval_runs") -> None:
    """Lưu kết quả để so sánh giữa các lần chạy."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    payload = {
        "scores": {k: (float(v) if isinstance(v, (int, float)) else str(v))
                   for k, v in scores.items()},
        "samples": [
            {"question": r.question, "answer": r.answer,
             "ground_truth": r.ground_truth, "num_contexts": len(r.contexts),
             "contexts": r.contexts}
            for r in results
        ],
    }
    path = Path(out_dir) / "latest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Eval results saved", path=str(path))


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
