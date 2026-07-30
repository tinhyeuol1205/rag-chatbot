from __future__ import annotations

"""
RAG Triad Metrics — Đo chất lượng pipeline RAG.

3 chỉ số cốt lõi (RAG Triad):

1. Context Relevance (0-1):
   "Chunks lấy về có liên quan đến câu hỏi?"
   → Cao = Retrieval tốt, ít nhiễu
   → Thấp = Search kém, lấy documents sai

2. Faithfulness (0-1):
   "Câu trả lời có HOÀN TOÀN dựa trên context?"
   → Cao = LLM trả lời dựa trên tài liệu
   → Thấp = LLM tự bịa (hallucination!)

3. Answer Relevance (0-1):
   "Câu trả lời có đúng trọng tâm câu hỏi?"
   → Cao = Trả lời đúng ý
   → Thấp = Trả lời lạc đề

Kịch bản chẩn đoán:
  Context Relevance ↑ + Faithfulness ↓ = LLM bỏ qua context, tự đoán
  Context Relevance ↓ + Faithfulness ↑ = Retrieval sai, LLM cố trả lời từ context sai
  Cả 3 ↓ = Pipeline cần sửa toàn diện

Tham khảo: rag_master.md — Module 8, mục 8.1
"""

from dataclasses import dataclass

from core import get_logger

logger = get_logger(__name__)


@dataclass
class EvalResult:
    """Kết quả đánh giá cho 1 câu hỏi."""

    question: str
    answer: str
    ground_truth: str
    contexts: list[str]
    context_relevance: float = 0.0
    faithfulness: float = 0.0
    answer_relevance: float = 0.0


def evaluate_with_ragas(results: list[EvalResult]) -> dict:
    """Chạy RAGAS evaluation trên tập kết quả.

    Args:
        results: Danh sách EvalResult (mỗi cái = 1 câu hỏi đã test)

    Returns:
        Dict chứa scores trung bình + chi tiết từng câu
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness

        # Chuyển sang format RAGAS yêu cầu
        data = {
            "question": [r.question for r in results],
            "answer": [r.answer for r in results],
            "ground_truth": [r.ground_truth for r in results],
            "contexts": [r.contexts for r in results],
        }

        dataset = Dataset.from_dict(data)

        # Chạy evaluation
        logger.info("Running RAGAS evaluation", num_samples=len(results))
        scores = evaluate(
            dataset=dataset,
            metrics=[context_precision, faithfulness, answer_relevancy],
        )

        logger.info("RAGAS evaluation complete", scores=scores)
        return dict(scores)

    except ImportError:
        logger.warning("RAGAS not installed, using simple evaluation fallback")
        return _simple_evaluate(results)


def _simple_evaluate(results: list[EvalResult]) -> dict:
    """Fallback evaluation đơn giản (không cần RAGAS).

    Đo lường cơ bản:
    - answer_length: Độ dài trung bình câu trả lời
    - has_context: Tỷ lệ câu có context
    - keyword_overlap: Overlap giữa answer và ground_truth
    """
    total = len(results)
    if total == 0:
        return {}

    keyword_scores = []
    for r in results:
        # Đo keyword overlap giữa answer và ground_truth
        answer_words = set(r.answer.lower().split())
        truth_words = set(r.ground_truth.lower().split())
        if truth_words:
            overlap = len(answer_words & truth_words) / len(truth_words)
            keyword_scores.append(overlap)

    return {
        "num_samples": total,
        "avg_answer_length": sum(len(r.answer) for r in results) / total,
        "avg_keyword_overlap": sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0,
        "has_context_ratio": sum(1 for r in results if r.contexts) / total,
    }
