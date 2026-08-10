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

from __future__ import annotations

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
    """Chạy RAGAS evaluation trên tập kết quả, fallback nếu thất bại.

    Args:
        results: Danh sách EvalResult (mỗi cái = 1 câu hỏi đã test)

    Returns:
        Dict chứa scores trung bình + chi tiết từng câu
    """
    if not results:
        return {}
    try:
        return _ragas_evaluate(results)
    except ImportError as e:
        logger.warning("RAGAS not installed, using simple fallback", error=str(e))
    except Exception:
        # ★ Bắt RỘNG: schema mismatch, judge API fail, timeout... đều phải fallback
        logger.exception("RAGAS evaluation failed, using simple fallback")
    return _simple_evaluate(results)


def _ragas_evaluate(results: list[EvalResult]) -> dict:
    """Chạy RAGAS API 0.2+ (SingleTurnSample + EvaluationDataset).

    Khác biệt API 0.2: tên field đổi từ question→user_input, answer→response,
    contexts→retrieved_contexts, ground_truth→reference.
    """
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness

    samples = [
        SingleTurnSample(
            user_input=r.question,
            response=r.answer,
            retrieved_contexts=r.contexts,
            reference=r.ground_truth,
        )
        for r in results
    ]
    dataset = EvaluationDataset(samples=samples)

    logger.info("Running RAGAS evaluation", num_samples=len(results))
    scores = evaluate(
        dataset=dataset,
        metrics=[ContextPrecision(), Faithfulness(), AnswerRelevancy()],
        llm=_build_judge_llm(),            # ★ dùng ĐÚNG provider của project
        embeddings=_build_judge_embeddings(),
    )
    logger.info("RAGAS evaluation complete")
    if hasattr(scores, "_repr_dict"):
        return {k: float(v) for k, v in scores._repr_dict.items()}
    return dict(scores)


def _build_judge_llm():
    """Judge LLM = đúng provider trong .env. Trả None → để RAGAS dùng default.

    Tránh phụ thuộc ngầm vào OpenAI: nếu project dùng provider khác (gemini),
    RAGAS sẽ cần OPENAI_API_KEY riêng.
    """
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    from core.config import settings

    if settings.LLM_PROVIDER.lower() != "openai":
        logger.warning(
            "RAGAS judge chỉ được cấu hình cho provider 'openai'. "
            "Đang dùng provider khác → RAGAS sẽ dùng default (cần OPENAI_API_KEY riêng).",
            provider=settings.LLM_PROVIDER,
        )
        return None
    kwargs = {"model": settings.OPENAI_MODEL_ID, "api_key": settings.OPENAI_API_KEY,
              "temperature": 0}
    if settings.OPENAI_BASE_URL:
        kwargs["base_url"] = settings.OPENAI_BASE_URL
    return LangchainLLMWrapper(ChatOpenAI(**kwargs))


def _build_judge_embeddings():
    """answer_relevancy cần embeddings — dùng model local, khỏi tốn API."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    from core.config import settings

    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_ID,
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
        )
    )


def _simple_evaluate(results: list[EvalResult]) -> dict:
    """Fallback evaluation đơn giản (không cần RAGAS).

    Đo lường cơ bản:
    - answer_length: Độ dài trung bình câu trả lời
    - has_context: Tỷ lệ câu có context
    - keyword_overlap: Overlap giữa answer và ground_truth

    LƯU Ý: metric keyword_overlap là recall từ khoá thô — KHÔNG đo được
    faithfulness hay hallucination. Đánh dấu rõ là fallback, không phải RAG Triad.
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
        "_mode": "SIMPLE_FALLBACK (không phải RAG Triad — chỉ đo keyword overlap)",
        "num_samples": total,
        "avg_answer_length": sum(len(r.answer) for r in results) / total,
        "avg_keyword_overlap": sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0,
        "has_context_ratio": sum(1 for r in results if r.contexts) / total,
    }
