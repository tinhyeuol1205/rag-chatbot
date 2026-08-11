"""
Query Condensation — Viết lại câu hỏi follow-up thành câu hỏi độc lập.

Vấn đề: "Can I carry them over?" — retrieval không hiểu "them" là gì.
Giải pháp: dùng LLM + history → "Can employees carry over unused annual leave?"

Chạy TRƯỚC Multi-Query Expansion trong pipeline:
  history + question → [Condense] → standalone question → [Multi-Query] → ...
"""

from __future__ import annotations

from core import get_logger
from core.llm import get_llm_service

logger = get_logger(__name__)

CONDENSE_PROMPT = """Given the conversation history and a follow-up question,
rewrite the follow-up question to be a standalone question that makes sense
without the history. Resolve all pronouns and references.

If the follow-up question is already standalone, return it UNCHANGED.
Return ONLY the rewritten question, nothing else.

Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""

MAX_HISTORY_TURNS = 3


class QueryCondenser:
    """Viết lại câu hỏi follow-up thành câu hỏi độc lập."""

    def __init__(self):
        self.llm = get_llm_service()

    def condense(self, question: str, history: list[tuple[str, str]]) -> str:
        """Trả về câu hỏi standalone. Không history → giữ nguyên.

        Args:
            question: Câu hỏi mới của user
            history: List các (user, assistant) — từ ít mới nhất

        Returns:
            Câu hỏi standalone để dùng cho retrieval
        """
        if not history:
            return question

        recent = history[-MAX_HISTORY_TURNS:]
        history_text = "\n".join(f"User: {u}\nAssistant: {a}" for u, a in recent)
        try:
            rewritten = self.llm.generate(
                user_prompt=CONDENSE_PROMPT.format(history=history_text, question=question),
                temperature=0.0, max_tokens=150,
            ).strip()
        except Exception as e:  # noqa: BLE001 - optional LLM enhancement must degrade
            # Cùng nguyên tắc graceful degradation như PR 1 / P1-7
            logger.warning("Condensation failed, using original question", error=str(e))
            return question

        if not rewritten or len(rewritten) > 500:
            logger.warning("Condensation output rejected, using original",
                           length=len(rewritten))
            return question
        logger.info("Query condensed", original=question[:60], rewritten=rewritten[:60])
        return rewritten
