from __future__ import annotations

"""
HyDE — Hypothetical Document Embedding ★ — Kỹ thuật RAG #4.

Vấn đề:
  Khi search, ta embed QUERY rồi tìm document gần nhất.
  Nhưng query ("laptop policy?") và document ("Employees receive Dell XPS...")
  nằm ở 2 không gian ngữ nghĩa KHÁC NHAU (câu hỏi vs câu trả lời).

Giải pháp:
  1. Dùng LLM sinh câu trả lời GIẢ ĐỊNH (có thể chưa chính xác)
  2. Embed câu trả lời giả → vector
  3. Search bằng vector này

  Tại sao? Vector của "câu trả lời giả" gần với "document chứa câu trả lời thật"
  hơn là vector của "câu hỏi".

  Query: "What is the laptop policy?"
  HyDE answer: "The company provides Dell XPS or MacBook Pro laptops to all employees..."
  → Vector của HyDE answer ≈ Vector của actual document ← MATCH TỐT HƠN

Tham khảo: rag_master.md — Module 4, mục 4.1, technique #2
"""

from core import get_logger
from core.llm import get_llm_service
from ingestion.embeddings import EmbeddingService

logger = get_logger(__name__)

HYDE_PROMPT = """Write a short paragraph that would answer the following question.
Write as if you are quoting from an official company document.
Do not explain or add disclaimers, just write the answer passage directly.

Question: {query}"""


class HyDEGenerator:
    """Sinh câu trả lời giả định → embed → dùng làm search vector."""

    def __init__(self):
        self.llm = get_llm_service()
        self.embedder = EmbeddingService()

    def generate_embedding(self, query: str) -> list[float] | None:
        """Tạo HyDE embedding cho query.

        Flow: query → LLM sinh hypothetical answer → embed answer → vector

        Args:
            query: Câu hỏi của user

        Returns:
            Vector 384d (embed từ hypothetical answer, KHÔNG phải từ query),
            hoặc None nếu HyDE fail → caller sẽ dùng dense search thường.
        """
        # Bước 1: LLM sinh câu trả lời giả
        try:
            hypothetical_answer = self._generate_hypothetical(query)
        except Exception as e:
            logger.warning("HyDE generation failed, skipping HyDE", error=str(e))
            return None
        if not hypothetical_answer.strip():
            logger.warning("HyDE returned empty text, skipping HyDE")
            return None

        # Bước 2: Embed câu trả lời giả (không phải embed query!)
        vector = self.embedder.embed_single(hypothetical_answer)

        logger.info(
            "HyDE embedding generated",
            query=query[:50],
            hypothetical=hypothetical_answer[:80],
        )
        return vector

    def _generate_hypothetical(self, query: str) -> str:
        """Dùng LLM sinh câu trả lời giả định."""
        return self.llm.generate(
            user_prompt=HYDE_PROMPT.format(query=query),
            temperature=0.5,  # Không quá creative, giữ sát chủ đề
            max_tokens=200,
        )

