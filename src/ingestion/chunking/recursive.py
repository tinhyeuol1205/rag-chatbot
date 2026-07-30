from __future__ import annotations

"""
Recursive Character Chunking — Chiến lược chunking cơ bản.

Cách hoạt động:
  1. Thử chia theo "\n\n" (paragraph) trước
  2. Nếu chunk vẫn quá lớn → chia theo "\n" (dòng)
  3. Vẫn quá lớn → chia theo ". " (câu)
  4. Cuối cùng → chia theo " " (từ)

Ưu điểm: Giữ nguyên cấu trúc paragraph/câu càng nhiều càng tốt.
Đây là chunking mặc định trong hầu hết RAG systems.

Tham khảo: rag_master.md — Module 2, mục 2.2, strategy #2
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core import get_logger
from core.config import settings

logger = get_logger(__name__)


def recursive_chunk(text: str) -> list[str]:
    """Chia text thành chunks bằng RecursiveCharacterTextSplitter.

    Args:
        text: Văn bản cần chia

    Returns:
        Danh sách các text chunks
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHILD_CHUNK_SIZE,      # 400 chars
        chunk_overlap=settings.CHILD_CHUNK_OVERLAP,  # 50 chars overlap
        separators=["\n\n", "\n", ". ", " ", ""],   # Thử chia theo thứ tự ưu tiên
    )

    chunks = splitter.split_text(text)

    logger.info(
        "Recursive chunking done",
        input_length=len(text),
        num_chunks=len(chunks),
        chunk_size=settings.CHILD_CHUNK_SIZE,
    )

    return chunks
