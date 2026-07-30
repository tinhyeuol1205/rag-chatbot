from __future__ import annotations

"""
Parent-Child Chunking ★ — Kỹ thuật RAG #5.

Vấn đề:
  - Chunk NHỎ (400 chars): search chính xác, nhưng LLM thiếu context
  - Chunk LỚN (2000 chars): LLM đủ context, nhưng search kém chính xác

Giải pháp:
  - Tạo PARENT chunks (lớn, 2000 chars) → lưu text vào Qdrant (không có vector)
  - Tạo CHILD chunks (nhỏ, 400 chars) → embed + lưu vector vào Qdrant
  - Mỗi child mang parent_id → link đến parent chunk

Khi retrieval:
  1. Search trên child_chunks (chính xác nhờ chunk nhỏ)
  2. Lấy parent_id từ child match
  3. Trả về parent chunk cho LLM (đầy đủ context)

Tham khảo: rag_master.md — Module 2, mục 2.2, strategy #4
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core import get_logger
from core.config import settings
from ingestion.models import Chunk, DocumentMetadata, RawDocument

logger = get_logger(__name__)


def parent_child_chunk(documents: list[RawDocument]) -> tuple[list[Chunk], list[Chunk]]:
    """Tạo parent chunks và child chunks từ danh sách documents.

    Args:
        documents: Danh sách RawDocument (từ parser)

    Returns:
        (parent_chunks, child_chunks) — 2 danh sách riêng biệt
        - parent_chunks: lưu vào Qdrant payload-only collection
        - child_chunks: embed + lưu vào Qdrant vector collection
    """

    # --- Bước 1: Gộp tất cả documents thành 1 text lớn ---
    # (vì 1 PDF có nhiều pages, ta muốn chunk xuyên suốt pages)
    full_text = "\n\n".join(doc.content for doc in documents)
    base_metadata = documents[0].metadata if documents else DocumentMetadata(
        file_name="unknown", file_type="unknown"
    )

    # --- Bước 2: Tạo PARENT chunks (lớn) ---
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.PARENT_CHUNK_SIZE,       # 2000 chars
        chunk_overlap=settings.PARENT_CHUNK_OVERLAP,  # 200 chars overlap
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    parent_texts = parent_splitter.split_text(full_text)

    parent_chunks = []
    for i, text in enumerate(parent_texts):
        parent_chunks.append(
            Chunk(
                content=text,
                is_parent=True,
                parent_id=None,  # Parent không có parent
                metadata=DocumentMetadata(
                    file_name=base_metadata.file_name,
                    file_type=base_metadata.file_type,
                    source_path=base_metadata.source_path,
                    section_title=f"Section {i + 1}",
                ),
            )
        )

    # --- Bước 3: Tạo CHILD chunks (nhỏ) từ mỗi parent ---
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHILD_CHUNK_SIZE,        # 400 chars
        chunk_overlap=settings.CHILD_CHUNK_OVERLAP,   # 50 chars overlap
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    child_chunks = []
    for parent in parent_chunks:
        child_texts = child_splitter.split_text(parent.content)

        for j, text in enumerate(child_texts):
            child_chunks.append(
                Chunk(
                    content=text,
                    is_parent=False,
                    parent_id=parent.chunk_id,  # ★ Link đến parent
                    metadata=DocumentMetadata(
                        file_name=parent.metadata.file_name,
                        file_type=parent.metadata.file_type,
                        source_path=parent.metadata.source_path,
                        section_title=parent.metadata.section_title,
                    ),
                )
            )

    logger.info(
        "Parent-Child chunking done",
        parents=len(parent_chunks),
        children=len(child_chunks),
        avg_children_per_parent=round(len(child_chunks) / max(len(parent_chunks), 1), 1),
    )

    return parent_chunks, child_chunks
