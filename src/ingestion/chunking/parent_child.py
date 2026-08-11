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

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core import get_logger
from core.config import settings
from ingestion.models import Chunk, DocumentMetadata, RawDocument

logger = get_logger(__name__)


def parent_child_chunk(documents: list[RawDocument]) -> tuple[list[Chunk], list[Chunk]]:
    """Tạo parent chunks và child chunks từ danh sách documents.

    Chunk theo TỪNG document (1 section MD / 1 trang PDF) để giữ nguyên
    section_title và page_number cho citation (bug P1-2).

    Args:
        documents: Danh sách RawDocument (từ parser)

    Returns:
        (parent_chunks, child_chunks) — 2 danh sách riêng biệt
        - parent_chunks: lưu vào Qdrant payload-only collection
        - child_chunks: embed + lưu vào Qdrant vector collection
    """
    if not documents:
        return [], []

    # Splitters — tạo 1 lần, dùng cho mọi document
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.PARENT_CHUNK_SIZE,       # 2000 chars
        chunk_overlap=settings.PARENT_CHUNK_OVERLAP,  # 200 chars overlap
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHILD_CHUNK_SIZE,        # 400 chars
        chunk_overlap=settings.CHILD_CHUNK_OVERLAP,   # 50 chars overlap
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    parent_chunks: list[Chunk] = []
    child_chunks: list[Chunk] = []

    # ★ Chunk theo TỪNG document → giữ metadata của document đó
    for doc_idx, doc in enumerate(documents):
        md = doc.metadata
        for p_idx, parent_text in enumerate(parent_splitter.split_text(doc.content)):
            # section_title: ưu tiên cái parser trích được; chỉ fallback khi thiếu
            if md.section_title:
                title = md.section_title
            elif md.page_number is not None:
                title = f"Page {md.page_number}"
            else:
                title = f"Section {doc_idx + 1}"

            parent = Chunk(
                content=parent_text,
                is_parent=True,
                parent_id=None,  # Parent không có parent
                position=f"{doc_idx}:{p_idx}",       # ★ chống collision (P2-5)
                metadata=DocumentMetadata(
                    file_name=md.file_name,
                    file_type=md.file_type,
                    page_number=md.page_number,      # ★ GIỮ
                    section_title=title,             # ★ GIỮ
                    source_path=md.source_path,
                    dataset_id=md.dataset_id,
                ),
            )
            parent_chunks.append(parent)

            for c_idx, child_text in enumerate(child_splitter.split_text(parent_text)):
                child_chunks.append(
                    Chunk(
                        content=child_text,
                        is_parent=False,
                        parent_id=parent.chunk_id,   # ★ Link đến parent
                        position=f"{doc_idx}:{p_idx}:{c_idx}",
                        metadata=parent.metadata.model_copy(),   # thừa hưởng metadata thật
                    )
                )

    logger.info(
        "Parent-Child chunking done",
        source_docs=len(documents),
        parents=len(parent_chunks),
        children=len(child_chunks),
        avg_children_per_parent=round(len(child_chunks) / max(len(parent_chunks), 1), 1),
    )

    return parent_chunks, child_chunks
