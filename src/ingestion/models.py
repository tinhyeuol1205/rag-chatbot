from __future__ import annotations

"""
Data Models cho Ingestion Pipeline.

Mỗi model đại diện cho dữ liệu tại 1 giai đoạn khác nhau:
  File → [Parser] → RawDocument → [Chunker] → Chunk → [Embedder] → EmbeddedChunk → Qdrant

Pattern: Giống llm-twin-course (models/raw.py → clean.py → chunk.py → embedded_chunk.py)
Mỗi bước xử lý nhận model A và trả về model B.
"""

import hashlib

from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    """Metadata đính kèm mỗi document/chunk — dùng cho filtering và source citation."""

    file_name: str
    file_type: str           # "pdf", "md", "docx"
    page_number: int | None = None
    section_title: str | None = None
    source_path: str = ""


class RawDocument(BaseModel):
    """
    Sau khi parse file → RawDocument.
    Mỗi RawDocument = 1 trang PDF hoặc 1 section Markdown.
    """

    content: str
    metadata: DocumentMetadata


class Chunk(BaseModel):
    """
    Sau khi chunking → Chunk.
    Mỗi Chunk có:
    - chunk_id: ID duy nhất (MD5 hash) — dùng làm Qdrant point ID
    - parent_id: Link đến parent chunk (cho Parent-Child Retrieval)
    - is_parent: True nếu đây là parent chunk (chunk lớn, chỉ lưu text)
    """

    chunk_id: str = ""
    content: str
    parent_id: str | None = None
    is_parent: bool = False
    metadata: DocumentMetadata

    def model_post_init(self, __context) -> None:
        """Tự động tạo chunk_id sau khi init nếu chưa có."""
        if not self.chunk_id:
            self.chunk_id = self._generate_id()

    def _generate_id(self) -> str:
        """Tạo ID duy nhất = MD5(content + file_name).
        Deterministic: cùng content + file → cùng ID (idempotent khi re-ingest).
        """
        key = f"{self.content[:200]}:{self.metadata.file_name}"
        return hashlib.md5(key.encode()).hexdigest()


class EmbeddedChunk(BaseModel):
    """
    Sau khi embedding → EmbeddedChunk.
    Thêm vector embedding (384 dimensions) để lưu vào Qdrant.
    """

    chunk_id: str
    content: str
    embedding: list[float]   # Vector 384d từ bge-small-en
    parent_id: str | None = None
    metadata: DocumentMetadata
