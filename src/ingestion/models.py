from __future__ import annotations

"""
Data Models cho Ingestion Pipeline.

Mỗi model đại diện cho dữ liệu tại 1 giai đoạn khác nhau:
  File → [Parser] → RawDocument → [Chunker] → Chunk → [Embedder] → EmbeddedChunk → Qdrant

Pattern: Giống llm-twin-course (models/raw.py → clean.py → chunk.py → embedded_chunk.py)
Mỗi bước xử lý nhận model A và trả về model B.
"""

import hashlib
import uuid

from pydantic import BaseModel

# Namespace cố định cho project — đảm bảo ID deterministic giữa các lần chạy
_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


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
    - chunk_id: UUIDv5 deterministic — dùng làm Qdrant point ID
    - parent_id: Link đến parent chunk (cho Parent-Child Retrieval)
    - position: vị trí trong file ("docIdx:parentIdx[:childIdx]") — chống collision
    - is_parent: True nếu đây là parent chunk (chunk lớn, chỉ lưu text)
    """

    chunk_id: str = ""
    content: str
    parent_id: str | None = None
    is_parent: bool = False
    position: str = ""
    metadata: DocumentMetadata

    def model_post_init(self, __context) -> None:
        """Tự động tạo chunk_id sau khi init nếu chưa có."""
        if not self.chunk_id:
            self.chunk_id = self._generate_id()

    def _generate_id(self) -> str:
        """UUIDv5 deterministic từ (file, vị trí, TOÀN BỘ content).

        - Dùng full content → không collision do prefix giống nhau (bug P2-5)
        - Kèm position → 2 đoạn text trùng nhau ở 2 vị trí vẫn là 2 chunk
        - Trả UUID chuẩn → Qdrant nhận trực tiếp, không cần normalize dấu '-'
        - Deterministic: cùng input → cùng ID (idempotent khi re-ingest content không đổi)
        """
        digest = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        key = f"{self.metadata.source_path or self.metadata.file_name}|{self.position}|{digest}"
        return str(uuid.uuid5(_NAMESPACE, key))


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
