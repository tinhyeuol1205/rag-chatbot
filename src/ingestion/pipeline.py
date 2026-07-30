from __future__ import annotations

"""
Ingestion Pipeline Orchestrator — Nối tất cả lại.

Luồng xử lý:
  1. Scan thư mục → tìm tất cả files (PDF, MD, DOCX)
  2. Parse mỗi file → list[RawDocument]
  3. Parent-Child Chunking → parent_chunks + child_chunks
  4. Embed child chunks → vectors 384d
  5. Store vào Qdrant:
     - child_chunks → collection có vectors (dùng để search)
     - parent_chunks → collection payload-only (dùng để trả context)

Usage:
    from ingestion.pipeline import IngestionPipeline
    pipeline = IngestionPipeline()
    pipeline.run("data/sample_docs/")
"""

from pathlib import Path

from qdrant_client.models import PointStruct

from core import get_logger
from core.config import settings
from core.db import QdrantConnector
from ingestion.chunking.parent_child import parent_child_chunk
from ingestion.embeddings import EmbeddingService
from ingestion.models import Chunk, EmbeddedChunk, RawDocument
from ingestion.parsers import ParserDispatcher

logger = get_logger(__name__)


class IngestionPipeline:
    """Orchestrator cho toàn bộ ingestion flow."""

    def __init__(self):
        self.qdrant = QdrantConnector()
        self.embedder = EmbeddingService()

    def run(self, data_dir: str) -> None:
        """Chạy toàn bộ pipeline: scan → parse → chunk → embed → store."""

        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Bước 0: Tạo collections trong Qdrant
        self._init_collections()

        # Bước 1: Scan & parse tất cả files
        all_documents = self._parse_all_files(data_path)
        if not all_documents:
            logger.warning("No documents found in directory", path=data_dir)
            return

        # Bước 2-5: Xử lý từng file
        for file_name, documents in all_documents.items():
            logger.info("Processing file", file=file_name, raw_docs=len(documents))

            # Bước 2: Parent-Child Chunking
            parent_chunks, child_chunks = parent_child_chunk(documents)

            # Bước 3: Embed child chunks
            embedded_children = self._embed_chunks(child_chunks)

            # Bước 4: Store vào Qdrant
            self._store_parents(parent_chunks)
            self._store_children(embedded_children)

            logger.info(
                "File processed",
                file=file_name,
                parents=len(parent_chunks),
                children=len(embedded_children),
            )

        logger.info("Ingestion pipeline complete")

    # ================================================================
    # Private methods — từng bước xử lý
    # ================================================================

    def _init_collections(self) -> None:
        """Tạo 2 collections trong Qdrant nếu chưa tồn tại."""
        self.qdrant.create_vector_collection(settings.CHILD_COLLECTION)
        self.qdrant.create_payload_collection(settings.PARENT_COLLECTION)

    def _parse_all_files(self, data_path: Path) -> dict[str, list[RawDocument]]:
        """Scan thư mục, parse tất cả files hỗ trợ.

        Returns:
            Dict mapping: file_name → list[RawDocument]
        """
        supported = ParserDispatcher.supported_extensions()
        files = [f for f in data_path.rglob("*") if f.suffix.lower() in supported]

        if not files:
            logger.warning("No supported files found", path=str(data_path), supported=supported)
            return {}

        logger.info("Found files", count=len(files))

        result: dict[str, list[RawDocument]] = {}
        for file_path in sorted(files):
            parser = ParserDispatcher.get_parser(file_path)
            documents = parser.parse(file_path)
            if documents:
                result[file_path.name] = documents

        return result

    def _embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed danh sách chunks → EmbeddedChunks (có vector)."""
        if not chunks:
            return []

        texts = [c.content for c in chunks]
        vectors = self.embedder.embed(texts)

        embedded = []
        for chunk, vector in zip(chunks, vectors):
            embedded.append(
                EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    embedding=vector,
                    parent_id=chunk.parent_id,
                    metadata=chunk.metadata,
                )
            )

        logger.info("Embedded chunks", count=len(embedded))
        return embedded

    def _store_children(self, chunks: list[EmbeddedChunk]) -> None:
        """Lưu child chunks vào Qdrant (CÓ vector)."""
        if not chunks:
            return

        points = [
            PointStruct(
                id=chunk.chunk_id,
                vector=chunk.embedding,
                payload={
                    "content": chunk.content,
                    "parent_id": chunk.parent_id,
                    "file_name": chunk.metadata.file_name,
                    "file_type": chunk.metadata.file_type,
                    "source_path": chunk.metadata.source_path,
                    "section_title": chunk.metadata.section_title,
                },
            )
            for chunk in chunks
        ]

        self.qdrant.upsert_points(settings.CHILD_COLLECTION, points)

    def _store_parents(self, chunks: list[Chunk]) -> None:
        """Lưu parent chunks vào Qdrant (KHÔNG có vector, chỉ payload)."""
        if not chunks:
            return

        points = [
            PointStruct(
                id=chunk.chunk_id,
                vector={},  # Payload-only: không có vector
                payload={
                    "content": chunk.content,
                    "file_name": chunk.metadata.file_name,
                    "file_type": chunk.metadata.file_type,
                    "source_path": chunk.metadata.source_path,
                    "section_title": chunk.metadata.section_title,
                },
            )
            for chunk in chunks
        ]

        self.qdrant.upsert_points(settings.PARENT_COLLECTION, points)
