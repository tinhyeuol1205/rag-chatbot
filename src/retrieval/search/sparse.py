from __future__ import annotations

"""
Sparse Search (BM25) — Tìm kiếm bằng từ khóa.

BM25 là thuật toán "cổ điển" dùng trong search engines (Google, Elasticsearch).
Nó đếm TẦN SUẤT từ khóa xuất hiện trong document, có điều chỉnh theo:
  - TF (Term Frequency): Từ xuất hiện nhiều lần trong doc → điểm cao
  - IDF (Inverse Document Frequency): Từ hiếm (chỉ xuất hiện ở ít docs) → quan trọng hơn
  - Document length normalization: Doc ngắn match 1 từ → quan trọng hơn doc dài match 1 từ

Ưu điểm: Chính xác với từ khóa, mã sản phẩm, tên riêng
  "TC-456" → BM25 tìm CHÍNH XÁC document chứa "TC-456" (kể cả khi dính dấu câu)

Nhược điểm: Không hiểu ngữ nghĩa
  "xe hơi" ≠ "ô tô" → BM25 coi là 2 từ KHÁC NHAU

Giới hạn: chỉ hoạt động tốt với ngôn ngữ có space phân từ (Anh, Việt có dấu cách).
  CJK (Trung/Nhật/Hàn) cần tokenizer riêng — chưa hỗ trợ.

Tham khảo: rag_master.md — Module 3, mục 3.1 (Sparse Embeddings)
"""

import re

from rank_bm25 import BM25Okapi

from core import get_logger
from core.config import settings
from core.db import QdrantConnector
from core.errors import RetrievalError

logger = get_logger(__name__)

# Tokenizer giữ được mã kiểu 'TC-456' kể cả khi dính dấu câu:
#   "see TC-456."    → ['tc-456']
#   "(TC-456), done" → ['tc-456', 'done']
_TOKEN_RE = re.compile(r"[0-9a-z]+(?:[-_][0-9a-z]+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize text: lowercase + giữ mã hiệu dính dấu câu."""
    return _TOKEN_RE.findall(text.lower())


class SparseSearcher:
    """Tìm kiếm bằng BM25 keyword matching."""

    def __init__(self):
        self.qdrant = QdrantConnector()
        self._index: BM25Okapi | None = None
        self._documents: list[dict] | None = None

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Search bằng BM25 keyword matching.

        Lần đầu gọi sẽ build index (load tất cả docs từ Qdrant).
        Các lần sau dùng index đã build (cached).

        Args:
            query: Câu hỏi của user
            top_k: Số kết quả trả về

        Returns:
            List[dict] — mỗi dict có: chunk_id, content, score, metadata
        """
        top_k = top_k or settings.TOP_K

        # Lazy build BM25 index
        if self._index is None:
            self._build_index()

        if not self._documents:
            return []

        # Tokenize query — giữ mã hiệu dính dấu câu
        query_tokens = tokenize(query)

        # BM25 scoring
        scores = self._index.get_scores(query_tokens)

        # Lấy top-K theo score
        scored_docs = list(zip(self._documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_results = scored_docs[:top_k]

        # Format kết quả
        results = []
        for doc, score in top_results:
            if score > 0:  # Chỉ lấy docs có match ít nhất 1 từ
                results.append({
                    "chunk_id": doc["chunk_id"],
                    "content": doc["content"],
                    "score": float(score),
                    "parent_id": doc.get("parent_id"),
                    "file_name": doc.get("file_name", ""),
                    "section_title": doc.get("section_title", ""),
                    "source": "sparse",  # Đánh dấu nguồn
                })

        logger.info("BM25 search done", query=query[:50], results=len(results))
        return results

    def _build_index(self) -> None:
        """Load tất cả documents từ Qdrant → build BM25 index.

        Gọi 1 lần duy nhất, kết quả cached cho các query sau.
        """
        logger.info("Building BM25 index...")

        # Đọc tất cả child chunks từ Qdrant
        try:
            points = self.qdrant.scroll_all(settings.CHILD_COLLECTION)
        except Exception as e:
            raise RetrievalError(
                f"Không đọc được collection '{settings.CHILD_COLLECTION}'. "
                f"Đã chạy 'make ingest' chưa? Lỗi gốc: {e}"
            ) from e

        self._documents = []
        corpus = []  # List of tokenized documents cho BM25

        for point in points:
            content = point.payload.get("content", "")
            self._documents.append({
                "chunk_id": point.id,
                "content": content,
                "parent_id": point.payload.get("parent_id"),
                "file_name": point.payload.get("file_name", ""),
                "section_title": point.payload.get("section_title", ""),
            })
            # Tokenize: giữ mã hiệu dính dấu câu
            corpus.append(tokenize(content))

        if corpus:
            self._index = BM25Okapi(corpus)

        logger.info("BM25 index built", total_documents=len(self._documents))
