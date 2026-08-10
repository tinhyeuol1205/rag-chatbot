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

import heapq
from threading import Lock

from rank_bm25 import BM25Okapi

from core import get_logger
from core.config import settings
from core.db import QdrantConnector
from core.errors import RetrievalError

logger = get_logger(__name__)

# Index được chia sẻ ở module level: tạo bao nhiêu instance cũng chỉ build 1 lần.
# Gọi invalidate_bm25_index() sau khi ingest để nạp lại (bug P1-9a).
_index_lock = Lock()
_shared: dict = {"index": None, "documents": None, "version": 0}

# Tokenizer giữ được mã kiểu 'TC-456' kể cả khi dính dấu câu:
#   "see TC-456."    → ['tc-456']
#   "(TC-456), done" → ['tc-456', 'done']
_TOKEN_RE = re.compile(r"[0-9a-z]+(?:[-_][0-9a-z]+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize text: lowercase + giữ mã hiệu dính dấu câu."""
    return _TOKEN_RE.findall(text.lower())


def invalidate_bm25_index() -> None:
    """Gọi sau khi ingest xong để BM25 nạp lại corpus (bug P1-9a).

    Lưu ý: chỉ có tác dụng trong CÙNG process. Nếu ingest ở terminal khác với
    server đang chạy → vẫn phải restart server để BM25 thấy dữ liệu mới.
    """
    with _index_lock:
        _shared["index"] = None
        _shared["documents"] = None
        _shared["version"] += 1
    logger.info("BM25 index invalidated", version=_shared["version"])


class SparseSearcher:
    """Tìm kiếm bằng BM25 keyword matching.

    Index được chia sẻ ở module level: tạo bao nhiêu instance cũng chỉ build 1 lần.
    Gọi invalidate_bm25_index() sau khi ingest để nạp lại.
    """

    def __init__(self):
        self.qdrant = QdrantConnector()

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

        index, documents = self._ensure_index()
        if not documents or index is None:
            return []

        # Tokenize query — giữ mã hiệu dính dấu câu
        query_tokens = tokenize(query)
        if not query_tokens:
            logger.info("BM25 search skipped — query has no usable tokens")
            return []

        # BM25 scoring
        scores = index.get_scores(query_tokens)

        # top-k bằng heap thay vì sort toàn bộ corpus (O(n log k) thay vì O(n log n))
        top_idx = heapq.nlargest(top_k, range(len(scores)), key=scores.__getitem__)

        # Format kết quả
        results = []
        for i in top_idx:
            if scores[i] <= 0:              # chỉ lấy docs match ít nhất 1 từ
                continue
            doc = documents[i]
            results.append({
                "chunk_id": doc["chunk_id"],
                "content": doc["content"],
                "score": float(scores[i]),
                "parent_id": doc.get("parent_id"),
                "file_name": doc.get("file_name", ""),
                "section_title": doc.get("section_title", ""),
                "page_number": doc.get("page_number"),
                "source": "sparse",  # Đánh dấu nguồn
            })

        logger.info("BM25 search done", query=query[:50], results=len(results))
        return results

    def _ensure_index(self):
        """Lấy index dùng chung, build nếu chưa có (thread-safe)."""
        with _index_lock:
            if _shared["index"] is None and _shared["documents"] is None:
                self._build_index_locked()
            return _shared["index"], _shared["documents"] or []

    def _build_index_locked(self) -> None:
        """Load tất cả documents từ Qdrant → build BM25 index.

        Gọi 1 lần duy nhất (đã nằm trong lock), kết quả cached cho các query sau.
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

        documents, corpus = [], []  # corpus = tokenized documents cho BM25

        for point in points:
            payload = point.payload or {}
            content = payload.get("content", "")
            if not content:
                continue
            documents.append({
                "chunk_id": point.id,
                "content": content,
                "parent_id": payload.get("parent_id"),
                "file_name": payload.get("file_name", ""),
                "section_title": payload.get("section_title", ""),
                "page_number": payload.get("page_number"),
            })
            # Tokenize: giữ mã hiệu dính dấu câu
            corpus.append(tokenize(content))

        _shared["documents"] = documents
        _shared["index"] = BM25Okapi(corpus) if corpus else None
        logger.info("BM25 index built", total_documents=len(documents))
