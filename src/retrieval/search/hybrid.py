from __future__ import annotations

"""
Hybrid Search + Reciprocal Rank Fusion (RRF) ★ — Kỹ thuật RAG #1.

Vấn đề:
  - Dense search (vector): tốt cho ngữ nghĩa, kém với từ khóa chính xác
  - Sparse search (BM25): tốt cho từ khóa, kém với ngữ nghĩa

Giải pháp: Kết hợp CẢ HAI bằng RRF (Reciprocal Rank Fusion).

RRF hoạt động dựa trên THỨ HẠNG (rank), không dựa trên điểm số (score):

  RRF_score(doc) = Σ  1 / (k + rank_m(doc))
                   m∈M

  Với: k = 60 (hằng số), rank_m = thứ hạng trong danh sách m

Ví dụ:
  Doc A: rank 1 trong Dense, rank 5 trong Sparse
  → RRF = 1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = 0.0318

  Doc B: rank 3 trong Dense, rank 2 trong Sparse
  → RRF = 1/(60+3) + 1/(60+2) = 0.0159 + 0.0161 = 0.0320

  → Doc B thắng (0.0320 > 0.0318) vì thứ hạng ĐỀU tốt ở cả 2 nguồn.

Tại sao dùng rank thay vì score?
  Score từ Dense (cosine: 0.0-1.0) và Sparse (BM25: 0-∞) KHÔNG cùng thang đo.
  RRF dùng rank → không cần normalize score → công bằng.

Tham khảo: rag_master.md — Module 4, mục 4.2
"""

from core import get_logger
from core.config import settings
from retrieval.search.dense import DenseSearcher
from retrieval.search.sparse import SparseSearcher

logger = get_logger(__name__)

# Hằng số RRF — giá trị chuẩn từ paper gốc
RRF_K = 60


class HybridSearcher:
    """Kết hợp Dense + Sparse search bằng RRF."""

    def __init__(self):
        self.dense = DenseSearcher()
        self.sparse = SparseSearcher()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        hyde_vector: list[float] | None = None,
    ) -> list[dict]:
        """Hybrid search: Dense + Sparse + RRF fusion.

        Args:
            query: Câu hỏi của user
            top_k: Số kết quả cuối cùng
            hyde_vector: Nếu có, dùng HyDE vector cho dense search thay vì embed query

        Returns:
            List[dict] đã merge và rank bằng RRF
        """
        top_k = top_k or settings.TOP_K

        # --- Bước 1: Chạy song song 2 search engines ---

        # Dense: dùng HyDE vector nếu có, nếu không thì embed query
        if hyde_vector:
            dense_results = self.dense.search_by_vector(hyde_vector, top_k=top_k)
        else:
            dense_results = self.dense.search(query, top_k=top_k)

        # Sparse: BM25 keyword search
        sparse_results = self.sparse.search(query, top_k=top_k)

        # --- Bước 2: RRF Fusion ---
        merged = self._rrf_fusion(dense_results, sparse_results)

        # --- Bước 3: Lấy top-K ---
        final = merged[:top_k]

        logger.info(
            "Hybrid search done",
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
            merged_count=len(merged),
            final_count=len(final),
        )

        return final

    def _rrf_fusion(self, *result_lists: list[dict]) -> list[dict]:
        """Reciprocal Rank Fusion — hợp nhất nhiều danh sách kết quả.

        Công thức: RRF_score(doc) = Σ 1/(k + rank) cho mỗi list chứa doc đó.

        Args:
            result_lists: Nhiều danh sách kết quả (đã sắp xếp theo relevance)

        Returns:
            Danh sách hợp nhất, sắp xếp theo RRF score giảm dần
        """
        rrf_scores: dict[str, float] = {}      # chunk_id → RRF score
        doc_store: dict[str, dict] = {}         # chunk_id → document data

        for result_list in result_lists:
            for rank, doc in enumerate(result_list):
                chunk_id = doc["chunk_id"]

                # Công thức RRF: 1 / (k + rank)
                # rank bắt đầu từ 0, nên rank 0 = vị trí #1
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (RRF_K + rank)

                # Lưu doc data (lấy lần đầu gặp)
                if chunk_id not in doc_store:
                    doc_store[chunk_id] = doc

        # Sắp xếp theo RRF score giảm dần
        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        # Build kết quả cuối
        merged = []
        for chunk_id in sorted_ids:
            doc = doc_store[chunk_id].copy()
            doc["rrf_score"] = rrf_scores[chunk_id]
            merged.append(doc)

        return merged
