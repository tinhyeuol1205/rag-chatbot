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
        """Delegate sang hàm module-level (giữ interface cũ)."""
        return rrf_fusion(*result_lists)


def rrf_fusion(
    *result_lists: list[dict],
    k: int = RRF_K,
    weights: list[float] | None = None,
) -> list[dict]:
    """Reciprocal Rank Fusion — hợp nhất N danh sách kết quả.

    Công thức: RRF_score(doc) = Σ w_m / (k + rank_m(doc))
    với rank_m bắt đầu từ 1 (đúng paper Cormack et al.).

    Điểm được CỘNG DỒN qua mọi list chứa doc — đây là tín hiệu chính của
    Multi-Query / Hybrid: "bao nhiêu query cùng đồng ý". (Chunk ở nhiều list
    thắng chunk chỉ ở 1 list.)

    Args:
        result_lists: Nhiều danh sách kết quả (đã sắp xếp theo relevance)
        k: Hằng số RRF (mặc định 60 theo paper)
        weights: Trọng số cho từng list (optional). Mặc định 1.0 cho mọi list.

    Returns:
        Danh sách hợp nhất, sắp xếp theo RRF score giảm dần.
        Mỗi doc có thêm 'rrf_score' và 'n_hits' (số list chứa doc đó).
    """
    if not result_lists:
        return []
    if weights is not None and len(weights) != len(result_lists):
        raise ValueError("weights phải có cùng số phần tử với result_lists")
    if weights is None:
        weights = [1.0] * len(result_lists)

    rrf_scores: dict[str, float] = {}      # chunk_id → RRF score
    hit_counts: dict[str, int] = {}        # chunk_id → số list chứa doc
    doc_store: dict[str, dict] = {}         # chunk_id → document data

    for w, result_list in zip(weights, result_lists):
        for rank, doc in enumerate(result_list, start=1):   # ★ rank tính từ 1
            chunk_id = doc["chunk_id"]

            # Công thức RRF: w / (k + rank), rank ≥ 1 (đúng paper Cormack et al.)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + w / (k + rank)
            hit_counts[chunk_id] = hit_counts.get(chunk_id, 0) + 1

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
        doc["n_hits"] = hit_counts[chunk_id]   # hữu ích để debug/log
        merged.append(doc)

    return merged
