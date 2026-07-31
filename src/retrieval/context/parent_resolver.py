from __future__ import annotations

"""
Parent Resolver — Map child chunks → parent chunks.

Đây là nửa sau của kỹ thuật Parent-Child Retrieval ★:
  - Ingestion tạo child_chunks (nhỏ, có vector) + parent_chunks (lớn, chỉ text)
  - Search tìm child_chunks match query
  - Parent Resolver lấy parent chunk cho mỗi child → đưa vào LLM

Ví dụ:
  Search tìm được child "Bước 2: Nhận laptop..." (400 chars)
  Parent Resolver → trả về parent "Toàn bộ Quy trình Onboarding..." (2000 chars)
  → LLM có đủ context để trả lời đầy đủ
"""

from core import get_logger
from core.config import settings
from core.db import QdrantConnector

logger = get_logger(__name__)


class ParentResolver:
    """Map child chunk IDs → parent chunks (full text)."""

    def __init__(self):
        self.qdrant = QdrantConnector()

    def resolve(self, child_results: list[dict]) -> list[dict]:
        """Thay thế child chunks bằng parent chunks.

        Args:
            child_results: Kết quả từ reranker (child chunks)

        Returns:
            List[dict] với content được thay bằng parent chunk content.
            Deduplicate: nếu 2 children cùng 1 parent → chỉ giữ 1 parent.
        """
        if not child_results:
            return []

        # Thu thập parent_ids (bỏ None, deduplicate)
        parent_ids = list(set(
            doc["parent_id"]
            for doc in child_results
            if doc.get("parent_id")
        ))

        if not parent_ids:
            # Không có parent_id → trả về child chunks nguyên bản
            logger.info("No parent_ids found, returning child chunks as-is")
            return child_results

        # Lấy parent chunks từ Qdrant
        parent_points = self.qdrant.get_by_ids(settings.PARENT_COLLECTION, parent_ids)

        # Tạo lookup: parent_id → parent content
        # ★ Normalize: bỏ dấu '-' vì Qdrant tự convert MD5 hex → UUID format
        #   Qdrant trả về: "249850ae-83ee-caac-ad59-5b9d037dd868" (có dấu -)
        #   Child payload:  "249850ae83eecaacad595b9d037dd868"     (không dấu -)
        parent_map = {
            str(point.id).replace("-", ""): point.payload
            for point in parent_points
        }

        # Thay child content bằng parent content
        resolved = []
        seen_parent_ids = set()

        for doc in child_results:
            pid = str(doc.get("parent_id", "")).replace("-", "")

            if pid and pid in parent_map and pid not in seen_parent_ids:
                # Thay content bằng parent (đầy đủ hơn)
                parent_payload = parent_map[pid]
                resolved.append({
                    "chunk_id": pid,
                    "content": parent_payload.get("content", doc["content"]),
                    "score": doc.get("rerank_score", doc.get("score", 0)),
                    "file_name": parent_payload.get("file_name", doc.get("file_name", "")),
                    "section_title": parent_payload.get("section_title", ""),
                    "is_parent": True,
                })
                seen_parent_ids.add(pid)  # Deduplicate

            elif not pid:
                # Không có parent → giữ nguyên child
                resolved.append(doc)

        logger.info(
            "Parent resolution done",
            children_in=len(child_results),
            parents_out=len(resolved),
            deduplicated=len(child_results) - len(resolved),
        )

        return resolved
