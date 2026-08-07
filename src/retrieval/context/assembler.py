from __future__ import annotations

"""
Context Assembler — Tổng hợp context + Lost-in-Middle reorder.

Lost-in-Middle là hiện tượng được phát hiện trong paper:
  "Lost in the Middle: How Language Models Use Long Contexts" (2023)

LLM chú ý tốt nhất ở ĐẦU và CUỐI context window, bỏ quên ở GIỮA:

  Context = [Doc1, Doc2, Doc3, Doc4, Doc5]
                         ↑↑↑
                    "Vùng mù" — LLM hay bỏ qua

Giải pháp: Sắp xếp lại theo pattern "zigzag":
  Input (ranked):    [1st, 2nd, 3rd, 4th, 5th]
  Output (reordered): [1st, 3rd, 5th, 4th, 2nd]

  → Doc quan trọng nhất (1st) ở ĐẦU
  → Doc quan trọng thứ 2 (2nd) ở CUỐI
  → Doc ít quan trọng nhất (3rd, 5th) ở GIỮA

Tham khảo: rag_master.md — Module 5, mục 5.2
"""

from core import get_logger
from core.config import settings

logger = get_logger(__name__)


class ContextAssembler:
    """Tổng hợp context với source citation + Lost-in-Middle reorder."""

    def assemble(self, documents: list[dict]) -> tuple[str, str]:
        """Tổng hợp documents thành context string cho LLM.

        Cả context block (có [Source N]) và sources summary ([N] ...) được build
        từ CÙNG list đã reorder → số [N] trong context khớp chính xác với số [N]
        trong danh sách Sources (bug P2-1: trước đây 2 list khác nhau).

        Args:
            documents: Danh sách documents (đã resolve parent)

        Returns:
            (context_text, sources_text) — 2 strings để đưa vào prompt
        """
        if not documents:
            return "", ""

        # Bước 1: Lost-in-Middle reorder
        reordered = self._lost_in_middle_reorder(documents)

        # Bước 2: Ghép context + sources từ CÙNG list reordered, kèm token budget
        context_parts, sources, seen = [], [], set()
        used_chars, dropped = 0, 0

        for i, doc in enumerate(reordered, 1):
            label = f"[Source {i}: {self._format_source(doc)}]"
            block = f"{label}\n{doc['content']}"

            # ★ P2-10: token budget — bỏ cả block, KHÔNG cắt giữa câu
            if used_chars + len(block) > settings.MAX_CONTEXT_CHARS and context_parts:
                dropped += 1
                continue

            context_parts.append(block)
            used_chars += len(block)

            key = self._source_key(doc)
            if key not in seen:
                seen.add(key)
                sources.append(f"[{i}] {self._format_source(doc)}")   # ★ CÙNG số i

        if dropped:
            logger.warning(
                "Context truncated by budget",
                dropped_docs=dropped,
                kept=len(context_parts),
                budget=settings.MAX_CONTEXT_CHARS,
                used=used_chars,
            )

        context_text = "\n\n---\n\n".join(context_parts)
        sources_text = "\n".join(sources)

        logger.info("Context assembled", chunks=len(context_parts), total_chars=len(context_text))
        return context_text, sources_text

    @staticmethod
    def _format_source(doc: dict) -> str:
        """Định dạng nguồn: 'file_name → section_title → p.N'."""
        parts = [doc.get("file_name") or "Unknown"]
        if doc.get("section_title"):
            parts.append(doc["section_title"])
        if doc.get("page_number") is not None:
            parts.append(f"p.{doc['page_number']}")
        return " → ".join(parts)

    @staticmethod
    def _source_key(doc: dict) -> str:
        """Key để dedupe nguồn trùng lặp (cùng file + section + page)."""
        return (
            f"{doc.get('file_name', '')}:{doc.get('section_title', '')}"
            f":{doc.get('page_number')}"
        )

    def _lost_in_middle_reorder(self, documents: list[dict]) -> list[dict]:
        """Sắp xếp lại theo Lost-in-Middle pattern.

        Input:  [1st, 2nd, 3rd, 4th, 5th]  (ranked by relevance)
        Output: [1st, 3rd, 5th, 4th, 2nd]  (important docs at start & end)
        """
        if len(documents) <= 2:
            return documents

        # Chia thành 2 nhóm: vị trí lẻ (1st, 3rd, 5th) và vị trí chẵn (2nd, 4th)
        odd_positions = [documents[i] for i in range(0, len(documents), 2)]   # [1st, 3rd, 5th]
        even_positions = [documents[i] for i in range(1, len(documents), 2)]  # [2nd, 4th]

        # Ghép: odd_positions + reversed(even_positions)
        # [1st, 3rd, 5th] + [4th, 2nd] = [1st, 3rd, 5th, 4th, 2nd]
        reordered = odd_positions + list(reversed(even_positions))

        return reordered
