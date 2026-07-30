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

logger = get_logger(__name__)


class ContextAssembler:
    """Tổng hợp context với source citation + Lost-in-Middle reorder."""

    def assemble(self, documents: list[dict]) -> tuple[str, str]:
        """Tổng hợp documents thành context string cho LLM.

        Args:
            documents: Danh sách documents (đã resolve parent)

        Returns:
            (context_text, sources_text) — 2 strings để đưa vào prompt
        """
        if not documents:
            return "", ""

        # Bước 1: Lost-in-Middle reorder
        reordered = self._lost_in_middle_reorder(documents)

        # Bước 2: Ghép context
        context_parts = []
        for i, doc in enumerate(reordered, 1):
            source = doc.get("file_name", "Unknown")
            section = doc.get("section_title", "")
            label = f"[Source {i}: {source}"
            if section:
                label += f" — {section}"
            label += "]"

            context_parts.append(f"{label}\n{doc['content']}")

        context_text = "\n\n---\n\n".join(context_parts)

        # Bước 3: Tạo sources summary
        sources = []
        seen = set()
        for doc in documents:
            fname = doc.get("file_name", "Unknown")
            section = doc.get("section_title", "")
            key = f"{fname}:{section}"
            if key not in seen:
                source_str = f"- {fname}"
                if section:
                    source_str += f" → {section}"
                sources.append(source_str)
                seen.add(key)

        sources_text = "\n".join(sources)

        logger.info("Context assembled", chunks=len(documents), total_chars=len(context_text))
        return context_text, sources_text

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
