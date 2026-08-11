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

from __future__ import annotations

from dataclasses import dataclass, field

from core import get_logger
from core.config import settings

logger = get_logger(__name__)


@dataclass(frozen=True)
class SourceRef:
    """Một nguồn duy nhất được gắn citation ID ổn định trong context."""

    citation_id: int
    file_name: str
    section_title: str | None = None
    page_number: int | None = None

    @property
    def label(self) -> str:
        """Định dạng nguồn: ``file → section → p.N``."""
        parts = [self.file_name or "Unknown"]
        if self.section_title:
            parts.append(self.section_title)
        if self.page_number is not None:
            parts.append(f"p.{self.page_number}")
        return " → ".join(parts)

    def as_dict(self) -> dict:
        """Chuyển sang payload JSON-safe cho API/SSE/UI."""
        return {
            "citation_id": self.citation_id,
            "file_name": self.file_name,
            "section_title": self.section_title,
            "page_number": self.page_number,
        }


@dataclass
class AssembledContext:
    """Context, sources và documents sau cùng một lần reorder/budget filtering."""

    text: str = ""
    sources: list[SourceRef] = field(default_factory=list)
    documents: list[dict] = field(default_factory=list)

    @property
    def sources_text(self) -> str:
        """Danh sách sources dùng trong prompt LLM."""
        return "\n".join(
            f"[{source.citation_id}] {source.label}" for source in self.sources
        )


class ContextAssembler:
    """Tổng hợp context với source citation + Lost-in-Middle reorder."""

    def assemble(self, documents: list[dict]) -> AssembledContext:
        """Tổng hợp context + sources từ cùng list sau reorder và budget filtering.

        Args:
            documents: Danh sách documents (đã resolve parent)

        Returns:
            ``AssembledContext`` chứa text, sources và documents thực sự được giữ.
        """
        if not documents:
            return AssembledContext()

        # Bước 1: Lost-in-Middle reorder
        reordered = self._lost_in_middle_reorder(documents)

        # Bước 2: citation ID thuộc về unique source, không thuộc vị trí block.
        source_ids: dict[tuple[str, str, int | None], int] = {}
        sources: list[SourceRef] = []
        kept_documents: list[dict] = []
        context_parts: list[str] = []
        used_chars, dropped = 0, 0

        for doc in reordered:
            key = self._source_key(doc)
            citation_id = source_ids.get(key)
            if citation_id is None:
                citation_id = len(source_ids) + 1

            source = SourceRef(
                citation_id=citation_id,
                file_name=doc.get("file_name") or "Unknown",
                section_title=doc.get("section_title") or None,
                page_number=doc.get("page_number"),
            )
            label = f"[Source {source.citation_id}: {source.label}]"
            block = f"{label}\n{doc['content']}"
            separator_size = len("\n\n---\n\n") if context_parts else 0

            # Hard limit: bỏ cả block, kể cả block đầu tiên nếu tự nó quá lớn.
            if used_chars + separator_size + len(block) > settings.MAX_CONTEXT_CHARS:
                dropped += 1
                continue

            # Chỉ commit source sau khi block thực sự được giữ; source bị drop
            # không được xuất hiện trong prompt/API.
            if key not in source_ids:
                source_ids[key] = citation_id
                sources.append(source)

            kept_documents.append({**doc, "citation_id": citation_id})
            context_parts.append(block)
            used_chars += separator_size + len(block)

        if dropped:
            logger.warning(
                "Context truncated by budget",
                dropped_docs=dropped,
                kept=len(context_parts),
                budget=settings.MAX_CONTEXT_CHARS,
                used=used_chars,
            )

        context_text = "\n\n---\n\n".join(context_parts)
        logger.info("Context assembled", chunks=len(kept_documents), total_chars=len(context_text))
        return AssembledContext(
            text=context_text,
            sources=sources,
            documents=kept_documents,
        )

    @staticmethod
    def _source_key(doc: dict) -> tuple[str, str, int | None]:
        """Key tuple để dedupe cùng file + section + page, không collision dấu ':'."""
        return (
            doc.get("file_name") or "",
            doc.get("section_title") or "",
            doc.get("page_number"),
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
