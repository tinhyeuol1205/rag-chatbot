from __future__ import annotations

"""
Multi-Query Expansion ★ — Kỹ thuật RAG #3.

Vấn đề:
  User hỏi "laptop policy" → chỉ search 1 query → có thể bỏ sót documents liên quan.

Giải pháp:
  Dùng LLM sinh 3 biến thể câu hỏi:
    "laptop policy"
    → "What is the company policy on laptop equipment?"
    → "How do employees receive their work computer?"
    → "What are the security requirements for company laptops?"

  Search cả 4 queries (gốc + 3 biến thể) → hợp nhất kết quả → recall cao hơn.

Tham khảo: rag_master.md — Module 4, mục 4.1, technique #1
"""

import re

from core import get_logger
from core.config import settings
from core.llm import get_llm_service

logger = get_logger(__name__)

# Regex làm sạch output LLM của Multi-Query
_NUMBERING_RE = re.compile(r"^\s*(?:\d+[\.\)]|[-*•])\s*")
_PREAMBLE_RE = re.compile(
    r"^(here (are|is)|sure|certainly|below are|alternative|these are)\b", re.IGNORECASE
)

MULTI_QUERY_PROMPT = """You are a helpful assistant. Your task is to generate {n} different versions
of the given user question to retrieve relevant documents from a knowledge base.

By generating multiple perspectives on the question, you help overcome limitations of
single-query similarity search.

Provide these alternative questions separated by newlines.
Do NOT include the original question. Do NOT number the questions.

Original question: {query}"""


class MultiQueryExpander:
    """Sinh nhiều biến thể câu hỏi để tăng recall khi search."""

    def __init__(self):
        self.llm = get_llm_service()

    def expand(self, query: str) -> list[str]:
        """Sinh N biến thể + trả về cùng query gốc.

        Args:
            query: Câu hỏi gốc của user

        Returns:
            [original_query, variant_1, variant_2, ...variant_N]
        """
        logger.info("Expanding query", original=query, n_variants=settings.EXPAND_N_QUERY)

        try:
            raw_output = self.llm.generate(
                user_prompt=MULTI_QUERY_PROMPT.format(
                    n=settings.EXPAND_N_QUERY,
                    query=query,
                ),
                temperature=0.7,  # Creativity cao để tạo biến thể đa dạng
                max_tokens=300,
            )
        except Exception as e:
            # ★ Degrade: không có variant vẫn search được bằng query gốc
            logger.warning("Multi-query expansion failed, using original query only",
                           error=str(e))
            return [query]

        variants = self._parse_variants(raw_output, exclude=query)

        # Ghép query gốc + các biến thể
        all_queries = [query] + variants[:settings.EXPAND_N_QUERY]

        logger.info("Query expanded", total_queries=len(all_queries), variants=variants)
        return all_queries

    def _parse_variants(self, raw_output: str, exclude: str) -> list[str]:
        """Làm sạch output LLM: bỏ numbering, preamble, dòng quá ngắn, trùng query gốc.

        LLM thường trả kèm "Here are 3 versions:", "1. ...", hoặc lặp lại câu hỏi gốc.
        Nếu không làm sạch, các dòng này bị đưa thẳng vào search như query thật —
        tốn 1 lượt hybrid search + bơm nhiễu vào RRF fusion (bug P3-5).
        """
        seen = {exclude.strip().lower()}
        variants: list[str] = []
        for line in raw_output.splitlines():
            line = _NUMBERING_RE.sub("", line.strip()).strip().strip('"')
            if len(line) < 8:                    # quá ngắn để là câu hỏi
                continue
            if _PREAMBLE_RE.match(line):         # "Here are 3 versions:"
                continue
            if line.endswith(":"):               # dòng tiêu đề
                continue
            key = line.lower()
            if key in seen:                      # trùng query gốc hoặc variant trước
                continue
            seen.add(key)
            variants.append(line)
        return variants

