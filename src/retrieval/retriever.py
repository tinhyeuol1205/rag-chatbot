"""
RAG Retriever — Main Orchestrator ★

Đây là "bộ não" kết nối TẤT CẢ kỹ thuật RAG lại với nhau.

Luồng xử lý đầy đủ:

  User Query
      │
      ├─① Multi-Query Expansion (LLM sinh 3 biến thể)
      │     → [query_gốc, variant_1, variant_2, variant_3]
      │
      ├─② HyDE (LLM sinh hypothetical answer → embed)
      │     → hyde_vector (384d)
      │
      ├─③ Hybrid Search cho MỖI query (Dense + BM25 + RRF)
      │     → 1 result_list cho mỗi query
      │
      ├─④ RRF Fusion trên TẤT CẢ queries (điểm cộng dồn)
      │     → unique_results
      │
      ├─⑤ Cross-Encoder Reranking
      │     → top_5 chunks (chính xác nhất)
      │
      ├─⑥ Parent Resolution (child → parent chunk)
      │     → full_context chunks
      │
      ├─⑦ Context Assembly + Lost-in-Middle reorder
      │     → context_text + sources_text
      │
      └─⑧ LLM Generation (provider-agnostic)
            → Final answer + source citation
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core import get_logger
from core.config import settings
from core.llm import get_llm_service
from retrieval.context.assembler import ContextAssembler
from retrieval.context.parent_resolver import ParentResolver
from retrieval.prompts import RAG_USER_PROMPT, SYSTEM_PROMPT
from retrieval.query_transform.condense import QueryCondenser
from retrieval.query_transform.hyde import HyDEGenerator
from retrieval.query_transform.multi_query import MultiQueryExpander
from retrieval.reranking.cross_encoder import CrossEncoderReranker
from retrieval.search.hybrid import HybridSearcher, rrf_fusion

logger = get_logger(__name__)

NO_CONTEXT_MSG = (
    "I don't have enough information in the company documents "
    "to answer this question."
)


@dataclass
class RAGResult:
    """Kết quả đầy đủ của 1 lượt RAG — dùng cho evaluation và debug."""

    answer: str
    contexts: list[str] = field(default_factory=list)
    sources: str = ""
    expanded_queries: list[str] = field(default_factory=list)
    num_candidates: int = 0


class RAGRetriever:
    """Main orchestrator — kết nối tất cả components."""

    def __init__(self):
        self.expander = MultiQueryExpander()
        self.hyde = HyDEGenerator()
        self.searcher = HybridSearcher()
        self.reranker = CrossEncoderReranker()
        self.parent_resolver = ParentResolver()
        self.assembler = ContextAssembler()
        self.llm = get_llm_service()
        self.condenser = QueryCondenser()

    def retrieve(self, user_query: str, history: list[tuple[str, str]] | None = None
                 ) -> tuple[list[dict], str, str, list[str], str]:
        """Chạy toàn bộ retrieval, trả về (resolved_docs, context, sources, expanded, search_query).

        Tách riêng khỏi generate để evaluation lấy được context THẬT đã đưa vào LLM
        (bug P0-4: trước đây evaluate tự search riêng → contexts là child chunk,
        không phải parent chunk LLM thật nhận).
        """
        logger.info("RAG query started", query=user_query[:80])

        # ⓿ Condense — resolve đại từ/tham chiếu từ history TRƯỚC khi retrieval
        search_query = self.condenser.condense(user_query, history or [])

        # ① + ② Song song hoá: expand và HyDE KHÔNG phụ thuộc nhau → tiết kiệm 1 LLM call
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_expand = pool.submit(self.expander.expand, search_query)
            f_hyde = pool.submit(self.hyde.generate_embedding, search_query)
            expanded_queries = f_expand.result()
            hyde_vector = f_hyde.result()

        # ③ Hybrid Search — thu từng result_list RIÊNG BIỆT (không gộp chung)
        # ★ Dùng search_query (đã condense) cho MỌI search — kể cả query đầu tiên
        result_lists = [self.searcher.search(search_query, hyde_vector=hyde_vector)]
        for eq in expanded_queries:
            result_lists.append(self.searcher.search(eq))

        # ④ RRF Fusion trên TẤT CẢ cùng lúc — điểm cộng dồn qua mọi query
        unique = rrf_fusion(*result_lists)
        logger.info(
            "Fusion done",
            n_lists=len(result_lists),
            total_rows=sum(len(r) for r in result_lists),
            unique=len(unique),
            top_n_hits=unique[0]["n_hits"] if unique else 0,
        )

        # ⑤ Cross-Encoder Reranking — rerank theo search_query (đã condense)
        top_chunks = self.reranker.rerank(search_query, unique)

        # ⑥ Parent Resolution
        resolved = self.parent_resolver.resolve(top_chunks)

        # ⑦ Context Assembly
        context_text, sources_text = self.assembler.assemble(resolved)

        return resolved, context_text, sources_text, expanded_queries, search_query

    def query(self, user_query: str, stream: bool = False,
              history: list[tuple[str, str]] | None = None):
        """Xử lý câu hỏi qua toàn bộ RAG pipeline.

        Args:
            user_query: Câu hỏi của user
            stream: True → trả về generator (SSE), False → trả về string
            history: List các (user, assistant) cho multi-turn — condense trước khi retrieval

        Returns:
            str hoặc generator — câu trả lời từ LLM
        """
        _, context, sources, _, search_query = self.retrieve(user_query, history=history)

        # ⑧ LLM Generation — short-circuit nếu context rỗng (khỏi tốn LLM call vô ích)
        if not context.strip():
            logger.warning("Empty context — skipping LLM call", query=user_query[:80])
            return iter([NO_CONTEXT_MSG]) if stream else NO_CONTEXT_MSG

        # ★ Dùng search_query (đã condense) cho generate — để LLM thấy câu hỏi độc lập,
        #   không phải câu gốc chứa đại từ mơ hồ ("change it?")
        if stream:
            return self._generate_stream(search_query, context, sources)
        else:
            return self._generate(search_query, context, sources)

    def query_with_context(self, user_query: str) -> RAGResult:
        """Dùng cho evaluation — trả về ĐÚNG context đã đưa vào LLM (bug P0-4)."""
        resolved, context, sources, expanded, search_query = self.retrieve(user_query)
        answer = (self._generate(search_query, context, sources)
                  if context.strip() else NO_CONTEXT_MSG)
        return RAGResult(
            answer=answer,
            contexts=[d["content"] for d in resolved],   # ★ context THẬT (parent chunk)
            sources=sources,
            expanded_queries=expanded,
            num_candidates=len(resolved),
        )

    def _generate(self, query: str, context: str, sources: str) -> str:
        """Gọi LLM sinh câu trả lời (non-streaming)."""
        user_prompt = RAG_USER_PROMPT.format(
            context=context, sources=sources, query=query
        )
        answer = self.llm.generate(
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.1,  # Thấp → trả lời sát context, ít hallucination
        )

        logger.info("Answer generated", length=len(answer))
        return answer

    def _generate_stream(self, query: str, context: str, sources: str):
        """Gọi LLM sinh câu trả lời (streaming — từng token)."""
        user_prompt = RAG_USER_PROMPT.format(
            context=context, sources=sources, query=query
        )
        yield from self.llm.generate_stream(
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.1,
        )

