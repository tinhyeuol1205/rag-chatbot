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
      │     → hyde_vector (1024d for BGE-M3)
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
      │     → AssembledContext (text + sources + kept documents)
      │
      └─⑧ LLM Generation (provider-agnostic)
            → Final answer + source citation
"""

from __future__ import annotations

from contextvars import copy_context
from dataclasses import dataclass, field
from unicodedata import normalize

from core import get_logger
from core.config import settings
from core.llm import get_llm_service
from retrieval.context.assembler import AssembledContext, ContextAssembler, SourceRef
from retrieval.context.parent_resolver import ParentResolver
from retrieval.prompts import RAG_USER_PROMPT, SYSTEM_PROMPT
from retrieval.query_transform.condense import QueryCondenser
from retrieval.query_transform.hyde import HyDEGenerator
from retrieval.query_transform.multi_query import MultiQueryExpander
from retrieval.reranking.cross_encoder import CrossEncoderReranker
from retrieval.scope import RetrievalScope, default_scope
from retrieval.search.hybrid import HybridSearcher, rrf_fusion

logger = get_logger(__name__)

NO_CONTEXT_MSG = (
    "I don't have enough information in the company documents "
    "to answer this question."
)


def _normalize_query_key(query: str) -> str:
    """Normalize query text for plan deduplication without changing search text."""
    return " ".join(normalize("NFKC", query).casefold().split())


def _dedupe_queries(queries: list[str]) -> list[str]:
    """Keep the first spelling of each normalized query in deterministic order."""
    seen: set[str] = set()
    unique: list[str] = []
    for query in queries:
        if not isinstance(query, str):
            continue
        key = _normalize_query_key(query)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(query.strip())
    return unique


@dataclass
class RAGResult:
    """Kết quả đầy đủ của 1 lượt RAG — dùng cho evaluation và debug."""

    answer: str
    contexts: list[str] = field(default_factory=list)
    sources: list[SourceRef] = field(default_factory=list)
    expanded_queries: list[str] = field(default_factory=list)
    num_candidates: int = 0


@dataclass
class RetrievalResult:
    """Kết quả retrieval sau cùng một lần assemble/reorder/budget filtering."""

    assembled: AssembledContext
    expanded_queries: list[str] = field(default_factory=list)
    search_query: str = ""
    num_reranked_candidates: int = 0


class RAGRetriever:
    """Main orchestrator — kết nối tất cả components."""

    def __init__(self, scope: RetrievalScope | None = None):
        # Scope is constructed by the server/configuration boundary.  It is not
        # read from ChatRequest, so a caller cannot switch datasets per request.
        self.scope = scope or default_scope()
        self.expander = MultiQueryExpander()
        self.hyde = HyDEGenerator()
        self.searcher = HybridSearcher()
        self.reranker = CrossEncoderReranker()
        self.parent_resolver = ParentResolver()
        self.assembler = ContextAssembler()
        self.llm = get_llm_service()
        self.condenser = QueryCondenser()

    def retrieve(
        self,
        user_query: str,
        history: list[tuple[str, str]] | None = None,
    ) -> RetrievalResult:
        """Chạy retrieval và trả về context có cấu trúc.

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
            # ContextVar carries the Redis reservation.  ThreadPoolExecutor
            # does not inherit it automatically, so copy a context per task;
            # otherwise optional LLM calls would see no reservation and be
            # rejected in production mode.
            f_expand = pool.submit(copy_context().run, self.expander.expand, search_query)
            f_hyde = pool.submit(copy_context().run, self.hyde.generate_embedding, search_query)
            expanded_queries = f_expand.result()
            hyde_vector = f_hyde.result()

        # ③ Build a deterministic query plan.  MultiQueryExpander historically
        # returned the original query as its first item; searching that list after
        # the direct hybrid search double-counted the sparse signal in RRF.
        unique_queries = _dedupe_queries([search_query, *expanded_queries])
        variant_queries = unique_queries[1:]

        # Direct channel: dense(query) + sparse(query), exactly once.
        result_lists = [self.searcher.search(search_query, scope=self.scope)]
        weights = [1.0]

        # HyDE is a separate dense-only vote.  Its sparse side would be identical
        # to the direct channel and must not be counted a second time.
        if hyde_vector:
            result_lists.append(
                self.searcher.search(
                    search_query,
                    hyde_vector=hyde_vector,
                    scope=self.scope,
                    include_sparse=False,
                )
            )
            weights.append(1.0)

        # Each distinct expansion gets one normal hybrid vote.
        for expanded_query in variant_queries:
            result_lists.append(
                self.searcher.search(expanded_query, scope=self.scope)
            )
            weights.append(1.0)

        # ④ RRF Fusion trên TẤT CẢ cùng lúc — điểm cộng dồn qua mọi query
        unique = rrf_fusion(*result_lists, weights=weights)
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
        resolved = self.parent_resolver.resolve(top_chunks, scope=self.scope)

        # ⑦ Context Assembly
        assembled = self.assembler.assemble(resolved)

        return RetrievalResult(
            assembled=assembled,
            expanded_queries=unique_queries,
            search_query=search_query,
            num_reranked_candidates=len(top_chunks),
        )

    def warmup(self) -> None:
        """Load các model nặng TRƯỚC khi nhận request (fail-fast, bug P2-14).

        Chỉ chạm embedding + reranker model (load về RAM 1 lần); KHÔNG gọi LLM
        API (không tốn tiền) và không chạy inference/query. Sample-free: truy cập
        property ``model`` là loader load model về RAM ngay (không embed thử).
        """
        if not self.searcher.dense.embedder.is_remote:
            _ = self.searcher.dense.embedder.model
        if not self.reranker.is_remote:
            _ = self.reranker.model
        logger.info("RAG retriever warmed up")

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
        retrieval = self.retrieve(user_query, history=history)
        context = retrieval.assembled.text
        sources = retrieval.assembled.sources_text
        search_query = retrieval.search_query

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

    def query_with_context(
        self,
        user_query: str,
        history: list[tuple[str, str]] | None = None,
    ) -> RAGResult:
        """Dùng cho evaluation — trả về ĐÚNG context đã đưa vào LLM (bug P0-4)."""
        retrieval = self.retrieve(user_query, history=history)
        assembled = retrieval.assembled
        answer = (
            self._generate(
                retrieval.search_query,
                assembled.text,
                assembled.sources_text,
            )
            if assembled.text.strip()
            else NO_CONTEXT_MSG
        )
        return RAGResult(
            answer=answer,
            contexts=[d["content"] for d in assembled.documents],
            sources=assembled.sources,
            expanded_queries=retrieval.expanded_queries,
            num_candidates=len(assembled.documents),
        )

    def stream_with_sources(
        self,
        user_query: str,
        history: list[tuple[str, str]] | None = None,
    ):
        """Stream token events rồi sources sau khi generation hoàn tất.

        Event data là Python object; API adapter chịu trách nhiệm serialize JSON cho
        SSE, còn Gradio có thể dùng trực tiếp để render source list.
        """
        retrieval = self.retrieve(user_query, history=history)
        assembled = retrieval.assembled

        if not assembled.text.strip():
            yield {"event": "token", "data": NO_CONTEXT_MSG}
        else:
            yield from (
                {"event": "token", "data": token}
                for token in self._generate_stream(
                    retrieval.search_query,
                    assembled.text,
                    assembled.sources_text,
                )
            )

        yield {
            "event": "sources",
            "data": [source.as_dict() for source in assembled.sources],
        }

    def _generate(self, query: str, context: str, sources: str) -> str:
        """Gọi LLM sinh câu trả lời (non-streaming)."""
        user_prompt = RAG_USER_PROMPT.format(
            context=context, sources=sources, query=query
        )
        answer = self.llm.generate(
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.1,  # Thấp → trả lời sát context, ít hallucination
            max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
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
            max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
        )
