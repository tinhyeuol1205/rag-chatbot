from __future__ import annotations

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
      │     → merged_results (20+ chunks)
      │
      ├─④ Deduplicate kết quả từ tất cả queries
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
      └─⑧ LLM Generation (GPT-4o-mini)
            → Final answer + source citation
"""

from openai import OpenAI

from core import get_logger
from core.config import settings
from retrieval.context.assembler import ContextAssembler
from retrieval.context.parent_resolver import ParentResolver
from retrieval.prompts import RAG_USER_PROMPT, SYSTEM_PROMPT
from retrieval.query_transform.hyde import HyDEGenerator
from retrieval.query_transform.multi_query import MultiQueryExpander
from retrieval.reranking.cross_encoder import CrossEncoderReranker
from retrieval.search.hybrid import HybridSearcher

logger = get_logger(__name__)


class RAGRetriever:
    """Main orchestrator — kết nối tất cả components."""

    def __init__(self):
        self.expander = MultiQueryExpander()
        self.hyde = HyDEGenerator()
        self.searcher = HybridSearcher()
        self.reranker = CrossEncoderReranker()
        self.parent_resolver = ParentResolver()
        self.assembler = ContextAssembler()
        # self.llm = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.llm = OpenAI(
            base_url = settings.OPENAI_BASE_URL,
            api_key = settings.OPENAI_API_KEY
        )

    def query(self, user_query: str, stream: bool = False):
        """Xử lý câu hỏi qua toàn bộ RAG pipeline.

        Args:
            user_query: Câu hỏi của user
            stream: True → trả về generator (SSE), False → trả về string

        Returns:
            str hoặc generator — câu trả lời từ LLM
        """
        logger.info("RAG query started", query=user_query[:80])

        # ① Multi-Query Expansion
        expanded_queries = self.expander.expand(user_query)

        # ② HyDE — sinh hypothetical answer embedding
        hyde_vector = self.hyde.generate_embedding(user_query)

        # ③ Hybrid Search cho mỗi query + HyDE
        all_results = []

        # Search bằng HyDE vector (query đầu tiên)
        hyde_results = self.searcher.search(user_query, hyde_vector=hyde_vector)
        all_results.extend(hyde_results)

        # Search bằng mỗi expanded query (không dùng HyDE)
        for eq in expanded_queries:
            results = self.searcher.search(eq)
            all_results.extend(results)

        # ④ Deduplicate
        unique = self._deduplicate(all_results)
        logger.info("After dedup", total=len(all_results), unique=len(unique))

        # ⑤ Cross-Encoder Reranking
        top_chunks = self.reranker.rerank(user_query, unique)

        # ⑥ Parent Resolution
        resolved = self.parent_resolver.resolve(top_chunks)

        # ⑦ Context Assembly
        context_text, sources_text = self.assembler.assemble(resolved)

        # ⑧ LLM Generation
        if stream:
            return self._generate_stream(user_query, context_text, sources_text)
        else:
            return self._generate(user_query, context_text, sources_text)

    def _generate(self, query: str, context: str, sources: str) -> str:
        """Gọi LLM sinh câu trả lời (non-streaming)."""
        response = self.llm.chat.completions.create(
            model=settings.OPENAI_MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": RAG_USER_PROMPT.format(
                    context=context, sources=sources, query=query
                )},
            ],
            temperature=0.1,  # Thấp → trả lời sát context, ít hallucination
        )

        answer = response.choices[0].message.content
        logger.info("Answer generated", length=len(answer))
        return answer

    def _generate_stream(self, query: str, context: str, sources: str):
        """Gọi LLM sinh câu trả lời (streaming — từng token)."""
        stream = self.llm.chat.completions.create(
            model=settings.OPENAI_MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": RAG_USER_PROMPT.format(
                    context=context, sources=sources, query=query
                )},
            ],
            temperature=0.1,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _deduplicate(self, results: list[dict]) -> list[dict]:
        """Loại bỏ duplicate chunks (giữ bản có score cao nhất)."""
        seen: dict[str, dict] = {}
        for doc in results:
            cid = doc["chunk_id"]
            if cid not in seen:
                seen[cid] = doc
            else:
                # Giữ bản có RRF score cao hơn
                existing_score = seen[cid].get("rrf_score", seen[cid].get("score", 0))
                new_score = doc.get("rrf_score", doc.get("score", 0))
                if new_score > existing_score:
                    seen[cid] = doc

        return list(seen.values())
