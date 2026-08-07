import re

from retrieval.context.assembler import ContextAssembler


def test_lost_in_middle_zigzag():
    docs = [{"chunk_id": str(i), "content": str(i), "file_name": "f"} for i in range(1, 6)]
    out = ContextAssembler()._lost_in_middle_reorder(docs)
    assert [d["content"] for d in out] == ["1", "3", "5", "4", "2"]


def test_reorder_short_lists_unchanged():
    for n in (0, 1, 2):
        docs = [{"content": str(i)} for i in range(n)]
        assert ContextAssembler()._lost_in_middle_reorder(docs) == docs


def test_source_numbers_match_context_labels():
    """★ Bug P2-1: số trong [Source N] phải khớp số trong danh sách Sources."""
    docs = [{"chunk_id": str(i), "content": f"c{i}",
             "file_name": f"f{i}.md", "section_title": f"S{i}"} for i in range(1, 6)]
    context, sources = ContextAssembler().assemble(docs)
    ctx_nums = re.findall(r"\[Source (\d+):", context)
    src_nums = re.findall(r"^\[(\d+)\]", sources, re.MULTILINE)
    assert ctx_nums == src_nums


def test_context_budget_drops_whole_blocks(monkeypatch):
    """★ P2-10: vượt budget thì bỏ cả block, không cắt giữa câu."""
    from core.config import settings
    monkeypatch.setattr(settings, "MAX_CONTEXT_CHARS", 200)
    docs = [{"chunk_id": str(i), "content": "x" * 150, "file_name": "f.md"}
            for i in range(5)]
    context, _ = ContextAssembler().assemble(docs)
    assert len(context) < 400            # chỉ giữ được 1 block
    assert "x" * 150 in context          # block được giữ nguyên vẹn


def test_empty_documents():
    assert ContextAssembler().assemble([]) == ("", "")


def test_format_source_with_page():
    doc = {"file_name": "handbook.pdf", "section_title": "Leave",
           "page_number": 12}
    assert ContextAssembler._format_source(doc) == "handbook.pdf → Leave → p.12"
