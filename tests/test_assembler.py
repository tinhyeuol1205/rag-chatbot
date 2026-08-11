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
    assembled = ContextAssembler().assemble(docs)
    ctx_nums = re.findall(r"\[Source (\d+):", assembled.text)
    src_nums = [str(source.citation_id) for source in assembled.sources]
    assert ctx_nums == src_nums


def test_context_budget_drops_whole_blocks(monkeypatch):
    """★ P2-10: vượt budget thì bỏ cả block, không cắt giữa câu."""
    from core.config import settings
    monkeypatch.setattr(settings, "MAX_CONTEXT_CHARS", 200)
    docs = [{"chunk_id": str(i), "content": "x" * 150, "file_name": "f.md"}
            for i in range(5)]
    assembled = ContextAssembler().assemble(docs)
    assert len(assembled.text) < 400       # chỉ giữ được 1 block
    assert "x" * 150 in assembled.text     # block được giữ nguyên vẹn


def test_empty_documents():
    assembled = ContextAssembler().assemble([])
    assert assembled.text == ""
    assert assembled.sources == []
    assert assembled.documents == []


def test_format_source_with_page():
    doc = {"file_name": "handbook.pdf", "section_title": "Leave",
           "page_number": 12}
    assembled = ContextAssembler().assemble([{**doc, "content": "text"}])
    assert assembled.sources[0].label == "handbook.pdf → Leave → p.12"


def test_duplicate_source_reuses_same_citation_id():
    docs = [
        {"content": "first", "file_name": "same.md", "section_title": "Policy"},
        {"content": "second", "file_name": "same.md", "section_title": "Policy"},
    ]
    assembled = ContextAssembler().assemble(docs)

    assert assembled.text.count("[Source 1:") == 2
    assert len(assembled.sources) == 1
    assert assembled.sources[0].citation_id == 1
    assert [doc["citation_id"] for doc in assembled.documents] == [1, 1]


def test_context_budget_is_hard_limit_including_separators(monkeypatch):
    from core.config import settings
    monkeypatch.setattr(settings, "MAX_CONTEXT_CHARS", 200)
    docs = [{"content": "x" * 70, "file_name": f"f{i}.md"} for i in range(5)]
    assembled = ContextAssembler().assemble(docs)
    assert len(assembled.text) <= 200


def test_dropped_source_is_not_exposed(monkeypatch):
    from core.config import settings
    monkeypatch.setattr(settings, "MAX_CONTEXT_CHARS", 100)
    docs = [
        {"content": "x" * 50, "file_name": "kept.md"},
        {"content": "y" * 50, "file_name": "dropped.md"},
    ]
    assembled = ContextAssembler().assemble(docs)
    assert {source.file_name for source in assembled.sources} == {
        doc["file_name"] for doc in assembled.documents
    }
