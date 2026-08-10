from ingestion.chunking.parent_child import parent_child_chunk
from ingestion.models import DocumentMetadata, RawDocument


def test_preserves_section_title():
    """★ Bug P1-2: section_title thật KHÔNG được ghi đè thành 'Section N'."""
    docs = [
        RawDocument(content="## Password Policy\n" + "x" * 500,
                    metadata=DocumentMetadata(file_name="p.md", file_type="md",
                                              section_title="Password Policy")),
    ]
    parents, children = parent_child_chunk(docs)
    assert all(p.metadata.section_title == "Password Policy" for p in parents)
    assert all(c.metadata.section_title == "Password Policy" for c in children)


def test_preserves_page_number():
    docs = [
        RawDocument(content="y" * 500,
                    metadata=DocumentMetadata(file_name="d.pdf", file_type="pdf",
                                              page_number=12)),
    ]
    parents, children = parent_child_chunk(docs)
    assert all(p.metadata.page_number == 12 for p in parents)
    assert all(c.metadata.page_number == 12 for c in children)


def test_children_link_to_correct_parent():
    docs = [
        RawDocument(content="a" * 3000,
                    metadata=DocumentMetadata(file_name="a.md", file_type="md")),
    ]
    parents, children = parent_child_chunk(docs)
    parent_ids = {p.chunk_id for p in parents}
    assert len(parents) > 1                      # đủ dài để tách nhiều parent
    assert all(c.parent_id in parent_ids for c in children)


def test_fallback_title_when_no_metadata():
    """Không có section_title/page_number → fallback 'Section N'."""
    docs = [
        RawDocument(content="z" * 100,
                    metadata=DocumentMetadata(file_name="a.md", file_type="md")),
    ]
    parents, _ = parent_child_chunk(docs)
    assert parents[0].metadata.section_title == "Section 1"


def test_empty_documents():
    assert parent_child_chunk([]) == ([], [])
