from pathlib import Path

from ingestion.parsers.markdown_parser import MarkdownParser


def test_extracts_section_titles(tmp_path: Path):
    f = tmp_path / "doc.md"
    f.write_text("# Title\nintro\n\n## Password Policy\nmin 12 chars\n", encoding="utf-8")
    docs = MarkdownParser().parse(f)
    titles = [d.metadata.section_title for d in docs]
    assert "Password Policy" in titles


def test_splits_h3_headers(tmp_path: Path):
    """### cũng phải tách section, không dồn vào 1 doc khổng lồ."""
    f = tmp_path / "doc.md"
    f.write_text("## A\ntext a\n\n### B\ntext b\n", encoding="utf-8")
    docs = MarkdownParser().parse(f)
    assert len(docs) == 2


def test_ignores_hash_inside_code_fence(tmp_path: Path):
    """Dòng '# comment' trong code block KHÔNG được coi là header."""
    f = tmp_path / "doc.md"
    f.write_text("## Setup\n```bash\n# install deps\npip install x\n```\ndone\n",
                 encoding="utf-8")
    docs = MarkdownParser().parse(f)
    assert len(docs) == 1
    assert "# install deps" in docs[0].content


def test_txt_file_type(tmp_path: Path):
    """File .txt qua MarkdownParser → file_type='txt', không phải 'md'."""
    f = tmp_path / "notes.txt"
    f.write_text("Some plain text\n", encoding="utf-8")
    docs = MarkdownParser().parse(f)
    assert docs[0].metadata.file_type == "txt"
