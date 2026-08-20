"""PR16 bounded PDF window parser tests."""

from types import SimpleNamespace

from pypdf import PdfWriter

from core.config import settings
from ingestion.parsers.pdf_parser import PDFParser


def _pdf(path, pages: int = 5):
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    with path.open("wb") as handle:
        writer.write(handle)


class _Element:
    def __init__(self, page: int | None, text: str):
        self.metadata = SimpleNamespace(page_number=page, coordinates=None)
        self.text = text

    def __str__(self):
        return self.text


def _element(page: int, text: str):
    return _Element(page, text)


def test_pdf_windows_use_bounded_slices_and_global_pages(tmp_path, monkeypatch):
    path = tmp_path / "large.pdf"
    _pdf(path, pages=5)
    monkeypatch.setattr(settings, "INGEST_PDF_PAGE_WINDOW", 2)
    monkeypatch.setattr(settings, "INGEST_PDF_OCR_MODE", "never")
    calls: list[tuple[str, int, int]] = []

    def fake_partition(_path, strategy, *, file=None, starting_page_number=1):
        assert file is not None
        payload = file.read()
        file.seek(0)
        calls.append((strategy, starting_page_number, len(payload)))
        # Unstructured reports page numbers local to the sliced PDF.
        count = 2 if starting_page_number < 5 else 1
        return [_element(index, f"PAGE-{starting_page_number + index - 1}") for index in range(1, count + 1)]

    parser = PDFParser()
    monkeypatch.setattr(parser, "_partition", fake_partition)
    windows = list(parser.iter_windows(path))

    assert [(window.start_page, window.end_page) for window in windows] == [(1, 2), (3, 4), (5, 5)]
    assert [document.metadata.page_number for window in windows for document in window.documents] == [1, 2, 3, 4, 5]
    assert all(call[1] in {1, 3, 5} for call in calls)
    assert len({call[2] for call in calls}) == 2


def test_pdf_ocr_only_fills_missing_pages_without_duplicate_citations(tmp_path, monkeypatch):
    path = tmp_path / "mixed.pdf"
    _pdf(path, pages=4)
    monkeypatch.setattr(settings, "INGEST_PDF_PAGE_WINDOW", 4)
    monkeypatch.setattr(settings, "INGEST_PDF_OCR_PAGE_WINDOW", 1)
    monkeypatch.setattr(settings, "INGEST_PDF_OCR_MODE", "missing_pages")

    def fake_partition(_path, strategy, *, file=None, starting_page_number=1):
        del file
        if strategy == settings.INGEST_PDF_FAST_STRATEGY:
            return [_element(1, "fast page one"), _element(3, "fast page three")]
        return [_element(1, f"ocr page {starting_page_number}")]

    parser = PDFParser()
    monkeypatch.setattr(parser, "_partition", fake_partition)
    window = next(parser.iter_windows(path))

    assert [document.metadata.page_number for document in window.documents] == [1, 2, 3, 4]
    assert [document.content for document in window.documents] == [
        "fast page one",
        "ocr page 2",
        "fast page three",
        "ocr page 4",
    ]
    assert window.quality.ocr_pages == 2


def test_pdf_missing_page_metadata_fails_closed(tmp_path, monkeypatch):
    path = tmp_path / "bad.pdf"
    _pdf(path, pages=1)

    def fake_partition(_path, _strategy, *, file=None, starting_page_number=1):
        del file, starting_page_number
        return [_Element(None, "text")]

    parser = PDFParser()
    monkeypatch.setattr(parser, "_partition", fake_partition)

    import pytest

    with pytest.raises(ValueError, match="missing_page_metadata"):
        next(parser.iter_windows(path))
