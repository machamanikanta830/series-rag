"""Tests for native-text PDF extraction and page-aware chunking."""

import pymupdf
import pytest

from app.chunking import chunk_document
from app.parsers.pdf import (
    PDFNoExtractableTextError,
    PDFParsingError,
    parse_pdf,
)


def _pdf_bytes(page_texts: list[str | None]) -> bytes:
    """Create a small in-memory PDF without relying on external fixtures."""
    pdf = pymupdf.open()
    for text in page_texts:
        page = pdf.new_page()
        if text is not None:
            page.insert_text((72, 72), text)
    content = pdf.tobytes()
    pdf.close()
    return content


def test_extracts_native_text_page_by_page() -> None:
    """Each nonempty page becomes an ordered immutable document section."""
    document = parse_pdf(
        _pdf_bytes(["First page lesson.", "Second page lesson."]),
        "course.pdf",
    )

    assert document.source_name == "course.pdf"
    assert document.text == "First page lesson.\n\nSecond page lesson."
    assert [section.text for section in document.sections] == [
        "First page lesson.",
        "Second page lesson.",
    ]
    assert [section.metadata["page_number"] for section in document.sections] == [
        "1",
        "2",
    ]
    assert document.metadata == {
        "filename": "course.pdf",
        "page_count": "2",
        "source_type": "pdf",
    }


def test_empty_pages_are_skipped_without_losing_physical_page_numbers() -> None:
    """A later text page retains its PDF page number after a blank page."""
    document = parse_pdf(
        _pdf_bytes(["First page.", None, "Third page."]),
        "pages.pdf",
    )

    assert [section.metadata["page_number"] for section in document.sections] == [
        "1",
        "3",
    ]
    assert document.metadata["page_count"] == "3"


def test_chunking_preserves_page_provenance_without_crossing_pages() -> None:
    """Short adjacent pages remain separate chunks with explicit page metadata."""
    document = parse_pdf(
        _pdf_bytes(["alpha beta", "gamma delta"]),
        "pages.pdf",
    )

    chunks = chunk_document(document, chunk_size=10, chunk_overlap=2)

    assert [chunk.text for chunk in chunks] == ["alpha beta", "gamma delta"]
    assert [(chunk.start_word, chunk.end_word) for chunk in chunks] == [(0, 2), (2, 4)]
    assert [chunk.metadata["page_number"] for chunk in chunks] == ["1", "2"]
    assert all(chunk.metadata["filename"] == "pages.pdf" for chunk in chunks)


def test_rejects_a_malformed_pdf() -> None:
    """Invalid bytes become one stable parser-level error."""
    with pytest.raises(PDFParsingError, match="malformed or unreadable"):
        parse_pdf(b"not a PDF", "broken.pdf")


def test_rejects_a_pdf_without_extractable_text() -> None:
    """Blank or image-only-style pages do not proceed to ingestion."""
    with pytest.raises(PDFNoExtractableTextError, match="no extractable native text"):
        parse_pdf(_pdf_bytes([None, None]), "blank.pdf")
