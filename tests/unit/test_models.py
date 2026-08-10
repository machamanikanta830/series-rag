"""Tests for immutable document and chunk models."""

from dataclasses import FrozenInstanceError

import pytest

from app.models import Chunk, Document, DocumentSection, SearchResult


def test_document_and_metadata_are_immutable() -> None:
    """Documents copy metadata and prevent later mutation."""
    source_metadata = {"topic": "cloud"}
    document = Document(
        document_id="document-1",
        source_name="lesson.txt",
        source_path="lesson.txt",
        text="A short lesson.",
        metadata=source_metadata,
    )

    source_metadata["topic"] = "changed"

    assert document.metadata == {"topic": "cloud"}
    with pytest.raises(FrozenInstanceError):
        document.text = "Changed text."
    with pytest.raises(TypeError):
        document.metadata["topic"] = "changed"


def test_search_result_is_immutable() -> None:
    """A ranked result cannot be changed after search returns it."""
    document = Document(
        document_id="document-1",
        source_name="lesson.txt",
        source_path="lesson.txt",
        text="A short lesson.",
    )
    result = SearchResult(
        chunk=Chunk(
            chunk_id="chunk-1",
            document_id=document.document_id,
            source_name=document.source_name,
            text=document.text,
            chunk_index=0,
            start_word=0,
            end_word=3,
        ),
        score=0.9,
    )

    with pytest.raises(FrozenInstanceError):
        result.score = 0.1


def test_document_sections_and_their_metadata_are_immutable() -> None:
    """Page-aware sections cannot be reordered or have metadata rewritten."""
    section_metadata = {"page_number": "1"}
    sections = [DocumentSection(text="Page text.", metadata=section_metadata)]

    document = Document(
        document_id="document-1",
        source_name="lesson.pdf",
        source_path="lesson.pdf",
        text="Page text.",
        sections=tuple(sections),
    )
    section_metadata["page_number"] = "2"
    sections.clear()

    assert document.sections[0].metadata == {"page_number": "1"}
    with pytest.raises(TypeError):
        document.sections[0].metadata["page_number"] = "2"
