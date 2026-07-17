"""Tests for immutable document and chunk models."""

from dataclasses import FrozenInstanceError

import pytest

from app.models import Chunk, Document, SearchResult


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
