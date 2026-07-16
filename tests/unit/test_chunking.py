"""Tests for explicit, fixed-size word chunking."""

from collections.abc import Mapping

import pytest

from app.chunking import chunk_document
from app.models import Document


def _document(words: list[str], metadata: Mapping[str, str] | None = None) -> Document:
    """Create a document with predictable source information for chunk tests."""
    return Document(
        document_id="document-1",
        source_name="lesson.txt",
        source_path="lesson.txt",
        text=" ".join(words),
        metadata={} if metadata is None else metadata,
    )


def test_short_document_produces_one_chunk() -> None:
    """A document shorter than the chunk size remains one chunk."""
    document = _document(["one", "two", "three"])

    [chunk] = chunk_document(document, chunk_size=6, chunk_overlap=2)

    assert chunk.text == "one two three"
    assert (chunk.chunk_index, chunk.start_word, chunk.end_word) == (0, 0, 3)


def test_exact_chunk_size_produces_one_chunk() -> None:
    """A document exactly the chunk size does not create an extra chunk."""
    document = _document(["one", "two", "three", "four"])

    chunks = chunk_document(document, chunk_size=4, chunk_overlap=1)

    assert [chunk.text for chunk in chunks] == ["one two three four"]
    assert [(chunk.start_word, chunk.end_word) for chunk in chunks] == [(0, 4)]


def test_multiple_chunks_use_the_requested_overlap_and_offsets() -> None:
    """Chunk boundaries preserve word order and use a two-word overlap."""
    words = [f"word-{index}" for index in range(10)]
    document = _document(words)

    chunks = chunk_document(document, chunk_size=6, chunk_overlap=2)

    assert [chunk.text for chunk in chunks] == [
        "word-0 word-1 word-2 word-3 word-4 word-5",
        "word-4 word-5 word-6 word-7 word-8 word-9",
    ]
    assert [(chunk.start_word, chunk.end_word) for chunk in chunks] == [(0, 6), (4, 10)]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]


def test_zero_overlap_advances_by_the_full_chunk_size() -> None:
    """Zero overlap produces adjacent, non-overlapping chunks."""
    document = _document([f"word-{index}" for index in range(10)])

    chunks = chunk_document(document, chunk_size=4, chunk_overlap=0)

    assert [(chunk.start_word, chunk.end_word) for chunk in chunks] == [
        (0, 4),
        (4, 8),
        (8, 10),
    ]


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (-1, 0), (3, -1), (3, 3), (3, 4)],
)
def test_invalid_chunk_configurations_are_rejected(
    chunk_size: int, chunk_overlap: int
) -> None:
    """Invalid settings cannot create a safe chunking step."""
    document = _document(["one"])

    with pytest.raises(ValueError):
        chunk_document(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_chunks_have_deterministic_ids_and_reconstruct_original_word_order() -> None:
    """Repeated chunking is stable and has no gaps outside intentional overlap."""
    words = [f"word-{index}" for index in range(12)]
    document = _document(words)

    first_chunks = chunk_document(document, chunk_size=5, chunk_overlap=2)
    second_chunks = chunk_document(document, chunk_size=5, chunk_overlap=2)

    assert [chunk.chunk_id for chunk in first_chunks] == [
        chunk.chunk_id for chunk in second_chunks
    ]

    rebuilt_words = first_chunks[0].text.split()
    for chunk in first_chunks[1:]:
        rebuilt_words.extend(chunk.text.split()[2:])

    assert rebuilt_words == words


def test_chunks_copy_document_metadata_without_mutating_it() -> None:
    """Each chunk receives an independent immutable copy of document metadata."""
    source_metadata = {"topic": "cloud"}
    document = _document(["one", "two", "three", "four"], source_metadata)

    chunks = chunk_document(document, chunk_size=2, chunk_overlap=0)
    source_metadata["topic"] = "changed"

    assert document.metadata == {"topic": "cloud"}
    assert all(chunk.metadata == {"topic": "cloud"} for chunk in chunks)
    assert all(chunk.metadata is not document.metadata for chunk in chunks)
