"""Tests for deterministic document and chunk identifiers."""

from app.identifiers import create_chunk_id, create_document_id


def test_same_document_inputs_produce_the_same_id() -> None:
    """Document IDs are deterministic for identical canonical inputs."""
    first_id = create_document_id("notes/lesson.txt", "Normalized text.")
    second_id = create_document_id("notes/lesson.txt", "Normalized text.")

    assert first_id == second_id


def test_different_source_paths_produce_different_document_ids() -> None:
    """The canonical source path participates in the document ID."""
    first_id = create_document_id("notes/lesson.txt", "Normalized text.")
    second_id = create_document_id("other/lesson.txt", "Normalized text.")

    assert first_id != second_id


def test_different_normalized_text_produces_different_document_ids() -> None:
    """The normalized document content participates in the document ID."""
    first_id = create_document_id("notes/lesson.txt", "First text.")
    second_id = create_document_id("notes/lesson.txt", "Second text.")

    assert first_id != second_id


def test_chunk_index_and_text_participate_in_chunk_ids() -> None:
    """Both the index and chunk text differentiate chunk IDs."""
    document_id = "document-1"
    base_id = create_chunk_id(document_id, 0, "Chunk text.")

    assert base_id != create_chunk_id(document_id, 1, "Chunk text.")
    assert base_id != create_chunk_id(document_id, 0, "Different chunk text.")
