"""Tests for immutable document and chunk models."""

from dataclasses import FrozenInstanceError

import pytest

from app.models import Document


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
