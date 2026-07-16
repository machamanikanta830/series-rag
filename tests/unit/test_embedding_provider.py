"""Tests for the lazy Sentence Transformers embedding provider."""

from unittest.mock import Mock, patch

import pytest

from app.embeddings.sentence_transformers import (
    DEFAULT_MODEL_NAME,
    SentenceTransformerEmbeddingProvider,
)


def test_provider_loads_once_and_returns_normalized_document_embeddings() -> None:
    """One cached model produces one normalized vector per input text."""
    model = Mock()
    model.get_embedding_dimension.return_value = 2
    model.encode.return_value = [[3.0, 4.0], [6.0, 8.0]]

    with patch(
        "app.embeddings.sentence_transformers.SentenceTransformer",
        return_value=model,
    ) as model_constructor:
        provider = SentenceTransformerEmbeddingProvider()

        assert model_constructor.call_count == 0
        document_embeddings = provider.embed_documents(["first", "second"])
        query_embedding = provider.embed_query("query")

    assert model_constructor.call_count == 1
    model_constructor.assert_called_once_with(DEFAULT_MODEL_NAME)
    assert document_embeddings == [[0.6, 0.8], [0.6, 0.8]]
    assert query_embedding == [0.6, 0.8]
    assert len(document_embeddings) == 2
    assert all(
        len(embedding) == provider.embedding_dimension()
        for embedding in document_embeddings
    )


def test_provider_rejects_empty_document_list_without_loading_a_model() -> None:
    """Input validation happens before any model work."""
    provider = SentenceTransformerEmbeddingProvider()

    with pytest.raises(ValueError, match="Document texts must not be empty"):
        provider.embed_documents([])


@pytest.mark.parametrize("query", ["", "   "])
def test_provider_rejects_empty_query(query: str) -> None:
    """Queries must contain visible text."""
    provider = SentenceTransformerEmbeddingProvider()

    with pytest.raises(ValueError, match="Query text must not be empty"):
        provider.embed_query(query)


def test_provider_wraps_model_loading_errors_with_context() -> None:
    """A failed lazy model load explains how to diagnose it."""
    with patch(
        "app.embeddings.sentence_transformers.SentenceTransformer",
        side_effect=OSError("unavailable"),
    ):
        provider = SentenceTransformerEmbeddingProvider()

        with pytest.raises(RuntimeError, match="Unable to load Sentence Transformers"):
            provider.embedding_dimension()
