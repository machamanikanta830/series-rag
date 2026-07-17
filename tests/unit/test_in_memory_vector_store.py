"""Tests for explicit in-memory vector storage and brute-force search."""

from collections.abc import Mapping

import pytest

from app.models import Chunk
from app.vector_stores.in_memory import InMemoryVectorStore


def _chunk(
    chunk_id: str,
    text: str = "Example chunk text.",
    metadata: Mapping[str, str] | None = None,
) -> Chunk:
    """Create a minimal immutable chunk for vector-store tests."""
    return Chunk(
        chunk_id=chunk_id,
        document_id="document-1",
        source_name="source.txt",
        text=text,
        chunk_index=0,
        start_word=0,
        end_word=3,
        metadata={} if metadata is None else metadata,
    )


def test_upsert_inserts_chunks_and_embeddings() -> None:
    """A successful upsert makes a chunk available to search."""
    store = InMemoryVectorStore()
    chunk = _chunk("chunk-1")

    store.upsert([chunk], [[1.0, 0.0]])

    [result] = store.search([1.0, 0.0])
    assert result.chunk == chunk
    assert result.score == pytest.approx(1.0)


def test_upsert_rejects_mismatched_chunk_and_embedding_counts() -> None:
    """Each chunk needs exactly one corresponding embedding."""
    store = InMemoryVectorStore()

    with pytest.raises(ValueError, match="same length"):
        store.upsert([_chunk("chunk-1")], [])


def test_upsert_rejects_empty_embeddings() -> None:
    """A vector must contain at least one numeric dimension."""
    store = InMemoryVectorStore()

    with pytest.raises(ValueError, match="must not be empty"):
        store.upsert([_chunk("chunk-1")], [[]])


def test_upsert_rejects_inconsistent_dimensions_within_one_batch() -> None:
    """All vectors in a batch must describe the same embedding space."""
    store = InMemoryVectorStore()

    with pytest.raises(ValueError, match="same dimension"):
        store.upsert([_chunk("first"), _chunk("second")], [[1.0, 0.0], [1.0]])


def test_upsert_rejects_dimension_mismatch_with_existing_store() -> None:
    """The first valid batch establishes a store-wide vector dimension."""
    store = InMemoryVectorStore()
    store.upsert([_chunk("first")], [[1.0, 0.0]])

    with pytest.raises(ValueError, match="existing store dimension"):
        store.upsert([_chunk("second")], [[1.0, 0.0, 0.0]])


def test_upsert_replaces_a_duplicate_chunk_id() -> None:
    """Upserting the same chunk ID replaces its vector and chunk payload."""
    store = InMemoryVectorStore()
    original = _chunk("shared", text="Original text")
    replacement = _chunk("shared", text="Replacement text")
    store.upsert([original], [[0.0, 1.0]])

    store.upsert([replacement], [[1.0, 0.0]])

    results = store.search([1.0, 0.0])
    assert len(results) == 1
    assert results[0].chunk == replacement


def test_invalid_batch_does_not_partially_mutate_existing_data() -> None:
    """All batch validation completes before any entry is inserted or replaced."""
    store = InMemoryVectorStore()
    original = _chunk("original")
    store.upsert([original], [[1.0, 0.0]])

    with pytest.raises(ValueError, match="must not be empty"):
        store.upsert([_chunk("new"), _chunk("invalid")], [[0.0, 1.0], []])

    results = store.search([1.0, 0.0])
    assert [result.chunk.chunk_id for result in results] == ["original"]


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_upsert_rejects_non_finite_values(invalid_value: float) -> None:
    """NaN and infinite values do not produce meaningful similarity scores."""
    store = InMemoryVectorStore()

    with pytest.raises(ValueError, match="finite values"):
        store.upsert([_chunk("chunk-1")], [[invalid_value, 0.0]])


def test_search_returns_empty_results_for_an_empty_store() -> None:
    """An empty store has no matches to rank."""
    assert InMemoryVectorStore().search([1.0]) == []


def test_search_ranks_exact_match_first_and_sorts_descending() -> None:
    """Brute-force search orders all chunks from highest score to lowest score."""
    store = InMemoryVectorStore()
    store.upsert(
        [_chunk("exact"), _chunk("related"), _chunk("orthogonal")],
        [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]],
    )

    results = store.search([1.0, 0.0])

    assert [result.chunk.chunk_id for result in results] == [
        "exact",
        "related",
        "orthogonal",
    ]
    assert [result.score for result in results] == sorted(
        (result.score for result in results), reverse=True
    )


def test_search_respects_top_k_and_handles_a_larger_limit() -> None:
    """The result limit can be smaller or larger than the number of entries."""
    store = InMemoryVectorStore()
    store.upsert(
        [_chunk("first"), _chunk("second")],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    assert len(store.search([1.0, 0.0], top_k=1)) == 1
    assert len(store.search([1.0, 0.0], top_k=10)) == 2


@pytest.mark.parametrize("top_k", [0, -1])
def test_search_rejects_an_invalid_result_limit(top_k: int) -> None:
    """A search must request at least one result."""
    store = InMemoryVectorStore()

    with pytest.raises(ValueError, match="greater than zero"):
        store.search([1.0], top_k=top_k)


def test_search_rejects_dimension_mismatch_and_empty_query_embedding() -> None:
    """A query must share the store dimension and contain values."""
    store = InMemoryVectorStore()
    store.upsert([_chunk("first")], [[1.0, 0.0]])

    with pytest.raises(ValueError, match="store dimension"):
        store.search([1.0])
    with pytest.raises(ValueError, match="must not be empty"):
        store.search([])


def test_search_uses_chunk_id_for_deterministic_tie_ordering() -> None:
    """Equal scores sort by chunk ID instead of insertion order."""
    store = InMemoryVectorStore()
    store.upsert(
        [_chunk("zulu"), _chunk("alpha")],
        [[1.0, 0.0], [1.0, 0.0]],
    )

    results = store.search([1.0, 0.0])

    assert [result.chunk.chunk_id for result in results] == ["alpha", "zulu"]


def test_search_results_preserve_chunk_metadata() -> None:
    """SearchResult returns the immutable Chunk rather than copied fields."""
    store = InMemoryVectorStore()
    chunk = _chunk("metadata", metadata={"topic": "security"})
    store.upsert([chunk], [[1.0, 0.0]])

    [result] = store.search([1.0, 0.0])
    assert result.chunk is chunk
    assert result.chunk.metadata == {"topic": "security"}
