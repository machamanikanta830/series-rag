"""Integration tests for Qdrant point upsert and nearest-neighbor search."""

from collections.abc import Iterator
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.models import Chunk
from app.vector_stores.qdrant import QdrantVectorStore

pytestmark = pytest.mark.integration


@pytest.fixture
def qdrant_client() -> Iterator[QdrantClient]:
    """Provide a real local Qdrant client and close it after each test."""
    client = QdrantClient(host="localhost", port=6333)
    try:
        client.get_collections()
    except (ResponseHandlingException, UnexpectedResponse) as error:
        client.close()
        pytest.fail(
            "Qdrant is not reachable. Start it with 'docker compose up -d qdrant'. "
            f"Original error: {error}"
        )

    yield client
    client.close()


@pytest.fixture
def collection_name(qdrant_client: QdrantClient) -> Iterator[str]:
    """Provide an isolated collection and remove it after each test."""
    name = f"seriesrag-vector-test-{uuid4().hex}"
    yield name

    if qdrant_client.collection_exists(name):
        qdrant_client.delete_collection(name)


def _chunk(
    chunk_id: str,
    text: str = "Example chunk text.",
    metadata: dict[str, str] | None = None,
) -> Chunk:
    """Create a small immutable chunk for Qdrant integration tests."""
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


def _store(collection_name: str, client: QdrantClient) -> QdrantVectorStore:
    """Create a three-dimensional store using the test's real client."""
    return QdrantVectorStore(
        collection_name=collection_name,
        embedding_dimension=3,
        client=client,
    )


def test_upsert_persists_payload_and_search_reconstructs_chunk(
    qdrant_client: QdrantClient,
    collection_name: str,
) -> None:
    """Stored point payloads contain all fields required to rebuild a Chunk."""
    store = _store(collection_name, qdrant_client)
    chunk = _chunk("chunk-1", metadata={"topic": "security"})

    store.upsert([chunk], [[1.0, 0.0, 0.0]])

    [point] = qdrant_client.query_points(
        collection_name=collection_name,
        query=[1.0, 0.0, 0.0],
        limit=1,
        with_payload=True,
    ).points
    assert point.payload is not None
    assert point.payload["chunk_id"] == "chunk-1"
    assert point.payload["metadata"] == {"topic": "security"}

    [result] = store.search([1.0, 0.0, 0.0])
    assert result.chunk == chunk
    assert result.score == pytest.approx(1.0)


def test_repeated_upsert_replaces_the_same_logical_chunk(
    qdrant_client: QdrantClient,
    collection_name: str,
) -> None:
    """The deterministic point ID makes the second upsert replace the first."""
    store = _store(collection_name, qdrant_client)
    original = _chunk("shared", text="Original text")
    replacement = _chunk("shared", text="Replacement text")
    store.upsert([original], [[0.0, 1.0, 0.0]])

    store.upsert([replacement], [[1.0, 0.0, 0.0]])

    results = store.search([1.0, 0.0, 0.0])
    assert len(results) == 1
    assert results[0].chunk == replacement


def test_upsert_rejects_mismatched_counts(
    qdrant_client: QdrantClient,
    collection_name: str,
) -> None:
    """Each chunk requires exactly one embedding."""
    store = _store(collection_name, qdrant_client)

    with pytest.raises(ValueError, match="same length"):
        store.upsert([_chunk("chunk-1")], [])


@pytest.mark.parametrize("embedding", [[], [1.0, 0.0]])
def test_upsert_rejects_empty_or_wrong_dimension_vectors(
    qdrant_client: QdrantClient,
    collection_name: str,
    embedding: list[float],
) -> None:
    """Every vector must be present and match the configured collection size."""
    store = _store(collection_name, qdrant_client)

    with pytest.raises(ValueError):
        store.upsert([_chunk("chunk-1")], [embedding])


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_upsert_rejects_non_finite_vectors(
    qdrant_client: QdrantClient,
    collection_name: str,
    invalid_value: float,
) -> None:
    """NaN and infinite vector values are rejected before Qdrant writes."""
    store = _store(collection_name, qdrant_client)

    with pytest.raises(ValueError, match="finite values"):
        store.upsert([_chunk("chunk-1")], [[invalid_value, 0.0, 0.0]])


def test_invalid_batch_does_not_partially_write_points(
    qdrant_client: QdrantClient,
    collection_name: str,
) -> None:
    """Qdrant receives no points until every embedding in the batch is valid."""
    store = _store(collection_name, qdrant_client)

    with pytest.raises(ValueError, match="must not be empty"):
        store.upsert(
            [_chunk("valid"), _chunk("invalid")],
            [[1.0, 0.0, 0.0], []],
        )

    assert store.search([1.0, 0.0, 0.0]) == []


def test_search_empty_collection_returns_no_results(
    qdrant_client: QdrantClient,
    collection_name: str,
) -> None:
    """An initialized collection without points has no nearest neighbors."""
    assert _store(collection_name, qdrant_client).search([1.0, 0.0, 0.0]) == []


def test_search_ranks_results_by_descending_cosine_score(
    qdrant_client: QdrantClient,
    collection_name: str,
) -> None:
    """For cosine collections, Qdrant returns higher scores for closer vectors."""
    store = _store(collection_name, qdrant_client)
    store.upsert(
        [_chunk("exact"), _chunk("related"), _chunk("orthogonal")],
        [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.0, 1.0, 0.0]],
    )

    results = store.search([1.0, 0.0, 0.0])

    assert [result.chunk.chunk_id for result in results] == [
        "exact",
        "related",
        "orthogonal",
    ]
    assert [result.score for result in results] == sorted(
        (result.score for result in results), reverse=True
    )
    assert results[0].score == pytest.approx(1.0)


def test_search_respects_top_k_and_larger_limits(
    qdrant_client: QdrantClient,
    collection_name: str,
) -> None:
    """The nearest-neighbor limit can be smaller or larger than the collection."""
    store = _store(collection_name, qdrant_client)
    store.upsert(
        [_chunk("first"), _chunk("second")],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )

    assert len(store.search([1.0, 0.0, 0.0], top_k=1)) == 1
    assert len(store.search([1.0, 0.0, 0.0], top_k=10)) == 2


@pytest.mark.parametrize("top_k", [0, -1])
def test_search_rejects_invalid_top_k(
    qdrant_client: QdrantClient,
    collection_name: str,
    top_k: int,
) -> None:
    """Qdrant searches require at least one requested result."""
    store = _store(collection_name, qdrant_client)

    with pytest.raises(ValueError, match="greater than zero"):
        store.search([1.0, 0.0, 0.0], top_k=top_k)


@pytest.mark.parametrize("query", [[], [1.0, 0.0], [float("nan"), 0.0, 0.0]])
def test_search_rejects_invalid_query_vectors(
    qdrant_client: QdrantClient,
    collection_name: str,
    query: list[float],
) -> None:
    """Queries must be finite vectors with the configured dimension."""
    store = _store(collection_name, qdrant_client)

    with pytest.raises(ValueError):
        store.search(query)


def test_search_preserves_chunk_metadata(
    qdrant_client: QdrantClient,
    collection_name: str,
) -> None:
    """Metadata survives point payload serialization and reconstruction."""
    store = _store(collection_name, qdrant_client)
    store.upsert(
        [_chunk("metadata", metadata={"topic": "security"})],
        [[1.0, 0.0, 0.0]],
    )

    [result] = store.search([1.0, 0.0, 0.0])
    assert result.chunk.metadata == {"topic": "security"}


def test_search_rejects_a_malformed_stored_payload(
    qdrant_client: QdrantClient,
    collection_name: str,
) -> None:
    """Corrupt points raise an error instead of being silently skipped."""
    store = _store(collection_name, qdrant_client)
    store.ensure_collection()
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=uuid4(),
                vector=[1.0, 0.0, 0.0],
                payload={"chunk_id": "incomplete"},
            )
        ],
        wait=True,
    )

    with pytest.raises(ValueError, match="missing required field"):
        store.search([1.0, 0.0, 0.0])
