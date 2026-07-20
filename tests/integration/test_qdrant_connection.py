"""Integration tests for local Qdrant connection and collection lifecycle setup."""

from collections.abc import Iterator
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.vector_stores.qdrant import QdrantVectorStore

pytestmark = pytest.mark.integration


@pytest.fixture
def qdrant_client() -> Iterator[QdrantClient]:
    """Provide a real local client and close its connections after each test."""
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
    """Provide a unique collection name and remove it after the test."""
    name = f"seriesrag-test-{uuid4().hex}"
    yield name

    if qdrant_client.collection_exists(name):
        qdrant_client.delete_collection(name)


def test_qdrant_connection_is_ready(
    qdrant_client: QdrantClient,
    collection_name: str,
) -> None:
    """The adapter can inspect the local Qdrant service."""
    store = QdrantVectorStore(
        collection_name=collection_name,
        embedding_dimension=3,
        client=qdrant_client,
    )

    assert store.is_ready()


def test_initialization_creates_a_missing_collection(
    qdrant_client: QdrantClient,
    collection_name: str,
) -> None:
    """A missing collection is created with cosine distance and the requested size."""
    store = QdrantVectorStore(
        collection_name=collection_name,
        embedding_dimension=3,
        client=qdrant_client,
    )

    store.ensure_collection()

    collection = qdrant_client.get_collection(collection_name)
    vectors_config = collection.config.params.vectors
    assert isinstance(vectors_config, models.VectorParams)
    assert vectors_config.size == 3
    assert vectors_config.distance == models.Distance.COSINE


def test_repeated_initialization_leaves_a_compatible_collection_unchanged(
    qdrant_client: QdrantClient,
    collection_name: str,
) -> None:
    """Initialization is idempotent for the same collection configuration."""
    store = QdrantVectorStore(
        collection_name=collection_name,
        embedding_dimension=3,
        client=qdrant_client,
    )

    store.ensure_collection()
    store.ensure_collection()

    assert qdrant_client.collection_exists(collection_name)


@pytest.mark.parametrize("collection_name", ["", "   "])
def test_invalid_collection_name_is_rejected(collection_name: str) -> None:
    """Collection names must contain visible text before creating a client."""
    with pytest.raises(ValueError, match="must not be empty"):
        QdrantVectorStore(collection_name=collection_name, embedding_dimension=3)


@pytest.mark.parametrize("embedding_dimension", [0, -1])
def test_invalid_embedding_dimension_is_rejected(embedding_dimension: int) -> None:
    """Embedding dimensions must be positive before creating a client."""
    with pytest.raises(ValueError, match="greater than zero"):
        QdrantVectorStore(
            collection_name="valid-collection",
            embedding_dimension=embedding_dimension,
        )


def test_incompatible_existing_dimension_is_rejected(
    qdrant_client: QdrantClient,
    collection_name: str,
) -> None:
    """An existing collection cannot silently use another embedding dimension."""
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
    )
    store = QdrantVectorStore(
        collection_name=collection_name,
        embedding_dimension=3,
        client=qdrant_client,
    )

    with pytest.raises(ValueError, match="dimension does not match"):
        store.ensure_collection()


def test_incompatible_existing_distance_is_rejected(
    qdrant_client: QdrantClient,
    collection_name: str,
) -> None:
    """An existing collection must use the cosine metric chosen for embeddings."""
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=3, distance=models.Distance.EUCLID),
    )
    store = QdrantVectorStore(
        collection_name=collection_name,
        embedding_dimension=3,
        client=qdrant_client,
    )

    with pytest.raises(ValueError, match="distance must be cosine"):
        store.ensure_collection()
