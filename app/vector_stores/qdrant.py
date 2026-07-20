"""Qdrant connection and collection lifecycle setup for Milestone 5A."""

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse


class QdrantVectorStore:
    """Connect to Qdrant and ensure one compatible cosine-distance collection.

    This adapter intentionally manages only connection checks and collection
    lifecycle. Vector writes and searches are added in Milestone 5B.
    """

    def __init__(
        self,
        collection_name: str,
        embedding_dimension: int,
        client: QdrantClient | None = None,
        host: str = "localhost",
        port: int = 6333,
    ) -> None:
        """Store explicit connection settings without making a network call yet."""
        if not collection_name.strip():
            raise ValueError("Qdrant collection name must not be empty")
        if embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be greater than zero")

        self._collection_name = collection_name
        self._embedding_dimension = embedding_dimension
        self._client = client or QdrantClient(host=host, port=port)

    def is_ready(self) -> bool:
        """Return whether the configured Qdrant client can inspect collections."""
        try:
            self._client.get_collections()
        except (ResponseHandlingException, UnexpectedResponse):
            return False
        return True

    def ensure_collection(self) -> None:
        """Create the collection when absent or validate an existing collection."""
        if not self.is_ready():
            raise RuntimeError(
                "Qdrant is not ready. Start the local service with "
                "'docker compose up -d qdrant' and try again."
            )

        if not self._client.collection_exists(self._collection_name):
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=self._embedding_dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            return

        collection = self._client.get_collection(self._collection_name)
        vectors_config = collection.config.params.vectors
        if not isinstance(vectors_config, models.VectorParams):
            raise ValueError(
                "Existing Qdrant collection must use one unnamed vector configuration"
            )
        if vectors_config.size != self._embedding_dimension:
            raise ValueError(
                "Existing Qdrant collection dimension does not match the configured "
                "embedding dimension"
            )
        if vectors_config.distance != models.Distance.COSINE:
            raise ValueError("Existing Qdrant collection distance must be cosine")
