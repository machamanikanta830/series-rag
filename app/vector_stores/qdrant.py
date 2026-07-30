"""Qdrant-backed vector storage with explicit point payloads."""

from collections.abc import Mapping
from math import isfinite
from uuid import NAMESPACE_URL, UUID, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.models import Chunk, SearchResult
from app.vector_stores.base import VectorStore

_POINT_ID_NAMESPACE = NAMESPACE_URL


class QdrantVectorStore(VectorStore):
    """Connect to Qdrant and ensure one compatible cosine-distance collection.

    Chunk IDs are mapped deterministically to UUIDv5 point IDs because Qdrant
    point IDs must be UUIDs or integers. The original SHA-256 chunk ID remains
    in each payload and is restored when search results are reconstructed.
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

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Validate a full batch, then persist its chunks and vectors as points."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        validated_embeddings = [
            _validate_embedding(embedding, self._embedding_dimension)
            for embedding in embeddings
        ]
        points = [
            models.PointStruct(
                id=_point_id_for_chunk(chunk.chunk_id),
                vector=embedding,
                payload=_chunk_to_payload(chunk),
            )
            for chunk, embedding in zip(chunks, validated_embeddings, strict=True)
        ]

        self.ensure_collection()
        if points:
            self._client.upsert(
                collection_name=self._collection_name,
                points=points,
                wait=True,
            )

    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[SearchResult]:
        """Query Qdrant and rebuild ranked immutable chunks from point payloads."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        validated_query = _validate_embedding(
            query_embedding,
            self._embedding_dimension,
        )
        self.ensure_collection()
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=validated_query,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        results = [
            SearchResult(
                chunk=_chunk_from_payload(point.payload),
                score=point.score,
            )
            for point in response.points
        ]
        return sorted(results, key=lambda result: result.score, reverse=True)

    def delete_document(self, document_id: str) -> None:
        """Delete every point whose payload belongs to the document."""
        self.ensure_collection()
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
            wait=True,
        )


def _validate_embedding(embedding: list[float], expected_dimension: int) -> list[float]:
    """Copy one finite vector after checking its exact collection dimension."""
    if not embedding:
        raise ValueError("Embedding vectors must not be empty")
    if len(embedding) != expected_dimension:
        raise ValueError(
            "Embedding dimension does not match the configured collection dimension"
        )
    if not all(isfinite(value) for value in embedding):
        raise ValueError("Embedding vectors must contain only finite values")
    return list(embedding)


def _point_id_for_chunk(chunk_id: str) -> UUID:
    """Map a stable chunk ID to a Qdrant-compatible stable UUID point ID."""
    return uuid5(_POINT_ID_NAMESPACE, f"series-rag:chunk:{chunk_id}")


def _chunk_to_payload(chunk: Chunk) -> dict[str, object]:
    """Serialize every Chunk field explicitly into a JSON-compatible payload."""
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "source_name": chunk.source_name,
        "text": chunk.text,
        "chunk_index": chunk.chunk_index,
        "start_word": chunk.start_word,
        "end_word": chunk.end_word,
        "metadata": dict(chunk.metadata),
    }


def _chunk_from_payload(payload: Mapping[str, object] | None) -> Chunk:
    """Reconstruct a Chunk or raise a clear error for corrupt point payloads."""
    if payload is None:
        raise ValueError("Qdrant search result is missing its chunk payload")

    return Chunk(
        chunk_id=_required_string(payload, "chunk_id"),
        document_id=_required_string(payload, "document_id"),
        source_name=_required_string(payload, "source_name"),
        text=_required_string(payload, "text"),
        chunk_index=_required_integer(payload, "chunk_index"),
        start_word=_required_integer(payload, "start_word"),
        end_word=_required_integer(payload, "end_word"),
        metadata=_required_metadata(payload),
    )


def _required_string(payload: Mapping[str, object], field_name: str) -> str:
    """Read a required string field from a Qdrant point payload."""
    value = _required_value(payload, field_name)
    if not isinstance(value, str):
        raise ValueError(f"Qdrant chunk payload field {field_name!r} must be a string")
    return value


def _required_integer(payload: Mapping[str, object], field_name: str) -> int:
    """Read a required integer field from a Qdrant point payload."""
    value = _required_value(payload, field_name)
    if type(value) is not int:
        raise ValueError(
            f"Qdrant chunk payload field {field_name!r} must be an integer"
        )
    return value


def _required_metadata(payload: Mapping[str, object]) -> dict[str, str]:
    """Read metadata while preserving the Chunk model's string mapping contract."""
    value = _required_value(payload, "metadata")
    if not isinstance(value, dict):
        raise ValueError("Qdrant chunk payload field 'metadata' must be an object")

    metadata: dict[str, str] = {}
    for key, metadata_value in value.items():
        if not isinstance(key, str) or not isinstance(metadata_value, str):
            raise ValueError(
                "Qdrant chunk payload metadata must map strings to strings"
            )
        metadata[key] = metadata_value
    return metadata


def _required_value(payload: Mapping[str, object], field_name: str) -> object:
    """Read one required field or identify the malformed stored payload."""
    if field_name not in payload:
        raise ValueError(
            f"Qdrant chunk payload is missing required field {field_name!r}"
        )
    return payload[field_name]
