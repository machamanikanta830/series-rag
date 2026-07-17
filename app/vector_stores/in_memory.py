"""An explicit, brute-force in-memory vector store for learning retrieval."""

from math import isfinite

from app.embeddings.base import cosine_similarity
from app.models import Chunk, SearchResult
from app.vector_stores.base import VectorStore


class InMemoryVectorStore(VectorStore):
    """Store vectors in a dictionary and compare every vector for each search.

    A chunk ID is the logical identity: upserting an existing ID replaces its
    chunk and embedding instead of adding a duplicate search result.
    """

    def __init__(self) -> None:
        """Create an empty store with no established embedding dimension."""
        self._entries: dict[str, tuple[Chunk, list[float]]] = {}
        self._dimension: int | None = None

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Validate a complete batch, then insert or replace its entries."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        normalized_embeddings, batch_dimension = _validate_batch(embeddings)
        if batch_dimension is None:
            return
        if self._dimension is not None and batch_dimension != self._dimension:
            raise ValueError(
                "Embedding dimension does not match the existing store dimension"
            )

        entries_to_upsert = list(zip(chunks, normalized_embeddings, strict=True))
        for chunk, embedding in entries_to_upsert:
            self._entries[chunk.chunk_id] = (chunk, embedding)

        if self._dimension is None:
            self._dimension = batch_dimension

    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[SearchResult]:
        """Score every stored vector and return the best deterministic matches."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not self._entries:
            return []

        validated_query = _validate_embedding(query_embedding)
        if len(validated_query) != self._dimension:
            raise ValueError(
                "Query embedding dimension does not match the store dimension"
            )

        results = [
            SearchResult(
                chunk=chunk,
                score=cosine_similarity(validated_query, embedding),
            )
            for chunk, embedding in self._entries.values()
        ]
        ranked_results = sorted(
            results,
            key=lambda result: (-result.score, result.chunk.chunk_id),
        )
        return ranked_results[:top_k]


def _validate_batch(
    embeddings: list[list[float]],
) -> tuple[list[list[float]], int | None]:
    """Copy and validate all embeddings before an upsert can mutate the store."""
    validated_embeddings: list[list[float]] = []
    batch_dimension: int | None = None

    for embedding in embeddings:
        validated_embedding = _validate_embedding(embedding)
        embedding_dimension = len(validated_embedding)
        if batch_dimension is None:
            batch_dimension = embedding_dimension
        elif embedding_dimension != batch_dimension:
            raise ValueError("All embeddings in one batch must have the same dimension")
        validated_embeddings.append(validated_embedding)

    return validated_embeddings, batch_dimension


def _validate_embedding(embedding: list[float]) -> list[float]:
    """Return a copied finite vector with at least one value."""
    if not embedding:
        raise ValueError("Embedding vectors must not be empty")
    if not all(isfinite(value) for value in embedding):
        raise ValueError("Embedding vectors must contain only finite values")
    return list(embedding)
