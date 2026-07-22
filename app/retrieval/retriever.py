"""Orchestrate query embeddings and vector-store retrieval."""

from app.embeddings.base import EmbeddingProvider
from app.models import SearchResult
from app.vector_stores.base import VectorStore


class SemanticRetriever:
    """Convert one query into an embedding and delegate ranked search."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        """Store the configured abstractions without calling either dependency."""
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return the vector store's ranked results for one validated query."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_embedding = self._embedding_provider.embed_query(normalized_query)
        return self._vector_store.search(query_embedding, top_k=top_k)
