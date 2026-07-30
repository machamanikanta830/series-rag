"""A small interface for storing and searching chunk embeddings."""

from abc import ABC, abstractmethod

from app.models import Chunk, SearchResult


class VectorStore(ABC):
    """Store chunk vectors and return ranked similarity results."""

    @abstractmethod
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert or replace embeddings using each chunk ID as its identity."""

    @abstractmethod
    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[SearchResult]:
        """Return up to ``top_k`` chunks ordered by descending similarity."""

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Idempotently remove every chunk belonging to one document."""
