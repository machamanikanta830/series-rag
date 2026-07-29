"""Read-only document metadata storage with an explicit ingestion write boundary."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.models import Chunk, Document


@dataclass(frozen=True, slots=True)
class CatalogDocument:
    """One immutable catalog entry and its ordered source chunks."""

    document: Document
    chunks: tuple[Chunk, ...]


class DocumentCatalog(ABC):
    """Store document metadata separately from vector-search implementation details."""

    @abstractmethod
    def record(self, document: Document, chunks: list[Chunk]) -> None:
        """Insert or replace one document after successful ingestion."""

    @abstractmethod
    def list_documents(self) -> tuple[CatalogDocument, ...]:
        """Return catalog entries in their original upload order."""

    @abstractmethod
    def get_document(self, document_id: str) -> CatalogDocument | None:
        """Return one catalog entry, or ``None`` when it does not exist."""


class InMemoryDocumentCatalog(DocumentCatalog):
    """Keep immutable document entries in insertion order for local development."""

    def __init__(self) -> None:
        """Create an empty in-process catalog."""
        self._documents: dict[str, CatalogDocument] = {}

    def record(self, document: Document, chunks: list[Chunk]) -> None:
        """Copy chunks into index order and insert or replace the document."""
        ordered_chunks = tuple(
            sorted(chunks, key=lambda chunk: (chunk.chunk_index, chunk.chunk_id))
        )
        self._documents[document.document_id] = CatalogDocument(
            document=document,
            chunks=ordered_chunks,
        )

    def list_documents(self) -> tuple[CatalogDocument, ...]:
        """Return immutable entries in dictionary insertion order."""
        return tuple(self._documents.values())

    def get_document(self, document_id: str) -> CatalogDocument | None:
        """Look up one document without exposing mutable internal state."""
        return self._documents.get(document_id)
