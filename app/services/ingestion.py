"""Orchestrate normalization, chunking, embedding, and vector storage."""

from dataclasses import dataclass

from app.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_document,
)
from app.document_catalog import DocumentCatalog
from app.embeddings.base import EmbeddingProvider
from app.models import Document, DocumentSection
from app.normalization import normalize_text
from app.vector_stores.base import VectorStore


@dataclass(frozen=True, slots=True)
class IngestionStatistics:
    """Immutable summary of one successfully completed document ingestion."""

    document_id: str
    chunks_created: int
    embedding_dimension: int
    vector_store_name: str


class IngestionService:
    """Coordinate existing ingestion components without owning their logic."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        document_catalog: DocumentCatalog | None = None,
    ) -> None:
        """Store injected dependencies and explicit chunking configuration."""
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._document_catalog = document_catalog

    def ingest(self, document: Document) -> IngestionStatistics:
        """Normalize and store one document's chunk embeddings."""
        normalized_sections = _normalize_sections(document.sections)
        normalized_document = Document(
            document_id=document.document_id,
            source_name=document.source_name,
            source_path=document.source_path,
            text=normalize_text(document.text),
            metadata=document.metadata,
            sections=normalized_sections,
        )
        chunks = chunk_document(
            normalized_document,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        embedding_dimension = self._embedding_provider.embedding_dimension()

        if chunks:
            embeddings = self._embedding_provider.embed_documents(
                [chunk.text for chunk in chunks]
            )
            self._vector_store.upsert(chunks, embeddings)

        if self._document_catalog is not None:
            self._document_catalog.record(normalized_document, chunks)

        return IngestionStatistics(
            document_id=document.document_id,
            chunks_created=len(chunks),
            embedding_dimension=embedding_dimension,
            vector_store_name=type(self._vector_store).__name__,
        )


def _normalize_sections(
    sections: tuple[DocumentSection, ...],
) -> tuple[DocumentSection, ...]:
    """Normalize nonempty sections without discarding their provenance metadata."""
    normalized_sections: list[DocumentSection] = []
    for section in sections:
        normalized_text = normalize_text(section.text)
        if normalized_text:
            normalized_sections.append(
                DocumentSection(
                    text=normalized_text,
                    metadata=section.metadata,
                )
            )
    return tuple(normalized_sections)
