"""Coordinate document deletion across catalog and vector storage."""

from app.document_catalog import DocumentCatalog
from app.vector_stores.base import VectorStore


class DocumentDeletionService:
    """Remove one document from both storage abstractions."""

    def __init__(
        self,
        document_catalog: DocumentCatalog,
        vector_store: VectorStore,
    ) -> None:
        """Store the two explicit deletion dependencies."""
        self._document_catalog = document_catalog
        self._vector_store = vector_store

    def delete(self, document_id: str) -> bool:
        """Delete one document, returning whether it existed in the catalog.

        Catalog removal happens first so its immutable entry can be restored if
        vector deletion fails. Vector-store deletion is idempotent, so callers can
        safely retry after an error.
        """
        catalog_document = self._document_catalog.delete_document(document_id)
        if catalog_document is None:
            return False

        try:
            self._vector_store.delete_document(document_id)
        except Exception:
            self._document_catalog.record(
                catalog_document.document,
                list(catalog_document.chunks),
            )
            raise

        return True
