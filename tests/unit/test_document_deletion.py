"""Tests for document deletion across catalog and vector storage."""

import pytest

from app.document_catalog import InMemoryDocumentCatalog
from app.models import Chunk, Document
from app.services.deletion import DocumentDeletionService
from app.vector_stores.in_memory import InMemoryVectorStore


class FailingDeleteVectorStore(InMemoryVectorStore):
    """Raise one configured failure without changing stored vectors."""

    def delete_document(self, document_id: str) -> None:
        """Fail before deleting any matching entries."""
        raise RuntimeError("vector deletion failed")


class FailingDeleteCatalog(InMemoryDocumentCatalog):
    """Raise one configured failure without changing catalog entries."""

    def delete_document(self, document_id: str) -> None:
        """Fail before deleting the matching catalog entry."""
        raise RuntimeError("catalog deletion failed")


def _document(document_id: str, source_name: str) -> Document:
    """Create one immutable document for deletion tests."""
    return Document(
        document_id=document_id,
        source_name=source_name,
        source_path=source_name,
        text=f"Text for {source_name}.",
    )


def _chunk(
    document: Document,
    chunk_id: str,
    chunk_index: int,
) -> Chunk:
    """Create one chunk belonging to the supplied document."""
    return Chunk(
        chunk_id=chunk_id,
        document_id=document.document_id,
        source_name=document.source_name,
        text=f"Text for {chunk_id}.",
        chunk_index=chunk_index,
        start_word=chunk_index * 3,
        end_word=(chunk_index + 1) * 3,
    )


def _stored_service() -> tuple[
    DocumentDeletionService,
    InMemoryDocumentCatalog,
    InMemoryVectorStore,
]:
    """Create two cataloged documents with three stored vectors."""
    first_document = _document("document-1", "first.md")
    second_document = _document("document-2", "second.md")
    first_chunks = [
        _chunk(first_document, "first-0", 0),
        _chunk(first_document, "first-1", 1),
    ]
    second_chunks = [_chunk(second_document, "second-0", 0)]

    catalog = InMemoryDocumentCatalog()
    catalog.record(first_document, first_chunks)
    catalog.record(second_document, second_chunks)
    vector_store = InMemoryVectorStore()
    vector_store.upsert(
        [*first_chunks, *second_chunks],
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
    )
    return DocumentDeletionService(catalog, vector_store), catalog, vector_store


def test_delete_removes_catalog_entry_and_all_document_vectors() -> None:
    """One service call removes every representation of the target document."""
    service, catalog, vector_store = _stored_service()

    deleted = service.delete("document-1")

    assert deleted is True
    assert catalog.get_document("document-1") is None
    assert [entry.document.document_id for entry in catalog.list_documents()] == [
        "document-2"
    ]
    results = vector_store.search([1.0, 0.0], top_k=10)
    assert [result.chunk.chunk_id for result in results] == ["second-0"]


def test_delete_leaves_unrelated_documents_untouched() -> None:
    """Deleting one identity preserves the other document and its vector."""
    service, catalog, vector_store = _stored_service()

    service.delete("document-1")

    remaining = catalog.get_document("document-2")
    assert remaining is not None
    assert [chunk.chunk_id for chunk in remaining.chunks] == ["second-0"]
    [result] = vector_store.search([0.0, 1.0], top_k=10)
    assert result.chunk.document_id == "document-2"


def test_storage_deletion_is_idempotent_for_an_unknown_document() -> None:
    """An absent catalog identity reports false and leaves both stores unchanged."""
    service, catalog, vector_store = _stored_service()

    assert service.delete("unknown") is False
    assert service.delete("unknown") is False
    assert len(catalog.list_documents()) == 2
    assert len(vector_store.search([1.0, 0.0], top_k=10)) == 3


def test_vector_failure_restores_the_catalog_and_propagates() -> None:
    """Catalog-first deletion is compensated when vector deletion fails."""
    document = _document("document-1", "lesson.md")
    chunk = _chunk(document, "chunk-1", 0)
    catalog = InMemoryDocumentCatalog()
    catalog.record(document, [chunk])
    vector_store = FailingDeleteVectorStore()
    vector_store.upsert([chunk], [[1.0, 0.0]])
    service = DocumentDeletionService(catalog, vector_store)

    with pytest.raises(RuntimeError, match="vector deletion failed"):
        service.delete("document-1")

    restored = catalog.get_document("document-1")
    assert restored is not None
    assert restored.chunks == (chunk,)
    assert vector_store.search([1.0, 0.0])[0].chunk == chunk


def test_catalog_failure_propagates_without_touching_vectors() -> None:
    """A catalog failure occurs before the vector-store delete boundary."""
    document = _document("document-1", "lesson.md")
    chunk = _chunk(document, "chunk-1", 0)
    catalog = FailingDeleteCatalog()
    catalog.record(document, [chunk])
    vector_store = InMemoryVectorStore()
    vector_store.upsert([chunk], [[1.0, 0.0]])
    service = DocumentDeletionService(catalog, vector_store)

    with pytest.raises(RuntimeError, match="catalog deletion failed"):
        service.delete("document-1")

    assert catalog.get_document("document-1") is not None
    assert vector_store.search([1.0, 0.0])[0].chunk == chunk
