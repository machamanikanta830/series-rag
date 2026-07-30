"""Tests for read-only document catalog endpoints."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_document_catalog,
    get_document_deletion_service,
    reset_development_application_state,
)
from app.api.main import app
from app.document_catalog import (
    CatalogDocument,
    DocumentCatalog,
    InMemoryDocumentCatalog,
)
from app.models import Chunk, Document
from app.services.deletion import DocumentDeletionService

client = TestClient(app)


class FailingDocumentCatalog(DocumentCatalog):
    """Raise one configured failure from catalog reads."""

    def __init__(self, error: RuntimeError) -> None:
        self.error = error

    def record(self, document: Document, chunks: list[Chunk]) -> None:
        """Reject writes because this test double supports reads only."""
        raise AssertionError("record should not be called")

    def list_documents(self) -> tuple[CatalogDocument, ...]:
        """Raise the configured failure while listing."""
        raise self.error

    def get_document(self, document_id: str) -> CatalogDocument | None:
        """Raise the configured failure while looking up a document."""
        raise self.error

    def delete_document(self, document_id: str) -> CatalogDocument | None:
        """Raise the configured failure while deleting a document."""
        raise self.error


class StubDocumentDeletionService(DocumentDeletionService):
    """Return or raise a configured deletion outcome while recording IDs."""

    def __init__(
        self,
        result: bool | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[str] = []

    def delete(self, document_id: str) -> bool:
        """Record the requested identity before returning or raising."""
        self.calls.append(document_id)
        if self.error is not None:
            raise self.error
        if self.result is None:
            raise AssertionError("StubDocumentDeletionService needs a result or error")
        return self.result


@pytest.fixture(autouse=True)
def isolate_application_state() -> Iterator[None]:
    """Reset catalog state and dependency overrides around every test."""
    reset_development_application_state()
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()
    reset_development_application_state()


def _document(document_id: str, source_name: str) -> Document:
    """Create deterministic document metadata for catalog response tests."""
    return Document(
        document_id=document_id,
        source_name=source_name,
        source_path=source_name,
        text=f"Full text for {source_name}.",
        metadata={"filename": source_name},
    )


def _chunk(
    document: Document,
    *,
    chunk_id: str,
    chunk_index: int,
    text: str,
) -> Chunk:
    """Create one immutable source chunk."""
    return Chunk(
        chunk_id=chunk_id,
        document_id=document.document_id,
        source_name=document.source_name,
        text=text,
        chunk_index=chunk_index,
        start_word=chunk_index * 3,
        end_word=(chunk_index + 1) * 3,
        metadata=document.metadata,
    )


def _override_catalog(catalog: DocumentCatalog) -> None:
    """Install one test-owned catalog dependency."""
    app.dependency_overrides[get_document_catalog] = lambda: catalog


def _override_deletion_service(service: StubDocumentDeletionService) -> None:
    """Install one test-owned deletion dependency."""
    app.dependency_overrides[get_document_deletion_service] = lambda: service


def test_empty_catalog_returns_an_empty_list() -> None:
    """A process with no uploads exposes an empty document collection."""
    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == []


def test_multiple_documents_preserve_upload_order() -> None:
    """The default shared catalog lists documents in ingestion order."""
    first_upload = client.post(
        "/documents",
        files={"file": ("first.md", b"First public lesson.", "text/markdown")},
    )
    second_upload = client.post(
        "/documents",
        files={"file": ("second.txt", b"Second public lesson.", "text/plain")},
    )

    response = client.get("/documents")

    assert first_upload.status_code == 201
    assert second_upload.status_code == 201
    assert response.status_code == 200
    assert response.json() == [
        {
            "document_id": first_upload.json()["document_id"],
            "filename": "first.md",
            "chunk_count": 1,
        },
        {
            "document_id": second_upload.json()["document_id"],
            "filename": "second.txt",
            "chunk_count": 1,
        },
    ]


def test_document_lookup_serializes_chunks_without_embeddings_or_scores() -> None:
    """Detail responses expose source text fields but no vector internals."""
    upload_response = client.post(
        "/documents",
        files={
            "file": (
                "lesson.md",
                b"AWS secures physical cloud infrastructure.",
                "text/markdown",
            )
        },
    )
    document_id = upload_response.json()["document_id"]

    response = client.get(f"/documents/{document_id}")

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "document_id": document_id,
        "filename": "lesson.md",
        "chunk_count": 1,
        "chunks": [
            {
                "chunk_id": body["chunks"][0]["chunk_id"],
                "chunk_index": 0,
                "text": "AWS secures physical cloud infrastructure.",
            }
        ],
    }
    assert "embedding" not in response.text
    assert "score" not in response.text


def test_document_chunks_are_returned_in_chunk_index_order() -> None:
    """The catalog normalizes chunk presentation to source order."""
    catalog = InMemoryDocumentCatalog()
    document = _document("document-1", "ordered.md")
    catalog.record(
        document,
        [
            _chunk(document, chunk_id="chunk-2", chunk_index=2, text="Third."),
            _chunk(document, chunk_id="chunk-0", chunk_index=0, text="First."),
            _chunk(document, chunk_id="chunk-1", chunk_index=1, text="Second."),
        ],
    )
    _override_catalog(catalog)

    response = client.get("/documents/document-1")

    assert response.status_code == 200
    assert [chunk["chunk_index"] for chunk in response.json()["chunks"]] == [0, 1, 2]
    assert [chunk["text"] for chunk in response.json()["chunks"]] == [
        "First.",
        "Second.",
        "Third.",
    ]


def test_unknown_document_returns_not_found() -> None:
    """A missing catalog identity maps to a stable HTTP 404 response."""
    response = client.get("/documents/unknown")

    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found."}


@pytest.mark.parametrize("path", ["/documents", "/documents/document-1"])
def test_unexpected_catalog_failures_return_stable_server_error(path: str) -> None:
    """Internal catalog errors do not leak their details through the API."""
    _override_catalog(FailingDocumentCatalog(RuntimeError("private catalog failure")))

    response = client.get(path)

    assert response.status_code == 500
    assert response.json() == {"detail": "The document catalog could not be read."}
    assert "private catalog failure" not in response.text


def test_dependency_override_supplies_an_isolated_catalog() -> None:
    """Catalog reads use the override without changing default application state."""
    catalog = InMemoryDocumentCatalog()
    document = _document("override-document", "override.md")
    catalog.record(document, [])
    _override_catalog(catalog)

    overridden_response = client.get("/documents")
    app.dependency_overrides.clear()
    default_response = client.get("/documents")

    assert overridden_response.json() == [
        {
            "document_id": "override-document",
            "filename": "override.md",
            "chunk_count": 0,
        }
    ]
    assert default_response.json() == []


def test_upload_and_catalog_reads_share_default_application_state() -> None:
    """An uploaded document is immediately visible through both catalog routes."""
    upload_response = client.post(
        "/documents",
        files={"file": ("shared.txt", b"Shared application state.", "text/plain")},
    )
    document_id = upload_response.json()["document_id"]

    list_response = client.get("/documents")
    detail_response = client.get(f"/documents/{document_id}")

    assert list_response.json()[0]["document_id"] == document_id
    assert detail_response.json()["document_id"] == document_id
    assert detail_response.json()["chunks"][0]["text"] == "Shared application state."


def test_delete_returns_no_content_and_removes_catalog_and_vectors() -> None:
    """Successful deletion removes all observable forms of an uploaded document."""
    upload_response = client.post(
        "/documents",
        files={
            "file": (
                "nutrition.md",
                b"Bananas contain potassium and support nutrition.",
                "text/markdown",
            )
        },
    )
    document_id = upload_response.json()["document_id"]
    before_delete = client.post(
        "/query",
        json={"question": "What do bananas contain for nutrition?", "top_k": 1},
    )

    delete_response = client.delete(f"/documents/{document_id}")
    list_response = client.get("/documents")
    detail_response = client.get(f"/documents/{document_id}")
    after_delete = client.post(
        "/query",
        json={"question": "What do bananas contain for nutrition?", "top_k": 5},
    )

    assert before_delete.json()["sources"][0]["document_id"] == document_id
    assert delete_response.status_code == 204
    assert delete_response.content == b""
    assert list_response.json() == []
    assert detail_response.status_code == 404
    assert all(
        source["document_id"] != document_id
        for source in after_delete.json()["sources"]
    )


def test_delete_one_of_multiple_documents_leaves_the_other_document() -> None:
    """Deletion targets one document identity without changing unrelated uploads."""
    first_upload = client.post(
        "/documents",
        files={"file": ("first.md", b"First public lesson.", "text/markdown")},
    )
    second_upload = client.post(
        "/documents",
        files={"file": ("second.md", b"Second public lesson.", "text/markdown")},
    )

    response = client.delete(f"/documents/{first_upload.json()['document_id']}")

    assert response.status_code == 204
    assert client.get("/documents").json() == [
        {
            "document_id": second_upload.json()["document_id"],
            "filename": "second.md",
            "chunk_count": 1,
        }
    ]


def test_deleting_an_unknown_or_already_deleted_document_returns_not_found() -> None:
    """Storage deletion is idempotent while repeated API deletion remains HTTP 404."""
    upload_response = client.post(
        "/documents",
        files={"file": ("lesson.md", b"Public lesson.", "text/markdown")},
    )
    document_id = upload_response.json()["document_id"]

    first_response = client.delete(f"/documents/{document_id}")
    repeated_response = client.delete(f"/documents/{document_id}")
    unknown_response = client.delete("/documents/unknown")

    assert first_response.status_code == 204
    assert repeated_response.status_code == 404
    assert repeated_response.json() == {"detail": "Document not found."}
    assert unknown_response.status_code == 404


def test_delete_uses_the_overrideable_service_dependency() -> None:
    """The API delegates one document ID to a test-owned deletion service."""
    service = StubDocumentDeletionService(result=True)
    _override_deletion_service(service)

    response = client.delete("/documents/document-1")

    assert response.status_code == 204
    assert service.calls == ["document-1"]


def test_delete_failure_returns_stable_server_error() -> None:
    """Unexpected deletion failures do not expose internal storage details."""
    service = StubDocumentDeletionService(
        error=RuntimeError("private vector-store failure")
    )
    _override_deletion_service(service)

    response = client.delete("/documents/document-1")

    assert response.status_code == 500
    assert response.json() == {"detail": "The document could not be deleted."}
    assert "private vector-store failure" not in response.text
