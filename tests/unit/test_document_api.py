"""Tests for bounded text and Markdown document uploads."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_ingestion_service,
    reset_development_application_state,
)
from app.api.main import MAX_UPLOAD_BYTES, app
from app.models import Document
from app.services.ingestion import IngestionService, IngestionStatistics

client = TestClient(app)


class StubIngestionService(IngestionService):
    """Return or raise a configured ingestion outcome while recording documents."""

    def __init__(
        self,
        result: IngestionStatistics | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.documents: list[Document] = []

    def ingest(self, document: Document) -> IngestionStatistics:
        """Record one document before returning or raising the configured outcome."""
        self.documents.append(document)
        if self.error is not None:
            raise self.error
        if self.result is None:
            raise AssertionError("StubIngestionService needs a result or error")
        return self.result


@pytest.fixture(autouse=True)
def isolate_application_state() -> Iterator[None]:
    """Reset dependency overrides and shared development vectors per test."""
    reset_development_application_state()
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()
    reset_development_application_state()


def _statistics() -> IngestionStatistics:
    """Return deterministic statistics for upload response tests."""
    return IngestionStatistics(
        document_id="document-1",
        chunks_created=2,
        embedding_dimension=3,
        vector_store_name="TestVectorStore",
    )


def _override_ingestion_service(service: StubIngestionService) -> None:
    """Install a test-owned ingestion dependency for one test."""
    app.dependency_overrides[get_ingestion_service] = lambda: service


@pytest.mark.parametrize("filename", ["lesson.txt", "lesson.md", "lesson.markdown"])
def test_supported_text_upload_returns_created_response(filename: str) -> None:
    """Every supported extension produces a structured HTTP 201 response."""
    service = StubIngestionService(result=_statistics())
    _override_ingestion_service(service)

    response = client.post(
        "/documents",
        files={"file": (filename, b"Public lesson text.", "application/octet-stream")},
    )

    assert response.status_code == 201
    assert response.json() == {
        "document_id": "document-1",
        "filename": filename,
        "chunks_created": 2,
        "embedding_dimension": 3,
        "vector_store_name": "TestVectorStore",
    }
    [document] = service.documents
    assert document.source_name == filename
    assert document.source_path == filename
    assert document.text == "Public lesson text."
    assert document.metadata == {"filename": filename}


def test_upload_rejects_an_unsupported_extension_regardless_of_mime_type() -> None:
    """A trusted-looking MIME type cannot make a PDF a supported text source."""
    response = client.post(
        "/documents",
        files={"file": ("lesson.pdf", b"not a PDF", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json() == {
        "detail": "Only .txt, .md, and .markdown files are supported."
    }


@pytest.mark.parametrize("filename", ["", ".", ".."])
def test_upload_rejects_a_missing_or_unusable_filename(filename: str) -> None:
    """Uploads need a safe source name for identity and later citations."""
    response = client.post(
        "/documents",
        files={"file": (filename, b"Text", "text/plain")},
    )

    assert response.status_code == 422


def test_upload_rejects_content_larger_than_the_explicit_limit() -> None:
    """The endpoint detects oversize content with a limit-plus-one bounded read."""
    response = client.post(
        "/documents",
        files={"file": ("large.txt", b"x" * (MAX_UPLOAD_BYTES + 1), "text/plain")},
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "Uploaded file exceeds the 1 MB size limit."}


def test_upload_rejects_invalid_utf8() -> None:
    """Invalid bytes are not guessed or passed into document ingestion."""
    response = client.post(
        "/documents",
        files={"file": ("lesson.txt", b"\xff\xfe", "text/plain")},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Uploaded file must contain valid UTF-8 text."}


@pytest.mark.parametrize("content", [b"", b"  \n\t", b"\xef\xbb\xbf  "])
def test_upload_rejects_empty_or_whitespace_only_text(content: bytes) -> None:
    """Decoded content must remain non-empty after conservative normalization."""
    response = client.post(
        "/documents",
        files={"file": ("empty.md", content, "text/markdown")},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Uploaded file must contain non-empty text."}


def test_upload_maps_expected_ingestion_failure_to_client_error() -> None:
    """Service validation errors become stable HTTP 422 responses."""
    service = StubIngestionService(error=ValueError("private validation detail"))
    _override_ingestion_service(service)

    response = client.post(
        "/documents",
        files={"file": ("lesson.txt", b"Lesson", "text/plain")},
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Uploaded document failed ingestion validation."
    }
    assert "private validation detail" not in response.text


def test_upload_maps_unexpected_ingestion_failure_to_server_error() -> None:
    """Unexpected service failures do not expose internal provider details."""
    service = StubIngestionService(error=RuntimeError("private provider failure"))
    _override_ingestion_service(service)

    response = client.post(
        "/documents",
        files={"file": ("lesson.txt", b"Lesson", "text/plain")},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "The uploaded document could not be ingested."}
    assert "private provider failure" not in response.text


def test_dependency_override_isolation_restores_default_ingestion() -> None:
    """The fixture removes ingestion overrides left by previous tests."""
    assert get_ingestion_service not in app.dependency_overrides

    response = client.post(
        "/documents",
        files={"file": ("default.txt", b"Default text.", "text/plain")},
    )

    assert response.status_code == 201
    assert response.json()["vector_store_name"] == "InMemoryVectorStore"


def test_uploaded_content_is_queryable_through_the_shared_default_store() -> None:
    """Default ingestion and query dependencies share one in-process store."""
    upload_response = client.post(
        "/documents",
        files={
            "file": (
                "nutrition.md",
                b"Oranges contain vitamin C and support nutrition.",
                "text/markdown",
            )
        },
    )
    document_id = upload_response.json()["document_id"]

    query_response = client.post(
        "/query",
        json={"question": "What do oranges contain for nutrition?", "top_k": 1},
    )

    assert upload_response.status_code == 201
    assert query_response.status_code == 200
    assert [source["document_id"] for source in query_response.json()["sources"]] == [
        document_id
    ]
    assert query_response.json()["sources"][0]["source_name"] == "nutrition.md"
