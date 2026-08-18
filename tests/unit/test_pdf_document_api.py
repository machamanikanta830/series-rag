"""Tests for bounded native-text PDF document uploads."""

from collections.abc import Iterator

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_document_catalog,
    get_ingestion_service,
    reset_development_application_state,
)
from app.api.main import MAX_UPLOAD_BYTES, app
from app.models import Document
from app.services.ingestion import IngestionService, IngestionStatistics

client = TestClient(app)


class StubIngestionService(IngestionService):
    """Return fixed statistics while recording parsed PDF documents."""

    def __init__(self) -> None:
        self.documents: list[Document] = []

    def ingest(self, document: Document) -> IngestionStatistics:
        """Record the parser output and return deterministic statistics."""
        self.documents.append(document)
        return IngestionStatistics(
            document_id=document.document_id,
            chunks_created=len(document.sections),
            embedding_dimension=3,
            vector_store_name="TestVectorStore",
        )


@pytest.fixture(autouse=True)
def isolate_application_state() -> Iterator[None]:
    """Reset shared storage and dependency overrides around each API test."""
    reset_development_application_state()
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()
    reset_development_application_state()


def _pdf_bytes(page_texts: list[str | None]) -> bytes:
    """Generate a small PDF entirely in memory for one API request."""
    pdf = pymupdf.open()
    for text in page_texts:
        page = pdf.new_page()
        if text is not None:
            page.insert_text((72, 72), text)
    content = pdf.tobytes()
    pdf.close()
    return content


def test_native_text_pdf_upload_returns_serialized_ingestion_response() -> None:
    """The upload adapter sends parsed pages through the existing service boundary."""
    service = StubIngestionService()
    app.dependency_overrides[get_ingestion_service] = lambda: service

    response = client.post(
        "/documents",
        files={
            "file": (
                "course.pdf",
                _pdf_bytes(["Provider responsibility.", "Customer responsibility."]),
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 201
    [document] = service.documents
    assert response.json() == {
        "document_id": document.document_id,
        "filename": "course.pdf",
        "chunks_created": 2,
        "embedding_dimension": 3,
        "vector_store_name": "TestVectorStore",
    }
    assert [section.metadata["page_number"] for section in document.sections] == [
        "1",
        "2",
    ]


def test_uploaded_pdf_chunks_preserve_page_and_filename_metadata() -> None:
    """The shared catalog receives page-aware chunks after normal ingestion."""
    upload_response = client.post(
        "/documents",
        files={
            "file": (
                "pages.pdf",
                _pdf_bytes(["First page content.", "Second page content."]),
                "application/pdf",
            )
        },
    )
    document_id = upload_response.json()["document_id"]

    catalog_document = get_document_catalog().get_document(document_id)
    detail_response = client.get(f"/documents/{document_id}")

    assert upload_response.status_code == 201
    assert catalog_document is not None
    assert [chunk.metadata["page_number"] for chunk in catalog_document.chunks] == [
        "1",
        "2",
    ]
    assert all(
        chunk.metadata["filename"] == "pages.pdf" for chunk in catalog_document.chunks
    )
    assert [
        chunk["metadata"]["page_number"] for chunk in detail_response.json()["chunks"]
    ] == ["1", "2"]


def test_malformed_pdf_returns_stable_unprocessable_response() -> None:
    """Unreadable bytes are rejected before the ingestion service is invoked."""
    response = client.post(
        "/documents",
        files={"file": ("broken.pdf", b"not a PDF", "application/pdf")},
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Uploaded PDF is malformed, password-protected, or unreadable."
    }


def test_pdf_without_extractable_text_returns_stable_response() -> None:
    """A blank native PDF explains the image-only and OCR limitation."""
    response = client.post(
        "/documents",
        files={"file": ("blank.pdf", _pdf_bytes([None]), "application/pdf")},
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": (
            "Uploaded PDF contains no extractable text. "
            "Scanned or image-only PDFs are not supported."
        )
    }


def test_oversized_pdf_is_rejected_before_parsing() -> None:
    """PDF uploads retain the existing one-megabyte bounded-read limit."""
    response = client.post(
        "/documents",
        files={
            "file": (
                "large.pdf",
                b"x" * (MAX_UPLOAD_BYTES + 1),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "Uploaded file exceeds the 1 MB size limit."}


def test_uploaded_pdf_content_is_queryable_through_the_shared_store() -> None:
    """Native PDF text follows the existing offline retrieval path unchanged."""
    upload_response = client.post(
        "/documents",
        files={
            "file": (
                "nutrition.pdf",
                _pdf_bytes(["Oranges contain vitamin C and support nutrition."]),
                "application/pdf",
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
    assert query_response.json()["sources"][0]["document_id"] == document_id
    assert query_response.json()["sources"][0]["source_name"] == "nutrition.pdf"


def test_dependency_override_isolation_restores_default_pdf_ingestion() -> None:
    """PDF tests do not leak a stub ingestion service into later requests."""
    assert get_ingestion_service not in app.dependency_overrides

    response = client.post(
        "/documents",
        files={
            "file": (
                "default.pdf",
                _pdf_bytes(["Default PDF text."]),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 201
    assert response.json()["vector_store_name"] == "InMemoryVectorStore"
