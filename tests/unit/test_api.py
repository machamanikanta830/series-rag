"""Tests for SeriesRAG's FastAPI HTTP adapter."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_rag_pipeline,
    reset_development_application_state,
)
from app.api.main import API_DESCRIPTION, API_TITLE, API_VERSION, app
from app.models import Chunk, SearchResult
from app.pipeline.rag_pipeline import RAGPipeline, RAGPipelineResult

client = TestClient(app)


def _search_result(
    chunk_id: str,
    *,
    score: float,
    source_name: str,
    chunk_index: int,
) -> SearchResult:
    """Create a deterministic domain result for API serialization tests."""
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="document-1",
            source_name=source_name,
            text=f"Text for {chunk_id}.",
            chunk_index=chunk_index,
            start_word=0,
            end_word=3,
            metadata={"topic": "api-test"},
        ),
        score=score,
    )


class StubPipeline(RAGPipeline):
    """Return or raise a configured value while recording API calls."""

    def __init__(
        self,
        result: RAGPipelineResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[str, int | None]] = []

    def answer(self, question: str, top_k: int | None = None) -> RAGPipelineResult:
        """Record one call before returning or raising the configured outcome."""
        self.calls.append((question, top_k))
        if self.error is not None:
            raise self.error
        if self.result is None:
            raise AssertionError("StubPipeline needs a result or error")
        return self.result


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Iterator[None]:
    """Keep FastAPI dependency overrides isolated between API tests."""
    reset_development_application_state()
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()
    reset_development_application_state()


def _override_pipeline(pipeline: StubPipeline) -> None:
    """Install one test-owned pipeline dependency for the current test."""
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline


def _successful_result() -> RAGPipelineResult:
    """Return two ordered sources and transparent answer details."""
    first = _search_result(
        "first",
        score=0.91,
        source_name="first.md",
        chunk_index=0,
    )
    second = _search_result(
        "second",
        score=0.73,
        source_name="second.md",
        chunk_index=2,
    )
    return RAGPipelineResult(
        answer="Grounded answer.",
        context="Built context.",
        prompt="Completed prompt.",
        search_results=(first, second),
        included_chunks=(first.chunk, second.chunk),
    )


def test_root_returns_api_metadata() -> None:
    """The root route identifies this API without using RAG dependencies."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
    }


def test_health_returns_ok() -> None:
    """The health route remains independent of model and service readiness."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_openapi_endpoint_describes_the_api() -> None:
    """FastAPI's OpenAPI document includes the working query route."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["info"] == {
        "title": API_TITLE,
        "description": API_DESCRIPTION,
        "version": API_VERSION,
    }
    assert "/query" in response.json()["paths"]


def test_query_serializes_pipeline_result_and_preserves_source_order() -> None:
    """The API returns answer details and ordered source citation fields."""
    pipeline = StubPipeline(result=_successful_result())
    _override_pipeline(pipeline)

    response = client.post(
        "/query",
        json={"question": "What is grounded?", "top_k": 2},
    )

    assert response.status_code == 200
    assert pipeline.calls == [("What is grounded?", 2)]
    assert response.json() == {
        "answer": "Grounded answer.",
        "context": "Built context.",
        "prompt": "Completed prompt.",
        "sources": [
            {
                "chunk_id": "first",
                "document_id": "document-1",
                "source_name": "first.md",
                "chunk_index": 0,
                "text": "Text for first.",
                "score": 0.91,
            },
            {
                "chunk_id": "second",
                "document_id": "document-1",
                "source_name": "second.md",
                "chunk_index": 2,
                "text": "Text for second.",
                "score": 0.73,
            },
        ],
    }


def test_query_requires_a_question_field() -> None:
    """Pydantic validation rejects requests that omit the required question."""
    response = client.post("/query", json={"top_k": 5})

    assert response.status_code == 422


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_query_rejects_empty_or_whitespace_only_questions(question: str) -> None:
    """A query must contain visible question text."""
    response = client.post("/query", json={"question": question, "top_k": 5})

    assert response.status_code == 422


@pytest.mark.parametrize("top_k", [0, -1])
def test_query_rejects_non_positive_top_k(top_k: int) -> None:
    """A query must request at least one ranked result."""
    response = client.post(
        "/query", json={"question": "What is shared responsibility?", "top_k": top_k}
    )

    assert response.status_code == 422


def test_query_maps_no_context_failure_to_client_error() -> None:
    """Expected no-evidence failures become a clear client-facing response."""
    pipeline = StubPipeline(error=ValueError("Retrieved context is empty"))
    _override_pipeline(pipeline)

    response = client.post("/query", json={"question": "Unknown?", "top_k": 5})

    assert response.status_code == 422
    assert response.json() == {
        "detail": "No usable context was found for this question."
    }


def test_query_maps_unexpected_pipeline_failure_to_server_error() -> None:
    """Unexpected failures return a stable response without internal details."""
    pipeline = StubPipeline(error=RuntimeError("private internal failure"))
    _override_pipeline(pipeline)

    response = client.post("/query", json={"question": "Question?", "top_k": 5})

    assert response.status_code == 500
    assert response.json() == {"detail": "The query could not be completed."}
    assert "private internal failure" not in response.text


def test_default_dependency_is_restored_and_runs_entirely_offline() -> None:
    """Override cleanup restores the deterministic development pipeline."""
    assert get_rag_pipeline not in app.dependency_overrides

    response = client.post(
        "/query",
        json={"question": "Who protects cloud infrastructure?", "top_k": 1},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "This is a deterministic development answer."
    assert [source["chunk_id"] for source in response.json()["sources"]] == [
        "development-provider-responsibility"
    ]
