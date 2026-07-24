"""Tests for SeriesRAG's FastAPI foundation routes."""

import pytest
from fastapi.testclient import TestClient

from app.api.main import API_DESCRIPTION, API_TITLE, API_VERSION, app

client = TestClient(app)


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
    """The health route is available before dependency checks are added."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_openapi_endpoint_describes_the_api() -> None:
    """FastAPI's OpenAPI document remains enabled for future consumers."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["info"] == {
        "title": API_TITLE,
        "description": API_DESCRIPTION,
        "version": API_VERSION,
    }
    assert "/query" in response.json()["paths"]


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


def test_query_returns_not_implemented_after_validation() -> None:
    """The public route reports that RAG service wiring is a later milestone."""
    response = client.post(
        "/query", json={"question": "What is shared responsibility?", "top_k": 5}
    )

    assert response.status_code == 501
    assert response.json() == {
        "detail": (
            "Query endpoint is not implemented yet; it will be connected to the "
            "RAG service in the next milestone."
        )
    }
