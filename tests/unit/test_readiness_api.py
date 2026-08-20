"""Tests for process liveness and configured dependency readiness."""

from collections.abc import Iterator
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_runtime_state,
    reset_development_application_state,
)
from app.api.main import app
from app.config import (
    EmbeddingProviderKind,
    GenerationProviderKind,
    RuntimeSettings,
    VectorStoreKind,
)
from app.generation.ollama import OllamaGenerationProvider
from app.runtime import RuntimeState
from app.vector_stores.qdrant import QdrantVectorStore

client = TestClient(app)


class StubOllamaProvider(OllamaGenerationProvider):
    """Expose deterministic readiness without making an HTTP request."""

    def __init__(self, ready: bool, error: Exception | None = None) -> None:
        super().__init__(model="configured-model")
        self._ready = ready
        self._error = error

    def is_ready(self) -> bool:
        if self._error is not None:
            raise self._error
        return self._ready


@pytest.fixture(autouse=True)
def isolate_application_state() -> Iterator[None]:
    """Keep readiness dependency overrides isolated between tests."""
    reset_development_application_state()
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()
    reset_development_application_state()


def _qdrant_runtime(*, ready: bool) -> RuntimeState:
    """Return a Qdrant-configured state backed by a non-networking fake client."""
    base_runtime = get_runtime_state()
    qdrant_client = MagicMock()
    if not ready:
        qdrant_client.get_collections.side_effect = RuntimeError(
            "private qdrant connection details"
        )
    vector_store = QdrantVectorStore(
        collection_name="readiness-test",
        embedding_dimension=2,
        client=qdrant_client,
    )
    settings = RuntimeSettings(
        embedding_provider=EmbeddingProviderKind.SENTENCE_TRANSFORMER,
        vector_store=VectorStoreKind.QDRANT,
    )
    return replace(
        base_runtime,
        settings=settings,
        vector_store=vector_store,
    )


def _ollama_runtime(
    *,
    ready: bool,
    error: Exception | None = None,
) -> RuntimeState:
    """Return an Ollama-configured state with deterministic readiness."""
    base_runtime = get_runtime_state()
    settings = RuntimeSettings(
        generation_provider=GenerationProviderKind.OLLAMA,
        ollama_model="configured-model",
    )
    return replace(
        base_runtime,
        settings=settings,
        generation_provider=StubOllamaProvider(ready, error),
    )


def _override_runtime(runtime: RuntimeState) -> None:
    """Provide one test-owned runtime through FastAPI's dependency boundary."""
    app.dependency_overrides[get_runtime_state] = lambda: runtime


def test_health_is_network_free_even_when_runtime_dependency_would_fail() -> None:
    """Liveness remains independent from all runtime dependency checks."""

    def fail_if_called() -> RuntimeState:
        raise AssertionError("health must not resolve runtime dependencies")

    app.dependency_overrides[get_runtime_state] = fail_if_called

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_development_runtime_reports_ready_with_structured_components() -> None:
    """Offline development components require no external readiness checks."""
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "components": {
            "embedding": {"provider": "development", "ready": True},
            "vector_store": {"provider": "in_memory", "ready": True},
            "generation": {"provider": "fake", "ready": True},
        },
    }


def test_unavailable_qdrant_returns_not_ready_without_internal_details() -> None:
    """Qdrant failures produce HTTP 503 and do not expose provider exceptions."""
    _override_runtime(_qdrant_runtime(ready=False))

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["components"]["vector_store"] == {
        "provider": "qdrant",
        "ready": False,
    }
    assert "private qdrant" not in response.text


def test_unavailable_ollama_returns_not_ready() -> None:
    """An unavailable configured generation service makes the runtime unready."""
    _override_runtime(_ollama_runtime(ready=False))

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["components"]["generation"] == {
        "provider": "ollama",
        "ready": False,
    }


def test_mixed_component_readiness_preserves_each_component_state() -> None:
    """One unavailable dependency does not hide other component results."""
    qdrant_runtime = _qdrant_runtime(ready=True)
    settings = replace(
        qdrant_runtime.settings,
        generation_provider=GenerationProviderKind.OLLAMA,
        ollama_model="configured-model",
    )
    mixed_runtime = replace(
        qdrant_runtime,
        settings=settings,
        generation_provider=StubOllamaProvider(ready=False),
    )
    _override_runtime(mixed_runtime)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "components": {
            "embedding": {"provider": "sentence_transformer", "ready": True},
            "vector_store": {"provider": "qdrant", "ready": True},
            "generation": {"provider": "ollama", "ready": False},
        },
    }


def test_internal_provider_exception_is_contained() -> None:
    """Unexpected dependency errors become stable readiness state only."""
    _override_runtime(
        _ollama_runtime(
            ready=False,
            error=RuntimeError("secret internal provider failure"),
        )
    )

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert "secret internal" not in response.text


def test_ready_endpoint_uses_fastapi_dependency_override() -> None:
    """Tests can replace shared runtime state without changing application code."""
    overridden_runtime = _ollama_runtime(ready=True)
    _override_runtime(overridden_runtime)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["components"]["generation"] == {
        "provider": "ollama",
        "ready": True,
    }
