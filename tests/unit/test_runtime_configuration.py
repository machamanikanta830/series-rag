"""Tests for environment settings and the shared runtime dependency graph."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

import app.runtime as runtime_module
from app.api.dependencies import (
    get_document_catalog,
    get_document_deletion_service,
    get_ingestion_service,
    get_rag_pipeline,
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
from app.embeddings.base import EmbeddingProvider
from app.generation.fake import FakeGenerationProvider
from app.generation.ollama import OllamaGenerationProvider
from app.models import Chunk, SearchResult
from app.pipeline.rag_pipeline import RAGPipeline, RAGPipelineResult
from app.vector_stores.in_memory import InMemoryVectorStore
from app.vector_stores.qdrant import QdrantVectorStore


class StubEmbeddingProvider(EmbeddingProvider):
    """Expose a fixed dimension without loading a real embedding model."""

    def __init__(self, model_name: str = "stub-model") -> None:
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def embedding_dimension(self) -> int:
        return 3


class OverridePipeline(RAGPipeline):
    """Return one fixed value without constructing the real pipeline graph."""

    def answer(self, question: str, top_k: int | None = None) -> RAGPipelineResult:
        chunk = Chunk(
            chunk_id="override-chunk",
            document_id="override-document",
            source_name="override.md",
            text="Override evidence.",
            chunk_index=0,
            start_word=0,
            end_word=2,
        )
        search_result = SearchResult(chunk=chunk, score=0.75)
        return RAGPipelineResult(
            answer="Override answer.",
            context="Override context.",
            prompt="Override prompt.",
            search_results=(search_result,),
            included_chunks=(chunk,),
        )


@pytest.fixture(autouse=True)
def isolate_application_state() -> Iterator[None]:
    """Keep runtime state and dependency overrides independent across tests."""
    reset_development_application_state()
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()
    reset_development_application_state()


def test_default_configuration_is_offline_and_deterministic() -> None:
    """An empty environment selects providers usable without external services."""
    settings = RuntimeSettings.from_environment({})
    state = runtime_module.build_runtime_state(settings)

    assert settings.embedding_provider is EmbeddingProviderKind.DEVELOPMENT
    assert settings.vector_store is VectorStoreKind.IN_MEMORY
    assert settings.generation_provider is GenerationProviderKind.FAKE
    assert settings.context_max_characters == 4_000
    assert settings.default_top_k == 5
    assert isinstance(state.vector_store, InMemoryVectorStore)
    assert isinstance(state.generation_provider, FakeGenerationProvider)


def test_qdrant_selection_uses_configured_url_collection_and_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qdrant wiring uses mocked third-party boundaries without network access."""
    client_sentinel = object()
    received_urls: list[str] = []

    def build_client(*, url: str) -> object:
        received_urls.append(url)
        return client_sentinel

    monkeypatch.setattr(
        runtime_module,
        "SentenceTransformerEmbeddingProvider",
        StubEmbeddingProvider,
    )
    monkeypatch.setattr(runtime_module, "QdrantClient", build_client)
    settings = RuntimeSettings.from_environment(
        {
            "SERIESRAG_EMBEDDING_PROVIDER": "sentence_transformer",
            "SERIESRAG_VECTOR_STORE": "qdrant",
            "SERIESRAG_QDRANT_URL": "http://qdrant.internal:6333",
            "SERIESRAG_QDRANT_COLLECTION": "course-notes",
        }
    )

    state = runtime_module.build_runtime_state(settings)

    assert isinstance(state.embedding_provider, StubEmbeddingProvider)
    assert isinstance(state.vector_store, QdrantVectorStore)
    assert received_urls == ["http://qdrant.internal:6333"]
    assert state.vector_store._collection_name == "course-notes"
    assert state.vector_store._embedding_dimension == 3
    assert state.vector_store._client is client_sentinel


def test_ollama_selection_uses_configured_url_and_model() -> None:
    """Ollama can replace fake generation without changing other components."""
    settings = RuntimeSettings.from_environment(
        {
            "SERIESRAG_GENERATION_PROVIDER": "ollama",
            "SERIESRAG_OLLAMA_URL": "http://ollama.internal:11434",
            "SERIESRAG_OLLAMA_MODEL": "llama-test",
        }
    )

    state = runtime_module.build_runtime_state(settings)

    assert isinstance(state.generation_provider, OllamaGenerationProvider)
    assert state.generation_provider._base_url == "http://ollama.internal:11434"
    assert state.generation_provider._model == "llama-test"


@pytest.mark.parametrize(
    ("name", "value", "expected_message"),
    [
        (
            "SERIESRAG_VECTOR_STORE",
            "pinecone",
            "Unsupported SERIESRAG_VECTOR_STORE value 'pinecone'",
        ),
        (
            "SERIESRAG_GENERATION_PROVIDER",
            "openai",
            "Unsupported SERIESRAG_GENERATION_PROVIDER value 'openai'",
        ),
    ],
)
def test_unsupported_provider_names_are_rejected(
    name: str,
    value: str,
    expected_message: str,
) -> None:
    """Requested providers never silently fall back to development defaults."""
    with pytest.raises(ValueError, match=expected_message):
        RuntimeSettings.from_environment({name: value})


@pytest.mark.parametrize(
    ("environment", "expected_message"),
    [
        (
            {
                "SERIESRAG_VECTOR_STORE": "qdrant",
                "SERIESRAG_EMBEDDING_PROVIDER": "development",
            },
            "requires SERIESRAG_EMBEDDING_PROVIDER=sentence_transformer",
        ),
        (
            {
                "SERIESRAG_VECTOR_STORE": "qdrant",
                "SERIESRAG_EMBEDDING_PROVIDER": "sentence_transformer",
                "SERIESRAG_QDRANT_URL": " ",
            },
            "SERIESRAG_QDRANT_URL is required",
        ),
        (
            {
                "SERIESRAG_VECTOR_STORE": "qdrant",
                "SERIESRAG_EMBEDDING_PROVIDER": "sentence_transformer",
                "SERIESRAG_QDRANT_COLLECTION": " ",
            },
            "SERIESRAG_QDRANT_COLLECTION is required",
        ),
        (
            {
                "SERIESRAG_GENERATION_PROVIDER": "ollama",
                "SERIESRAG_OLLAMA_MODEL": " ",
            },
            "SERIESRAG_OLLAMA_MODEL is required",
        ),
    ],
)
def test_provider_specific_required_configuration_is_validated(
    environment: dict[str, str],
    expected_message: str,
) -> None:
    """Incomplete production-style selections fail before serving requests."""
    with pytest.raises(ValueError, match=expected_message):
        RuntimeSettings.from_environment(environment)


def test_dependency_getters_reuse_one_runtime_graph() -> None:
    """Every API dependency resolves a component from the same process state."""
    first_state = get_runtime_state()
    second_state = get_runtime_state()

    assert first_state is second_state
    assert get_rag_pipeline() is first_state.rag_pipeline
    assert get_ingestion_service() is first_state.ingestion_service
    assert get_document_catalog() is first_state.document_catalog
    assert get_document_deletion_service() is first_state.deletion_service


def test_fastapi_pipeline_dependency_remains_overrideable() -> None:
    """Runtime wiring does not prevent an offline endpoint dependency override."""
    override_pipeline = OverridePipeline.__new__(OverridePipeline)
    app.dependency_overrides[get_rag_pipeline] = lambda: override_pipeline

    response = TestClient(app).post(
        "/query",
        json={"question": "Use the override", "top_k": 2},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Override answer."
