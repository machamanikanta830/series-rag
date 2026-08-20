"""Evaluate configured runtime dependencies without performing application work."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from app.config import GenerationProviderKind, VectorStoreKind
from app.generation.ollama import OllamaGenerationProvider
from app.runtime import RuntimeState
from app.vector_stores.qdrant import QdrantVectorStore

ReadinessStatus = Literal["ready", "not_ready"]


@dataclass(frozen=True, slots=True)
class ComponentReadiness:
    """Public-safe readiness state for one configured component."""

    provider: str
    ready: bool


@dataclass(frozen=True, slots=True)
class ReadinessComponents:
    """Readiness states for the runtime's configurable dependencies."""

    embedding: ComponentReadiness
    vector_store: ComponentReadiness
    generation: ComponentReadiness


@dataclass(frozen=True, slots=True)
class ApplicationReadiness:
    """Overall readiness plus inspectable component states."""

    status: ReadinessStatus
    components: ReadinessComponents


def check_runtime_readiness(runtime: RuntimeState) -> ApplicationReadiness:
    """Check only the external services selected by validated configuration."""
    settings = runtime.settings
    embedding = ComponentReadiness(
        provider=settings.embedding_provider.value,
        ready=True,
    )
    vector_store = ComponentReadiness(
        provider=settings.vector_store.value,
        ready=_vector_store_is_ready(runtime),
    )
    generation = ComponentReadiness(
        provider=settings.generation_provider.value,
        ready=_generation_provider_is_ready(runtime),
    )
    components = ReadinessComponents(
        embedding=embedding,
        vector_store=vector_store,
        generation=generation,
    )
    is_ready = embedding.ready and vector_store.ready and generation.ready
    status: ReadinessStatus = "ready" if is_ready else "not_ready"
    return ApplicationReadiness(status=status, components=components)


def _vector_store_is_ready(runtime: RuntimeState) -> bool:
    """Check Qdrant only when it is the explicitly configured vector store."""
    if runtime.settings.vector_store is VectorStoreKind.IN_MEMORY:
        return True
    if not isinstance(runtime.vector_store, QdrantVectorStore):
        return False
    return _safe_readiness_check(runtime.vector_store.is_ready)


def _generation_provider_is_ready(runtime: RuntimeState) -> bool:
    """Check Ollama only when it is the explicitly configured generator."""
    if runtime.settings.generation_provider is GenerationProviderKind.FAKE:
        return True
    if not isinstance(runtime.generation_provider, OllamaGenerationProvider):
        return False
    return _safe_readiness_check(runtime.generation_provider.is_ready)


def _safe_readiness_check(check: Callable[[], bool]) -> bool:
    """Keep third-party failures out of the public readiness response."""
    try:
        return check() is True
    except Exception:
        # This is the external dependency boundary: any provider failure means
        # unavailable, while the endpoint returns only a stable boolean state.
        return False
