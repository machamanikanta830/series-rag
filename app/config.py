"""Explicit environment-driven settings for the SeriesRAG runtime graph."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from app.embeddings.sentence_transformers import DEFAULT_MODEL_NAME


class EmbeddingProviderKind(StrEnum):
    """Embedding implementations available to the runtime."""

    DEVELOPMENT = "development"
    SENTENCE_TRANSFORMER = "sentence_transformer"


class VectorStoreKind(StrEnum):
    """Vector-store implementations available to the runtime."""

    IN_MEMORY = "in_memory"
    QDRANT = "qdrant"


class GenerationProviderKind(StrEnum):
    """Generation implementations available to the runtime."""

    FAKE = "fake"
    OLLAMA = "ollama"


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Validated configuration used to construct one application runtime."""

    embedding_provider: EmbeddingProviderKind = EmbeddingProviderKind.DEVELOPMENT
    vector_store: VectorStoreKind = VectorStoreKind.IN_MEMORY
    generation_provider: GenerationProviderKind = GenerationProviderKind.FAKE
    embedding_model: str = DEFAULT_MODEL_NAME
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "seriesrag"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = ""
    context_max_characters: int = 4_000
    default_top_k: int = 5

    def __post_init__(self) -> None:
        """Normalize text and reject incompatible or incomplete settings."""
        _require_kind(
            self.embedding_provider,
            EmbeddingProviderKind,
            "SERIESRAG_EMBEDDING_PROVIDER",
        )
        _require_kind(
            self.vector_store,
            VectorStoreKind,
            "SERIESRAG_VECTOR_STORE",
        )
        _require_kind(
            self.generation_provider,
            GenerationProviderKind,
            "SERIESRAG_GENERATION_PROVIDER",
        )

        for field_name in (
            "embedding_model",
            "qdrant_url",
            "qdrant_collection",
            "ollama_url",
            "ollama_model",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")
            object.__setattr__(self, field_name, value.strip())

        _require_positive_integer(
            self.context_max_characters,
            "SERIESRAG_CONTEXT_MAX_CHARACTERS",
        )
        _require_positive_integer(self.default_top_k, "SERIESRAG_DEFAULT_TOP_K")

        if (
            self.embedding_provider is EmbeddingProviderKind.SENTENCE_TRANSFORMER
            and not self.embedding_model
        ):
            raise ValueError(
                "SERIESRAG_EMBEDDING_MODEL is required when "
                "SERIESRAG_EMBEDDING_PROVIDER=sentence_transformer"
            )

        if self.vector_store is VectorStoreKind.QDRANT:
            if (
                self.embedding_provider
                is not EmbeddingProviderKind.SENTENCE_TRANSFORMER
            ):
                raise ValueError(
                    "SERIESRAG_VECTOR_STORE=qdrant requires "
                    "SERIESRAG_EMBEDDING_PROVIDER=sentence_transformer"
                )
            if not self.qdrant_url:
                raise ValueError(
                    "SERIESRAG_QDRANT_URL is required when "
                    "SERIESRAG_VECTOR_STORE=qdrant"
                )
            if not self.qdrant_collection:
                raise ValueError(
                    "SERIESRAG_QDRANT_COLLECTION is required when "
                    "SERIESRAG_VECTOR_STORE=qdrant"
                )

        if self.generation_provider is GenerationProviderKind.OLLAMA:
            if not self.ollama_url:
                raise ValueError(
                    "SERIESRAG_OLLAMA_URL is required when "
                    "SERIESRAG_GENERATION_PROVIDER=ollama"
                )
            if not self.ollama_model:
                raise ValueError(
                    "SERIESRAG_OLLAMA_MODEL is required when "
                    "SERIESRAG_GENERATION_PROVIDER=ollama"
                )

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> "RuntimeSettings":
        """Read one validated settings object from a process environment mapping."""
        values = os.environ if environment is None else environment
        return cls(
            embedding_provider=_read_kind(
                values,
                "SERIESRAG_EMBEDDING_PROVIDER",
                EmbeddingProviderKind,
                EmbeddingProviderKind.DEVELOPMENT,
            ),
            vector_store=_read_kind(
                values,
                "SERIESRAG_VECTOR_STORE",
                VectorStoreKind,
                VectorStoreKind.IN_MEMORY,
            ),
            generation_provider=_read_kind(
                values,
                "SERIESRAG_GENERATION_PROVIDER",
                GenerationProviderKind,
                GenerationProviderKind.FAKE,
            ),
            embedding_model=values.get(
                "SERIESRAG_EMBEDDING_MODEL",
                DEFAULT_MODEL_NAME,
            ),
            qdrant_url=values.get(
                "SERIESRAG_QDRANT_URL",
                "http://localhost:6333",
            ),
            qdrant_collection=values.get(
                "SERIESRAG_QDRANT_COLLECTION",
                "seriesrag",
            ),
            ollama_url=values.get(
                "SERIESRAG_OLLAMA_URL",
                "http://localhost:11434",
            ),
            ollama_model=values.get("SERIESRAG_OLLAMA_MODEL", ""),
            context_max_characters=_read_positive_integer(
                values,
                "SERIESRAG_CONTEXT_MAX_CHARACTERS",
                4_000,
            ),
            default_top_k=_read_positive_integer(
                values,
                "SERIESRAG_DEFAULT_TOP_K",
                5,
            ),
        )


def _read_kind[KindT: StrEnum](
    environment: Mapping[str, str],
    name: str,
    kind_type: type[KindT],
    default: KindT,
) -> KindT:
    """Read one enum setting with an error listing every supported value."""
    raw_value = environment.get(name, default.value).strip().lower()
    try:
        return kind_type(raw_value)
    except ValueError as error:
        supported_values = ", ".join(kind.value for kind in kind_type)
        raise ValueError(
            f"Unsupported {name} value {raw_value!r}. "
            f"Expected one of: {supported_values}."
        ) from error


def _read_positive_integer(
    environment: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    """Read a positive integer without accepting decimal or boolean text."""
    raw_value = environment.get(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    _require_positive_integer(value, name)
    return value


def _require_positive_integer(value: int, name: str) -> None:
    """Reject booleans, non-integers, and non-positive integer settings."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_kind[KindT: StrEnum](
    value: object, kind_type: type[KindT], name: str
) -> None:
    """Reject direct construction with a value outside the declared enum."""
    if not isinstance(value, kind_type):
        supported_values = ", ".join(kind.value for kind in kind_type)
        raise ValueError(
            f"Unsupported {name} value {value!r}. Expected one of: {supported_values}."
        )
