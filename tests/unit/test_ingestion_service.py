"""Tests for explicit document-ingestion orchestration."""

from dataclasses import FrozenInstanceError

import pytest

from app.embeddings.base import EmbeddingProvider
from app.models import Chunk, Document
from app.services.ingestion import IngestionService, IngestionStatistics
from app.vector_stores.in_memory import InMemoryVectorStore


class FakeEmbeddingProvider(EmbeddingProvider):
    """Return deterministic vectors while recording document embedding calls."""

    def __init__(
        self,
        dimension: int = 2,
        error: RuntimeError | None = None,
    ) -> None:
        self.dimension = dimension
        self.error = error
        self.document_calls: list[list[str]] = []
        self.dimension_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Record texts and return one finite vector per input."""
        self.document_calls.append(texts)
        if self.error is not None:
            raise self.error
        return [[1.0, *([0.0] * (self.dimension - 1))] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return a compatible vector for store assertions."""
        return [1.0, *([0.0] * (self.dimension - 1))]

    def embedding_dimension(self) -> int:
        """Record and return the configured dimension."""
        self.dimension_calls += 1
        return self.dimension


class RecordingInMemoryVectorStore(InMemoryVectorStore):
    """Record upsert batches before delegating to the real in-memory store."""

    def __init__(self) -> None:
        super().__init__()
        self.upsert_calls: list[tuple[list[Chunk], list[list[float]]]] = []

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Record the exact batch and preserve normal store behavior."""
        self.upsert_calls.append((chunks, embeddings))
        super().upsert(chunks, embeddings)


class FailingInMemoryVectorStore(InMemoryVectorStore):
    """Raise one configured storage failure from upsert."""

    def __init__(self, error: RuntimeError) -> None:
        super().__init__()
        self.error = error

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Propagate the configured vector-store failure."""
        raise self.error


def _document(text: str = "one two three four five") -> Document:
    """Create one immutable document with source metadata."""
    return Document(
        document_id="document-1",
        source_name="lesson.md",
        source_path="course/lesson.md",
        text=text,
        metadata={"course": "cloud"},
    )


def test_service_normalizes_chunks_embeds_and_upserts_a_document() -> None:
    """Successful ingestion passes each explicit intermediate value forward."""
    provider = FakeEmbeddingProvider()
    store = RecordingInMemoryVectorStore()
    service = IngestionService(provider, store, chunk_size=3, chunk_overlap=1)

    statistics = service.ingest(_document(" one   two\r\nthree four five "))

    assert provider.document_calls == [["one two three", "three four five"]]
    [(chunks, embeddings)] = store.upsert_calls
    assert [chunk.text for chunk in chunks] == ["one two three", "three four five"]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert all(chunk.metadata == {"course": "cloud"} for chunk in chunks)
    assert embeddings == [[1.0, 0.0], [1.0, 0.0]]
    assert statistics.chunks_created == 2


def test_service_handles_a_document_that_normalizes_to_zero_chunks() -> None:
    """Empty content reports zero chunks without embedding or writing a batch."""
    provider = FakeEmbeddingProvider(dimension=4)
    store = RecordingInMemoryVectorStore()

    statistics = IngestionService(provider, store).ingest(_document(" \r\n\t\x00 "))

    assert statistics.chunks_created == 0
    assert statistics.embedding_dimension == 4
    assert provider.document_calls == []
    assert provider.dimension_calls == 1
    assert store.upsert_calls == []


def test_statistics_report_document_dimension_and_vector_store() -> None:
    """The result identifies the document and injected storage implementation."""
    provider = FakeEmbeddingProvider(dimension=3)
    store = InMemoryVectorStore()

    statistics = IngestionService(provider, store).ingest(_document())

    assert statistics == IngestionStatistics(
        document_id="document-1",
        chunks_created=1,
        embedding_dimension=3,
        vector_store_name="InMemoryVectorStore",
    )


def test_statistics_are_immutable() -> None:
    """Callers cannot rewrite a completed ingestion summary."""
    statistics = IngestionService(
        FakeEmbeddingProvider(), InMemoryVectorStore()
    ).ingest(_document())

    with pytest.raises(FrozenInstanceError):
        statistics.chunks_created = 99  # type: ignore[misc]


def test_repeated_ingestion_replaces_existing_chunks_without_duplicates() -> None:
    """Deterministic chunk IDs preserve vector-store upsert semantics."""
    provider = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    service = IngestionService(provider, store, chunk_size=3, chunk_overlap=1)
    document = _document()

    first_statistics = service.ingest(document)
    second_statistics = service.ingest(document)
    results = store.search(provider.embed_query("query"), top_k=10)

    assert first_statistics == second_statistics
    assert len(results) == 2
    assert len({result.chunk.chunk_id for result in results}) == 2
    assert len(provider.document_calls) == 2


def test_embedding_provider_failure_propagates_without_storage_write() -> None:
    """The service does not hide provider errors or continue to vector storage."""
    error = RuntimeError("embedding failed")
    provider = FakeEmbeddingProvider(error=error)
    store = RecordingInMemoryVectorStore()

    with pytest.raises(RuntimeError, match="embedding failed"):
        IngestionService(provider, store).ingest(_document())

    assert store.upsert_calls == []


def test_vector_store_failure_propagates() -> None:
    """The service preserves failures raised by the configured vector store."""
    error = RuntimeError("storage failed")
    service = IngestionService(
        FakeEmbeddingProvider(),
        FailingInMemoryVectorStore(error),
    )

    with pytest.raises(RuntimeError, match="storage failed"):
        service.ingest(_document())
