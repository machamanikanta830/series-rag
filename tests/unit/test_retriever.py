"""Tests for query-time embedding and vector-search orchestration."""

import pytest

from app.embeddings.base import EmbeddingProvider
from app.models import Chunk, SearchResult
from app.retrieval.retriever import SemanticRetriever
from app.vector_stores.base import VectorStore


def _result(chunk_id: str, score: float, topic: str) -> SearchResult:
    """Create a small immutable result with recognizable metadata."""
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="document-1",
            source_name="source.txt",
            text=f"Text for {chunk_id}",
            chunk_index=0,
            start_word=0,
            end_word=3,
            metadata={"topic": topic},
        ),
        score=score,
    )


class RecordingEmbeddingProvider(EmbeddingProvider):
    """A deterministic provider that records query calls without loading a model."""

    def __init__(self, query_embeddings: list[list[float]]) -> None:
        self._query_embeddings = query_embeddings
        self.queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """This unit-test provider is only used for query embeddings."""
        raise AssertionError(f"embed_documents was called with {texts!r}")

    def embed_query(self, text: str) -> list[float]:
        """Return the next configured vector and record the supplied query."""
        self.queries.append(text)
        return self._query_embeddings.pop(0)

    def embedding_dimension(self) -> int:
        """Return the dimension of the next configured test vector."""
        return len(self._query_embeddings[0])


class RecordingVectorStore(VectorStore):
    """A deterministic store that records searches without storing data."""

    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.search_calls: list[tuple[list[float], int]] = []

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """This unit-test store is only used for searches."""
        raise AssertionError(f"upsert was called with {chunks!r} and {embeddings!r}")

    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[SearchResult]:
        """Record the query and return the configured result list unchanged."""
        self.search_calls.append((query_embedding, top_k))
        return self._results


def test_retrieve_strips_query_and_returns_store_results_unchanged() -> None:
    """The retriever delegates exactly once and preserves result identity and order."""
    query_embedding = [0.1, 0.2]
    results = [_result("first", 0.9, "security"), _result("second", 0.4, "network")]
    provider = RecordingEmbeddingProvider([query_embedding])
    store = RecordingVectorStore(results)
    retriever = SemanticRetriever(provider, store)

    retrieved = retriever.retrieve("  shared responsibility  ", top_k=2)

    assert provider.queries == ["shared responsibility"]
    assert store.search_calls == [(query_embedding, 2)]
    assert store.search_calls[0][0] is query_embedding
    assert retrieved is results
    assert [result.chunk.chunk_id for result in retrieved] == ["first", "second"]
    assert [result.score for result in retrieved] == [0.9, 0.4]
    assert retrieved[0].chunk.metadata == {"topic": "security"}


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_retrieve_rejects_empty_or_whitespace_only_queries(query: str) -> None:
    """Queries must include visible text before embedding starts."""
    retriever = SemanticRetriever(
        RecordingEmbeddingProvider([[0.1, 0.2]]),
        RecordingVectorStore([]),
    )

    with pytest.raises(ValueError, match="must not be empty"):
        retriever.retrieve(query)


@pytest.mark.parametrize("query", [None, 42])
def test_retrieve_rejects_non_string_queries(query: object) -> None:
    """Only strings are valid query inputs."""
    retriever = SemanticRetriever(
        RecordingEmbeddingProvider([[0.1, 0.2]]),
        RecordingVectorStore([]),
    )

    with pytest.raises(TypeError, match="must be a string"):
        retriever.retrieve(query)  # type: ignore[arg-type]


@pytest.mark.parametrize("top_k", [0, -1])
def test_retrieve_rejects_an_invalid_result_limit(top_k: int) -> None:
    """A retrieval request must ask for at least one result."""
    retriever = SemanticRetriever(
        RecordingEmbeddingProvider([[0.1, 0.2]]),
        RecordingVectorStore([]),
    )

    with pytest.raises(ValueError, match="greater than zero"):
        retriever.retrieve("query", top_k=top_k)


def test_embedding_provider_errors_are_propagated() -> None:
    """The retriever does not hide a clear embedding-provider failure."""
    provider = FailingEmbeddingProvider()
    retriever = SemanticRetriever(provider, RecordingVectorStore([]))

    with pytest.raises(RuntimeError, match="embedding failed"):
        retriever.retrieve("query")


def test_vector_store_errors_are_propagated() -> None:
    """The retriever does not hide a clear vector-store failure."""
    store = FailingVectorStore()
    retriever = SemanticRetriever(RecordingEmbeddingProvider([[0.1, 0.2]]), store)

    with pytest.raises(RuntimeError, match="search failed"):
        retriever.retrieve("query")


def test_repeated_retrievals_embed_each_query_without_reusing_a_vector() -> None:
    """No retriever cache leaks one query's embedding into the next request."""
    first_embedding = [0.1, 0.2]
    second_embedding = [0.3, 0.4]
    provider = RecordingEmbeddingProvider([first_embedding, second_embedding])
    store = RecordingVectorStore([])
    retriever = SemanticRetriever(provider, store)

    retriever.retrieve("first query")
    retriever.retrieve("second query")

    assert provider.queries == ["first query", "second query"]
    assert store.search_calls == [(first_embedding, 5), (second_embedding, 5)]
    assert store.search_calls[0][0] is first_embedding
    assert store.search_calls[1][0] is second_embedding


class FailingEmbeddingProvider(EmbeddingProvider):
    """A provider that always raises a recognizable query-embedding failure."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """This fake never embeds documents."""
        raise AssertionError(f"embed_documents was called with {texts!r}")

    def embed_query(self, text: str) -> list[float]:
        """Raise the error whose propagation the test verifies."""
        raise RuntimeError("embedding failed")

    def embedding_dimension(self) -> int:
        """Return an unused fixed test dimension."""
        return 2


class FailingVectorStore(VectorStore):
    """A store that always raises a recognizable search failure."""

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """This fake never stores embeddings."""
        raise AssertionError(f"upsert was called with {chunks!r} and {embeddings!r}")

    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[SearchResult]:
        """Raise the error whose propagation the test verifies."""
        raise RuntimeError("search failed")
