"""Tests for orchestration across the existing RAG components."""

import pytest

from app.context.builder import ContextBuilder, ContextBuildResult
from app.generation.base import GenerationProvider
from app.generation.fake import FakeGenerationProvider
from app.models import Chunk, SearchResult
from app.pipeline.rag_pipeline import RAGPipeline
from app.prompts.builder import PromptBuilder
from app.retrieval.retriever import SemanticRetriever


def _search_result(chunk_id: str, score: float) -> SearchResult:
    """Create an immutable result with source metadata for pipeline tests."""
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="document-1",
            source_name="source.md",
            text=f"Text for {chunk_id}.",
            chunk_index=0,
            start_word=0,
            end_word=3,
            metadata={"topic": "testing"},
        ),
        score=score,
    )


class RecordingRetriever(SemanticRetriever):
    """Return configured results while recording each pipeline retrieval call."""

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Record and return the ranked list without changing it."""
        self.calls.append((query, top_k))
        return self.results


class RecordingContextBuilder(ContextBuilder):
    """Return configured context while recording the exact ranked result list."""

    def __init__(self, result: ContextBuildResult) -> None:
        self.result = result
        self.calls: list[list[SearchResult]] = []

    def build(self, results: list[SearchResult]) -> ContextBuildResult:
        """Record the unchanged list and return the configured context result."""
        self.calls.append(results)
        return self.result


class RecordingPromptBuilder(PromptBuilder):
    """Return configured prompt text while recording its two input values."""

    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        self.calls: list[tuple[str, str]] = []

    def build(self, question: str, context: str) -> str:
        """Record the inputs and return one predetermined completed prompt."""
        self.calls.append((question, context))
        return self.prompt


class FailingRetriever(SemanticRetriever):
    """Raise a configured failure from retrieval."""

    def __init__(self, error: RuntimeError) -> None:
        self.error = error

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Propagate the configured retrieval failure."""
        raise self.error


class FailingContextBuilder(ContextBuilder):
    """Raise a configured failure from context construction."""

    def __init__(self, error: RuntimeError) -> None:
        self.error = error

    def build(self, results: list[SearchResult]) -> ContextBuildResult:
        """Propagate the configured context-building failure."""
        raise self.error


class FailingPromptBuilder(PromptBuilder):
    """Raise a configured failure from prompt construction."""

    def __init__(self, error: RuntimeError) -> None:
        self.error = error

    def build(self, question: str, context: str) -> str:
        """Propagate the configured prompt-building failure."""
        raise self.error


class FailingGenerationProvider(GenerationProvider):
    """Raise a configured failure after shared prompt validation."""

    def __init__(self, error: RuntimeError) -> None:
        self.error = error

    def _generate(self, prompt: str) -> str:
        """Propagate the configured generation failure."""
        raise self.error


def _pipeline(
    retriever: SemanticRetriever,
    context_builder: ContextBuilder,
    prompt_builder: PromptBuilder,
    generation_provider: GenerationProvider,
    top_k: int = 2,
) -> RAGPipeline:
    """Build a pipeline using only injected test collaborators."""
    return RAGPipeline(
        retriever,
        context_builder,
        prompt_builder,
        generation_provider,
        top_k=top_k,
    )


def test_pipeline_orchestrates_each_component_once_and_returns_details() -> None:
    """One answer preserves every intermediate value for inspection."""
    results = [_search_result("first", 0.9), _search_result("second", 0.8)]
    context = "[Source: source.md | Chunk 0]\n\nText for first."
    retriever = RecordingRetriever(results)
    context_builder = RecordingContextBuilder(
        ContextBuildResult(context, (results[0].chunk,))
    )
    prompt_builder = RecordingPromptBuilder("Completed prompt")
    generation_provider = FakeGenerationProvider("Generated answer")
    pipeline = _pipeline(
        retriever,
        context_builder,
        prompt_builder,
        generation_provider,
    )

    result = pipeline.answer("Original question")

    assert retriever.calls == [("Original question", 2)]
    assert context_builder.calls == [results]
    assert context_builder.calls[0] is results
    assert prompt_builder.calls == [("Original question", context)]
    assert generation_provider.received_prompts == ["Completed prompt"]
    assert result.answer == "Generated answer"
    assert result.context == context
    assert result.prompt == "Completed prompt"
    assert result.search_results == tuple(results)
    assert result.included_chunks == (results[0].chunk,)


def test_pipeline_uses_a_per_call_top_k_without_changing_its_default() -> None:
    """Request-specific limits do not mutate later pipeline retrieval behavior."""
    results = [_search_result("first", 0.9)]
    retriever = RecordingRetriever(results)
    pipeline = _pipeline(
        retriever,
        RecordingContextBuilder(ContextBuildResult("Context", (results[0].chunk,))),
        RecordingPromptBuilder("Prompt"),
        FakeGenerationProvider("Answer"),
        top_k=2,
    )

    pipeline.answer("First question", top_k=1)
    pipeline.answer("Second question")

    assert retriever.calls == [("First question", 1), ("Second question", 2)]


@pytest.mark.parametrize("top_k", [0, -1])
def test_pipeline_rejects_non_positive_top_k(top_k: int) -> None:
    """The pipeline cannot request zero or negative retrieval results."""
    results = [_search_result("first", 0.9)]

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        _pipeline(
            RecordingRetriever(results),
            RecordingContextBuilder(ContextBuildResult("Context", (results[0].chunk,))),
            RecordingPromptBuilder("Prompt"),
            FakeGenerationProvider("Answer"),
            top_k=top_k,
        )


def test_pipeline_propagates_retrieval_failures() -> None:
    """The pipeline leaves retriever exceptions unchanged."""
    error = RuntimeError("retrieval failed")
    context_builder = RecordingContextBuilder(ContextBuildResult("Context", ()))
    pipeline = _pipeline(
        FailingRetriever(error),
        context_builder,
        RecordingPromptBuilder("Prompt"),
        FakeGenerationProvider("Answer"),
    )

    with pytest.raises(RuntimeError, match="retrieval failed"):
        pipeline.answer("Question")

    assert context_builder.calls == []


def test_pipeline_propagates_context_building_failures() -> None:
    """The pipeline leaves context-builder exceptions unchanged."""
    error = RuntimeError("context failed")
    pipeline = _pipeline(
        RecordingRetriever([_search_result("first", 0.9)]),
        FailingContextBuilder(error),
        RecordingPromptBuilder("Prompt"),
        FakeGenerationProvider("Answer"),
    )

    with pytest.raises(RuntimeError, match="context failed"):
        pipeline.answer("Question")


def test_pipeline_propagates_prompt_building_failures() -> None:
    """The pipeline leaves prompt-builder exceptions unchanged."""
    error = RuntimeError("prompt failed")
    result = _search_result("first", 0.9)
    pipeline = _pipeline(
        RecordingRetriever([result]),
        RecordingContextBuilder(ContextBuildResult("Context", (result.chunk,))),
        FailingPromptBuilder(error),
        FakeGenerationProvider("Answer"),
    )

    with pytest.raises(RuntimeError, match="prompt failed"):
        pipeline.answer("Question")


def test_pipeline_propagates_generation_failures() -> None:
    """The pipeline leaves generation-provider exceptions unchanged."""
    error = RuntimeError("generation failed")
    result = _search_result("first", 0.9)
    pipeline = _pipeline(
        RecordingRetriever([result]),
        RecordingContextBuilder(ContextBuildResult("Context", (result.chunk,))),
        RecordingPromptBuilder("Prompt"),
        FailingGenerationProvider(error),
    )

    with pytest.raises(RuntimeError, match="generation failed"):
        pipeline.answer("Question")


@pytest.mark.parametrize(
    ("context", "included_chunks"),
    [("", ()), ("Context without chunks", ())],
)
def test_pipeline_rejects_empty_retrieval_context(
    context: str,
    included_chunks: tuple[Chunk, ...],
) -> None:
    """The pipeline never generates an answer without included source evidence."""
    result = _search_result("first", 0.9)
    prompt_builder = RecordingPromptBuilder("Prompt")
    generation_provider = FakeGenerationProvider("Answer")
    pipeline = _pipeline(
        RecordingRetriever([result]),
        RecordingContextBuilder(ContextBuildResult(context, included_chunks)),
        prompt_builder,
        generation_provider,
    )

    with pytest.raises(ValueError, match="Retrieved context is empty"):
        pipeline.answer("Question")

    assert prompt_builder.calls == []
    assert generation_provider.received_prompts == []
