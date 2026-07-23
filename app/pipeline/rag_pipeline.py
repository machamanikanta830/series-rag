"""Compose retrieval, context, prompting, and generation into one answer path."""

from dataclasses import dataclass

from app.context.builder import ContextBuilder
from app.generation.base import GenerationProvider
from app.models import Chunk, SearchResult
from app.prompts.builder import PromptBuilder
from app.retrieval.retriever import SemanticRetriever


@dataclass(frozen=True, slots=True)
class RAGPipelineResult:
    """The inspectable intermediate values and final answer from one pipeline run."""

    answer: str
    context: str
    prompt: str
    search_results: tuple[SearchResult, ...]
    included_chunks: tuple[Chunk, ...]


class RAGPipeline:
    """Coordinate existing RAG components without changing their behavior."""

    def __init__(
        self,
        retriever: SemanticRetriever,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        generation_provider: GenerationProvider,
        top_k: int = 5,
    ) -> None:
        """Store injected components and one validated retrieval result limit."""
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        self._retriever = retriever
        self._context_builder = context_builder
        self._prompt_builder = prompt_builder
        self._generation_provider = generation_provider
        self._top_k = top_k

    def answer(self, question: str) -> RAGPipelineResult:
        """Run one complete source-grounded answer path for the original question."""
        search_results = self._retriever.retrieve(question, top_k=self._top_k)
        context_result = self._context_builder.build(search_results)

        if not context_result.text.strip() or not context_result.included_chunks:
            raise ValueError(
                "Retrieved context is empty; unable to generate a grounded answer."
            )

        prompt = self._prompt_builder.build(question, context_result.text)
        generated_answer = self._generation_provider.generate(prompt)

        return RAGPipelineResult(
            answer=generated_answer,
            context=context_result.text,
            prompt=prompt,
            search_results=tuple(search_results),
            included_chunks=context_result.included_chunks,
        )
