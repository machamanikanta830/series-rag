"""Demonstrate the complete RAG path using local embeddings and a fake response."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.context.builder import ContextBuilder  # noqa: E402
from app.embeddings.base import EmbeddingProvider  # noqa: E402
from app.embeddings.sentence_transformers import (  # noqa: E402
    SentenceTransformerEmbeddingProvider,
)
from app.generation.fake import FakeGenerationProvider  # noqa: E402
from app.models import Chunk  # noqa: E402
from app.pipeline.rag_pipeline import RAGPipeline  # noqa: E402
from app.prompts.builder import PromptBuilder  # noqa: E402
from app.retrieval.retriever import SemanticRetriever  # noqa: E402
from app.vector_stores.in_memory import InMemoryVectorStore  # noqa: E402


class OfflineDemoEmbeddingProvider(EmbeddingProvider):
    """Embed this demo's fixed examples without a model download or network call."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one deterministic unit vector for each known demo chunk."""
        return [self._embedding_for(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return a deterministic unit vector for the known demo question."""
        return self._embedding_for(text)

    def embedding_dimension(self) -> int:
        """Report the explicit dimensionality of the demo-only vectors."""
        return 3

    def _embedding_for(self, text: str) -> list[float]:
        """Map each example's topic to a visibly related unit vector."""
        lowered_text = text.lower()
        if "infrastructure" in lowered_text or "data centers" in lowered_text:
            return [1.0, 0.0, 0.0]
        if "iam" in lowered_text or "identity" in lowered_text:
            return [0.0, 1.0, 0.0]
        if "banana" in lowered_text or "potassium" in lowered_text:
            return [0.0, 0.0, 1.0]

        raise ValueError("Offline demo embedding is unavailable for this text")


def _embed_demo_chunks(
    chunks: list[Chunk],
) -> tuple[EmbeddingProvider, list[list[float]]]:
    """Prefer cached Sentence Transformers embeddings, with an offline fallback."""
    provider = SentenceTransformerEmbeddingProvider()

    try:
        return provider, provider.embed_documents([chunk.text for chunk in chunks])
    except RuntimeError:
        fallback_provider = OfflineDemoEmbeddingProvider()
        return fallback_provider, fallback_provider.embed_documents(
            [chunk.text for chunk in chunks]
        )


def main() -> None:
    """Run retrieval, context construction, prompting, and fake generation."""
    chunks = [
        Chunk(
            chunk_id="provider-infrastructure",
            document_id="cloud-course",
            source_name="shared-responsibility.md",
            text="AWS protects the physical cloud infrastructure and data centers.",
            chunk_index=0,
            start_word=0,
            end_word=9,
            metadata={"topic": "provider responsibility"},
        ),
        Chunk(
            chunk_id="customer-configuration",
            document_id="cloud-course",
            source_name="shared-responsibility.md",
            text="Customers configure IAM permissions and protect their data.",
            chunk_index=1,
            start_word=9,
            end_word=17,
            metadata={"topic": "customer responsibility"},
        ),
        Chunk(
            chunk_id="fruit-nutrition",
            document_id="nutrition-course",
            source_name="fruit.md",
            text="Bananas contain potassium and other nutrients.",
            chunk_index=0,
            start_word=0,
            end_word=6,
            metadata={"topic": "nutrition"},
        ),
    ]
    question = "Who protects the physical cloud infrastructure?"

    embedding_provider, document_embeddings = _embed_demo_chunks(chunks)
    vector_store = InMemoryVectorStore()
    vector_store.upsert(chunks, document_embeddings)
    pipeline = RAGPipeline(
        retriever=SemanticRetriever(embedding_provider, vector_store),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        generation_provider=FakeGenerationProvider(
            "AWS protects the physical cloud infrastructure and data centers "
            "[Source: shared-responsibility.md | Chunk 0]."
        ),
        top_k=2,
    )

    result = pipeline.answer(question)

    print(f"Embedding provider: {embedding_provider.__class__.__name__}")
    print("Question")
    print("--------")
    print(question)
    print("\nRanked retrieved chunks")
    print("-----------------------")
    for rank, search_result in enumerate(result.search_results, start=1):
        chunk = search_result.chunk
        print(f"{rank}. score={search_result.score:.3f}")
        print(f"   source={chunk.source_name}, chunk_index={chunk.chunk_index}")
        print(f"   text={chunk.text}")
    print("\nRetrieved context")
    print("-----------------")
    print(result.context)
    print("\nCompleted prompt")
    print("----------------")
    print(result.prompt)
    print("\nGenerated answer")
    print("----------------")
    print(result.answer)


if __name__ == "__main__":
    main()
