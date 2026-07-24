"""Overrideable FastAPI dependencies for application-level components."""

from app.context.builder import ContextBuilder
from app.embeddings.base import EmbeddingProvider
from app.generation.fake import FakeGenerationProvider
from app.models import Chunk
from app.pipeline.rag_pipeline import RAGPipeline
from app.prompts.builder import PromptBuilder
from app.retrieval.retriever import SemanticRetriever
from app.vector_stores.in_memory import InMemoryVectorStore


class _DevelopmentEmbeddingProvider(EmbeddingProvider):
    """Provide deterministic vectors for the small default development corpus."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one explicit topic vector for each development text."""
        if not texts:
            raise ValueError("Document texts must not be empty")
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return one explicit topic vector for a development query."""
        if not text.strip():
            raise ValueError("Query text must not be empty")
        return self._embed(text)

    def embedding_dimension(self) -> int:
        """Return the dimension of the development vectors."""
        return 2

    def _embed(self, text: str) -> list[float]:
        """Map cloud-provider and customer topics to two visible dimensions."""
        lowered_text = text.lower()
        provider_terms = ("aws", "cloud", "infrastructure", "data center")
        customer_terms = ("customer", "iam", "permission", "their data")
        provider_score = float(sum(term in lowered_text for term in provider_terms))
        customer_score = float(sum(term in lowered_text for term in customer_terms))

        if provider_score == 0.0 and customer_score == 0.0:
            return [1.0, 1.0]
        return [provider_score, customer_score]


def get_rag_pipeline() -> RAGPipeline:
    """Build a deterministic offline pipeline for the development API default."""
    chunks = [
        Chunk(
            chunk_id="development-provider-responsibility",
            document_id="development-shared-responsibility",
            source_name="development-shared-responsibility.md",
            text="AWS protects the physical cloud infrastructure and data centers.",
            chunk_index=0,
            start_word=0,
            end_word=9,
            metadata={"topic": "provider responsibility"},
        ),
        Chunk(
            chunk_id="development-customer-responsibility",
            document_id="development-shared-responsibility",
            source_name="development-shared-responsibility.md",
            text="Customers configure IAM permissions and protect their data.",
            chunk_index=1,
            start_word=9,
            end_word=17,
            metadata={"topic": "customer responsibility"},
        ),
    ]
    embedding_provider = _DevelopmentEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    vector_store.upsert(
        chunks,
        embedding_provider.embed_documents([chunk.text for chunk in chunks]),
    )

    return RAGPipeline(
        retriever=SemanticRetriever(embedding_provider, vector_store),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        generation_provider=FakeGenerationProvider(
            "This is a deterministic development answer."
        ),
    )
