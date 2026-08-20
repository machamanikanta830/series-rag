"""Construct one explicit, shared graph of configured application components."""

from dataclasses import dataclass

from qdrant_client import QdrantClient

from app.config import (
    EmbeddingProviderKind,
    GenerationProviderKind,
    RuntimeSettings,
    VectorStoreKind,
)
from app.context.builder import ContextBuilder
from app.document_catalog import DocumentCatalog, InMemoryDocumentCatalog
from app.embeddings.base import EmbeddingProvider
from app.embeddings.sentence_transformers import SentenceTransformerEmbeddingProvider
from app.generation.base import GenerationProvider
from app.generation.fake import FakeGenerationProvider
from app.generation.ollama import OllamaGenerationProvider
from app.models import Chunk
from app.pipeline.rag_pipeline import RAGPipeline
from app.prompts.builder import PromptBuilder
from app.retrieval.retriever import SemanticRetriever
from app.services.deletion import DocumentDeletionService
from app.services.ingestion import IngestionService
from app.vector_stores.base import VectorStore
from app.vector_stores.in_memory import InMemoryVectorStore
from app.vector_stores.qdrant import QdrantVectorStore

_DEVELOPMENT_ANSWER = "This is a deterministic development answer."


class DevelopmentEmbeddingProvider(EmbeddingProvider):
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


@dataclass(frozen=True, slots=True)
class RuntimeState:
    """Every application component that must share configured runtime state."""

    settings: RuntimeSettings
    embedding_provider: EmbeddingProvider
    vector_store: VectorStore
    document_catalog: DocumentCatalog
    ingestion_service: IngestionService
    deletion_service: DocumentDeletionService
    retriever: SemanticRetriever
    context_builder: ContextBuilder
    prompt_builder: PromptBuilder
    generation_provider: GenerationProvider
    rag_pipeline: RAGPipeline


def build_runtime_state(settings: RuntimeSettings) -> RuntimeState:
    """Build one component graph without silently replacing requested providers."""
    embedding_provider = _build_embedding_provider(settings)
    vector_store = _build_vector_store(settings, embedding_provider)
    document_catalog = InMemoryDocumentCatalog()

    if settings.embedding_provider is EmbeddingProviderKind.DEVELOPMENT:
        _seed_development_vectors(vector_store, embedding_provider)

    ingestion_service = IngestionService(
        embedding_provider,
        vector_store,
        document_catalog=document_catalog,
    )
    deletion_service = DocumentDeletionService(document_catalog, vector_store)
    retriever = SemanticRetriever(embedding_provider, vector_store)
    context_builder = ContextBuilder(max_characters=settings.context_max_characters)
    prompt_builder = PromptBuilder()
    generation_provider = _build_generation_provider(settings)
    rag_pipeline = RAGPipeline(
        retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        generation_provider=generation_provider,
        top_k=settings.default_top_k,
    )

    return RuntimeState(
        settings=settings,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        document_catalog=document_catalog,
        ingestion_service=ingestion_service,
        deletion_service=deletion_service,
        retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        generation_provider=generation_provider,
        rag_pipeline=rag_pipeline,
    )


def _build_embedding_provider(settings: RuntimeSettings) -> EmbeddingProvider:
    """Return exactly the configured embedding implementation."""
    if settings.embedding_provider is EmbeddingProviderKind.DEVELOPMENT:
        return DevelopmentEmbeddingProvider()
    return SentenceTransformerEmbeddingProvider(settings.embedding_model)


def _build_vector_store(
    settings: RuntimeSettings,
    embedding_provider: EmbeddingProvider,
) -> VectorStore:
    """Return exactly the configured shared vector-store implementation."""
    if settings.vector_store is VectorStoreKind.IN_MEMORY:
        return InMemoryVectorStore()

    client = QdrantClient(url=settings.qdrant_url)
    return QdrantVectorStore(
        collection_name=settings.qdrant_collection,
        embedding_dimension=embedding_provider.embedding_dimension(),
        client=client,
    )


def _build_generation_provider(settings: RuntimeSettings) -> GenerationProvider:
    """Return exactly the configured generation implementation."""
    if settings.generation_provider is GenerationProviderKind.FAKE:
        return FakeGenerationProvider(_DEVELOPMENT_ANSWER)
    return OllamaGenerationProvider(
        model=settings.ollama_model,
        base_url=settings.ollama_url,
    )


def _seed_development_vectors(
    vector_store: VectorStore,
    embedding_provider: EmbeddingProvider,
) -> None:
    """Keep the existing transparent offline corpus in development mode."""
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
    vector_store.upsert(
        chunks,
        embedding_provider.embed_documents([chunk.text for chunk in chunks]),
    )
