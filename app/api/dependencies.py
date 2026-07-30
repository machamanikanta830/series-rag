"""Overrideable FastAPI dependencies for application-level components."""

from dataclasses import dataclass

from app.context.builder import ContextBuilder
from app.document_catalog import DocumentCatalog, InMemoryDocumentCatalog
from app.embeddings.base import EmbeddingProvider
from app.generation.fake import FakeGenerationProvider
from app.models import Chunk
from app.pipeline.rag_pipeline import RAGPipeline
from app.prompts.builder import PromptBuilder
from app.retrieval.retriever import SemanticRetriever
from app.services.deletion import DocumentDeletionService
from app.services.ingestion import IngestionService
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


@dataclass(frozen=True, slots=True)
class _DevelopmentApplicationState:
    """Components that intentionally share one in-process vector store."""

    rag_pipeline: RAGPipeline
    ingestion_service: IngestionService
    document_catalog: DocumentCatalog
    deletion_service: DocumentDeletionService


def _build_development_application_state() -> _DevelopmentApplicationState:
    """Build deterministic offline query and ingestion dependencies."""
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
    document_catalog = InMemoryDocumentCatalog()
    vector_store.upsert(
        chunks,
        embedding_provider.embed_documents([chunk.text for chunk in chunks]),
    )

    rag_pipeline = RAGPipeline(
        retriever=SemanticRetriever(embedding_provider, vector_store),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        generation_provider=FakeGenerationProvider(
            "This is a deterministic development answer."
        ),
    )
    ingestion_service = IngestionService(
        embedding_provider,
        vector_store,
        document_catalog=document_catalog,
    )
    deletion_service = DocumentDeletionService(document_catalog, vector_store)
    return _DevelopmentApplicationState(
        rag_pipeline,
        ingestion_service,
        document_catalog,
        deletion_service,
    )


_development_application_state = _build_development_application_state()


def get_rag_pipeline() -> RAGPipeline:
    """Return the shared deterministic offline development query pipeline."""
    return _development_application_state.rag_pipeline


def get_ingestion_service() -> IngestionService:
    """Return the ingestion service backed by the shared development store."""
    return _development_application_state.ingestion_service


def get_document_catalog() -> DocumentCatalog:
    """Return the catalog shared with the development ingestion service."""
    return _development_application_state.document_catalog


def get_document_deletion_service() -> DocumentDeletionService:
    """Return the deletion service sharing the development catalog and vectors."""
    return _development_application_state.deletion_service


def reset_development_application_state() -> None:
    """Replace development state so tests cannot leak uploaded documents."""
    global _development_application_state
    _development_application_state = _build_development_application_state()
