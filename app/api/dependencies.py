"""Overrideable FastAPI dependencies backed by one configured runtime graph."""

from app.config import RuntimeSettings
from app.document_catalog import DocumentCatalog
from app.pipeline.rag_pipeline import RAGPipeline
from app.runtime import RuntimeState, build_runtime_state
from app.services.deletion import DocumentDeletionService
from app.services.ingestion import IngestionService

_runtime_state = build_runtime_state(RuntimeSettings.from_environment())


def get_runtime_state() -> RuntimeState:
    """Return the one configured component graph shared by this process."""
    return _runtime_state


def get_rag_pipeline() -> RAGPipeline:
    """Return the configured query pipeline from shared runtime state."""
    return _runtime_state.rag_pipeline


def get_ingestion_service() -> IngestionService:
    """Return the ingestion service sharing the configured vector store."""
    return _runtime_state.ingestion_service


def get_document_catalog() -> DocumentCatalog:
    """Return the configured in-process document catalog."""
    return _runtime_state.document_catalog


def get_document_deletion_service() -> DocumentDeletionService:
    """Return the deletion service sharing the catalog and vector store."""
    return _runtime_state.deletion_service


def reset_runtime_state(settings: RuntimeSettings | None = None) -> None:
    """Rebuild shared state from explicit settings or the process environment."""
    global _runtime_state
    configured_settings = settings or RuntimeSettings.from_environment()
    _runtime_state = build_runtime_state(configured_settings)


def reset_development_application_state() -> None:
    """Restore deterministic offline defaults so tests cannot leak state."""
    reset_runtime_state(RuntimeSettings())
