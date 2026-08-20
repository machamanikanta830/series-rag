"""Pydantic models for the public SeriesRAG HTTP API."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ApiMetadataResponse(BaseModel):
    """Basic identifying information returned by the root endpoint."""

    name: str
    version: str
    description: str


class HealthResponse(BaseModel):
    """Simple liveness status returned without checking dependencies."""

    status: str


class ReadinessComponentResponse(BaseModel):
    """Public readiness state for one configured runtime dependency."""

    provider: str
    ready: bool


class ReadinessComponentsResponse(BaseModel):
    """Configured dependency states returned by the readiness endpoint."""

    embedding: ReadinessComponentResponse
    vector_store: ReadinessComponentResponse
    generation: ReadinessComponentResponse


class ReadinessResponse(BaseModel):
    """Overall application readiness and its component-level evidence."""

    status: Literal["ready", "not_ready"]
    components: ReadinessComponentsResponse


class QueryRequest(BaseModel):
    """A future RAG query request with an explicit retrieval result limit."""

    question: str
    top_k: int = Field(default=5, gt=0)

    @field_validator("question")
    @classmethod
    def validate_question(cls, question: str) -> str:
        """Reject questions that contain only whitespace."""
        if not question.strip():
            raise ValueError("question must not be empty")
        return question


class SourceResponse(BaseModel):
    """One source chunk included in a generated answer's context."""

    chunk_id: str
    document_id: str
    source_name: str
    chunk_index: int
    text: str
    score: float
    metadata: dict[str, str]


class QueryResponse(BaseModel):
    """A generated answer and its inspectable grounding information."""

    answer: str
    context: str
    prompt: str
    sources: list[SourceResponse]


class DocumentIngestionResponse(BaseModel):
    """Public summary of one successfully ingested uploaded document."""

    document_id: str
    filename: str
    chunks_created: int
    embedding_dimension: int
    vector_store_name: str


class DocumentSummaryResponse(BaseModel):
    """Public metadata for one ingested document."""

    document_id: str
    filename: str
    chunk_count: int


class DocumentChunkResponse(BaseModel):
    """One source chunk exposed without its embedding vector."""

    chunk_id: str
    chunk_index: int
    text: str
    metadata: dict[str, str]


class DocumentDetailResponse(DocumentSummaryResponse):
    """One document summary and its chunks in source order."""

    chunks: list[DocumentChunkResponse]
