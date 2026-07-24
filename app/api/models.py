"""Pydantic models for the public SeriesRAG HTTP API."""

from pydantic import BaseModel, Field, field_validator


class ApiMetadataResponse(BaseModel):
    """Basic identifying information returned by the root endpoint."""

    name: str
    version: str
    description: str


class HealthResponse(BaseModel):
    """Simple health status returned without checking dependencies."""

    status: str


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


class QueryResponse(BaseModel):
    """A generated answer and its inspectable grounding information."""

    answer: str
    context: str
    prompt: str
    sources: list[SourceResponse]
