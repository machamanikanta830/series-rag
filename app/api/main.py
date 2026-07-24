"""FastAPI application and foundation routes for SeriesRAG."""

from fastapi import FastAPI, HTTPException, status

from app.api.models import ApiMetadataResponse, HealthResponse, QueryRequest

API_TITLE = "SeriesRAG API"
API_DESCRIPTION = "A learning-focused, source-grounded RAG API."
API_VERSION = "0.1.0"

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)


@app.get("/", response_model=ApiMetadataResponse)
def read_root() -> ApiMetadataResponse:
    """Return basic API metadata without invoking RAG components."""
    return ApiMetadataResponse(
        name=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
    )


@app.get("/health", response_model=HealthResponse)
def read_health() -> HealthResponse:
    """Return a process-level health response with no dependency checks yet."""
    return HealthResponse(status="ok")


@app.post("/query")
def query_rag(request: QueryRequest) -> None:
    """Validate a future RAG request until the pipeline is connected."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "Query endpoint is not implemented yet; it will be connected to the "
            "RAG service in the next milestone."
        ),
    )
