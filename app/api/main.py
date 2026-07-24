"""FastAPI application and HTTP adapters for SeriesRAG."""

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status

from app.api.dependencies import get_rag_pipeline
from app.api.models import (
    ApiMetadataResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceResponse,
)
from app.pipeline.rag_pipeline import RAGPipeline, RAGPipelineResult

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


@app.post("/query", response_model=QueryResponse)
def query_rag(
    request: QueryRequest,
    pipeline: Annotated[RAGPipeline, Depends(get_rag_pipeline)],
) -> QueryResponse:
    """Run one validated question through the configured RAG pipeline."""
    try:
        result = pipeline.answer(request.question, top_k=request.top_k)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="No usable context was found for this question.",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The query could not be completed.",
        ) from error

    return _to_query_response(result)


def _to_query_response(result: RAGPipelineResult) -> QueryResponse:
    """Serialize immutable domain results into public API response models."""
    included_chunk_ids = {chunk.chunk_id for chunk in result.included_chunks}
    serialized_chunk_ids: set[str] = set()
    sources: list[SourceResponse] = []

    for search_result in result.search_results:
        chunk = search_result.chunk
        if (
            chunk.chunk_id not in included_chunk_ids
            or chunk.chunk_id in serialized_chunk_ids
        ):
            continue

        sources.append(
            SourceResponse(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                source_name=chunk.source_name,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                score=search_result.score,
            )
        )
        serialized_chunk_ids.add(chunk.chunk_id)

    return QueryResponse(
        answer=result.answer,
        context=result.context,
        prompt=result.prompt,
        sources=sources,
    )
