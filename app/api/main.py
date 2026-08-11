"""FastAPI application and HTTP adapters for SeriesRAG."""

from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import Response

from app.api.dependencies import (
    get_document_catalog,
    get_document_deletion_service,
    get_ingestion_service,
    get_rag_pipeline,
)
from app.api.models import (
    ApiMetadataResponse,
    DocumentChunkResponse,
    DocumentDetailResponse,
    DocumentIngestionResponse,
    DocumentSummaryResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceResponse,
)
from app.document_catalog import CatalogDocument, DocumentCatalog
from app.identifiers import create_document_id
from app.models import Document
from app.normalization import normalize_text
from app.parsers.docx import (
    DOCXNoExtractableTextError,
    DOCXParsingError,
    parse_docx,
)
from app.parsers.pdf import (
    PDFNoExtractableTextError,
    PDFParsingError,
    parse_pdf,
)
from app.pipeline.rag_pipeline import RAGPipeline, RAGPipelineResult
from app.services.deletion import DocumentDeletionService
from app.services.ingestion import IngestionService

API_TITLE = "SeriesRAG API"
API_DESCRIPTION = "A learning-focused, source-grounded RAG API."
API_VERSION = "0.1.0"
MAX_UPLOAD_BYTES = 1_048_576
_SUPPORTED_UPLOAD_EXTENSIONS = frozenset({".txt", ".md", ".markdown", ".pdf", ".docx"})

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


@app.post(
    "/documents",
    response_model=DocumentIngestionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: Annotated[UploadFile, File(...)],
    ingestion_service: Annotated[
        IngestionService,
        Depends(get_ingestion_service),
    ],
) -> DocumentIngestionResponse:
    """Validate and synchronously ingest one bounded supported document."""
    filename = _validate_upload_filename(file.filename)
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="Uploaded file exceeds the 1 MB size limit.",
        )

    document = _parse_uploaded_document(filename, content)
    try:
        statistics = ingestion_service.ingest(document)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Uploaded document failed ingestion validation.",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The uploaded document could not be ingested.",
        ) from error

    return DocumentIngestionResponse(
        document_id=statistics.document_id,
        filename=filename,
        chunks_created=statistics.chunks_created,
        embedding_dimension=statistics.embedding_dimension,
        vector_store_name=statistics.vector_store_name,
    )


@app.get("/documents", response_model=list[DocumentSummaryResponse])
def list_documents(
    document_catalog: Annotated[
        DocumentCatalog,
        Depends(get_document_catalog),
    ],
) -> list[DocumentSummaryResponse]:
    """Return ingested document metadata in original upload order."""
    try:
        catalog_documents = document_catalog.list_documents()
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The document catalog could not be read.",
        ) from error

    return [_to_document_summary(entry) for entry in catalog_documents]


@app.get("/documents/{document_id}", response_model=DocumentDetailResponse)
def get_document(
    document_id: str,
    document_catalog: Annotated[
        DocumentCatalog,
        Depends(get_document_catalog),
    ],
) -> DocumentDetailResponse:
    """Return one ingested document and its chunks without vector data."""
    try:
        catalog_document = document_catalog.get_document(document_id)
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The document catalog could not be read.",
        ) from error

    if catalog_document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )
    return _to_document_detail(catalog_document)


@app.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_document(
    document_id: str,
    deletion_service: Annotated[
        DocumentDeletionService,
        Depends(get_document_deletion_service),
    ],
) -> Response:
    """Delete one cataloged document and all of its stored vectors."""
    try:
        document_existed = deletion_service.delete(document_id)
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The document could not be deleted.",
        ) from error

    if not document_existed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


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


def _validate_upload_filename(uploaded_filename: str | None) -> str:
    """Return a safe basename with an explicitly supported extension."""
    if uploaded_filename is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Uploaded file must include a usable filename.",
        )

    filename = uploaded_filename.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    filename = filename.strip()
    if not filename or filename in {".", ".."}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Uploaded file must include a usable filename.",
        )
    if Path(filename).suffix.lower() not in _SUPPORTED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only .txt, .md, .markdown, .pdf, and .docx files are supported.",
        )
    return filename


def _parse_uploaded_document(filename: str, content: bytes) -> Document:
    """Create one document through the parser appropriate to its extension."""
    extension = Path(filename).suffix.lower()
    if extension == ".pdf":
        try:
            return parse_pdf(content, filename)
        except PDFNoExtractableTextError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    "Uploaded PDF contains no extractable text. "
                    "Scanned or image-only PDFs are not supported."
                ),
            ) from error
        except PDFParsingError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Uploaded PDF is malformed, password-protected, or unreadable.",
            ) from error

    if extension == ".docx":
        try:
            return parse_docx(content, filename)
        except DOCXNoExtractableTextError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Uploaded DOCX contains no extractable text.",
            ) from error
        except DOCXParsingError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Uploaded DOCX is malformed or unreadable.",
            ) from error

    try:
        decoded_text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Uploaded file must contain valid UTF-8 text.",
        ) from error

    normalized_text = normalize_text(decoded_text)
    if not normalized_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Uploaded file must contain non-empty text.",
        )

    return Document(
        document_id=create_document_id(filename, normalized_text),
        source_name=filename,
        source_path=filename,
        text=decoded_text,
        metadata={"filename": filename},
    )


def _to_document_summary(entry: CatalogDocument) -> DocumentSummaryResponse:
    """Serialize one catalog entry without exposing its chunks or vectors."""
    return DocumentSummaryResponse(
        document_id=entry.document.document_id,
        filename=entry.document.source_name,
        chunk_count=len(entry.chunks),
    )


def _to_document_detail(entry: CatalogDocument) -> DocumentDetailResponse:
    """Serialize one catalog entry and its already ordered chunks."""
    return DocumentDetailResponse(
        **_to_document_summary(entry).model_dump(),
        chunks=[
            DocumentChunkResponse(
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
            )
            for chunk in entry.chunks
        ],
    )


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
