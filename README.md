# SeriesRAG

SeriesRAG is a learning-focused retrieval-augmented generation (RAG) project.
Its first phase deliberately builds semantic search from small, visible components
instead of delegating the core ideas to a framework.

The eventual application will ingest related transcripts, course notes, podcasts,
documentation, and other text sources, then retrieve the most relevant source
chunks for a question. The project now also has a small, locally testable
orchestration path that can pass retrieved context to a fake generation provider.
A small FastAPI adapter now exposes that development pipeline; it is not a
production chatbot or user interface.

## Phase 1 scope

Phase 1 implements this path:

```text
.txt, .md, native-text .pdf, and native .docx files
  → conservative normalization
  → fixed-size, word-based chunks
  → sentence-transformer embeddings
  → vector storage
  → ranked semantic retrieval
  → source-grounded chunks
  → context and prompt construction
  → generation provider
  → inspectable generated answer
```

The required components are:

- Python 3.12 and `pyproject.toml` project configuration
- a transparent chunker with default 200-word chunks and 40-word overlap
- deterministic document and chunk identifiers
- `sentence-transformers/all-MiniLM-L6-v2` embeddings
- an explicit NumPy in-memory cosine-similarity store
- a Qdrant-backed store run through Docker Compose
- Typer command-line interfaces and Rich result rendering
- unit tests with mocked embeddings, plus separate Qdrant integration tests

## Non-goals for Phase 1

- No production chatbot or user interface
- No OpenAI API, LangChain, or LlamaIndex
- No OCR, scanned/image-only PDF extraction, legacy `.doc`, or other Office formats
- No production frontend features, PostgreSQL, Redis, Celery, or Kubernetes
- No committed model files, Qdrant data, secrets, virtual environments, caches,
  or private source transcripts

## Development commands

Use the project virtual environment directly because the shell's `python` command
may resolve to a global Conda installation:

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy
.venv/bin/pytest
```

## Frontend foundation

The `frontend/` directory contains a responsive React and TypeScript application
shell built with Vite, React Router, and Tailwind CSS. It currently provides
navigation and placeholder pages at:

- `/`
- `/upload`
- `/chat`
- `/documents`

The upload page calls the SeriesRAG document-ingestion API, and the Documents
page reads the in-process catalog. Chat remains a placeholder for a later
milestone.

Start the FastAPI backend from the repository root:

```bash
.venv/bin/uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal, install the frontend packages and start Vite:

```bash
cd frontend
npm install
npm run dev
```

Open the local Vite URL, visit `/upload`, choose one supported file under 1 MB,
and submit it. During local development, Vite proxies `/documents` to FastAPI at
`http://localhost:8000`, so backend CORS configuration is not required.

After uploading, visit `/documents` to browse the catalog. Select a document to
inspect its chunks without leaving the page. PDF chunks show their source page;
DOCX chunks show available section and heading provenance. The empty catalog
links back to `/upload`, and transient list or detail failures can be retried.

`frontend/.env.example` documents `VITE_API_BASE_URL`. Leave it blank for the
local proxy workflow. Set it to a separately hosted API base URL only when that
API is configured to allow the frontend's origin; trailing slashes are removed
by the client.

Create a production build with:

```bash
cd frontend
npm run build
```

## Local RAG demonstration

Run the complete inspectable pipeline with a deterministic fake response:

```bash
.venv/bin/python scripts/rag_demo.py
```

The demo uses the local sentence-transformer cache and the in-memory vector
store when the model is available. Otherwise, it uses explicit deterministic
vectors for its fixed example content. It does not require Ollama or make a
generation network request.

## Query API

The development API accepts a question and retrieval limit:

```http
POST /query
Content-Type: application/json

{
  "question": "What is the shared responsibility model?",
  "top_k": 5
}
```

The response contains the answer, built context, completed prompt, and ordered
source chunks with their similarity scores. The default dependency uses a small
offline development corpus and deterministic fake generation; it is not
production configuration.

Upload one small UTF-8 text, Markdown, native-text PDF, or DOCX document for
in-process ingestion:

```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@sample-data/example.md"
```

Supported extensions are `.txt`, `.md`, `.markdown`, `.pdf`, and `.docx`;
uploads are limited to 1 MB. PDF text is extracted page by page, and each
resulting chunk keeps its one-based page number and original filename in
metadata. Chunking resets at page boundaries so a chunk never receives ambiguous
multi-page provenance.

Only PDFs containing extractable native text are supported. Scanned or
image-only PDFs require OCR and return HTTP 422; OCR is intentionally outside
this milestone. Uploaded documents and vectors remain in memory for the
application process and are not persisted.

DOCX extraction supports body headings, paragraphs, and tables in document
order. Tables become deterministic pipe-separated plain text. Images,
headers/footers, comments, tracked-change metadata, embedded files, legacy
`.doc`, and other Office formats are not extracted. DOCX sections preserve a
structural `section_type`; page numbers are never invented.

List uploaded documents in their original upload order:

```bash
curl http://localhost:8000/documents
```

Inspect one document and its source-ordered chunks:

```bash
curl http://localhost:8000/documents/{document_id}
```

Catalog responses contain document and chunk metadata but never expose embedding
vectors. The catalog is in memory and resets when the application process
restarts.

Delete an uploaded document and all of its stored chunk vectors:

```bash
curl -i -X DELETE http://localhost:8000/documents/{document_id}
```

A successful deletion returns HTTP 204 with no response body. An unknown or
already-deleted document returns HTTP 404. Deletion affects the in-process
catalog and vector store together; it does not delete source files from disk.

The ingestion and search commands will be added in later milestones:

```bash
python -m app.cli.ingest sample-data --collection aws-course
python -m app.cli.search "What is the shared responsibility model?" --collection aws-course
```
