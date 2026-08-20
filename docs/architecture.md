# SeriesRAG Architecture

SeriesRAG is intentionally a small, learning-first semantic retrieval system.
Each stage is visible in the code so that the path from source text to ranked
results can be inspected and tested without relying on a large framework.

## Current Architecture

Phase 1 establishes semantic search without an LLM. The intended data flow is:

```text
.txt / .md / native-text .pdf / native .docx source files
  → file loading
  → conservative text normalization
  → fixed-size word chunking
  → deterministic document and chunk IDs
  → embedding provider
  → vector store
  → cosine-similarity ranking
  → ranked source chunks in the CLI
```

The project foundation currently provides packaging, tooling, a minimal `app`
package, and test structure. Later Phase 1 milestones add each stage in the flow
individually, with unit tests before the Qdrant implementation is introduced.

Anticipated Phase 1 components:

- `app/models.py` for document, chunk, and retrieval-result dataclasses
- loading, normalization, and explicit word-chunking modules
- deterministic ID helpers based on canonical relative paths and normalized text
- an embedding-provider interface and a Sentence Transformers implementation
- an explicit NumPy in-memory vector store for learning cosine similarity
- a Qdrant adapter using Docker Compose
- thin ingestion and search orchestration modules
- separate Typer CLI modules for ingestion and search

The architecture stays simple because the goal is to understand the mechanics of
retrieval. Data models and small functions make intermediate results easy to
print, assert in tests, and reason about while debugging.

### Runtime configuration and dependency graph

`RuntimeSettings` reads and validates the process environment once, and
`build_runtime_state` constructs one explicit component graph:

```text
RuntimeSettings
  ├─ EmbeddingProvider
  ├─ VectorStore ───────────────┐
  ├─ InMemoryDocumentCatalog ───┼─ IngestionService
  │                             └─ DocumentDeletionService
  ├─ EmbeddingProvider + VectorStore → SemanticRetriever
  ├─ ContextBuilder
  ├─ PromptBuilder
  └─ GenerationProvider
       ↓
     RAGPipeline
```

FastAPI's dependency functions return components from one shared `RuntimeState`,
so uploads, queries, catalog reads, and deletions operate on the same objects.
They remain overrideable for offline HTTP tests. The default graph uses a small
deterministic embedding provider, `InMemoryVectorStore`, an in-memory catalog,
and `FakeGenerationProvider`. Explicit production-style settings replace the
embedding provider with Sentence Transformers, the vector store with
`QdrantVectorStore`, and generation with `OllamaGenerationProvider` as selected.
Unsupported or incomplete selections fail during configuration instead of
falling back silently.

Qdrant and Ollama clients are configured without readiness probes in this
runtime graph. The document catalog is still process-local even when Qdrant
stores vectors remotely, so this configuration does not yet provide durable
document-management metadata.

### Liveness and readiness

`GET /health` is a liveness signal: it proves the FastAPI process can answer a
request and deliberately does not resolve runtime dependencies or make network
calls. `GET /ready` is a readiness signal: a small evaluator inspects the shared
`RuntimeState` and reports the embedding, vector-store, and generation providers
individually. All required components produce HTTP 200; any unavailable required
component produces HTTP 503 with the same structured, public-safe body.

The in-memory vector store, deterministic development embeddings, and fake
generation have no external dependency, so validated configuration is sufficient
for them. Qdrant readiness delegates to its existing read-only collection-list
check and performs no writes. Ollama readiness calls `GET /api/tags` and requires
the configured model to be present; it neither generates text nor pulls a model.
Unexpected provider exceptions are contained at this boundary and appear only as
`ready: false`, without URLs, stack traces, or exception details in the response.

Sentence Transformer readiness is intentionally configuration-based. Calling
the readiness endpoint never asks the lazy provider to load a model, preventing
an operational probe from unexpectedly downloading a large artifact. In Qdrant
mode the model has already reported its dimension during runtime construction;
with in-memory storage, a later embedding operation remains the first model-load
boundary. This trade-off is documented rather than hidden behind an unsafe probe.

### FastAPI adapter

`app.api.main` exposes metadata at `/`, a process-level `/health` response,
OpenAPI documentation, and a working `/query` adapter around `RAGPipeline`.
Pydantic request and response models remain separate from internal immutable
domain models. The pipeline is obtained through an overrideable FastAPI
dependency backed by shared runtime state, allowing API tests to run entirely
offline without Sentence Transformers, Qdrant, or Ollama.

### React document adapters

The Vite frontend keeps HTTP details in `frontend/src/services` and public API
shapes in `frontend/src/types`. The upload page validates one selected file for
supported extension and the shared 1 MB limit, then delegates multipart request
construction and response parsing to `uploadDocument`. FastAPI remains the
authoritative validator; structured HTTP failures become stable, user-facing
messages rather than raw exception details.

The Documents page uses the same service boundary for `GET /documents` and
`GET /documents/{document_id}`. It keeps list selection and loading state local,
preserves catalog order, and renders chunk text without exposing embeddings.
Document-detail responses include the chunks' existing string metadata so PDF
page numbers and DOCX section provenance can cross the HTTP boundary. The UI
handles metadata as an open mapping because not every source type has the same
provenance fields.

Document deletion follows the same adapter boundary. The selected-document UI
asks for explicit confirmation, then `deleteDocument` sends
`DELETE /documents/{document_id}` and treats the empty HTTP 204 response as
success. The page removes only that document from local state and clears its
detail selection. A 404 also removes the stale local entry because the backend
has already reached the desired state; server and network failures leave the
document visible so the user can retry safely.

The Chat page delegates `POST /query` to a typed query service. A shared HTTP
client now owns the Vite API base URL, JSON parsing, network-error mapping, and
FastAPI error-detail extraction for both document and query services. The query
service validates the complete response shape before returning it to the page.
The page keeps the current question, retrieval limit, request state, and an
ordered `ConversationItem` list in local React state. Each completed item has a
client-generated identity, submitted question, plain-text answer, and its own
supporting sources. Successful requests append without replacing earlier
answers; failed requests leave the completed history unchanged. The
supporting-evidence list preserves backend order and exact retrieval scores.
Each source can be expanded locally to reveal its full chunk text, shortened
domain IDs, and all returned string provenance. `SourceResponse` includes a
backward-compatible copy of the chunk metadata, allowing PDF page numbers and
DOCX section details to reach the inspector without changing retrieval,
context-building, or generation behavior. Each conversation card owns its source
expansion and Clipboard API feedback state, so interactions remain isolated.

Conversation history is intentionally ephemeral: it is not sent to a
conversation API, written to local storage, or persisted by the backend. The
confirmed clear action empties only the frontend list and never deletes uploaded
documents. Embeddings remain private.

During `npm run dev`, Vite proxies `/documents` and `/query` to the local FastAPI
service on port 8000. This keeps local browser requests same-origin and avoids
adding CORS behavior to the backend. `VITE_API_BASE_URL` can replace the relative
base for environments that host an API separately and configure its allowed
origin.

### Document ingestion service

`IngestionService` owns the application-level write path for one existing
`Document`: conservative normalization, fixed-size chunking, document embedding,
and vector-store upsert. It delegates each operation to the existing component
and returns immutable statistics describing the completed ingestion. Documents
that normalize to empty text produce no chunks and no vector-store write.

### Document upload adapter

`POST /documents` accepts one bounded `.txt`, `.md`, `.markdown`, `.pdf`, or
`.docx` multipart upload and validates the filename extension independently of
its MIME type. Text formats retain their existing UTF-8 path. PDF and DOCX bytes
are passed to dedicated parsers before the resulting immutable `Document` is
delegated to `IngestionService`. The response exposes API-specific ingestion
statistics. The default ingestion and query dependencies share one explicitly
resettable in-memory development state, so uploaded content can be retrieved
later in the same application process without external services.

### Native-text PDF parser

`app.parsers.pdf` opens bounded in-memory PDF bytes with PyMuPDF and extracts
plain text one page at a time. It never invokes PyMuPDF's OCR facilities. Blank
pages are omitted from content while later pages keep their physical one-based
page numbers. Malformed, password-protected, unreadable, and no-text PDFs become
stable parser errors that the HTTP adapter maps to HTTP 422.

`Document.sections` is a backward-compatible immutable sequence for content with
more specific provenance. Each nonempty PDF page becomes one section. During
chunking, document metadata such as the original filename is merged with the
section's `page_number`. Chunking restarts at every section boundary, so chunks
never span pages. This can produce smaller chunks near page boundaries, but it
keeps page attribution explicit and avoids introducing a more complex span model.
The resulting chunks continue through the existing embedding, vector-storage,
retrieval, context, and generation components unchanged.

### Native DOCX parser

`app.parsers.docx` opens bounded in-memory DOCX packages with `python-docx` and
iterates body paragraphs and tables in document order. Nonempty headings and
paragraphs become separate `DocumentSection` values. Tables become one section
whose rows contain deterministic, pipe-separated cell text. This plain-text form
is inspectable and can enter the existing embedding pipeline without HTML or a
new rendering layer.

DOCX sections use `section_type` values of `heading`, `paragraph`, or `table`.
Heading sections also retain their Word heading style. Document-level metadata
supplies the original filename and `source_type`, which the chunker merges into
every chunk. DOCX has no reliable page model at this parsing boundary, so no page
number is invented. Section boundaries remain chunk boundaries, preserving
structural provenance at the cost of occasionally smaller chunks.

The parser intentionally ignores images, headers and footers, comments,
tracked-change metadata, embedded files, legacy `.doc`, and other Office formats.
Malformed packages and documents without supported usable text are rejected
before ingestion.

### Document catalog

The document catalog stores immutable document metadata and source-ordered chunks
after successful ingestion. `GET /documents` lists entries in upload order, and
`GET /documents/{document_id}` returns one entry with its chunks. These endpoints
never expose embeddings and do not inspect vector-store internals.

The catalog is deliberately separate from the `VectorStore` interface: vector
stores own similarity-search data, while the catalog owns document-management
metadata and upload ordering. The default development configuration uses an
in-memory catalog shared with `IngestionService`. A future durable catalog can
replace it without changing the in-memory or Qdrant retrieval implementations.

### Document deletion service

`DocumentDeletionService` coordinates `DELETE /documents/{document_id}` without
putting storage logic in the HTTP adapter. It removes the catalog entry and then
delegates removal of every matching chunk to the configured `VectorStore`.
Both the in-memory and Qdrant implementations expose the same idempotent
document-deletion operation. Qdrant selects points by the `document_id` stored
in each chunk payload.

For the in-memory development configuration, deletion is synchronous and
all-or-nothing. The service keeps the immutable catalog entry as a compensation
snapshot: if vector deletion raises, it records the document and chunks back
into the catalog before propagating the original error. A catalog failure occurs
before vector deletion, so vectors remain untouched. With a remote vector
database, a network failure can make the server-side outcome uncertain; the
catalog compensation is therefore best-effort rather than a distributed
transaction. Durable deployments would need a transactional outbox, retryable
operation log, or reconciliation process if that requirement appears.

### In-memory retrieval

The current in-memory vector store keeps each `Chunk` and its embedding in a
Python dictionary, keyed by `chunk_id`. For every query, it calculates cosine
similarity against every stored vector, sorts the scores, and returns the top
results. This is brute-force linear search: one query costs O(n) comparisons for
n stored vectors. It is intentionally easy to inspect, but it motivates the
vector database work in the next milestone.

### Qdrant collection setup

Qdrant runs as a separate local Docker service. The application uses
`qdrant-client` to connect to that service and create or inspect a collection.
A collection defines one vector space: its embedding dimension and cosine
distance must agree with the vectors the application will later write. Milestone
5B stores each `Chunk` as a Qdrant point. The embedding is the point vector, and
the chunk fields plus metadata are explicit payload values used to reconstruct an
immutable `Chunk` after searching. Qdrant persists these points outside the
Python process and performs nearest-neighbor search for query vectors. For cosine
collections, higher returned scores indicate closer semantic matches.

### Retriever orchestration

`SemanticRetriever` owns query-time orchestration. It validates and trims query
text, asks the configured embedding provider for one query embedding, and passes
that embedding to the configured vector store for nearest-neighbor search. It
returns the store's ranked source chunks unchanged; it does not generate answers
or contain storage-specific behavior.

### Retrieval demo and evaluation

The retrieval demo exercises the complete local path independently: document
embeddings, in-memory storage, the retriever, and ranked source chunks. The
lightweight evaluation utility measures whether expected chunk IDs appear at the
top of those rankings. Hit rate and reciprocal rank measure retrieval quality,
not answer quality; they help validate source retrieval before answer generation
is considered.

### Context builder

The context builder turns ranked `SearchResult` objects into readable,
source-attributed text for a later prompt-construction step. It preserves the
retriever's order, includes each `chunk_id` only once, and stops before adding a
fully formatted chunk that would exceed its character budget. It returns both
the context text and the immutable included chunks, so callers can inspect the
source metadata without parsing the formatted text. It does not build prompts
or call an LLM.

### Prompt builder

The prompt builder combines a validated user question and a previously built
context string into one deterministic, provider-neutral text prompt. Its
instructions require a future model to use only the supplied context, cite its
source labels, and state clearly when the context cannot support an answer. It
does not call an LLM or define provider-specific message objects.

### Generation interface

The generation interface accepts one completed prompt and returns plain text
without exposing an SDK, provider-specific chat-message type, or network
configuration. Its deterministic fake provider validates prompts, records them
per instance, and returns a configured fixed response. This makes a future
generation step testable without coupling the project to a real model provider.

### Ollama generation provider

`OllamaGenerationProvider` is the first real generation-provider implementation.
It sends a completed prompt to the non-streaming `/api/generate` endpoint of a
locally running Ollama service and returns only the generated text. Generation
remains behind the provider interface, so the pipeline can use it without
depending on Ollama-specific HTTP details.

### Complete RAG pipeline

`RAGPipeline` composes the existing components into one inspectable flow:

```text
question
  → SemanticRetriever
  → ranked SearchResult objects
  → ContextBuilder
  → PromptBuilder
  → GenerationProvider
  → RAGPipelineResult
```

It passes intermediate values through unchanged and returns the generated answer
alongside the built context, completed prompt, ranked results, and included
chunks. The pipeline contains no provider-specific logic, caching, retries, or
retrieval modifications. The local demo uses `FakeGenerationProvider`, so it
does not require Ollama. It prefers a cached Sentence Transformer model and uses
explicit deterministic vectors for its fixed examples when that model is not
available offline.

## Future Evolution

After Phase 1, future phases may evolve the project in this order:

1. **Retrieval quality:** add a small evaluation dataset and measurements for
   retrieval quality, then refine chunking and metadata deliberately.
2. **Collection management:** support clearer ingestion lifecycle operations,
   source updates, and reproducible collection metadata.
3. **Answer generation:** add a carefully scoped, source-grounded answer layer
   only after retrieval quality is understood.
4. **Additional source types:** consider OCR for scanned PDFs or additional
   Office formats only when a concrete need justifies their parsing complexity.
5. **User experience:** consider an API or interface only after the command-line
   workflow and retrieval behavior are reliable.

The retriever lets callers use either vector-store implementation through the
same query-time path while keeping embedding and storage decisions separate.

These are directions, not commitments. Any new layer should be introduced only
when it solves an observed requirement and preserves the project's inspectable,
learning-focused character.
