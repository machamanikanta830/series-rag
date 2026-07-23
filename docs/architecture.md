# SeriesRAG Architecture

SeriesRAG is intentionally a small, learning-first semantic retrieval system.
Each stage is visible in the code so that the path from source text to ranked
results can be inspected and tested without relying on a large framework.

## Current Architecture

Phase 1 establishes semantic search without an LLM. The intended data flow is:

```text
.txt / .md source files
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
4. **Additional source types:** consider formats such as PDF or DOCX only when a
   concrete need justifies their parsing complexity.
5. **User experience:** consider an API or interface only after the command-line
   workflow and retrieval behavior are reliable.

The retriever lets callers use either vector-store implementation through the
same query-time path while keeping embedding and storage decisions separate.

These are directions, not commitments. Any new layer should be introduced only
when it solves an observed requirement and preserves the project's inspectable,
learning-focused character.
