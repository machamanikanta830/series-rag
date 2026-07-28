# SeriesRAG Learning Journal

Use one copy of this template for each meaningful learning session or milestone.
Keep entries short and concrete; the goal is to capture understanding and open
questions, not to produce formal documentation.

## Entry Template

- **Date:**
- **Milestone:**

### Concepts learned

-

### Questions

-

### Future ideas

-

### Lessons from debugging

-
## Milestone 3

### Learned

- Embeddings are dense numerical representations.
- Similar meaning produces vectors with similar directions.
- Cosine similarity measures vector orientation.
- Normalized embeddings make cosine similarity equivalent to the dot product.

### Questions

- Why are embeddings 384 dimensions?
- How does the model learn semantic space?
- How do vector databases search millions of vectors efficiently?

## Milestone 4

### Learned

- Brute-force retrieval compares a query vector with every stored vector.
- Cosine similarity scores allow chunks to be ranked by semantic direction.
- Deterministic tie-breaking makes equal-score results reproducible.

### Questions

- How do approximate nearest-neighbor indexes avoid checking every vector?
- What trade-offs does a vector database make between speed and recall?

## Milestone 5A

### Learned

- Qdrant runs independently from the Python process as a local Docker service.
- A collection fixes the vector dimension and distance metric for stored vectors.
- Collection setup can be validated before adding vector writes or retrieval.

### Questions

- How does Qdrant index vectors after they are written?
- How should collection lifecycle work when source documents are re-ingested?

## Milestone 5B

### Learned

- A Qdrant point combines an identifier, an embedding vector, and a payload.
- Persisted payloads can reconstruct immutable application-level chunks.
- Cosine search returns higher scores for more closely aligned vectors.

### Questions

- How does Qdrant handle approximate nearest-neighbor recall at larger scales?
- How should collection contents be updated when a source document changes?

## Milestone 6A

### Learned

- A retriever coordinates query embedding and vector search without owning either.
- Query validation belongs at the caller-facing orchestration boundary.
- Returning store results unchanged preserves each store's ranking semantics.

### Questions

- Where should ingestion orchestration connect document chunks to a vector store?
- How should retrieval behavior be measured before generating answers?

## Milestone 6B

### Learned

- Hit rate checks whether an expected chunk appears within a chosen rank limit.
- Reciprocal rank rewards relevant chunks that appear earlier in a ranking.
- Retrieval quality can be measured before any answer-generation layer exists.

### Questions

- How can a small evaluation set represent the variety of real user questions?
- Which retrieval failures should lead to changes in chunking versus embeddings?

## Milestone 7B

### Learned

- A prompt can make grounding requirements explicit before a model is involved.
- Stable prompt formatting makes behavior easier to inspect and test.
- Source labels in retrieval context give a future answer layer concrete citation targets.

### Questions

- How should a future answer layer validate citations against included chunks?
- When should prompt instructions evolve based on observed retrieval failures?

## Milestone 8B

### Learned

- A provider adapter can isolate HTTP request and response details from the rest
  of an application.
- Non-streaming Ollama generation uses one JSON request containing a model,
  prompt, and `stream: false`.
- Network and response failures need clear application-level error messages.

### Questions

- How should a future answer layer select a local model for different tasks?
- Which provider behaviors should be tested through an optional local smoke test?

## Milestone 8C

### Learned

- A thin orchestration layer can connect RAG stages without duplicating their
  validation, ranking, or provider behavior.
- Returning intermediate values makes a RAG run easier to inspect and debug.
- A deterministic fake provider allows the complete pipeline to run locally
  without a model service.

### Questions

- Which pipeline results should a future user-facing interface display?
- How should future answer evaluation use the retrieved context and citations?

## Milestone 9B

### Learned

- FastAPI acts as an adapter that translates HTTP models to and from application
  pipeline values.
- Dependency overrides keep API tests independent of model and database services.
- Request-specific retrieval limits can be passed without mutating shared pipeline
  configuration.

### Questions

- Which application failures should become stable public API error codes?
- When should the development dependency be replaced with production wiring?

## Milestone 10A

### Learned

- A thin ingestion service can coordinate normalization, chunking, embeddings,
  and storage without reimplementing those operations.
- Deterministic chunk IDs make repeated vector-store upserts idempotent.
- Empty normalized documents should avoid unnecessary embedding and storage work.

### Questions

- How should ingestion report partial progress when processing many documents?
- When should a source update remove chunks that no longer exist?

## Milestone 10B

### Learned

- Multipart uploads require bounded reads before decoding content in memory.
- Filename extensions and UTF-8 content need independent validation at the HTTP
  boundary.
- Query and ingestion dependencies must reference the same vector store for
  uploaded content to become immediately retrievable.

### Questions

- How should application state move from in-memory development storage to a
  durable deployment configuration?
- Which document-level metadata will future filtering and source management need?
