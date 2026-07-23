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
