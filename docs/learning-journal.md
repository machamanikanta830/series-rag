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
