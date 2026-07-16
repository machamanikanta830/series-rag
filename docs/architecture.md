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

These are directions, not commitments. Any new layer should be introduced only
when it solves an observed requirement and preserves the project's inspectable,
learning-focused character.
