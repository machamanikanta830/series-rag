# SeriesRAG

SeriesRAG is a learning-focused retrieval-augmented generation (RAG) project.
Its first phase deliberately builds semantic search from small, visible components
instead of delegating the core ideas to a framework.

The eventual application will ingest related transcripts, course notes, podcasts,
documentation, and other text sources, then retrieve the most relevant source
chunks for a question. Phase 1 stops at retrieval: it does not generate answers.

## Phase 1 scope

Phase 1 implements this path:

```text
.txt and .md files
  → conservative normalization
  → fixed-size, word-based chunks
  → sentence-transformer embeddings
  → vector storage
  → ranked semantic retrieval
  → source-grounded chunks
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

- No chatbot or generated-answer layer
- No LLM, OpenAI API, LangChain, or LlamaIndex
- No PDF or DOCX ingestion
- No FastAPI, React, PostgreSQL, Redis, Celery, or Kubernetes
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

The ingestion and search commands will be added in later milestones:

```bash
python -m app.cli.ingest sample-data --collection aws-course
python -m app.cli.search "What is the shared responsibility model?" --collection aws-course
```
