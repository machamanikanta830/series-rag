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
.txt and .md files
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
- No PDF or DOCX ingestion
- No React, PostgreSQL, Redis, Celery, or Kubernetes
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

The ingestion and search commands will be added in later milestones:

```bash
python -m app.cli.ingest sample-data --collection aws-course
python -m app.cli.search "What is the shared responsibility model?" --collection aws-course
```
