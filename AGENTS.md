# SeriesRAG Development Guide

## Project Goal

SeriesRAG is a learning-focused Retrieval-Augmented Generation (RAG) project.
The primary goal is to understand document ingestion, chunking, embeddings, vector
databases, semantic retrieval, and evaluation. The goal is **not** to build a
chatbot quickly.

## Engineering Principles

- Prefer explicit implementations.
- Prioritize inspectability and learning.
- Avoid unnecessary abstractions.
- Introduce complexity only when a real requirement appears.

## Do Not Introduce Prematurely

Unless explicitly requested, do not introduce:

- LangChain or LlamaIndex
- FastAPI or React
- PostgreSQL, Redis, Celery, or Kubernetes
- Authentication
- Repository patterns, factory patterns, or dependency injection
- OpenAI APIs
- PDF parsing
- Evaluation frameworks

## Development Workflow

For every milestone:

1. Inspect existing files.
2. Modify only the files required for that milestone.
3. Explain the implementation and the concept it demonstrates.
4. Run Ruff, Mypy, and Pytest with the project virtual environment:

   ```bash
   .venv/bin/ruff check .
   .venv/bin/mypy
   .venv/bin/pytest
   ```

5. Explain the execution flow.
6. Suggest a commit message.
7. Stop before beginning the next milestone.

Never create commits unless explicitly instructed.

## Code Style

- Use Python 3.12.
- Add type hints everywhere.
- Prefer dataclasses for structured data.
- Prefer small, focused functions.
- Avoid hidden magic.
- Avoid broad exception handling; catch only errors that can be handled
  meaningfully.

## Testing

- Prefer unit tests.
- Mock embeddings where possible.
- Keep Qdrant tests isolated as integration tests.
- Do not require network access for unit tests.
