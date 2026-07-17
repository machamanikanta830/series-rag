"""Demonstrate transparent in-memory semantic search over a few chunks."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.embeddings.sentence_transformers import (  # noqa: E402
    SentenceTransformerEmbeddingProvider,
)
from app.models import Chunk  # noqa: E402
from app.vector_stores.in_memory import InMemoryVectorStore  # noqa: E402


def main() -> None:
    """Embed example chunks, search them, and print the ranked source context."""
    chunks = [
        Chunk(
            chunk_id="aws-security",
            document_id="aws-course",
            source_name="cloud-security.md",
            text="AWS protects the physical infrastructure and data centers.",
            chunk_index=0,
            start_word=0,
            end_word=8,
            metadata={"topic": "shared responsibility"},
        ),
        Chunk(
            chunk_id="customer-security",
            document_id="aws-course",
            source_name="cloud-security.md",
            text="Customers configure identity permissions and protect their data.",
            chunk_index=1,
            start_word=8,
            end_word=16,
            metadata={"topic": "shared responsibility"},
        ),
        Chunk(
            chunk_id="fruit",
            document_id="nutrition",
            source_name="fruit.md",
            text="Bananas contain potassium and other nutrients.",
            chunk_index=0,
            start_word=0,
            end_word=6,
            metadata={"topic": "nutrition"},
        ),
    ]
    query = "Who secures cloud data centers?"

    provider = SentenceTransformerEmbeddingProvider()
    store = InMemoryVectorStore()
    store.upsert(chunks, provider.embed_documents([chunk.text for chunk in chunks]))
    results = store.search(provider.embed_query(query), top_k=3)

    print(f"Query: {query}\n")
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        print(f"{rank}. score={result.score:.3f}")
        print(f"   source={chunk.source_name}, chunk_index={chunk.chunk_index}")
        print(f"   text={chunk.text}")


if __name__ == "__main__":
    main()
