"""Demonstrate complete semantic retrieval with in-memory vector storage."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.embeddings.sentence_transformers import (  # noqa: E402
    SentenceTransformerEmbeddingProvider,
)
from app.models import Chunk  # noqa: E402
from app.retrieval.retriever import SemanticRetriever  # noqa: E402
from app.vector_stores.in_memory import InMemoryVectorStore  # noqa: E402


def main() -> None:
    """Embed example chunks, retrieve them, and print ranked source context."""
    chunks = [
        Chunk(
            chunk_id="provider-infrastructure",
            document_id="cloud-course",
            source_name="shared-responsibility.md",
            text="AWS protects the physical cloud infrastructure and data centers.",
            chunk_index=0,
            start_word=0,
            end_word=9,
            metadata={"topic": "provider responsibility"},
        ),
        Chunk(
            chunk_id="customer-configuration",
            document_id="cloud-course",
            source_name="shared-responsibility.md",
            text="Customers configure identity permissions and protect their data.",
            chunk_index=1,
            start_word=9,
            end_word=17,
            metadata={"topic": "customer responsibility"},
        ),
        Chunk(
            chunk_id="fruit-nutrition",
            document_id="nutrition-course",
            source_name="fruit.md",
            text="Bananas contain potassium and other nutrients.",
            chunk_index=0,
            start_word=0,
            end_word=6,
            metadata={"topic": "nutrition"},
        ),
    ]
    queries = [
        "Who protects cloud data centers?",
        "What do customers configure for access?",
    ]

    provider = SentenceTransformerEmbeddingProvider()
    store = InMemoryVectorStore()
    store.upsert(chunks, provider.embed_documents([chunk.text for chunk in chunks]))
    retriever = SemanticRetriever(provider, store)

    for query in queries:
        print(f"Query: {query}")
        for rank, result in enumerate(retriever.retrieve(query, top_k=3), start=1):
            chunk = result.chunk
            print(f"{rank}. score={result.score:.3f}")
            print(f"   source={chunk.source_name}, chunk_index={chunk.chunk_index}")
            print(f"   text={chunk.text}")
        print()


if __name__ == "__main__":
    main()
