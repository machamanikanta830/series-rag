"""Demonstrate embeddings and cosine similarity with a few short sentences."""

import sys
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.embeddings.base import cosine_similarity  # noqa: E402
from app.embeddings.sentence_transformers import (  # noqa: E402
    SentenceTransformerEmbeddingProvider,
)

EXAMPLE_SENTENCES = [
    "AWS secures the cloud infrastructure.",
    "The cloud provider protects physical data centers.",
    "A banana contains potassium.",
]


def main() -> None:
    """Embed example sentences, compare them, and rank them by similarity."""
    provider = SentenceTransformerEmbeddingProvider()
    embeddings = provider.embed_documents(EXAMPLE_SENTENCES)

    print(f"Embedding dimension: {provider.embedding_dimension()}")
    print("\nFirst eight values from each normalized embedding:")
    for sentence, embedding in zip(EXAMPLE_SENTENCES, embeddings, strict=True):
        print(f"- {sentence}\n  {embedding[:8]}")

    print("\nPairwise cosine similarity:")
    for first_index, second_index in combinations(range(len(EXAMPLE_SENTENCES)), 2):
        similarity = cosine_similarity(
            embeddings[first_index], embeddings[second_index]
        )
        print(
            f"- {similarity:.3f}: {EXAMPLE_SENTENCES[first_index]} "
            f"<-> {EXAMPLE_SENTENCES[second_index]}"
        )

    reference_index = 0
    ranked_sentences = sorted(
        enumerate(EXAMPLE_SENTENCES),
        key=lambda item: cosine_similarity(
            embeddings[reference_index], embeddings[item[0]]
        ),
        reverse=True,
    )

    print(f"\nRanking by similarity to: {EXAMPLE_SENTENCES[reference_index]}")
    for rank, (sentence_index, sentence) in enumerate(ranked_sentences, start=1):
        similarity = cosine_similarity(
            embeddings[reference_index], embeddings[sentence_index]
        )
        print(f"{rank}. {similarity:.3f} — {sentence}")


if __name__ == "__main__":
    main()
