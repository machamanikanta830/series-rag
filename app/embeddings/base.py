"""Small, explicit interfaces and utilities for embedding vectors."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from math import sqrt


class EmbeddingProvider(ABC):
    """Generate numeric vector representations of documents and queries."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector for each document text."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Return one embedding vector for a query text."""

    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the number of values in each embedding vector."""


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity from the dot product and vector magnitudes."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimension")

    dot_product = sum(first * second for first, second in zip(a, b, strict=True))
    magnitude_a = sqrt(sum(value * value for value in a))
    magnitude_b = sqrt(sum(value * value for value in b))

    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors")

    return dot_product / (magnitude_a * magnitude_b)
