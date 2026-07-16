"""A lazy Sentence Transformers embedding provider."""

from collections.abc import Iterable
from math import sqrt

from sentence_transformers import SentenceTransformer

from app.embeddings.base import EmbeddingProvider

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Generate normalized embeddings with one lazily loaded model instance."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        """Store the model name without loading model files yet."""
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed non-empty document texts and return normalized Python lists."""
        if not texts:
            raise ValueError("Document texts must not be empty")
        if any(not text.strip() for text in texts):
            raise ValueError("Document texts must not contain empty values")

        model = self._get_model()
        expected_dimension = self.embedding_dimension()
        encoded_embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        normalized_embeddings = [
            _normalize_embedding(embedding) for embedding in encoded_embeddings
        ]

        has_unexpected_dimension = any(
            len(embedding) != expected_dimension for embedding in normalized_embeddings
        )
        if has_unexpected_dimension:
            raise RuntimeError(
                "Model returned an embedding with a dimension different from "
                "its reported dimension"
            )

        return normalized_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed one non-empty query text."""
        if not text.strip():
            raise ValueError("Query text must not be empty")
        return self.embed_documents([text])[0]

    def embedding_dimension(self) -> int:
        """Return the dimension reported by the loaded sentence-transformer model."""
        dimension = self._get_model().get_embedding_dimension()
        if dimension is None:
            raise RuntimeError("Loaded model did not report an embedding dimension")
        return dimension

    def _get_model(self) -> SentenceTransformer:
        """Load the configured model once and cache it for this provider instance."""
        if self._model is None:
            try:
                self._model = SentenceTransformer(self._model_name)
            except (OSError, RuntimeError, ValueError) as error:
                raise RuntimeError(
                    "Unable to load Sentence Transformers model "
                    f"{self._model_name!r}. Check the model name and network or "
                    "local model cache."
                ) from error
        return self._model


def _normalize_embedding(embedding: Iterable[float]) -> list[float]:
    """Return an L2-normalized Python list without relying on a library helper."""
    values = [float(value) for value in embedding]
    magnitude = sqrt(sum(value * value for value in values))

    if magnitude == 0:
        raise ValueError("Model returned a zero embedding, which cannot be normalized")

    return [value / magnitude for value in values]
