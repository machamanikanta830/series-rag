"""Tests for explicit cosine-similarity calculations."""

import pytest

from app.embeddings.base import cosine_similarity


def test_identical_vectors_have_similarity_one() -> None:
    """A vector has maximum cosine similarity with itself."""
    assert cosine_similarity([3.0, 4.0], [3.0, 4.0]) == pytest.approx(1.0)


def test_orthogonal_vectors_have_similarity_zero() -> None:
    """Perpendicular vectors have no directional overlap."""
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_dimension_mismatch_is_rejected() -> None:
    """Cosine similarity requires aligned vector dimensions."""
    with pytest.raises(ValueError, match="same dimension"):
        cosine_similarity([1.0, 2.0], [1.0])


@pytest.mark.parametrize(
    ("first", "second"),
    [([0.0, 0.0], [1.0, 0.0]), ([1.0, 0.0], [0.0, 0.0])],
)
def test_zero_vectors_are_rejected(first: list[float], second: list[float]) -> None:
    """A zero vector has no direction, so its cosine is undefined."""
    with pytest.raises(ValueError, match="zero vectors"):
        cosine_similarity(first, second)
