"""Tests for source-attributed, budget-limited retrieval context."""

import pytest

from app.context.builder import ContextBuilder
from app.models import Chunk, SearchResult


def _result(
    chunk_id: str,
    *,
    source_name: str = "source.md",
    chunk_index: int = 0,
    text: str = "Chunk text.",
    score: float = 0.9,
) -> SearchResult:
    """Create a ranked result with source metadata for context tests."""
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="document-1",
            source_name=source_name,
            text=text,
            chunk_index=chunk_index,
            start_word=0,
            end_word=2,
            metadata={"topic": "testing"},
        ),
        score=score,
    )


def test_builder_formats_results_in_retrieval_order_with_source_attribution() -> None:
    """The rendered context keeps result order and identifies each source."""
    results = [
        _result("first", source_name="aws.md", chunk_index=0, text="AWS text."),
        _result(
            "second",
            source_name="security.md",
            chunk_index=4,
            text="Security text.",
            score=0.8,
        ),
    ]

    built_context = ContextBuilder().build(results)

    assert built_context.text == (
        "[Source: aws.md | Chunk 0]\n\nAWS text.\n\n---\n\n"
        "[Source: security.md | Chunk 4]\n\nSecurity text."
    )
    assert [chunk.chunk_id for chunk in built_context.included_chunks] == [
        "first",
        "second",
    ]


def test_builder_removes_duplicate_chunk_ids_without_reordering() -> None:
    """Only the first occurrence of each logical chunk enters the context."""
    first = _result("duplicate", text="First version.")
    duplicate = _result("duplicate", text="Later version.", score=0.8)
    second = _result("second", text="Second chunk.", score=0.7)

    built_context = ContextBuilder().build([first, duplicate, second])

    assert "First version." in built_context.text
    assert "Later version." not in built_context.text
    assert [chunk.chunk_id for chunk in built_context.included_chunks] == [
        "duplicate",
        "second",
    ]


def test_builder_stops_before_a_chunk_that_exceeds_the_character_budget() -> None:
    """The complete context, including separators, stays within the budget."""
    first = _result("first", text="Short text.")
    second = _result("second", chunk_index=1, text="This text does not fit.")
    first_text = "[Source: source.md | Chunk 0]\n\nShort text."
    builder = ContextBuilder(max_characters=len(first_text))

    built_context = builder.build([first, second])

    assert built_context.text == first_text
    assert len(built_context.text) == len(first_text)
    assert [chunk.chunk_id for chunk in built_context.included_chunks] == ["first"]


def test_builder_returns_empty_context_when_the_first_chunk_does_not_fit() -> None:
    """Chunks are never partially truncated merely to fill the budget."""
    result = _result("first", text="This complete chunk is too long.")

    built_context = ContextBuilder(max_characters=10).build([result])

    assert built_context.text == ""
    assert built_context.included_chunks == ()


def test_builder_preserves_included_chunk_metadata() -> None:
    """Callers can inspect the source metadata for every included chunk."""
    result = _result("first", source_name="notes.md", chunk_index=2)

    built_context = ContextBuilder().build([result])

    included_chunk = built_context.included_chunks[0]
    assert included_chunk.source_name == "notes.md"
    assert included_chunk.chunk_index == 2
    assert included_chunk.metadata == {"topic": "testing"}


@pytest.mark.parametrize("max_characters", [0, -1])
def test_builder_rejects_non_positive_character_budgets(max_characters: int) -> None:
    """A character budget must leave room for at least one character."""
    with pytest.raises(ValueError, match="greater than zero"):
        ContextBuilder(max_characters=max_characters)


@pytest.mark.parametrize("max_characters", ["100", 1.5, True])
def test_builder_rejects_non_integer_character_budgets(
    max_characters: object,
) -> None:
    """Boolean and non-integer values are not meaningful character budgets."""
    with pytest.raises(TypeError, match="must be an integer"):
        ContextBuilder(max_characters=max_characters)  # type: ignore[arg-type]
