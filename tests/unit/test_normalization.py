"""Tests for conservative text normalization."""

from app.normalization import normalize_text


def test_normalizes_line_endings_nulls_and_excess_whitespace() -> None:
    """Line ending and whitespace cleanup keeps the text's words intact."""
    raw_text = "First\r\nSecond\rThird\x00  \n\n\n\nFourth\t\titem  "

    assert normalize_text(raw_text) == "First\nSecond\nThird\n\nFourth item"


def test_collapses_repeated_spaces_and_tabs_within_a_line() -> None:
    """Runs of inline spaces and tabs become one space."""
    assert normalize_text("one   two\t\tthree") == "one two three"


def test_preserves_punctuation_casing_and_paragraph_boundaries() -> None:
    """Meaningful wording and two-newline paragraph boundaries remain unchanged."""
    text = "## TITLE!\n\nHello, WORLD? (Yes.)\n\nNext paragraph."

    assert normalize_text(text) == text


def test_normalization_is_idempotent() -> None:
    """Normalizing an already normalized value does not change it."""
    normalized = normalize_text(" One\r\n\r\n\r\nTwo\t\twords ")

    assert normalize_text(normalized) == normalized
