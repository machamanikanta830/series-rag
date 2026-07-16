"""Conservative text normalization for plain-text source documents."""

import re

_INLINE_WHITESPACE = re.compile(r"[ \t]+")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving wording, punctuation, and paragraphs."""
    normalized_line_endings = text.replace("\r\n", "\n").replace("\r", "\n")
    without_nulls = normalized_line_endings.replace("\x00", "")
    normalized_lines = [
        _INLINE_WHITESPACE.sub(" ", line.rstrip(" \t"))
        for line in without_nulls.split("\n")
    ]
    normalized = "\n".join(normalized_lines)
    return _EXCESS_NEWLINES.sub("\n\n", normalized).strip()
