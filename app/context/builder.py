"""Build readable, source-attributed context from ranked retrieval results."""

from dataclasses import dataclass

from app.models import Chunk, SearchResult

_CHUNK_SEPARATOR = "\n\n---\n\n"


@dataclass(frozen=True, slots=True)
class ContextBuildResult:
    """Formatted context and the immutable chunks that contributed to it."""

    text: str
    included_chunks: tuple[Chunk, ...]


class ContextBuilder:
    """Format unique retrieval results within a fixed character budget."""

    def __init__(self, max_characters: int = 4_000) -> None:
        """Create a builder that never returns context longer than its budget."""
        if isinstance(max_characters, bool) or not isinstance(max_characters, int):
            raise TypeError("max_characters must be an integer")
        if max_characters <= 0:
            raise ValueError("max_characters must be greater than zero")

        self._max_characters = max_characters

    def build(self, results: list[SearchResult]) -> ContextBuildResult:
        """Format unique chunks in retrieval order without truncating a chunk.

        When the next unique, fully formatted chunk cannot fit, building stops.
        This keeps the returned text within the configured budget and preserves
        the rank order of all included chunks.
        """
        context_parts: list[str] = []
        included_chunks: list[Chunk] = []
        included_chunk_ids: set[str] = set()
        context_length = 0

        for result in results:
            chunk = result.chunk
            if chunk.chunk_id in included_chunk_ids:
                continue

            formatted_chunk = _format_chunk(chunk)
            separator = _CHUNK_SEPARATOR if context_parts else ""
            addition = f"{separator}{formatted_chunk}"
            if context_length + len(addition) > self._max_characters:
                break

            context_parts.append(formatted_chunk)
            included_chunks.append(chunk)
            included_chunk_ids.add(chunk.chunk_id)
            context_length += len(addition)

        return ContextBuildResult(
            text=_CHUNK_SEPARATOR.join(context_parts),
            included_chunks=tuple(included_chunks),
        )


def _format_chunk(chunk: Chunk) -> str:
    """Return one clearly attributed context section for a source chunk."""
    return f"[Source: {chunk.source_name} | Chunk {chunk.chunk_index}]\n\n{chunk.text}"
