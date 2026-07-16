"""A transparent fixed-size, word-based document chunker."""

from app.identifiers import create_chunk_id
from app.models import Chunk, Document

DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 40


def chunk_document(
    document: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split a document into overlapping chunks of whitespace-separated words."""
    _validate_chunk_configuration(chunk_size, chunk_overlap)

    words = document.text.split()
    chunks: list[Chunk] = []
    start_word = 0
    step_size = chunk_size - chunk_overlap

    while start_word < len(words):
        end_word = min(start_word + chunk_size, len(words))
        chunk_text = " ".join(words[start_word:end_word])
        chunk_index = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=create_chunk_id(document.document_id, chunk_index, chunk_text),
                document_id=document.document_id,
                source_name=document.source_name,
                text=chunk_text,
                chunk_index=chunk_index,
                start_word=start_word,
                end_word=end_word,
                metadata=dict(document.metadata),
            )
        )

        if end_word == len(words):
            break
        start_word += step_size

    return chunks


def _validate_chunk_configuration(chunk_size: int, chunk_overlap: int) -> None:
    """Ensure chunk configuration can create forward progress."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be zero or greater")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
