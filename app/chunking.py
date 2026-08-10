"""A transparent fixed-size, word-based document chunker."""

from app.identifiers import create_chunk_id
from app.models import Chunk, Document, DocumentSection, Metadata

DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 40


def chunk_document(
    document: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split a document into overlapping chunks of whitespace-separated words."""
    _validate_chunk_configuration(chunk_size, chunk_overlap)

    chunks: list[Chunk] = []
    sections = document.sections or (DocumentSection(text=document.text),)
    document_word_offset = 0

    for section in sections:
        section_words = section.text.split()
        section_metadata = dict(document.metadata)
        section_metadata.update(section.metadata)
        _append_section_chunks(
            chunks=chunks,
            document=document,
            words=section_words,
            metadata=section_metadata,
            document_word_offset=document_word_offset,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        document_word_offset += len(section_words)

    return chunks


def _append_section_chunks(
    *,
    chunks: list[Chunk],
    document: Document,
    words: list[str],
    metadata: Metadata,
    document_word_offset: int,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Append chunks for one section without crossing its provenance boundary."""
    section_start_word = 0
    step_size = chunk_size - chunk_overlap

    while section_start_word < len(words):
        section_end_word = min(section_start_word + chunk_size, len(words))
        chunk_text = " ".join(words[section_start_word:section_end_word])
        chunk_index = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=create_chunk_id(document.document_id, chunk_index, chunk_text),
                document_id=document.document_id,
                source_name=document.source_name,
                text=chunk_text,
                chunk_index=chunk_index,
                start_word=document_word_offset + section_start_word,
                end_word=document_word_offset + section_end_word,
                metadata=dict(metadata),
            )
        )

        if section_end_word == len(words):
            break
        section_start_word += step_size


def _validate_chunk_configuration(chunk_size: int, chunk_overlap: int) -> None:
    """Ensure chunk configuration can create forward progress."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be zero or greater")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
