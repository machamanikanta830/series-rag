"""Deterministic identifiers for documents and chunks."""

from hashlib import sha256

_DOCUMENT_ID_PREFIX = b"series-rag:document-id:v1\n"
_CHUNK_ID_PREFIX = b"series-rag:chunk-id:v1\n"


def _encode_part(value: str) -> bytes:
    """Length-prefix a UTF-8 value so identifier inputs are unambiguous."""
    encoded_value = value.encode("utf-8")
    byte_length = str(len(encoded_value)).encode("ascii")
    return byte_length + b":" + encoded_value + b"\n"


def _hash_parts(prefix: bytes, *parts: str) -> str:
    """Hash a versioned, length-prefixed sequence of text values."""
    payload = prefix + b"".join(_encode_part(part) for part in parts)
    return sha256(payload).hexdigest()


def create_document_id(canonical_source_path: str, normalized_text: str) -> str:
    """Create an ID from a canonical relative path and normalized document text."""
    return _hash_parts(_DOCUMENT_ID_PREFIX, canonical_source_path, normalized_text)


def create_chunk_id(document_id: str, chunk_index: int, chunk_text: str) -> str:
    """Create an ID from a document ID, a zero-based index, and chunk text."""
    return _hash_parts(_CHUNK_ID_PREFIX, document_id, str(chunk_index), chunk_text)
