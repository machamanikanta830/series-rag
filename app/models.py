"""Immutable data structures used by the ingestion and chunking pipeline."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

Metadata = Mapping[str, str]


def _freeze_metadata(metadata: Metadata) -> Metadata:
    """Copy metadata into a read-only mapping for an immutable model."""
    return MappingProxyType(dict(metadata))


@dataclass(frozen=True, slots=True)
class DocumentSection:
    """One traceable part of a document with section-specific metadata."""

    text: str
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Protect section metadata from mutation after creation."""
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class Document:
    """One normalized source document ready to be chunked."""

    document_id: str
    source_name: str
    source_path: str
    text: str
    metadata: Metadata = field(default_factory=dict)
    sections: tuple[DocumentSection, ...] = ()

    def __post_init__(self) -> None:
        """Protect metadata and section ordering after document creation."""
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        object.__setattr__(self, "sections", tuple(self.sections))


@dataclass(frozen=True, slots=True)
class Chunk:
    """A word-based slice of a document, including its source context."""

    chunk_id: str
    document_id: str
    source_name: str
    text: str
    chunk_index: int
    start_word: int
    end_word: int
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Protect metadata from mutation after chunk creation."""
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One ranked chunk returned by vector similarity search."""

    chunk: Chunk
    score: float
