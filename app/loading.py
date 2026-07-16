"""Load supported local text files into normalized document models."""

import os
from pathlib import Path

from app.identifiers import create_document_id
from app.models import Document
from app.normalization import normalize_text

_SUPPORTED_SUFFIXES = frozenset({".txt", ".md"})


def load_documents(source: str | Path) -> list[Document]:
    """Load one supported file or all supported files beneath a directory.

    Directory source paths are recorded relative to that directory. A single-file
    source is recorded relative to its parent directory, which produces its name.
    """
    source_path = Path(source)

    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")

    if source_path.is_file():
        _validate_supported_file(source_path)
        return [_load_file(source_path, source_path.parent)]

    if source_path.is_dir():
        if source_path.is_symlink():
            raise ValueError(f"Symlinked directories are not supported: {source_path}")
        files = _discover_supported_files(source_path)
        return [_load_file(file_path, source_path) for file_path in files]

    raise ValueError(f"Source path must be a file or directory: {source_path}")


def _discover_supported_files(root: Path) -> list[Path]:
    """Find supported files recursively without following symlinked directories."""
    discovered: list[Path] = []

    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        directory_names[:] = [
            name for name in directory_names if not (directory_path / name).is_symlink()
        ]

        for file_name in file_names:
            file_path = directory_path / file_name
            if _is_supported_file(file_path):
                discovered.append(file_path)

    return sorted(
        discovered,
        key=lambda file_path: file_path.relative_to(root).as_posix(),
    )


def _is_supported_file(path: Path) -> bool:
    """Return whether a path has a supported text-file suffix."""
    return path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES


def _validate_supported_file(path: Path) -> None:
    """Reject individual files that are not accepted source formats."""
    if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported source file: {path}. Only .txt and .md files are supported."
        )


def _load_file(file_path: Path, ingestion_root: Path) -> Document:
    """Read, normalize, and identify one source file."""
    normalized_text = normalize_text(file_path.read_text(encoding="utf-8"))
    if not normalized_text:
        raise ValueError(
            f"Source file is empty or contains only whitespace: {file_path}"
        )

    canonical_source_path = file_path.relative_to(ingestion_root).as_posix()
    return Document(
        document_id=create_document_id(canonical_source_path, normalized_text),
        source_name=file_path.name,
        source_path=canonical_source_path,
        text=normalized_text,
    )
