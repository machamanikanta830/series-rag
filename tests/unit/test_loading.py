"""Tests for loading supported source files into document models."""

from pathlib import Path

import pytest

from app.loading import load_documents


@pytest.mark.parametrize("suffix", [".txt", ".md"])
def test_loads_supported_utf8_file(tmp_path: Path, suffix: str) -> None:
    """The loader reads both supported formats as normalized UTF-8 text."""
    source_file = tmp_path / f"lesson{suffix}"
    source_file.write_text("Café\r\n\r\nLesson", encoding="utf-8")

    [document] = load_documents(source_file)

    assert document.source_name == source_file.name
    assert document.source_path == source_file.name
    assert document.text == "Café\n\nLesson"


def test_recursively_discovers_files_in_deterministic_order(tmp_path: Path) -> None:
    """Directory loading sorts canonical relative paths and skips unsupported files."""
    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()
    (tmp_path / "z-last.md").write_text("Last", encoding="utf-8")
    (nested_directory / "a-first.txt").write_text("First", encoding="utf-8")
    (tmp_path / "ignored.json").write_text("Ignored", encoding="utf-8")

    first_load = load_documents(tmp_path)
    second_load = load_documents(tmp_path)

    expected_paths = ["nested/a-first.txt", "z-last.md"]
    assert [document.source_path for document in first_load] == expected_paths
    assert [document.document_id for document in second_load] == [
        document.document_id for document in first_load
    ]


def test_rejects_a_missing_source_path(tmp_path: Path) -> None:
    """Missing source paths raise a clear filesystem error."""
    missing_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match="Source path does not exist"):
        load_documents(missing_path)


def test_rejects_an_unsupported_individual_file(tmp_path: Path) -> None:
    """An individual source must use a supported suffix."""
    source_file = tmp_path / "lesson.json"
    source_file.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Only .txt and .md files are supported"):
        load_documents(source_file)


@pytest.mark.parametrize("contents", ["", "  \n\t\r\n  "])
def test_rejects_empty_or_whitespace_only_files(tmp_path: Path, contents: str) -> None:
    """Files without normalized text cannot become documents."""
    source_file = tmp_path / "empty.txt"
    source_file.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="empty or contains only whitespace"):
        load_documents(source_file)
