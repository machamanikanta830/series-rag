"""Extract ordered, structured native text from DOCX bytes."""

from collections.abc import Iterable
from io import BytesIO
from zipfile import BadZipFile

from docx import Document as open_docx
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.identifiers import create_document_id
from app.models import Document, DocumentSection
from app.normalization import normalize_text


class DOCXParsingError(ValueError):
    """Raised when uploaded bytes cannot be read as a DOCX package."""


class DOCXNoExtractableTextError(ValueError):
    """Raised when a readable DOCX contains no usable supported text."""


def parse_docx(content: bytes, filename: str) -> Document:
    """Extract body paragraphs and tables as ordered document sections."""
    try:
        docx_document = open_docx(BytesIO(content))
        sections = _extract_sections(docx_document.iter_inner_content())
    except (
        BadZipFile,
        KeyError,
        PackageNotFoundError,
        SyntaxError,
        ValueError,
    ) as error:
        raise DOCXParsingError("Uploaded DOCX is malformed or unreadable") from error

    if not sections:
        raise DOCXNoExtractableTextError(
            "Uploaded DOCX contains no extractable supported text"
        )

    normalized_text = "\n\n".join(section.text for section in sections)
    return Document(
        document_id=create_document_id(filename, normalized_text),
        source_name=filename,
        source_path=filename,
        text=normalized_text,
        metadata={
            "filename": filename,
            "source_type": "docx",
        },
        sections=sections,
    )


def _extract_sections(
    blocks: Iterable[Paragraph | Table],
) -> tuple[DocumentSection, ...]:
    """Convert supported body blocks into immutable sections in source order."""
    sections: list[DocumentSection] = []
    for block in blocks:
        if isinstance(block, Paragraph):
            section = _paragraph_section(block)
        else:
            section = _table_section(block)

        if section is not None:
            sections.append(section)
    return tuple(sections)


def _paragraph_section(paragraph: Paragraph) -> DocumentSection | None:
    """Convert one nonempty paragraph or heading into a traceable section."""
    text = normalize_text(paragraph.text)
    if not text:
        return None

    style_name = paragraph.style.name if paragraph.style is not None else ""
    metadata = {"section_type": "paragraph"}
    if style_name.lower().startswith("heading"):
        metadata["section_type"] = "heading"
        metadata["heading_style"] = style_name

    return DocumentSection(text=text, metadata=metadata)


def _table_section(table: Table) -> DocumentSection | None:
    """Render one table as deterministic pipe-separated plain-text rows."""
    rendered_rows: list[str] = []
    for row in table.rows:
        cells = [_single_line_text(cell.text) for cell in row.cells]
        if any(cells):
            rendered_rows.append(" | ".join(cells))

    if not rendered_rows:
        return None
    return DocumentSection(
        text="\n".join(rendered_rows),
        metadata={"section_type": "table"},
    )


def _single_line_text(text: str) -> str:
    """Normalize a table cell and keep its content on one readable row."""
    return " ".join(normalize_text(text).split())
