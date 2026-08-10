"""Extract page-aware plain text from native-text PDF bytes."""

import pymupdf

from app.identifiers import create_document_id
from app.models import Document, DocumentSection
from app.normalization import normalize_text


class PDFParsingError(ValueError):
    """Raised when uploaded bytes cannot be read as a PDF."""


class PDFNoExtractableTextError(ValueError):
    """Raised when a readable PDF contains no usable native text."""


def parse_pdf(content: bytes, filename: str) -> Document:
    """Extract normalized native text and one traceable section per nonempty page."""
    try:
        with pymupdf.open(  # type: ignore[no-untyped-call]
            stream=content,
            filetype="pdf",
        ) as pdf_document:
            if pdf_document.needs_pass:
                raise PDFParsingError("Password-protected PDFs are not supported")

            page_count = pdf_document.page_count
            extracted_sections: list[DocumentSection] = []
            for page_number, page in enumerate(pdf_document, start=1):
                section = _extract_page_section(
                    page=page,
                    page_number=page_number,
                )
                if section is not None:
                    extracted_sections.append(section)
            sections = tuple(extracted_sections)
    except PDFParsingError:
        raise
    except (pymupdf.FileDataError, RuntimeError) as error:
        raise PDFParsingError("Uploaded PDF is malformed or unreadable") from error

    if not sections:
        raise PDFNoExtractableTextError(
            "Uploaded PDF contains no extractable native text"
        )

    normalized_text = "\n\n".join(section.text for section in sections)
    return Document(
        document_id=create_document_id(filename, normalized_text),
        source_name=filename,
        source_path=filename,
        text=normalized_text,
        metadata={
            "filename": filename,
            "page_count": str(page_count),
            "source_type": "pdf",
        },
        sections=sections,
    )


def _extract_page_section(
    *,
    page: pymupdf.Page,
    page_number: int,
) -> DocumentSection | None:
    """Return one nonempty normalized page with a one-based page number."""
    extracted_text = page.get_text(  # type: ignore[no-untyped-call]
        "text",
        sort=True,
    )
    page_text = normalize_text(extracted_text)
    if not page_text:
        return None
    return DocumentSection(
        text=page_text,
        metadata={"page_number": str(page_number)},
    )
