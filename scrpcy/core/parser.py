"""Academic document parser integrating PyMuPDF4LLM, Gemini extraction, and YAML frontmatter."""

from __future__ import annotations

import os
from pathlib import Path

import pymupdf
import pymupdf4llm

from scrpcy.core.extractor import GeminiMetadataExtractor, MetadataExtractor
from scrpcy.core.models import DocumentMetadata, ParsedDocument
from scrpcy.core.workspace import WorkspaceManager
from scrpcy.exceptions import CorruptedPDFError, PDFParseError


class DocumentParser:
    """Ingests academic PDFs, extracts markdown text and metadata, and caches formatted outputs."""

    def __init__(
        self,
        workspace: WorkspaceManager | None = None,
        extractor: MetadataExtractor | None = None,
    ) -> None:
        """Initialize DocumentParser.

        Args:
            workspace: Workspace directory manager. If None, uses default WorkspaceManager().
            extractor: MetadataExtractor instance. If None, defaults to GeminiMetadataExtractor().
        """
        self.workspace = workspace or WorkspaceManager()
        self.extractor = extractor or GeminiMetadataExtractor()

    def parse(
        self,
        file_path: str | Path,
        save_to_cache: bool = True,
        output_filename: str | None = None,
    ) -> ParsedDocument:
        """Parse a local PDF file into clean Markdown with prepended YAML frontmatter.

        Args:
            file_path: Local path to the PDF file.
            save_to_cache: Whether to persist the generated markdown in the workspace cache.
            output_filename: Optional custom filename for the cached file (defaults to <stem>.md).

        Returns:
            ParsedDocument containing paths, metadata, combined markdown, and body markdown.

        Raises:
            FileNotFoundError: If the input PDF file does not exist.
            CorruptedPDFError: If the PDF is unreadable, empty, truncated, or zero-paged.
            PDFParseError: If markdown extraction or processing fails.
        """
        target_path = Path(file_path).expanduser().resolve()

        if not target_path.exists():
            raise FileNotFoundError(f"PDF file not found at '{target_path}'")

        if not target_path.is_file():
            raise PDFParseError(f"Target path '{target_path}' is not a regular file.")

        try:
            file_size = target_path.stat().st_size
        except OSError as exc:
            raise PDFParseError(f"Cannot inspect file stats for '{target_path}': {exc}") from exc

        if file_size == 0:
            raise CorruptedPDFError(f"PDF file is empty (0 bytes): '{target_path}'")

        # Validate that the file is an openable PDF document with at least one page
        try:
            with pymupdf.open(target_path) as doc:
                if len(doc) == 0:
                    raise CorruptedPDFError(f"PDF document has 0 pages: '{target_path}'")

                # Extract first page representation for metadata extraction
                first_page_text = pymupdf4llm.to_markdown(doc, pages=[0])
                if not first_page_text or not first_page_text.strip():
                    first_page_text = doc[0].get_text("text")

                # Extract markdown for the full document
                clean_body_markdown = pymupdf4llm.to_markdown(doc)
        except CorruptedPDFError:
            raise
        except (pymupdf.FileDataError, pymupdf.EmptyFileError) as exc:
            raise CorruptedPDFError(
                f"Corrupted or invalid PDF format at '{target_path}': {exc}"
            ) from exc
        except Exception as exc:
            raise CorruptedPDFError(
                f"Failed to read or decode PDF at '{target_path}': {exc}"
            ) from exc

        # Extract structured metadata via the LLM extractor
        try:
            metadata = self.extractor.extract(first_page_text)
            metadata.ensure_cite_key()
        except Exception:
            # Graceful fallback metadata from file stem if extractor fails or no API key provided
            metadata = DocumentMetadata(
                title=target_path.stem.replace("_", " ").title(),
                cite_key=target_path.stem.lower().replace(" ", "_"),
            )
            metadata.ensure_cite_key()

        # Build YAML frontmatter and composite markdown
        frontmatter = metadata.to_frontmatter()
        combined_markdown = f"{frontmatter}\n{clean_body_markdown.strip()}\n"

        cache_path: Path | None = None
        if save_to_cache:
            target_cache_name = output_filename or f"{target_path.stem}.md"
            cache_path = self.workspace.save_to_cache(
                target_cache_name,
                combined_markdown,
            )

        return ParsedDocument(
            file_path=target_path,
            cache_path=cache_path,
            metadata=metadata,
            markdown=combined_markdown,
            body_markdown=clean_body_markdown,
        )


def parse_pdf(
    file_path: str | Path,
    workspace: WorkspaceManager | None = None,
    extractor: MetadataExtractor | None = None,
    save_to_cache: bool = True,
    output_filename: str | None = None,
) -> ParsedDocument:
    """Convenience functional wrapper around DocumentParser.parse."""
    parser = DocumentParser(workspace=workspace, extractor=extractor)
    return parser.parse(
        file_path=file_path,
        save_to_cache=save_to_cache,
        output_filename=output_filename,
    )
