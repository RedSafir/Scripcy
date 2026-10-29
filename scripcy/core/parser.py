"""Academic document parser integrating PyMuPDF4LLM, Gemini extraction, and YAML frontmatter."""

from __future__ import annotations

import os
from pathlib import Path

import pymupdf

from scripcy.core.extractor import GeminiMetadataExtractor, MetadataExtractor
from scripcy.core.models import DocumentMetadata, ParsedDocument
from scripcy.core.workspace import WorkspaceManager
from scripcy.exceptions import CorruptedPDFError, PDFParseError


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

    def _heuristic_extract_metadata(self, text: str, target_path: Path) -> DocumentMetadata:
        """Fallback deterministic heuristic metadata extraction from text content."""
        import re

        doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", text[:2500])
        doi = doi_match.group(0).rstrip(".,)") if doi_match else None

        year_match = re.search(r"(?:published|terbit|tahun|copyright|©|\(c\))\s*:?.*?(\b20\d{2}\b)", text[:2500], re.IGNORECASE)
        if not year_match:
            year_match = re.search(r"\b(20\d{2})\b", text[:2500])
        year = int(year_match.group(1)) if year_match else None

        authors = []
        auth_match = re.search(r"\*\*Penulis\*\*:\s*([^(\n]+)", text[:2500])
        if not auth_match:
            auth_match = re.search(r"(?:^|\n)\s*[_*]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)[_*]+", text[:2500])
        if auth_match:
            cand = auth_match.group(1).strip()
            if not any(w in cand.lower() for w in ["research article", "keywords", "abstrak", "abstract", "universitas", "university"]):
                authors.append(cand)

        title = target_path.stem.replace("_", " ").title()
        header_match = re.search(r"(?:^|\n)#+\s*\**([^\n*]+)\**", text[:2000])
        if header_match:
            cand_title = header_match.group(1).strip()
            if len(cand_title) > 8 and not cand_title.lower().startswith("research article"):
                title = cand_title

        return DocumentMetadata(
            title=title,
            authors=authors,
            year=year,
            doi=doi,
            cite_key=target_path.stem.lower().replace(" ", "_"),
        )

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

        import pymupdf4llm

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
            # Graceful deterministic fallback from first page text before raw stem fallback
            metadata = self._heuristic_extract_metadata(first_page_text or clean_body_markdown, target_path)
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
