"""Evidence Highlighting Engine for PDF and Web Sources.

Provides:
1. Native PDF Stabilo / Highlight Annotations via PyMuPDF (stroke=(1.0, 0.9, 0.2)).
   Saves highlighted evidence snapshots in vault/annotated/ without modifying the original document.
2. W3C Scroll-to-Text Fragment URLs for Web sources (opens directly to the highlighted sentence in Chrome/Edge/Firefox).
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
import urllib.parse
from typing import Any, Optional, Tuple

import pymupdf as fitz

logger = logging.getLogger(__name__)


class EvidenceHighlighter:
    """Creates deep links and highlighted visual evidence for both PDF and Web documents."""

    @staticmethod
    def highlight_pdf_passage(
        pdf_path: Path,
        page_num: int,
        quote_snippet: str,
        output_dir: Path,
        prefix_id: str = "ev",
    ) -> Path | None:
        """Add permanent bright yellow highlight annotation to PDF passage and save in annotated/ folder."""
        if not pdf_path.exists() or not quote_snippet or not quote_snippet.strip():
            return None

        output_dir.mkdir(parents=True, exist_ok=True)
        dest_filename = f"{pdf_path.stem}_{prefix_id}_p{page_num}.pdf"
        dest_path = output_dir / dest_filename

        try:
            doc = fitz.open(str(pdf_path))
            target_page_idx = max(0, min(page_num - 1, len(doc) - 1))
            page = doc[target_page_idx]

            # Clean quote query to improve match rate (remove markdown, extra whitespace)
            clean_quote = re.sub(r"[#*_`\"']", "", quote_snippet).strip()
            # Try full match first
            quads = page.search_for(clean_quote)

            # If no direct match, try first 6-8 words
            if not quads:
                words = clean_quote.split()
                if len(words) >= 4:
                    shorter = " ".join(words[:6])
                    quads = page.search_for(shorter)

            if quads:
                annot = page.add_highlight_annot(quads)
                # Bright fluorescent yellow stabilo
                annot.set_colors(stroke=(1.0, 0.9, 0.2))
                annot.update()

            doc.save(str(dest_path), garbage=4, deflate=True)
            doc.close()
            return dest_path
        except Exception as e:
            logger.debug(f"Failed highlighting PDF evidence for {pdf_path.name}: {e}")
            return None

    @staticmethod
    def generate_web_highlight_url(
        web_url: str,
        quote_snippet: str,
    ) -> str:
        """Generate W3C Scroll-to-Text Fragment URL that highlights text in browsers."""
        if not web_url:
            return ""

        clean_quote = re.sub(r"[#*_`\"']", "", quote_snippet).strip()
        words = clean_quote.split()
        if not words:
            return web_url

        # Use first 8-12 words for robust browser text fragment matching
        target_phrase = " ".join(words[:10])
        encoded_text = urllib.parse.quote(target_phrase)
        base_url = web_url.split("#:~:text=")[0]
        return f"{base_url}#:~:text={encoded_text}"
