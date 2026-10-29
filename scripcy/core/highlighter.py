"""Evidence Highlighting Engine for PDF and Web Sources.

Provides:
1. Native PDF Stabilo / Highlight Annotations via PyMuPDF (stroke=(1.0, 0.9, 0.2)).
   Saves highlighted evidence snapshots in vault/annotated/ without modifying the original document.
2. W3C Scroll-to-Text Fragment URLs for Web sources (opens directly to the highlighted sentence in Chrome/Edge/Firefox).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
import re
import urllib.parse
from typing import Any, Optional, Tuple

import pymupdf as fitz

logger = logging.getLogger(__name__)

_ID_EN_STOPWORDS = {
    "yang", "dalam", "pada", "untuk", "dengan", "dari", "atau", "serta", "adalah",
    "sebagai", "terhadap", "melalui", "dapat", "lebih", "secara", "oleh", "telah",
    "merupakan", "tersebut", "sehingga", "antara", "seperti", "juga", "akan",
    "namun", "karena", "ketika", "tidak", "bisa", "bagi", "suatu", "berbagai",
    "this", "that", "with", "from", "have", "been", "were", "their", "which",
    "through", "between", "into", "such", "also", "more", "than", "these",
}


class EvidenceHighlighter:
    """Creates deep links and highlighted visual evidence for both PDF and Web documents."""

    @staticmethod
    def extract_pdf_sentence_index(pdf_path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Extract all substantive sentences across all pages of a PDF (filtering running headers/footers)
        along with document frequencies for fast IDF-weighted claim matching.
        """
        sentences: list[dict[str, Any]] = []
        df: dict[str, int] = {}
        if not pdf_path or not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
            return sentences, df

        try:
            with fitz.open(str(pdf_path)) as doc:
                for p_idx in range(len(doc)):
                    page = doc[p_idx]
                    p_num = p_idx + 1
                    p_height = page.rect.height
                    blocks = page.get_text("blocks")
                    for b in blocks:
                        if len(b) >= 5 and b[6] == 0 and 35 <= b[1] <= (p_height - 35):
                            txt = b[4].replace("-\n", "").replace("\n", " ").strip()
                            for sent in re.split(r"(?<=[.!?])\s+", txt):
                                clean_s = sent.strip()
                                if len(clean_s) >= 25:
                                    tokens = {
                                        w for w in re.findall(r"\b[a-zA-Z]{4,}\b", clean_s.lower())
                                        if w not in _ID_EN_STOPWORDS
                                    }
                                    if len(tokens) >= 2:
                                        for w in tokens:
                                            df[w] = df.get(w, 0) + 1
                                        sentences.append({
                                            "page_num": p_num,
                                            "sentence": clean_s,
                                            "tokens": tokens,
                                        })
        except Exception as e:
            logger.debug(f"Failed indexing sentences for {pdf_path.name}: {e}")

        return sentences, df

    @staticmethod
    def match_claim_to_pdf_sentence(
        claim_text: str,
        sentence_index: list[dict[str, Any]],
        df: dict[str, int],
        fallback_chunks: list[dict[str, Any]] | None = None,
    ) -> tuple[int, str]:
        """Find the exact (page_num, verbatim_sentence) in the PDF that best supports the given claim/sentence."""
        if not sentence_index:
            if fallback_chunks:
                return fallback_chunks[0]["page_num"], fallback_chunks[0]["quote"]
            return 1, ""

        claim_tokens = {
            w for w in re.findall(r"\b[a-zA-Z]{4,}\b", claim_text.lower())
            if w not in _ID_EN_STOPWORDS
        }
        total_n = len(sentence_index)
        best_page = sentence_index[0]["page_num"]
        best_sent = sentence_index[0]["sentence"]
        best_score = -1.0

        for item in sentence_index:
            common = claim_tokens & item["tokens"]
            if not common:
                continue
            # IDF-weighted overlap score: rare technical terms carry much higher weight
            score = sum(math.log(1.0 + total_n / (1.0 + df.get(w, 1))) for w in common)
            if score > best_score:
                best_score = score
                best_page = item["page_num"]
                best_sent = item["sentence"]

        if best_score <= 0 and fallback_chunks:
            return fallback_chunks[0]["page_num"], fallback_chunks[0]["quote"]

        return best_page, best_sent

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

            # Clean quote query to improve match rate (remove markdown, punctuation, extra whitespace)
            clean_quote = re.sub(r"[#*_`\"'()]", "", quote_snippet).strip()
            # Try full match first
            quads = page.search_for(clean_quote)

            # If no direct match, try first 6-8 words
            if not quads:
                words = clean_quote.split()
                if len(words) >= 4:
                    shorter = " ".join(words[:6])
                    quads = page.search_for(shorter)

            # If still no match, slide a 3-to-4 word window across the quote snippet
            if not quads:
                words = [w.strip(".,;:?!") for w in clean_quote.split() if len(w.strip(".,;:?!")) > 2]
                for w_len in (4, 3):
                    for i in range(len(words) - w_len + 1):
                        window = " ".join(words[i : i + w_len])
                        quads = page.search_for(window)
                        if quads:
                            break
                    if quads:
                        break

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
