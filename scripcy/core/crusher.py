"""SmartCrusher & Markdown Minifier for LLM Token Optimization.

Provides structural compression (JSON-to-TSV, null pruning, schema compaction)
and semantic-preserving Markdown minification for RAG chunks and agent context.
"""

from __future__ import annotations

import json
import re
from typing import Any


class MarkdownMinifier:
    """Minifies Markdown and RAG evidence chunks while preserving tables and citations."""

    @staticmethod
    def minify(text: str, preserve_tables: bool = True) -> str:
        """Strip non-semantic bloat, collapse whitespace, and compact links/decorations."""
        if not text:
            return ""

        # Normalize line endings
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")

        # Strip HTML comments
        cleaned = re.sub(r"<!--.*?-->", "", cleaned, flags=re.DOTALL)

        # Strip non-structural inline HTML tags (span, div, b, i, font, a) while preserving inner text
        cleaned = re.sub(r"<(?:span|div|b|i|strong|em|font|p)\b[^>]*>(.*?)</(?:span|div|b|i|strong|em|font|p)>", r"\1", cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"</?(?:span|div|b|i|strong|em|font|p)\b[^>]*>", "", cleaned, flags=re.IGNORECASE)

        # Replace markdown image embeds with concise text labels or empty
        cleaned = re.sub(r"!\[(.*?)\]\([^\)]+\)", lambda m: f"[Gambar: {m.group(1)}]" if m.group(1).strip() else "", cleaned)

        # Simplify markdown links with long tracking query strings: [Text](https://url?utm=...) -> [Text](https://url)
        cleaned = re.sub(r"\[([^\]]+)\]\((https?://[^\s\)]+)\)", lambda m: f"[{m.group(1)}]({m.group(2).split('?')[0]})", cleaned)

        # Remove repetitive horizontal rules (---, ***, ___ on own line)
        cleaned = re.sub(r"^[ \t]*[-*_]{3,}[ \t]*$", "", cleaned, flags=re.MULTILINE)

        # Process line by line to strip trailing spaces while preserving code block and table structure
        lines = cleaned.split("\n")
        processed_lines: list[str] = []
        in_code_block = False

        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                processed_lines.append(line.rstrip())
                continue

            if in_code_block:
                processed_lines.append(line)
                continue

            trimmed = line.rstrip()
            # Collapse multiple spaces inside plain text lines (avoiding markdown pipe tables)
            if preserve_tables and "|" in trimmed:
                processed_lines.append(trimmed)
            else:
                collapsed = re.sub(r"[ \t]{2,}", " ", trimmed)
                processed_lines.append(collapsed)

        result = "\n".join(processed_lines)

        # Collapse 3 or more consecutive newlines into at most 2 newlines
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()


class SmartCrusher:
    """Compresses JSON structures, extracts compact TSV, and eliminates redundant keys."""

    @staticmethod
    def to_compact_tsv(records: list[dict[str, Any]], columns: list[str] | None = None) -> str:
        """Convert list of uniform dictionaries into a compact tab-separated table."""
        if not records:
            return ""

        cols = columns or list(records[0].keys())
        lines = ["\t".join(cols)]

        for r in records:
            row_vals: list[str] = []
            for col in cols:
                val = r.get(col, "")
                if val is None:
                    val = ""
                # Flatten any internal newlines or tabs to preserve TSV grid integrity
                str_val = str(val).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()
                row_vals.append(str_val)
            lines.append("\t".join(row_vals))

        return "\n".join(lines)

    @staticmethod
    def prune_nulls(data: Any) -> Any:
        """Recursively remove None, empty strings, and empty collections from dictionaries/lists."""
        if isinstance(data, dict):
            pruned: dict[str, Any] = {}
            for k, v in data.items():
                pv = SmartCrusher.prune_nulls(v)
                if pv is None or pv == "" or pv == [] or pv == {}:
                    continue
                pruned[k] = pv
            return pruned
        elif isinstance(data, list):
            pruned_list = [SmartCrusher.prune_nulls(v) for v in data]
            return [v for v in pruned_list if v is not None and v != "" and v != [] and v != {}]
        return data

    @staticmethod
    def crush_json(data: Any) -> str:
        """Prune empty values and serialize JSON with compact tokenizer-friendly delimiters."""
        pruned = SmartCrusher.prune_nulls(data)
        if pruned is None or pruned == "" or pruned == []:
            return "{}"
        return json.dumps(pruned, ensure_ascii=False, separators=(",", ": "))

    @staticmethod
    def crush_verified_facts_ledger(facts: list[Any]) -> str:
        """Serialize a list of ExtractedFact objects into an ultra-compact TSV block."""
        if not facts:
            return ""

        rows: list[dict[str, Any]] = []
        for f in facts:
            claim = getattr(f, "claim_or_metric", "").strip()
            stem = getattr(f, "citation_stem", "").strip()
            year = getattr(f, "year", "") or ""
            vis = getattr(f, "visual_ref", "") or ""
            if claim and stem:
                rows.append({
                    "Fakta_Klaim": claim,
                    "Sumber": stem,
                    "Tahun": year,
                    "Visual": vis,
                })

        if not rows:
            return ""

        # Build compact TSV with header
        cols = ["Fakta_Klaim", "Sumber", "Tahun"]
        if any(r["Visual"] for r in rows):
            cols.append("Visual")

        return SmartCrusher.to_compact_tsv(rows, columns=cols)
