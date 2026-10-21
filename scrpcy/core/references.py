"""Automatic Vault Bibliography Generator for Scrpcy.

Resolves in-text citation keys and evidence passages against project vault metadata,
formatting bibliographic references according to project citation style (APA/IEEE)
and displaying transparent vault filename tags.
"""

from __future__ import annotations

import re
from typing import Any
from pathlib import Path

from scrpcy.core.engine import EvidenceChunk
from scrpcy.core.mcp_client import (
    CitationItem,
    format_reference_entry_apa,
    format_reference_entry_ieee,
)
from scrpcy.core.models import DocumentMetadata
from scrpcy.core.polisher import AcademicPolisher
from scrpcy.core.vault import VaultItem, VaultManager


def resolve_vault_reference_entries(
    text: str,
    evidence: list[EvidenceChunk] | None = None,
    vault_manager: VaultManager | None = None,
    style: str = "apa",
) -> list[dict[str, Any]]:
    """Resolve cited papers and evidence chunks into formatted bibliography entries.

    Returns a list of dictionaries with:
    - key: In-text citation key or index (e.g. '[Page2002]' or '[1]')
    - citation_item: Normalized CitationItem
    - formatted_entry: Formatted APA or IEEE reference string
    - filename: Physical vault filename (e.g. 'peatland_fire_2002.pdf')
    - path: Absolute path to vault file
    """
    if not vault_manager:
        return []

    active_items = vault_manager.list_active_files()
    if not active_items:
        return []

    # Map stems and cite keys to VaultItem and DocumentMetadata
    item_by_stem: dict[str, VaultItem] = {}
    item_by_citekey: dict[str, VaultItem] = {}
    meta_by_stem: dict[str, DocumentMetadata] = {}

    for item in active_items:
        stem_norm = item.stem.lower()
        item_by_stem[stem_norm] = item
        cache_path = vault_manager.workspace.get_cache_path(f"{item.stem}.md")
        meta = vault_manager._read_frontmatter_metadata(cache_path) if cache_path.exists() else None
        if not meta:
            meta = DocumentMetadata(
                title=item.title or item.stem.replace("_", " ").title(),
                cite_key=item.cite_key or item.stem,
            )
        meta_by_stem[stem_norm] = meta
        if meta.cite_key:
            item_by_citekey[meta.cite_key.lower().strip("[]")] = item

    # 1. Extract in-text citation keys from response text
    raw_cites = AcademicPolisher.extract_citations(text)
    clean_keys = [c.strip("[]").strip() for c in raw_cites]

    # 2. Extract evidence source stems
    evidence_stems: list[str] = []
    if evidence:
        for c in evidence:
            s = c.source.lower().strip()
            # source might be 'peatland.md', 'peatland_c1', or stem
            stem = Path(s).stem
            stem = re.sub(r"_c\d+$", "", stem)
            if stem and stem not in evidence_stems:
                evidence_stems.append(stem)

    matched_stems: list[tuple[str, str]] = []  # (stem, label)
    seen_stems: set[str] = set()

    # Match in-text citations to vault items
    for k in clean_keys:
        k_lower = k.lower()
        # Direct cite_key match
        if k_lower in item_by_citekey:
            stem = item_by_citekey[k_lower].stem.lower()
            if stem not in seen_stems:
                matched_stems.append((stem, f"[{k}]"))
                seen_stems.add(stem)
            continue
        # Direct stem match
        if k_lower in item_by_stem:
            if k_lower not in seen_stems:
                matched_stems.append((k_lower, f"[{k}]"))
                seen_stems.add(k_lower)
            continue
        # Partial match on author/year
        found = False
        for stem_key, v_item in item_by_stem.items():
            meta = meta_by_stem.get(stem_key)
            if meta:
                # Check author name or year in key
                author_match = any(a.lower() in k_lower for a in meta.authors)
                year_match = str(meta.year) in k_lower if meta.year else False
                if author_match and year_match:
                    if stem_key not in seen_stems:
                        matched_stems.append((stem_key, f"[{k}]"))
                        seen_stems.add(stem_key)
                    found = True
                    break
        if found:
            continue

        # Handle numeric citation like [1], [2] matching evidence order
        if k.isdigit():
            idx = int(k) - 1
            if 0 <= idx < len(evidence_stems):
                ev_stem = evidence_stems[idx]
                if ev_stem in item_by_stem and ev_stem not in seen_stems:
                    matched_stems.append((ev_stem, f"[{k}]"))
                    seen_stems.add(ev_stem)

    # If no in-text citations matched, but evidence was provided, include evidence documents
    if not matched_stems and evidence_stems:
        for idx, ev_stem in enumerate(evidence_stems, 1):
            if ev_stem in item_by_stem and ev_stem not in seen_stems:
                label = f"[{idx}]" if style.lower() == "ieee" else f"[{item_by_stem[ev_stem].cite_key or ev_stem}]"
                matched_stems.append((ev_stem, label))
                seen_stems.add(ev_stem)

    # Format bibliographic entries
    entries: list[dict[str, Any]] = []
    norm_style = style.lower().strip()

    for idx, (stem, label) in enumerate(matched_stems, start=1):
        v_item = item_by_stem[stem]
        meta = meta_by_stem.get(stem) or DocumentMetadata(title=v_item.title)
        cite_item = CitationItem.from_any(meta)
        if not cite_item.title:
            cite_item.title = v_item.title or v_item.stem.replace("_", " ").title()

        if norm_style == "ieee":
            formatted = format_reference_entry_ieee(cite_item, idx)
        else:
            formatted = format_reference_entry_apa(cite_item)

        entries.append({
            "key": label,
            "citation_item": cite_item,
            "formatted_entry": formatted,
            "filename": v_item.filename,
            "path": v_item.path,
        })

    return entries


def append_vault_bibliography(
    text: str,
    evidence: list[EvidenceChunk] | None = None,
    vault_manager: VaultManager | None = None,
    style: str = "apa",
) -> str:
    """Append formatted bibliography with vault file mappings to AI response text.

    If a reference section is already present or no vault references are detected,
    returns text unchanged.
    """
    if not text or not vault_manager:
        return text

    # Avoid duplicate bibliography if already present in text
    lower_text = text.lower()
    if (
        "### referensi" in lower_text
        or "### references" in lower_text
        or "### daftar pustaka" in lower_text
        or "## referensi" in lower_text
        or "## daftar pustaka" in lower_text
    ):
        return text

    entries = resolve_vault_reference_entries(
        text=text,
        evidence=evidence,
        vault_manager=vault_manager,
        style=style,
    )
    if not entries:
        return text

    norm_style = style.upper()
    lines = [
        "",
        "---",
        f"### 📚 Referensi Vault ({norm_style}):",
    ]

    for idx, entry in enumerate(entries, 1):
        key = entry["key"]
        ref_text = entry["formatted_entry"]
        filename = entry["filename"]
        if norm_style == "IEEE":
            lines.append(f"{ref_text}  \n   `[File: {filename}]`")
        else:
            lines.append(f"{idx}. **{key}** {ref_text}  \n   `[File: {filename}]`")

    return text.rstrip() + "\n" + "\n".join(lines) + "\n"
