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


def extract_academic_citations(text: str) -> list[tuple[str, str]]:
    """Extract both bracketed [Key] and APA parenthetical/narrative citations.

    Returns a list of tuples: (normalized_lookup_key, presentation_label).
    Example:
    - '[Page2002]' -> ('Page2002', '[Page2002]')
    - '(Artès et al., 2019)' -> ('Artès et al., 2019', '(Artès et al., 2019)')
    - 'Page (2002)' -> ('Page, 2002', '(Page, 2002)')
    """
    results: list[tuple[str, str]] = []
    seen_labels: set[str] = set()

    # 1. Bracketed: [Page2002], [1], [Vaswani et al., 2017] (excluding markdown links [text](url))
    bracket_matches = re.finditer(
        r"\[([A-Za-z0-9_,\s-]+(?:\s+et\s+al\.)?(?:,\s*\d{4})?)\](?!\()",
        text,
    )
    for m in bracket_matches:
        raw = m.group(1).strip()
        lbl = f"[{raw}]"
        if lbl not in seen_labels and raw:
            results.append((raw, lbl))
            seen_labels.add(lbl)

    # 2. Parenthetical APA: (Page et al., 2002), (Artès, 2019), (Smith & Jones, 2021)
    paren_matches = re.finditer(
        r"\(([A-Z][A-Za-z\u00C0-\u024F\s\.\-]+?(?:,\s*|\s+et\s+al\.,?\s*|\s*&\s*[A-Z][A-Za-z\u00C0-\u024F\s\.\-]+,\s*)\d{4}[a-z]?)\)",
        text,
    )
    for m in paren_matches:
        raw = m.group(1).strip()
        lbl = f"({raw})"
        if lbl not in seen_labels and raw:
            results.append((raw, lbl))
            seen_labels.add(lbl)

    # 3. Narrative APA: Artès et al. (2019) or Page (2002)
    narr_matches = re.finditer(
        r"\b([A-Z][A-Za-z\u00C0-\u024F\-]+(?:\s+et\s+al\.)?)\s+\((\d{4}[a-z]?)\)",
        text,
    )
    for m in narr_matches:
        author = m.group(1).strip()
        year = m.group(2).strip()
        raw = f"{author}, {year}"
        lbl = f"({raw})"
        if lbl not in seen_labels and raw:
            results.append((raw, lbl))
            seen_labels.add(lbl)

    return results


def resolve_vault_reference_entries(
    text: str,
    evidence: list[EvidenceChunk] | None = None,
    vault_manager: VaultManager | None = None,
    style: str = "apa",
    active_documents: set[str] | list[str] | None = None,
) -> list[dict[str, Any]]:
    """Resolve cited papers and evidence chunks into formatted bibliography entries.

    Returns a list of dictionaries with:
    - key: In-text citation key or index (e.g. '[Page2002]', '(Artès et al., 2019)', or '[1]')
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

    # 1. Extract all in-text citations (both bracketed and parenthetical/narrative APA)
    extracted_citations = extract_academic_citations(text)

    # 2. Extract evidence source stems
    evidence_stems: list[str] = []
    if evidence:
        for c in evidence:
            s = c.source.lower().strip()
            stem = Path(s).stem
            stem = re.sub(r"_c\d+$", "", stem)
            if stem and stem not in evidence_stems:
                evidence_stems.append(stem)

    matched_stems: list[tuple[str, str]] = []  # (stem, label)
    seen_stems: set[str] = set()

    def _get_default_label(target_stem: str, default_idx: int) -> str:
        v_item = item_by_stem.get(target_stem)
        if not v_item:
            return f"[{default_idx}]"
        meta = meta_by_stem.get(target_stem)
        if style.lower() == "ieee":
            return f"[{default_idx}]"
        if meta and meta.authors and meta.year:
            first_auth = meta.authors[0].split(",")[0].strip().split()[-1]
            if len(meta.authors) > 2:
                return f"({first_auth} et al., {meta.year})"
            elif len(meta.authors) == 2:
                second_auth = meta.authors[1].split(",")[0].strip().split()[-1]
                return f"({first_auth} & {second_auth}, {meta.year})"
            return f"({first_auth}, {meta.year})"
        return f"[{v_item.cite_key or target_stem}]"

    # Match in-text citations to vault items
    for raw_k, disp_label in extracted_citations:
        k_lower = raw_k.lower().strip()

        # Direct cite_key match
        if k_lower in item_by_citekey:
            stem = item_by_citekey[k_lower].stem.lower()
            if stem not in seen_stems:
                matched_stems.append((stem, disp_label))
                seen_stems.add(stem)
            continue

        # Direct stem match
        if k_lower in item_by_stem:
            if k_lower not in seen_stems:
                matched_stems.append((k_lower, disp_label))
                seen_stems.add(k_lower)
            continue

        # Match author surname and/or year against metadata
        found = False
        for stem_key, v_item in item_by_stem.items():
            meta = meta_by_stem.get(stem_key)
            if meta:
                # Check author surnames
                author_match = False
                if meta.authors:
                    for a in meta.authors:
                        clean_a = a.lower().replace(",", " ")
                        parts_a = [p for p in clean_a.split() if len(p) > 2]
                        if any(p in k_lower for p in parts_a):
                            author_match = True
                            break

                year_match = str(meta.year) in k_lower if meta.year else False
                if (author_match and year_match) or (author_match and len(item_by_stem) == 1):
                    if stem_key not in seen_stems:
                        matched_stems.append((stem_key, disp_label))
                        seen_stems.add(stem_key)
                    found = True
                    break
        if found:
            continue

        # Handle numeric citation like [1], [2] matching evidence order
        if raw_k.isdigit():
            idx = int(raw_k) - 1
            if 0 <= idx < len(evidence_stems):
                ev_stem = evidence_stems[idx]
                if ev_stem in item_by_stem and ev_stem not in seen_stems:
                    matched_stems.append((ev_stem, disp_label))
                    seen_stems.add(ev_stem)

    # 3. Fallback: if evidence chunks were provided, ensure evidence papers are included
    if evidence_stems:
        for idx, ev_stem in enumerate(evidence_stems, 1):
            if ev_stem in item_by_stem and ev_stem not in seen_stems:
                label = _get_default_label(ev_stem, len(matched_stems) + 1)
                matched_stems.append((ev_stem, label))
                seen_stems.add(ev_stem)

    # 4. Fallback: if active_documents were specified and still no match, include active documents
    if not matched_stems and active_documents:
        for doc_stem in active_documents:
            stem_norm = doc_stem.lower().strip()
            if stem_norm in item_by_stem and stem_norm not in seen_stems:
                label = _get_default_label(stem_norm, len(matched_stems) + 1)
                matched_stems.append((stem_norm, label))
                seen_stems.add(stem_norm)

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
    active_documents: set[str] | list[str] | None = None,
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
        "### 📚 referensi vault" in lower_text
        or "### referensi" in lower_text
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
        active_documents=active_documents,
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
