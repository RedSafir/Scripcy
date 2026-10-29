"""Automatic Vault Bibliography Generator for Scripcy.

Resolves in-text citation keys and evidence passages against project vault metadata,
formatting bibliographic references according to project citation style (APA/IEEE)
and displaying transparent vault filename tags.
"""

from __future__ import annotations

import re
from typing import Any
from pathlib import Path

from scripcy.core.engine import EvidenceChunk
from scripcy.core.highlighter import EvidenceHighlighter
from scripcy.core.mcp_client import (
    CitationItem,
    format_reference_entry_apa,
    format_reference_entry_ieee,
)
from scripcy.core.models import DocumentMetadata
from scripcy.core.polisher import AcademicPolisher
from scripcy.core.vault import VaultItem, VaultManager


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

        # Match author surname, title keywords, and/or year against metadata
        found = False
        k_year_match = re.search(r"\b(19\d{2}|20\d{2})\b", k_lower)
        k_year = int(k_year_match.group(1)) if k_year_match else None

        for stem_key, v_item in item_by_stem.items():
            meta = meta_by_stem.get(stem_key)
            if meta:
                # Dynamic heuristic hydration if metadata authors are missing
                if not meta.authors and vault_manager:
                    c_path = vault_manager.workspace.get_cache_path(f"{v_item.stem}.md")
                    if c_path.exists():
                        try:
                            c_text = c_path.read_text(encoding="utf-8", errors="replace")[:2500]
                            h_meta = vault_manager.pdf_parser._heuristic_extract_metadata(c_text, v_item.path)
                            if h_meta.authors:
                                meta.authors = h_meta.authors
                            if not meta.year and h_meta.year:
                                meta.year = h_meta.year
                            if not meta.doi and h_meta.doi:
                                meta.doi = h_meta.doi
                        except Exception:
                            pass

                # Check author surnames
                author_match = False
                if meta.authors:
                    for a in meta.authors:
                        clean_a = a.lower().replace(",", " ")
                        parts_a = [p for p in clean_a.split() if len(p) > 2]
                        if any(p in k_lower for p in parts_a):
                            author_match = True
                            break

                # Check title / stem keywords (e.g. 'bayesian' in 'Integration of Bayesian Methods')
                title_keyword_match = False
                title_tokens = [
                    w for w in re.findall(r"\b[a-zA-Z]{5,}\b", (meta.title or stem_key).lower())
                    if w not in {"review", "study", "analysis", "empirical", "theoretical", "method", "systematic"}
                ]
                if any(w in k_lower for w in title_tokens):
                    title_keyword_match = True

                year_match = False
                if meta.year:
                    if k_year:
                        year_match = (meta.year == k_year or abs(meta.year - k_year) <= 1)
                    else:
                        year_match = str(meta.year) in k_lower

                if ((author_match or title_keyword_match) and (year_match or not k_year)) or (author_match and len(item_by_stem) == 1):
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


def strip_manual_bibliography(text: str) -> str:
    """Cleanly strip any manual bibliography or references section written by an LLM.

    Preserves verified '### 📚 Daftar Pustaka:' sections generated by Scripcy.
    """
    if not text:
        return ""
    if "### 📚 daftar pustaka:" in text.lower():
        return text

    pattern = re.compile(
        r"(?i)\n+(?:---*\s*\n+)?#{1,4}\s*(?:📚\s*)?(?:daftar\s+pustaka|referensi\s+vault|referensi|references|bibliography)[\s\S]*$",
    )
    return pattern.sub("", text).rstrip()


def append_vault_bibliography(
    text: str,
    evidence: list[EvidenceChunk] | None = None,
    vault_manager: VaultManager | None = None,
    style: str = "apa",
    active_documents: set[str] | list[str] | None = None,
) -> str:
    """Append formatted bibliography with vault file mappings to AI response text.

    Strips any unverified manual bibliography written by the model, resolves in-text citations
    against vault metadata, and appends a standardized '### 📚 Daftar Pustaka:' section.
    """
    if not text or not vault_manager:
        return text

    if "### 📚 daftar pustaka:" in text.lower():
        return text

    cleaned_text = strip_manual_bibliography(text)

    entries = resolve_vault_reference_entries(
        text=cleaned_text,
        evidence=evidence,
        vault_manager=vault_manager,
        style=style,
        active_documents=active_documents,
    )
    if not entries:
        return cleaned_text

    norm_style = style.upper()
    lines = [
        "",
        "---",
        "### 📚 Daftar Pustaka:",
    ]

    base_vault_dir = getattr(vault_manager, "vault_dir", None) or Path("vault")
    annotated_dir = Path(base_vault_dir) / "annotated"

    # Pre-generate annotated visual evidence per evidence chunk (fine-grained claim level)
    # Map (paper_stem, page_num) -> link_url
    chunk_annot_urls: list[dict[str, Any]] = []
    if evidence:
        for ev_idx, ev in enumerate(evidence, 1):
            ev_source_clean = re.sub(r"_c\d+$", "", Path(ev.source).stem).lower()
            ev_page = ev.page_num or 1
            ev_quote = ev.quote_snippet or ""
            target_p = None
            for e in entries:
                if e.get("path") and ev_source_clean in Path(e["path"]).stem.lower():
                    target_p = Path(e["path"])
                    break

            ev_url = ""
            if target_p and target_p.exists() and target_p.suffix.lower() == ".pdf" and ev_quote:
                annot_pdf = EvidenceHighlighter.highlight_pdf_passage(
                    pdf_path=target_p,
                    page_num=ev_page,
                    quote_snippet=ev_quote,
                    output_dir=annotated_dir,
                    prefix_id=f"ev_p{ev_page}_{ev_idx}",
                )
                tgt_f = annot_pdf if annot_pdf and annot_pdf.exists() else target_p
                clean_fp = tgt_f.resolve().as_posix()
                if not clean_fp.startswith("/"):
                    clean_fp = f"/{clean_fp}"
                ev_url = f"file://{clean_fp}#page={ev_page}"
            elif target_p and target_p.exists():
                clean_fp = target_p.resolve().as_posix()
                if not clean_fp.startswith("/"):
                    clean_fp = f"/{clean_fp}"
                ev_url = f"file://{clean_fp}#page={ev_page}"

            if ev_url:
                chunk_annot_urls.append({
                    "stem": ev_source_clean,
                    "page_num": ev_page,
                    "quote": ev_quote,
                    "text": ev.text,
                    "url": ev_url,
                })

    for idx, entry in enumerate(entries, 1):
        key = entry["key"]
        ref_text = entry["formatted_entry"]
        filename = entry["filename"]
        raw_path = entry.get("path")
        p_obj = Path(raw_path) if raw_path else None
        p_stem = p_obj.stem.lower() if p_obj else filename.lower()

        # Find all matching evidence chunks for this entry
        matching_chunks = [c for c in chunk_annot_urls if c["stem"] in p_stem or p_stem in c["stem"]]

        # Extract full-PDF sentence index so EVERY citation occurrence in the draft can match its exact page & sentence
        sentence_index: list[dict[str, Any]] = []
        sent_df: dict[str, int] = {}
        if p_obj and p_obj.exists() and p_obj.suffix.lower() == ".pdf":
            sentence_index, sent_df = EvidenceHighlighter.extract_pdf_sentence_index(p_obj)

        # Track all distinct (page_num, quote, url) proofs actually used across inline citations + evidence
        used_proofs: list[dict[str, Any]] = []
        annot_cache: dict[tuple[int, str], str] = {}

        def _get_or_create_annot_url(p_num: int, q_text: str, occ_id: int) -> str:
            cache_key = (p_num, q_text[:60])
            if cache_key in annot_cache:
                return annot_cache[cache_key]
            if p_obj and p_obj.exists() and p_obj.suffix.lower() == ".pdf" and q_text:
                annot_pdf = EvidenceHighlighter.highlight_pdf_passage(
                    pdf_path=p_obj,
                    page_num=p_num,
                    quote_snippet=q_text,
                    output_dir=annotated_dir,
                    prefix_id=f"c{idx}_p{p_num}_{occ_id}",
                )
                tgt_f = annot_pdf if annot_pdf and annot_pdf.exists() else p_obj
                clean_fp = tgt_f.resolve().as_posix()
                if not clean_fp.startswith("/"):
                    clean_fp = f"/{clean_fp}"
                res_url = f"file://{clean_fp}#page={p_num}"
                annot_cache[cache_key] = res_url
                used_proofs.append({"page_num": p_num, "quote": q_text, "url": res_url})
                return res_url
            elif matching_chunks:
                return matching_chunks[0]["url"]
            elif p_obj and p_obj.exists():
                clean_fp = p_obj.resolve().as_posix()
                if not clean_fp.startswith("/"):
                    clean_fp = f"/{clean_fp}"
                return f"file://{clean_fp}#page={p_num}"
            return ""

        # 1. Hyperlink EVERY inline citation occurrence at the line/sentence/table-row level
        if key:
            clean_k = key.strip("()[]")
            k_esc = re.escape(clean_k)
            paren_regex = re.compile(r"(?<!\[)\(" + k_esc + r"\)(?!\])")
            narr_regex = None
            k_auth, k_yr = "", ""
            if ", " in clean_k:
                k_auth, k_yr = clean_k.split(", ", 1)
                narr_regex = re.compile(r"(?<!\[)" + re.escape(k_auth) + r"\s*\(" + re.escape(k_yr) + r"\)(?!\])")

            occ_counter = [0]
            body_lines = cleaned_text.split("\n")
            new_body_lines = []

            for b_line in body_lines:
                has_paren = bool(paren_regex.search(b_line))
                has_narr = bool(narr_regex and narr_regex.search(b_line))
                if not has_paren and not has_narr:
                    new_body_lines.append(b_line)
                    continue

                # Split line into sentences (unless it is a Markdown table row, where the whole row is the claim)
                is_table_row = b_line.lstrip().startswith("|")
                if is_table_row:
                    occ_counter[0] += 1
                    p_num, q_sent = EvidenceHighlighter.match_claim_to_pdf_sentence(
                        b_line, sentence_index, sent_df, fallback_chunks=matching_chunks
                    )
                    best_url = _get_or_create_annot_url(p_num, q_sent, occ_counter[0])
                    if best_url:
                        b_line = paren_regex.sub(f"[({clean_k} • Hal. {p_num})]({best_url})", b_line)
                        if narr_regex:
                            b_line = narr_regex.sub(f"[{k_auth} ({k_yr}, Hal. {p_num})]({best_url})", b_line)
                    new_body_lines.append(b_line)
                else:
                    # Process per sentence within the line so multiple citations in one paragraph map to their exact sentences
                    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\*\(\[])", b_line)
                    new_sents = []
                    for s_part in sents:
                        if paren_regex.search(s_part) or (narr_regex and narr_regex.search(s_part)):
                            occ_counter[0] += 1
                            p_num, q_sent = EvidenceHighlighter.match_claim_to_pdf_sentence(
                                s_part, sentence_index, sent_df, fallback_chunks=matching_chunks
                            )
                            best_url = _get_or_create_annot_url(p_num, q_sent, occ_counter[0])
                            if best_url:
                                s_part = paren_regex.sub(f"[({clean_k} • Hal. {p_num})]({best_url})", s_part)
                                if narr_regex:
                                    s_part = narr_regex.sub(f"[{k_auth} ({k_yr}, Hal. {p_num})]({best_url})", s_part)
                        new_sents.append(s_part)
                    new_body_lines.append(" ".join(new_sents))

            cleaned_text = "\n".join(new_body_lines)

        # 2. Format Bibliography links (combine distinct page proofs from inline claims + retrieved chunks)
        all_candidate_proofs = used_proofs + matching_chunks
        proof_links = []
        seen_pages = set()
        for mc in all_candidate_proofs:
            p_no = mc["page_num"]
            if p_no not in seen_pages:
                q_snip = f' > _"{mc["quote"][:120].strip()}"_' if mc["quote"] else ""
                proof_links.append(f"- [`Halaman {p_no} (Tersorot Stabilo)`]({mc['url']}){(' ' + q_snip) if q_snip else ''}")
                seen_pages.add(p_no)

        if proof_links:
            proof_str = "\n   " + "\n   ".join(sorted(proof_links, key=lambda x: int(re.search(r"Halaman (\d+)", x).group(1)) if re.search(r"Halaman (\d+)", x) else 999))
        else:
            default_url = f"file:///{p_obj.resolve().as_posix()}" if p_obj else ""
            proof_str = f"  \n   [`{filename}`]({default_url})" if default_url else ""

        if norm_style == "IEEE":
            lines.append(f"{ref_text}{proof_str}")
        else:
            lines.append(f"{idx}. **{key}** {ref_text}{proof_str}")

    return cleaned_text.rstrip() + "\n" + "\n".join(lines) + "\n"
