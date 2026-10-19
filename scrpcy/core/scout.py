"""Intelligent Literature Scout pipeline combining LLM query formulation, OpenAlex, and AI curation."""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from scrpcy.core.engine import LLMEngine
from scrpcy.core.models import DocumentMetadata, ParsedDocument
from scrpcy.core.parser import DocumentParser
from scrpcy.core.search import PaperSearcher, SearchResult, sanitize_filename
from scrpcy.core.workspace import WorkspaceManager
from scrpcy.exceptions import SearchError

try:
    import pyalex
    from pyalex import Works
    PYALEX_AVAILABLE = True
except ImportError:
    PYALEX_AVAILABLE = False


class ScoutQuery(BaseModel):
    """Structured search parameters extracted from user natural language query."""

    keywords: str = Field(..., description="Target academic search query and keywords in English/Indonesian.")
    year_start: int | None = Field(default=None, description="Start publication year filter (e.g. 2018).")
    year_end: int | None = Field(default=None, description="End publication year filter (e.g. 2020).")
    authors: list[str] = Field(default_factory=list, description="Author names if specified.")
    concepts: list[str] = Field(default_factory=list, description="Key conceptual domain tags.")
    is_oa_only: bool = Field(default=False, description="Filter solely for open-access papers.")
    summary: str = Field(default="", description="Brief synthesis of the user's research intent.")


class ScoutResult(BaseModel):
    """Normalized literature discovery candidate with verification badges."""

    title: str = Field(..., description="Formal title of the research paper.")
    authors: list[str] = Field(default_factory=list, description="Author names.")
    year: int | None = Field(default=None, description="Publication year.")
    doi: str | None = Field(default=None, description="Digital Object Identifier.")
    abstract: str | None = Field(default=None, description="Abstract summary.")
    pdf_url: str | None = Field(default=None, description="Direct URL to open-access PDF.")
    has_pdf: bool = Field(default=False, description="Whether open-access PDF is available.")
    citation_count: int | None = Field(default=None, description="Citation metrics count.")
    source: str = Field(default="openalex", description="Origin source badge (openalex, semanticscholar, arxiv, ai_recommended).")
    relevance_reason: str = Field(default="", description="AI explanation of why this paper matches the user's request.")
    cite_key: str = Field(default="", description="Citation key identifier.")

    def ensure_cite_key(self) -> str:
        """Derive standard citation key if not already present."""
        if self.cite_key:
            return self.cite_key
        author = "unknown"
        if self.authors:
            parts = self.authors[0].strip().split()
            if parts:
                author = re.sub(r"[^\w]", "", parts[-1].lower())
        year_str = str(self.year) if self.year else "nodate"
        slug = ""
        if self.title:
            words = [w for w in re.sub(r"[^\w\s]", "", self.title.lower()).split() if len(w) > 3]
            if words:
                slug = words[0]
        self.cite_key = f"{author}{year_str}{slug}"
        return self.cite_key


def reconstruct_inverted_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """Reconstruct full text abstract from OpenAlex inverted index dictionary."""
    if not inverted_index:
        return ""
    indexed_words: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            indexed_words.append((pos, word))
    indexed_words.sort(key=lambda item: item[0])
    return " ".join(item[1] for item in indexed_words)


class LiteratureScout:
    """Orchestrates natural language intent parsing, OpenAlex scraping, and AI curation."""

    def __init__(
        self,
        workspace: WorkspaceManager | None = None,
        searcher: PaperSearcher | None = None,
        parser: DocumentParser | None = None,
    ) -> None:
        self.workspace = workspace or WorkspaceManager()
        self.searcher = searcher or PaperSearcher(workspace=self.workspace)
        self.parser = parser or DocumentParser(workspace=self.workspace)

    def extract_search_criteria(
        self,
        user_input: str,
        engine: LLMEngine | None = None,
    ) -> ScoutQuery:
        """Extract structured query parameters from freeform natural language prompt.

        Args:
            user_input: Raw user request (e.g. 'saya butuh paper tentang kesehatan mental remaja 2018-2019').
            engine: Optional LLMEngine for structured AI parameter extraction.

        Returns:
            Validated ScoutQuery instance.
        """
        raw_text = user_input.strip()
        if not raw_text:
            return ScoutQuery(keywords="academic research", summary="General research inquiry")

        # Try LLM structured formulation if engine is available with an active API key
        if engine:
            model_name = engine.config.default_model
            api_key = engine.config.get_api_key_for_model(model_name)
            if api_key:
                try:
                    return self._extract_criteria_via_llm(raw_text, engine)
                except Exception:
                    pass

        # Robust heuristic fallback (works 100% offline without API key)
        return self._extract_criteria_heuristic(raw_text)

    def _extract_criteria_heuristic(self, text: str) -> ScoutQuery:
        """Heuristic rule-based extractor for years, open-access intent, and keywords."""
        cleaned = text.strip()

        # Extract 4-digit years (e.g. 2018-2019 or tahun 2020)
        years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", cleaned)]
        year_start: int | None = None
        year_end: int | None = None
        if len(years) >= 2:
            year_start = min(years[0], years[1])
            year_end = max(years[0], years[1])
        elif len(years) == 1:
            year_start = years[0]
            year_end = years[0]

        # Detect open access preference
        is_oa = bool(re.search(r"\b(open[\s\-]?access|oa|pdf|gratis|free)\b", cleaned, re.IGNORECASE))

        # Filter out common Indonesian and English conversational filler prefixes
        filler_patterns = [
            r"^(saya\s+)?(ingin|butuh|tolong\s+carikan|cari(kan)?|mau)\s+(paper|jurnal|artikel|makalah|penelitian)\s+(tentang|mengenai)?",
            r"^(i\s+)?(need|want|find\s+me|search\s+for|look\s+for)\s+(papers?|journals?|articles?)\s+(about|on|regarding)?",
            r"\b(tahun|year)\s+(19\d{2}|20\d{2})(\s*[-–—]\s*(19\d{2}|20\d{2}))?\b",
            r"\b(open[\s\-]?access|oa|pdf|gratis|free)\b",
        ]
        keyword_candidate = cleaned
        for pat in filler_patterns:
            keyword_candidate = re.sub(pat, "", keyword_candidate, flags=re.IGNORECASE).strip()

        keyword_candidate = re.sub(r"\s+", " ", keyword_candidate).strip()
        if not keyword_candidate:
            keyword_candidate = cleaned

        return ScoutQuery(
            keywords=keyword_candidate,
            year_start=year_start,
            year_end=year_end,
            is_oa_only=is_oa,
            summary=f"Inquiry for literature regarding '{keyword_candidate}'",
        )

    def _extract_criteria_via_llm(self, text: str, engine: LLMEngine) -> ScoutQuery:
        """Prompt LLM to extract formal academic search parameters as JSON."""
        import litellm

        system_prompt = (
            "You are an expert academic librarian. Analyze the user's natural language request for papers. "
            "Extract structured academic search parameters into JSON matching this schema:\n"
            "- keywords: Optimized English academic search keywords (e.g. 'adolescent mental health intervention').\n"
            "- year_start: Earliest publication year as integer or null.\n"
            "- year_end: Latest publication year as integer or null.\n"
            "- concepts: List of 2-4 broad academic field/concept tags.\n"
            "- is_oa_only: Boolean true if user explicitly asks for open access/free/pdf, else false.\n"
            "- summary: A concise 1-sentence summary of the research intent.\n"
            "Respond ONLY with a valid JSON object."
        )

        model_name = engine.config.default_model
        api_key = engine.config.get_api_key_for_model(model_name)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Request: {text}"},
        ]

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        if api_key:
            kwargs["api_key"] = api_key

        response = litellm.completion(**kwargs)
        raw_json = response.choices[0].message.content or "{}"
        clean_json = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", raw_json).strip()
        data = json.loads(clean_json)

        return ScoutQuery(
            keywords=data.get("keywords") or text,
            year_start=data.get("year_start"),
            year_end=data.get("year_end"),
            concepts=data.get("concepts") or [],
            is_oa_only=bool(data.get("is_oa_only", False)),
            summary=data.get("summary") or text,
        )

    def search_openalex(self, query: ScoutQuery, limit: int = 15) -> list[ScoutResult]:
        """Query OpenAlex using pyalex with full text search and metadata filters."""
        if not PYALEX_AVAILABLE:
            return []

        try:
            works_query = Works().search(query.keywords)

            # Apply publication year filter
            if query.year_start and query.year_end:
                if query.year_start == query.year_end:
                    works_query = works_query.filter(publication_year=query.year_start)
                else:
                    works_query = works_query.filter(
                        publication_year=f">{query.year_start - 1}",
                    )
            elif query.year_start:
                works_query = works_query.filter(publication_year=f">{query.year_start - 1}")
            elif query.year_end:
                works_query = works_query.filter(publication_year=f"<{query.year_end + 1}")

            if query.is_oa_only:
                works_query = works_query.filter(is_oa=True)

            raw_records = works_query.get()
        except Exception:
            return []

        results: list[ScoutResult] = []
        for w in raw_records[:limit]:
            title = w.get("title")
            if not title:
                continue

            authors = [
                a.get("author", {}).get("display_name")
                for a in w.get("authorships", [])
                if a.get("author", {}).get("display_name")
            ]
            year = w.get("publication_year")
            doi_raw = w.get("doi")
            doi_clean = doi_raw.replace("https://doi.org/", "") if doi_raw else None

            oa_info = w.get("open_access", {}) or {}
            is_oa = bool(oa_info.get("is_oa", False))
            pdf_url = oa_info.get("oa_url")
            citation_count = w.get("cited_by_count")

            abstract = reconstruct_inverted_abstract(w.get("abstract_inverted_index"))

            item = ScoutResult(
                title=title,
                authors=authors,
                year=year,
                doi=doi_clean,
                abstract=abstract,
                pdf_url=pdf_url,
                has_pdf=is_oa and bool(pdf_url),
                citation_count=citation_count,
                source="openalex",
                relevance_reason="Matches OpenAlex global scholarly catalog.",
            )
            item.ensure_cite_key()
            results.append(item)

        return results

    async def search_multi_source(
        self,
        query: ScoutQuery,
        limit: int = 15,
    ) -> list[ScoutResult]:
        """Execute parallel search across OpenAlex, Semantic Scholar, and ArXiv."""
        loop = asyncio.get_running_loop()

        # Run pyalex in worker thread since pyalex is synchronous
        openalex_task = loop.run_in_executor(None, self.search_openalex, query, limit)
        s2_task = self.searcher.search(query.keywords, limit=limit)

        openalex_res, s2_res = await asyncio.gather(
            openalex_task,
            s2_task,
            return_exceptions=True,
        )

        candidates: list[ScoutResult] = []

        if isinstance(openalex_res, list):
            candidates.extend(openalex_res)

        if isinstance(s2_res, list):
            for r in s2_res:
                # Apply year filtering if specified
                if query.year_start and r.year and r.year < query.year_start:
                    continue
                if query.year_end and r.year and r.year > query.year_end:
                    continue
                if query.is_oa_only and not r.has_pdf:
                    continue

                item = ScoutResult(
                    title=r.title,
                    authors=r.authors,
                    year=r.year,
                    doi=r.doi,
                    abstract=r.abstract,
                    pdf_url=r.pdf_url,
                    has_pdf=r.has_pdf,
                    source="semanticscholar",
                    relevance_reason="Found via Semantic Scholar / ArXiv index.",
                )
                item.ensure_cite_key()
                candidates.append(item)

        # Deduplicate candidates by normalized title
        seen_titles: set[str] = set()
        deduped: list[ScoutResult] = []
        for c in candidates:
            norm = re.sub(r"[^\w]", "", c.title.lower())
            if norm and norm not in seen_titles:
                seen_titles.add(norm)
                deduped.append(c)

        return deduped[:limit]

    def curate_and_rank(
        self,
        user_input: str,
        candidates: list[ScoutResult],
        engine: LLMEngine | None = None,
        max_output: int = 8,
    ) -> list[ScoutResult]:
        """Rank candidates according to user intent and optionally inject seminal papers."""
        if not candidates:
            return []

        # If LLM engine with active key is available, use AI ranking and curation
        if engine:
            model_name = engine.config.default_model
            api_key = engine.config.get_api_key_for_model(model_name)
            if api_key:
                try:
                    return self._curate_via_llm(user_input, candidates, engine, max_output)
                except Exception:
                    pass

        # Offline ranking heuristic: sort by citation count and open-access availability
        sorted_candidates = sorted(
            candidates,
            key=lambda c: (c.has_pdf, c.citation_count or 0),
            reverse=True,
        )
        return sorted_candidates[:max_output]

    def _curate_via_llm(
        self,
        user_input: str,
        candidates: list[ScoutResult],
        engine: LLMEngine,
        max_output: int,
    ) -> list[ScoutResult]:
        """Prompt LLM to evaluate relevance, provide reasons, and add seminal literature."""
        import litellm

        brief_catalog = []
        for idx, c in enumerate(candidates, 1):
            brief_catalog.append({
                "index": idx,
                "title": c.title,
                "authors": c.authors[:2],
                "year": c.year,
                "abstract": (c.abstract[:200] + "...") if c.abstract else "No abstract",
            })

        system_prompt = (
            "You are an academic literature curator. The user requested: '{user_input}'.\n"
            "Below is a list of real candidates retrieved from academic databases.\n"
            "Select and rank up to {max_output} most relevant papers.\n"
            "For each paper selected, provide a 1-sentence relevance reason.\n"
            "If there is a classic, seminal, or groundbreaking paper in this exact field that was NOT in the list, "
            "you may also append it as a recommendation with 'is_ai_seminal': true.\n"
            "Return valid JSON array of objects with keys:\n"
            "- candidate_index: integer corresponding to candidate list (null if seminal recommendation)\n"
            "- title: string\n"
            "- authors: list of strings\n"
            "- year: integer\n"
            "- relevance_reason: string\n"
            "- is_ai_seminal: boolean"
        ).format(user_input=user_input, max_output=max_output)

        model_name = engine.config.default_model
        api_key = engine.config.get_api_key_for_model(model_name)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Candidates:\n{json.dumps(brief_catalog, indent=2)}"},
        ]

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        if api_key:
            kwargs["api_key"] = api_key

        response = litellm.completion(**kwargs)
        raw_json = response.choices[0].message.content or "{}"
        clean_json = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", raw_json).strip()
        data = json.loads(clean_json)

        # Handle object wrapping array
        items = data.get("papers") or data.get("results") or data if isinstance(data, list) else []
        if isinstance(data, dict) and not items:
            for v in data.values():
                if isinstance(v, list):
                    items = v
                    break

        final_curated: list[ScoutResult] = []
        for entry in items:
            c_idx = entry.get("candidate_index")
            if c_idx and isinstance(c_idx, int) and 1 <= c_idx <= len(candidates):
                base = candidates[c_idx - 1]
                base.relevance_reason = entry.get("relevance_reason") or base.relevance_reason
                final_curated.append(base)
            elif entry.get("is_ai_seminal"):
                # Seminal paper recommended by AI knowledge
                seminal = ScoutResult(
                    title=entry.get("title") or "Seminal Literature Recommendation",
                    authors=entry.get("authors") or ["Seminal Authors"],
                    year=entry.get("year"),
                    source="ai_recommended",
                    has_pdf=False,
                    relevance_reason=entry.get("relevance_reason") or "Seminal groundbreaking paper recommended by AI.",
                )
                seminal.ensure_cite_key()
                final_curated.append(seminal)

        return final_curated or candidates[:max_output]

    async def download_scout_paper(self, paper: ScoutResult) -> Path:
        """Download open-access paper PDF into the master project vault and cache it."""
        if not paper.has_pdf or not paper.pdf_url:
            raise SearchError(f"No open-access PDF URL available for '{paper.title}'")

        self.workspace.ensure_directories()
        safe_stem = sanitize_filename(paper.title)
        target_pdf = self.workspace.vault_dir / f"{safe_stem}.pdf"

        # Stream download directly
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(paper.pdf_url)
            if resp.status_code != 200:
                raise SearchError(f"Failed to download PDF ({resp.status_code}) from {paper.pdf_url}")
            target_pdf.write_bytes(resp.content)

        # Parse or create markdown frontmatter cache
        try:
            self.parser.parse(target_pdf, save_to_cache=True)
        except Exception:
            meta = DocumentMetadata(
                title=paper.title,
                authors=paper.authors,
                year=paper.year,
                doi=paper.doi,
                cite_key=paper.cite_key or safe_stem,
            )
            clean_md = f"# {paper.title}\n\nAbstract:\n{paper.abstract or ''}\n"
            cache_file = f"{safe_stem}.md"
            self.workspace.save_to_cache(cache_file, f"{meta.to_frontmatter()}\n{clean_md}")

        return target_pdf
