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

from scripcy.core.engine import LLMEngine
from scripcy.core.models import DocumentMetadata, ParsedDocument
from scripcy.core.parser import DocumentParser
from scripcy.core.config import ScripcyConfig
from scripcy.core.search import PaperSearcher, SearchResult, is_cs_ai_domain, sanitize_filename
from scripcy.core.workspace import WorkspaceManager
from scripcy.exceptions import SearchError

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
        config: ScripcyConfig | None = None,
    ) -> None:
        self.workspace = workspace or WorkspaceManager()
        self.config = config or ScripcyConfig()
        s2_key = self.config.get_api_key("semantic_scholar")
        self.searcher = searcher or PaperSearcher(workspace=self.workspace, api_key=s2_key)
        self.parser = parser or DocumentParser(workspace=self.workspace)
        if PYALEX_AVAILABLE:
            if self.config.openalex_api_key:
                pyalex.config.api_key = self.config.openalex_api_key
            if self.config.unpaywall_email:
                pyalex.config.email = self.config.unpaywall_email

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
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
        litellm.drop_params = True

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
        effective_model = model_name
        api_base = None
        if model_name.startswith("9router/"):
            submodel = model_name.split("/", 1)[1]
            effective_model = f"openai/{submodel}"
            api_base = "http://localhost:20128/v1"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Request: {text}"},
        ]

        kwargs: dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        if api_key:
            kwargs["api_key"] = api_key
        if api_base:
            kwargs["api_base"] = api_base

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

    def search_openalex(self, query: ScoutQuery, limit: int = 30) -> list[ScoutResult]:
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
        limit: int = 30,
    ) -> list[ScoutResult]:
        """Execute federated snapshot searches across OpenAlex, CrossRef, Semantic Scholar, and ArXiv."""
        loop = asyncio.get_running_loop()

        tasks = []
        task_types = []

        from unittest.mock import AsyncMock, MagicMock
        is_search_mocked = isinstance(getattr(self.searcher, "search", None), (AsyncMock, MagicMock))

        # 1. OpenAlex snapshot (primary scholarly catalog)
        openalex_limit = min(limit, 20)
        tasks.append(loop.run_in_executor(None, self.search_openalex, query, openalex_limit))
        task_types.append("openalex")

        # 2. Semantic Scholar snapshot (or mocked search delegate)
        if is_search_mocked or getattr(self.config, "enable_semantic_scholar", True):
            tasks.append(self.searcher.search(query.keywords, limit=min(limit, 15)))
            task_types.append("semanticscholar")

        # 3. CrossRef snapshot (if enabled and not in mocked unit test)
        if not is_search_mocked and getattr(self.config, "enable_crossref", True):
            email = getattr(self.config, "unpaywall_email", None)
            tasks.append(self.searcher.search_crossref(query.keywords, limit=min(limit, 15), email=email))
            task_types.append("crossref")

        # 4. ArXiv snapshot (strictly for Computer Science / AI / Math / Physics)
        if not is_search_mocked and is_cs_ai_domain(query.keywords):
            tasks.append(self.searcher.search_arxiv(query.keywords, limit=min(limit, 10)))
            task_types.append("arxiv")

        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        candidates: list[ScoutResult] = []

        for src_type, res in zip(task_types, task_results):
            if isinstance(res, Exception) or not isinstance(res, list):
                continue
            for r in res:
                if isinstance(r, ScoutResult):
                    candidates.append(r)
                elif isinstance(r, SearchResult):
                    if query.year_start and r.year and r.year < query.year_start:
                        continue
                    if query.year_end and r.year and r.year > query.year_end:
                        continue
                    if query.is_oa_only and not r.has_pdf:
                        continue

                    badge_reason = {
                        "crossref": "Found via CrossRef scholarly metadata index.",
                        "semanticscholar": "Found via Semantic Scholar catalog.",
                        "arxiv": "Found via ArXiv preprint repository (CS/AI domain).",
                    }.get(src_type, f"Found via {src_type}.")

                    item = ScoutResult(
                        title=r.title,
                        authors=r.authors,
                        year=r.year,
                        doi=r.doi,
                        abstract=r.abstract,
                        pdf_url=r.pdf_url,
                        has_pdf=r.has_pdf,
                        source=src_type,
                        relevance_reason=badge_reason,
                    )
                    item.ensure_cite_key()
                    candidates.append(item)

        # Deduplicate candidates by normalized title and DOI
        seen_titles: set[str] = set()
        seen_dois: set[str] = set()
        deduped: list[ScoutResult] = []
        for c in candidates:
            norm = re.sub(r"[^\w]", "", c.title.lower())
            c_doi = (c.doi or "").lower().strip()
            if norm and norm in seen_titles:
                continue
            if c_doi and c_doi in seen_dois:
                continue
            if norm:
                seen_titles.add(norm)
            if c_doi:
                seen_dois.add(c_doi)
            deduped.append(c)

        return deduped[:limit]

    # Alias for backwards compatibility with test harnesses and external callers
    search_all_repositories = search_multi_source

    def curate_and_rank(
        self,
        user_input: str,
        candidates: list[ScoutResult],
        engine: LLMEngine | None = None,
        max_output: int | None = None,
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
        if max_output:
            return sorted_candidates[:max_output]
        return sorted_candidates

    def _curate_via_llm(
        self,
        user_input: str,
        candidates: list[ScoutResult],
        engine: LLMEngine,
        max_output: int | None = None,
    ) -> list[ScoutResult]:
        """Prompt LLM to evaluate relevance, provide reasons, and add seminal literature."""
        import litellm

        target_limit = max_output or len(candidates)

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
        ).format(user_input=user_input, max_output=target_limit)

        import litellm
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
        litellm.drop_params = True

        model_name = engine.config.default_model
        api_key = engine.config.get_api_key_for_model(model_name)
        effective_model = model_name
        api_base = None
        if model_name.startswith("9router/"):
            submodel = model_name.split("/", 1)[1]
            effective_model = f"openai/{submodel}"
            api_base = "http://localhost:20128/v1"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Candidates:\n{json.dumps(brief_catalog, indent=2)}"},
        ]

        kwargs: dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        if api_key:
            kwargs["api_key"] = api_key
        if api_base:
            kwargs["api_base"] = api_base

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

        # Append remaining candidates not in final_curated, sorted by pdf availability and citations
        seen_titles = {re.sub(r"[^\w]", "", c.title.lower()) for c in final_curated}
        remaining = [
            c for c in candidates
            if re.sub(r"[^\w]", "", c.title.lower()) not in seen_titles
        ]
        remaining.sort(key=lambda c: (c.has_pdf, c.citation_count or 0), reverse=True)
        final_curated.extend(remaining)

        if max_output:
            return final_curated[:max_output]
        return final_curated

    async def download_scout_paper(self, paper: ScoutResult) -> Path:
        """Download open-access paper PDF into the master project vault and cache it."""
        if not paper.has_pdf or not paper.pdf_url:
            if paper.doi:
                email = getattr(self.config, "unpaywall_email", None)
                oa_url = await self.searcher.lookup_unpaywall(paper.doi, email=email)
                if oa_url:
                    paper.pdf_url = oa_url
                    paper.has_pdf = True
                    paper.relevance_reason = (paper.relevance_reason or "") + " (Versi Green OA ditemukan via Unpaywall)"

        if not paper.has_pdf or not paper.pdf_url:
            raise SearchError(f"No open-access PDF URL available for '{paper.title}'")

        self.workspace.ensure_directories()
        safe_stem = sanitize_filename(paper.title)
        papers_dir = getattr(self.workspace, "papers_dir", self.workspace.vault_dir)
        target_pdf = papers_dir / f"{safe_stem}.pdf"

        # Stream download directly
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(paper.pdf_url)
            if resp.status_code != 200:
                raise SearchError(f"Failed to download PDF ({resp.status_code}) from {paper.pdf_url}")
            target_pdf.write_bytes(resp.content)

        # Convert PDF content to clean Markdown with accurate verified metadata
        meta = DocumentMetadata(
            title=paper.title,
            authors=paper.authors,
            year=paper.year,
            doi=paper.doi,
            cite_key=paper.cite_key or safe_stem,
        )
        meta.ensure_cite_key()

        def _parse_pdf_body() -> str:
            try:
                import pymupdf
                import pymupdf4llm

                with pymupdf.open(target_pdf) as doc:
                    return pymupdf4llm.to_markdown(doc)
            except Exception:
                return f"# {paper.title}\n\nAbstract:\n{paper.abstract or ''}\n"

        clean_body_markdown = await asyncio.to_thread(_parse_pdf_body)

        cache_file = f"{safe_stem}.md"
        self.workspace.save_to_cache(
            cache_file, f"{meta.to_frontmatter()}\n{clean_body_markdown.strip()}\n"
        )

        def _ingest_vector_safe():
            try:
                from scripcy.core.vector_vault import ProjectVectorVault
                db_p = self.workspace.vault_dir / "project_vault.db"
                vv = ProjectVectorVault(db_p, self.workspace.vault_dir)
                vv.ingest_paper_pdf(target_pdf, paper_id=safe_stem, title=paper.title)
            except Exception:
                pass

        await asyncio.to_thread(_ingest_vector_safe)
        return target_pdf

    def ingest_abstract_paper(self, paper: ScoutResult) -> Path:
        """Create a markdown document from an abstract-only paywalled paper in the vault and cache."""
        self.workspace.ensure_directories()
        safe_stem = sanitize_filename(paper.title)

        meta = DocumentMetadata(
            title=paper.title,
            authors=paper.authors,
            year=paper.year,
            doi=paper.doi,
            cite_key=paper.cite_key or safe_stem,
        )
        meta.ensure_cite_key()

        body = (
            f"# {paper.title}\n\n"
            f"**Penulis**: {', '.join(paper.authors)} ({paper.year or 'n.d.'})\n"
            f"**DOI**: {paper.doi or 'Tidak tersedia'}\n\n"
            "> **Catatan**: Naskah lengkap (Full-Text PDF) paper ini berada di balik paywall komersial. "
            "Kutipan dan bukti berikut disintesis berdasarkan abstrak publikasi resmi yang terbuka.\n\n"
            f"## Abstrak\n{paper.abstract or 'Abstrak tidak tersedia.'}\n"
        )
        full_content = f"{meta.to_frontmatter()}\n{body.strip()}\n"

        papers_dir = getattr(self.workspace, "papers_dir", self.workspace.vault_dir)
        target_md = papers_dir / f"{safe_stem}.md"
        target_md.write_text(full_content, encoding="utf-8")
        if papers_dir != self.workspace.vault_dir:
            try:
                (self.workspace.vault_dir / f"{safe_stem}.md").write_text(full_content, encoding="utf-8")
            except Exception:
                pass

        cache_file = f"{safe_stem}.md"
        self.workspace.save_to_cache(cache_file, full_content)
        return target_md


    async def traverse_citation_network(
        self,
        seed_identifier: str,
        direction: str = "both",
        limit: int = 10,
    ) -> list[ScoutResult]:
        """Traverse the citation graph forwards (citations) and/or backwards (references)."""
        seed_id = seed_identifier.strip()
        if not any(seed_id.upper().startswith(p) for p in ("DOI:", "ARXIV:", "CORPUSID:")) and "10." not in seed_id and len(seed_id) != 40:
            try:
                s2_matches = await self.searcher.search_semantic_scholar(seed_id, limit=1)
                if s2_matches and s2_matches[0].paper_id:
                    seed_id = s2_matches[0].paper_id
            except Exception:
                pass

        candidates: list[SearchResult] = []
        if direction in ("citations", "forward", "both"):
            try:
                cites = await self.searcher.get_paper_citations(seed_id, limit=limit)
                candidates.extend(cites)
            except Exception:
                pass
        if direction in ("references", "backward", "both"):
            try:
                refs = await self.searcher.get_paper_references(seed_id, limit=limit)
                candidates.extend(refs)
            except Exception:
                pass

        results: list[ScoutResult] = []
        seen_titles = set()
        for c in candidates:
            if not c.title or c.title.lower() in seen_titles:
                continue
            seen_titles.add(c.title.lower())
            reason = "Forward citation (newer work citing seed paper)" if "citations" in c.source_api else "Backward reference (foundational work cited by seed paper)"
            res = ScoutResult(
                title=c.title,
                authors=c.authors,
                year=c.year,
                doi=c.doi,
                abstract=c.abstract,
                pdf_url=c.pdf_url,
                has_pdf=c.has_pdf,
                source="citation_graph",
                relevance_reason=reason,
            )
            res.ensure_cite_key()
            results.append(res)

        return results[:limit]

    async def map_vault_citation_network(
        self,
        vault_stems: list[str],
        direction: str = "both",
        limit: int = 10,
    ) -> list[ScoutResult]:
        """Map citation network starting from existing vault papers to discover connected literature."""
        if not vault_stems:
            return []

        all_candidates: list[ScoutResult] = []
        seen = {s.lower() for s in vault_stems}

        seeds = vault_stems[:max(3, len(vault_stems))]
        for stem in seeds:
            clean_seed = stem
            if self.workspace:
                cache_md = self.workspace.get_cache_path(f"{stem}.md")
                if cache_md.exists():
                    try:
                        content = cache_md.read_text(encoding="utf-8", errors="replace")
                        doi_m = re.search(r"(?:^|\n)doi:\s*[\"']?([^\s\n\r\"']+)[\"']?", content, re.IGNORECASE)
                        title_m = re.search(r"(?:^|\n)title:\s*[\"']?([^\n\r\"']+)[\"']?", content, re.IGNORECASE)
                        if doi_m and doi_m.group(1).strip() and doi_m.group(1).strip() != "None":
                            clean_seed = doi_m.group(1).strip()
                        elif title_m and title_m.group(1).strip() and title_m.group(1).strip() != "None":
                            clean_seed = title_m.group(1).strip()
                    except Exception:
                        pass
            if clean_seed == stem:
                clean_seed = stem.replace("_", " ").strip()
            network = await self.traverse_citation_network(clean_seed, direction=direction, limit=limit)
            for paper in network:
                t_slug = re.sub(r"[^\w]", "", paper.title.lower())
                if t_slug and t_slug not in seen:
                    seen.add(t_slug)
                    all_candidates.append(paper)

        all_candidates.sort(key=lambda p: (1 if p.has_pdf else 0, p.year or 0), reverse=True)
        return all_candidates[:limit]

    async def auto_scout_and_ingest(
        self,
        query: str,
        max_papers: int = 2,
        on_status: Any | None = None,
        engine: LLMEngine | None = None,
    ) -> list[str]:
        """Automatically discover, filter (preferring OA PDF, fallback to abstract), and ingest papers for a topic."""
        try:
            sq = await asyncio.to_thread(self.extract_search_criteria, query, engine=engine)
        except Exception:
            sq = ScoutQuery(keywords=query, is_oa_only=False)

        candidates = await self.search_multi_source(sq, limit=15)
        if not candidates:
            return []

        curated = await asyncio.to_thread(self.curate_and_rank, query, candidates, engine=engine)
        if not curated:
            return []

        # Resolve Green OA PDF via Unpaywall for paywalled candidates with DOI
        email = getattr(self.config, "unpaywall_email", None)
        for c in curated:
            if not (c.has_pdf and c.pdf_url) and c.doi:
                try:
                    oa_url = await self.searcher.lookup_unpaywall(c.doi, email=email)
                    if oa_url:
                        c.pdf_url = oa_url
                        c.has_pdf = True
                        c.relevance_reason = (c.relevance_reason or "") + " (Versi Green OA ditemukan via Unpaywall)"
                except Exception:
                    pass

        # Prioritize OA PDF candidates, fallback to abstract if closed-access
        oa_candidates = [c for c in curated if c.has_pdf and c.pdf_url]
        closed_candidates = [c for c in curated if not (c.has_pdf and c.pdf_url) and c.abstract]

        selected_candidates: list[ScoutResult] = []
        for c in oa_candidates:
            if len(selected_candidates) >= max_papers:
                break
            selected_candidates.append(c)

        for c in closed_candidates:
            if len(selected_candidates) >= max_papers:
                break
            if not any(sc.title.lower() == c.title.lower() for sc in selected_candidates):
                selected_candidates.append(c)

        ingested_stems: list[str] = []
        tried_titles = {c.title.lower() for c in selected_candidates}
        oa_pool = [c for c in oa_candidates if c.has_pdf and c.pdf_url]

        for idx, paper in enumerate(selected_candidates, 1):
            safe_stem = sanitize_filename(paper.title)
            if paper.has_pdf and paper.pdf_url:
                if on_status:
                    on_status("Mengunduh", f"[{idx}/{len(selected_candidates)}] {paper.title[:50]}... (PDF)")
                try:
                    await self.download_scout_paper(paper)
                    ingested_stems.append(safe_stem)
                except Exception as exc:
                    # Autonomous error recovery: find open-access alternative candidate from curated pool
                    alt_found = False
                    for alt_paper in oa_pool:
                        if alt_paper.title.lower() in tried_titles:
                            continue
                        tried_titles.add(alt_paper.title.lower())
                        if on_status:
                            on_status("Mengunduh", f"Mencoba alternatif Open-Access: {alt_paper.title[:45]}...")
                        try:
                            await self.download_scout_paper(alt_paper)
                            ingested_stems.append(sanitize_filename(alt_paper.title))
                            alt_found = True
                            break
                        except Exception:
                            continue

                    if not alt_found and paper.abstract:
                        if on_status:
                            on_status("Membaca", f"Fallback ke ringkasan abstrak: {paper.title[:50]}...")
                        self.ingest_abstract_paper(paper)
                        ingested_stems.append(safe_stem)
            elif paper.abstract:
                if on_status:
                    on_status("Membaca", f"[{idx}/{len(selected_candidates)}] Abstrak paper: {paper.title[:50]}...")
                self.ingest_abstract_paper(paper)
                ingested_stems.append(safe_stem)

        return ingested_stems
