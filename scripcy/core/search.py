"""Academic paper search and PDF retrieval across Semantic Scholar and ArXiv."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Literal
import xml.etree.ElementTree as ET

import httpx
from pydantic import BaseModel, ConfigDict, Field, computed_field

from scripcy.core.workspace import WorkspaceManager
from scripcy.exceptions import (
    DownloadError,
    NoOpenAccessPDFError,
    RateLimitError,
    SearchError,
)

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
ARXIV_QUERY_URL = "https://export.arxiv.org/api/query"
CROSSREF_WORKS_URL = "https://api.crossref.org/works"
UNPAYWALL_API_URL = "https://api.unpaywall.org/v2"
OPENCITATIONS_COCI_URL = "https://opencitations.net/index/coci/api/v1/citations"

CS_AI_KEYWORDS = {
    "computer science", "artificial intelligence", "machine learning", "deep learning",
    "neural", "llm", "transformer", "nlp", "natural language", "computer vision",
    "algorithm", "cryptography", "reinforcement learning", "quantum", "physics",
    "mathematics", "cybersecurity", "software", "database", "distributed systems",
    "compiler", "operating system", "data science", "robotics", "speech recognition",
    "generative ai", "image processing", "graph neural", "information retrieval",
    "benchmark", "convolutional", "recurrent", "lstm", "encoder", "decoder", "arxiv",
    "network", "networks", "residual", "resnet",
    "yolo", "yolos", "vit", "vision transformer", "detr", "rcnn", "cnn", "attention",
    "deteksi objek", "kecerdasan buatan", "pembelajaran mesin", "pembelajaran mendalam",
    "penglihatan komputer", "klasifikasi citra", "segmentasi citra", "jaringan saraf",
    "object detection", "image classification", "feature extraction",
}


def is_cs_ai_domain(query: str) -> bool:
    """Check whether a search query falls within Computer Science, AI, Math, or Physics domains."""
    q = query.lower()
    return any(re.search(rf"\b{re.escape(k)}\b", q) for k in CS_AI_KEYWORDS)

ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def sanitize_filename(name: str) -> str:
    """Produce a safe filename for local storage."""
    cleaned = re.sub(r"[^\w\-_\.]", "_", name.strip())
    cleaned = re.sub(r"__+", "_", cleaned)
    return cleaned[:80] or "paper"


class SearchResult(BaseModel):
    """Structured academic paper search metadata."""

    model_config = ConfigDict(extra="ignore")

    title: str = Field(description="Full title of the paper.")
    authors: list[str] = Field(default_factory=list, description="List of author names.")
    abstract: str | None = Field(default=None, description="Abstract summary if available.")
    year: int | None = Field(default=None, description="Publication or preprint year.")
    doi: str | None = Field(default=None, description="Digital Object Identifier.")
    pdf_url: str | None = Field(default=None, description="Direct URL to open-access PDF.")
    source_api: str = Field(default="semantic_scholar", description="API source (semantic_scholar or arxiv).")
    paper_id: str | None = Field(default=None, description="Unique identifier from source API.")
    cite_key: str | None = Field(default=None, description="Citation key for reference.")

    @computed_field
    @property
    def has_pdf(self) -> bool:
        """Check whether an open-access PDF download URL is available."""
        return bool(self.pdf_url and self.pdf_url.strip())

    def ensure_cite_key(self) -> str:
        """Return existing cite key or compute a deterministic key."""
        if self.cite_key:
            return self.cite_key

        first_author = "unknown"
        if self.authors:
            parts = self.authors[0].strip().split()
            if parts:
                first_author = re.sub(r"[^\w]", "", parts[-1].lower())

        year_str = str(self.year) if self.year else "nodate"
        slug = ""
        if self.title:
            words = [w for w in re.sub(r"[^\w\s]", "", self.title.lower()).split() if len(w) > 3]
            if words:
                slug = words[0]

        generated = f"{first_author}{year_str}{slug}"
        self.cite_key = generated
        return generated


class PaperSearcher:
    """Asynchronous client for searching academic literature and downloading open-access PDFs."""

    def __init__(
        self,
        workspace: WorkspaceManager | None = None,
        client: httpx.AsyncClient | None = None,
        api_key: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.workspace = workspace or WorkspaceManager()
        self._client = client
        self._owns_client = client is None
        self.api_key = api_key
        self.timeout = timeout

    async def _get_client(self) -> httpx.AsyncClient:
        """Return existing HTTP client or instantiate a managed client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
            self._owns_client = True
        return self._client

    async def close(self) -> None:
        """Close managed HTTP client resources."""
        if self._owns_client and self._client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> PaperSearcher:
        await self._get_client()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def search(
        self,
        query: str,
        limit: int = 10,
        source: Literal["auto", "semantic_scholar", "arxiv"] = "auto",
    ) -> list[SearchResult]:
        """Search academic papers, falling back from Semantic Scholar to ArXiv when needed."""
        clean_query = query.strip()
        if not clean_query:
            return []

        if source == "semantic_scholar":
            return await self.search_semantic_scholar(clean_query, limit=limit)
        if source == "arxiv":
            return await self.search_arxiv(clean_query, limit=limit)
        if source == "openalex":
            return await self.search_openalex(clean_query, limit=limit)

        # Automatic fallback: 1) Semantic Scholar, 2) OpenAlex global catalog, 3) ArXiv
        try:
            results = await self.search_semantic_scholar(clean_query, limit=limit)
            if results:
                return results
        except Exception:
            pass

        try:
            results = await self.search_openalex(clean_query, limit=limit)
            if results:
                return results
        except Exception:
            pass

        if is_cs_ai_domain(clean_query):
            return await self.search_arxiv(clean_query, limit=limit)
        return []

    async def search_semantic_scholar(
        self,
        query: str,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Query Semantic Scholar Graph API for matching literature."""
        client = await self._get_client()
        params = {
            "query": query,
            "limit": limit,
            "fields": "paperId,title,abstract,year,authors,externalIds,openAccessPdf,url",
        }
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            response = await client.get(
                SEMANTIC_SCHOLAR_SEARCH_URL,
                params=params,
                headers=headers,
            )
        except httpx.TimeoutException as exc:
            raise SearchError(f"Semantic Scholar request timed out: {exc}") from exc
        except httpx.RequestError as exc:
            raise SearchError(f"Semantic Scholar connection error: {exc}") from exc

        if response.status_code == 429:
            raise RateLimitError("Semantic Scholar rate limit reached (HTTP 429).")
        if response.status_code >= 400:
            raise SearchError(f"Semantic Scholar API returned HTTP {response.status_code}: {response.text}")

        try:
            data = response.json().get("data", [])
        except Exception as exc:
            raise SearchError(f"Failed to parse Semantic Scholar JSON: {exc}") from exc

        results: list[SearchResult] = []
        for item in data:
            title = (item.get("title") or "").strip()
            if not title:
                continue

            authors = [
                a.get("name", "").strip()
                for a in item.get("authors", [])
                if a.get("name") and a.get("name").strip()
            ]

            ext_ids = item.get("externalIds") or {}
            doi = ext_ids.get("DOI")
            arxiv_id = ext_ids.get("ArXiv")

            open_access = item.get("openAccessPdf") or {}
            pdf_url = open_access.get("url") if isinstance(open_access, dict) else None
            if not pdf_url and arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            result = SearchResult(
                title=title,
                authors=authors,
                abstract=item.get("abstract"),
                year=item.get("year"),
                doi=doi,
                pdf_url=pdf_url,
                source_api="semantic_scholar",
                paper_id=item.get("paperId"),
            )
            result.ensure_cite_key()
            results.append(result)

        return results

    async def batch_get_papers_semantic_scholar(
        self,
        paper_ids: list[str],
    ) -> list[SearchResult]:
        """Fetch metadata for multiple papers in bulk via Semantic Scholar Batch API."""
        if not paper_ids:
            return []

        client = await self._get_client()
        params = {
            "fields": "paperId,title,abstract,year,authors,externalIds,openAccessPdf,url",
        }
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        results: list[SearchResult] = []
        chunk_size = 100
        for i in range(0, len(paper_ids), chunk_size):
            chunk = paper_ids[i : i + chunk_size]
            try:
                response = await client.post(
                    SEMANTIC_SCHOLAR_BATCH_URL,
                    json={"ids": chunk},
                    params=params,
                    headers=headers,
                )
                if response.status_code == 200:
                    items = response.json()
                    for item in items:
                        if not item:
                            continue
                        title = (item.get("title") or "").strip()
                        if not title:
                            continue
                        authors = [
                            a.get("name", "").strip()
                            for a in item.get("authors", [])
                            if a.get("name") and a.get("name").strip()
                        ]
                        ext_ids = item.get("externalIds") or {}
                        doi = ext_ids.get("DOI")
                        arxiv_id = ext_ids.get("ArXiv")
                        open_access = item.get("openAccessPdf") or {}
                        pdf_url = open_access.get("url") if isinstance(open_access, dict) else None
                        if not pdf_url and arxiv_id:
                            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

                        res = SearchResult(
                            title=title,
                            authors=authors,
                            abstract=item.get("abstract"),
                            year=item.get("year"),
                            doi=doi,
                            pdf_url=pdf_url,
                            source_api="semantic_scholar",
                            paper_id=item.get("paperId"),
                        )
                        res.ensure_cite_key()
                        results.append(res)
            except Exception:
                continue
        return results

    async def get_paper_citations(
        self,
        paper_id_or_doi: str,
        limit: int = 15,
    ) -> list[SearchResult]:
        """Fetch papers citing the specified seed paper via Semantic Scholar Graph API."""
        client = await self._get_client()
        clean_id = paper_id_or_doi.strip()
        encoded = f"DOI:{clean_id}" if ("10." in clean_id and not clean_id.startswith("DOI:")) else clean_id
        url = f"https://api.semanticscholar.org/graph/v1/paper/{encoded}/citations"
        params = {
            "limit": limit,
            "fields": "citingPaper.paperId,citingPaper.title,citingPaper.abstract,citingPaper.year,citingPaper.authors,citingPaper.externalIds,citingPaper.openAccessPdf",
        }
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            response = await client.get(url, params=params, headers=headers)
        except Exception:
            return []

        if response.status_code >= 400:
            return []

        try:
            data = response.json().get("data", [])
        except Exception:
            return []

        results: list[SearchResult] = []
        for entry in data:
            item = entry.get("citingPaper") or {}
            title = (item.get("title") or "").strip()
            if not title:
                continue
            authors = [
                a.get("name", "").strip()
                for a in item.get("authors", [])
                if a.get("name") and a.get("name").strip()
            ]
            ext_ids = item.get("externalIds") or {}
            doi = ext_ids.get("DOI")
            arxiv_id = ext_ids.get("ArXiv")
            open_access = item.get("openAccessPdf") or {}
            pdf_url = open_access.get("url") if isinstance(open_access, dict) else None
            if not pdf_url and arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            res = SearchResult(
                title=title,
                authors=authors,
                abstract=item.get("abstract"),
                year=item.get("year"),
                doi=doi,
                pdf_url=pdf_url,
                source_api="semantic_scholar_citations",
                paper_id=item.get("paperId"),
            )
            res.ensure_cite_key()
            results.append(res)
        return results

    async def get_paper_references(
        self,
        paper_id_or_doi: str,
        limit: int = 15,
    ) -> list[SearchResult]:
        """Fetch papers referenced by the specified seed paper via Semantic Scholar Graph API."""
        client = await self._get_client()
        clean_id = paper_id_or_doi.strip()
        encoded = f"DOI:{clean_id}" if ("10." in clean_id and not clean_id.startswith("DOI:")) else clean_id
        url = f"https://api.semanticscholar.org/graph/v1/paper/{encoded}/references"
        params = {
            "limit": limit,
            "fields": "citedPaper.paperId,citedPaper.title,citedPaper.abstract,citedPaper.year,citedPaper.authors,citedPaper.externalIds,citedPaper.openAccessPdf",
        }
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            response = await client.get(url, params=params, headers=headers)
        except Exception:
            return []

        if response.status_code >= 400:
            return []

        try:
            data = response.json().get("data", [])
        except Exception:
            return []

        results: list[SearchResult] = []
        for entry in data:
            item = entry.get("citedPaper") or {}
            title = (item.get("title") or "").strip()
            if not title:
                continue
            authors = [
                a.get("name", "").strip()
                for a in item.get("authors", [])
                if a.get("name") and a.get("name").strip()
            ]
            ext_ids = item.get("externalIds") or {}
            doi = ext_ids.get("DOI")
            arxiv_id = ext_ids.get("ArXiv")
            open_access = item.get("openAccessPdf") or {}
            pdf_url = open_access.get("url") if isinstance(open_access, dict) else None
            if not pdf_url and arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            res = SearchResult(
                title=title,
                authors=authors,
                abstract=item.get("abstract"),
                year=item.get("year"),
                doi=doi,
                pdf_url=pdf_url,
                source_api="semantic_scholar_references",
                paper_id=item.get("paperId"),
            )
            res.ensure_cite_key()
            results.append(res)
        return results

    async def search_openalex(
        self,
        query: str,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Query OpenAlex works API for global literature discovery with inverted-index abstract reconstruction."""
        client = await self._get_client()
        clean = query.strip()
        params = {
            "search": clean,
            "per_page": min(limit, 25),
        }
        headers = {
            "User-Agent": "Scripcy/2.0 (mailto:scholar@scripcy.local)",
        }
        try:
            resp = await client.get("https://api.openalex.org/works", params=params, headers=headers, timeout=12)
            if resp.status_code != 200:
                return []
            data = resp.json()
            raw_works = data.get("results", [])
        except Exception:
            return []

        results: list[SearchResult] = []
        for w in raw_works[:limit]:
            title = (w.get("title") or "").strip()
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
            pdf_url = oa_info.get("oa_url")
            if not pdf_url:
                best_loc = w.get("best_oa_location") or {}
                pdf_url = best_loc.get("pdf_url")

            # Reconstruct abstract from inverted index
            inverted = w.get("abstract_inverted_index")
            abstract = None
            if inverted and isinstance(inverted, dict):
                try:
                    words_pos: list[tuple[int, str]] = []
                    for word, positions in inverted.items():
                        for pos in positions:
                            words_pos.append((pos, word))
                    words_pos.sort(key=lambda x: x[0])
                    abstract = " ".join(word for _, word in words_pos)
                except Exception:
                    abstract = None

            res = SearchResult(
                title=title,
                authors=authors,
                year=year,
                doi=doi_clean,
                abstract=abstract,
                pdf_url=pdf_url,
                source_api="openalex",
                paper_id=w.get("id"),
            )
            res.ensure_cite_key()
            results.append(res)
        return results

    async def search_arxiv(
        self,
        query: str,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Query ArXiv Atom API for preprints matching query terms."""
        client = await self._get_client()
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit,
        }

        try:
            response = await client.get(ARXIV_QUERY_URL, params=params)
        except httpx.TimeoutException as exc:
            raise SearchError(f"ArXiv request timed out: {exc}") from exc
        except httpx.RequestError as exc:
            raise SearchError(f"ArXiv connection error: {exc}") from exc

        if response.status_code == 429:
            raise RateLimitError("ArXiv API rate limit reached (HTTP 429).")
        if response.status_code >= 400:
            raise SearchError(f"ArXiv API returned HTTP {response.status_code}: {response.text}")

        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as exc:
            raise SearchError(f"Failed to parse ArXiv XML response: {exc}") from exc

        results: list[SearchResult] = []
        for entry in root.findall("atom:entry", ATOM_NS):
            title_elem = entry.find("atom:title", ATOM_NS)
            if title_elem is None or not title_elem.text:
                continue

            # Strip internal newlines from XML title and summary
            title = " ".join(title_elem.text.split())

            summary_elem = entry.find("atom:summary", ATOM_NS)
            abstract = " ".join(summary_elem.text.split()) if summary_elem is not None and summary_elem.text else None

            authors: list[str] = []
            for author_node in entry.findall("atom:author", ATOM_NS):
                name_node = author_node.find("atom:name", ATOM_NS)
                if name_node is not None and name_node.text:
                    authors.append(name_node.text.strip())

            published_elem = entry.find("atom:published", ATOM_NS)
            year: int | None = None
            if published_elem is not None and published_elem.text:
                match = re.search(r"\b(19\d\d|20\d\d)\b", published_elem.text)
                if match:
                    year = int(match.group(1))

            doi_elem = entry.find("arxiv:doi", ATOM_NS)
            doi = doi_elem.text.strip() if doi_elem is not None and doi_elem.text else None

            id_elem = entry.find("atom:id", ATOM_NS)
            paper_id = id_elem.text.strip() if id_elem is not None and id_elem.text else None

            # Resolve PDF URL from link attributes or construct from ArXiv ID
            pdf_url: str | None = None
            pdf_link = entry.find("atom:link[@title='pdf']", ATOM_NS)
            if pdf_link is not None:
                pdf_url = pdf_link.attrib.get("href")
            else:
                for link in entry.findall("atom:link", ATOM_NS):
                    if link.attrib.get("type") == "application/pdf" or "/pdf/" in link.attrib.get("href", ""):
                        pdf_url = link.attrib.get("href")
                        break

            if pdf_url and pdf_url.startswith("http://"):
                pdf_url = "https://" + pdf_url[7:]
            elif not pdf_url and paper_id and "/abs/" in paper_id:
                clean_id = paper_id.split("/abs/")[-1]
                pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"

            result = SearchResult(
                title=title,
                authors=authors,
                abstract=abstract,
                year=year,
                doi=doi,
                pdf_url=pdf_url,
                source_api="arxiv",
                paper_id=paper_id,
            )
            result.ensure_cite_key()
            results.append(result)

        return results

    async def search_crossref(
        self,
        query: str,
        limit: int = 10,
        email: str | None = None,
    ) -> list[SearchResult]:
        """Query CrossRef REST API for scholarly literature matching query terms."""
        clean_query = query.strip()
        if not clean_query:
            return []

        client = await self._get_client()
        clean_email = email.strip() if email else "researcher@scripcy.org"
        params = {
            "query": clean_query,
            "rows": limit,
        }
        headers = {
            "User-Agent": f"scripcy/1.0 (mailto:{clean_email})",
        }

        try:
            response = await client.get(
                CROSSREF_WORKS_URL,
                params=params,
                headers=headers,
                follow_redirects=True,
            )
        except httpx.TimeoutException as exc:
            raise SearchError(f"CrossRef request timed out: {exc}") from exc
        except httpx.RequestError as exc:
            raise SearchError(f"CrossRef connection error: {exc}") from exc

        if response.status_code == 429:
            raise RateLimitError("CrossRef API rate limit reached (HTTP 429).")
        if response.status_code >= 400:
            raise SearchError(f"CrossRef API returned HTTP {response.status_code}: {response.text}")

        try:
            items = response.json().get("message", {}).get("items", [])
        except Exception as exc:
            raise SearchError(f"Failed to parse CrossRef JSON: {exc}") from exc

        results: list[SearchResult] = []
        for item in items:
            title_list = item.get("title") or []
            title = title_list[0].strip() if title_list else ""
            if not title:
                continue

            authors = []
            for a in item.get("author", []):
                given = a.get("given", "").strip()
                family = a.get("family", "").strip()
                name = f"{given} {family}".strip() or family or given
                if name:
                    authors.append(name)

            date_parts = item.get("published", {}).get("date-parts", [[None]])
            year = date_parts[0][0] if date_parts and date_parts[0] and isinstance(date_parts[0][0], int) else None

            raw_abstract = item.get("abstract") or ""
            abstract = re.sub(r"<[^>]+>", "", raw_abstract).strip() if raw_abstract else None

            doi = item.get("DOI")

            pdf_url = None
            for link in item.get("link", []):
                if link.get("content-type") == "application/pdf" and link.get("URL"):
                    pdf_url = link.get("URL")
                    break

            res = SearchResult(
                title=title,
                authors=authors,
                abstract=abstract,
                year=year,
                doi=doi,
                pdf_url=pdf_url,
                source_api="crossref",
                paper_id=doi,
            )
            res.ensure_cite_key()
            results.append(res)
        return results

    async def lookup_unpaywall(
        self,
        doi: str,
        email: str | None = None,
    ) -> str | None:
        """Query Unpaywall API to discover Green Open Access PDF location for a DOI."""
        if not doi:
            return None
        clean_doi = doi.strip().replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
        client = await self._get_client()
        clean_email = email.strip() if email else "researcher@scripcy.org"
        url = f"{UNPAYWALL_API_URL}/{clean_doi}"
        params = {"email": clean_email}

        try:
            resp = await client.get(url, params=params, follow_redirects=True)
            if resp.status_code == 200:
                data = resp.json()
                best_oa = data.get("best_oa_location") or {}
                return best_oa.get("url_for_pdf") or best_oa.get("url")
        except Exception:
            pass
        return None

    async def get_opencitations(
        self,
        doi: str,
        api_token: str | None = None,
    ) -> list[str]:
        """Query OpenCitations COCI API to retrieve citing DOIs for a given DOI."""
        if not doi:
            return []
        clean_doi = doi.strip().replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
        client = await self._get_client()
        url = f"{OPENCITATIONS_COCI_URL}/{clean_doi}"
        headers = {}
        if api_token:
            headers["authorization"] = api_token

        try:
            resp = await client.get(url, headers=headers, follow_redirects=True)
            if resp.status_code == 200:
                data = resp.json()
                return [item.get("citing") for item in data if item.get("citing")]
        except Exception:
            pass
        return []

    async def download_paper(
        self,
        paper: SearchResult,
        filename: str | None = None,
        target_dir: Path | None = None,
    ) -> Path:
        """Download open-access PDF for a search result into the workspace vault."""
        if not paper.has_pdf or not paper.pdf_url:
            raise NoOpenAccessPDFError(f"No open-access PDF URL available for '{paper.title}'")

        client = await self._get_client()
        dest_dir = target_dir or self.workspace.vault_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        target_filename = filename or f"{sanitize_filename(paper.ensure_cite_key())}.pdf"
        if not target_filename.lower().endswith(".pdf"):
            target_filename = f"{target_filename}.pdf"

        target_path = dest_dir / target_filename

        try:
            async with client.stream("GET", paper.pdf_url, follow_redirects=True) as response:
                if response.status_code == 429:
                    raise RateLimitError(f"Rate limit exceeded while downloading PDF from {paper.pdf_url}")
                if response.status_code >= 400:
                    raise DownloadError(f"Download failed with HTTP status {response.status_code}")

                chunks: list[bytes] = []
                async for chunk in response.aiter_bytes():
                    chunks.append(chunk)

                content = b"".join(chunks)
                if not content:
                    raise DownloadError(f"Downloaded PDF from {paper.pdf_url} is empty (0 bytes).")

                target_path.write_bytes(content)
                return target_path
        except (RateLimitError, DownloadError):
            raise
        except httpx.TimeoutException as exc:
            raise DownloadError(f"PDF download timed out: {exc}") from exc
        except httpx.RequestError as exc:
            raise DownloadError(f"Network error during PDF download: {exc}") from exc
        except OSError as exc:
            raise DownloadError(f"Filesystem error saving PDF to '{target_path}': {exc}") from exc
