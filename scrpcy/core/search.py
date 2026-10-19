"""Academic paper search and PDF retrieval across Semantic Scholar and ArXiv."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Literal
import xml.etree.ElementTree as ET

import httpx
from pydantic import BaseModel, ConfigDict, Field, computed_field

from scrpcy.core.workspace import WorkspaceManager
from scrpcy.exceptions import (
    DownloadError,
    NoOpenAccessPDFError,
    RateLimitError,
    SearchError,
)

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_QUERY_URL = "https://export.arxiv.org/api/query"

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

        # Automatic fallback: try Semantic Scholar, fallback to ArXiv on error or rate-limiting
        try:
            results = await self.search_semantic_scholar(clean_query, limit=limit)
            if results:
                return results
        except (RateLimitError, SearchError, httpx.RequestError):
            pass

        return await self.search_arxiv(clean_query, limit=limit)

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
