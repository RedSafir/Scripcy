"""Comprehensive unit test suite for academic paper search and PDF retrieval."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import httpx
import pytest

from scrpcy.core.search import (
    PaperSearcher,
    SearchResult,
    sanitize_filename,
)
from scrpcy.core.workspace import WorkspaceManager
from scrpcy.exceptions import (
    DownloadError,
    NoOpenAccessPDFError,
    RateLimitError,
    SearchError,
)

SAMPLE_SEMANTIC_SCHOLAR_JSON = {
    "total": 1,
    "offset": 0,
    "data": [
        {
            "paperId": "204e3073870fae3f05bcbc65e8236986d7734eee",
            "title": "Attention Is All You Need",
            "abstract": "The dominant sequence transduction models are based on complex recurrent networks.",
            "year": 2017,
            "authors": [
                {"authorId": "1", "name": "Ashish Vaswani"},
                {"authorId": "2", "name": "Noam Shazeer"},
            ],
            "externalIds": {
                "DOI": "10.48550/arXiv.1706.03762",
                "ArXiv": "1706.03762",
            },
            "openAccessPdf": {
                "url": "https://arxiv.org/pdf/1706.03762.pdf",
                "status": "GOLD",
            },
        }
    ],
}

SAMPLE_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1512.03385v1</id>
    <published>2015-12-10T18:00:00Z</published>
    <title>Deep Residual Learning for Image Recognition</title>
    <summary>Deeper neural networks are more difficult to train.</summary>
    <author><name>Kaiming He</name></author>
    <author><name>Xiangyu Zhang</name></author>
    <arxiv:doi>10.1109/CVPR.2016.90</arxiv:doi>
    <link title="pdf" href="https://arxiv.org/pdf/1512.03385.pdf" type="application/pdf"/>
  </entry>
</feed>"""


class TestSearchResultModel:
    """Tests for SearchResult validation, cite keys, and helpers."""

    def test_sanitize_filename(self) -> None:
        assert sanitize_filename("Attention: Is All You Need! / 2017") == "Attention_Is_All_You_Need_2017"
        assert sanitize_filename("   ") == "paper"
        long_name = "a" * 120
        assert len(sanitize_filename(long_name)) <= 80

    def test_search_result_cite_key_synthesis(self) -> None:
        result = SearchResult(
            title="Attention Is All You Need",
            authors=["Ashish Vaswani", "Noam Shazeer"],
            year=2017,
            pdf_url="https://example.com/paper.pdf",
        )
        assert result.has_pdf is True
        cite_key = result.ensure_cite_key()
        assert "vaswani2017attention" in cite_key


class TestSemanticScholarSearch:
    """Tests for Semantic Scholar Graph API query parsing and rate limiting."""

    @pytest.mark.asyncio
    async def test_search_semantic_scholar_success(self, temp_workspace: WorkspaceManager) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "semanticscholar.org" in str(request.url)
            assert "query=transformer" in str(request.url)
            return httpx.Response(200, json=SAMPLE_SEMANTIC_SCHOLAR_JSON)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            searcher = PaperSearcher(workspace=temp_workspace, client=client)
            results = await searcher.search_semantic_scholar("transformer", limit=5)

            assert len(results) == 1
            paper = results[0]
            assert paper.title == "Attention Is All You Need"
            assert paper.authors == ["Ashish Vaswani", "Noam Shazeer"]
            assert paper.year == 2017
            assert paper.doi == "10.48550/arXiv.1706.03762"
            assert paper.pdf_url == "https://arxiv.org/pdf/1706.03762.pdf"
            assert paper.source_api == "semantic_scholar"

    @pytest.mark.asyncio
    async def test_search_semantic_scholar_arxiv_fallback_pdf(self, temp_workspace: WorkspaceManager) -> None:
        data_no_open_access = {
            "data": [
                {
                    "title": "ArXiv Linked Paper",
                    "authors": [{"name": "Jane Doe"}],
                    "year": 2023,
                    "externalIds": {"ArXiv": "2301.12345"},
                    "openAccessPdf": None,
                }
            ]
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=data_no_open_access)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            searcher = PaperSearcher(workspace=temp_workspace, client=client)
            results = await searcher.search_semantic_scholar("query")
            assert len(results) == 1
            assert results[0].pdf_url == "https://arxiv.org/pdf/2301.12345.pdf"

    @pytest.mark.asyncio
    async def test_search_semantic_scholar_rate_limit(self, temp_workspace: WorkspaceManager) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"message": "Too Many Requests"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            searcher = PaperSearcher(workspace=temp_workspace, client=client)
            with pytest.raises(RateLimitError, match="rate limit"):
                await searcher.search_semantic_scholar("query")

    @pytest.mark.asyncio
    async def test_search_semantic_scholar_http_error(self, temp_workspace: WorkspaceManager) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            searcher = PaperSearcher(workspace=temp_workspace, client=client)
            with pytest.raises(SearchError, match="HTTP 500"):
                await searcher.search_semantic_scholar("query")


class TestArXivSearch:
    """Tests for ArXiv Atom XML query parsing and fallback operation."""

    @pytest.mark.asyncio
    async def test_search_arxiv_success(self, temp_workspace: WorkspaceManager) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "arxiv.org" in str(request.url)
            return httpx.Response(200, text=SAMPLE_ARXIV_XML)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            searcher = PaperSearcher(workspace=temp_workspace, client=client)
            results = await searcher.search_arxiv("residual networks", limit=5)

            assert len(results) == 1
            paper = results[0]
            assert paper.title == "Deep Residual Learning for Image Recognition"
            assert paper.authors == ["Kaiming He", "Xiangyu Zhang"]
            assert paper.year == 2015
            assert paper.doi == "10.1109/CVPR.2016.90"
            assert paper.pdf_url == "https://arxiv.org/pdf/1512.03385.pdf"
            assert paper.source_api == "arxiv"

    @pytest.mark.asyncio
    async def test_search_arxiv_rate_limit(self, temp_workspace: WorkspaceManager) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, text="Rate limit")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            searcher = PaperSearcher(workspace=temp_workspace, client=client)
            with pytest.raises(RateLimitError, match="rate limit"):
                await searcher.search_arxiv("query")

    @pytest.mark.asyncio
    async def test_search_arxiv_malformed_xml(self, temp_workspace: WorkspaceManager) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<unclosed_tag>broken xml")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            searcher = PaperSearcher(workspace=temp_workspace, client=client)
            with pytest.raises(SearchError, match="parse ArXiv XML"):
                await searcher.search_arxiv("query")


class TestSearchFallbackRouting:
    """Tests for automatic fallback from Semantic Scholar to ArXiv."""

    @pytest.mark.asyncio
    async def test_search_auto_prefers_semantic_scholar_when_available(
        self,
        temp_workspace: WorkspaceManager,
    ) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if "semanticscholar.org" in url_str:
                calls.append("semantic_scholar")
                return httpx.Response(200, json=SAMPLE_SEMANTIC_SCHOLAR_JSON)
            calls.append("arxiv")
            return httpx.Response(200, text=SAMPLE_ARXIV_XML)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            searcher = PaperSearcher(workspace=temp_workspace, client=client)
            results = await searcher.search("transformer", source="auto")

            assert len(results) == 1
            assert results[0].source_api == "semantic_scholar"
            assert calls == ["semantic_scholar"]

    @pytest.mark.asyncio
    async def test_search_auto_falls_back_to_arxiv_on_rate_limit(
        self,
        temp_workspace: WorkspaceManager,
    ) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if "semanticscholar.org" in url_str:
                calls.append("semantic_scholar")
                return httpx.Response(429, json={"error": "Rate limit"})
            calls.append("arxiv")
            return httpx.Response(200, text=SAMPLE_ARXIV_XML)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            searcher = PaperSearcher(workspace=temp_workspace, client=client)
            results = await searcher.search("residual networks", source="auto")

            assert len(results) == 1
            assert results[0].source_api == "arxiv"
            assert "semantic_scholar" in calls
            assert "arxiv" in calls

    @pytest.mark.asyncio
    async def test_search_auto_falls_back_to_arxiv_on_server_error(
        self,
        temp_workspace: WorkspaceManager,
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if "semanticscholar.org" in str(request.url):
                return httpx.Response(503, text="Service Unavailable")
            return httpx.Response(200, text=SAMPLE_ARXIV_XML)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            searcher = PaperSearcher(workspace=temp_workspace, client=client)
            results = await searcher.search("deep learning", source="auto")

            assert len(results) == 1
            assert results[0].source_api == "arxiv"

    @pytest.mark.asyncio
    async def test_search_empty_query_returns_empty_list(self, temp_workspace: WorkspaceManager) -> None:
        searcher = PaperSearcher(workspace=temp_workspace)
        results = await searcher.search("   ")
        assert results == []


class TestPDFDownloadHelper:
    """Tests for downloading open-access PDF files into the workspace vault."""

    @pytest.mark.asyncio
    async def test_download_paper_success(self, temp_workspace: WorkspaceManager) -> None:
        pdf_payload = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"

        def handler(request: httpx.Request) -> httpx.Response:
            if "arxiv.org/pdf" in str(request.url):
                return httpx.Response(
                    200,
                    content=pdf_payload,
                    headers={"Content-Type": "application/pdf"},
                )
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            searcher = PaperSearcher(workspace=temp_workspace, client=client)
            paper = SearchResult(
                title="Attention Is All You Need",
                authors=["Ashish Vaswani"],
                year=2017,
                pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
            )

            downloaded_path = await searcher.download_paper(paper)

            assert downloaded_path.is_file()
            assert downloaded_path.parent == temp_workspace.vault_dir
            assert downloaded_path.read_bytes() == pdf_payload

    @pytest.mark.asyncio
    async def test_download_paper_custom_target_and_filename(
        self,
        tmp_path: Path,
        temp_workspace: WorkspaceManager,
    ) -> None:
        pdf_payload = b"%PDF-1.4 sample content"
        custom_dir = tmp_path / "custom_downloads"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=pdf_payload)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            searcher = PaperSearcher(workspace=temp_workspace, client=client)
            paper = SearchResult(
                title="Sample Paper",
                pdf_url="https://example.com/paper.pdf",
            )

            downloaded = await searcher.download_paper(
                paper,
                filename="target_doc.pdf",
                target_dir=custom_dir,
            )

            assert downloaded.name == "target_doc.pdf"
            assert downloaded.parent == custom_dir
            assert downloaded.read_bytes() == pdf_payload

    @pytest.mark.asyncio
    async def test_download_paper_no_open_access_raises_exception(
        self,
        temp_workspace: WorkspaceManager,
    ) -> None:
        searcher = PaperSearcher(workspace=temp_workspace)
        closed_paper = SearchResult(
            title="Paywalled Paper",
            authors=["Restricted Author"],
            pdf_url=None,
        )

        with pytest.raises(NoOpenAccessPDFError, match="No open-access PDF"):
            await searcher.download_paper(closed_paper)

    @pytest.mark.asyncio
    async def test_download_paper_rate_limit_raises_exception(
        self,
        temp_workspace: WorkspaceManager,
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, text="Too Many Requests")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            searcher = PaperSearcher(workspace=temp_workspace, client=client)
            paper = SearchResult(
                title="Rate Limited Paper",
                pdf_url="https://example.com/paper.pdf",
            )

            with pytest.raises(RateLimitError, match="Rate limit exceeded"):
                await searcher.download_paper(paper)

    @pytest.mark.asyncio
    async def test_download_paper_http_error_raises_exception(
        self,
        temp_workspace: WorkspaceManager,
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="Not Found")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            searcher = PaperSearcher(workspace=temp_workspace, client=client)
            paper = SearchResult(
                title="Missing Paper",
                pdf_url="https://example.com/not_found.pdf",
            )

            with pytest.raises(DownloadError, match="HTTP status 404"):
                await searcher.download_paper(paper)

    @pytest.mark.asyncio
    async def test_download_paper_empty_content_raises_exception(
        self,
        temp_workspace: WorkspaceManager,
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            searcher = PaperSearcher(workspace=temp_workspace, client=client)
            paper = SearchResult(
                title="Empty Paper",
                pdf_url="https://example.com/empty.pdf",
            )

            with pytest.raises(DownloadError, match="empty"):
                await searcher.download_paper(paper)
