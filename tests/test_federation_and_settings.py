"""Unit tests for multi-source academic federation, arXiv domain guard, Unpaywall Green OA, and DESIGN.md settings."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from scripcy.core.config import ScripcyConfig
from scripcy.core.scout import LiteratureScout, ScoutQuery, ScoutResult
from scripcy.core.search import (
    PaperSearcher,
    SearchResult,
    is_cs_ai_domain,
)
from scripcy.studio.app import CLIStudio


def test_arxiv_domain_guard() -> None:
    """Verify that only CS, AI, Math, and Physics queries trigger arXiv search."""
    # CS / AI queries should match
    assert is_cs_ai_domain("deep learning transformers for NLP")
    assert is_cs_ai_domain("machine learning classification algorithm")
    assert is_cs_ai_domain("quantum computing and cryptography")
    assert is_cs_ai_domain("reinforcement learning robotics")
    assert is_cs_ai_domain("graph neural networks")

    # Non-CS domains must NOT trigger arXiv
    assert not is_cs_ai_domain("karhutla di indonesia 5 tahun terakhir")
    assert not is_cs_ai_domain("kebijakan hukum pidana lingkungan hidup")
    assert not is_cs_ai_domain("analisis manajemen rantai pasok kelapa sawit")
    assert not is_cs_ai_domain("dampak kebakaran hutan terhadap keanekaragaman hayati")
    assert not is_cs_ai_domain("psikologi perkembangan anak usia dini")


@pytest.mark.asyncio
async def test_paper_searcher_arxiv_fallback_guard() -> None:
    """Ensure PaperSearcher falls back to arXiv ONLY if the domain is CS/AI."""
    searcher = PaperSearcher()

    # When Semantic Scholar and OpenAlex fail, non-CS query returns empty list instead of irrelevant arXiv papers
    with patch.object(searcher, "search_semantic_scholar", side_effect=Exception("S2 429")):
        with patch.object(searcher, "search_openalex", new_callable=AsyncMock, return_value=[]):
            with patch.object(searcher, "search_arxiv", new_callable=AsyncMock) as mock_arxiv:
                results = await searcher.search("karhutla di indonesia")
                assert results == []
                mock_arxiv.assert_not_called()

    # When Semantic Scholar and OpenAlex fail, CS query DOES fall back to arXiv
    with patch.object(searcher, "search_semantic_scholar", side_effect=Exception("S2 429")):
        with patch.object(searcher, "search_openalex", new_callable=AsyncMock, return_value=[]):
            with patch.object(searcher, "search_arxiv", new_callable=AsyncMock) as mock_arxiv:
                mock_arxiv.return_value = [SearchResult(title="Transformer Models", authors=["Author"])]
                results = await searcher.search("transformers for deep learning")
                assert len(results) == 1
                mock_arxiv.assert_called_once()


@pytest.mark.asyncio
async def test_unpaywall_lookup() -> None:
    """Verify Unpaywall API lookup extracts PDF URL."""
    searcher = PaperSearcher()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "is_oa": True,
        "best_oa_location": {
            "url_for_pdf": "https://repository.univ.edu/paper.pdf",
            "url": "https://repository.univ.edu/paper.pdf",
            "version": "acceptedVersion",
        },
    }

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    with patch.object(searcher, "_get_client", return_value=mock_client):
        pdf_url = await searcher.lookup_unpaywall("10.1234/test.doi", email="researcher@scripcy.org")
        assert pdf_url == "https://repository.univ.edu/paper.pdf"


@pytest.mark.asyncio
async def test_scout_unpaywall_resolution(tmp_path: Path) -> None:
    """Verify that paywalled papers with DOI resolve to Green OA via Unpaywall."""
    cfg = ScripcyConfig(unpaywall_email="test@univ.ac.id")
    scout = LiteratureScout(config=cfg)

    paper = ScoutResult(
        title="Impact of Wildfires on Peatlands",
        authors=["Siti Rahma"],
        year=2021,
        doi="10.1016/j.envsci.2021.05.001",
        has_pdf=False,
        pdf_url=None,
    )

    with patch.object(
        scout.searcher,
        "lookup_unpaywall",
        new_callable=AsyncMock,
        return_value="https://repo.ugm.ac.id/green-oa.pdf",
    ):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"%PDF-1.4 Mock PDF content"

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            with patch("pymupdf.open", side_effect=Exception("skip parsing")):
                target = await scout.download_scout_paper(paper)
                assert paper.has_pdf is True
                assert paper.pdf_url == "https://repo.ugm.ac.id/green-oa.pdf"
                assert "Unpaywall" in paper.relevance_reason
                assert target.name.endswith(".pdf")


def test_settings_cmd_toggles_and_keys() -> None:
    """Test /settings commands for toggling providers and setting API keys."""
    test_console = Console(record=True, width=100)
    studio = CLIStudio(config=ScripcyConfig(), console=test_console)

    # 1. Toggle Semantic Scholar
    studio.cmd_settings("toggle semantic_scholar")
    assert studio.config.enable_semantic_scholar is False
    studio.cmd_settings("toggle semantic_scholar")
    assert studio.config.enable_semantic_scholar is True

    # 2. Toggle CrossRef
    studio.cmd_settings("toggle crossref")
    assert studio.config.enable_crossref is False

    # 3. Toggle OpenCitations
    studio.cmd_settings("toggle opencitations")
    assert studio.config.enable_opencitations is True

    # 4. Set OpenAlex API Key
    studio.cmd_settings("set-key openalex oa_key_123456")
    assert studio.config.openalex_api_key == "oa_key_123456"

    # 5. Set Semantic Scholar API Key
    studio.cmd_settings("set-key semantic_scholar s2_key_abcdef")
    assert studio.config.semantic_scholar_api_key == "s2_key_abcdef"

    # 6. Set Unpaywall Email
    studio.cmd_settings("set-email researcher@campus.ac.id")
    assert studio.config.unpaywall_email == "researcher@campus.ac.id"

    # 7. Check show output adheres to DESIGN.md
    studio.cmd_settings("show")
    output = test_console.export_text()
    assert "OpenAlex (Utama)" in output
    assert "CrossRef" in output
    assert "Semantic Scholar" in output
    assert "OpenCitations" in output
    assert "ArXiv Domain Guard" in output
    assert "researcher@campus.ac.id" in output
