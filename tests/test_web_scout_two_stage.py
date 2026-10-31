"""Unit tests for WebScout Two-Stage Search & AI Candidate Selection."""

import asyncio
import json
import pytest

from scripcy.core.web_scout import WebScout, WebSearchResult
from scripcy.core.workspace import WorkspaceManager


@pytest.fixture
def web_scout(temp_workspace: WorkspaceManager) -> WebScout:
    return WebScout(workspace=temp_workspace)


def _make_candidate(title: str, url: str, domain: str, snippet: str, is_official: bool, authority_score: float) -> WebSearchResult:
    return WebSearchResult(
        title=title,
        url=url,
        snippet=snippet,
        domain=domain,
        authority_score=authority_score,
        is_official=is_official,
        institution=domain,
        authority_tier=1 if is_official else 4,
        reasons=["Test match"],
        relevance_score=0.9,
        final_score=authority_score * 0.9,
    )


@pytest.mark.asyncio
async def test_ai_candidate_selection_filters_slop(web_scout: WebScout):
    candidates = [
        _make_candidate(
            title="Official Climate Change Indicators 2024",
            url="https://www.epa.gov/climate-indicators/greenhouse-gases",
            domain="epa.gov",
            snippet="Comprehensive empirical greenhouse gas emissions data compiled by sovereign authorities.",
            is_official=True,
            authority_score=1.0,
        ),
        _make_candidate(
            title="5 Easy Ways To Prevent Climate Change Fast",
            url="https://clickbait-blog.xyz/easy-climate-tips",
            domain="clickbait-blog.xyz",
            snippet="Read our top tips to stop emissions right now with zero effort.",
            is_official=False,
            authority_score=0.1,
        ),
        _make_candidate(
            title="WMO Statement on the State of the Global Climate",
            url="https://wmo.int/publication-series/wmo-statement-state-of-global-climate",
            domain="wmo.int",
            snippet="Official multilateral report on global atmospheric metrics and temperature anomalies.",
            is_official=True,
            authority_score=1.0,
        ),
    ]

    # Mock LLM that selects IDs 1 and 3 (the authoritative sources, ignoring the blog)
    mock_llm_called = []

    async def mock_llm_selector(prompt: str) -> str:
        mock_llm_called.append(prompt)
        assert "epa.gov" in prompt
        assert "clickbait-blog.xyz" in prompt
        assert "wmo.int" in prompt
        return json.dumps({
            "selected_ids": [1, 3],
            "reason": "EPA and WMO are official sovereign and multilateral primary scientific sources."
        })

    selected = await web_scout.select_authoritative_candidates_with_ai(
        candidates,
        topic="Global Greenhouse Gas Emissions Trends",
        llm_client=mock_llm_selector,
        max_select=2,
    )

    assert len(mock_llm_called) == 1
    assert len(selected) == 2
    domains = [s.domain for s in selected]
    assert "epa.gov" in domains
    assert "wmo.int" in domains
    assert "clickbait-blog.xyz" not in domains


@pytest.mark.asyncio
async def test_ai_candidate_selection_fallback_when_offline(web_scout: WebScout):
    candidates = [
        _make_candidate("Title 1", "https://who.int/1", "who.int", "Snippet 1", True, 1.0),
        _make_candidate("Title 2", "https://cdc.gov/2", "cdc.gov", "Snippet 2", True, 0.9),
        _make_candidate("Title 3", "https://blog.com/3", "blog.com", "Snippet 3", False, 0.2),
    ]

    # When llm_client is None (zero-token or offline mode), falls back to top waterfall candidates
    selected = await web_scout.select_authoritative_candidates_with_ai(
        candidates,
        topic="Infectious Disease Tracking",
        llm_client=None,
        max_select=2,
    )
    assert len(selected) == 2
    assert selected[0].domain == "who.int"
    assert selected[1].domain == "cdc.gov"
