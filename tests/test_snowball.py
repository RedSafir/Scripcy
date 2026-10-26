"""Unit tests for Multi-Thread Cross-Citation Snowballing and Convergence Scoring."""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripcy.core.search import PaperSearcher, SearchResult
from scripcy.core.snowball import (
    CitationSnowballer,
    ConvergenceResult,
    SeedThread,
    calculate_recency_score,
    calculate_semantic_relevance,
    calculate_thread_overlap,
    token_cosine_similarity,
)


def test_calculate_thread_overlap() -> None:
    """Verify thread overlap calculation across independent threads."""
    # 3 out of 3 threads
    assert calculate_thread_overlap(["Thread A", "Thread B", "Thread C"], 3) == 1.0
    # Duplicate thread mentions in same thread should not inflate count
    assert calculate_thread_overlap(["Thread A", "Thread A", "Thread B"], 3) == pytest.approx(2 / 3)
    # 1 out of 3 threads
    assert calculate_thread_overlap(["Thread A"], 3) == pytest.approx(1 / 3)
    # 0 threads
    assert calculate_thread_overlap([], 3) == 0.0
    assert calculate_thread_overlap(["Thread A"], 0) == 0.0


def test_calculate_recency_score() -> None:
    """Verify recency decay score with exponential lambda decay."""
    # Current year (2026) -> delta = 0 -> score = 1.0
    assert calculate_recency_score(2026, current_year=2026, decay_rate=0.05) == 1.0

    # 10 years ago (2016) -> delta = 10 -> exp(-0.05 * 10) = exp(-0.5) ~ 0.6065
    expected_2016 = math.exp(-0.05 * 10)
    assert calculate_recency_score(2016, current_year=2026, decay_rate=0.05) == pytest.approx(expected_2016)

    # Missing year falls back to delta = 10
    assert calculate_recency_score(None, current_year=2026, decay_rate=0.05) == pytest.approx(expected_2016)

    # Future or current year should not exceed 1.0
    assert calculate_recency_score(2027, current_year=2026, decay_rate=0.05) == 1.0


def test_token_cosine_similarity_pruning() -> None:
    """Verify token cosine similarity detects in-domain vs. out-of-domain noise."""
    query = "karhutla di indonesia mitigasi lahan gambut"

    in_domain_text = (
        "Analisis spasial dinamika kebakaran hutan dan lahan gambut (karhutla) di Indonesia "
        "serta strategi mitigasi emisi karbon."
    )
    sim_in = token_cosine_similarity(query, in_domain_text)
    assert sim_in > 0.45

    # Universal methodological noise (e.g. PyTorch framework)
    out_domain_noise = (
        "PyTorch: An Imperative Style, High-Performance Deep Learning Library. "
        "We present PyTorch, a Python package providing GPU-accelerated tensor computation."
    )
    sim_out = token_cosine_similarity(query, out_domain_noise)
    assert sim_out < 0.10


@pytest.mark.asyncio
async def test_decompose_query_heuristic() -> None:
    """Verify heuristic thread decomposition when engine is absent or offline."""
    snowballer = CitationSnowballer(engine=None)
    threads = await snowballer.decompose_query_into_threads("transformer attention mechanisms")

    assert len(threads) == 3
    names = [t.thread_name for t in threads]
    assert "Theoretical / Foundational" in names
    assert "Applied / Architectural" in names
    assert "Evaluation / Empirical" in names
    assert all("transformer attention mechanisms" in t.query for t in threads)


@pytest.mark.asyncio
async def test_decompose_query_llm() -> None:
    """Verify LLM structured thread decomposition when engine is active."""
    mock_engine = MagicMock()
    mock_engine.config.default_model = "openai/gpt-4o"
    mock_engine.config.get_api_key_for_model.return_value = "sk-valid-key"

    llm_payload = {
        "threads": [
            {"thread_name": "Mathematical Foundations", "query": "self-attention mathematical theory"},
            {"thread_name": "Efficient Architectures", "query": "linear transformer linear attention"},
            {"thread_name": "Benchmark Evaluation", "query": "long-range arena transformer benchmarks"},
        ]
    }

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content=f"```json\n{json_dumps(llm_payload)}\n```"))]

    with patch("litellm.completion", return_value=mock_resp):
        snowballer = CitationSnowballer(engine=mock_engine)
        threads = await snowballer.decompose_query_into_threads("transformer attention")

        assert len(threads) == 3
        assert threads[0].thread_name == "Mathematical Foundations"
        assert threads[1].thread_name == "Efficient Architectures"
        assert threads[2].thread_name == "Benchmark Evaluation"


@pytest.mark.asyncio
async def test_snowball_convergence_scoring_and_ranking() -> None:
    """Test multi-thread cross-citation graph traversal and multi-factor convergence scoring."""
    searcher = PaperSearcher()
    snowballer = CitationSnowballer(searcher=searcher, min_relevance=0.15, current_year=2026)

    # 3 Seed Threads
    s1 = SearchResult(title="Theory of Peatland Hydrology", doi="10.1001/seed1")
    s2 = SearchResult(title="Remote Sensing of Peat Fires", doi="10.1001/seed2")
    s3 = SearchResult(title="Carbon Emissions and Fire Policy", doi="10.1001/seed3")

    threads = [
        SeedThread(thread_name="Hydrology Theory", query="peat hydrology", seed_papers=[s1]),
        SeedThread(thread_name="Remote Sensing", query="fire remote sensing", seed_papers=[s2]),
        SeedThread(thread_name="Policy & Emissions", query="emissions policy", seed_papers=[s3]),
    ]

    # Candidate 1: Cited across ALL 3 threads (True Hub Paper)
    hub_paper = SearchResult(
        title="Fire and Peatland Ecosystems in Indonesia",
        abstract="Comprehensive assessment of tropical peatland fires, hydrology, and carbon release in Indonesia.",
        year=2022,
        doi="10.1001/hub_paper",
    )

    # Candidate 2: Cited across 2 threads (Moderate Hub)
    mod_paper = SearchResult(
        title="Peat Moisture Detection via Satellites",
        abstract="Satellite radar estimation of peat moisture and wildfire risk in Southeast Asia.",
        year=2023,
        doi="10.1001/mod_paper",
    )

    # Candidate 3: Cited across 1 thread only (Niche Paper)
    niche_paper = SearchResult(
        title="Local Peat Soil Microbes in Riau",
        abstract="Microbial diversity in Indonesian peat swamps under wildfire conditions.",
        year=2020,
        doi="10.1001/niche_paper",
    )

    # Candidate 4: Cited across all 3 threads BUT out-of-domain noise (e.g. PyTorch / general tool)
    noise_paper = SearchResult(
        title="PyTorch Machine Learning Library",
        abstract="General purpose automatic differentiation library for Python and GPU deep learning.",
        year=2019,
        doi="10.1001/pytorch_tool",
    )

    # Mock citation references per seed paper
    async def mock_get_references(seed_id: str, limit: int = 20) -> list[SearchResult]:
        if "seed1" in seed_id:
            # Seed 1 cites: Hub, Mod, Noise
            return [hub_paper, mod_paper, noise_paper]
        elif "seed2" in seed_id:
            # Seed 2 cites: Hub, Mod, Noise
            return [hub_paper, mod_paper, noise_paper]
        elif "seed3" in seed_id:
            # Seed 3 cites: Hub, Niche, Noise
            return [hub_paper, niche_paper, noise_paper]
        return []

    with patch.object(searcher, "get_paper_references", side_effect=mock_get_references):
        with patch.object(searcher, "get_paper_citations", new_callable=AsyncMock, return_value=[]):
            results = await snowballer.snowball_and_score(
                user_query="kebakaran hutan dan lahan gambut di indonesia",
                threads=threads,
                include_forward=False,
                max_hub_papers=5,
            )

    # 1. Verify noise paper was PRUNED due to low semantic relevance (< 0.15)
    result_dois = [r.doi for r in results]
    assert "10.1001/pytorch_tool" not in result_dois, "Universal library noise must be pruned!"

    # 2. Verify remaining candidates
    assert len(results) == 3

    # 3. Candidate 1 (Hub) must be ranked #1
    top = results[0]
    assert top.doi == "10.1001/hub_paper"
    assert top.overlap_count == 3
    assert top.thread_overlap == 1.0
    assert len(top.citing_threads) == 3
    assert "Hydrology Theory" in top.citing_threads
    assert "Remote Sensing" in top.citing_threads
    assert "Policy & Emissions" in top.citing_threads
    assert "Cited across 3/3 independent research angles" in top.reasoning

    # 4. Candidate 2 (Moderate) ranked #2
    second = results[1]
    assert second.doi == "10.1001/mod_paper"
    assert second.overlap_count == 2
    assert second.thread_overlap == pytest.approx(2 / 3, abs=1e-3)

    # 5. Candidate 3 (Niche) ranked #3
    third = results[2]
    assert third.doi == "10.1001/niche_paper"
    assert third.overlap_count == 1
    assert third.thread_overlap == pytest.approx(1 / 3, abs=1e-3)

    # 6. Verify strictly monotonic final scores
    assert top.final_score > second.final_score > third.final_score


@pytest.mark.asyncio
async def test_discover_hub_papers_end_to_end() -> None:
    """Verify full end-to-end discover_hub_papers execution with seed retrieval."""
    searcher = PaperSearcher()
    snowballer = CitationSnowballer(searcher=searcher, min_relevance=0.10)

    # Mock search for seeds
    async def mock_search(query: str, limit: int = 2) -> list[SearchResult]:
        return [SearchResult(title=f"Seed for {query[:20]}", doi=f"10.seed/{hash(query) % 1000}")]

    # Mock hub paper
    hub = SearchResult(
        title="Deep Attention Mechanisms for Neural Processing",
        abstract="Fundamental principles of attention mechanisms in deep neural networks.",
        year=2021,
        doi="10.1000/attention_hub",
    )

    with patch.object(searcher, "search", side_effect=mock_search):
        with patch.object(searcher, "get_paper_references", new_callable=AsyncMock, return_value=[hub]):
            with patch.object(searcher, "get_paper_citations", new_callable=AsyncMock, return_value=[]):
                hubs = await snowballer.discover_hub_papers(
                    user_query="attention mechanisms deep neural networks",
                    papers_per_thread=1,
                    include_forward=False,
                    max_hub_papers=3,
                )

    assert len(hubs) >= 1
    assert hubs[0].doi == "10.1000/attention_hub"
    assert hubs[0].overlap_count == 3  # Cited across all 3 generated threads
    assert hubs[0].final_score > 0.60


def json_dumps(obj: Any) -> str:
    import json
    return json.dumps(obj)
