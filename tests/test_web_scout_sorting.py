"""Tests for official website scoring, multi-part TLD parsing, and search result re-ranking."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from scripcy.core.web_scout import (
    OfficialDomainFilter,
    OfficialDomainScorer,
    WebScout,
    WebSearchResult,
)
from scripcy.core.workspace import WorkspaceManager


class TestOfficialDomainScorer:
    """Detailed unit tests for OfficialDomainScorer heuristics and multi-signal ranking."""

    def test_multi_part_public_suffix_parsing(self):
        # Indonesian government and academic suffixes
        p1 = OfficialDomainScorer.parse_domain_parts("https://sipongi.menlhk.go.id/hotspot/data")
        assert p1["suffix"] == "go.id"
        assert p1["registered_domain"] == "menlhk.go.id"
        assert p1["subdomain"] == "sipongi"

        p2 = OfficialDomainScorer.parse_domain_parts("https://ugm.ac.id/id/penelitian")
        assert p2["suffix"] == "ac.id"
        assert p2["registered_domain"] == "ugm.ac.id"
        assert p2["subdomain"] == ""

        # UK and Australian government suffixes
        p3 = OfficialDomainScorer.parse_domain_parts("https://data.gov.uk/dataset/fire-statistics")
        assert p3["suffix"] == "gov.uk"
        assert p3["registered_domain"] == "data.gov.uk"
        assert p3["subdomain"] == ""

        p3_sub = OfficialDomainScorer.parse_domain_parts("https://stats.data.gov.uk/dataset")
        assert p3_sub["suffix"] == "gov.uk"
        assert p3_sub["registered_domain"] == "data.gov.uk"
        assert p3_sub["subdomain"] == "stats"

        p4 = OfficialDomainScorer.parse_domain_parts("https://www.bom.gov.au/climate/")
        assert p4["suffix"] == "gov.au"
        assert p4["registered_domain"] == "bom.gov.au"

    def test_indonesian_sovereign_ministries_and_institutions(self):
        # SiPongi KLHK (Karhutla monitoring)
        r_sipongi = OfficialDomainScorer.evaluate("https://sipongi.menlhk.go.id/publikasi/statistik")
        assert r_sipongi.is_authoritative is True
        assert r_sipongi.authority_score >= 0.85
        assert r_sipongi.authority_tier == 1
        assert "SiPongi" in r_sipongi.institution_name or "KLHK" in r_sipongi.institution_name

        # BPS (Badan Pusat Statistik)
        r_bps = OfficialDomainScorer.evaluate("https://www.bps.go.id/id/statistics-table/2/MTA0MSMy/luas-kebakaran-hutan-dan-lahan.html")
        assert r_bps.is_authoritative is True
        assert r_bps.authority_score >= 0.85
        assert "Badan Pusat Statistik" in r_bps.institution_name

        # BMKG
        r_bmkg = OfficialDomainScorer.evaluate("https://www.bmkg.go.id/cuaca/karhutla.bmkg")
        assert r_bmkg.is_authoritative is True
        assert "BMKG" in r_bmkg.institution_name

    def test_entity_grounding_and_custom_whitelist(self):
        # Target official domain match (e.g. from Wikidata P856 resolution)
        r_whitelisted = OfficialDomainScorer.evaluate(
            "https://perhutani.co.id/laporan-tahunan",
            target_official_domains={"perhutani.co.id"},
        )
        assert r_whitelisted.is_authoritative is True
        assert r_whitelisted.authority_score == 1.0
        assert r_whitelisted.authority_tier == 1
        assert "Verified target entity official domain match" in r_whitelisted.reasons[0]

    def test_anti_seo_farm_path_penalties(self):
        # Site using a generic commercial TLD with clickbait / tag patterns
        r_tag = OfficialDomainScorer.evaluate("https://berita-harian-online.com/tag/data-karhutla-terbaru-lengkap-2024.html")
        assert r_tag.is_authoritative is False
        assert any("SEO farm tag" in reason for reason in r_tag.reasons)

        r_category = OfficialDomainScorer.evaluate("https://agregator-portal.net/kategori/kebijakan-lingkungan/")
        assert r_category.is_authoritative is False
        assert any("SEO farm tag" in reason for reason in r_category.reasons)

    def test_cheap_spam_tld_penalties(self):
        spam_urls = [
            "https://info-terbaru.xyz/karhutla-indonesia",
            "https://download-data.top/laporan-resmi",
            "https://free-stats.click/gambut",
            "https://clickbait.site/update-2024",
        ]
        for url in spam_urls:
            r = OfficialDomainScorer.evaluate(url)
            assert r.is_authoritative is False
            assert r.authority_score <= 0.20
            assert any("High-risk or spam TLD" in reason for reason in r.reasons)

    def test_ugc_and_free_blogging_platforms_penalized(self):
        blog_urls = [
            "https://myenvironmentalblog.blogspot.com/2024/01/karhutla.html",
            "https://greenpeacewatch.wordpress.com/2023/report",
            "https://medium.com/@analyst/analisis-karhutla-5-tahun-terakhir",
            "https://kompasiana.com/user123/kebakaran-hutan-dan-lahan",
        ]
        for url in blog_urls:
            r = OfficialDomainScorer.evaluate(url)
            assert r.is_authoritative is False
            assert r.authority_score <= 0.20
            assert any("free blogging platform" in reason.lower() for reason in r.reasons)


class TestWebScoutReranking:
    """Tests for two-stage candidate retrieval and authority re-ranking."""

    @pytest.mark.asyncio
    async def test_search_official_web_sorts_by_authority_and_filters_farms(self, tmp_path: Path):
        mock_html = """
        <div class="result">
            <h2 class="result__title"><a href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fblog-amatir.blogspot.com%2Fkarhutla">Blog Karhutla Pribadi</a></h2>
            <div class="result__snippet">Catatan pribadi tentang kebakaran hutan di Riau.</div>
            <a class="result__url" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fblog-amatir.blogspot.com%2Fkarhutla"></a>
        </div>
        <div class="result">
            <h2 class="result__title"><a href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fsipongi.menlhk.go.id%2Fpublikasi%2Fdata-karhutla-5-tahun">SiPongi KLHK - Rekapitulasi Luas Kebakaran Hutan Indonesia</a></h2>
            <div class="result__snippet">Data resmi luasan karhutla tahun 2019 hingga 2024 dari Kementerian Lingkungan Hidup dan Kehutanan.</div>
            <a class="result__url" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fsipongi.menlhk.go.id%2Fpublikasi%2Fdata-karhutla-5-tahun"></a>
        </div>
        <div class="result">
            <h2 class="result__title"><a href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fspamsite.xyz%2Ftag%2Fkarhutla-terbaru-lengkap">Spam Site Tag Aggregator</a></h2>
            <div class="result__snippet">Kumpulan artikel berita copy-paste tentang api hutan.</div>
            <a class="result__url" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fspamsite.xyz%2Ftag%2Fkarhutla-terbaru-lengkap"></a>
        </div>
        <div class="result">
            <h2 class="result__title"><a href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.bps.go.id%2Fstatictable%2Fkarhutla-nasional">Badan Pusat Statistik - Statistik Lingkungan Hidup</a></h2>
            <div class="result__snippet">Statistik tahunan emisi dan luas kebakaran hutan BPS RI.</div>
            <a class="result__url" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.bps.go.id%2Fstatictable%2Fkarhutla-nasional"></a>
        </div>
        """

        ws = WorkspaceManager(root_dir=tmp_path)
        ws.ensure_directories()
        scout = WebScout(workspace=ws)

        mock_resp = AsyncMock()
        mock_resp.text = mock_html
        mock_resp.raise_for_status = lambda: None

        with patch.object(scout, "_get_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_fn.return_value = mock_client

            results = await scout.search_official_web("data karhutla indonesia 5 tahun terakhir", limit=2, require_official=True)

            assert len(results) == 2
            # The top results MUST be the sovereign official portals (SiPongi KLHK & BPS), NOT blogspot or spam
            domains = [r.domain for r in results]
            assert "sipongi.menlhk.go.id" in domains[0] or "bps.go.id" in domains[0]
            assert "sipongi.menlhk.go.id" in domains[1] or "bps.go.id" in domains[1]
            assert all(r.is_official for r in results)
            assert all(r.authority_score >= 0.80 for r in results)

            # Blogspot and .xyz spam MUST NOT be in the official results
            assert not any("blogspot" in r.domain for r in results)
            assert not any("spamsite.xyz" in r.domain for r in results)

    def test_critic_report_generates_directed_web_queries(self):
        from scripcy.core.critic import DevilsAdvocate
        critic = DevilsAdvocate()
        report = critic.critique_heuristic(
            query="carikan aku data karhutla di indonesia 5 tahun terakhir",
            answer="Kebakaran hutan sering terjadi di musim kemarau dan menyebabkan kabut asap.",
        )
        assert len(report.directed_search_queries) > 0
        assert len(report.directed_web_queries) > 0
        assert any("resmi" in q or "statistik" in q or "pemerintah" in q for q in report.directed_web_queries)


class TestQueryDisambiguator:
    """Tests for Stanford STORM and GPT-Researcher inspired QueryDisambiguator."""

    def test_strips_conversational_and_generic_tokens(self):
        from scripcy.core.web_scout import QueryDisambiguator
        d = QueryDisambiguator.disambiguate("carikan aku data terbaru karhutla di indonesia")
        # Core entity must be isolated to karhutla
        assert "karhutla" in d.core_entity_terms
        assert "indonesia" in d.context_terms
        # Generic meta-tokens like 'data' and 'terbaru' must NOT be in core entities
        assert "data" not in d.core_entity_terms
        assert "terbaru" not in d.core_entity_terms
        assert "carikan" not in d.core_entity_terms
        assert "aku" not in d.core_entity_terms

        # Generated queries must contain domain operators and metric indicators
        assert len(d.targeted_queries) >= 2
        assert any("site:go.id" in q for q in d.targeted_queries)
        assert any("-pengertian" in q for q in d.targeted_queries)

    def test_anchors_karhutla_to_sipongi_and_klhk(self):
        from scripcy.core.web_scout import QueryDisambiguator
        d = QueryDisambiguator.disambiguate("rekapitulasi titik api karhutla 2024")
        assert "karhutla" in d.core_entity_terms
        assert any("sipongi.menlhk.go.id" in dom for dom in d.anchored_domains)
        assert "kebakaran hutan" in d.synonyms
        assert any("site:sipongi.menlhk.go.id" in q for q in d.targeted_queries)


class TestCascadedWaterfallAndRelevanceGate:
    """Tests for FlashRAG and MindSearch inspired cascaded relevance gating."""

    def test_root_landing_page_detection(self):
        from scripcy.core.web_scout import is_root_or_generic_landing
        # Generic root / shallow pages must return True
        assert is_root_or_generic_landing("https://www.bps.go.id/") is True
        assert is_root_or_generic_landing("https://www.bps.go.id/id") is True
        assert is_root_or_generic_landing("https://data.go.id/") is True
        assert is_root_or_generic_landing("https://indonesia.go.id/instantion") is True

        # Deep substantive content pages must return False
        assert is_root_or_generic_landing("https://sipongi.menlhk.go.id/publikasi/statistik-2024") is False
        assert is_root_or_generic_landing("https://www.bps.go.id/id/statistics-table/2/MTA0MSMy/luas-kebakaran-hutan-dan-lahan.html") is False

    def test_topic_relevance_computation(self):
        from scripcy.core.web_scout import compute_topic_relevance
        # Completely irrelevant page
        rel_zero = compute_topic_relevance(
            title="Badan Pusat Statistik Indonesia",
            snippet="Portal resmi BPS RI AllStats data dan sensus penduduk.",
            url="https://www.bps.go.id/id",
            core_terms=["karhutla"],
        )
        assert rel_zero == 0.0

        # Highly relevant page
        rel_high = compute_topic_relevance(
            title="SiPongi KLHK - Rekapitulasi Luas Kebakaran Hutan dan Lahan",
            snippet="Data resmi luasan karhutla tahun 2024 per provinsi di Indonesia.",
            url="https://sipongi.menlhk.go.id/publikasi/karhutla-2024",
            core_terms=["karhutla"],
            synonyms=["kebakaran hutan"],
        )
        assert rel_high >= 0.85

    @pytest.mark.asyncio
    async def test_rejects_irrelevant_official_homepage_in_search(self, tmp_path: Path):
        from scripcy.core.web_scout import WebScout
        mock_html = """
        <div class="result">
            <h2 class="result__title"><a href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.bps.go.id%2Fid">Badan Pusat Statistik Indonesia</a></h2>
            <div class="result__snippet">Portal resmi Badan Pusat Statistik untuk statistik ekonomi dan sensus.</div>
            <a class="result__url" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.bps.go.id%2Fid"></a>
        </div>
        <div class="result">
            <h2 class="result__title"><a href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fsipongi.menlhk.go.id%2Fpublikasi%2Fluas-karhutla-2024">SiPongi KLHK - Rekapitulasi Luas Karhutla Indonesia 2024</a></h2>
            <div class="result__snippet">Data rekapitulasi luas area kebakaran hutan dan lahan tahun 2024.</div>
            <a class="result__url" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fsipongi.menlhk.go.id%2Fpublikasi%2Fluas-karhutla-2024"></a>
        </div>
        """

        ws = WorkspaceManager(root_dir=tmp_path)
        ws.ensure_directories()
        scout = WebScout(workspace=ws)

        mock_resp = AsyncMock()
        mock_resp.text = mock_html
        mock_resp.raise_for_status = lambda: None

        with patch.object(scout, "_get_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_fn.return_value = mock_client

            results = await scout.search_official_web("carikan aku data terbaru karhutla di indonesia", limit=5, require_official=True)

            # bps.go.id/id MUST be dropped because it is a generic root homepage with 0 relevance to karhutla
            # only SiPongi KLHK should survive
            assert len(results) == 1
            assert results[0].domain == "sipongi.menlhk.go.id"
            assert "karhutla" in results[0].title.lower()

    @pytest.mark.asyncio
    async def test_post_fetch_verification_rejects_missing_topic(self, tmp_path: Path):
        from scripcy.core.web_scout import WebScout
        from scripcy.exceptions import OfficialSourceRejectedError

        ws = WorkspaceManager(root_dir=tmp_path)
        ws.ensure_directories()
        scout = WebScout(workspace=ws)

        # Mock HTML for a shell page with no karhutla text
        mock_shell_html = """
        <html>
            <head><title>Portal Resmi</title></head>
            <body><p>Selamat datang di portal informasi resmi.</p></body>
        </html>
        """
        mock_resp = AsyncMock()
        mock_resp.text = mock_shell_html
        mock_resp.url = "https://kementerian.go.id/portal"
        mock_resp.raise_for_status = lambda: None

        with patch.object(scout, "_get_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_fn.return_value = mock_client

            with pytest.raises(OfficialSourceRejectedError) as exc_info:
                await scout.ingest_url(
                    "https://kementerian.go.id/portal",
                    required_topic_terms=["karhutla", "kebakaran"],
                )
            assert "does not discuss requested topic terms" in str(exc_info.value)
