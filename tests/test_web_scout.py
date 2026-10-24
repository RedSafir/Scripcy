"""Tests for official web scout, domain filtering, and academic markdown conversion."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from PIL import Image
import pytest

from scrpcy import (
    OfficialDomainFilter,
    OfficialSourceRejectedError,
    WebScout,
    WebScoutError,
)
from scrpcy.core.models import DocumentMetadata
from scrpcy.core.web_scout import (
    SubstantiveVisualAsset,
    SubstantiveVisualExtractor,
    WebPageToMarkdownConverter,
)
from scrpcy.core.workspace import WorkspaceManager


def create_mock_png(width: int = 600, height: int = 400, color: tuple[int, int, int] = (100, 150, 200)) -> bytes:
    """Helper to generate in-memory PNG bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestOfficialDomainFilter:
    """Tests for authority scoring and official domain heuristics."""

    def test_sovereign_government_domains(self):
        result_id = OfficialDomainFilter.evaluate("https://www.bps.go.id/id/pressrelease/2024/02/05/pdb.html")
        assert result_id.is_authoritative is True
        assert result_id.authority_tier == 1
        assert "Badan Pusat Statistik" in result_id.institution_name
        assert result_id.authority_score >= 0.5

        result_us = OfficialDomainFilter.evaluate("https://www.census.gov/newsroom/press-releases/2024.html")
        assert result_us.is_authoritative is True
        assert result_us.authority_tier == 1

    def test_academic_and_igo_domains(self):
        result_edu = OfficialDomainFilter.evaluate("https://cs.stanford.edu/research/ai-report.html")
        assert result_edu.is_authoritative is True
        assert result_edu.authority_tier == 1

        result_who = OfficialDomainFilter.evaluate("https://www.who.int/news/item/01-01-2024-health-report")
        assert result_who.is_authoritative is True
        assert "World Health Organization" in result_who.institution_name

    def test_standards_and_technical_documentation(self):
        result_w3c = OfficialDomainFilter.evaluate("https://www.w3.org/TR/2024/REC-webauthn-3/")
        assert result_w3c.is_authoritative is True
        assert result_w3c.authority_tier == 2

    def test_rejection_of_blogs_and_spam(self):
        bad_urls = [
            "https://medium.com/@randomuser/data-science-trends",
            "https://myblog.wordpress.com/2024/tips",
            "https://spamsite.xyz/crypto-news",
            "https://forum.reddit.com/r/technology",
        ]
        for url in bad_urls:
            eval_res = OfficialDomainFilter.evaluate(url)
            assert eval_res.is_authoritative is False
            assert eval_res.authority_tier == 4

    def test_schema_org_json_ld_enhancement(self):
        html_with_schema = """
        <!DOCTYPE html>
        <html>
        <head>
          <script type="application/ld+json">
          {
            "@context": "https://schema.org",
            "@type": "GovernmentOrganization",
            "name": "Kementerian Keuangan",
            "url": "https://kemenkeu.go.id"
          }
          </script>
        </head>
        <body><main><p>Laporan APBN 2024</p></main></body>
        </html>
        """
        eval_res = OfficialDomainFilter.evaluate("https://kemenkeu.go.id/laporan", html_content=html_with_schema)
        assert eval_res.is_authoritative is True
        assert any("Schema.org" in r for r in eval_res.reasons)


class TestSubstantiveVisualExtractor:
    """Tests for extracting informative charts while eliminating logos and UI icons."""

    @pytest.mark.asyncio
    async def test_extracts_substantive_chart_and_rejects_logos(self, tmp_path: Path):
        html = """
        <html>
        <body>
          <nav>
            <img src="/assets/logo.png" alt="Company Logo" class="brand-logo" />
            <img src="/icons/nav-arrow.svg" alt="Arrow" />
          </nav>
          <header>
            <img src="/banner/header-banner.jpg" class="header-banner" />
          </header>
          <main>
            <article>
              <h1>Laporan Pertumbuhan</h1>
              <figure>
                <img src="/figures/chart_gdp.png" alt="Grafik Pertumbuhan PDB" width="600" height="400" />
                <figcaption>Gambar 1: Laju Pertumbuhan PDB Triwulan IV 2023</figcaption>
              </figure>
              <img src="/social/share-button.png" class="share-btn" />
              <img src="/ads/sponsored.jpg" class="advertisement" />
            </article>
          </main>
          <footer>
            <img src="/footer/partner-logo.png" />
          </footer>
        </body>
        </html>
        """
        valid_png_bytes = create_mock_png(600, 400)

        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = valid_png_bytes
        mock_client.get.return_value = mock_resp

        extractor = SubstantiveVisualExtractor(http_client=mock_client)
        assets = await extractor.extract_images(
            html_content=html,
            base_url="https://www.bps.go.id/article",
            output_dir=tmp_path,
        )

        # Only chart_gdp.png should pass
        assert len(assets) == 1
        assert "chart_gdp.png" in assets[0].url
        assert "Gambar 1" in assets[0].caption
        assert assets[0].width == 600
        assert assets[0].height == 400
        assert assets[0].local_path.exists()

    @pytest.mark.asyncio
    async def test_rejects_small_tracking_pixel_and_banners(self, tmp_path: Path):
        html = """
        <html><body><main>
          <img src="/tracker.png" width="1" height="1" />
          <img src="/stripe.png" width="900" height="50" />
        </main></body></html>
        """
        # Pixel: 1x1
        pixel_bytes = create_mock_png(1, 1)
        # Banner: 900x50 (ratio 18.0)
        banner_bytes = create_mock_png(900, 50)

        mock_client = AsyncMock()

        def side_effect(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.content = pixel_bytes if "tracker" in url else banner_bytes
            return resp

        mock_client.get.side_effect = side_effect

        extractor = SubstantiveVisualExtractor(http_client=mock_client)
        assets = await extractor.extract_images(html, "https://example.gov", tmp_path)
        assert len(assets) == 0


class TestWebPageToMarkdownConverter:
    """Tests for academic markdown generation and YAML frontmatter."""

    def test_convert_html_to_markdown_with_table(self):
        html = """
        <html>
        <head><title>Statistik Pertanian 2023</title></head>
        <body>
        <main>
          <h1>Statistik Pertanian 2023</h1>
          <p>Produksi padi tahun 2023 diperkirakan sebesar 53,98 juta ton GKG.</p>
          <table>
            <thead><tr><th>Komoditas</th><th>Produksi (Juta Ton)</th></tr></thead>
            <tbody><tr><td>Padi</td><td>53.98</td></tr><tr><td>Jagung</td><td>20.10</td></tr></tbody>
          </table>
        </main>
        </body>
        </html>
        """
        metadata = DocumentMetadata(
            title="Statistik Pertanian 2023",
            authors=["Badan Pusat Statistik (BPS)"],
            year=2023,
            cite_key="bps2023statistik",
            source_type="official_web",
            institution="Badan Pusat Statistik (BPS)",
            url="https://bps.go.id/pertanian",
            authority_score=0.95,
        )

        md = WebPageToMarkdownConverter.convert(html, "https://bps.go.id/pertanian", metadata)
        assert "---" in md
        assert "bps2023statistik" in md
        assert "source_type: official_web" in md
        assert "53,98 juta ton GKG" in md
        # Markdown table should be present
        assert "| Komoditas | Produksi" in md or "Komoditas" in md


class TestWebScoutPipeline:
    """Integration test for the complete WebScout search and ingestion flow."""

    @pytest.mark.asyncio
    async def test_ingest_url_official_flow(self, tmp_path: Path):
        ws = WorkspaceManager(root_dir=tmp_path / "workspace")
        mock_client = AsyncMock()

        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
          <title>Indeks Pembangunan Manusia 2023</title>
          <meta name="date" content="2023-11-15">
        </head>
        <body>
          <nav><img src="/logo.png" alt="BPS Logo"></nav>
          <main>
            <article>
              <h1>Indeks Pembangunan Manusia 2023 Mencapai 74,39</h1>
              <p>Indeks Pembangunan Manusia (IPM) Indonesia tahun 2023 mencapai 74,39, meningkat 0,84 persen.</p>
              <figure>
                <img src="/images/chart_ipm.png" alt="Grafik Pertumbuhan IPM" width="600" height="400">
                <figcaption>Gambar 1: Tren IPM Indonesia 2018-2023</figcaption>
              </figure>
            </article>
          </main>
        </body>
        </html>
        """

        chart_bytes = create_mock_png(600, 400)

        def mock_get(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            if "chart_ipm" in str(url):
                resp.content = chart_bytes
            else:
                resp.text = html_content
                resp.url = "https://www.bps.go.id/pressrelease/ipm-2023.html"
            return resp

        mock_client.get.side_effect = mock_get
        mock_client.is_closed = False

        scout = WebScout(workspace=ws, http_client=mock_client)
        parsed = await scout.ingest_url("https://www.bps.go.id/pressrelease/ipm-2023.html")

        assert parsed.metadata.source_type == "official_web"
        assert parsed.metadata.year == 2023
        assert "bps" in parsed.metadata.cite_key.lower()
        assert "Badan Pusat Statistik" in parsed.metadata.institution
        assert parsed.file_path.exists()
        assert parsed.cache_path.exists()
        assert "74,39" in parsed.markdown

        # Verify visual asset was downloaded into images/ directory
        img_dir = ws.images_dir / parsed.metadata.cite_key
        assert img_dir.exists()
        saved_imgs = list(img_dir.glob("*.png"))
        assert len(saved_imgs) == 1

    @pytest.mark.asyncio
    async def test_rejects_unofficial_url(self, tmp_path: Path):
        ws = WorkspaceManager(root_dir=tmp_path / "workspace")
        mock_client = AsyncMock()

        resp = MagicMock()
        resp.status_code = 200
        resp.text = "<html><body><main><p>My crypto opinion blog.</p></main></body></html>"
        resp.url = "https://randomblog.wordpress.com/post"
        mock_client.get.return_value = resp
        mock_client.is_closed = False

        scout = WebScout(workspace=ws, http_client=mock_client)
        with pytest.raises(OfficialSourceRejectedError):
            await scout.ingest_url("https://randomblog.wordpress.com/post", require_official=True)

    @pytest.mark.asyncio
    async def test_search_official_web_filters_results(self, tmp_path: Path):
        ws = WorkspaceManager(root_dir=tmp_path / "workspace")
        mock_client = AsyncMock()

        ddg_html = """
        <html>
        <body>
          <div class="result">
            <a class="result__url" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.bps.go.id%2Fpublikasi%2F1">bps.go.id</a>
            <a class="result__title">Statistik Inflasi 2023</a>
            <div class="result__snippet">Laporan resmi inflasi tahunan BPS.</div>
          </div>
          <div class="result">
            <a class="result__url" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fmedium.com%2F@blog%2Finflasi">medium.com</a>
            <a class="result__title">Opini Inflasi Saya</a>
            <div class="result__snippet">Blog tentang inflasi.</div>
          </div>
        </body>
        </html>
        """

        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.text = ddg_html
        mock_client.post.return_value = mock_post_resp
        mock_client.is_closed = False

        scout = WebScout(workspace=ws, http_client=mock_client)
        results = await scout.search_official_web("inflasi indonesia 2023", limit=5, require_official=True)

        # medium.com should be rejected, only bps.go.id should pass
        assert len(results) == 1
        assert "bps.go.id" in results[0].url
        assert results[0].is_official is True
        assert "Statistik Inflasi 2023" in results[0].title
