"""Official web discovery, authority filtering, and clean academic markdown conversion."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
import json
import logging
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

from bs4 import BeautifulSoup
import httpx
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from scrpcy.core.models import DocumentMetadata, ParsedDocument
from scrpcy.core.workspace import WorkspaceManager
from scrpcy.exceptions import OfficialSourceRejectedError, WebScoutError

logger = logging.getLogger(__name__)

OFFICIAL_GOV_TLD_RE = re.compile(
    r"(\.gov|\.go\.id|\.gov\.[a-z]{2}|\.gouv\.[a-z]{2}|\.gob\.[a-z]{2}|\.gc\.ca|\.go\.jp)$",
    re.IGNORECASE,
)
OFFICIAL_EDU_TLD_RE = re.compile(
    r"(\.edu|\.ac\.id|\.ac\.[a-z]{2}|\.edu\.[a-z]{2}|\.ac\.jp|\.edu\.sg)$",
    re.IGNORECASE,
)
OFFICIAL_IGO_DOMAINS = {
    "who.int", "worldbank.org", "un.org", "imf.org", "oecd.org",
    "adb.org", "wto.org", "unesco.org", "ipcc.ch", "cern.ch",
    "fao.org", "wipo.int", "itu.int", "ilo.org", "unicef.org",
}
OFFICIAL_STANDARDS_DOMAINS = {
    "w3.org", "iso.org", "ieee.org", "ietf.org", "nist.gov",
    "python.org", "pytorch.org", "docs.kernel.org", "developer.mozilla.org",
    "tensorflow.org", "golang.org", "rust-lang.org",
}
SPAM_DOMAINS_RE = re.compile(
    r"(\.blogspot\.com|\.wordpress\.com|\.wixsite\.com|\.weebly\.com|\.medium\.com|\.substack\.com|\.dev\.to|\.hashnode\.dev|\.kompasiana\.com|\.kaskus\.co\.id)$",
    re.IGNORECASE,
)
SPAM_TLD_RE = re.compile(
    r"(\.xyz|\.top|\.click|\.buzz|\.club|\.info|\.tk|\.ml|\.ga|\.cf|\.gq)$",
    re.IGNORECASE,
)

KNOWN_INSTITUTIONS: dict[str, str] = {
    "bps.go.id": "Badan Pusat Statistik (BPS)",
    "kemkes.go.id": "Kementerian Kesehatan RI",
    "menlhk.go.id": "Kementerian Lingkungan Hidup dan Kehutanan",
    "kemenkeu.go.id": "Kementerian Keuangan RI",
    "kemdikbud.go.id": "Kementerian Pendidikan, Kebudayaan, Riset, dan Teknologi",
    "bappenas.go.id": "Kementerian PPN / Bappenas",
    "bi.go.id": "Bank Indonesia",
    "ojk.go.id": "Otoritas Jasa Keuangan",
    "who.int": "World Health Organization (WHO)",
    "worldbank.org": "World Bank",
    "un.org": "United Nations",
    "oecd.org": "OECD",
    "imf.org": "International Monetary Fund (IMF)",
    "w3.org": "World Wide Web Consortium (W3C)",
    "iso.org": "International Organization for Standardization (ISO)",
    "nist.gov": "National Institute of Standards and Technology (NIST)",
    "nih.gov": "National Institutes of Health (NIH)",
    "cdc.gov": "Centers for Disease Control and Prevention (CDC)",
    "nasa.gov": "NASA",
}

FIGURE_CAPTION_RE = re.compile(
    r"^(?:Figure|Fig\.|FIG\.|Gambar|Grafik|Diagram|Bagan|Peta|Chart)\s*(\d+[a-zA-Z]?|[IVXLCDM]+)[\s.:—–-]+(.*)",
    re.IGNORECASE,
)

JUNK_IMG_RE = re.compile(
    r"(logo|avatar|icon|badge|button|share|social|advert|ad-|banner|sponsor|"
    r"gravatar|sprite|nav|pixel|analytics|favicon|author|profile|widget|"
    r"arrow|chevron|spinner|loader|cookie|header|footer|thumb)",
    re.IGNORECASE,
)


class WebAuthorityResult(BaseModel):
    """Authority trust score evaluation for a web resource."""

    is_authoritative: bool
    authority_score: float
    authority_tier: int
    institution_name: str
    domain: str
    reasons: list[str] = Field(default_factory=list)


class SubstantiveVisualAsset(BaseModel):
    """Substantive figure, chart, or infographic extracted from an official web page."""

    url: str
    local_path: Path
    caption: str
    alt_text: str
    width: int
    height: int


class WebSearchResult(BaseModel):
    """Search hit discovered via web scout search."""

    title: str
    url: str
    snippet: str
    domain: str
    authority_score: float
    is_official: bool
    institution: str = ""


class OfficialDomainFilter:
    """Evaluates whether web pages originate from official, sovereign, or accredited entities."""

    @classmethod
    def evaluate(cls, url: str, html_content: str = "") -> WebAuthorityResult:
        """Score authority based on TLD, domain hierarchy, Schema.org, and academic metadata."""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]

        score = 0.0
        reasons: list[str] = []
        tier = 4
        institution = ""

        # Match known institutional domains
        for known_dom, known_inst in KNOWN_INSTITUTIONS.items():
            if domain == known_dom or domain.endswith("." + known_dom):
                institution = known_inst
                break

        # 1. TLD & Sovereign Domain Scoring
        if OFFICIAL_GOV_TLD_RE.search(domain) or any(domain == igo or domain.endswith("." + igo) for igo in OFFICIAL_IGO_DOMAINS):
            score += 0.55
            tier = 1
            reasons.append("Sovereign government or official intergovernmental domain")
        elif OFFICIAL_EDU_TLD_RE.search(domain):
            score += 0.50
            tier = 1
            reasons.append("Accredited academic or higher education domain")
        elif any(domain == std or domain.endswith("." + std) for std in OFFICIAL_STANDARDS_DOMAINS):
            score += 0.50
            tier = 2
            reasons.append("Official technical standard, specification, or software documentation")
        elif domain.startswith("docs.") or domain.startswith("developer.") or domain.startswith("research."):
            score += 0.25
            tier = 2
            reasons.append("Primary technical documentation or research subdomain")

        # Negative checks
        if SPAM_DOMAINS_RE.search(domain):
            score -= 0.60
            reasons.append("Identified free blogging platform, UGC forum, or non-institutional domain")
        if SPAM_TLD_RE.search(domain):
            score -= 0.45
            reasons.append("High-risk or spam TLD detected")

        # 2. Structural HTML Metadata Analysis (if HTML provided)
        if html_content:
            soup = BeautifulSoup(html_content, "html.parser")

            # Schema.org JSON-LD parsing
            for script in soup.find_all("script", type="application/ld+json"):
                try:
                    data = json.loads(script.string or "{}")
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        stype = str(item.get("@type", ""))
                        if any(t in stype for t in ["Government", "NGO", "CollegeOrUniversity", "ResearchOrganization", "ScholarlyArticle", "Report", "Dataset"]):
                            score += 0.25
                            reasons.append(f"Authoritative Schema.org entity '{stype}' detected")
                            pub = item.get("publisher", {})
                            if isinstance(pub, dict) and pub.get("name") and not institution:
                                institution = str(pub.get("name"))
                except Exception:
                    continue

            # Highwire Press & Dublin Core academic meta tags
            if soup.find("meta", attrs={"name": re.compile(r"^citation_", re.I)}):
                score += 0.15
                reasons.append("Highwire Press academic citation tags present")
            if soup.find("meta", attrs={"name": re.compile(r"^dc\.", re.I)}):
                score += 0.15
                reasons.append("Dublin Core publication metadata present")

            # Check for excessive ad scripts
            ad_scripts = soup.find_all("script", src=re.compile(r"adsbygoogle|taboola|outbrain|adnxs", re.I))
            if ad_scripts:
                score -= 0.25
                reasons.append("Commercial ad network scripts detected")

            # Text-to-HTML ratio heuristic
            text_len = len(soup.get_text())
            html_len = max(1, len(html_content))
            if (text_len / html_len) < 0.05 and score < 0.5:
                score -= 0.20
                reasons.append("Low text-to-code ratio indicates potential content aggregator")

        final_score = max(0.0, min(1.0, score))
        is_authoritative = final_score >= 0.50

        if not institution:
            parts = domain.split(".")
            institution = parts[-2].upper() if len(parts) >= 2 else domain.upper()

        return WebAuthorityResult(
            is_authoritative=is_authoritative,
            authority_score=round(final_score, 2),
            authority_tier=tier if is_authoritative else 4,
            institution_name=institution,
            domain=domain,
            reasons=reasons,
        )


class SubstantiveVisualExtractor:
    """Extracts informative figures, charts, and infographics while eliminating logos and UI clutter."""

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._client = http_client

    async def extract_images(
        self,
        html_content: str,
        base_url: str,
        output_dir: Path,
        max_images: int = 5,
    ) -> list[SubstantiveVisualAsset]:
        """Parse DOM, download candidate images, apply dimension/ratio filters, and return valid assets."""
        soup = BeautifulSoup(html_content, "html.parser")
        output_dir.mkdir(parents=True, exist_ok=True)
        valid_assets: list[SubstantiveVisualAsset] = []

        # Target content container: <main>, <article>, or body
        content_root = soup.find("main") or soup.find("article") or soup.body or soup

        # Decompose navigation, headers, footers, and sidebars from content root copy
        for junk in content_root.find_all(["header", "footer", "nav", "aside", "form"]):
            junk.decompose()

        candidate_imgs = content_root.find_all("img")
        client = self._client or httpx.AsyncClient(timeout=10.0, follow_redirects=True)
        should_close = self._client is None

        try:
            for img in candidate_imgs:
                if len(valid_assets) >= max_images:
                    break

                src = img.get("src") or img.get("data-src") or ""
                if not src or src.startswith("data:image"):
                    continue

                # 1. Lexical pattern filtering
                img_ident = f"{src} {img.get('class', '')} {img.get('id', '')} {img.get('alt', '')}"
                if JUNK_IMG_RE.search(img_ident):
                    continue

                # 2. File extension check: accept raster; reject SVG UI icons, GIF animations, ICO
                parsed_src = urlparse(src)
                ext = Path(parsed_src.path).suffix.lower()
                if ext in [".svg", ".gif", ".ico"]:
                    continue

                full_url = urljoin(base_url, src)

                # 3. Caption identification
                caption = ""
                parent_figure = img.find_parent("figure")
                if parent_figure:
                    fc = parent_figure.find("figcaption")
                    if fc:
                        caption = fc.get_text(strip=True)

                if not caption:
                    sibling = img.find_next_sibling(["p", "span", "div"], class_=re.compile(r"caption|keterangan|deskripsi", re.I))
                    if sibling:
                        caption = sibling.get_text(strip=True)

                alt_text = str(img.get("alt", "")).strip()

                # 4. Fetch and validate physical image dimensions with PIL
                try:
                    resp = await client.get(full_url, headers={"User-Agent": "Mozilla/5.0 (ScrpcyAcademicBot/1.0)"})
                    if resp.status_code != 200 or not resp.content:
                        continue

                    img_bytes = resp.content
                    with Image.open(BytesIO(img_bytes)) as pil_img:
                        w, h = pil_img.size
                        # Dimension heuristics: min 250x150, min area 37500 px
                        if w < 250 or h < 150 or (w * h) < 37500:
                            continue
                        ratio = w / float(h)
                        # Aspect ratio between 0.25 (tall) and 4.0 (wide)
                        if ratio < 0.25 or ratio > 4.0:
                            continue

                    idx = len(valid_assets) + 1
                    target_filename = f"figure_{idx}.png"
                    target_path = output_dir / target_filename
                    target_path.write_bytes(img_bytes)

                    valid_assets.append(SubstantiveVisualAsset(
                        url=full_url,
                        local_path=target_path,
                        caption=caption,
                        alt_text=alt_text,
                        width=w,
                        height=h,
                    ))
                except Exception as dl_err:
                    logger.debug(f"Image validation skipped for {full_url}: {dl_err}")
                    continue
        finally:
            if should_close:
                await client.aclose()

        return valid_assets


class WebPageToMarkdownConverter:
    """Converts cleaned HTML bodies into academic Markdown with YAML frontmatter."""

    @classmethod
    def convert(
        cls,
        html_content: str,
        url: str,
        metadata: DocumentMetadata,
        images: list[SubstantiveVisualAsset] | None = None,
    ) -> str:
        """Convert HTML to GFM Markdown with table preservation and frontmatter."""
        body_md = ""

        # Attempt high-fidelity extraction via trafilatura
        try:
            import trafilatura
            body_md = trafilatura.extract(
                html_content,
                url=url,
                output_format="markdown",
                include_tables=True,
                include_images=False,
                include_links=True,
            ) or ""
        except Exception as traf_err:
            logger.debug(f"Trafilatura extraction fallback triggered: {traf_err}")

        # Resilient fallback via BeautifulSoup + markdownify
        if not body_md or len(body_md.strip()) < 100:
            soup = BeautifulSoup(html_content, "lxml")
            for junk in soup(["header", "footer", "nav", "aside", "form", "script", "style", "noscript", "iframe"]):
                junk.decompose()
            for junk_cls in soup.find_all(class_=re.compile(r"cookie|advert|banner|social|share|sidebar|widget|popup", re.I)):
                junk_cls.decompose()

            content_el = soup.find("main") or soup.find("article") or soup.find("div", role="main") or soup.body or soup
            try:
                from markdownify import markdownify as md
                body_md = md(str(content_el), heading_style="ATX", strip=["script", "style"])
            except Exception:
                body_md = content_el.get_text(separator="\n\n")

        clean_body = body_md.strip()

        # Append substantive visual assets with relative local links and captions
        if images:
            clean_body += "\n\n## Visual & Bagan Pendukung\n"
            cite_dir = metadata.cite_key or "web_assets"
            for idx, img_asset in enumerate(images, 1):
                caption_text = img_asset.caption or img_asset.alt_text or f"Gambar {idx}"
                rel_path = f"../images/{cite_dir}/{img_asset.local_path.name}"
                clean_body += f"\n![{caption_text}]({rel_path})\n*{caption_text}*\n"

        frontmatter = metadata.to_frontmatter()
        return f"{frontmatter}\n{clean_body}\n"


class WebScout:
    """Pipeline orchestrator for searching, filtering, and ingesting official web sources."""

    def __init__(
        self,
        workspace: WorkspaceManager | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.workspace = workspace or WorkspaceManager()
        self.web_vault_dir = getattr(self.workspace, "web_dir", self.workspace.vault_dir / "web")
        self.web_vault_dir.mkdir(parents=True, exist_ok=True)
        self._client = http_client

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 ScrpcyAcademic/1.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "id,en-US;q=0.9,en;q=0.8",
            }
            self._client = httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers=headers)
        return self._client

    async def search_official_web(
        self,
        query: str,
        limit: int = 5,
        require_official: bool = True,
    ) -> list[WebSearchResult]:
        """Search DuckDuckGo HTML endpoint and filter candidates using OfficialDomainFilter."""
        client = await self._get_client()
        search_url = "https://html.duckduckgo.com/html/"
        data = {"q": query, "b": "", "kl": ""}

        try:
            resp = await client.post(search_url, data=data)
            resp.raise_for_status()
        except Exception as exc:
            raise WebScoutError(f"DuckDuckGo search query failed for '{query}': {exc}") from exc

        soup = BeautifulSoup(resp.text, "html.parser")
        results: list[WebSearchResult] = []

        for result_node in soup.select(".result"):
            title_node = result_node.select_one(".result__title")
            snippet_node = result_node.select_one(".result__snippet")
            url_node = result_node.select_one(".result__url")

            if not title_node or not url_node:
                continue

            raw_href = url_node.get("href", "") or (title_node.find("a") or {}).get("href", "")
            if not raw_href:
                continue

            # Decode DuckDuckGo redirect wrapper
            parsed_href = urlparse(raw_href)
            if "duckduckgo.com" in parsed_href.netloc:
                qs = parse_qs(parsed_href.query)
                clean_url = qs.get("uddg", [raw_href])[0]
            else:
                clean_url = raw_href

            title_text = title_node.get_text(strip=True)
            snippet_text = snippet_node.get_text(strip=True) if snippet_node else ""

            auth = OfficialDomainFilter.evaluate(clean_url)
            if require_official and not auth.is_authoritative:
                continue

            results.append(WebSearchResult(
                title=title_text,
                url=clean_url,
                snippet=snippet_text,
                domain=auth.domain,
                authority_score=auth.authority_score,
                is_official=auth.is_authoritative,
                institution=auth.institution_name,
            ))

            if len(results) >= limit:
                break

        return results

    async def ingest_url(
        self,
        url: str,
        custom_cite_key: str | None = None,
        save_to_cache: bool = True,
        require_official: bool = True,
        vector_vault: Any | None = None,
    ) -> ParsedDocument:
        """Fetch official web page, extract visuals, format academic Markdown, and persist to vault."""
        client = await self._get_client()
        try:
            resp = await client.get(url)
            resp.raise_for_status()
        except Exception as exc:
            raise WebScoutError(f"Failed to retrieve web document at '{url}': {exc}") from exc

        html_text = resp.text
        auth = OfficialDomainFilter.evaluate(str(resp.url), html_text)

        if require_official and not auth.is_authoritative:
            reason_str = "; ".join(auth.reasons) or "Domain does not meet official criteria."
            raise OfficialSourceRejectedError(
                f"URL '{url}' rejected as official source (score: {auth.authority_score:.2f}). Reason: {reason_str}"
            )

        # Extract page title
        soup = BeautifulSoup(html_text, "html.parser")
        title = ""
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = str(og_title["content"]).strip()
        elif soup.title and soup.title.string:
            title = soup.title.string.strip()
        if not title:
            h1 = soup.find("h1")
            title = h1.get_text(strip=True) if h1 else Path(urlparse(url).path).stem.replace("-", " ").title()

        # Extract publication year
        year = None
        date_meta = soup.find("meta", attrs={"name": re.compile(r"(date|published_time)", re.I)})
        if date_meta and date_meta.get("content"):
            m = re.search(r"\b(20\d\d|19\d\d)\b", str(date_meta["content"]))
            if m:
                year = int(m.group(1))
        if year is None:
            year = datetime.now().year

        # Determine deterministic cite_key
        author_inst = auth.institution_name
        if custom_cite_key:
            cite_key = custom_cite_key
        else:
            m_acr = re.search(r"\(([A-Za-z0-9]+)\)", author_inst)
            slug = m_acr.group(1).lower() if m_acr else re.sub(r"[^\w]", "", author_inst.split()[0]).lower()
            title_slug = ""
            words = [w for w in re.sub(r"[^\w\s]", "", title.lower()).split() if len(w) > 3]
            if words:
                title_slug = words[0]
            cite_key = f"{slug}{year}{title_slug}"

        # Extract substantive visuals
        img_output_dir = self.workspace.images_dir / cite_key
        visual_extractor = SubstantiveVisualExtractor(http_client=client)
        substantive_images = await visual_extractor.extract_images(
            html_content=html_text,
            base_url=str(resp.url),
            output_dir=img_output_dir,
        )

        metadata = DocumentMetadata(
            title=title,
            authors=[author_inst],
            year=year,
            cite_key=cite_key,
            source_type="official_web",
            url=str(resp.url),
            institution=author_inst,
            access_date=datetime.now().strftime("%Y-%m-%d"),
            authority_score=auth.authority_score,
            authority_tier=auth.authority_tier,
        )

        full_markdown = WebPageToMarkdownConverter.convert(
            html_content=html_text,
            url=str(resp.url),
            metadata=metadata,
            images=substantive_images,
        )

        # Save HTML snapshot in vault/web/
        snapshot_filename = f"{cite_key}.html"
        snapshot_path = self.web_vault_dir / snapshot_filename
        snapshot_path.write_text(html_text, encoding="utf-8", errors="replace")

        # Save Markdown in cache/
        cache_path: Path | None = None
        if save_to_cache:
            cache_path = self.workspace.save_to_cache(f"{cite_key}.md", full_markdown)

        # Index in Vector Vault if available
        if vector_vault is not None:
            try:
                img_dicts = [
                    {
                        "id": f"{cite_key}_fig_{i+1}",
                        "paper_id": cite_key,
                        "page_num": 1,
                        "file_path": str(img.local_path),
                        "caption": img.caption or img.alt_text or f"Visual {i+1}",
                        "ai_analysis": f"Informative figure from {author_inst}.",
                        "width": img.width,
                        "height": img.height,
                    }
                    for i, img in enumerate(substantive_images)
                ]
                vector_vault.ingest_web_document(
                    paper_id=cite_key,
                    title=title,
                    markdown_text=full_markdown,
                    source_path=snapshot_path,
                    images=img_dicts,
                )
            except Exception as vec_err:
                logger.debug(f"Vector vault ingestion for web page failed: {vec_err}")

        _, body_markdown = DocumentMetadata.from_frontmatter(full_markdown)

        return ParsedDocument(
            file_path=snapshot_path,
            cache_path=cache_path,
            metadata=metadata,
            markdown=full_markdown,
            body_markdown=body_markdown,
        )

    async def close(self) -> None:
        """Close underlying HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
