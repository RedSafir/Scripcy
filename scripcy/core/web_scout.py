"""Official web discovery, authority filtering, and clean academic markdown conversion."""

from __future__ import annotations

import base64
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

from scripcy.core.models import DocumentMetadata, ParsedDocument
from scripcy.core.workspace import WorkspaceManager
from scripcy.exceptions import OfficialSourceRejectedError, WebScoutError

logger = logging.getLogger(__name__)

# Multi-part Sovereign Government & Public Suffix TLDs
OFFICIAL_GOV_SUFFIXES = {
    "go.id", "gov", "gov.uk", "gov.au", "mil.id", "mil",
    "desa.id", "asn.id", "gc.ca", "go.jp", "gob.es", "gouv.fr",
    "gov.br", "gov.in", "gov.za", "gov.sg", "govt.nz",
}

# Accredited Academic & Research Suffixes
OFFICIAL_EDU_SUFFIXES = {
    "ac.id", "edu", "edu.au", "ac.uk", "edu.sg",
    "ac.jp", "edu.my", "ac.nz", "edu.cn",
}

OFFICIAL_IGO_DOMAINS = {
    "who.int", "worldbank.org", "un.org", "imf.org", "oecd.org",
    "adb.org", "wto.org", "unesco.org", "ipcc.ch", "cern.ch",
    "fao.org", "wipo.int", "itu.int", "ilo.org", "unicef.org",
    "unep.org", "undp.org", "wmo.int",
}

OFFICIAL_STANDARDS_DOMAINS = {
    "w3.org", "iso.org", "ieee.org", "ietf.org", "nist.gov",
    "python.org", "pytorch.org", "docs.kernel.org", "developer.mozilla.org",
    "tensorflow.org", "golang.org", "rust-lang.org", "ecma-international.org",
}

SPAM_DOMAINS_RE = re.compile(
    r"(?:^|\.)(blogspot\.com|wordpress\.com|wixsite\.com|weebly\.com|medium\.com|"
    r"substack\.com|dev\.to|hashnode\.dev|kompasiana\.com|kaskus\.co\.id|"
    r"squarespace\.com|hubspotpagebuilder\.com)$",
    re.IGNORECASE,
)

SPAM_TLD_RE = re.compile(
    r"(\.xyz|\.top|\.click|\.buzz|\.club|\.info|\.tk|\.ml|\.ga|\.cf|\.gq|\.site|\.fun|\.online|\.rest|\.icu)$",
    re.IGNORECASE,
)

SEO_FARM_PATH_RE = re.compile(
    r"(?:/tag/|/tags/|/kategori/|/category/|/topik/|/topics/|/search\?|"
    r"-(?:terbaru|lengkap|terpercaya|biaya|syarat|cara-mudah|download-gratis|update-20\d\d)"
    r"(?:\.html?|/)?$)",
    re.IGNORECASE,
)

KNOWN_INSTITUTIONS: dict[str, str] = {
    # Indonesian Government, Ministries & Sovereign Agencies
    "bps.go.id": "Badan Pusat Statistik (BPS)",
    "kemkes.go.id": "Kementerian Kesehatan RI",
    "menlhk.go.id": "Kementerian Lingkungan Hidup dan Kehutanan (KLHK)",
    "sipongi.menlhk.go.id": "SiPongi - Karhutla Monitoring Sistem KLHK",
    "kemenkeu.go.id": "Kementerian Keuangan RI",
    "kemdikbud.go.id": "Kementerian Pendidikan, Kebudayaan, Riset, dan Teknologi",
    "bappenas.go.id": "Kementerian PPN / Bappenas",
    "bi.go.id": "Bank Indonesia",
    "ojk.go.id": "Otoritas Jasa Keuangan (OJK)",
    "bmkg.go.id": "Badan Meteorologi, Klimatologi, dan Geofisika (BMKG)",
    "bnpb.go.id": "Badan Nasional Penanggulangan Bencana (BNPB)",
    "brin.go.id": "Badan Riset dan Inovasi Nasional (BRIN)",
    "kominfo.go.id": "Kementerian Komunikasi dan Informatika RI",
    "esdm.go.id": "Kementerian Energi dan Sumber Daya Mineral",
    "pertanian.go.id": "Kementerian Pertanian RI",
    "kkp.go.id": "Kementerian Kelautan dan Perikanan",
    "big.go.id": "Badan Informasi Geospasial (BIG)",
    "setneg.go.id": "Kementerian Sekretariat Negara RI",
    "jdih.maritim.go.id": "JDIH Kemenko Marves",
    "pajak.go.id": "Direktorat Jenderal Pajak RI",

    # International Intergovernmental & Accredited Organizations
    "who.int": "World Health Organization (WHO)",
    "worldbank.org": "World Bank Group",
    "un.org": "United Nations (PBB)",
    "oecd.org": "OECD",
    "imf.org": "International Monetary Fund (IMF)",
    "ipcc.ch": "Intergovernmental Panel on Climate Change (IPCC)",
    "w3.org": "World Wide Web Consortium (W3C)",
    "iso.org": "International Organization for Standardization (ISO)",
    "nist.gov": "National Institute of Standards and Technology (NIST)",
    "nih.gov": "National Institutes of Health (NIH)",
    "cdc.gov": "Centers for Disease Control and Prevention (CDC)",
    "nasa.gov": "NASA",
    "noaa.gov": "National Oceanic and Atmospheric Administration (NOAA)",
    "usgs.gov": "United States Geological Survey (USGS)",
    "bls.gov": "Bureau of Labor Statistics (BLS)",
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
    authority_tier: int = 4
    reasons: list[str] = Field(default_factory=list)
    relevance_score: float = 1.0
    final_score: float = 1.0


# Sectoral Topic Authority Anchors for Specialized Official Portals
TOPIC_AUTHORITY_ANCHORS: dict[str, dict[str, Any]] = {
    "karhutla": {
        "synonyms": ["karhutla", "kebakaran hutan", "titik api", "titik panas", "hotspot", "api hutan"],
        "domains": ["sipongi.menlhk.go.id", "menlhk.go.id", "bmkg.go.id", "bnpb.go.id", "bps.go.id"],
        "metric_terms": ["luas", "hektar", "hotspot", "sebaran"],
        "institution": "SiPongi - Karhutla Monitoring Sistem KLHK",
    },
    "kebakaran hutan": {
        "synonyms": ["kebakaran hutan", "karhutla", "titik api", "titik panas", "hotspot", "api hutan"],
        "domains": ["sipongi.menlhk.go.id", "menlhk.go.id", "bmkg.go.id", "bnpb.go.id", "bps.go.id"],
        "metric_terms": ["luas", "hektar", "hotspot", "sebaran"],
        "institution": "SiPongi - Karhutla Monitoring Sistem KLHK",
    },
    "stunting": {
        "synonyms": ["stunting", "gizi buruk", "tengkes", "ssgi", "ski"],
        "domains": ["kemkes.go.id", "bps.go.id", "bkkbn.go.id"],
        "metric_terms": ["prevalensi", "persentase", "balita", "survei"],
        "institution": "Kementerian Kesehatan RI",
    },
    "inflasi": {
        "synonyms": ["inflasi", "ihk", "indeks harga konsumen", "deflasi"],
        "domains": ["bi.go.id", "bps.go.id", "kemenkeu.go.id"],
        "metric_terms": ["ihk", "persentase", "yoy", "mtm"],
        "institution": "Bank Indonesia",
    },
    "kemiskinan": {
        "synonyms": ["kemiskinan", "garis kemiskinan", "penduduk miskin"],
        "domains": ["bps.go.id", "kemensos.go.id", "bappenas.go.id"],
        "metric_terms": ["persentase", "garis kemiskinan", "susenas"],
        "institution": "Badan Pusat Statistik (BPS)",
    },
}


def is_root_or_generic_landing(url: str) -> bool:
    """Detect if a URL is merely a top-level root or shallow generic landing page
    (e.g., bps.go.id/id, data.go.id/, indonesia.go.id/instantion) without deep content path."""
    try:
        parsed = urlparse(url)
        clean_path = parsed.path.strip("/")
        if not clean_path:
            return True
        shallow_segments = {
            "", "id", "en", "home", "beranda", "index.html", "index.php",
            "instantion", "tentang", "about", "contact", "hubungi-kami", "profil",
        }
        parts = [p.lower() for p in clean_path.split("/") if p]
        if len(parts) == 1 and parts[0] in shallow_segments:
            return True
        if len(parts) == 2 and parts[0] in shallow_segments and parts[1] in shallow_segments:
            return True
        return False
    except Exception:
        return False


def compute_topic_relevance(
    title: str,
    snippet: str,
    url: str,
    core_terms: list[str],
    synonyms: list[str] | None = None,
) -> float:
    """Compute semantic and lexical overlap between search hit and core research topic.
    Implements MindSearch and FlashRAG pre-fetch lexical anchor gate.
    Returns score between 0.0 and 1.0."""
    if not core_terms:
        return 1.0

    haystack = f"{title} {snippet} {url}".lower()
    all_check_terms: set[str] = {t.lower().strip() for t in core_terms if t.strip()}
    if synonyms:
        all_check_terms.update(s.lower().strip() for s in synonyms if s.strip())

    # Lexical Anchor Gate: at least one core term or synonym MUST match
    matched = [t for t in all_check_terms if t in haystack]
    if not matched:
        return 0.0

    title_lower = title.lower()
    snippet_lower = snippet.lower()
    url_lower = url.lower()

    score = 0.40  # Base score for passing the lexical anchor
    if any(t in title_lower for t in all_check_terms):
        score += 0.35  # Strong boost for title match
    if any(t in snippet_lower for t in all_check_terms):
        score += 0.15  # Boost for snippet match
    if any(t in url_lower for t in all_check_terms):
        score += 0.10  # Boost for URL path match

    return min(1.0, score)


class DisambiguatedQuery(BaseModel):
    """Normalized query parameters formulated by QueryDisambiguator."""

    raw_query: str
    core_entity_terms: list[str] = Field(default_factory=list)
    context_terms: list[str] = Field(default_factory=list)
    clean_topic_phrase: str = ""
    synonyms: list[str] = Field(default_factory=list)
    targeted_queries: list[str] = Field(default_factory=list)
    anchored_domains: list[str] = Field(default_factory=list)
    primary_institution: str = ""


class QueryDisambiguator:
    """Disambiguates conversational search prompts, extracts core topical anchors,
    strips generic meta-tokens, and generates targeted search query strategies
    based on Stanford STORM and GPT-Researcher principles."""

    CONVERSATIONAL_PREFIXES = {
        "carikan", "tolong", "coba", "aku", "saya", "kami", "kita", "kamu", "anda",
        "berikan", "carikanlah", "tolonglah", "bagaimana", "mengapa", "apa",
        "apakah", "siapa", "kapan", "dimana", "di mana", "mengenai", "tentang",
        "seperti", "yaitu", "adalah", "ini", "itu", "tersebut", "secara", "pada",
        "please", "find", "search", "give", "tell", "show", "what", "how", "who", "where",
    }

    GENERIC_META_TOKENS = {
        "data", "informasi", "laporan", "statistik", "rekapitulasi", "rekap",
        "angka", "tabel", "grafik", "diagram", "dokumen", "berkas", "file",
        "terbaru", "terkini", "update", "teranyar", "mutakhir", "terupdate",
        "pengertian", "definisi", "arti", "makna", "maksud", "contoh", "fungsi",
        "manfaat", "tujuan", "metode", "cara", "analisis", "kajian", "studi",
        "report", "overview", "info", "information", "statistics", "dataset",
        "latest", "newest", "recent", "definition", "meaning", "analysis",
    }

    @classmethod
    def disambiguate(cls, raw_query: str) -> DisambiguatedQuery:
        """Formulate targeted query strategies without generic meta-token pollution."""
        clean_raw = raw_query.strip().lower()
        tokens = [w for w in re.findall(r"\b[\w\-]+\b", clean_raw) if len(w) > 1]

        core_terms: list[str] = []
        context_terms: list[str] = []
        synonyms: list[str] = []
        anchored_domains: list[str] = []
        institution = ""
        metric_terms: list[str] = []

        # 1. Match against known topic authority anchors
        for anchor_key, anchor_info in TOPIC_AUTHORITY_ANCHORS.items():
            if anchor_key in clean_raw or any(syn in clean_raw for syn in anchor_info["synonyms"]):
                if anchor_key not in core_terms:
                    core_terms.append(anchor_key)
                synonyms.extend(anchor_info["synonyms"])
                anchored_domains.extend(anchor_info["domains"])
                metric_terms.extend(anchor_info.get("metric_terms", []))
                if not institution:
                    institution = anchor_info.get("institution", "")

        # 2. Filter remaining tokens into core vs context vs meta
        for t in tokens:
            if t in cls.CONVERSATIONAL_PREFIXES:
                continue
            if t in cls.GENERIC_META_TOKENS:
                continue
            # Check if t is temporal or geographic
            if re.match(r"^20\d\d$", t) or t in ("tahun", "terakhir", "indonesia", "nasional", "provinsi"):
                if t not in context_terms:
                    context_terms.append(t)
            elif t not in core_terms:
                core_terms.append(t)

        if not core_terms:
            # Fallback if query was entirely meta words
            core_terms = [t for t in tokens if t not in cls.CONVERSATIONAL_PREFIXES]
            if not core_terms:
                core_terms = ["riset"]

        clean_topic = " ".join(core_terms + context_terms)
        core_quoted = f'"{core_terms[0]}"' if core_terms else clean_topic

        # Formulate targeted query list (Stanford STORM perspective pattern)
        metrics_str = f"({' OR '.join(metric_terms[:3])})" if metric_terms else "luas OR angka"
        ctx_str = " ".join(context_terms)

        targeted: list[str] = []
        # Target 1: Official government scoped query with metric expansion & definition exclusion
        targeted.append(f"{core_quoted} {metrics_str} {ctx_str} site:go.id -pengertian -definisi -kamus".strip())

        # Target 2: Anchored primary portal query (if known authority exists)
        if anchored_domains:
            prim_dom = anchored_domains[0]
            targeted.append(f"site:{prim_dom} {core_quoted} {ctx_str}".strip())

        # Target 3: Broad official fallback
        targeted.append(f"{core_quoted} {clean_topic} resmi".strip())

        return DisambiguatedQuery(
            raw_query=raw_query,
            core_entity_terms=core_terms,
            context_terms=context_terms,
            clean_topic_phrase=clean_topic,
            synonyms=list(dict.fromkeys(synonyms)),
            targeted_queries=targeted,
            anchored_domains=list(dict.fromkeys(anchored_domains)),
            primary_institution=institution,
        )


class OfficialDomainScorer:
    """Multi-signal composite authority scorer and official website classifier."""

    MULTI_PART_SUFFIXES = [
        "go.id", "ac.id", "co.id", "mil.id", "desa.id", "asn.id", "sch.id", "net.id", "or.id", "web.id",
        "gov.uk", "ac.uk", "co.uk", "org.uk", "mil.uk",
        "gov.au", "edu.au", "com.au", "org.au",
        "gc.ca", "gov.br", "gov.in", "gov.za", "gov.sg", "edu.sg", "govt.nz", "ac.nz",
        "go.jp", "ac.jp", "co.jp",
        "gob.es", "gouv.fr", "gov.pl", "gov.it",
    ]

    @classmethod
    def parse_domain_parts(cls, url: str) -> dict[str, str]:
        """Extract hostname, registered domain, subdomain, and multi-part public suffix."""
        parsed = urlparse(url)
        hostname = (parsed.netloc or url).lower().strip()
        if ":" in hostname:
            hostname = hostname.split(":")[0]
        if hostname.startswith("www."):
            hostname = hostname[4:]

        matched_suffix = ""
        for sfx in cls.MULTI_PART_SUFFIXES:
            if hostname == sfx or hostname.endswith("." + sfx):
                matched_suffix = sfx
                break

        if matched_suffix:
            prefix = hostname[: -(len(matched_suffix) + 1)] if hostname != matched_suffix else ""
            prefix_parts = prefix.split(".") if prefix else []
            registered_name = prefix_parts[-1] if prefix_parts else ""
            registered_domain = f"{registered_name}.{matched_suffix}" if registered_name else matched_suffix
            subdomain = ".".join(prefix_parts[:-1]) if len(prefix_parts) > 1 else ""
        else:
            parts = hostname.split(".")
            if len(parts) >= 2:
                matched_suffix = parts[-1]
                registered_name = parts[-2]
                registered_domain = f"{registered_name}.{matched_suffix}"
                subdomain = ".".join(parts[:-2]) if len(parts) > 2 else ""
            else:
                matched_suffix = ""
                registered_domain = hostname
                subdomain = ""

        return {
            "hostname": hostname,
            "registered_domain": registered_domain,
            "subdomain": subdomain,
            "suffix": matched_suffix,
        }

    @classmethod
    def evaluate(
        cls,
        url: str,
        html_content: str = "",
        target_official_domains: set[str] | None = None,
        custom_whitelist: set[str] | None = None,
    ) -> WebAuthorityResult:
        """Score domain authority based on TLD, entity registry, Schema.org, and anti-spam heuristics."""
        parts = cls.parse_domain_parts(url)
        hostname = parts["hostname"]
        reg_domain = parts["registered_domain"]
        suffix = parts["suffix"]
        path = urlparse(url).path

        score = 0.0
        reasons: list[str] = []
        tier = 4
        institution = ""

        # 1. Whitelist & Verified Entity Grounding
        if target_official_domains and (hostname in target_official_domains or reg_domain in target_official_domains):
            score = 1.0
            tier = 1
            reasons.append("Verified target entity official domain match (Wikidata / Registry)")
        elif custom_whitelist and (hostname in custom_whitelist or reg_domain in custom_whitelist):
            score = 1.0
            tier = 1
            reasons.append("Verified custom whitelist match")

        # 2. Institutional Registry Check
        for known_dom, known_inst in KNOWN_INSTITUTIONS.items():
            if hostname == known_dom or hostname.endswith("." + known_dom):
                institution = known_inst
                if score < 1.0:
                    score += 0.55
                    is_standards = any(hostname == std or hostname.endswith("." + std) for std in OFFICIAL_STANDARDS_DOMAINS)
                    tier = 2 if is_standards else min(tier, 1)
                    reasons.append(f"Official institution registry match: {known_inst}")
                break

        # 3. Sovereign Government & Public Suffix Scoring
        if score < 1.0:
            if suffix in OFFICIAL_GOV_SUFFIXES or hostname.endswith(".gov"):
                score += 0.55
                tier = min(tier, 1)
                reasons.append(f"Sovereign government TLD authority: .{suffix or 'gov'}")
            elif suffix in OFFICIAL_EDU_SUFFIXES or hostname.endswith(".edu"):
                score += 0.50
                tier = min(tier, 1)
                reasons.append(f"Accredited academic or higher education institution TLD: .{suffix or 'edu'}")
            elif any(hostname == igo or hostname.endswith("." + igo) for igo in OFFICIAL_IGO_DOMAINS):
                score += 0.50
                tier = min(tier, 1)
                reasons.append("Official intergovernmental organization domain")
            elif any(hostname == std or hostname.endswith("." + std) for std in OFFICIAL_STANDARDS_DOMAINS):
                score += 0.50
                tier = min(tier, 2)
                reasons.append("Official technical standard, specification, or software documentation")

            # 4. Authoritative Subdomain Check
            if parts["subdomain"] in ("docs", "developer", "research", "data", "statistik", "sipongi", "jdih"):
                score += 0.20
                tier = min(tier, 2)
                reasons.append(f"Authoritative institutional subdomain '{parts['subdomain']}'")

            # 5. Penalties (Anti-SEO Farm, UGC, Spam TLD)
            if SPAM_DOMAINS_RE.search(hostname):
                score -= 0.55
                reasons.append("Identified free blogging platform, UGC forum, or non-institutional hosting")
            if SPAM_TLD_RE.search(hostname):
                score -= 0.45
                reasons.append("High-risk or spam TLD detected")
            if SEO_FARM_PATH_RE.search(path):
                score -= 0.35
                reasons.append("SEO farm tag, category aggregator, or clickbait slug pattern detected in URL path")

        # 6. Structural HTML Metadata Analysis
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

            # Commercial ad network script detection
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
            parts_h = hostname.split(".")
            institution = parts_h[-2].upper() if len(parts_h) >= 2 else hostname.upper()

        return WebAuthorityResult(
            is_authoritative=is_authoritative,
            authority_score=round(final_score, 2),
            authority_tier=tier if is_authoritative else 4,
            institution_name=institution,
            domain=hostname,
            reasons=reasons,
        )


class OfficialDomainFilter:
    """Evaluates whether web pages originate from official, sovereign, or accredited entities."""

    @classmethod
    def evaluate(
        cls,
        url: str,
        html_content: str = "",
        target_official_domains: set[str] | None = None,
        custom_whitelist: set[str] | None = None,
    ) -> WebAuthorityResult:
        """Proxy to OfficialDomainScorer.evaluate."""
        return OfficialDomainScorer.evaluate(
            url=url,
            html_content=html_content,
            target_official_domains=target_official_domains,
            custom_whitelist=custom_whitelist,
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
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "id,en-US;q=0.9,en;q=0.8",
            }
            self._client = httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers=headers)
        return self._client

    @staticmethod
    def _decode_bing_url(href: str) -> str:
        """Decode target URL from Bing click tracking redirects."""
        if not href:
            return ""
        if "bing.com/ck/a" in href:
            parsed = urlparse(href)
            qs = parse_qs(parsed.query)
            u_val = qs.get("u", [""])[0]
            if u_val.startswith("a1"):
                raw_b64 = u_val[2:]
                raw_b64 += "=" * ((4 - len(raw_b64) % 4) % 4)
                try:
                    decoded = base64.urlsafe_b64decode(raw_b64).decode("utf-8", errors="ignore")
                    if decoded.startswith("http"):
                        return decoded
                except Exception:
                    pass
        return href

    @classmethod
    def _parse_bing_results(cls, html_text: str) -> list[tuple[str, str, str]]:
        """Parse (title, url, snippet) from Bing search HTML."""
        soup = BeautifulSoup(html_text, "html.parser")
        hits: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for node in soup.select("li.b_algo"):
            h2 = node.find("h2")
            a = h2.find("a") if h2 else None
            p = node.select_one(".b_caption p") or node.select_one("p")
            if not a:
                continue
            title = a.get_text(strip=True)
            raw_href = a.get("href", "")
            clean_url = cls._decode_bing_url(raw_href)
            if not clean_url or not clean_url.startswith("http") or clean_url in seen:
                continue
            seen.add(clean_url)
            snippet = p.get_text(strip=True) if p else ""
            hits.append((title, clean_url, snippet))
        return hits

    @staticmethod
    def _parse_yahoo_results(html_text: str) -> list[tuple[str, str, str]]:
        """Parse (title, url, snippet) from Yahoo search HTML."""
        soup = BeautifulSoup(html_text, "html.parser")
        hits: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for node in soup.select(".algo"):
            h3 = node.find("h3")
            a = h3.find("a") if h3 else None
            p = node.select_one(".compText") or node.select_one("p")
            if not a:
                continue
            title = a.get_text(strip=True)
            clean_url = a.get("href", "")
            if not clean_url or not clean_url.startswith("http") or clean_url in seen:
                continue
            seen.add(clean_url)
            snippet = p.get_text(strip=True) if p else ""
            hits.append((title, clean_url, snippet))
        return hits

    @staticmethod
    def _parse_ddg_results(html_text: str) -> list[tuple[str, str, str]]:
        """Parse (title, url, snippet) from DuckDuckGo search HTML."""
        soup = BeautifulSoup(html_text, "html.parser")
        hits: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for result_node in soup.select(".result"):
            title_node = result_node.select_one(".result__title")
            snippet_node = result_node.select_one(".result__snippet")
            url_node = result_node.select_one(".result__url")
            if not title_node or not url_node:
                continue
            raw_href = url_node.get("href", "") or (title_node.find("a") or {}).get("href", "")
            if not raw_href:
                continue
            parsed_href = urlparse(raw_href)
            if "duckduckgo.com" in parsed_href.netloc:
                qs = parse_qs(parsed_href.query)
                clean_url = qs.get("uddg", [raw_href])[0]
            else:
                clean_url = raw_href
            if not clean_url or not clean_url.startswith("http") or clean_url in seen:
                continue
            seen.add(clean_url)
            title = title_node.get_text(strip=True)
            snippet = snippet_node.get_text(strip=True) if snippet_node else ""
            hits.append((title, clean_url, snippet))
        return hits

    async def search_official_web(
        self,
        query: str,
        limit: int = 5,
        require_official: bool = True,
        target_official_domains: set[str] | None = None,
        custom_whitelist: set[str] | None = None,
    ) -> list[WebSearchResult]:
        """Search authoritative web resources using QueryDisambiguator, Cascaded Waterfall filtering,
        and multi-engine fallbacks (Bing, Yahoo, DuckDuckGo)."""
        client = await self._get_client()
        disambiguated = QueryDisambiguator.disambiguate(query)
        primary_query = disambiguated.targeted_queries[0] if disambiguated.targeted_queries else query

        raw_hits: list[tuple[str, str, str]] = []

        # 1. Primary: Bing HTML search
        try:
            resp = await client.get("https://www.bing.com/search", params={"q": primary_query})
            text = getattr(resp, "text", "")
            if isinstance(text, str) and text.strip():
                raw_hits = self._parse_bing_results(text)
            if not raw_hits and primary_query != query:
                resp = await client.get("https://www.bing.com/search", params={"q": query})
                text = getattr(resp, "text", "")
                if isinstance(text, str) and text.strip():
                    raw_hits = self._parse_bing_results(text)
        except Exception as b_err:
            logger.debug(f"Bing search failed, falling back: {b_err}")

        # 2. Secondary fallback: Yahoo HTML search
        if not raw_hits:
            try:
                resp = await client.get("https://search.yahoo.com/search", params={"q": primary_query})
                text = getattr(resp, "text", "")
                if isinstance(text, str) and text.strip():
                    raw_hits = self._parse_yahoo_results(text)
            except Exception as y_err:
                logger.debug(f"Yahoo search failed, falling back: {y_err}")

        # 3. Tertiary fallback: DuckDuckGo HTML endpoint
        if not raw_hits:
            try:
                search_url = "https://html.duckduckgo.com/html/"
                data = {"q": primary_query, "b": "", "kl": ""}
                resp = await client.post(search_url, data=data)
                text = getattr(resp, "text", "")
                if isinstance(text, str) and text.strip():
                    raw_hits = self._parse_ddg_results(text)
                if not raw_hits and primary_query != query:
                    data = {"q": query, "b": "", "kl": ""}
                    resp = await client.post(search_url, data=data)
                    text = getattr(resp, "text", "")
                    if isinstance(text, str) and text.strip():
                        raw_hits = self._parse_ddg_results(text)
            except Exception as ddg_err:
                logger.debug(f"DuckDuckGo search failed: {ddg_err}")

        # Merge anchored domains if available
        effective_target_domains = set(target_official_domains or [])
        if disambiguated.anchored_domains:
            effective_target_domains.update(disambiguated.anchored_domains)

        candidates: list[WebSearchResult] = []
        for title_text, clean_url, snippet_text in raw_hits:
            auth = OfficialDomainFilter.evaluate(
                clean_url,
                target_official_domains=effective_target_domains or None,
                custom_whitelist=custom_whitelist,
            )

            # Compute semantic/lexical overlap against core topic terms (MindSearch pattern)
            relevance = compute_topic_relevance(
                title=title_text,
                snippet=snippet_text,
                url=clean_url,
                core_terms=disambiguated.core_entity_terms,
                synonyms=disambiguated.synonyms,
            )

            # Detect root or generic landing page (e.g. bps.go.id/id, data.go.id/)
            is_root = is_root_or_generic_landing(clean_url)
            if is_root and relevance < 0.70:
                relevance = 0.0

            # Cascaded Waterfall Scoring (FlashRAG formula)
            if relevance >= 0.30:
                final_score = relevance * (1.0 + 0.5 * auth.authority_score)
            else:
                final_score = 0.0

            # Skip spam, content farms, or candidates that failed relevance gating
            if require_official and (auth.authority_score < 0.20 or final_score <= 0.0):
                continue

            candidates.append(WebSearchResult(
                title=title_text,
                url=clean_url,
                snippet=snippet_text,
                domain=auth.domain,
                authority_score=auth.authority_score,
                is_official=auth.is_authoritative,
                institution=auth.institution_name,
                authority_tier=auth.authority_tier,
                reasons=auth.reasons,
                relevance_score=relevance,
                final_score=final_score,
            ))

        # Re-rank by official status, final score, and authority score
        candidates.sort(key=lambda r: (r.is_official, r.final_score, r.authority_score), reverse=True)

        if require_official:
            official_candidates = [c for c in candidates if c.is_official and c.final_score > 0.0]
            if official_candidates:
                return official_candidates[:limit]
            return [c for c in candidates if c.authority_score >= 0.50 and c.final_score > 0.0][:limit]

        return candidates[:limit]

    async def ingest_url(
        self,
        url: str,
        custom_cite_key: str | None = None,
        save_to_cache: bool = True,
        require_official: bool = True,
        vector_vault: Any | None = None,
        required_topic_terms: list[str] | None = None,
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

        # MindSearch Post-Fetch Content Gate: verify topic relevance before persisting
        if required_topic_terms:
            content_lower = f"{title} {full_markdown}".lower()
            has_topic = any(term.lower() in content_lower for term in required_topic_terms)
            if not has_topic and len(full_markdown) < 500:
                raise OfficialSourceRejectedError(
                    f"Content at '{url}' does not discuss requested topic terms: {required_topic_terms}"
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
