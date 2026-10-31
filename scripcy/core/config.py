"""Configuration manager for local scripcy settings and BYOK credentials."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Literal
import urllib.request

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scripcy.exceptions import ConfigError

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_CONFIG_ENV = "SCRIPCY_CONFIG"
DEFAULT_CONFIG_PATH = Path.home() / ".scripcy" / "config.json"


def detect_9router_key() -> str | None:
    """Attempt to discover active 9Router API key from local 9router installation database."""
    candidates: list[Path] = []
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(Path(appdata) / "9router" / "db" / "data.sqlite")
    candidates.append(Path.home() / ".config" / "9router" / "db" / "data.sqlite")
    candidates.append(Path.home() / ".9router" / "db" / "data.sqlite")

    for db_path in candidates:
        if db_path.is_file():
            try:
                con = sqlite3.connect(str(db_path), timeout=1.0)
                cur = con.cursor()
                row = cur.execute("SELECT key FROM apiKeys WHERE isActive = 1 ORDER BY createdAt DESC LIMIT 1;").fetchone()
                con.close()
                if row and row[0]:
                    return str(row[0]).strip()
            except Exception:
                continue
    return None


def fetch_9router_models(base_url: str = "http://localhost:20128/v1", timeout: float = 1.0) -> list[str]:
    """Fetch list of available models from a running 9Router or OpenAI-compatible gateway."""
    endpoint = f"{base_url.rstrip('/')}/models"
    try:
        req = urllib.request.Request(endpoint, headers={"User-Agent": "scripcy"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                models: list[str] = []
                for item in data.get("data", []):
                    m_id = item.get("id")
                    if m_id:
                        models.append(str(m_id))
                return models
    except Exception:
        pass
    return []


class EffortTierConfig(BaseModel):
    """Configuration bounds for a specific worker search effort tier."""

    min_sources: int = Field(default=1, ge=0, description="Minimum distinct literature sources required.")
    max_sources: int = Field(default=3, ge=1, description="Upper bound of literature sources to fetch.")
    max_search_queries: int = Field(default=3, ge=1, le=20, description="Maximum external search queries allowed.")
    max_download_items: int = Field(default=3, ge=0, le=20, description="Maximum new documents allowed to download.")


class CitationPreferences(BaseModel):
    """Preferences for evidence synthesis and reference citation."""

    style: str = Field(default="apa", description="Citation style (e.g. apa, bibtex, ieee, inline).")
    include_doi: bool = Field(default=True, description="Whether to append DOI links when available.")
    max_evidence_chunks: int = Field(default=5, ge=1, le=50, description="Maximum evidence passages to inject.")
    context_token_budget: int = Field(default=4000, ge=256, le=128000, description="Max token budget for retrieved context.")


class ScripcyConfig(BaseModel):
    """Local configuration schema for scripcy AI studio."""

    model_config = ConfigDict(extra="ignore")

    default_model: str = Field(
        default="gemini/gemini-2.5-pro",
        description="Default LiteLLM model identifier or local endpoint.",
    )
    api_keys: dict[str, str] = Field(
        default_factory=dict,
        description="Key-value mapping of provider names to API keys.",
    )
    api_bases: dict[str, str] = Field(
        default_factory=dict,
        description="Custom base URLs per provider (e.g. 9router or local endpoints).",
    )
    scout_timeout: float = Field(
        default=180.0,
        ge=5.0,
        le=900.0,
        description="Foreground timeout in seconds for automatic literature scouting before backgrounding.",
    )
    citation: CitationPreferences = Field(
        default_factory=CitationPreferences,
        description="Citation and evidence synthesis settings.",
    )
    max_agent_turns: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Maximum agent loop turns for synthesis, verification, and citation loops.",
    )
    max_refinement_turns: int = Field(
        default=8,
        ge=2,
        le=25,
        description="Maximum iterative refinement rounds for FactChecker and Devil's Advocate loops.",
    )
    max_output_tokens: int = Field(
        default=8192,
        ge=512,
        le=65536,
        description="Maximum output token generation limit (prevents truncated responses).",
    )
    contract_unlimited_tokens: bool = Field(
        default=True,
        description="Backward compatible toggle for unlimited token contracts.",
    )
    default_effort_tier: Literal["low", "medium", "extra"] = Field(
        default="medium",
        description="Default baseline worker search effort tier (low, medium, extra).",
    )
    effort_tier_low: EffortTierConfig = Field(
        default_factory=lambda: EffortTierConfig(
            min_sources=0,
            max_sources=1,
            max_search_queries=1,
            max_download_items=1,
        ),
        description="Low effort tier bounds: primarily local vault or deductive reasoning (0-1 sources).",
    )
    effort_tier_medium: EffortTierConfig = Field(
        default_factory=lambda: EffortTierConfig(
            min_sources=1,
            max_sources=3,
            max_search_queries=3,
            max_download_items=3,
        ),
        description="Medium effort tier bounds: standard empirical search (1-3 sources).",
    )
    effort_tier_extra: EffortTierConfig = Field(
        default_factory=lambda: EffortTierConfig(
            min_sources=3,
            max_sources=7,
            max_search_queries=6,
            max_download_items=6,
        ),
        description="Extra effort tier bounds: comprehensive multi-channel synthesis (3-7 sources).",
    )
    enable_query_clarification: bool = Field(
        default=True,
        description="Whether to clarify vague or under-specified queries before Promisor dispatch.",
    )
    max_worker_expansions: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Batas maksimum putaran ekspansi/follow-up search saat Devil's Advocate mendeteksi gap pada temuan worker.",
    )
    workspace_path: str = Field(
        default="~/.scripcy/workspace",
        description="Base path for workspace caches and document vaults.",
    )
    skills_dir: str = Field(
        default="./skills",
        description="Directory containing custom skill markdown prompts.",
    )
    openalex_api_key: str | None = Field(
        default=None,
        description="Optional API key for OpenAlex (grants high-volume daily limits).",
    )
    semantic_scholar_api_key: str | None = Field(
        default=None,
        description="Optional API key for Semantic Scholar.",
    )
    enable_semantic_scholar: bool = Field(
        default=True,
        description="Whether Semantic Scholar is enabled in literature search federation.",
    )
    enable_crossref: bool = Field(
        default=True,
        description="Whether CrossRef is enabled in literature search federation.",
    )
    enable_opencitations: bool = Field(
        default=False,
        description="Whether OpenCitations is enabled in literature search federation.",
    )
    opencitations_api_key: str | None = Field(
        default=None,
        description="Optional API token for OpenCitations.",
    )
    unpaywall_email: str | None = Field(
        default=None,
        description="Email address used for Unpaywall and polite-pool API queries.",
    )

    def get_api_key(self, provider: str) -> str | None:
        """Resolve API key for a provider, checking stored config then environment variables."""
        norm_provider = provider.lower().strip()
        if norm_provider == "openalex" and self.openalex_api_key:
            return self.openalex_api_key
        if norm_provider in ("semantic_scholar", "semanticscholar") and self.semantic_scholar_api_key:
            return self.semantic_scholar_api_key
        if norm_provider == "opencitations" and self.opencitations_api_key:
            return self.opencitations_api_key

        if norm_provider in self.api_keys and self.api_keys[norm_provider]:
            return self.api_keys[norm_provider]

        env_mapping = {
            "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
            "google": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
            "openai": ["OPENAI_API_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY"],
            "cohere": ["COHERE_API_KEY"],
            "groq": ["GROQ_API_KEY"],
            "deepseek": ["DEEPSEEK_API_KEY"],
            "mistral": ["MISTRAL_API_KEY"],
            "openrouter": ["OPENROUTER_API_KEY"],
            "9router": ["ROUTER9_API_KEY", "NINE_ROUTER_API_KEY", "OPENAI_API_KEY"],
            "openalex": ["OPENALEX_API_KEY"],
            "semantic_scholar": ["SEMANTIC_SCHOLAR_API_KEY", "S2_API_KEY"],
            "semanticscholar": ["SEMANTIC_SCHOLAR_API_KEY", "S2_API_KEY"],
            "opencitations": ["OPENCITATIONS_API_KEY"],
        }

        for env_var in env_mapping.get(norm_provider, [f"{norm_provider.upper()}_API_KEY"]):
            val = os.environ.get(env_var)
            if val:
                return val
        return None

    def get_api_key_for_model(self, model: str) -> str | None:
        """Derive provider from model string and return matching API key."""
        model_str = model.lower()
        if model_str.startswith("9router/"):
            return self.get_api_key("9router") or detect_9router_key() or "sk-9router-local"
        if "/" in model_str:
            prefix = model_str.split("/", 1)[0]
            return self.get_api_key(prefix)
        if "gpt" in model_str or "o1" in model_str or "o3" in model_str:
            return self.get_api_key("openai")
        if "claude" in model_str:
            return self.get_api_key("anthropic")
        if "gemini" in model_str:
            return self.get_api_key("gemini")
        return None

    def get_api_base_for_model(self, model: str) -> str | None:
        """Resolve custom base URL for a model or provider."""
        model_str = model.lower()
        if model_str.startswith("9router/"):
            return self.api_bases.get("9router") or os.environ.get("ROUTER9_BASE_URL") or "http://127.0.0.1:20128/v1"
        if "/" in model_str:
            prefix = model_str.split("/", 1)[0]
            if prefix in self.api_bases and self.api_bases[prefix]:
                return self.api_bases[prefix]
            if prefix == "openai":
                return self.api_bases.get("openai") or os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
        return None

    def set_api_key(self, provider: str, key: str) -> None:
        """Store API key for a provider."""
        self.api_keys[provider.lower().strip()] = key.strip()

    def set_api_base(self, provider: str, base_url: str) -> None:
        """Store custom base URL for a provider."""
        self.api_bases[provider.lower().strip()] = base_url.strip().rstrip("/")

    def setup_9router(
        self,
        port: int = 20128,
        host: str = "localhost",
        api_key: str | None = None,
        default_model: str | None = None,
    ) -> None:
        """Convenience method to configure 9Router local smart proxy."""
        base_url = f"http://{host}:{port}/v1"
        self.set_api_base("9router", base_url)
        resolved_key = api_key or self.get_api_key("9router") or detect_9router_key() or "sk-9router-local"
        self.set_api_key("9router", resolved_key)
        if default_model:
            self.default_model = default_model
        else:
            models = fetch_9router_models(base_url)
            if models:
                self.default_model = f"9router/{models[0]}"

    def save(self, path: Path | str | None = None) -> Path:
        """Save configuration to JSON file."""
        target_path = resolve_config_path(path)
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(
                self.model_dump_json(indent=2),
                encoding="utf-8",
            )
            return target_path
        except OSError as exc:
            raise ConfigError(f"Failed to write configuration file at '{target_path}': {exc}") from exc

    @classmethod
    def load(cls, path: Path | str | None = None) -> ScripcyConfig:
        """Load configuration from JSON file or return default if nonexistent."""
        target_path = resolve_config_path(path)
        if not target_path.is_file():
            return cls()

        try:
            raw_text = target_path.read_text(encoding="utf-8")
            data = json.loads(raw_text)
            return cls.model_validate(data)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise ConfigError(f"Failed to read or parse configuration from '{target_path}': {exc}") from exc


def resolve_config_path(path: Path | str | None = None) -> Path:
    """Resolve active configuration path using override, environment, or default."""
    if path is not None:
        return Path(path).expanduser().resolve()
    env_path = os.environ.get(DEFAULT_CONFIG_ENV) or os.environ.get("SCRPCY_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()
    target = DEFAULT_CONFIG_PATH.expanduser().resolve()
    if not target.is_file():
        legacy = (Path.home() / ".scrpcy" / "config.json").expanduser().resolve()
        if legacy.is_file():
            return legacy
    return target


# Backward-compatibility alias
ScrpcyConfig = ScripcyConfig


class ConfigManager:
    """High-level manager for configuration updates, reloads, and persistence."""

    def __init__(self, config_path: Path | str | None = None, auto_load: bool = True) -> None:
        self.config_path = resolve_config_path(config_path)
        self.config: ScripcyConfig = ScripcyConfig.load(self.config_path) if auto_load else ScripcyConfig()

    def reload(self) -> ScripcyConfig:
        """Reload configuration from disk."""
        self.config = ScripcyConfig.load(self.config_path)
        return self.config

    def save(self) -> Path:
        """Persist current configuration to disk."""
        return self.config.save(self.config_path)

    def update(self, **kwargs: Any) -> ScripcyConfig:
        """Update configuration fields and persist changes."""
        current_data = self.config.model_dump()
        current_data.update(kwargs)
        try:
            self.config = ScripcyConfig.model_validate(current_data)
        except ValidationError as exc:
            raise ConfigError(f"Configuration update validation failed: {exc}") from exc
        self.save()
        return self.config

    def update_effort_tier(
        self,
        tier: str,
        min_sources: int,
        max_sources: int | None = None,
        max_search_queries: int | None = None,
        max_download_items: int | None = None,
    ) -> ScripcyConfig:
        """Update source limits for a specific effort tier and persist changes."""
        tier_clean = tier.strip().lower()
        attr_name = f"effort_tier_{tier_clean}"
        if not hasattr(self.config, attr_name):
            raise ConfigError(f"Tingkat effort '{tier}' tidak valid. Pilih: 'low', 'medium', atau 'extra'.")
        curr_tier: EffortTierConfig = getattr(self.config, attr_name)
        new_min = max(0, int(min_sources))
        new_max = max(new_min, int(max_sources)) if max_sources is not None else max(new_min + 1, curr_tier.max_sources)
        new_queries = int(max_search_queries) if max_search_queries is not None else curr_tier.max_search_queries
        new_downloads = int(max_download_items) if max_download_items is not None else curr_tier.max_download_items

        new_tier_obj = EffortTierConfig(
            min_sources=new_min,
            max_sources=new_max,
            max_search_queries=new_queries,
            max_download_items=new_downloads,
        )
        return self.update(**{attr_name: new_tier_obj})
