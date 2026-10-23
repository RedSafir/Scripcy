"""Configuration manager for local scrpcy settings and BYOK credentials."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
from typing import Any
import urllib.request

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scrpcy.exceptions import ConfigError

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_CONFIG_ENV = "SCRPCY_CONFIG"
DEFAULT_CONFIG_PATH = Path.home() / ".scrpcy" / "config.json"


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
        req = urllib.request.Request(endpoint, headers={"User-Agent": "scrpcy"})
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


class CitationPreferences(BaseModel):
    """Preferences for evidence synthesis and reference citation."""

    style: str = Field(default="apa", description="Citation style (e.g. apa, bibtex, ieee, inline).")
    include_doi: bool = Field(default=True, description="Whether to append DOI links when available.")
    max_evidence_chunks: int = Field(default=5, ge=1, le=50, description="Maximum evidence passages to inject.")
    context_token_budget: int = Field(default=4000, ge=256, le=128000, description="Max token budget for retrieved context.")


class ScrpcyConfig(BaseModel):
    """Local configuration schema for scrpcy AI studio."""

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
    workspace_path: str = Field(
        default="~/.scrpcy/workspace",
        description="Base path for workspace caches and document vaults.",
    )
    skills_dir: str = Field(
        default="./skills",
        description="Directory containing custom skill markdown prompts.",
    )

    def get_api_key(self, provider: str) -> str | None:
        """Resolve API key for a provider, checking stored config then environment variables."""
        norm_provider = provider.lower().strip()
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
            return self.api_bases.get("9router") or os.environ.get("ROUTER9_BASE_URL") or "http://localhost:20128/v1"
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
    def load(cls, path: Path | str | None = None) -> ScrpcyConfig:
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
    env_path = os.environ.get(DEFAULT_CONFIG_ENV)
    if env_path:
        return Path(env_path).expanduser().resolve()
    return DEFAULT_CONFIG_PATH.expanduser().resolve()


class ConfigManager:
    """High-level manager for configuration updates, reloads, and persistence."""

    def __init__(self, config_path: Path | str | None = None, auto_load: bool = True) -> None:
        self.config_path = resolve_config_path(config_path)
        self.config: ScrpcyConfig = ScrpcyConfig.load(self.config_path) if auto_load else ScrpcyConfig()

    def reload(self) -> ScrpcyConfig:
        """Reload configuration from disk."""
        self.config = ScrpcyConfig.load(self.config_path)
        return self.config

    def save(self) -> Path:
        """Persist current configuration to disk."""
        return self.config.save(self.config_path)

    def update(self, **kwargs: Any) -> ScrpcyConfig:
        """Update configuration fields and persist changes."""
        current_data = self.config.model_dump()
        current_data.update(kwargs)
        try:
            self.config = ScrpcyConfig.model_validate(current_data)
        except ValidationError as exc:
            raise ConfigError(f"Configuration update validation failed: {exc}") from exc
        self.save()
        return self.config
