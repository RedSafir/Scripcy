"""Markdown skills loader for dynamic system prompts in scrpcy."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
import yaml

from scrpcy.exceptions import SkillNotFoundError

FRONTMATTER_PATTERN = re.compile(
    r"^\s*---\r?\n(.*?)\r?\n---\r?\n?(.*)$",
    re.DOTALL,
)

DEFAULT_ACADEMIC_WRITER_PROMPT = """You are an elite academic research assistant and scientific writer.
Your role is to synthesize complex academic literature into clear, rigorous, and citation-backed scientific prose.

Guidelines:
1. Always ground your claims directly in the provided evidence context.
2. Attribute every claim with precise citation keys (e.g. [Vaswani2017]).
3. If the provided evidence is insufficient to answer a question, clearly acknowledge the gap rather than speculating.
4. Maintain academic neutrality, precision, and formal scientific tone.
"""

DEFAULT_LITERATURE_SYNTHESIZER_PROMPT = """You are a systematic literature review specialist.
Analyze and compare findings across the provided academic paper excerpts.
Highlight methodologies, consensus findings, contradictions, and open research problems.
Use precise in-text citations for all synthesized assertions.
"""


class Skill(BaseModel):
    """Represents a dynamic persona or skill prompt loaded from Markdown."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="Unique identifier for the skill (e.g. academic_writer).")
    description: str = Field(default="", description="Short summary of the skill purpose.")
    system_prompt: str = Field(description="System instructions injected into LLM context.")
    model: str | None = Field(default=None, description="Optional model override for this skill.")
    temperature: float | None = Field(default=None, description="Optional sampling temperature override.")
    source_path: Path = Field(description="Filepath from which this skill was loaded.")
    mtime: float = Field(default=0.0, description="Last modification timestamp on disk.")

    def format_prompt(self, **kwargs: Any) -> str:
        """Render prompt with variable substitutions if format placeholders exist."""
        if not kwargs:
            return self.system_prompt
        try:
            return self.system_prompt.format(**kwargs)
        except (KeyError, IndexError):
            return self.system_prompt


class SkillsLoader:
    """Manages discovery, dynamic loading, and hot-reloading of markdown skill prompts."""

    def __init__(self, skills_dir: Path | str = "./skills", auto_init: bool = True) -> None:
        self.skills_dir = Path(skills_dir).expanduser().resolve()
        self._skills_cache: dict[str, Skill] = {}

        if auto_init:
            self.ensure_skills_dir()

    def ensure_skills_dir(self) -> None:
        """Create skills directory and seed default skills if empty."""
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        default_skills = {
            "academic_writer": {
                "description": "Synthesizes literature into rigorous, cited academic prose.",
                "prompt": DEFAULT_ACADEMIC_WRITER_PROMPT,
                "temperature": 0.2,
            },
            "literature_synthesizer": {
                "description": "Systematic literature review and cross-paper comparison.",
                "prompt": DEFAULT_LITERATURE_SYNTHESIZER_PROMPT,
                "temperature": 0.1,
            },
        }

        for name, spec in default_skills.items():
            file_path = self.skills_dir / f"{name}.md"
            if not file_path.exists():
                self.create_or_update_skill(
                    name=name,
                    system_prompt=spec["prompt"],
                    description=spec["description"],
                    temperature=spec["temperature"],
                )

    def _parse_skill_file(self, file_path: Path) -> Skill:
        """Parse a markdown skill file with optional YAML frontmatter."""
        text = file_path.read_text(encoding="utf-8")
        current_mtime = file_path.stat().st_mtime

        match = FRONTMATTER_PATTERN.match(text)
        if match:
            frontmatter_raw = match.group(1)
            body = match.group(2).strip()
            try:
                meta = yaml.safe_load(frontmatter_raw) or {}
            except yaml.YAMLError:
                meta = {}
            name = meta.get("name") or file_path.stem
            description = meta.get("description", "")
            model = meta.get("model")
            temperature = meta.get("temperature")
            system_prompt = body if body else text.strip()
        else:
            name = file_path.stem
            description = ""
            model = None
            temperature = None
            system_prompt = text.strip()

        return Skill(
            name=name,
            description=description,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            source_path=file_path,
            mtime=current_mtime,
        )

    def get_skill(self, name: str) -> Skill:
        """Retrieve a skill by name, hot-reloading from disk if modified."""
        norm_name = Path(name).stem
        file_path = self.skills_dir / f"{norm_name}.md"

        if not file_path.is_file():
            raise SkillNotFoundError(f"Skill '{name}' not found at '{file_path}'")

        current_mtime = file_path.stat().st_mtime
        cached = self._skills_cache.get(norm_name)

        if cached is None or cached.mtime < current_mtime:
            skill = self._parse_skill_file(file_path)
            self._skills_cache[norm_name] = skill
            return skill

        return cached

    def list_skills(self) -> list[Skill]:
        """Scan skills directory and return all discovered skills."""
        if not self.skills_dir.is_dir():
            return []

        discovered: list[Skill] = []
        for file_path in sorted(self.skills_dir.glob("*.md")):
            skill = self.get_skill(file_path.stem)
            discovered.append(skill)
        return discovered

    def create_or_update_skill(
        self,
        name: str,
        system_prompt: str,
        description: str = "",
        model: str | None = None,
        temperature: float | None = None,
    ) -> Skill:
        """Write a skill definition to disk and update internal cache."""
        norm_name = Path(name).stem
        file_path = self.skills_dir / f"{norm_name}.md"

        frontmatter_dict: dict[str, Any] = {
            "name": norm_name,
            "description": description,
        }
        if model:
            frontmatter_dict["model"] = model
        if temperature is not None:
            frontmatter_dict["temperature"] = temperature

        yaml_str = yaml.safe_dump(frontmatter_dict, sort_keys=False, allow_unicode=True)
        content = f"---\n{yaml_str}---\n\n{system_prompt.strip()}\n"

        self.skills_dir.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

        skill = self._parse_skill_file(file_path)
        self._skills_cache[norm_name] = skill
        return skill

    def reload_all(self) -> dict[str, Skill]:
        """Force reload of all skills from disk."""
        self._skills_cache.clear()
        for skill in self.list_skills():
            self._skills_cache[skill.name] = skill
        return self._skills_cache
