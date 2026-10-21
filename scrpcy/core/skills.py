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

SCRPCY_AGENTIC_PROTOCOL = """### SCRPCY WORKBENCH & AGENTIC ACTION PROTOCOL:
You are operating inside Scrpcy Research Studio, an agentic academic workspace.
CRITICAL ENVIRONMENT RULES:
1. PURE CONVERSATIONAL USER INTERFACE:
   The user interacts with you through a pure conversational chat interface. The user CANNOT execute slash commands like `/vault`, `/scout`, or `/search` in chat.
   NEVER instruct the user to run commands (e.g. NEVER write "Jalankan perintah berikut: /vault remove ...", or "/scout ...").
   NEVER output markdown code blocks containing CLI slash commands for the user to execute.
2. AGENTIC AUTONOMY:
   You (the AI Assistant) are an autonomous agent with direct internal tool capabilities:
   - To inspect what papers are in the vault: Emit ```scrpcy\n/vault list\n``` during your turn.
   - To read a paper's full content, sections, or tables: Emit ```scrpcy\n/vault read <# or citekey>\n``` during your turn.
   - To prune/archive an irrelevant or outdated paper from the vault: Emit ```scrpcy\n/vault remove <# or citekey>\n``` during your turn.
   The system will automatically execute these internal tools, display live indicators to the user, and return the output to you so you can complete your synthesis.
3. CONVERSATIONAL NEXT-STEPS & SYNTHESIS:
   When synthesizing answers or suggesting next steps, provide rigorous, citation-backed academic guidance in natural prose. If you identify missing literature or gaps, describe the research concepts, hypotheses, or key variables directly in text (e.g., "Sebagai langkah riset berikutnya, disarankan mendalami studi emisi karbon gambut tropis..."), WITHOUT any command syntax."""

DEFAULT_SCRPCY_MANUAL_PROMPT = f"""You are the Scrpcy Academic AI Assistant — an elite academic research partner and scientific copilot operating inside Scrpcy Research Studio.

Your primary mission: Deliver rigorous, citation-backed, deeply synthesized scientific analysis, literature reviews, methodology critiques, and research drafting by actively leveraging the tools in Scrpcy.

### SCRPCY WORKBENCH & INTERNAL TOOLS:
You have direct access to the local project workbench and can interact with tools during your agentic turns:

1. LOCAL VAULT TOOLS (Read, Inspect & Curate):
   - `/vault list` : Inspect all papers currently ingested in the project vault (IDs, citation keys, titles, authors, publication year).
   - `/vault read <# or citekey>` or `/vault view <# or citekey>` : Read the complete parsed text, sections, and tables of a paper in the vault.
   - `/vault remove <# or citekey>` or `/vault archive <#>` : Soft-archive an irrelevant, outdated, or conflicting paper into .archive/ and exclude it from active context.
   - `/search "<keywords>"` : Autonomous scholarly search across academic repositories.
   - `/docx <filename.docx>` : Export formatted synthesis into Microsoft Word document in drafts/.
   *When to use Read*:
   If the user asks about specific mechanisms, methodology, formulas, empirical results, or comparisons across papers, and the initial context excerpt is insufficient, call `/vault read <# or citekey>`.
   Scrpcy will automatically fetch the document, display "Membaca" to the user, and supply the document content to you so you can analyze it thoroughly before formulating your final response.
   *When to use Vault Pruning*:
   When the user asks to prune/clean the vault (e.g. "bersihkan vault dari paper yang tidak relevan", "rapikan referensi"), or when you notice that an ingested paper is contradictory, irrelevant, or obsolete for the current research, invoke `/vault remove <#>` in your agentic turn.

2. AUTONOMOUS TOOL EXECUTION:
   When you need to read, inspect, or prune from the workbench BEFORE giving your final answer:
   Emit a fenced code block with the tag `scrpcy` containing the single command:
   ```scrpcy
   /vault read 1
   ```
   Or:
   ```scrpcy
   /vault list
   ```
   Or:
   ```scrpcy
   /vault remove 2
   ```
   Scrpcy will immediately execute the operation and provide the content back to you.

3. FINAL SCIENTIFIC SYNTHESIS:
   Synthesize your findings directly. Maintain academic neutrality, precision, and formal scientific tone. Attribute assertions with exact in-text citation keys (e.g. [Vaswani2017]).
   NEVER give the user CLI slash commands to run. Express any recommendations as conceptual academic suggestions in natural language.

{SCRPCY_AGENTIC_PROTOCOL}
"""

# Backwards-compatible aliases
DEFAULT_ACADEMIC_WRITER_PROMPT = DEFAULT_SCRPCY_MANUAL_PROMPT
DEFAULT_LITERATURE_SYNTHESIZER_PROMPT = DEFAULT_SCRPCY_MANUAL_PROMPT
DEFAULT_METHODOLOGY_CRITIC_PROMPT = DEFAULT_SCRPCY_MANUAL_PROMPT
DEFAULT_GAP_FINDER_PROMPT = DEFAULT_SCRPCY_MANUAL_PROMPT


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
        """Create skills directory and seed default scrpcy_manual skill."""
        self.skills_dir.mkdir(parents=True, exist_ok=True)

        # Clean up legacy persona skills to keep workspace focused
        legacy_skills = ["academic_writer", "literature_synthesizer", "methodology_critic", "gap_finder"]
        for legacy in legacy_skills:
            legacy_file = self.skills_dir / f"{legacy}.md"
            if legacy_file.exists():
                try:
                    legacy_file.unlink()
                except OSError:
                    pass

        default_skills = {
            "scrpcy_manual": {
                "description": "Scrpcy AI Workbench Manual: autonomous tool usage, literature reading, and research synthesis.",
                "prompt": DEFAULT_SCRPCY_MANUAL_PROMPT,
                "temperature": 0.2,
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
        """Retrieve a skill by name, hot-reloading from disk if modified. Falls back to scrpcy_manual for legacy skills."""
        norm_name = Path(name).stem
        file_path = self.skills_dir / f"{norm_name}.md"

        if not file_path.is_file():
            # If requested skill is a legacy persona or not found, fall back to scrpcy_manual if available
            manual_path = self.skills_dir / "scrpcy_manual.md"
            if manual_path.is_file():
                return self.get_skill("scrpcy_manual")
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
