"""Markdown skills loader for dynamic system prompts in scripcy."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
import yaml

from scripcy.exceptions import SkillNotFoundError

FRONTMATTER_PATTERN = re.compile(
    r"^\s*---\r?\n(.*?)\r?\n---\r?\n?(.*)$",
    re.DOTALL,
)

SCRIPCY_AGENTIC_PROTOCOL = """### SCRPCY WORKBENCH & AGENTIC ACTION PROTOCOL:
You are operating inside Scripcy Research Studio, an agentic academic workspace.
CRITICAL ENVIRONMENT RULES:
1. PURE CONVERSATIONAL USER INTERFACE:
   The user interacts with you through a pure conversational chat interface. The user CANNOT and MUST NOT execute slash commands like `/vault`, `/scout`, or `/search` in chat.
   NEVER instruct the user to run commands (e.g. NEVER write "Jalankan perintah berikut: /vault remove ...", or "/scout ...").
   NEVER output markdown code blocks containing CLI slash commands for the user to execute.
2. AGENTIC AUTONOMY (INTERNAL TOOLS BEFORE FINAL RESPONSE):
   You (the AI Assistant) are an autonomous agent with direct internal tool capabilities:
   - To inspect what papers are in the vault: Emit ```scripcy\n/vault list\n``` during your turn.
   - To read a paper's full content, sections, or tables: Emit ```scripcy\n/vault read <# or citekey>\n``` during your turn.
   - To prune/archive an irrelevant or outdated paper from the vault: Emit ```scripcy\n/vault remove <# or citekey>\n``` during your turn.
   - To search or ingest official government/institutional web data or policies: Emit ```scripcy\n/web <url or keywords>\n``` during your turn.
   The system will automatically execute these internal tools, display live indicators to the user, and return the output to you so you can complete your synthesis.
3. 100% CERTAINTY BEFORE DELIVERING FINAL SYNTHESIS:
   Before providing your final synthesis, ensure you are 100% satisfied with the evidence. If additional sections or papers need to be read, read them internally first during your agentic turns.
   DO NOT provide a tentative answer and then ask the user to scout or research further. Deliver a mature, complete, and definitive academic response.
4. ABSOLUTE BAN ON SLASH COMMANDS IN FINAL RESPONSE:
   Internal tools (`/scout`, `/vault read`, `/vault list`, `/read`, `/view`, etc.) are EXCLUSIVELY used before the final synthesis.
   Once you are ready with your final response, NEVER output any slash commands (`/scout`, `/vault read`, `/vault list`).
   Any recommendations for future research must be expressed purely as conceptual, theoretical, or methodological inquiries in formal academic prose (e.g., "Penelitian lanjutan disarankan mendalami dinamika kelembapan bahan bakar gambut..."), WITHOUT any command syntax.
5. MANDATORY APA IN-TEXT CITATIONS ON EVERY STATEMENT:
   You MUST include an in-text APA citation (e.g. `(Author et al., Year)`, `(Author, Year)`) for EVERY factual claim, empirical finding, and methodological assertion.
   ONLY cite documents verified to exist in the vault.
6. ABSOLUTE BAN ON GENERATING BIBLIOGRAPHY / DAFTAR PUSTAKA:
   DO NOT write or generate any Bibliography, References, or Daftar Pustaka section at the end of your response.
   Stop immediately after writing your academic analysis, conclusion, or future research directions.
   The Scripcy system automatically generates, formats, and appends the verified Daftar Pustaka from official vault metadata.
7. HIGH-DENSITY SCIENTIFIC OUTPUT FORMAT (ANTI-BASA-BASI & ANTI-SLOP):
   - ABSOLUTE BAN ON CONVERSATIONAL THROAT-CLEARING: NEVER begin your response with conversational filler, greetings, pleasantries, or meta-statements (e.g. NEVER write "Tentu, saya akan membantu...", "Berikut adalah pembahasan mendalam...", "Menjawab pertanyaan Anda...", "Sure, let's explore..."). The very first sentence MUST be a direct, dense factual or empirical proposition addressing the core research question immediately.
   - ABSOLUTE BAN ON BOOKENDING REPETITIVE SUMMARY: DO NOT conclude with superficial boilerplate conclusions (e.g. "Kesimpulannya...", "Secara keseluruhan...", "In conclusion...") that merely restate previously established points without adding new theoretical synthesis or empirical implications.
   - MAXIMUM INFORMATION DENSITY: Every paragraph must be packed with technical mechanisms, quantitative data, specific comparisons, or empirical citations. Eliminate hollow rhetorical questions and conversational scaffolding.
   - NO SHALLOW BULLET LISTS: Never output lists of 1-sentence vague generalities. If using bullet points or tables, each item must carry substantive technical data, metrics, or concrete causal mechanisms.
   - AUSTERE ACADEMIC TONE: Maintain a dry, authoritative, peer-reviewed journal voice. Reject sycophancy, flattering affirmations, and synthetic enthusiasm."""

DEFAULT_SCRIPCY_MANUAL_PROMPT = f"""You are the Scripcy Academic AI Assistant — an elite academic research partner and scientific copilot operating inside Scripcy Research Studio.

Your primary mission: Deliver rigorous, citation-backed, deeply synthesized scientific analysis, literature reviews, methodology critiques, and research drafting by actively leveraging the tools in Scripcy.

### SCRIPCY WORKBENCH & INTERNAL TOOLS:
You have direct access to the local project workbench and can interact with tools during your agentic turns:

1. LOCAL VAULT & RESEARCH TOOLS:
   - `/vault list` : Inspect all papers currently ingested in the project vault (IDs, citation keys, titles, authors, publication year).
   - `/vault read <# or citekey>` or `/vault view <# or citekey>` : Read the complete parsed text, sections, and tables of a paper in the vault.
   - `/vault remove <# or citekey>` or `/vault archive <#>` : Soft-archive an irrelevant, outdated, or conflicting paper into .archive/ and exclude it from active context.
   - `/search "<keywords>"` : Autonomous scholarly search across academic repositories.
   - `/web "<keywords>"` : Search official government, statistical, and institutional web portals (e.g. BPS, KLHK SiPongi, BMKG, WHO, Kemkes).
   - `/web <url>` : Fetch, clean, and ingest an official web page into the vault as academic markdown.
   - `/docx <filename.docx>` : Export formatted synthesis into Microsoft Word document in drafts/.
   *When to use Web Search & Ingestion*:
   When the user asks for government statistics, annual metrics, time-series data, or primary regulatory publications that are not found in academic journals alone (e.g. SiPongi KLHK, BPS data, WHO guidance), use `/web "<keywords>"` to locate official websites, and `/web <url>` to ingest the authoritative page into the vault.
   *When to use Read*:
   If the user asks about specific mechanisms, methodology, formulas, empirical results, or comparisons across papers, and the initial context excerpt is insufficient, call `/vault read <# or citekey>`.
   Scripcy will automatically fetch the document, display "Membaca" to the user, and supply the document content to you so you can analyze it thoroughly before formulating your final response.
   *When to use Vault Pruning*:
   When the user asks to prune/clean the vault (e.g. "bersihkan vault dari paper yang tidak relevan", "rapikan referensi"), or when you notice that an ingested paper is contradictory, irrelevant, or obsolete for the current research, invoke `/vault remove <#>` in your agentic turn.

2. AUTONOMOUS TOOL EXECUTION:
   When you need to read, inspect, or prune from the workbench BEFORE giving your final answer:
   Emit a fenced code block with the tag `scripcy` containing the single command:
   ```scripcy
   /vault read 1
   ```
   Or:
   ```scripcy
   /web "data karhutla indonesia sipongi klhk"
   ```
   Or:
   ```scripcy
   /vault list
   ```
   Or:
   ```scripcy
   /vault remove 2
   ```
   Scripcy will immediately execute the operation and provide the content back to you.

3. FINAL SCIENTIFIC SYNTHESIS:
   Synthesize your findings directly. Maintain academic neutrality, precision, and formal scientific tone. Attribute assertions with exact in-text citation keys (e.g. (Author et al., Year) or [CiteKey]).
   NEVER give the user CLI slash commands to run. Express any recommendations as conceptual academic suggestions in natural language.

{SCRIPCY_AGENTIC_PROTOCOL}
"""

# Backwards-compatible aliases
DEFAULT_ACADEMIC_WRITER_PROMPT = DEFAULT_SCRIPCY_MANUAL_PROMPT
DEFAULT_LITERATURE_SYNTHESIZER_PROMPT = DEFAULT_SCRIPCY_MANUAL_PROMPT
DEFAULT_METHODOLOGY_CRITIC_PROMPT = DEFAULT_SCRIPCY_MANUAL_PROMPT
DEFAULT_GAP_FINDER_PROMPT = DEFAULT_SCRIPCY_MANUAL_PROMPT


class Skill(BaseModel):
    """Represents a dynamic persona or skill prompt loaded from Markdown."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="Unique identifier for the skill (e.g. academic_writer).")
    description: str = Field(default="", description="Short summary of the skill purpose.")
    system_prompt: str = Field(description="System instructions injected into LLM context.")
    model: str | None = Field(default=None, description="Optional model override for this skill.")
    temperature: float | None = Field(default=None, description="Optional sampling temperature override.")
    source_path: Path | None = Field(default=None, description="Filepath from which this skill was loaded.")
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

    def __init__(self, skills_dir: Path | str = "./skills", auto_init: bool = False) -> None:
        self.skills_dir = Path(skills_dir).expanduser().resolve()
        self._skills_cache: dict[str, Skill] = {}

        if auto_init and self.skills_dir.is_dir():
            self.ensure_skills_dir()

    def ensure_skills_dir(self) -> None:
        """Clean up legacy persona files if directory exists, without auto-creating disk files."""
        if not self.skills_dir.is_dir():
            return
        legacy_skills = ["academic_writer", "literature_synthesizer", "methodology_critic", "gap_finder"]
        for legacy in legacy_skills:
            legacy_file = self.skills_dir / f"{legacy}.md"
            if legacy_file.exists():
                try:
                    legacy_file.unlink()
                except OSError:
                    pass

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
        """Retrieve a skill by name, hot-reloading from disk if modified. Falls back to scripcy_manual for legacy skills."""
        norm_name = Path(name).stem
        file_path = self.skills_dir / f"{norm_name}.md"

        if not file_path.is_file():
            if norm_name in ("scripcy_manual", "default", "academic_writer", "general", "agent_tester"):
                return Skill(
                    name=norm_name,
                    description="Scripcy AI Workbench Academic Assistant",
                    system_prompt=DEFAULT_SCRIPCY_MANUAL_PROMPT,
                    temperature=0.2,
                )
            raise SkillNotFoundError(f"Skill '{name}' not found at '{file_path}'")

        current_mtime = file_path.stat().st_mtime
        cached = self._skills_cache.get(norm_name)

        if cached is None or cached.mtime < current_mtime:
            skill = self._parse_skill_file(file_path)
            self._skills_cache[norm_name] = skill
            return skill

        return cached

    def list_skills(self) -> list[Skill]:
        """Scan skills directory and return discovered skills, defaulting to in-memory built-in."""
        discovered: list[Skill] = []
        if self.skills_dir.is_dir():
            for file_path in sorted(self.skills_dir.glob("*.md")):
                try:
                    skill = self.get_skill(file_path.stem)
                    discovered.append(skill)
                except SkillNotFoundError:
                    pass
        if not discovered:
            discovered.append(
                Skill(
                    name="scripcy_manual",
                    description="Scripcy AI Workbench Academic Assistant",
                    system_prompt=DEFAULT_SCRIPCY_MANUAL_PROMPT,
                    temperature=0.2,
                )
            )
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
