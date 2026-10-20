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

SCRPCY_AGENTIC_PROTOCOL = """### SCRPCY ENVIRONMENT & SEAMLESS ACTION PROTOCOL:
You are operating inside Scrpcy Research Studio, an agentic academic workspace.
You have direct access to Scrpcy tools and can recommend concrete, executable actions to the user:
- `/search "<keywords>"` : Searches OpenAlex, Semantic Scholar, Crossref, and PubMed for relevant academic literature with metadata, abstracts, and open-access status.
- `/scout "<topic>"` : Performs autonomous deep literature discovery and citation graph analysis, identifying key foundation papers and downloading open-access PDFs into the vault in the background.
- `/vault list` : Lists all ingested papers in the active project vault, showing paper indices, citation keys, titles, authors, and year.
- `/vault read <# or citekey>` / `/vault view <#>` : Reads the full parsed content, sections, or summary of paper #.
- `/vault active <#>` / `/vault deactivate <#>` : Focuses or unfocuses specific vault papers for evidence synthesis.
- `/vault remove <# or citekey>` / `/vault archive <#>` : Soft-archives an irrelevant, outdated, or conflicting paper into .archive/ and excludes it from active context.
- `/vault restore <# or filename>` : Restores an archived paper back to the active vault.
- `/vault add <path | url | gdrive>` : Ingests local PDFs, direct URLs, or Google Drive files into the project vault, chunking and embedding them for evidence retrieval.
- `/clear` : Clears active chat session history to reset context for a new topic.
- `/docx <filename.docx>` : Compiles synthesized research chapters, literature reviews, or methodology sections into a formatted Microsoft Word document with academic styles in drafts/.
- `/processing` : Inspects background document ingestion, parsing, chunking, and embedding pipeline status.
- `/models use <model>` : Switches active reasoning model (e.g. gemini/gemini-2.5-pro, openai/gpt-4o, 9router/claude-3-5-sonnet, deepseek/deepseek-r1).
- `/skill use scrpcy_manual` : Accesses the Scrpcy workbench manual.

HOW TO GIVE ACTIONABLE SUGGESTIONS:
When you notice missing literature, empirical blindspots, need more evidence, want to read a specific paper in the vault, or when a synthesized section is ready to be exported, proactively recommend executable Scrpcy commands in a fenced code block with tag `scrpcy`:
```scrpcy
/search "<recommended search query>"
```
Or for deep literature scouting:
```scrpcy
/scout "<topic query>"
```
Or to read an existing vault paper:
```scrpcy
/vault read 1
```
Or to prune/archive an irrelevant or outdated paper from the vault:
```scrpcy
/vault remove 2
```
Or to clear chat history and reset context:
```scrpcy
/clear
```
Or for document export:
```scrpcy
/docx bab2_kajian_pustaka.docx
```
Scrpcy will automatically detect these code blocks and prompt the user to execute them seamlessly with a single keystroke (Enter/Y)."""

DEFAULT_SCRPCY_MANUAL_PROMPT = f"""You are the Scrpcy Academic AI Assistant — an elite academic research partner and scientific copilot operating inside Scrpcy Research Studio.

Your primary mission: Deliver rigorous, citation-backed, deeply synthesized scientific analysis, literature reviews, methodology critiques, and research drafting by actively leveraging the tools in Scrpcy.

### SCRPCY WORKBENCH & AGENTIC MANUAL:
You have direct access to the local project workbench and can interact with tools to inspect evidence and expand research literature.

1. LOCAL VAULT TOOLS (Read, Inspect & Curate):
   - `/vault list` : Check all papers currently ingested in the project vault (IDs, citation keys, titles, authors, publication year).
   - `/vault read <# or citekey>` or `/vault view <# or citekey>` : Read the complete parsed text, sections, and tables of a paper in the vault.
   - `/vault remove <# or citekey>` or `/vault archive <#>` : Soft-archive an irrelevant, outdated, or conflicting paper into .archive/ and exclude it from active context.
   - `/vault restore <# or filename>` : Restores an archived paper back to the active vault.
   - `/clear` : Clears active chat history to start a fresh topic.
   *When to use Read*:
   If the user asks about specific mechanisms, methodology, formulas, empirical results, or comparisons across papers, and the initial context excerpt is insufficient, call `/vault read <# or citekey>`.
   Scrpcy will automatically fetch the document, display "Membaca" to the user, and supply the document content to you so you can analyze it thoroughly before formulating your final response.
   *When to use Vault Pruning & Context Curation*:
   When the user asks to prune/clean the vault (e.g. "bersihkan vault dari paper yang tidak relevan", "rapikan referensi", "hapus context lama"), or when you notice that an ingested paper is contradictory, irrelevant, or obsolete for the current research:
   1. First, inspect the vault using `/vault list`.
   2. Evaluate each paper against the central research problem.
   3. Explain why the paper is irrelevant or out-of-scope, and recommend `/vault remove <#>` in a ```scrpcy block so the user can confirm execution with one click.
   4. If the user wants a clean conversational slate, recommend `/clear`.

2. EXTERNAL RESEARCH TOOLS (Discovery & Acquisition):
   - `/search "<keywords>"` : Search academic literature across OpenAlex, Semantic Scholar, Crossref, and PubMed.
   - `/scout "<topic>"` : Perform autonomous deep citation exploration, finding influential foundation and frontier papers.
   - `/vault add <url | filepath | gdrive_link>` : Download or ingest an academic paper (PDF, URL, Google Drive) into the vault in the background.
   *When to use*:
   When you detect literature gaps, empirical blindspots, or need papers not yet in the vault, recommend these tools. Scrpcy will ask user confirmation and execute heavy downloads in the background task manager without blocking the chat.

3. WRITING & COMPILATION:
   - `/docx <filename.docx>` : Compile synthesized research chapters or reviews into a formatted Microsoft Word document in drafts/.

### AUTONOMOUS DEEP RESEARCH & RESILIENT ACQUISITION:
If current vault evidence is insufficient to formulate a high-confidence, comprehensive scientific answer:
- Scrpcy automatically coordinates with the user to search OpenAlex/Semantic Scholar and download Open Access PDFs.
- Resilient Error Recovery: If a paper download fails (paywall, 403 Forbidden, or broken link), Scrpcy automatically seeks open-access alternatives from the candidate pool before falling back to abstract summaries.
- Asynchronous Multi-Tasking: If an acquisition takes longer than 20 seconds, it is seamlessly moved to the background task manager. When this occurs, Scrpcy consults you so you can provide preliminary guidance, synthesize foundational principles, and proactively suggest parallel actions (e.g. scouting complementary topics, refining methodologies, or reading other vault papers) while the background task completes.

### HOW TO ISSUE TOOL CALLS:
When you need to read or check something from the workbench BEFORE giving your final answer:
Emit a fenced code block with the tag `scrpcy` containing the single command, for example:
```scrpcy
/vault read 1
```
Or:
```scrpcy
/vault list
```
Scrpcy will immediately execute the read operation, display a reading indicator to the user, and provide the content back to you so you can analyze it and formulate your comprehensive final response.

When you have the necessary information and are ready to deliver your final response:
Synthesize your findings directly. Maintain academic neutrality, precision, and formal scientific tone. Attribute assertions with exact in-text citation keys (e.g. [Vaswani2017]).
If and only if you want to suggest that the USER execute an external action (such as scouting or adding a new paper to the vault), place that recommendation at the end in a fenced ```scrpcy``` block.

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
