"""Arch Linux-inspired interactive CLI research studio for scrpcy."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import re
import shlex
import sys
import threading
import time
from typing import Any
import webbrowser

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from scrpcy.core.config import ConfigManager, ScrpcyConfig
from scrpcy.core.engine import LLMEngine
from scrpcy.core.mcp_client import WordCraftClient
from scrpcy.core.models import DocumentMetadata, ParsedDocument
from scrpcy.core.parser import DocumentParser
from scrpcy.core.search import PaperSearcher, SearchResult
from scrpcy.core.skills import SkillsLoader
from scrpcy.core.workspace import WorkspaceManager

# Safe UTF-8 console output for Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ARCH_LOGO = """
       /\\
      /  \\
     /\\   \\
    /      \\
   /   ,,   \\
  /   |  |  -\\
 /_-''    ''-_\\
  S C R P C Y
"""


class CLIStudio:
    """Terminal-native interactive academic studio with Arch Linux aesthetic."""

    def __init__(
        self,
        workspace: WorkspaceManager | None = None,
        config: ScrpcyConfig | None = None,
        console: Console | None = None,
        parser: DocumentParser | None = None,
        wordcraft: WordCraftClient | None = None,
        engine: LLMEngine | None = None,
    ) -> None:
        self.workspace = workspace or WorkspaceManager()
        self.workspace.ensure_directories()
        self.config_manager = ConfigManager()
        self.config = config or self.config_manager.config

        self.skills_loader = SkillsLoader(skills_dir=self.config.skills_dir)
        self.engine = engine or LLMEngine(config=self.config, skills_loader=self.skills_loader)
        self.parser = parser or DocumentParser(workspace=self.workspace)
        self.wordcraft = wordcraft or WordCraftClient(workspace=self.workspace)

        self.console = console or Console(force_terminal=True)
        self.active_skill_name = "academic_writer"
        self.active_documents: set[str] = set()
        self.last_search_results: list[SearchResult] = []
        self.conversation_history: list[dict[str, str]] = []
        self.start_time = time.time()
        self._web_server_thread: threading.Thread | None = None

        # Auto-activate all existing documents on start
        self._sync_vault_cache()

    def _get_cached_stems(self) -> list[str]:
        """Return list of document stems in the workspace cache."""
        return [p.stem for p in self.workspace.list_cached_documents()]

    def _sync_vault_cache(self) -> list[str]:
        """Index available markdown caches in the workspace."""
        cached_files = self._get_cached_stems()
        for stem in cached_files:
            self.active_documents.add(stem)
        return cached_files

    def get_banner(self) -> Table:
        """Construct the Arch Linux-style Neofetch/Fastfetch status grid."""
        table = Table.grid(padding=(0, 3))
        table.add_column(style="bold cyan", justify="center")
        table.add_column(justify="left")

        cached_docs = self._get_cached_stems()
        active_count = len(self.active_documents)
        total_count = len(cached_docs)
        uptime_sec = int(time.time() - self.start_time)
        uptime_str = f"{uptime_sec // 60}m {uptime_sec % 60}s"

        configured_keys = [k for k, v in self.config.api_keys.items() if v]
        key_status = ", ".join(configured_keys) if configured_keys else "[dim]None (search & docx offline mode)[/dim]"

        info_lines = [
            f"[bold cyan]scrpcy[/bold cyan]@[bold white]studio[/bold white]",
            "[cyan]---------------------------------[/cyan]",
            f"[bold cyan]OS[/bold cyan]           : [white]{sys.platform.capitalize()} (Arch Linux CLI Studio)[/white]",
            f"[bold cyan]Uptime[/bold cyan]       : [white]{uptime_str}[/white]",
            f"[bold cyan]Workspace[/bold cyan]    : [white]{self.workspace.root_dir}[/white]",
            f"[bold cyan]Vault[/bold cyan]        : [white]{total_count} papers ({active_count} active rujukan)[/white]",
            f"[bold cyan]Engine Model[/bold cyan] : [bold bright_white]{self.config.default_model}[/bold bright_white]",
            f"[bold cyan]Active Skill[/bold cyan] : [white]{self.active_skill_name}[/white]",
            f"[bold cyan]Citation[/bold cyan]     : [white]{self.config.citation.style.upper()}[/white]",
            f"[bold cyan]Config Keys[/bold cyan]  : [white]{key_status}[/white]",
            f"[bold cyan]MCP Backend[/bold cyan]  : [white]WordCraft (python-docx generator ready)[/white]",
            f"[bold cyan]Paper Search[/bold cyan] : [green]Semantic Scholar + ArXiv (Online)[/green]",
            "",
            "[bold red]●[/bold red] [bold green]●[/bold green] [bold yellow]●[/bold yellow] [bold blue]●[/bold blue] [bold magenta]●[/bold magenta] [bold cyan]●[/bold cyan] [bold white]●[/bold white]",
        ]

        table.add_row(ARCH_LOGO.strip("\n"), "\n".join(info_lines))
        return table

    def print_banner(self) -> None:
        """Display the studio banner and basic instruction tips."""
        self.console.print()
        self.console.print(self.get_banner())
        self.console.print()
        self.console.print("[dim]Type your research question directly to query active papers.[/dim]")
        self.console.print("[dim]Type [bold cyan]/help[/bold cyan] or [bold cyan]?[/bold cyan] to view all available commands.[/dim]")
        self.console.print()

    def print_help(self) -> None:
        """Render a clean Linux-style table of all CLI commands."""
        table = Table(
            title="scrpcy CLI Studio Commands",
            title_style="bold cyan",
            border_style="cyan",
            header_style="bold bright_cyan",
            show_lines=False,
        )
        table.add_column("Command", style="bold green", no_wrap=True)
        table.add_column("Arguments", style="yellow")
        table.add_column("Description", style="white")

        commands = [
            ("/vault", "", "List all indexed papers in workspace vault (alias: /docs, /ls)"),
            ("/active", "<#|stem|all|clear>", "Toggle active document sources for citation context"),
            ("/parse", "<path.pdf>", "Ingest a local PDF into the vault & extract markdown"),
            ("/view", "<#|stem>", "Inspect the frontmatter and content of an indexed paper"),
            ("/search", "<keywords>", "Search academic literature across Semantic Scholar & ArXiv"),
            ("/download", "<#>", "Download a paper from search results into the vault"),
            ("/skills", "", "List all available research skill personas"),
            ("/skill", "<name>", "Switch the active persona (e.g. academic_writer)"),
            ("/model", "<model_name>", "Switch the default LLM model (e.g. gemini/gemini-2.5-pro)"),
            ("/style", "<apa|ieee>", "Change document citation formatting style"),
            ("/docx", "[filename]", "Compile current research chat/draft into a Word document"),
            ("/config", "[set-key <prov> <key>]", "Manage API keys and settings"),
            ("/web", "[port]", "Launch the 3-panel Web Studio server and open browser"),
            ("/status", "", "Re-display the Arch Linux system fetch banner"),
            ("/clear", "", "Clear the terminal screen"),
            ("/exit", "", "Exit the scrpcy research studio (alias: /quit, q)"),
        ]

        for cmd, args, desc in commands:
            table.add_row(cmd, args, desc)

        self.console.print(table)
        self.console.print("\n[dim]Note: Any text entered without a '/' prefix is treated as a research query.[/dim]\n")

    def run(self) -> None:
        """Start the interactive REPL loop."""
        self.print_banner()

        while True:
            try:
                # Linux-style prompt
                prompt_text = "[bold cyan]scrpcy[/bold cyan]@[bold white]studio[/bold white]:[bold blue]~$[/bold blue] "
                raw_input = self.console.input(prompt_text).strip()

                if not raw_input:
                    continue

                if raw_input in ("/exit", "/quit", "exit", "quit", "q"):
                    self.console.print("\n[bold cyan]Closing scrpcy research studio. Goodbye![/bold cyan]")
                    break

                if raw_input.startswith("/"):
                    self.handle_command(raw_input)
                elif raw_input in ("help", "?"):
                    self.print_help()
                elif raw_input == "clear":
                    self.clear_screen()
                else:
                    self.handle_chat(raw_input)

            except (KeyboardInterrupt, EOFError):
                self.console.print("\n\n[bold cyan]Session terminated. Goodbye![/bold cyan]")
                break
            except Exception as exc:
                self.console.print(f"[bold red][!] Error:[/bold red] {exc}\n")

    def clear_screen(self) -> None:
        """Clear the terminal screen and reprint banner."""
        os.system("cls" if os.name == "nt" else "clear")
        self.print_banner()

    def handle_command(self, cmd_line: str) -> None:
        """Route slash commands to their respective handlers."""
        parts = cmd_line.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/help", "/?"):
            self.print_help()
        elif cmd in ("/vault", "/docs", "/ls"):
            self.cmd_vault()
        elif cmd in ("/active", "/select"):
            self.cmd_active(args)
        elif cmd == "/parse":
            self.cmd_parse(args)
        elif cmd == "/view":
            self.cmd_view(args)
        elif cmd == "/search":
            self.cmd_search(args)
        elif cmd == "/download":
            self.cmd_download(args)
        elif cmd == "/skills":
            self.cmd_skills()
        elif cmd == "/skill":
            self.cmd_skill(args)
        elif cmd == "/model":
            self.cmd_model(args)
        elif cmd == "/style":
            self.cmd_style(args)
        elif cmd == "/config":
            self.cmd_config(args)
        elif cmd in ("/docx", "/export"):
            self.cmd_docx(args)
        elif cmd in ("/status", "/fetch", "/info"):
            self.console.print(self.get_banner())
        elif cmd == "/clear":
            self.clear_screen()
        elif cmd == "/web":
            self.cmd_web(args)
        else:
            self.console.print(f"[bold red]Unknown command:[/bold red] {cmd}. Type [bold cyan]/help[/bold cyan] for list of commands.\n")

    def cmd_vault(self) -> None:
        """Display list of papers in the local vault with active selection state."""
        stems = self._get_cached_stems()
        if not stems:
            self.console.print("[dim]Vault is currently empty. Use [bold cyan]/search[/bold cyan] or [bold cyan]/parse <file.pdf>[/bold cyan] to add papers.[/dim]\n")
            return

        table = Table(
            title=f"Indexed Research Papers ({len(stems)} total, {len(self.active_documents)} active)",
            title_style="bold cyan",
            border_style="cyan",
            header_style="bold bright_cyan",
        )
        table.add_column("#", justify="right", style="dim", width=4)
        table.add_column("Active", justify="center", width=8)
        table.add_column("Cite Key", style="bold green", overflow="fold")
        table.add_column("Title", style="white")
        table.add_column("Year", justify="center", width=6)
        table.add_column("Authors", style="dim")

        for idx, stem in enumerate(stems, 1):
            is_active = stem in self.active_documents
            active_marker = "[bold green][✓][/bold green]" if is_active else "[dim][ ][/dim]"

            content = self.workspace.read_from_cache(f"{stem}.md")
            meta, _ = DocumentMetadata.from_frontmatter(content)
            title = meta.title or stem
            year = str(meta.year) if meta.year else "n.d."
            authors_str = ", ".join(meta.authors[:2]) if meta.authors else "Unknown"

            table.add_row(str(idx), active_marker, meta.cite_key or stem, title, year, authors_str)

        self.console.print(table)
        self.console.print("[dim]Use [bold cyan]/active <#|stem>[/bold cyan] to toggle papers in/out of context.[/dim]\n")

    def cmd_active(self, arg: str) -> None:
        """Toggle active papers by index, stem, or batch words."""
        if not arg:
            self.console.print("[yellow]Usage: /active <# | stem | all | clear>[/yellow]\n")
            return

        stems = self._get_cached_stems()
        if not stems:
            self.console.print("[dim]No papers in vault to activate.[/dim]\n")
            return

        if arg == "all":
            self.active_documents.update(stems)
            self.console.print(f"[bold green][✓] Activated all {len(stems)} papers for citation context.[/bold green]\n")
            return

        if arg == "clear":
            self.active_documents.clear()
            self.console.print("[bold yellow][!] Cleared all active papers. Context is now empty.[/bold yellow]\n")
            return

        # Check if argument is a numeric index
        target_stem: str | None = None
        if arg.isdigit():
            idx = int(arg) - 1
            if 0 <= idx < len(stems):
                target_stem = stems[idx]
        else:
            matching = [s for s in stems if s.lower() == arg.lower() or arg.lower() in s.lower()]
            if matching:
                target_stem = matching[0]

        if not target_stem:
            self.console.print(f"[red][x] Document '{arg}' not found in vault.[/red]\n")
            return

        if target_stem in self.active_documents:
            self.active_documents.discard(target_stem)
            self.console.print(f"[yellow][-] Deactivated:[/yellow] [bold]{target_stem}[/bold] (Excluded from context)\n")
        else:
            self.active_documents.add(target_stem)
            self.console.print(f"[green][+] Activated:[/green] [bold]{target_stem}[/bold] (Included as rujukan)\n")

    def cmd_parse(self, arg: str) -> None:
        """Ingest a local PDF document into the workspace vault."""
        if not arg:
            self.console.print("[yellow]Usage: /parse <path/to/paper.pdf>[/yellow]\n")
            return

        pdf_path = Path(arg).expanduser().resolve()
        if not pdf_path.exists() or not pdf_path.is_file():
            self.console.print(f"[red][x] PDF file not found at: '{pdf_path}'[/red]\n")
            return

        with self.console.status(f"[cyan]Parsing PDF and extracting metadata: {pdf_path.name}...[/cyan]"):
            try:
                parsed = self.parser.parse(pdf_path, save_to_cache=True)
                self.active_documents.add(parsed.file_path.stem)
            except Exception as exc:
                self.console.print(f"[bold red][x] Failed to parse PDF:[/bold red] {exc}\n")
                return

        self.console.print(f"[bold green][✓] Successfully ingested:[/bold green] {parsed.metadata.title}")
        self.console.print(f"    Cite Key : [cyan]{parsed.metadata.cite_key}[/cyan]")
        self.console.print(f"    Authors  : {', '.join(parsed.metadata.authors) or 'Unknown'}")
        self.console.print(f"    Year     : {parsed.metadata.year or 'n.d.'}")
        self.console.print(f"    Cached at: [dim]{parsed.cache_path}[/dim]\n")

    def cmd_view(self, arg: str) -> None:
        """Inspect the frontmatter and beginning of a paper's markdown text."""
        if not arg:
            self.console.print("[yellow]Usage: /view <# | stem>[/yellow]\n")
            return

        stems = self._get_cached_stems()
        target_stem: str | None = None
        if arg.isdigit():
            idx = int(arg) - 1
            if 0 <= idx < len(stems):
                target_stem = stems[idx]
        else:
            matching = [s for s in stems if s.lower() == arg.lower() or arg.lower() in s.lower()]
            if matching:
                target_stem = matching[0]

        if not target_stem:
            self.console.print(f"[red][x] Document '{arg}' not found.[/red]\n")
            return

        content = self.workspace.read_from_cache(f"{target_stem}.md")
        preview = "\n".join(content.splitlines()[:35]) + "\n\n... (truncated) ..."
        self.console.print(Panel(preview, title=f"Paper: {target_stem}", border_style="cyan"))
        self.console.print()

    def cmd_search(self, arg: str) -> None:
        """Search academic literature asynchronously and store results for download."""
        if not arg:
            self.console.print("[yellow]Usage: /search <keywords> [--limit N][/yellow]\n")
            return

        limit = 5
        query = arg
        if "--limit" in arg:
            match = re.search(r"--limit\s+(\d+)", arg)
            if match:
                limit = int(match.group(1))
                query = re.sub(r"--limit\s+\d+", "", arg).strip()

        with self.console.status(f"[cyan]Searching literature across Semantic Scholar & ArXiv for '{query}'...[/cyan]"):
            try:
                results = asyncio.run(self._execute_search(query, limit))
                self.last_search_results = results
            except Exception as exc:
                self.console.print(f"[bold red][x] Search failed:[/bold red] {exc}\n")
                return

        if not results:
            self.console.print(f"[yellow]No academic results found for '{query}'.[/yellow]\n")
            return

        table = Table(
            title=f"Literature Search Results for '{query}' ({len(results)} found)",
            title_style="bold cyan",
            border_style="cyan",
            header_style="bold bright_cyan",
        )
        table.add_column("#", justify="right", style="dim", width=4)
        table.add_column("PDF Status", justify="center", width=14)
        table.add_column("Title", style="bold white")
        table.add_column("Year", justify="center", width=6)
        table.add_column("Authors", style="dim")

        for idx, res in enumerate(results, 1):
            pdf_badge = "[bold green][Open Access][/bold green]" if res.has_pdf else "[dim][Paywalled][/dim]"
            authors_str = ", ".join(res.authors[:2]) if res.authors else "Unknown"
            table.add_row(str(idx), pdf_badge, res.title, str(res.year or "n.d."), authors_str)

        self.console.print(table)
        self.console.print("[dim]Use [bold cyan]/download <#>[/bold cyan] to download and ingest an open-access paper.[/dim]\n")

    async def _execute_search(self, query: str, limit: int) -> list[SearchResult]:
        async with PaperSearcher(workspace=self.workspace) as searcher:
            return await searcher.search(query, limit=limit)

    def cmd_download(self, arg: str) -> None:
        """Download and ingest a paper from the last search result by index."""
        if not arg or not arg.isdigit():
            self.console.print("[yellow]Usage: /download <# from search results>[/yellow]\n")
            return

        idx = int(arg) - 1
        if not self.last_search_results or idx < 0 or idx >= len(self.last_search_results):
            self.console.print("[red][x] Invalid search index. Run /search first.[/red]\n")
            return

        target_paper = self.last_search_results[idx]
        if not target_paper.has_pdf:
            self.console.print(f"[red][x] '{target_paper.title}' does not have an open-access PDF available.[/red]\n")
            return

        with self.console.status(f"[cyan]Downloading open-access PDF for '{target_paper.title}'...[/cyan]"):
            try:
                downloaded_pdf = asyncio.run(self._execute_download(target_paper))
                try:
                    parsed = self.parser.parse(downloaded_pdf, save_to_cache=True)
                except Exception:
                    meta = DocumentMetadata(
                        title=target_paper.title,
                        authors=target_paper.authors,
                        year=target_paper.year,
                        doi=target_paper.doi,
                        cite_key=target_paper.cite_key,
                    )
                    meta.ensure_cite_key()
                    clean_md = f"# {target_paper.title}\n\n{target_paper.abstract or ''}\n"
                    target_cache_name = f"{downloaded_pdf.stem}.md"
                    cache_path = self.workspace.save_to_cache(
                        target_cache_name,
                        f"{meta.to_frontmatter()}\n{clean_md}",
                    )
                    parsed = ParsedDocument(
                        file_path=downloaded_pdf,
                        cache_path=cache_path,
                        metadata=meta,
                        markdown=f"{meta.to_frontmatter()}\n{clean_md}",
                        body_markdown=clean_md,
                    )
                self.active_documents.add(parsed.file_path.stem)
            except Exception as exc:
                self.console.print(f"[bold red][x] Download or ingestion failed:[/bold red] {exc}\n")
                return

        self.console.print(f"[bold green][✓] Successfully downloaded and added to vault:[/bold green] {target_paper.title}")
        self.console.print(f"    Cite Key : [cyan]{target_paper.cite_key}[/cyan]")
        self.console.print(f"    Vault PDF: [dim]{downloaded_pdf}[/dim]\n")

    async def _execute_download(self, paper: SearchResult) -> Path:
        async with PaperSearcher(workspace=self.workspace) as searcher:
            return await searcher.download_paper(paper)

    def cmd_skills(self) -> None:
        """List all available skill markdown prompt files."""
        skills = self.skills_loader.list_skills()
        table = Table(
            title="Available Research Skills / Personas",
            title_style="bold cyan",
            border_style="cyan",
            header_style="bold bright_cyan",
        )
        table.add_column("Skill Name", style="bold green")
        table.add_column("Active", justify="center", width=8)
        table.add_column("Description", style="white")

        for s in skills:
            is_active = "[bold green][Active][/bold green]" if s.name == self.active_skill_name else ""
            table.add_row(s.name, is_active, s.description or s.title or "Standard skill prompt")

        self.console.print(table)
        self.console.print("[dim]Use [bold cyan]/skill <name>[/bold cyan] to switch persona.[/dim]\n")

    def cmd_skill(self, arg: str) -> None:
        """Switch current active persona skill."""
        if not arg:
            self.console.print(f"[dim]Current active skill:[/dim] [bold cyan]{self.active_skill_name}[/bold cyan]\n")
            return

        try:
            self.skills_loader.get_skill(arg)
            self.active_skill_name = arg
            self.console.print(f"[bold green][✓] Switched active skill to:[/bold green] [bold cyan]{arg}[/bold cyan]\n")
        except Exception:
            available = ", ".join(s.name for s in self.skills_loader.list_skills())
            self.console.print(f"[red][x] Skill '{arg}' not found. Available skills: {available}[/red]\n")

    def cmd_model(self, arg: str) -> None:
        """Switch current default LLM model."""
        if not arg:
            self.console.print(f"[dim]Current model:[/dim] [bold cyan]{self.config.default_model}[/bold cyan]\n")
            return

        self.config_manager.update(default_model=arg)
        self.config = self.config_manager.config
        self.console.print(f"[bold green][✓] Default model updated to:[/bold green] [bold bright_white]{arg}[/bold bright_white]\n")

    def cmd_style(self, arg: str) -> None:
        """Switch citation style between APA and IEEE."""
        style = arg.lower().strip()
        if style not in ("apa", "ieee"):
            self.console.print("[yellow]Usage: /style <apa | ieee>[/yellow]\n")
            return

        self.config.citation.style = style
        self.console.print(f"[bold green][✓] Citation formatting style set to:[/bold green] [bold cyan]{style.upper()}[/bold cyan]\n")

    def cmd_config(self, arg: str) -> None:
        """Display configuration or set API key."""
        if not arg or arg == "show":
            self.console.print(Panel(
                f"[bold cyan]Default Model [/bold cyan]: {self.config.default_model}\n"
                f"[bold cyan]Workspace Path[/bold cyan]: {self.config.workspace_path}\n"
                f"[bold cyan]Skills Dir    [/bold cyan]: {self.config.skills_dir}\n"
                f"[bold cyan]Citation Style[/bold cyan]: {self.config.citation.style}\n"
                f"[bold cyan]API Keys      [/bold cyan]: {', '.join(k for k, v in self.config.api_keys.items() if v) or 'None'}",
                title="scrpcy Configuration",
                border_style="cyan",
            ))
            self.console.print()
            return

        parts = shlex.split(arg)
        if len(parts) >= 3 and parts[0] == "set-key":
            provider, key = parts[1], parts[2]
            self.config_manager.config.set_api_key(provider, key)
            self.config_manager.save()
            self.config = self.config_manager.config
            self.console.print(f"[bold green][✓] Successfully saved API key for provider '{provider}'.[/bold green]\n")
        else:
            self.console.print("[yellow]Usage: /config show OR /config set-key <provider> <key>[/yellow]\n")

    def cmd_docx(self, arg: str) -> None:
        """Export current research synthesis session into a Word (.docx) manuscript."""
        filename = arg.strip() or f"research_draft_{int(time.time())}.docx"
        if not filename.endswith(".docx"):
            filename += ".docx"

        dest_path = self.workspace.vault_dir / filename

        with self.console.status("[cyan]Compiling manuscript and formatting citations via WordCraft...[/cyan]"):
            try:
                # Build sections from conversation history or active vault documents
                sections = []
                if self.conversation_history:
                    for idx, item in enumerate(self.conversation_history, 1):
                        heading = f"{idx}. {item['query'][:60]}"
                        sections.append({
                            "heading": heading,
                            "level": 1,
                            "content": item["response"],
                        })
                else:
                    sections.append({
                        "heading": "1. Literature Summary",
                        "level": 1,
                        "content": "This draft summarizes the active papers currently curated in the scrpcy research vault.",
                    })

                # Compile reference entries from active documents
                references = []
                for stem in self.active_documents:
                    content = self.workspace.read_from_cache(f"{stem}.md")
                    meta, _ = DocumentMetadata.from_frontmatter(content)
                    references.append(meta.model_dump())

                self.wordcraft.create_document(title=f"Academic Research Synthesis - {self.config.citation.style.upper()}")
                for sec in sections:
                    self.wordcraft.append_heading(sec["heading"], level=sec.get("level", 1))
                    self.wordcraft.add_paragraph_with_citation(sec["content"])

                if references:
                    self.wordcraft.append_reference_list(references, style=self.config.citation.style)

                self.wordcraft.save(dest_path)
            except Exception as exc:
                self.console.print(f"[bold red][x] Document export failed:[/bold red] {exc}\n")
                return

        self.console.print(f"[bold green][✓] Microsoft Word manuscript generated:[/bold green] [bold cyan]{dest_path}[/bold cyan]")
        self.console.print(f"    Style: {self.config.citation.style.upper()} Citation Standards\n")

    def cmd_web(self, arg: str) -> None:
        """Launch the web server in background and open the browser."""
        port = int(arg) if arg and arg.isdigit() else 8000
        host = "127.0.0.1"

        if self._web_server_thread and self._web_server_thread.is_alive():
            self.console.print(f"[yellow][!] Web Studio server is already running at http://{host}:{port}[/yellow]")
            webbrowser.open(f"http://{host}:{port}")
            self.console.print()
            return

        def _run_server() -> None:
            import uvicorn
            from scrpcy.server import create_app
            app = create_app(workspace=self.workspace, config=self.config)
            config = uvicorn.Config(app=app, host=host, port=port, log_level="warning")
            server = uvicorn.Server(config)
            server.run()

        self._web_server_thread = threading.Thread(target=_run_server, daemon=True)
        self._web_server_thread.start()

        time.sleep(1.0)
        webbrowser.open(f"http://{host}:{port}")
        self.console.print(f"[bold green][✓] Web Studio launched in background at:[/bold green] [bold cyan]http://{host}:{port}[/bold cyan]\n")

    def handle_chat(self, query: str) -> None:
        """Process research query with evidence synthesis and real-time streaming output."""
        active_paths = [self.workspace.get_cache_path(f"{s}.md") for s in self.active_documents]
        existing_paths = [p for p in active_paths if p.exists()]

        # Retrieve evidence chunks for preview
        evidence = self.engine.evidence_synthesizer.retrieve_evidence(
            query=query,
            documents=existing_paths,
            max_chunks=self.config.citation.max_evidence_chunks,
        )

        if evidence:
            sources_summary = ", ".join({c.source for c in evidence})
            self.console.print(f"[dim]● Sourced {len(evidence)} evidence chunks from: {sources_summary}[/dim]")
        elif existing_paths:
            self.console.print("[dim]● No high-relevance evidence chunks identified from active papers.[/dim]")
        else:
            self.console.print("[dim]● Notice: No papers are active in the vault. Operating in general synthesis mode.[/dim]")

        # Check API key resolution
        model_name = self.config.default_model
        api_key = self.config.get_api_key_for_model(model_name)

        if not api_key:
            self.console.print()
            self.console.print(Panel(
                f"[bold yellow][!] No API key configured for model '{model_name}'.[/bold yellow]\n\n"
                f"To enable live AI generation, run:\n"
                f"  [bold cyan]/config set-key gemini <YOUR_API_KEY>[/bold cyan]\n\n"
                f"[dim]Tips: Dapatkan kunci API Gemini resmi 100% gratis tanpa kartu kredit di:\n"
                f"https://aistudio.google.com/apikey[/dim]",
                title="API Key Required for Live LLM",
                border_style="yellow",
            ))
            self.console.print()
            return

        accumulated_tokens: list[str] = []
        self.console.print()

        try:
            async def _stream_chat() -> None:
                async for token in self.engine.stream_query(
                    query=query,
                    documents=existing_paths,
                    skill_name=self.active_skill_name,
                ):
                    accumulated_tokens.append(token)
                    # Print token directly to terminal for real-time streaming feeling
                    sys.stdout.write(token)
                    sys.stdout.flush()

            asyncio.run(_stream_chat())
            self.console.print("\n")

            full_response = "".join(accumulated_tokens)
            self.conversation_history.append({
                "query": query,
                "response": full_response,
            })

        except Exception as exc:
            self.console.print(f"\n[bold red][x] Generation error:[/bold red] {exc}\n")
