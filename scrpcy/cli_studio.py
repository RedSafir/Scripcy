"""Interactive CLI research studio for scrpcy with project workspace management."""

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
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from scrpcy.core.config import ConfigManager, ScrpcyConfig
from scrpcy.core.engine import LLMEngine
from scrpcy.core.mcp_client import WordCraftClient
from scrpcy.core.models import DocumentMetadata, ParsedDocument
from scrpcy.core.parser import DocumentParser
from scrpcy.core.project import ProjectInfo, ProjectManager
from scrpcy.core.scout import LiteratureScout, ScoutQuery, ScoutResult
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

SCRPCY_LOGO = r"""
   _____ _____ _____  _____   _______     __
  / ____/ ____|  __ \|  __ \ / ____\ \   / /
 | (___| |    | |__) | |__) | |     \ \_/ / 
  \___ \ |    |  _  /|  ___/| |      \   /  
  ____) | |___| | \ \| |    | |____   | |   
 |_____/ \_____|_|  \_\_|     \_____|  |_|   
"""
SCRPCY_VERSION = "v0.1.0"


class CLIStudio:
    """Terminal-native research studio with Dashboard & Master Project modes."""

    def __init__(
        self,
        workspace: WorkspaceManager | None = None,
        config: ScrpcyConfig | None = None,
        console: Console | None = None,
        project_manager: ProjectManager | None = None,
        parser: DocumentParser | None = None,
        wordcraft: WordCraftClient | None = None,
        engine: LLMEngine | None = None,
        initial_project: str | Path | None = None,
    ) -> None:
        self.config_manager = ConfigManager()
        self.config = config or self.config_manager.config
        self.console = console or Console(force_terminal=True)
        self.project_manager = project_manager or ProjectManager()

        self._injected_parser = parser
        self._injected_wordcraft = wordcraft
        self._injected_engine = engine

        self.skills_loader = SkillsLoader(skills_dir=self.config.skills_dir)
        self.active_skill_name = "academic_writer"
        self.active_documents: set[str] = set()
        self.last_search_results: list[SearchResult] = []
        self.last_scout_results: list[ScoutResult] = []
        self.conversation_history: list[dict[str, str]] = []
        self._web_server_thread: threading.Thread | None = None

        self.current_project: ProjectInfo | None = None
        self.workspace: WorkspaceManager | None = None
        self.parser: DocumentParser | None = None
        self.wordcraft: WordCraftClient | None = None
        self.engine: LLMEngine | None = None
        self.scout: LiteratureScout | None = None

        # If a workspace was explicitly provided, bind it
        if workspace:
            self._bind_workspace(workspace)

        # If an initial project was requested, connect to it
        if initial_project:
            try:
                self.open_project(initial_project)
            except Exception:
                pass

    def _bind_workspace(self, ws: WorkspaceManager) -> None:
        """Bind workspace and initialize dependent components."""
        self.workspace = ws
        self.workspace.ensure_directories()
        self.parser = self._injected_parser or DocumentParser(workspace=self.workspace)
        self.wordcraft = self._injected_wordcraft or WordCraftClient(workspace=self.workspace)
        self.engine = self._injected_engine or LLMEngine(
            config=self.config,
            skills_loader=self.skills_loader,
        )
        self.scout = LiteratureScout(workspace=self.workspace, parser=self.parser)
        self.active_documents.clear()
        self._sync_vault_cache()

    def _get_cached_stems(self) -> list[str]:
        """Return list of document stems in the active workspace cache."""
        if not self.workspace:
            return []
        return [p.stem for p in self.workspace.list_cached_documents()]

    def _sync_vault_cache(self) -> list[str]:
        """Index available markdown caches in the active workspace."""
        cached_files = self._get_cached_stems()
        for stem in cached_files:
            self.active_documents.add(stem)
        return cached_files

    def print_banner(self) -> None:
        """Render the large ASCII art logo with version and direct command prompt."""
        self.console.print(f"[bold cyan]{SCRPCY_LOGO}[/bold cyan]")
        self.console.print(f" [dim]{SCRPCY_VERSION} - Local-First Academic Research Studio[/dim]\n")

    def run(self) -> None:
        """Main interaction loop handling Dashboard and Project modes."""
        self.print_banner()

        if self.current_project:
            self.print_project_banner()
        else:
            self.print_dashboard_help()

        while True:
            try:
                if self.current_project:
                    prompt = f"[bold green]{self.current_project.name}[/bold green][bold cyan]>[/bold cyan] "
                else:
                    prompt = "[bold cyan]>[/bold cyan] "

                raw_input = self.console.input(prompt).strip()

                if not raw_input:
                    continue

                if raw_input in ("/exit", "/quit", "exit", "quit", "q"):
                    self.console.print("\n[bold cyan]Closing scrpcy research studio. Goodbye![/bold cyan]")
                    break

                if self.current_project is None:
                    self.handle_dashboard_command(raw_input)
                else:
                    self.handle_project_interaction(raw_input)

            except (KeyboardInterrupt, EOFError):
                self.console.print("\n\n[bold cyan]Session terminated. Goodbye![/bold cyan]")
                break
            except Exception as exc:
                self.console.print(f"[bold red][!] Error:[/bold red] {exc}\n")

    # =========================================================================
    # DASHBOARD MODE HANDLERS
    # =========================================================================

    def print_dashboard_help(self) -> None:
        """Render instructions for Dashboard mode."""
        self.console.print("[bold bright_white]Dashboard Commands:[/bold bright_white]")
        table = Table(box=None, padding=(0, 2), show_header=False)
        table.add_column("Command", style="bold green")
        table.add_column("Args", style="yellow")
        table.add_column("Description", style="white")

        table.add_row("projects", "", "List registered research projects (alias: list, ls)")
        table.add_row("new", "<name> [folder_path]", "Create and connect a master project folder")
        table.add_row("open", "<# | name | folder_path>", "Open an existing project folder")
        table.add_row("delete", "<# | name>", "Remove project from Scrpcy (alias: remove, rm)")
        table.add_row("skills", "", "List available AI persona skills (alias: skill)")
        table.add_row("status", "", "View LLM model, API key status, and system settings")
        table.add_row("config", "[set-key <prov> <key>]", "Manage API keys and default model")
        table.add_row("clear", "", "Clear terminal screen")
        table.add_row("exit", "", "Exit the studio (alias: quit, q)")

        self.console.print(table)
        self.console.print("\n[dim]Tip: Type 'new <name>' or 'open <path>' to begin working on a project.[/dim]\n")

    def handle_dashboard_command(self, raw_line: str) -> None:
        """Process commands while on the initial dashboard."""
        parts = shlex.split(raw_line) if raw_line else []
        if not parts:
            return

        cmd = parts[0].lower().lstrip("/")
        args = parts[1:]

        if cmd in ("help", "?"):
            self.print_dashboard_help()
        elif cmd in ("projects", "list", "ls"):
            self.dashboard_list_projects()
        elif cmd in ("new", "create"):
            self.dashboard_new_project(args)
        elif cmd in ("open", "load", "connect"):
            self.dashboard_open_project(args)
        elif cmd in ("delete", "remove", "rm"):
            self.dashboard_delete_project(args)
        elif cmd in ("skills", "skill"):
            self.render_skills_catalog(is_project_mode=False)
        elif cmd in ("status", "info"):
            self.dashboard_status()
        elif cmd == "config":
            self.cmd_config(" ".join(args))
        elif cmd == "clear":
            self.clear_screen()
        else:
            self.console.print(f"[yellow]Unknown dashboard command: '{raw_line}'.[/yellow]")
            self.console.print("[dim]Type 'help' to see available dashboard commands, or 'new <name>' to create a project.[/dim]\n")

    def dashboard_status(self) -> None:
        """Display system status: model, API keys, citation style, and project count."""
        model_name = self.config.default_model
        api_key = self.config.get_api_key_for_model(model_name)
        key_status = "[bold green][Configured][/bold green]" if api_key else "[bold yellow][Missing API Key][/bold yellow]"

        configured_keys = [k for k, v in self.config.api_keys.items() if v]
        all_keys_str = ", ".join(configured_keys) if configured_keys else "[dim]None[/dim]"

        projects = self.project_manager.list_projects()

        table = Table(
            title="scrpcy System & Model Status",
            title_style="bold cyan",
            border_style="cyan",
            header_style="bold bright_cyan",
        )
        table.add_column("Property", style="bold white", width=22)
        table.add_column("Current Setting / Value", style="white")

        table.add_row("Default Model", f"{model_name} {key_status}")
        table.add_row("Configured Providers", all_keys_str)
        table.add_row("Citation Style", self.config.citation.style.upper())
        table.add_row("Known Projects", f"{len(projects)} registered")
        table.add_row("Active Project", self.current_project.name if self.current_project else "[dim]None (Dashboard)[/dim]")
        table.add_row("Network Search", "[green]Semantic Scholar + ArXiv (Online)[/green]")
        table.add_row("Document Generator", "[green]WordCraft / python-docx (Ready)[/green]")

        self.console.print(table)
        self.console.print()

    def dashboard_list_projects(self) -> None:
        """Display list of registered research projects."""
        projects = self.project_manager.list_projects()
        if not projects:
            self.console.print("[dim]No projects registered yet.[/dim]")
            self.console.print("[dim]Create your first project using: [bold green]new <project_name> [folder_path][/bold green][/dim]\n")
            return

        table = Table(
            title="Registered Research Projects",
            title_style="bold cyan",
            border_style="cyan",
            header_style="bold bright_cyan",
        )
        table.add_column("#", justify="right", style="dim", width=4)
        table.add_column("Project Name", style="bold green", width=20)
        table.add_column("Master Directory Path", style="white")
        table.add_column("Last Accessed", style="dim", width=20)

        for idx, p in enumerate(projects, 1):
            last_date = p.last_opened[:19].replace("T", " ") if p.last_opened else "Unknown"
            table.add_row(str(idx), p.name, str(p.path), last_date)

        self.console.print(table)
        self.console.print("[dim]Use [bold cyan]open <# | name | path>[/bold cyan] to connect to a project.[/dim]\n")

    def dashboard_new_project(self, args: list[str]) -> None:
        """Create a new master project folder and activate it."""
        name = args[0].strip() if args else ""
        custom_path: str | None = args[1].strip() if len(args) > 1 else None

        # Step 1: Interactive prompt for project name if not provided
        if not name:
            self.console.print()
            self.console.print(Panel(
                "[bold bright_white]Create New Master Research Project[/bold bright_white]\n\n"
                "Project master folder will organize your research papers, cache, synthesis notes, and drafts.",
                title="[bold cyan]New Project Wizard[/bold cyan]",
                border_style="cyan",
            ))
            try:
                name = self.console.input("[bold cyan]project name[/bold cyan] [bold green]❯[/bold green] ").strip()
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[yellow]Project creation canceled.[/yellow]\n")
                return

            if not name or name.lower() in ("q", "quit", "cancel"):
                self.console.print("[yellow]Project creation canceled.[/yellow]\n")
                return

        # Step 2: Interactive location selection if not explicitly provided as 2nd arg
        if custom_path is None:
            clean_name = re.sub(r"[^\w\-_]", "_", name.strip().lower())
            curr_dir = Path.cwd() / clean_name

            self.console.print()
            self.console.print("[bold bright_white]Pilih lokasi penyimpanan master project:[/bold bright_white]")
            self.console.print(f"  [bold cyan][1][/bold cyan] Folder saat ini      : [dim]{curr_dir}[/dim]")
            self.console.print(f"  [bold cyan][2][/bold cyan] Folder lain / kustom : [dim]Tentukan path sendiri (misal: D:\\Riset\\{clean_name})[/dim]")
            self.console.print()

            try:
                choice = self.console.input("[bold cyan]pilih lokasi [1/2, default: 1][/bold cyan] [bold green]❯[/bold green] ").strip()
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[yellow]Project creation canceled.[/yellow]\n")
                return

            if choice.lower() in ("q", "quit", "cancel"):
                self.console.print("[yellow]Project creation canceled.[/yellow]\n")
                return

            if choice == "2":
                try:
                    custom_path = self.console.input("[bold cyan]path folder tujuan[/bold cyan] [bold green]❯[/bold green] ").strip()
                except (KeyboardInterrupt, EOFError):
                    self.console.print("\n[yellow]Project creation canceled.[/yellow]\n")
                    return
                if not custom_path or custom_path.lower() in ("q", "quit", "cancel"):
                    self.console.print("[yellow]Project creation canceled.[/yellow]\n")
                    return
            else:
                # Default option: current directory
                custom_path = str(curr_dir)

        with self.console.status(f"[cyan]Initializing master project directory for '{name}'...[/cyan]"):
            try:
                proj = self.project_manager.create_project(name=name, path=custom_path)
                self.open_project(proj)
            except Exception as exc:
                self.console.print(f"[bold red][x] Failed to create project:[/bold red] {exc}\n")
                return

        self.console.print(f"[bold green][✓] Master Project '{proj.name}' initialized and connected![/bold green]")
        self.console.print(f"    Master Folder: [bold cyan]{proj.path}[/bold cyan]")
        self.console.print(f"    Vault : [dim]{proj.resolved_path / 'vault'}[/dim]")
        self.console.print(f"    Drafts: [dim]{proj.resolved_path / 'drafts'}[/dim]\n")
        self.console.print("[dim]Type /help to see project commands, or ask research questions directly.[/dim]\n")

    def dashboard_open_project(self, args: list[str]) -> None:
        """Connect to an existing master project by name, index, or folder path."""
        if not args:
            self.console.print("[yellow]Usage: open <# | project_name | folder_path>[/yellow]\n")
            return

        target = " ".join(args).strip()
        with self.console.status(f"[cyan]Connecting to project '{target}'...[/cyan]"):
            try:
                proj = self.project_manager.open_project(target)
                self.open_project(proj)
            except Exception as exc:
                self.console.print(f"[bold red][x] Could not open project:[/bold red] {exc}\n")
                return

        cached_count = len(self._get_cached_stems())
        self.console.print(f"[bold green][✓] Connected to Master Project:[/bold green] [bold cyan]{proj.name}[/bold cyan]")
        self.console.print(f"    Location: [dim]{proj.path}[/dim]")
        self.console.print(f"    Papers in Vault: [white]{cached_count} indexed[/white]\n")

    def dashboard_delete_project(self, args: list[str]) -> None:
        """Remove a project from the Scrpcy registry without deleting physical files."""
        projects = self.project_manager.list_projects()
        if not projects:
            self.console.print("[dim]No registered projects found to delete.[/dim]\n")
            return

        target_identifier = args[0].strip() if args else ""
        if not target_identifier:
            self.dashboard_list_projects()
            try:
                target_identifier = self.console.input("[bold cyan]delete project (# or name)[/bold cyan] [bold green]❯[/bold green] ").strip()
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[yellow]Deletion canceled.[/yellow]\n")
                return

        if not target_identifier or target_identifier.lower() in ("q", "quit", "cancel"):
            self.console.print("[yellow]Deletion canceled.[/yellow]\n")
            return

        project_info = self.project_manager.get_project(target_identifier)
        if not project_info:
            self.console.print(f"[bold red][x] Project '{target_identifier}' not found in registry.[/bold red]\n")
            return

        try:
            confirm = self.console.input(
                f"Are you sure you want to remove project '[bold cyan]{project_info.name}[/bold cyan]' from Scrpcy? (y/N): "
            ).strip().lower()
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[yellow]Deletion canceled.[/yellow]\n")
            return

        if confirm not in ("y", "yes"):
            self.console.print("[yellow]Deletion canceled.[/yellow]\n")
            return

        if self.current_project and self.current_project.name == project_info.name:
            self.current_project = None
            self.workspace = None
            self.engine = None
            self.scout = None
            self.active_documents.clear()

        removed = self.project_manager.remove_project(project_info.name)
        if removed:
            self.console.print(f"[bold green][✓] Project '{project_info.name}' removed from Scrpcy.[/bold green]")
            self.console.print(f"[dim]Note: Your master files at '{project_info.path}' remain safe and untouched on disk.[/dim]\n")
        else:
            self.console.print(f"[bold red][x] Failed to remove project '{project_info.name}'.[/bold red]\n")

    def open_project(self, target: ProjectInfo | str | Path, show_banner: bool = True) -> None:
        """Switch studio context to a designated master project."""
        if isinstance(target, ProjectInfo):
            proj_info = target
        else:
            proj_info = self.project_manager.open_project(target)

        self.current_project = proj_info
        ws = WorkspaceManager(root_dir=proj_info.resolved_path)
        self._bind_workspace(ws)
        if show_banner:
            self.print_project_banner()

    def print_project_banner(self) -> None:
        """Render distinct visual status panel confirming active Project Workspace mode."""
        if not self.current_project or not self.workspace:
            return

        cached_count = len(self._get_cached_stems())
        active_count = len(self.active_documents)
        model_name = self.config.default_model

        panel_content = (
            f"[bold bright_white]MODE       :[/bold bright_white] [bold green]● PROJECT WORKSPACE (Active)[/bold green]\n"
            f"[bold bright_white]PROJECT    :[/bold bright_white] [bold cyan]{self.current_project.name}[/bold cyan]\n"
            f"[bold bright_white]MASTER PATH:[/bold bright_white] {self.current_project.path}\n"
            f"[bold bright_white]VAULT      :[/bold bright_white] {cached_count} indexed paper(s) ({active_count} active in citation context)\n"
            f"[bold bright_white]LLM MODEL  :[/bold bright_white] {model_name}\n\n"
            f"[bold bright_yellow]Quick Commands:[/bold bright_yellow]\n"
            f"  • Ketik pertanyaan riset Anda langsung di prompt untuk sintesis AI\n"
            f"  • [bold cyan]/scout[/bold cyan]     -> Cari paper baru via OpenAlex/S2 & unduh ke vault\n"
            f"  • [bold cyan]/vault[/bold cyan]     -> Lihat koleksi paper di vault & toggle status aktif (/active)\n"
            f"  • [bold cyan]/help[/bold cyan]      -> Tampilkan semua perintah project\n"
            f"  • [bold yellow]/back[/bold yellow]      -> Kembali ke Dashboard (tutup sesi project)"
        )

        self.console.print()
        self.console.print(Panel(
            panel_content,
            title=f"[bold green] PROJECT: {self.current_project.name} [/bold green]",
            border_style="green",
        ))
        self.console.print()

    # =========================================================================
    # PROJECT MODE HANDLERS
    # =========================================================================

    def handle_project_interaction(self, raw_input: str) -> None:
        """Route input when inside an active project."""
        if raw_input.startswith("/"):
            self.handle_project_command(raw_input)
        elif raw_input in ("help", "?"):
            self.print_project_help()
        elif raw_input == "clear":
            self.clear_screen()
        else:
            self.handle_chat(raw_input)

    def print_project_help(self) -> None:
        """Render a clean table of project commands."""
        table = Table(
            title=f"Project Commands [{self.current_project.name if self.current_project else 'Active'}]",
            title_style="bold cyan",
            border_style="cyan",
            header_style="bold bright_cyan",
        )
        table.add_column("Command", style="bold green", no_wrap=True)
        table.add_column("Arguments", style="yellow")
        table.add_column("Description", style="white")

        commands = [
            ("/scout", "", "Natural-language AI literature scout (OpenAlex + Semantic Scholar)"),
            ("/vault", "", "List papers in this project's vault (alias: /docs, /ls)"),
            ("/active", "<#|stem|all|clear>", "Toggle papers in/out of citation context"),
            ("/parse", "<path.pdf>", "Ingest a local PDF into this project's vault"),
            ("/view", "<#|stem>", "Inspect paper markdown and metadata in terminal"),
            ("/search", "<keywords>", "Search literature across Semantic Scholar & ArXiv"),
            ("/download", "<#>", "Download paper into this project's master vault"),
            ("/docx", "[filename]", "Compile current draft into Word (.docx) in drafts/"),
            ("/skills", "", "List available research skill personas (alias: /skill)"),
            ("/skill", "[# | name]", "Switch active research persona by number or name (e.g. /skill 2)"),
            ("/model", "<model_name>", "Switch default LLM model"),
            ("/style", "<apa|ieee>", "Change citation formatting style"),
            ("/dashboard", "", "Return to Dashboard (alias: /close, /back)"),
            ("/delete", "[# | name]", "Remove project from Scrpcy (alias: /remove, /rm)"),
            ("/status", "", "View system and project status"),
            ("/web", "[port]", "Launch Web Studio for this project in browser"),
            ("/clear", "", "Clear terminal screen"),
            ("/exit", "", "Exit scrpcy"),
        ]

        for cmd, args, desc in commands:
            table.add_row(cmd, args, desc)

        self.console.print(table)
        self.console.print("\n[dim]Note: Type any text directly to ask questions to the AI using active papers.[/dim]\n")

    def handle_project_command(self, cmd_line: str) -> None:
        """Route slash commands in Project mode."""
        parts = cmd_line.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/help", "/?"):
            self.print_project_help()
        elif cmd in ("/scout", "/paper-scout"):
            self.cmd_scout()
        elif cmd in ("/dashboard", "/close", "/back"):
            self.cmd_return_dashboard()
        elif cmd in ("/delete", "/remove", "/rm"):
            self.cmd_delete_project(args)
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
        elif cmd in ("/skills", "/skill"):
            self.cmd_skill(args)
        elif cmd == "/model":
            self.cmd_model(args)
        elif cmd == "/style":
            self.cmd_style(args)
        elif cmd == "/config":
            self.cmd_config(args)
        elif cmd in ("/docx", "/export"):
            self.cmd_docx(args)
        elif cmd in ("/status", "/info"):
            self.cmd_project_status()
        elif cmd == "/clear":
            self.clear_screen()
        elif cmd == "/web":
            self.cmd_web(args)
        else:
            self.console.print(f"[bold red]Unknown project command:[/bold red] {cmd}. Type [bold cyan]/help[/bold cyan] for list of commands.\n")

    def cmd_scout(self) -> None:
        """Interactive natural-language literature scout workflow with OpenAlex."""
        if not self.workspace or not self.scout:
            self.console.print("[red][x] Please open a project first to use /scout.[/red]\n")
            return

        self.console.print()
        self.console.print(Panel(
            "[bold bright_white]Intelligent Literature Scout Pipeline[/bold bright_white]\n\n"
            "Tuliskan kebutuhan riset / jurnal Anda secara bebas (bahasa alami).\n"
            "Sistem akan merumuskan parameter pencarian, menarik paper nyata via [bold cyan]OpenAlex (pyalex)[/bold cyan] "
            "dan Semantic Scholar, lalu mengkurasi paper yang paling relevan untuk project Anda.\n\n"
            "[dim]Contoh: 'Saya butuh paper tentang kesehatan mental remaja tahun 2018-2020 yang open access'[/dim]",
            title="[bold cyan]Literature Scout[/bold cyan]",
            border_style="cyan",
        ))

        try:
            user_input = self.console.input("[bold cyan]scout[/bold cyan] [bold green]❯[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[yellow]Scout canceled.[/yellow]\n")
            return

        if not user_input or user_input.lower() in ("q", "quit", "cancel", "exit"):
            self.console.print("[yellow]Scout canceled.[/yellow]\n")
            return

        # Phase 1: Parameter Formulation
        with self.console.status("[cyan]Formulating academic query strategy from your description...[/cyan]"):
            query_spec = self.scout.extract_search_criteria(user_input, engine=self.engine)

        self.console.print()
        self.console.print(Panel(
            f"[bold cyan]Keywords   [/bold cyan]: {query_spec.keywords}\n"
            f"[bold cyan]Year Filter[/bold cyan]: {f'{query_spec.year_start} - {query_spec.year_end}' if query_spec.year_start else 'All Years'}\n"
            f"[bold cyan]Open Access[/bold cyan]: {'Yes (OA Only)' if query_spec.is_oa_only else 'Any'}\n"
            f"[bold cyan]Summary    [/bold cyan]: {query_spec.summary}",
            title="[green]Extracted Query Strategy[/green]",
            border_style="green",
        ))
        self.console.print()

        # Phase 2: Multi-Source Scraping
        with self.console.status("[cyan]Scraping verified scholarly works across OpenAlex & Semantic Scholar...[/cyan]"):
            try:
                candidates = asyncio.run(self.scout.search_multi_source(query_spec, limit=15))
            except Exception as exc:
                self.console.print(f"[bold red][x] Search scraping failed:[/bold red] {exc}\n")
                return

        if not candidates:
            self.console.print("[yellow]No scholarly candidates found matching the criteria.[/yellow]\n")
            return

        # Phase 3: AI Curation & Ranking
        with self.console.status(f"[cyan]Evaluating and ranking {len(candidates)} candidates against your research intent...[/cyan]"):
            try:
                curated = self.scout.curate_and_rank(user_input, candidates, engine=self.engine, max_output=8)
            except Exception:
                curated = candidates[:8]

        self.last_scout_results = curated

        # Phase 4: Render Table
        table = Table(
            title=f"Curated Literature Results ({len(curated)} papers)",
            title_style="bold cyan",
            border_style="cyan",
            header_style="bold bright_cyan",
        )
        table.add_column("#", justify="right", style="dim", width=3)
        table.add_column("Badge", justify="center", width=14)
        table.add_column("Title", style="bold white", width=34, overflow="fold")
        table.add_column("Authors", style="dim", width=18, overflow="fold")
        table.add_column("Year", justify="center", width=6)
        table.add_column("PDF Status", justify="center", width=13)
        table.add_column("Relevance / Summary", style="italic white", overflow="fold")

        for idx, item in enumerate(curated, 1):
            if item.source == "openalex":
                source_badge = "[bold cyan][OpenAlex][/bold cyan]"
            elif item.source == "ai_recommended":
                source_badge = "[bold magenta][AI Seminal][/bold magenta]"
            else:
                source_badge = "[bold blue][S2/ArXiv][/bold blue]"

            pdf_status = "[bold green][Open Access][/bold green]" if item.has_pdf else "[dim][Paywalled][/dim]"
            authors_str = ", ".join(item.authors[:2]) if item.authors else "Unknown"

            table.add_row(
                str(idx),
                source_badge,
                item.title,
                authors_str,
                str(item.year or "n.d."),
                pdf_status,
                item.relevance_reason or "Matched search parameters.",
            )

        self.console.print(table)
        self.console.print()

        # Phase 5: Interactive Ingestion
        self.console.print("[bold yellow]Ingest to Vault:[/bold yellow] [dim]Ketik nomor paper yang ingin diunduh ke vault project (contoh: '1, 2' atau '1' atau 'all', atau 'q' untuk selesai):[/dim]")
        try:
            dl_choice = self.console.input("[bold cyan]download[/bold cyan] [bold green]❯[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n")
            return

        if not dl_choice or dl_choice.lower() in ("q", "quit", "cancel", "n", "no"):
            self.console.print("[dim]No papers ingested.[/dim]\n")
            return

        to_download_indices = []
        if dl_choice.lower() == "all":
            to_download_indices = list(range(len(curated)))
        else:
            for part in dl_choice.replace(",", " ").split():
                if part.isdigit():
                    num = int(part) - 1
                    if 0 <= num < len(curated):
                        to_download_indices.append(num)

        if not to_download_indices:
            self.console.print("[dim]No valid paper numbers selected.[/dim]\n")
            return

        for num in to_download_indices:
            paper_to_dl = curated[num]
            if not paper_to_dl.has_pdf:
                self.console.print(f"[yellow][!] Paper #{num + 1} '{paper_to_dl.title[:45]}...' is paywalled (no open-access PDF).[/yellow]")
                continue

            with self.console.status(f"[cyan]Downloading '{paper_to_dl.title[:40]}...' into project vault...[/cyan]"):
                try:
                    dl_path = asyncio.run(self.scout.download_scout_paper(paper_to_dl))
                    self.active_documents.add(dl_path.stem)
                    self.console.print(f"[bold green][✓] Added to project vault & active rujukan:[/bold green] {paper_to_dl.title}")
                except Exception as exc:
                    self.console.print(f"[bold red][x] Failed to download #{num + 1}:[/bold red] {exc}")

        self.console.print()

    def cmd_return_dashboard(self) -> None:
        """Disconnect from active project and return to dashboard."""
        proj_name = self.current_project.name if self.current_project else "project"
        self.current_project = None
        self.workspace = None
        self.engine = None
        self.scout = None
        self.active_documents.clear()
        self.console.print(f"[yellow][←] Closed project '{proj_name}'. Returned to [bold cyan]Dashboard Mode[/bold cyan].[/yellow]\n")
        self.print_dashboard_help()

    def cmd_delete_project(self, arg: str = "") -> None:
        """Remove project from Scrpcy registry, closing it if currently active."""
        target_name = arg.strip() if arg else (self.current_project.name if self.current_project else "")
        if not target_name:
            self.console.print("[red][x] No active project or target name specified.[/red]\n")
            return

        project_info = self.project_manager.get_project(target_name)
        if not project_info:
            self.console.print(f"[bold red][x] Project '{target_name}' not found in registry.[/bold red]\n")
            return

        is_current = bool(self.current_project and self.current_project.name == project_info.name)
        prompt_label = f"active project '[bold cyan]{project_info.name}[/bold cyan]'" if is_current else f"project '[bold cyan]{project_info.name}[/bold cyan]'"

        try:
            confirm = self.console.input(
                f"Are you sure you want to remove {prompt_label} from Scrpcy? (y/N): "
            ).strip().lower()
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[yellow]Deletion canceled.[/yellow]\n")
            return

        if confirm not in ("y", "yes"):
            self.console.print("[yellow]Deletion canceled.[/yellow]\n")
            return

        if is_current:
            self.current_project = None
            self.workspace = None
            self.engine = None
            self.scout = None
            self.active_documents.clear()

        removed = self.project_manager.remove_project(project_info.name)
        if removed:
            self.console.print(f"[bold green][✓] Project '{project_info.name}' removed from Scrpcy.[/bold green]")
            self.console.print(f"[dim]Note: Your master files at '{project_info.path}' remain safe and untouched on disk.[/dim]\n")
            if is_current:
                self.console.print("[yellow][←] Returned to Dashboard.[/yellow]\n")
                self.print_dashboard_help()
        else:
            self.console.print(f"[bold red][x] Failed to remove project '{project_info.name}'.[/bold red]\n")

    def cmd_project_status(self) -> None:
        """Display status for the current active master project."""
        if not self.current_project or not self.workspace:
            self.dashboard_status()
            return

        cached_docs = self._get_cached_stems()
        active_count = len(self.active_documents)
        model_name = self.config.default_model

        table = Table(
            title=f"Master Project Status: {self.current_project.name}",
            title_style="bold cyan",
            border_style="cyan",
            header_style="bold bright_cyan",
        )
        table.add_column("Property", style="bold white", width=22)
        table.add_column("Details", style="white")

        table.add_row("Project Name", self.current_project.name)
        table.add_row("Master Path", str(self.workspace.root_dir))
        table.add_row("Vault Papers", f"{len(cached_docs)} indexed ({active_count} active rujukan)")
        table.add_row("Engine Model", model_name)
        table.add_row("Active Skill", self.active_skill_name)
        table.add_row("Citation Style", self.config.citation.style.upper())
        table.add_row("Drafts Folder", str(self.workspace.root_dir / "drafts"))

        self.console.print(table)
        self.console.print()

    def clear_screen(self) -> None:
        """Clear the terminal screen and reprint logo."""
        os.system("cls" if os.name == "nt" else "clear")
        self.print_banner()

    def cmd_vault(self) -> None:
        """Display list of papers in the local vault with active selection state."""
        if not self.workspace:
            self.console.print("[dim]No active workspace.[/dim]\n")
            return

        stems = self._get_cached_stems()
        if not stems:
            self.console.print("[dim]Project vault is currently empty. Use [bold cyan]/search[/bold cyan] or [bold cyan]/parse <file.pdf>[/bold cyan] to add papers.[/dim]\n")
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

        if not self.workspace or not self.parser:
            self.console.print("[red][x] No active project workspace to ingest document into.[/red]\n")
            return

        pdf_path = Path(arg).expanduser().resolve()
        if not pdf_path.exists() or not pdf_path.is_file():
            self.console.print(f"[red][x] PDF file not found at: '{pdf_path}'[/red]\n")
            return

        with self.console.status(f"[cyan]Parsing PDF and extracting metadata: {pdf_path.name}...[/cyan]"):
            try:
                parsed = self.parser.parse(pdf_path, save_to_cache=True)
            except Exception:
                meta = DocumentMetadata(
                    title=pdf_path.stem.replace("_", " ").title(),
                    authors=["Unknown Author"],
                    cite_key=pdf_path.stem.lower(),
                )
                clean_md = f"# {pdf_path.stem}\n\n[Ingested content]\n"
                target_cache_name = f"{pdf_path.stem}.md"
                cache_path = self.workspace.save_to_cache(
                    target_cache_name,
                    f"{meta.to_frontmatter()}\n{clean_md}",
                )
                parsed = ParsedDocument(
                    file_path=pdf_path,
                    cache_path=cache_path,
                    metadata=meta,
                    markdown=f"{meta.to_frontmatter()}\n{clean_md}",
                    body_markdown=clean_md,
                )

            self.active_documents.add(parsed.file_path.stem)

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

        if not self.workspace:
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

        if not self.workspace or not self.parser:
            self.console.print("[red][x] No active project to download paper into.[/red]\n")
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

    def render_skills_catalog(self, is_project_mode: bool = True) -> None:
        """Render a formatted table catalog of available research persona skills."""
        skills = self.skills_loader.list_skills()
        title = "Available Research Personas / Skills" if is_project_mode else "Installed AI Research Personas / Skills"
        table = Table(
            title=title,
            title_style="bold cyan",
            border_style="cyan",
            header_style="bold bright_cyan",
        )
        table.add_column("#", justify="right", style="dim", width=3)
        table.add_column("Persona / Skill", style="bold green", width=24)
        table.add_column("Status", justify="center", width=12)
        table.add_column("Description", style="white")

        for idx, s in enumerate(skills, start=1):
            if is_project_mode:
                is_active = (s.name == self.active_skill_name)
                status = "[bold green][Active][/bold green]" if is_active else "[dim][Standby][/dim]"
            else:
                status = "[bold cyan][Installed][/bold cyan]"

            table.add_row(
                str(idx),
                s.name,
                status,
                s.description or "Standard research persona",
            )

        self.console.print(table)
        if is_project_mode:
            self.console.print("[dim]Switch persona: [bold cyan]/skill <# | name>[/bold cyan] (e.g. [bold cyan]/skill 2[/bold cyan] or [bold cyan]/skill methodology_critic[/bold cyan])[/dim]")
        else:
            self.console.print("[dim]Activate persona inside a project: open or create a project, then run [bold cyan]/skill <# | name>[/bold cyan][/dim]")
        self.console.print(f"[dim]Add custom persona: Create a markdown file (.md) in [bold white]{self.skills_loader.skills_dir}[/bold white][/dim]\n")

    def cmd_skills(self) -> None:
        """List all available research persona skills."""
        self.render_skills_catalog(is_project_mode=True)

    def cmd_skill(self, arg: str) -> None:
        """Switch current active persona skill, or view catalog if no argument given."""
        arg = arg.strip()
        if not arg:
            self.render_skills_catalog(is_project_mode=True)
            return

        skills = self.skills_loader.list_skills()
        target_skill_name = None

        if arg.isdigit():
            idx = int(arg) - 1
            if 0 <= idx < len(skills):
                target_skill_name = skills[idx].name
            else:
                self.console.print(f"[bold red][x] Invalid skill number: {arg}.[/bold red] Choose a number between 1 and {len(skills)}.\n")
                return
        else:
            norm_arg = arg.lower().replace(" ", "_").replace("-", "_")
            for s in skills:
                if s.name == arg or s.name.lower() == norm_arg:
                    target_skill_name = s.name
                    break

        if target_skill_name:
            try:
                self.skills_loader.get_skill(target_skill_name)
                self.active_skill_name = target_skill_name
                self.console.print(f"[bold green][✓] Switched active skill to:[/bold green] [bold cyan]{target_skill_name}[/bold cyan]\n")
            except Exception as exc:
                self.console.print(f"[bold red][x] Error loading skill '{target_skill_name}':[/bold red] {exc}\n")
        else:
            available = ", ".join(f"{i}. {s.name}" for i, s in enumerate(skills, start=1))
            self.console.print(f"[bold red][x] Skill '{arg}' not found.[/bold red] Available skills: {available}\n")

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
        elif len(parts) >= 2 and parts[0] == "set-model":
            model = parts[1]
            self.config_manager.update(default_model=model)
            self.config = self.config_manager.config
            self.console.print(f"[bold green][✓] Updated default model to '{model}'.[/bold green]\n")
        else:
            self.console.print("[yellow]Usage: config show OR config set-key <provider> <key>[/yellow]\n")

    def cmd_docx(self, arg: str) -> None:
        """Export current research synthesis session into a Word (.docx) manuscript."""
        if not self.workspace or not self.wordcraft:
            self.console.print("[red][x] No active project to export manuscript from.[/red]\n")
            return

        filename = arg.strip() or f"research_draft_{int(time.time())}.docx"
        if not filename.endswith(".docx"):
            filename += ".docx"

        drafts_dir = self.workspace.root_dir / "drafts"
        drafts_dir.mkdir(parents=True, exist_ok=True)
        dest_path = drafts_dir / filename

        with self.console.status("[cyan]Compiling manuscript and formatting citations via WordCraft...[/cyan]"):
            try:
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

                references = []
                for stem in self.active_documents:
                    content = self.workspace.read_from_cache(f"{stem}.md")
                    meta, _ = DocumentMetadata.from_frontmatter(content)
                    references.append(meta.model_dump())

                self.wordcraft.create_document(
                    title=f"Academic Research Synthesis - {self.config.citation.style.upper()}",
                    path=dest_path,
                )
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
        if not self.workspace:
            self.console.print("[yellow][!] Please open a project first or launch web mode with 'scrpcy start --web'.[/yellow]\n")
            return

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
        if not self.workspace or not self.engine:
            self.console.print("[red][x] No active project workspace loaded.[/red]\n")
            return

        active_paths = [self.workspace.get_cache_path(f"{s}.md") for s in self.active_documents]
        existing_paths = [p for p in active_paths if p.exists()]

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
