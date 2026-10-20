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
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from scrpcy.core.chats import ChatManager, ChatSession
from scrpcy.core.config import ConfigManager, ScrpcyConfig, detect_9router_key, fetch_9router_models
from scrpcy.core.engine import LLMEngine, estimate_tokens
from scrpcy.core.gdrive import GoogleDriveManager, extract_drive_file_id
from scrpcy.core.mcp_client import WordCraftClient
from scrpcy.core.models import DocumentMetadata, ParsedDocument
from scrpcy.core.parser import DocumentParser
from scrpcy.core.project import ProjectInfo, ProjectManager
from scrpcy.core.scout import LiteratureScout, ScoutQuery, ScoutResult
from scrpcy.core.search import PaperSearcher, SearchResult
from scrpcy.core.skills import SkillsLoader
from scrpcy.core.tasks import TaskInfo, TaskManager
from scrpcy.core.tracker import TokenTracker
from scrpcy.core.vault import VaultItem, VaultManager
from scrpcy.core.workspace import WorkspaceManager

# Safe UTF-8 console output for Windows terminal and quiet litellm output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import litellm
    litellm.suppress_debug_info = True
    litellm.set_verbose = False
    litellm.drop_params = True
except Exception:
    pass

from scrpcy.ui.interactive import (
    _format_model_display_name,
    interactive_action_confirm,
    interactive_carousel_table,
    interactive_help_view,
    interactive_input_prompt,
    interactive_model_selector,
    interactive_skill_selector,
    interactive_token_warning_confirm,
)
from scrpcy.ui.formatters import render_academic_markdown

SCRPCY_LOGO = r"""
  ███████╗ ██████╗██████╗ ██████╗  ██████╗██╗   ██╗
  ██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝╚██╗ ██╔╝
  ███████╗██║     ██████╔╝██████╔╝██║      ╚████╔╝ 
  ╚════██║██║     ██╔══██╗██╔═══╝ ██║       ╚██╔╝  
  ███████║╚██████╗██║  ██║██║     ╚██████╗   ██║   
  ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝      ╚═════╝   ╚═╝   
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

        self.task_manager = TaskManager()
        self.gdrive_manager = GoogleDriveManager()
        self.vault_manager: VaultManager | None = None
        self.chat_manager: ChatManager | None = None
        self.sub_mode: str | None = None
        self.active_chat_session: ChatSession | None = None
        self._has_printed_banner: bool = False

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
        proj_name = self.current_project.name if self.current_project else None
        proj_path = str(self.workspace.root_dir) if self.workspace else None
        self.engine = self._injected_engine or LLMEngine(
            config=self.config,
            skills_loader=self.skills_loader,
            project_name=proj_name,
            project_path=proj_path,
        )
        self.engine.default_model = self.get_active_project_model()
        self.scout = LiteratureScout(workspace=self.workspace, parser=self.parser)
        self.vault_manager = VaultManager(workspace=self.workspace)
        self.chat_manager = ChatManager(self.workspace.root_dir / "chats.db")
        self._ensure_active_chat_session()
        self.sub_mode = None
        self.active_documents.clear()
        self._sync_vault_cache()

    def get_active_project_model(self) -> str:
        """Get the model configured for the current project, or fallback default."""
        if self.current_project:
            stored = self.project_manager.get_project_model(self.current_project.name)
            if stored:
                self.current_project.model = stored
                return stored
            if self.current_project.model:
                return self.current_project.model
        return self.config.default_model

    def get_active_project_effort(self) -> str:
        """Get the reasoning effort level configured for current project."""
        if self.current_project:
            return self.project_manager.get_project_effort(self.current_project.name)
        return "high"

    def _ensure_active_chat_session(self) -> ChatSession:
        """Ensure there is an active chat session bound, creating chat1 if none exists."""
        if not self.chat_manager:
            raise RuntimeError("No chat manager initialized.")
        if self.active_chat_session:
            return self.active_chat_session

        sessions = self.chat_manager.list_sessions()
        if sessions:
            self.active_chat_session = sessions[0]
        else:
            self.active_chat_session = self.chat_manager.create_session(
                name="chat1",
                active_skill=self.active_skill_name,
            )
        return self.active_chat_session

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
        """Render the large ASCII art logo at most once."""
        if not self._has_printed_banner:
            self.console.print(f"[bold cyan]{SCRPCY_LOGO.strip()}[/bold cyan]\n")
            self.console.print(f" [dim]{SCRPCY_VERSION} • Local-First Academic Research Studio[/dim]\n")
            self._has_printed_banner = True

    def print_dashboard_banner(self) -> None:
        """Render minimal borderless dashboard status without command suggestion blocks."""
        projects = self.project_manager.list_projects()
        p_count = len(projects)
        self.console.print(
            f" [bold cyan]scrpcy[/bold cyan] [dim]•[/dim] [dim]dashboard ({p_count} projects)[/dim]"
        )

    def render_mode_divider(self) -> None:
        """Render horizontal divider when switching between modes or resources."""
        width = self.console.width or 80
        self.console.print(Text("─" * width, style="grey23"))

    def run(self) -> None:
        """Main interaction loop handling Dashboard and Project modes."""
        self.print_banner()

        if self.current_project:
            self.print_project_banner()
        else:
            self.print_dashboard_banner()

        while True:
            try:
                eff = self.get_active_project_effort() if self.current_project else "high"
                mod = self.get_active_project_model() if self.current_project else self.config.default_model

                if self.sub_mode == "vault":
                    prompt_lbl = f"{self.current_project.name}/vault> " if self.current_project else "vault> "
                    resolved_mode = "vault"
                elif self.sub_mode == "processing":
                    prompt_lbl = f"{self.current_project.name}/processing> " if self.current_project else "processing> "
                    resolved_mode = "processing"
                elif self.sub_mode == "chat_hub":
                    prompt_lbl = f"{self.current_project.name}/chat> " if self.current_project else "chat> "
                    resolved_mode = "chat_hub"
                elif self.sub_mode == "chat":
                    session = self._ensure_active_chat_session()
                    sess_name = session.name if session else "chat1"
                    prompt_lbl = f"{self.current_project.name}/{sess_name}> " if self.current_project else f"{sess_name}> "
                    resolved_mode = "chat"
                elif self.current_project:
                    prompt_lbl = f"{self.current_project.name}> "
                    resolved_mode = "project"
                else:
                    prompt_lbl = "> "
                    resolved_mode = "dashboard"

                ctx_lbl = self.current_project.name if self.current_project else ""

                raw_input = interactive_input_prompt(
                    console=self.console,
                    prompt_label=prompt_lbl,
                    is_project_mode=(self.current_project is not None),
                    is_chat_mode=(self.sub_mode == "chat"),
                    mode=resolved_mode,
                    context_label=ctx_lbl,
                    current_model=mod,
                    current_effort=eff,
                ).strip()

                if not raw_input:
                    continue

                if self.sub_mode is None and raw_input in ("/exit", "/quit", "exit", "quit", "q"):
                    self.console.print("\n[bold cyan]Closing scrpcy research studio. Goodbye![/bold cyan]")
                    break

                if raw_input in ("?", "/help"):
                    interactive_help_view(
                        console=self.console,
                        current_model=mod,
                        current_effort=eff,
                        is_project_mode=(self.current_project is not None),
                        is_chat_mode=(self.sub_mode == "chat"),
                        mode=resolved_mode,
                        context_label=ctx_lbl,
                    )
                    continue

                if self.current_project is None:
                    self.handle_dashboard_command(raw_input)
                else:
                    self.handle_project_interaction(raw_input)

            except KeyboardInterrupt:
                if self.sub_mode == "chat":
                    self.render_mode_divider()
                    self.sub_mode = "chat_hub"
                    self.console.print(f"\n[dim]Returned to chat hub: {self.current_project.name}/chat>[/dim]\n")
                    continue
                elif self.sub_mode is not None:
                    self.render_mode_divider()
                    self.sub_mode = None
                    self.console.print(f"\n[dim]Returned to project root: {self.current_project.name if self.current_project else 'dashboard'}>[/dim]\n")
                    continue
                self.console.print("\n\n[bold cyan]Session terminated. Goodbye![/bold cyan]")
                break
            except EOFError:
                self.console.print("\n\n[bold cyan]Session terminated. Goodbye![/bold cyan]")
                break
            except Exception as exc:
                self.console.print(f"[bold red][!] Error:[/bold red] {escape(str(exc))}\n")

    # =========================================================================
    # DASHBOARD MODE HANDLERS
    # =========================================================================

    def print_dashboard_help(self) -> None:
        """Render simple available commands for Dashboard mode."""
        interactive_help_view(
            console=self.console,
            current_model=self.config.default_model,
            current_effort="high",
            mode="dashboard",
            context_label="Dashboard",
        )

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
            if args:
                self.dashboard_status(args)
            else:
                self.console.print(
                    "[dim]Perintah 'status' digunakan di dalam project untuk melihat model dan konsumsi token.\n"
                    "Buka project terlebih dahulu (contoh: [bold cyan]open <nama>[/bold cyan]).[/dim]\n"
                )
        elif cmd in ("models", "model", "config"):
            self.cmd_models(" ".join(args))
        elif cmd in ("settings", "setting"):
            self.cmd_settings(" ".join(args))
        elif cmd == "clear":
            self.clear_screen()
        else:
            self.console.print(f"[yellow]Unknown dashboard command: '{raw_line}'.[/yellow]")
            self.console.print("[dim]Type 'help' to see available dashboard commands, or 'new <name>' to create a project.[/dim]\n")

    def dashboard_status(self, args: list[str] | None = None) -> None:
        """Display per-project status and token consumption."""
        projects = self.project_manager.list_projects()
        if not projects:
            self.console.print("[yellow][!] No research projects registered yet.[/yellow]")
            self.console.print("[dim]Create a new project using: [bold cyan]new <project_name>[/bold cyan][/dim]\n")
            return

        target_name = args[0] if args else None
        target_project = None
        if target_name:
            for p in projects:
                if p.name.lower() == target_name.lower():
                    target_project = p
                    break
            if not target_project:
                self.console.print(f"[bold red][x] Project '{target_name}' not found.[/bold red]")
                self.console.print(f"[dim]Available projects: {', '.join(p.name for p in projects)}[/dim]\n")
                return
        else:
            target_project = self.project_manager.get_active_project() or projects[0]

        vault_dir = Path(target_project.path) / "vault"
        vault_count = len([f for f in vault_dir.iterdir() if f.is_file()]) if vault_dir.is_dir() else 0
        token_stats = TokenTracker.get_project_tokens(target_project.path)

        self.console.print(f"\n[bold bright_white]Research Project Status: {target_project.name}[/bold bright_white]")
        table = Table(box=None, padding=(0, 2), show_header=False)
        table.add_column("Property", style="dim", width=22)
        table.add_column("Details", style="white")

        table.add_row("Project Name", f"[bold bright_white]{target_project.name}[/bold bright_white]")
        table.add_row("Master Path", str(target_project.path))
        table.add_row("Vault Papers", f"{vault_count} documents in vault")
        proj_model = target_project.model or f"[dim]Not assigned (default: {self.config.default_model})[/dim]"
        table.add_row("Project Model", f"[bold cyan]{proj_model}[/bold cyan]")
        table.add_row("Citation Style", self.config.citation.style.upper())

        token_display = (
            f"Input: {token_stats.input_tokens:,} | Output: {token_stats.output_tokens:,} | "
            f"Total: {token_stats.total_tokens:,} tokens ({token_stats.request_count} reqs)"
        )
        table.add_row("Project Token Usage", token_display)
        if token_stats.last_active:
            table.add_row("Last Activity", token_stats.last_active.replace("T", " ")[:19] + " UTC")

        self.console.print(table)
        self.console.print(f"[dim]To open this project: [bold cyan]open {target_project.name}[/bold cyan][/dim]\n")

    def dashboard_list_projects(self) -> None:
        """Display list of registered research projects."""
        projects = self.project_manager.list_projects()
        if not projects:
            self.console.print("[dim]No projects registered yet.[/dim]")
            self.console.print("[dim]Create your first project using: [bold cyan]new <project_name> [folder_path][/bold cyan][/dim]\n")
            return

        self.console.print("\n[bold bright_white]Registered Research Projects[/bold bright_white]")
        table = Table(
            box=None,
            padding=(0, 2),
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("#", justify="right", style="dim", width=4)
        table.add_column("Project Name", style="bold white", width=20)
        table.add_column("Master Directory Path", style="dim")
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
            self.console.print("\n[bold bright_white]New Project Wizard — Create Master Research Project[/bold bright_white]")
            self.console.print("[dim]Project master folder will organize your research papers, cache, synthesis notes, and drafts.[/dim]\n")
            try:
                name = self.console.input("[bold cyan]project name > [/bold cyan]").strip()
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
                self.render_mode_divider()
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
        """Render distinct visual status confirming active Project Workspace mode without heavy boxing."""
        if not self.current_project or not self.workspace:
            return

        cached_count = len(self._get_cached_stems())
        active_count = len(self.active_documents)
        model_name = self.get_active_project_model()
        effort = self.get_active_project_effort()
        model_label = _format_model_display_name(model_name)

        self.console.print()
        self.console.print(
            f" [bold cyan]scrpcy[/bold cyan] [dim]•[/dim] [bold bright_white]{self.current_project.name}[/bold bright_white]  "
            f"[dim]•[/dim]  [bold cyan]{model_label}[/bold cyan] [dim]({effort})[/dim]  "
            f"[dim]•[/dim]  [dim]vault: {cached_count} indexed ({active_count} active)[/dim]"
        )

    # =========================================================================
    # PROJECT MODE HANDLERS
    # =========================================================================

    def handle_project_interaction(self, raw_input: str) -> None:
        """Route input when inside an active project across root and sub-modes."""
        clean = raw_input.strip()
        if not clean:
            return

        if self.sub_mode == "vault":
            self.handle_vault_interaction(clean)
        elif self.sub_mode == "processing":
            self.handle_processing_interaction(clean)
        elif self.sub_mode == "chat_hub":
            self.handle_chat_hub_interaction(clean)
        elif self.sub_mode == "chat":
            self.handle_chat_submode_interaction(clean)
        else:
            self.handle_project_root_interaction(clean)

    def handle_project_root_interaction(self, clean: str) -> None:
        """Process commands at project root prompt (<project_name>>). Commands do NOT require leading '/'."""
        cmd_line = clean if clean.startswith("/") else f"/{clean}"
        self.handle_project_command(cmd_line)

    def print_project_help(self) -> None:
        """Render clean available commands for Project mode."""
        proj_name = self.current_project.name if self.current_project else "Active"
        interactive_help_view(
            console=self.console,
            current_model=self.get_active_project_model(),
            current_effort=self.get_active_project_effort(),
            mode="project",
            context_label=proj_name,
        )

    def handle_project_command(self, cmd_line: str) -> None:
        """Route slash commands in Project mode."""
        parts = cmd_line.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/help", "/?"):
            self.print_project_help()
        elif cmd in ("/findpaper", "/scout", "/paper-scout"):
            self.cmd_findpaper(args)
        elif cmd in ("/dashboard", "/close", "/back"):
            self.cmd_return_dashboard()
        elif cmd in ("/delete", "/remove", "/rm"):
            self.cmd_delete_project(args)
        elif cmd in ("/vault", "/docs", "/v"):
            self.render_mode_divider()
            if args:
                self.handle_vault_interaction(args)
            else:
                self.sub_mode = "vault"
                self.render_vault_landing()
        elif cmd in ("/processing", "/proc", "/p"):
            self.render_mode_divider()
            if args:
                self.handle_processing_interaction(args)
            else:
                self.sub_mode = "processing"
                self.render_processing_summary()
        elif cmd in ("/chat", "/c"):
            self.render_mode_divider()
            if args:
                self.handle_chat_hub_interaction(args)
            else:
                self.sub_mode = "chat_hub"
                self.cmd_chat_list_sessions()
        elif cmd in ("/rename",):
            self.cmd_rename_project(args)
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
        elif cmd in ("/models", "/model", "/config"):
            self.cmd_models(args)
        elif cmd in ("/tokens", "/usage"):
            self.cmd_models(f"tokens {args}".strip())
        elif cmd in ("/gdrive",):
            self.cmd_vault_gdrive(args)
        elif cmd in ("/settings", "/setting", "/style"):
            self.cmd_settings(args)
        elif cmd in ("/docx", "/export"):
            self.cmd_docx(args)
        elif cmd in ("/status", "/info"):
            self.cmd_project_status()
        elif cmd == "/clear":
            self.clear_screen()
        elif cmd == "/web":
            self.cmd_web(args)
        else:
            clean_cmd = cmd_line.lstrip("/")
            self.console.print(f"[yellow]Perintah tidak dikenal: '{clean_cmd}'.[/yellow]")
            self.console.print("[dim]Ketik 'help' untuk melihat daftar perintah, atau 'chat' untuk mulai berdiskusi dengan AI.[/dim]\n")

    def cmd_scout(self) -> None:
        """Interactive natural-language literature scout workflow with OpenAlex."""
        if not self.workspace or not self.scout:
            self.console.print("[red][x] Please open a project first to use /scout.[/red]\n")
            return

        self.console.print()
        self.console.print("\n[bold bright_white]Intelligent Literature Scout Pipeline[/bold bright_white]")
        self.console.print(
            "Tuliskan kebutuhan riset / jurnal Anda secara bebas (bahasa alami).\n"
            "Sistem akan merumuskan parameter pencarian, menarik paper nyata via [bold cyan]OpenAlex[/bold cyan] "
            "dan Semantic Scholar, lalu mengkurasi paper yang paling relevan untuk project Anda.\n"
            "[dim]Contoh: 'Paper tentang kesehatan mental remaja tahun 2018-2020 yang open access'[/dim]\n"
        )

        try:
            user_input = self.console.input("[bold cyan]scout > [/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[yellow]Scout canceled.[/yellow]\n")
            return

        if not user_input or user_input.lower() in ("q", "quit", "cancel", "exit"):
            self.console.print("[yellow]Scout canceled.[/yellow]\n")
            return

        # Phase 1: Parameter Formulation
        with self.console.status("[cyan]Formulating academic query strategy from your description...[/cyan]"):
            query_spec = self.scout.extract_search_criteria(user_input, engine=self.engine)

        self.console.print("\n[bold bright_white]Extracted Query Strategy[/bold bright_white]")
        q_table = Table(box=None, padding=(0, 2), show_header=False)
        q_table.add_column("Field", style="dim", width=14)
        q_table.add_column("Value", style="white")
        q_table.add_row("Keywords", query_spec.keywords)
        q_table.add_row("Year Filter", f"{query_spec.year_start} - {query_spec.year_end}" if query_spec.year_start else "All Years")
        q_table.add_row("Open Access", "Yes (OA Only)" if query_spec.is_oa_only else "Any")
        q_table.add_row("Summary", query_spec.summary)
        self.console.print(q_table)
        self.console.print()

        # Phase 2: Multi-Source Scraping
        with self.console.status("[cyan]Scraping verified scholarly works across OpenAlex & Semantic Scholar...[/cyan]"):
            try:
                candidates = asyncio.run(self.scout.search_multi_source(query_spec, limit=30))
            except Exception as exc:
                self.console.print(f"[bold red][x] Search scraping failed:[/bold red] {exc}\n")
                return

        if not candidates:
            self.console.print("[yellow]No scholarly candidates found matching the criteria.[/yellow]\n")
            return

        # Phase 3: AI Curation & Ranking (No hard limit, ranked by relevance with most relevant on top)
        with self.console.status(f"[cyan]Evaluating and ranking {len(candidates)} candidates against your research intent...[/cyan]"):
            try:
                curated = self.scout.curate_and_rank(user_input, candidates, engine=self.engine, max_output=None)
            except Exception:
                curated = candidates

        self.last_scout_results = curated

        # Phase 4 & 5: Paginated Carousel Table & Interactive Ingestion
        columns = [
            {"name": "#", "justify": "right", "style": "dim", "width": 3},
            {"name": "Badge", "justify": "center", "width": 14},
            {"name": "Title", "style": "bold white", "width": 34, "overflow": "fold"},
            {"name": "Authors", "style": "dim", "width": 18, "overflow": "fold"},
            {"name": "Year", "justify": "center", "width": 6, "style": "dim"},
            {"name": "PDF Status", "justify": "center", "width": 13},
            {"name": "Relevance / Summary", "style": "dim", "overflow": "fold"},
        ]

        rows = []
        for idx, item in enumerate(curated, 1):
            if item.source == "openalex":
                source_badge = "[bold cyan][OpenAlex][/bold cyan]"
            elif item.source == "ai_recommended":
                source_badge = "[bold cyan][AI Seminal][/bold cyan]"
            else:
                source_badge = "[dim][S2/ArXiv][/dim]"

            pdf_status = "[bold cyan][Open Access][/bold cyan]" if item.has_pdf else "[dim][Paywalled][/dim]"
            authors_str = ", ".join(item.authors[:2]) if item.authors else "Unknown"

            rows.append([
                str(idx),
                source_badge,
                item.title,
                authors_str,
                str(item.year or "n.d."),
                pdf_status,
                item.relevance_reason or "Matched search parameters.",
            ])

        dl_choice = interactive_carousel_table(
            title="Curated Literature Results",
            columns=columns,
            rows=rows,
            console=self.console,
            prompt_label="download > ",
            help_text="Ketik nomor paper yang ingin diunduh ke vault project (contoh: '1, 2' atau 'all', 'q' untuk batal):",
            item_label="papers",
        )

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

        selected_papers = [curated[i] for i in to_download_indices]
        task = self.task_manager.submit(
            name=f"Ingest {len(selected_papers)} paper(s) to vault",
            task_type="download",
            target_fn=self._background_download_worker,
            papers=selected_papers,
        )
        self.console.print()
        self.console.print(
            f"[bold green][✓] Background Download Agent Dispatched:[/bold green] [bold cyan]{task.id}[/bold cyan] ({len(selected_papers)} paper(s) queued)"
        )
        self.console.print(
            " [dim]• Aktivitas Anda tidak ter-block: Anda dapat langsung melanjutkan chat, riset, atau membaca draft.[/dim]"
        )
        self.console.print(
            " [dim]• Ketik [bold cyan]processing[/bold cyan] atau [bold cyan]proc[/bold cyan] untuk memantau progres unduhan & log real-time.[/dim]\n"
        )

    def _background_download_worker(self, task_id: str, papers: list[ScoutResult]) -> str:
        """Background agent worker downloading and ingesting academic papers into vault."""
        total = len(papers)
        success = 0
        failed = 0

        for idx, paper in enumerate(papers, 1):
            if not paper.has_pdf or not paper.pdf_url:
                self.task_manager.log(task_id, f"[-] Skipped #{idx} '{paper.title[:45]}': No open-access PDF available.")
                failed += 1
                continue

            self.task_manager.log(task_id, f"[{idx}/{total}] Downloading: '{paper.title[:45]}...'")

            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    dl_path = loop.run_until_complete(self.scout.download_scout_paper(paper))
                finally:
                    loop.close()

                self.active_documents.add(dl_path.stem)
                self._sync_vault_cache()
                self.task_manager.log(task_id, f"[✓] Added #{idx} to vault & indexed: {paper.title[:50]}")
                success += 1
            except Exception as exc:
                self.task_manager.log(task_id, f"[x] Error downloading #{idx} '{paper.title[:40]}': {exc}")
                failed += 1

        summary = f"Downloaded {success}/{total} paper(s) into vault ({failed} failed/skipped)."
        self.task_manager.log(task_id, f"Finished: {summary}")
        return summary

    def _background_scout_worker(self, task_id: str, query: str, max_download: int = 3) -> str:
        """Background agent worker that formulates criteria, searches multi-source, ranks, and downloads into vault."""
        clean_query = query.strip().strip("'\"")
        self.task_manager.log(task_id, f"Formulating query strategy for: '{clean_query}'")
        try:
            query_spec = self.scout.extract_search_criteria(clean_query, engine=self.engine)
        except Exception as exc:
            self.task_manager.log(task_id, f"[!] Criteria formulation fallback: {exc}")
            from scrpcy.core.scout import ScoutQuery
            query_spec = ScoutQuery(keywords=clean_query, summary=clean_query)

        self.task_manager.log(task_id, f"Scraping OpenAlex & Semantic Scholar (keywords: '{query_spec.keywords}')...")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            candidates = loop.run_until_complete(self.scout.search_multi_source(query_spec, limit=15))
        except Exception as exc:
            self.task_manager.log(task_id, f"[x] Multi-source search error: {exc}")
            raise
        finally:
            loop.close()

        if not candidates:
            msg = f"No scholarly candidates found for '{query}'."
            self.task_manager.log(task_id, msg)
            return msg

        self.task_manager.log(task_id, f"Curating & ranking {len(candidates)} candidates against research intent...")
        try:
            curated = self.scout.curate_and_rank(query, candidates, engine=self.engine, max_output=None)
        except Exception:
            curated = candidates

        self.last_scout_results = curated

        oa_papers = [p for p in curated if p.has_pdf and p.pdf_url][:max_download]
        if not oa_papers:
            msg = f"Found {len(curated)} candidates, but none had direct open-access PDF links."
            self.task_manager.log(task_id, msg)
            return msg

        self.task_manager.log(task_id, f"Downloading top {len(oa_papers)} open-access paper(s) into vault...")
        downloaded = 0

        for idx, paper in enumerate(oa_papers, 1):
            self.task_manager.log(task_id, f"[{idx}/{len(oa_papers)}] Downloading '{paper.title[:45]}...'")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                dl_path = loop.run_until_complete(self.scout.download_scout_paper(paper))
                self.active_documents.add(dl_path.stem)
                self._sync_vault_cache()
                self.task_manager.log(task_id, f"[✓] Added #{idx} to vault & indexed: {paper.title[:50]}")
                downloaded += 1
            except Exception as exc:
                self.task_manager.log(task_id, f"[x] Error downloading #{idx} '{paper.title[:35]}': {exc}")
            finally:
                loop.close()

        summary = f"Scout & Ingest finished: Added {downloaded}/{len(oa_papers)} paper(s) to vault."
        self.task_manager.log(task_id, summary)
        return summary

    def cmd_return_dashboard(self) -> None:
        """Disconnect from active project and return to dashboard."""
        self.render_mode_divider()
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
        """Display status, assigned model, and token consumption for the active master project."""
        if not self.current_project or not self.workspace:
            self.console.print("[dim]Belum ada project yang aktif. Buka project terlebih dahulu dengan 'open <nama>'.[/dim]\n")
            return

        cached_docs = self._get_cached_stems()
        active_count = len(self.active_documents)
        model_name = self.get_active_project_model()
        model_display = _format_model_display_name(model_name)
        effort = self.get_active_project_effort()
        token_stats = TokenTracker.get_project_tokens(self.workspace.root_dir)

        self.console.print(f"\n[bold bright_white]Status Project: {self.current_project.name}[/bold bright_white]")
        table = Table(box=None, padding=(0, 2), show_header=False)
        table.add_column("Property", style="dim", width=22)
        table.add_column("Details", style="white")

        table.add_row("Project Name", f"[bold bright_white]{self.current_project.name}[/bold bright_white]")
        table.add_row("Project Model", f"[bold cyan]{model_display}[/bold cyan] [dim]({effort} effort)[/dim]")

        token_display = (
            f"Total: [bold bright_white]{token_stats.total_tokens:,}[/bold bright_white] tokens "
            f"[dim](In: {token_stats.input_tokens:,} | Out: {token_stats.output_tokens:,} | {token_stats.request_count} reqs)[/dim]"
        )
        table.add_row("Project Token Usage", token_display)
        if token_stats.last_active:
            table.add_row("Last LLM Activity", token_stats.last_active.replace("T", " ")[:19] + " UTC")

        table.add_row("Active Skill", self.active_skill_name)
        table.add_row("Vault Papers", f"{len(cached_docs)} indexed ({active_count} active rujukan)")
        table.add_row("Citation Style", self.config.citation.style.upper())
        table.add_row("Master Path", str(self.workspace.root_dir))
        table.add_row("Drafts Folder", str(self.workspace.root_dir / "drafts"))

        self.console.print(table)
        self.console.print()

    def clear_screen(self) -> None:
        """Clear the terminal screen and reprint logo."""
        os.system("cls" if os.name == "nt" else "clear")
        self.print_banner()

    def cmd_rename_project(self, new_name: str) -> None:
        """Rename the currently active project."""
        if not self.current_project:
            self.console.print("[red][x] No active project to rename.[/red]\n")
            return

        new_name = new_name.strip()
        if not new_name:
            self.console.print("[yellow]Usage: rename <new_name>[/yellow]\n")
            return

        try:
            old_name = self.current_project.name
            renamed = self.project_manager.rename_project(old_name, new_name)
            self.current_project = renamed
            self.console.print(f"[bold green][✓] Project '{old_name}' successfully renamed to:[/bold green] [bold cyan]{renamed.name}[/bold cyan]\n")
        except Exception as exc:
            self.console.print(f"[bold red][x] Failed to rename project:[/bold red] {exc}\n")

    def cmd_findpaper(self, query: str = "") -> None:
        """Search and scout academic literature on OpenAlex and Semantic Scholar in background."""
        clean_query = query.strip().strip("'\"")
        if not clean_query:
            self.cmd_scout()
            return

        if not self.workspace or not self.scout:
            self.console.print("[red][x] Please open a project first.[/red]\n")
            return

        task = self.task_manager.submit(
            name=f"Literature Scout & Ingest: {clean_query[:40]}",
            task_type="scout",
            target_fn=self._background_scout_worker,
            query=clean_query,
            max_download=3,
        )
        self.console.print()
        self.console.print(
            f"[bold green][✓] Background Literature Scout Dispatched:[/bold green] [bold cyan]{task.id}[/bold cyan]"
        )
        self.console.print(
            f" [dim]• Topik pencarian : '{clean_query}'[/dim]"
        )
        self.console.print(
            " [dim]• Proses pencarian di OpenAlex/Semantic Scholar dan download PDF berjalan di background.[/dim]"
        )
        self.console.print(
            " [dim]• Ketik [bold cyan]processing[/bold cyan] atau [bold cyan]proc[/bold cyan] untuk memantau progres & log real-time.[/dim]\n"
        )

    # =========================================================================
    # VAULT SUB-MODE (skripsi/vault>)
    # =========================================================================

    def render_vault_landing(self) -> None:
        """Render concise status overview of vault resources."""
        if not self.vault_manager or not self.workspace:
            return

        active_items = self.vault_manager.list_active_files()
        cached_count = sum(1 for it in active_items if it.is_cached)
        needs_reload = len(active_items) - cached_count
        archived_count = len(self.vault_manager.list_archived_files())
        vault_path = self.workspace.root_dir / "vault"

        self.console.print(f"\n[bold bright_white]Vault Resources [{self.current_project.name if self.current_project else 'Active'}][/bold bright_white]")
        v_table = Table(box=None, padding=(0, 2), show_header=False)
        v_table.add_column("Label", style="dim", width=18)
        v_table.add_column("Value", style="white")
        v_table.add_row("Status", "[bold cyan]Aktif[/bold cyan]")
        v_table.add_row("Folder Vault", str(vault_path))
        v_table.add_row("Total Berkas", f"{len(active_items)} berkas aktif ({cached_count} terindeks Markdown, {needs_reload} perlu reload)")
        v_table.add_row("Diarsipkan", f"{archived_count} berkas di .archive/ (tidak aktif di konteks AI)")
        v_table.add_row("Format Dokumen", "[dim]PDF, DOCX, PPTX, XLSX, TXT, HTML, MD (didukung Microsoft MarkItDown)[/dim]")
        self.console.print(v_table)
        self.console.print()

    def render_vault_view(self) -> None:
        """Render active vault items table and usage hint."""
        if not self.vault_manager:
            return
        items = self.vault_manager.list_active_files()
        if not items:
            self.console.print("\n[dim]Vault masih kosong (0 berkas aktif).[/dim]")
            self.console.print("[dim]Gunakan 'add <path>', 'findpaper <topik>', atau 'gdrive' untuk menambahkan berkas.[/dim]\n")
            return

        self.console.print(f"\n[bold bright_white]Daftar Berkas Vault [{self.current_project.name if self.current_project else 'Vault'}][/bold bright_white]")
        table = Table(
            box=None,
            padding=(0, 2),
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("#", justify="right", style="dim", width=3)
        table.add_column("Filename", style="bold white")
        table.add_column("Format", justify="center", width=8, style="dim")
        table.add_column("Size", justify="right", width=10, style="dim")
        table.add_column("Markdown Status", justify="center", width=18)
        table.add_column("Cite Key / Title", style="cyan")

        for idx, item in enumerate(items, 1):
            md_status = "[bold cyan][Ready / Cached][/bold cyan]" if item.is_cached else "[dim][Needs Reload][/dim]"
            table.add_row(
                str(idx),
                item.filename,
                item.extension.upper().lstrip("."),
                item.readable_size,
                md_status,
                item.title or item.cite_key,
            )

        self.console.print(table)
        archived_count = len(self.vault_manager.list_archived_files())
        archive_hint = f" ({archived_count} files in .archive/)" if archived_count else ""
        self.console.print(
            f"[dim]Ketik 'view <#>' untuk baca isi markdown, 'remove <#>' untuk arsip, atau 'back' untuk kembali{archive_hint}.[/dim]\n"
        )

    def print_vault_help(self) -> None:
        """Render instructions for Vault sub-mode."""
        interactive_help_view(
            console=self.console,
            current_model=self.get_active_project_model(),
            current_effort=self.get_active_project_effort(),
            mode="vault",
            context_label=self.current_project.name if self.current_project else "",
        )

    def handle_vault_interaction(self, raw_input: str) -> None:
        """Route input when inside vault sub-mode (skripsi/vault>)."""
        clean = raw_input.strip()
        if not clean:
            return

        parts = clean.split(maxsplit=1)
        cmd = parts[0].lower().lstrip("/")
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("back", "exit", "quit", "cd..", "close") or clean in ("cd ..", ".."):
            self.render_mode_divider()
            self.sub_mode = None
            self.console.print(f"[dim]Returned to project root: {self.current_project.name}>[/dim]\n")
            return

        if cmd in ("help", "?"):
            self.print_vault_help()
        elif cmd in ("vault", "status"):
            self.render_vault_landing()
        elif cmd in ("list", "ls"):
            if arg in ("archive", "archived", ".archive"):
                self.cmd_vault_list_archived()
            else:
                self.render_vault_view()
        elif cmd in ("archived", "archive"):
            self.cmd_vault_list_archived()
        elif cmd in ("reload", "refresh"):
            self.cmd_vault_reload()
        elif cmd == "add":
            self.cmd_vault_add(arg)
        elif cmd in ("findpaper", "scout"):
            self.cmd_findpaper(arg)
        elif cmd == "gdrive":
            self.cmd_vault_gdrive(arg)
        elif cmd in ("remove", "rm"):
            self.cmd_vault_remove(arg)
        elif cmd in ("restore", "unarchive"):
            self.cmd_vault_restore(arg)
        elif cmd == "view":
            self.cmd_view(arg)
        elif cmd == "clear":
            self.clear_screen()
        else:
            self.console.print(f"[yellow]Unknown vault command: '{clean}'. Type 'help' to see vault commands, or 'back' to return.[/yellow]\n")

    def cmd_vault_reload(self) -> None:
        """Scan vault folder and convert all new or updated files to markdown using MarkItDown."""
        if not self.vault_manager or not self.workspace:
            self.console.print("[red][x] Vault manager is not active.[/red]\n")
            return

        with self.console.status("[cyan]Scanning vault and converting documents with Microsoft MarkItDown...[/cyan]"):
            total, converted = self.vault_manager.reload_vault(force=False)
            self._sync_vault_cache()

        self.console.print(f"[bold green][✓] Vault reload completed:[/bold green] {total} total resource(s), [bold cyan]{converted} converted/refreshed[/bold cyan] into Markdown cache.")
        self.console.print(f"[dim]Active citation context updated ({len(self.active_documents)} items active).[/dim]\n")

    def _background_vault_add_worker(self, task_id: str, path_str: str) -> str:
        """Background worker ingesting local document and converting to Markdown."""
        src_path = Path(path_str).expanduser().resolve()
        self.task_manager.log(task_id, f"Ingesting '{src_path.name}' into vault...")
        try:
            item = self.vault_manager.add_resource(src_path)
            self.active_documents.add(item.path.stem)
            self._sync_vault_cache()
            summary = f"Added {item.filename} ({item.readable_size}) to vault"
            self.task_manager.log(task_id, f"[✓] {summary}")
            return summary
        except Exception as exc:
            self.task_manager.log(task_id, f"[x] Ingest failed: {exc}")
            raise

    def cmd_vault_add(self, path_str: str) -> None:
        """Add external file into project vault and convert with MarkItDown."""
        if not self.vault_manager or not self.workspace:
            self.console.print("[red][x] Vault manager is not active.[/red]\n")
            return

        if not path_str:
            self.console.print("[yellow]Usage: add <path/to/file>[/yellow]\n")
            return

        src_path = Path(path_str).expanduser().resolve()
        if not src_path.exists():
            self.console.print(f"[bold red][x] File not found at:[/bold red] '{src_path}'\n")
            return

        if self.sub_mode == "chat":
            task = self.task_manager.submit(
                name=f"Ingest Document: {src_path.name}",
                task_type="convert",
                target_fn=self._background_vault_add_worker,
                path_str=str(src_path),
            )
            self.console.print()
            self.console.print(
                f"[bold green][✓] Proses ingest berkas dikirim ke antrean Processing:[/bold green] [bold cyan]{task.id}[/bold cyan]"
            )
            self.console.print(f" [dim]• Berkas : {src_path.name}[/dim]")
            self.console.print(" [dim]• Konversi markdown dan sinkronisasi cache berjalan di background.[/dim]")
            self.console.print(" [dim]• Ketik [bold cyan]/proc[/bold cyan] untuk memantau status.[/dim]\n")
            return

        with self.console.status(f"[cyan]Adding and converting '{src_path.name}' with MarkItDown...[/cyan]"):
            try:
                item = self.vault_manager.add_resource(src_path)
                self.active_documents.add(item.path.stem)
                self._sync_vault_cache()
            except Exception as exc:
                self.console.print(f"[bold red][x] Failed to add resource:[/bold red] {exc}\n")
                return

        self.console.print(f"[bold green][✓] Added resource to vault & converted to Markdown:[/bold green] [bold cyan]{item.filename}[/bold cyan]")
        self.console.print(f"    Format: {item.extension.upper().lstrip('.')} | Size: {item.readable_size}\n")

    def cmd_vault_remove(self, identifier: str) -> None:
        """Soft-remove a file from active project by moving it to .archive/."""
        if not self.vault_manager:
            return

        if not identifier:
            self.console.print("[yellow]Usage: remove <# | filename>[/yellow]\n")
            return

        try:
            archived = self.vault_manager.archive_resource(identifier)
            self.active_documents.discard(archived.path.stem)
            self.console.print(f"[yellow][-] File '{archived.filename}' moved to .archive/.[/yellow]")
            self.console.print(f"[dim]Excluded from active AI context. Use 'restore {archived.filename}' to re-include anytime.[/dim]\n")
        except Exception as exc:
            self.console.print(f"[bold red][x] Failed to remove file:[/bold red] {exc}\n")

    def cmd_vault_restore(self, identifier: str) -> None:
        """Restore an archived file back to active vault/."""
        if not self.vault_manager:
            return

        if not identifier:
            self.console.print("[yellow]Usage: restore <# | filename>[/yellow]\n")
            return

        try:
            restored = self.vault_manager.restore_resource(identifier)
            self.active_documents.add(restored.path.stem)
            self._sync_vault_cache()
            self.console.print(f"[bold green][✓] File '{restored.filename}' restored to active vault and re-indexed into AI context.[/bold green]\n")
        except Exception as exc:
            self.console.print(f"[bold red][x] Failed to restore file:[/bold red] {exc}\n")

    def cmd_vault_list_archived(self) -> None:
        """List soft-removed files stored inside .archive/."""
        if not self.vault_manager:
            return

        archived = self.vault_manager.list_archived_files()
        if not archived:
            self.console.print("[dim]No archived files in .archive/. All vault files are currently active.[/dim]\n")
            return

        self.console.print(f"\n[bold bright_white]Archived Vault Files ({len(archived)} files in .archive/)[/bold bright_white]")
        table = Table(
            box=None,
            padding=(0, 2),
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("#", justify="right", style="dim", width=3)
        table.add_column("Filename", style="bold white")
        table.add_column("Format", justify="center", width=8, style="dim")
        table.add_column("Size", justify="right", width=10, style="dim")
        table.add_column("Status", justify="center", width=14, style="dim")

        for idx, f in enumerate(archived, 1):
            table.add_row(
                str(idx),
                f.filename,
                f.extension.upper().lstrip("."),
                f.readable_size,
                "[dim][Archived][/dim]",
            )

        self.console.print(table)
        self.console.print("[dim]Use [bold cyan]restore <# | name>[/bold cyan] to move an archived file back to active vault.[/dim]\n")

    def _background_gdrive_worker(self, task_id: str, file_id: str) -> str:
        """Background worker downloading Google Drive file and ingesting into vault."""
        self.task_manager.log(task_id, f"Connecting to Google Drive to download file ID: {file_id}")
        try:
            downloaded = self.gdrive_manager.download_public_file(
                file_id=file_id,
                dest_dir=self.workspace.vault_dir,
            )
            self.task_manager.log(task_id, f"Downloaded '{downloaded.name}'. Converting to Markdown via MarkItDown...")
            self.vault_manager.convert_file(downloaded, force=True)
            self.active_documents.add(downloaded.stem)
            self._sync_vault_cache()
            summary = f"Downloaded & indexed Google Drive file: {downloaded.name}"
            self.task_manager.log(task_id, f"[✓] {summary}")
            return summary
        except Exception as exc:
            self.task_manager.log(task_id, f"[x] GDrive download error: {exc}")
            raise

    def cmd_vault_gdrive(self, arg: str = "") -> None:
        """Download and ingest files from Google Drive into vault with MarkItDown conversion."""
        if not self.workspace or not self.vault_manager:
            self.console.print("[red][x] Please open a project first.[/red]\n")
            return

        clean_arg = arg.strip()
        if not clean_arg:
            self.console.print("\n[bold bright_white]Google Drive Vault Ingestion[/bold bright_white]")
            self.console.print(
                "Anda dapat mengunduh dokumen dari Google Drive langsung ke vault:\n"
                "  1. Masukkan [bold cyan]link berbagi (share link)[/bold cyan] atau [bold cyan]File ID[/bold cyan] Google Drive.\n"
                "  2. Atau ketik [bold cyan]auth[/bold cyan] untuk menghasilkan tautan OAuth login Google Drive Anda.\n\n"
                "[dim]Contoh link: https://drive.google.com/file/d/1A2B3C4D5E.../view?usp=sharing[/dim]\n"
            )
            try:
                clean_arg = self.console.input("[bold cyan]gdrive > [/bold cyan]").strip()
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n")
                return

        if not clean_arg or clean_arg.lower() in ("q", "quit", "cancel", "back"):
            return

        if clean_arg.lower() == "auth":
            client_secrets = Path.home() / ".scrpcy" / "credentials.json"
            if not client_secrets.exists():
                self.console.print(
                    f"[yellow][!] File 'credentials.json' belum ditemukan di:[/yellow] [dim]{client_secrets}[/dim]\n"
                    "Untuk login OAuth akun pribadi:\n"
                    "1. Unduh OAuth client ID (Desktop App) dari Google Cloud Console\n"
                    f"2. Simpan sebagai '{client_secrets}'\n"
                    "Atau gunakan link shareable langsung dengan opsi 'Anyone with the link'.\n"
                )
                return

            try:
                auth_url, flow = self.gdrive_manager.get_oauth_auth_url(client_secrets)
                self.console.print("\n[bold green]Silakan buka tautan OAuth berikut di browser Anda:[/bold green]")
                self.console.print(f"[link={auth_url}][bold cyan]{auth_url}[/bold cyan][/link]\n")
                code = self.console.input("Masukkan kode otorisasi dari browser: ").strip()
                if code:
                    self.gdrive_manager.finish_oauth_with_code(flow, code)
                    self.console.print("[bold green][✓] Autentikasi Google Drive berhasil disimpan![/bold green]\n")
            except Exception as exc:
                self.console.print(f"[bold red][x] Autentikasi gagal:[/bold red] {exc}\n")
            return

        file_id = extract_drive_file_id(clean_arg)
        if not file_id:
            self.console.print(f"[bold red][x] Format link atau ID Google Drive tidak valid:[/bold red] '{clean_arg}'\n")
            return

        task = self.task_manager.submit(
            name=f"Google Drive Ingest: {file_id[:12]}...",
            task_type="gdrive",
            target_fn=self._background_gdrive_worker,
            file_id=file_id,
        )
        self.console.print()
        self.console.print(
            f"[bold green][✓] Proses download Google Drive dikirim ke antrean Processing:[/bold green] [bold cyan]{task.id}[/bold cyan]"
        )
        self.console.print(f" [dim]• File ID : {file_id}[/dim]")
        self.console.print(" [dim]• Proses unduh dan konversi markdown berjalan di background.[/dim]")
        self.console.print(" [dim]• Ketik [bold cyan]processing[/bold cyan] atau [bold cyan]/proc[/bold cyan] untuk memantau status & log real-time.[/dim]\n")

    # =========================================================================
    # PROCESSING SUB-MODE (skripsi/processing>)
    # =========================================================================

    def render_processing_summary(self) -> None:
        """Render concise status overview of background processing tasks."""
        tasks = self.task_manager.list_tasks()
        running_tasks = [t for t in tasks if t.status == "running"]
        completed_tasks = [t for t in tasks if t.status == "completed"]
        failed_tasks = [t for t in tasks if t.status == "failed"]
        cancelled_tasks = [t for t in tasks if t.status == "cancelled"]

        status_text = "[bold cyan]Active Processing[/bold cyan]" if running_tasks else "[dim]Idle[/dim]"

        self.console.print(f"\n[bold bright_white]Processing Status [{self.current_project.name if self.current_project else 'Studio'}][/bold bright_white]")
        table = Table(box=None, padding=(0, 2), show_header=False)
        table.add_column("Property", style="dim", width=18)
        table.add_column("Value", style="white")

        table.add_row("Status", status_text)
        table.add_row("Running Tasks", f"[bold cyan]{len(running_tasks)}[/bold cyan] proses sedang berjalan" if running_tasks else "0 proses")
        table.add_row("Completed Tasks", f"[bold white]{len(completed_tasks)}[/bold white] selesai")
        table.add_row("Failed / Cancelled", f"{len(failed_tasks)} gagal, {len(cancelled_tasks)} dibatalkan")
        table.add_row("Total Riwayat", f"{len(tasks)} tasks terdaftar")

        self.console.print(table)
        self.console.print()

    def render_processing_view(self) -> None:
        """Render borderless duotone table of background tasks."""
        tasks = self.task_manager.list_tasks()
        table = Table(
            title=f"Background Processing Tasks [{self.current_project.name if self.current_project else 'Studio'}]",
            box=None,
            padding=(0, 2),
            title_style="bold bright_white",
            header_style="bold cyan",
        )
        table.add_column("#", justify="right", style="dim", width=3)
        table.add_column("Task ID", style="bold cyan", width=10)
        table.add_column("Type", justify="center", style="dim", width=10)
        table.add_column("Description", style="white")
        table.add_column("Status", justify="center", width=14)
        table.add_column("Duration", justify="right", style="dim", width=9)
        table.add_column("Progress / Note", style="dim")

        for idx, t in enumerate(tasks, 1):
            if t.status == "running":
                status_badge = "[bold cyan][Running][/bold cyan]"
            elif t.status == "completed":
                status_badge = "[bold white][Completed][/bold white]"
            elif t.status == "cancelled":
                status_badge = "[dim][Cancelled][/dim]"
            else:
                status_badge = "[bold red][Failed][/bold red]"

            duration_str = f"{t.duration_seconds:.1f}s"
            table.add_row(
                str(idx),
                t.id,
                t.task_type.upper(),
                t.name,
                status_badge,
                duration_str,
                t.progress[:35],
            )

        self.console.print(table)
        if not tasks:
            self.console.print("[dim]No background tasks currently registered.[/dim]")
        self.console.print("[dim]Commands: [bold cyan]list[/bold cyan], [bold cyan]detail <#|id>[/bold cyan], [bold cyan]log <#|id>[/bold cyan], [bold cyan]cancel <#|id>[/bold cyan], [bold cyan]clear[/bold cyan], [bold cyan]back[/bold cyan][/dim]\n")

    def print_processing_help(self) -> None:
        """Render instructions for Processing sub-mode."""
        interactive_help_view(
            console=self.console,
            current_model=self.get_active_project_model(),
            current_effort=self.get_active_project_effort(),
            mode="processing",
            context_label=self.current_project.name if self.current_project else "",
        )

    def handle_processing_interaction(self, raw_input: str) -> None:
        """Route input when inside processing sub-mode (skripsi/processing>)."""
        clean = raw_input.strip()
        if not clean:
            return

        parts = clean.split(maxsplit=1)
        cmd = parts[0].lower().lstrip("/")
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("back", "exit", "quit", "cd..", "close") or clean in ("cd ..", ".."):
            self.render_mode_divider()
            self.sub_mode = None
            self.console.print(f"[dim]Returned to project root: {self.current_project.name}>[/dim]\n")
            return

        if cmd in ("help", "?"):
            self.print_processing_help()
        elif cmd in ("list", "ls"):
            self.render_processing_view()
        elif cmd in ("status", "summary"):
            self.render_processing_summary()
        elif cmd in ("detail", "view", "show"):
            self.cmd_processing_detail(arg)
        elif cmd in ("log", "logs"):
            self.cmd_processing_log(arg)
        elif cmd in ("cancel", "kill", "abort", "stop"):
            self.cmd_processing_cancel(arg)
        elif cmd == "clear":
            count = self.task_manager.clear_completed()
            self.console.print(f"[green][✓] Cleared {count} completed/cancelled task(s) from history.[/green]\n")
        else:
            self.console.print(f"[yellow]Unknown processing command: '{clean}'. Type '?' or '/' for available commands, or 'back' to return.[/yellow]\n")

    def cmd_processing_detail(self, arg: str) -> None:
        """Display detailed metadata and result summary for a task."""
        if not arg:
            self.console.print("[yellow]Usage: detail <# | task_id>[/yellow]\n")
            return

        task = self.task_manager.get_task(arg)
        if not task:
            self.console.print(f"[bold red][x] Task '{arg}' not found.[/bold red]\n")
            return

        self.console.print(f"\n[bold bright_white]Task Detail: {task.id}[/bold bright_white]")
        t_table = Table(box=None, padding=(0, 2), show_header=False)
        t_table.add_column("Property", style="dim", width=16)
        t_table.add_column("Details", style="white")
        t_table.add_row("ID", f"[bold cyan]{task.id}[/bold cyan]")
        t_table.add_row("Name", task.name)
        t_table.add_row("Type", task.task_type.upper())
        t_table.add_row("Status", task.status.upper())
        t_table.add_row("Started At", str(task.started_at))
        t_table.add_row("Finished At", str(task.finished_at or 'Running...'))
        t_table.add_row("Duration", f"{task.duration_seconds:.2f} seconds")
        t_table.add_row("Progress", task.progress)
        if task.error:
            t_table.add_row("Error", f"[bold red]{task.error}[/bold red]")
        if task.result_summary:
            t_table.add_row("Result", f"[bold cyan]{task.result_summary}[/bold cyan]")
        self.console.print(t_table)
        self.console.print()

    def cmd_processing_log(self, arg: str) -> None:
        """Display all captured log lines for a background task."""
        if not arg:
            self.console.print("[yellow]Usage: log <# | task_id>[/yellow]\n")
            return

        task = self.task_manager.get_task(arg)
        if not task:
            self.console.print(f"[bold red][x] Task '{arg}' not found.[/bold red]\n")
            return

        if not task.logs:
            self.console.print(f"[dim]No logs recorded yet for task {task.id}.[/dim]\n")
            return

        self.console.print(f"[bold cyan]Logs for Task {task.id} ({task.name}):[/bold cyan]")
        for line in task.logs:
            self.console.print(f"  {line}")
        self.console.print()

    def cmd_processing_cancel(self, arg: str) -> None:
        """Cancel a running background job."""
        if not arg:
            self.console.print("[yellow]Usage: cancel <# | task_id>[/yellow]\n")
            return

        cancelled = self.task_manager.cancel_task(arg)
        if cancelled:
            self.console.print(f"[yellow][!] Cancellation signal sent to task '{arg}'.[/yellow]\n")
        else:
            self.console.print(f"[bold red][x] Could not cancel task '{arg}' (not found or already completed).[/bold red]\n")

    # =========================================================================
    # CHAT SUB-MODE (skripsi/chat1>)
    # =========================================================================

    def cmd_chat(self, arg: str = "") -> None:
        """Route to chat hub or directly process chat command."""
        self.render_mode_divider()
        clean = arg.strip()
        if clean:
            self.handle_chat_hub_interaction(clean)
        else:
            self.sub_mode = "chat_hub"
            self.cmd_chat_list_sessions()

    def handle_chat_root_command(self, arg_str: str) -> None:
        """Alias for handle_chat_hub_interaction for backwards compatibility."""
        self.handle_chat_hub_interaction(arg_str)

    def handle_chat_hub_interaction(self, raw_input: str) -> None:
        """Process commands when in chat hub (skripsi/chat>)."""
        clean = raw_input.strip()
        if not clean:
            return

        parts = clean.split(maxsplit=1)
        sub = parts[0].lower().lstrip("/")
        param = parts[1].strip() if len(parts) > 1 else ""

        if sub in ("back", "exit", "quit", "cd..", "close", ".."):
            self.render_mode_divider()
            self.sub_mode = None
            self.console.print(f"[dim]Returned to project root: {self.current_project.name}>[/dim]\n")
            return

        if sub in ("list", "ls", "sessions"):
            self.cmd_chat_list_sessions()
            return

        if sub == "new":
            if not self.chat_manager:
                return
            session = self.chat_manager.create_session(
                name=param if param else None,
                active_skill=self.active_skill_name,
            )
            self.render_mode_divider()
            self.active_chat_session = session
            self.sub_mode = "chat"
            self.console.print(f"[bold green][✓] Sesi chat baru dibuat:[/bold green] [bold cyan]{session.name}[/bold cyan]")
            self.console.print("[dim]Ketik pertanyaan riset Anda langsung di bawah ini. Gunakan '/' untuk perintah (contoh: /back).[/dim]\n")
            return

        if sub in ("resume", "open", "load", "switch"):
            if not self.chat_manager:
                return
            target = param
            if not target:
                self.cmd_chat_list_sessions()
                try:
                    target = self.console.input("Masukkan nomor atau nama chat yang ingin dibuka: ").strip()
                except (KeyboardInterrupt, EOFError):
                    return
            session = self.chat_manager.get_session(target)
            if session:
                self.render_mode_divider()
                self.active_chat_session = session
                self.sub_mode = "chat"
                self.console.print(f"[bold green][✓] Masuk ke sesi percakapan:[/bold green] [bold cyan]{session.name}[/bold cyan] ({session.message_count} pesan)")
                self.console.print("[dim]Ketik pertanyaan riset Anda langsung di bawah ini. Gunakan '/' untuk perintah (contoh: /back).[/dim]\n")
            else:
                self.console.print(f"[red][x] Sesi chat '{target}' tidak ditemukan.[/red]\n")
            return

        if sub in ("delete", "remove", "rm"):
            if not self.chat_manager:
                return
            target = param
            if not target:
                self.cmd_chat_list_sessions()
                try:
                    target = self.console.input("Masukkan nomor atau nama chat yang ingin dihapus: ").strip()
                except (KeyboardInterrupt, EOFError):
                    return
            session = self.chat_manager.get_session(target)
            if session:
                deleted_name = session.name
                self.chat_manager.delete_session(session.id)
                if self.active_chat_session and self.active_chat_session.id == session.id:
                    self.active_chat_session = None
                self.console.print(f"[bold green][✓] Sesi chat '{deleted_name}' berhasil dihapus.[/bold green]\n")
                self.cmd_chat_list_sessions()
            else:
                self.console.print(f"[red][x] Sesi chat '{target}' tidak ditemukan.[/red]\n")
            return

        if sub == "clear":
            self.clear_screen()
            return

        if sub in ("help", "?"):
            interactive_help_view(
                console=self.console,
                current_model=self.get_active_project_model(),
                current_effort=self.get_active_project_effort(),
                mode="chat_hub",
                context_label=self.current_project.name if self.current_project else "",
            )
            return

        # Direct number or session name shorthand (e.g. typing "1" or "chat1" directly)
        if self.chat_manager:
            session = self.chat_manager.get_session(sub)
            if session:
                self.render_mode_divider()
                self.active_chat_session = session
                self.sub_mode = "chat"
                self.console.print(f"[bold green][✓] Masuk ke sesi percakapan:[/bold green] [bold cyan]{session.name}[/bold cyan] ({session.message_count} pesan)")
                self.console.print("[dim]Ketik pertanyaan riset Anda langsung di bawah ini. Gunakan '/' untuk perintah (contoh: /back).[/dim]\n")
                return

        self.console.print(f"[yellow]Perintah tidak dikenal: '{clean}'. Ketik '?' atau '/' untuk melihat perintah yang tersedia.[/yellow]\n")

    def handle_chat_submode_interaction(self, raw_input: str) -> None:
        """Process input when inside a specific chat session (skripsi/chat1>)."""
        clean = raw_input.strip()
        if not clean:
            return

        # In chat mode, commands MUST start with "/"
        if clean.startswith("/"):
            parts = clean.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("/back", "/exit", "/quit", "/close", "/cd..", "/.."):
                self.render_mode_divider()
                self.sub_mode = "chat_hub"
                self.console.print(f"[dim]Kembali ke menu chat: {self.current_project.name}/chat>[/dim]\n")
                return
            elif cmd in ("/help", "/?"):
                interactive_help_view(
                    console=self.console,
                    current_model=self.get_active_project_model(),
                    current_effort=self.get_active_project_effort(),
                    mode="chat",
                    context_label=self.current_project.name if self.current_project else "",
                )
            elif cmd == "/history":
                self.cmd_chat_history()
            elif cmd == "/clear":
                if self.active_chat_session and self.chat_manager:
                    self.chat_manager.clear_messages(self.active_chat_session.id)
                self.console.print("[dim][✓] Riwayat pesan untuk sesi ini telah dibersihkan.[/dim]\n")
            elif cmd in ("/status", "/info"):
                self.cmd_project_status()
            elif cmd in ("/skill", "/persona"):
                self.cmd_skill(arg)
                if self.active_chat_session and self.chat_manager:
                    self.chat_manager.update_session_skill(self.active_chat_session.id, self.active_skill_name)
            elif cmd == "/new":
                if not self.chat_manager:
                    return
                session = self.chat_manager.create_session(
                    name=arg if arg else None,
                    active_skill=self.active_skill_name,
                )
                self.render_mode_divider()
                self.active_chat_session = session
                self.console.print(f"[bold green][✓] Sesi chat baru dibuat:[/bold green] [bold cyan]{session.name}[/bold cyan]")
                self.console.print("[dim]Ketik pertanyaan riset Anda langsung di bawah ini. Gunakan '/' untuk perintah (contoh: /back).[/dim]\n")
            elif cmd in ("/resume", "/open", "/switch"):
                if not self.chat_manager:
                    return
                session = self.chat_manager.get_session(arg)
                if session:
                    self.render_mode_divider()
                    self.active_chat_session = session
                    self.console.print(f"[bold green][✓] Berpindah ke sesi percakapan:[/bold green] [bold cyan]{session.name}[/bold cyan] ({session.message_count} pesan)\n")
                else:
                    self.console.print(f"[red][x] Sesi chat '{arg}' tidak ditemukan.[/red]\n")
            elif cmd in ("/export",):
                self.cmd_chat_export()
            else:
                self.handle_project_command(clean)
            return

        # Any input without leading "/" in chat mode is strictly treated as an AI conversation prompt
        self.handle_chat(clean)

    def cmd_chat_list_sessions(self) -> None:
        """Render table of all chat sessions stored in this project."""
        if not self.chat_manager:
            return

        sessions = self.chat_manager.list_sessions()
        if not sessions:
            self.console.print("[dim]No saved chat sessions found in this project. Type 'chat new' to start one.[/dim]\n")
            return

        self.console.print(f"\n[bold bright_white]Saved Chat Sessions [{self.current_project.name if self.current_project else 'Active'}][/bold bright_white]")
        table = Table(
            box=None,
            padding=(0, 2),
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("#", justify="right", style="dim", width=3)
        table.add_column("Session Name", style="bold white")
        table.add_column("Persona", style="cyan")
        table.add_column("Messages", justify="center", width=10, style="dim")
        table.add_column("Last Updated", style="dim")

        for idx, s in enumerate(sessions, 1):
            is_active = self.active_chat_session and s.id == self.active_chat_session.id
            active_badge = f" [bold cyan](Current)[/bold cyan]" if is_active else ""
            table.add_row(
                str(idx),
                f"{s.name}{active_badge}",
                s.active_skill,
                str(s.message_count),
                s.updated_at[:19].replace("T", " "),
            )

        self.console.print(table)
        self.console.print("[dim]Use [bold cyan]resume <# | name>[/bold cyan] to open a session.[/dim]\n")

    def cmd_chat_history(self) -> None:
        """Display message dialogue history for the active chat session."""
        if not self.chat_manager or not self.active_chat_session:
            self.console.print("[yellow]No active chat session.[/yellow]\n")
            return

        messages = self.chat_manager.get_messages(self.active_chat_session.id)
        if not messages:
            self.console.print(f"[dim]Session '{self.active_chat_session.name}' has no messages yet. Type a question to begin.[/dim]\n")
            return

        self.console.print(f"\n[bold cyan]History for '{self.active_chat_session.name}' ({len(messages)} messages):[/bold cyan]\n")
        for msg in messages:
            if msg.role == "user":
                self.console.print(f"[bold bright_white]User:[/bold bright_white] {msg.content}")
            else:
                self.console.print(f"[bold cyan]AI ({self.active_chat_session.active_skill}):[/bold cyan]\n{msg.content}")
                if msg.citations:
                    self.console.print(f"[dim]Citations: {', '.join(msg.citations)}[/dim]")
            self.console.print()

    def cmd_chat_export(self) -> None:
        """Export current chat session to Markdown."""
        if not self.chat_manager or not self.active_chat_session or not self.workspace:
            self.console.print("[yellow]No active chat session to export.[/yellow]\n")
            return

        drafts_dir = self.workspace.root_dir / "drafts"
        dest_file = drafts_dir / f"{self.active_chat_session.name}.md"
        out_path = self.chat_manager.export_to_markdown(self.active_chat_session.id, dest_file)
        self.console.print(f"[bold green][✓] Chat dialogue exported to Markdown:[/bold green] [bold cyan]{out_path}[/bold cyan]\n")

    def cmd_vault(self) -> None:
        """Display list of papers in the local vault with active selection state."""
        if not self.workspace:
            self.console.print("[dim]No active workspace.[/dim]\n")
            return

        stems = self._get_cached_stems()
        if not stems:
            self.console.print("[dim]Project vault is currently empty. Use [bold cyan]/search[/bold cyan] or [bold cyan]/parse <file.pdf>[/bold cyan] to add papers.[/dim]\n")
            return

        self.console.print(f"\n[bold bright_white]Indexed Research Papers ({len(stems)} total, {len(self.active_documents)} active)[/bold bright_white]")
        table = Table(
            box=None,
            padding=(0, 2),
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("#", justify="right", style="dim", width=4)
        table.add_column("Active", justify="center", width=8)
        table.add_column("Cite Key", style="bold white", overflow="fold")
        table.add_column("Title", style="white")
        table.add_column("Year", justify="center", width=6, style="dim")
        table.add_column("Authors", style="dim")

        for idx, stem in enumerate(stems, 1):
            is_active = stem in self.active_documents
            active_marker = "[bold cyan][✓][/bold cyan]" if is_active else "[dim][ ][/dim]"

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
        self.console.print(f"\n[bold bright_white]Paper Preview: {target_stem}[/bold bright_white]")
        self.console.print(preview)
        self.console.print()

    def cmd_search(self, arg: str) -> None:
        """Search academic literature asynchronously and store results for download."""
        if not arg:
            self.console.print("[yellow]Usage: /search <keywords> [--limit N][/yellow]\n")
            return

        limit = 5
        clean_arg = arg.strip()
        if "--limit" in clean_arg:
            match = re.search(r"--limit\s+(\d+)", clean_arg)
            if match:
                limit = int(match.group(1))
                clean_arg = re.sub(r"--limit\s+\d+", "", clean_arg).strip()
        query = clean_arg.strip().strip("'\"")

        with self.console.status(f"[cyan]Searching literature across Semantic Scholar & ArXiv for '{query}'...[/cyan]"):
            try:
                results = asyncio.run(self._execute_search(query, limit))
                self.last_search_results = results
            except Exception as exc:
                self.console.print(f"[bold red][x] Search failed:[/bold red] {exc}\n")
                if not getattr(self, "_is_recovering", False) and self.workspace and self.engine:
                    self._is_recovering = True
                    try:
                        self.console.print("[bold cyan]●[/bold cyan] [bold bright_white]Menganalisis kendala:[/bold bright_white] [dim]Melaporkan kendala pencarian ke AI...[/dim]\n")
                        self.handle_chat(
                            f"Terjadi kesalahan saat mencari literatur '{query}': {exc}. "
                            f"Evaluasi kendala ini dan rekomendasikan strategi alternatif atau periksa paper di vault dengan /vault list."
                        )
                    finally:
                        self._is_recovering = False
                return

        if not results:
            self.console.print(f"[yellow]No academic results found for '{query}'.[/yellow]\n")
            if not getattr(self, "_is_recovering", False) and self.workspace and self.engine:
                self._is_recovering = True
                try:
                    self.console.print("[bold cyan]●[/bold cyan] [bold bright_white]Menganalisis kendala:[/bold bright_white] [dim]Meminta AI mengevaluasi pencarian dan menyusun kata kunci alternatif...[/dim]\n")
                    self.handle_chat(
                        f"Pencarian literatur dengan kata kunci '{query}' tidak menemukan hasil paper di repositori akademik terbuka (Semantic Scholar & ArXiv). "
                        f"Tolong evaluasi penyebabnya (apakah kata kunci terlalu spesifik/panjang), "
                        f"rekomendasikan kata kunci pencarian alternatif yang lebih efektif menggunakan blok ```scrpcy, "
                        f"atau sarankan untuk memeriksa koleksi paper yang sudah ada di vault dengan /vault list."
                    )
                finally:
                    self._is_recovering = False
            return

        self.console.print(f"\n[bold bright_white]Literature Search Results for '{query}' ({len(results)} found)[/bold bright_white]")
        table = Table(
            box=None,
            padding=(0, 2),
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("#", justify="right", style="dim", width=4)
        table.add_column("PDF Status", justify="center", width=14)
        table.add_column("Title", style="bold white")
        table.add_column("Year", justify="center", width=6, style="dim")
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
                    import pymupdf
                    import pymupdf4llm

                    with pymupdf.open(downloaded_pdf) as doc:
                        clean_body = pymupdf4llm.to_markdown(doc)
                except Exception:
                    clean_body = f"# {target_paper.title}\n\n{target_paper.abstract or ''}\n"

                meta = DocumentMetadata(
                    title=target_paper.title,
                    authors=target_paper.authors,
                    year=target_paper.year,
                    doi=target_paper.doi,
                    cite_key=target_paper.cite_key,
                )
                meta.ensure_cite_key()
                target_cache_name = f"{downloaded_pdf.stem}.md"
                cache_path = self.workspace.save_to_cache(
                    target_cache_name,
                    f"{meta.to_frontmatter()}\n{clean_body.strip()}\n",
                )
                parsed = ParsedDocument(
                    file_path=downloaded_pdf,
                    cache_path=cache_path,
                    metadata=meta,
                    markdown=f"{meta.to_frontmatter()}\n{clean_body.strip()}\n",
                    body_markdown=clean_body,
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
            box=None,
            padding=(0, 2),
            show_header=True,
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
            self.console.print("[dim]Switch persona: [bold cyan]/skill <# | name>[/bold cyan] (e.g. [bold cyan]/skill 2[/bold cyan])[/dim]")
        else:
            self.console.print("[dim]Activate persona inside a project: open or create a project, then run [bold cyan]/skill <# | name>[/bold cyan][/dim]")
        self.console.print(f"[dim]Add custom persona: Create a markdown file (.md) in [bold white]{self.skills_loader.skills_dir}[/bold white][/dim]\n")

    def cmd_skills(self) -> None:
        """List or interactively select research persona skills."""
        if sys.stdin.isatty():
            skills = self.skills_loader.list_skills()
            chosen = interactive_skill_selector(self.active_skill_name, skills, console=self.console)
            if chosen:
                self.active_skill_name = chosen
                if self.active_chat_session and self.chat_manager:
                    self.chat_manager.update_session_skill(self.active_chat_session.id, chosen)
                self.console.print(f"\n[bold green][✓] Switched active persona to:[/bold green] [bold cyan]{chosen}[/bold cyan]\n")
                return
        self.render_skills_catalog(is_project_mode=True)

    def cmd_skill(self, arg: str) -> None:
        """Switch current active persona skill, or view catalog if no argument given."""
        arg = arg.strip()
        if not arg:
            if sys.stdin.isatty():
                skills = self.skills_loader.list_skills()
                chosen = interactive_skill_selector(self.active_skill_name, skills, console=self.console)
                if chosen:
                    self.active_skill_name = chosen
                    if self.active_chat_session and self.chat_manager:
                        self.chat_manager.update_session_skill(self.active_chat_session.id, chosen)
                    self.console.print(f"\n[bold green][✓] Switched active persona to:[/bold green] [bold cyan]{chosen}[/bold cyan]\n")
                    return
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
        """Alias for /models use <model>."""
        if not arg:
            self.cmd_models("list")
        else:
            self.cmd_models(f"use {arg}")

    def cmd_style(self, arg: str) -> None:
        """Alias for /settings set-citation <style>."""
        if not arg:
            self.cmd_settings("show")
        else:
            self.cmd_settings(f"set-citation {arg}")

    def cmd_models(self, arg: str = "") -> None:
        """Manage LLM models, per-project model assignments, API keys, 9Router, and tokens."""
        parts = shlex.split(arg) if arg else []
        sub = parts[0].lower() if parts else "list"

        if sub in ("list", "show", "ls", "switch") or not parts:
            r9_base = self.config.api_bases.get("9router", "http://localhost:20128/v1")
            r9_key = self.config.get_api_key("9router") or detect_9router_key()
            live_r9_models = fetch_9router_models(r9_base)

            # If interactive terminal, launch the Antigravity TUI selector
            if sys.stdin.isatty() and (not parts or sub in ("switch", "list")):
                all_options: list[str] = [f"9router/{m}" for m in live_r9_models]
                presets = [
                    "gemini/gemini-2.5-flash",
                    "gemini/gemini-2.5-pro",
                    "anthropic/claude-3-5-sonnet",
                    "deepseek/deepseek-chat",
                    "openai/gpt-4o",
                ]
                for p in presets:
                    if p not in all_options:
                        all_options.append(p)

                curr_m = self.get_active_project_model()
                curr_eff = self.get_active_project_effort()
                chosen = interactive_model_selector(
                    current_model=curr_m,
                    current_effort=curr_eff,
                    available_models=all_options,
                    console=self.console,
                )
                if chosen:
                    chosen_m, chosen_eff = chosen
                    target_proj = self.current_project or self.project_manager.get_active_project()
                    if target_proj:
                        self.project_manager.set_project_model(target_proj.name, chosen_m)
                        self.project_manager.set_project_effort(target_proj.name, chosen_eff)
                        if self.current_project:
                            self.current_project.model = chosen_m
                        if self.engine:
                            self.engine.default_model = chosen_m
                        m_label = _format_model_display_name(chosen_m)
                        self.console.print(
                            f"\n[bold green][✓] Active model updated to:[/bold green] [bold cyan]{m_label}[/bold cyan] [dim]({chosen_eff} effort)[/dim]\n"
                        )
                    else:
                        self.config.default_model = chosen_m
                        self.config.save()
                        if self.engine:
                            self.engine.default_model = chosen_m
                        self.console.print(
                            f"\n[bold green][✓] Updated global fallback model to:[/bold green] [bold cyan]{chosen_m}[/bold cyan]\n"
                        )
                    return

            if live_r9_models:
                r9_status = f"[bold green][Online - {len(live_r9_models)} models detected][/bold green]"
            elif r9_key:
                r9_status = "[bold green][Configured][/bold green]"
            else:
                r9_status = "[dim][Standby / Not Configured][/dim]"

            all_keys = []
            for provider, val in self.config.api_keys.items():
                if val:
                    masked = f"{val[:3]}...{val[-4:]}" if len(val) > 8 else "***"
                    all_keys.append(f"{provider} ({masked})")
            if detect_9router_key() and "9router" not in [k for k, v in self.config.api_keys.items() if v]:
                all_keys.append("9router (Auto-detected)")
            keys_display = ", ".join(all_keys) if all_keys else "[dim]No API keys stored locally[/dim]"

            self.console.print()
            table = Table(
                title="scrpcy Model & Provider Inventory",
                box=None,
                padding=(0, 2),
                show_header=True,
                header_style="bold bright_cyan",
            )
            table.add_column("Property", style="bold cyan")
            table.add_column("Configuration / Status", style="white")

            if self.current_project:
                active_m = self.get_active_project_model()
                eff = self.get_active_project_effort()
                table.add_row("Active Project", f"[bold green]{self.current_project.name}[/bold green]")
                table.add_row("Project Model", f"[bold bright_cyan]{active_m}[/bold bright_cyan] [dim]({eff} effort)[/dim]")
                if self.workspace:
                    p_tok = TokenTracker.get_project_tokens(self.workspace.root_dir)
                    table.add_row("Project Token Usage", f"{p_tok.total_tokens:,} tokens (In: {p_tok.input_tokens:,}, Out: {p_tok.output_tokens:,})")
            else:
                projects = self.project_manager.list_projects()
                if projects:
                    proj_lines = [f"{p.name}: [cyan]{p.model or 'Not assigned'}[/cyan]" for p in projects[:5]]
                    table.add_row("Project Models", ", ".join(proj_lines))
                else:
                    table.add_row("Project Models", "[dim]None registered yet[/dim]")

            table.add_row("9Router Local Gateway", f"{r9_base} {r9_status}")
            table.add_row("Configured API Keys", keys_display)
            self.console.print(table)
            self.console.print()

            # 1. Live 9Router Models Table (if 9Router is active)
            if live_r9_models:
                r9_table = Table(
                    title=f"Live 9Router Models Detected (at {r9_base})",
                    box=None,
                    padding=(0, 2),
                    show_header=True,
                    header_style="bold bright_green",
                )
                r9_table.add_column("#", justify="right", style="dim", width=3)
                r9_table.add_column("Model Identifier (for Scrpcy)", style="bold cyan")
                r9_table.add_column("Command to Use", style="yellow")

                for idx, m_id in enumerate(live_r9_models, 1):
                    cmd_example = f"models use 9router/{m_id}"
                    r9_table.add_row(str(idx), f"9router/{m_id}", cmd_example)

                self.console.print(r9_table)
                self.console.print()

            # 2. Supported Model Presets Catalog
            cat_table = Table(
                title="Katalog Model yang Didukung (Supported Model Catalog)",
                box=None,
                padding=(0, 2),
                show_header=True,
                header_style="bold cyan",
            )
            cat_table.add_column("Provider / Router", style="bold white", width=18)
            cat_table.add_column("Contoh Model Identifiers", style="cyan")
            cat_table.add_column("Keterangan & Kebutuhan Kunci", style="dim")

            cat_table.add_row("9Router (Local)", "9router/<model_id>", "Proxy lokal :20128, multi-provider otomatis")
            cat_table.add_row("Google Gemini", "gemini/gemini-2.5-flash, gemini/gemini-2.5-pro", "Gratis di aistudio.google.com (GEMINI_API_KEY)")
            cat_table.add_row("Anthropic Claude", "anthropic/claude-3-5-sonnet", "ANTHROPIC_API_KEY")
            cat_table.add_row("DeepSeek", "deepseek/deepseek-chat, deepseek/deepseek-reasoner", "DEEPSEEK_API_KEY")
            cat_table.add_row("OpenAI", "openai/gpt-4o, openai/gpt-4o-mini", "OPENAI_API_KEY")
            cat_table.add_row("Groq", "groq/llama-3.3-70b-versatile", "GROQ_API_KEY")
            cat_table.add_row("Ollama", "ollama/llama3.2", "Server Ollama lokal (:11434)")

            self.console.print(cat_table)
            self.console.print()

            # 3. Descriptive Commands Table
            cmd_table = Table(
                title="Panduan Perintah 'models' (Commands Guide)",
                box=None,
                padding=(0, 2),
                show_header=True,
                header_style="bold cyan",
            )
            cmd_table.add_column("Perintah", style="bold cyan", width=34)
            cmd_table.add_column("Deskripsi & Contoh", style="white")

            cmd_table.add_row("/models", "Buka menu pemilihan model interaktif (cursor & effort)")
            cmd_table.add_row("/models list", "Menampilkan katalog model teks lengkap")
            cmd_table.add_row("/models use <model> [project]", "Mengatur model LLM (contoh: /models use 9router/ag/gemini-3.8-flash)")
            cmd_table.add_row("/models 9router [port] [model]", "Menghubungkan 9Router lokal dan memindai model aktif")
            cmd_table.add_row("/models set-key <provider> <key>", "Menyimpan API key (contoh: /models set-key 9router <key>)")
            cmd_table.add_row("/models tokens [project]", "Melihat statistik token input, output, dan riwayat sintesis")
            cmd_table.add_row("/models status [project]", "Melihat status project dan model aktif")

            self.console.print(cmd_table)
            self.console.print()
            return

        if sub in ("tokens", "usage"):
            filter_proj = parts[1] if len(parts) > 1 else None
            tracker = TokenTracker()
            summary = tracker.get_summary()

            self.console.print()
            table = Table(
                title="Global Token Consumption Summary",
                box=None,
                padding=(0, 2),
                title_style="bold bright_white",
                header_style="bold cyan",
            )
            table.add_column("Metric", style="bold white", width=24)
            table.add_column("Count", justify="right", style="bold cyan")
            table.add_row("Total Input Tokens", f"{summary.total_input_tokens:,}")
            table.add_row("Total Output Tokens", f"{summary.total_output_tokens:,}")
            table.add_row("Grand Total Tokens", f"[bold bright_white]{summary.total_tokens:,}[/bold bright_white]")
            table.add_row("Total Requests", f"{summary.total_requests:,}")
            self.console.print(table)
            self.console.print()

            if summary.projects:
                p_table = Table(
                    title="Token Usage Breakdown by Project",
                    box=None,
                    padding=(0, 2),
                    title_style="bold bright_white",
                    header_style="bold cyan",
                )
                p_table.add_column("Project Name", style="bold white", width=22)
                p_table.add_column("Input", justify="right", style="cyan")
                p_table.add_column("Output", justify="right", style="cyan")
                p_table.add_column("Total", justify="right", style="bold white")
                p_table.add_column("Requests", justify="right", style="dim", width=10)
                p_table.add_column("Last Active (UTC)", style="dim", width=20)


                for p_name, p_sum in summary.projects.items():
                    if filter_proj and filter_proj.lower() not in p_name.lower():
                        continue
                    last_act = p_sum.last_active.replace("T", " ")[:19] if p_sum.last_active else "-"
                    p_table.add_row(
                        p_name,
                        f"{p_sum.input_tokens:,}",
                        f"{p_sum.output_tokens:,}",
                        f"{p_sum.total_tokens:,}",
                        f"{p_sum.request_count:,}",
                        last_act,
                    )
                self.console.print(p_table)
                self.console.print()
            elif self.workspace:
                p_tok = TokenTracker.get_project_tokens(self.workspace.root_dir)
                self.console.print(f"[bold green]Current Project Tokens:[/bold green] {p_tok.total_tokens:,} (In: {p_tok.input_tokens:,}, Out: {p_tok.output_tokens:,})\n")
            else:
                self.console.print("[dim]No token consumption recorded yet.[/dim]\n")

        elif sub in ("use", "set", "set-model"):
            if len(parts) < 2:
                self.console.print("[yellow]Usage: models use <model_name> [project_name][/yellow]\n")
                return
            model = parts[1]
            target_proj = parts[2] if len(parts) >= 3 else None

            if self.current_project:
                self.current_project.model = model
                self.project_manager.set_project_model(self.current_project.name, model)
                if self.engine:
                    self.engine.default_model = model
                self.console.print(f"[bold green][✓] Updated model for project '{self.current_project.name}' to:[/bold green] [bold cyan]{model}[/bold cyan]\n")
            elif target_proj:
                try:
                    p = self.project_manager.set_project_model(target_proj, model)
                    self.console.print(f"[bold green][✓] Updated model for project '{p.name}' to:[/bold green] [bold cyan]{model}[/bold cyan]\n")
                except Exception as exc:
                    self.console.print(f"[bold red][x] Error:[/bold red] {exc}\n")
            else:
                active_p = self.project_manager.get_active_project()
                if active_p:
                    self.project_manager.set_project_model(active_p.name, model)
                    self.console.print(f"[bold green][✓] Updated model for active project '{active_p.name}' to:[/bold green] [bold cyan]{model}[/bold cyan]\n")
                else:
                    self.config_manager.update(default_model=model)
                    self.console.print(f"[bold green][✓] Set default fallback model to:[/bold green] [bold cyan]{model}[/bold cyan]\n")

        elif sub == "9router":
            port = 20128
            chosen_model = None
            if len(parts) >= 2 and parts[1].isdigit():
                port = int(parts[1])
                if len(parts) >= 3:
                    chosen_model = parts[2]
            elif len(parts) >= 2:
                chosen_model = parts[1]

            base_url = f"http://localhost:{port}/v1"
            live_models = fetch_9router_models(base_url)
            auto_key = detect_9router_key()
            if auto_key:
                self.config_manager.config.set_api_key("9router", auto_key)
            self.config_manager.config.set_api_base("9router", base_url)

            if not chosen_model and live_models:
                chosen_model = f"9router/{live_models[0]}"

            if chosen_model:
                if self.current_project:
                    self.current_project.model = chosen_model
                    self.project_manager.set_project_model(self.current_project.name, chosen_model)
                    if self.engine:
                        self.engine.default_model = chosen_model
                else:
                    active_p = self.project_manager.get_active_project()
                    if active_p:
                        self.project_manager.set_project_model(active_p.name, chosen_model)

            self.config_manager.save()
            self.config = self.config_manager.config
            detected_str = f" ({len(live_models)} live models detected)" if live_models else ""
            model_disp = f" Assigned project model: '{chosen_model}'." if chosen_model else ""
            self.console.print(f"[bold green][✓] Connected to 9Router at {base_url}{detected_str}.{model_disp}[/bold green]\n")

        elif sub == "status":
            target = [parts[1]] if len(parts) > 1 else None
            if self.current_project:
                self.cmd_project_status()
            else:
                self.dashboard_status(target)

        elif sub == "set-key":
            if len(parts) < 3:
                self.console.print("[yellow]Usage: models set-key <provider> <key>[/yellow]\n")
                return
            provider, key = parts[1], parts[2]
            self.config_manager.config.set_api_key(provider, key)
            self.config_manager.save()
            self.config = self.config_manager.config
            self.console.print(f"[bold green][✓] Successfully saved API key for provider '{provider}'.[/bold green]\n")

        elif sub == "set-base":
            if len(parts) < 3:
                self.console.print("[yellow]Usage: models set-base <provider> <url>[/yellow]\n")
                return
            provider, url = parts[1], parts[2]
            self.config_manager.config.set_api_base(provider, url)
            self.config_manager.save()
            self.config = self.config_manager.config
            self.console.print(f"[bold green][✓] Set custom base URL for '{provider}' -> {url}[/bold green]\n")

        else:
            self.console.print("[yellow]Usage: models [list | use <model> [project] | set-key <provider> <key> | 9router [port] [model] | tokens | status][/yellow]\n")

    def cmd_settings(self, arg: str = "") -> None:
        """Display or update general workspace and citation settings."""
        parts = shlex.split(arg) if arg else []
        sub = parts[0].lower() if parts else "show"

        if sub in ("show", "list", "ls") or not parts:
            self.console.print("\n[bold bright_white]Workspace & Citation Settings[/bold bright_white]")
            s_table = Table(box=None, padding=(0, 2), show_header=False)
            s_table.add_column("Property", style="dim", width=18)
            s_table.add_column("Value", style="white")
            s_table.add_row("Workspace Path", str(self.config.workspace_path))
            s_table.add_row("Skills Dir", str(self.config.skills_dir))
            s_table.add_row("Citation Style", self.config.citation.style.upper())
            s_table.add_row("Include DOI", str(self.config.citation.include_doi))
            s_table.add_row("Max Evidence", f"{self.config.citation.max_evidence_chunks} chunks")
            self.console.print(s_table)
            self.console.print("[dim]Commands: settings set-citation <apa|ieee> | settings set-workspace <path>[/dim]\n")
            return

        if sub in ("set-citation", "citation", "style"):
            if len(parts) < 2:
                self.console.print("[yellow]Usage: settings set-citation <apa | ieee>[/yellow]\n")
                return
            new_style = parts[1].lower().strip()
            if new_style not in ("apa", "ieee", "bibtex"):
                self.console.print("[yellow]Supported styles: apa, ieee, bibtex[/yellow]\n")
                return
            self.config_manager.config.citation.style = new_style
            self.config_manager.save()
            self.config = self.config_manager.config
            self.console.print(f"[bold green][✓] Citation formatting style set to:[/bold green] [bold cyan]{new_style.upper()}[/bold cyan]\n")
        elif sub in ("set-workspace", "workspace"):
            if len(parts) < 2:
                self.console.print("[yellow]Usage: settings set-workspace <path>[/yellow]\n")
                return
            new_path = parts[1].strip()
            self.config_manager.config.workspace_path = new_path
            self.config_manager.save()
            self.config = self.config_manager.config
            self.console.print(f"[bold green][✓] Workspace path set to:[/bold green] [bold cyan]{new_path}[/bold cyan]\n")
        else:
            self.console.print("[yellow]Usage: settings [show | set-citation <apa|ieee> | set-workspace <path>][/yellow]\n")

    def cmd_config(self, arg: str) -> None:
        """Alias for backward compatibility directing to cmd_models."""
        self.cmd_models(arg)

    def _background_docx_worker(self, task_id: str, filename: str) -> str:
        """Background worker compiling conversation synthesis into Word (.docx) manuscript."""
        self.task_manager.log(task_id, f"Initializing manuscript compilation: {filename}")
        drafts_dir = self.workspace.root_dir / "drafts"
        drafts_dir.mkdir(parents=True, exist_ok=True)
        dest_path = drafts_dir / filename

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

        self.task_manager.log(task_id, f"Formatting {len(sections)} sections and {len(references)} citations via WordCraft...")
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
        summary = f"Generated manuscript: {dest_path.name}"
        self.task_manager.log(task_id, f"[✓] {summary}")
        return summary

    def cmd_docx(self, arg: str, in_background: bool = False) -> None:
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

        if in_background or self.sub_mode == "chat":
            task = self.task_manager.submit(
                name=f"WordCraft Export: {filename}",
                task_type="convert",
                target_fn=self._background_docx_worker,
                filename=filename,
            )
            self.console.print()
            self.console.print(
                f"[bold green][✓] Proses kompilasi Word (.docx) dikirim ke antrean Processing:[/bold green] [bold cyan]{task.id}[/bold cyan]"
            )
            self.console.print(f" [dim]• File Target : drafts/{filename}[/dim]")
            self.console.print(f" [dim]• Gaya Sitasi : {self.config.citation.style.upper()}[/dim]")
            self.console.print(" [dim]• Ketik [bold cyan]/proc[/bold cyan] untuk memantau status pembuatan berkas.[/dim]\n")
            return

        with self.console.status("[cyan]Compiling manuscript and formatting citations via WordCraft...[/cyan]"):
            try:
                self._background_docx_worker("sync", filename)
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

        # Check if user query in chat explicitly contains a Google Drive link or download intent
        gdrive_id = extract_drive_file_id(query)
        if gdrive_id and ("download" in query.lower() or "unduh" in query.lower() or "ingest" in query.lower() or "masuk" in query.lower() or query.strip().startswith("http")):
            self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Mendeteksi link Google Drive[/bold bright_white] [cyan]ID: {gdrive_id[:12]}...[/cyan]")
            task = self.task_manager.submit(
                name=f"Google Drive Ingest: {gdrive_id[:12]}...",
                task_type="gdrive",
                target_fn=self._background_gdrive_worker,
                file_id=gdrive_id,
            )
            self.console.print(
                f"[bold green][✓] Proses download Google Drive dikirim ke antrean Processing:[/bold green] [bold cyan]{task.id}[/bold cyan]"
            )
            self.console.print(" [dim]• File akan otomatis diunduh, dikonversi ke Markdown, dan diindeks ke dalam vault.[/dim]")
            self.console.print(" [dim]• Ketik [bold cyan]/proc[/bold cyan] untuk memantau status atau log.[/dim]\n")
            return

        if self.vault_manager:
            try:
                self.vault_manager.reload_vault()
            except Exception:
                pass
        self._sync_vault_cache()

        active_paths = [self.workspace.get_cache_path(f"{s}.md") for s in self.active_documents]
        existing_paths = [p for p in active_paths if p.exists()]

        evidence = self.engine.evidence_synthesizer.retrieve_evidence(
            query=query,
            documents=existing_paths,
            max_chunks=self.config.citation.max_evidence_chunks,
        )

        # High Token Consumption Warning & Confirmation (if > 5 chunks)
        if len(evidence) > 5:
            sources = sorted(list({c.source for c in evidence}))
            evidence_tokens = sum(c.token_estimate or estimate_tokens(c.text) for c in evidence)
            est_total_input = estimate_tokens(query) + 250 + evidence_tokens

            choice = interactive_token_warning_confirm(
                chunk_count=len(evidence),
                doc_count=len(sources),
                token_estimate=est_total_input,
                console=self.console,
            )

            if choice == "cancel":
                self.console.print("[dim]● Pertanyaan dibatalkan oleh pengguna (hemat token).[/dim]\n")
                return
            elif choice == "limit":
                evidence = evidence[:5]
                self.console.print("[dim]● Dibatasi ke 5 chunks teratas untuk penghematan token.[/dim]")

        # 1. Action Indicator: Membaca
        if evidence:
            sources = sorted(list({c.source for c in evidence}))
            for src in sources:
                count = sum(1 for c in evidence if c.source == src)
                self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Membaca[/bold bright_white] [cyan]cache/{src}[/cyan] [dim]({count} evidence chunks)[/dim]")
        elif existing_paths:
            self.console.print("[bold cyan]●[/bold cyan] [dim]Membaca: Tidak ada chunk relevan dari paper aktif[/dim]")
        else:
            self.console.print("[bold cyan]●[/bold cyan] [dim]Membaca: Tidak ada paper aktif di vault (mode sintesis umum)[/dim]")

        model_name = self.get_active_project_model()
        api_key = self.config.get_api_key_for_model(model_name)
        effort = self.get_active_project_effort()
        model_label = _format_model_display_name(model_name)

        if not api_key:
            provider_hint = "9router" if model_name.startswith("9router/") else (model_name.split("/")[0] if "/" in model_name else "gemini")
            self.console.print()
            self.console.print(f"[bold yellow][!] No API key configured for model '{model_name}'.[/bold yellow]")
            self.console.print(f"  To configure API key, run:")
            self.console.print(f"    [bold cyan]/models set-key {provider_hint} <YOUR_API_KEY>[/bold cyan]")
            self.console.print(f"  [dim]Tips: Jika menggunakan 9Router lokal, periksa http://localhost:20128[/dim]\n")
            return

        session = self._ensure_active_chat_session()
        if self.chat_manager:
            self.chat_manager.add_message(session.id, "user", query)

        # 2. Action Indicator: Berpikir
        self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Berpikir[/bold bright_white] [dim]Menganalisis literatur dan menyusun sintesis ilmiah...[/dim]")
        self.console.print(f"\n[bold cyan]✦ {model_label}[/bold cyan] [dim]({self.active_skill_name} • {effort} effort)[/dim]\n")

        accumulated_tokens: list[str] = []
        start_time = time.time()
        try:
            async def _stream_chat() -> None:
                async for token in self.engine.stream_query(
                    query=query,
                    documents=existing_paths,
                    skill_name=self.active_skill_name,
                    model=model_name,
                    max_evidence_chunks=self.config.citation.max_evidence_chunks,
                ):
                    accumulated_tokens.append(token)

            asyncio.run(_stream_chat())
            elapsed = time.time() - start_time

            full_response = "".join(accumulated_tokens)

            # 3. Clean academic Markdown with LaTeX math beautification
            render_academic_markdown(full_response, console=self.console)
            self.console.print()

            # 4. Accurate token consumption telemetry
            evidence_tokens = sum(c.token_estimate or estimate_tokens(c.text) for c in evidence)
            input_tokens = estimate_tokens(query) + 250 + evidence_tokens
            output_tokens = estimate_tokens(full_response)
            total_tokens = input_tokens + output_tokens
            style_name = self.config.citation.style.upper()
            self.console.print(
                f"[dim]── Tokens: {input_tokens:,} in · {output_tokens:,} out · {total_tokens:,} total ({model_label}) • {elapsed:.1f}s • {style_name} style ──[/dim]\n"
            )

            if self.chat_manager:
                self.chat_manager.add_message(session.id, "assistant", full_response)
            self.conversation_history.append({
                "query": query,
                "response": full_response,
            })

            # 5. Suggested actions & Action Indicator: Mengubah / Menulis
            self._handle_suggested_actions(full_response)

        except (KeyboardInterrupt, asyncio.CancelledError):
            self.console.print("\n\n[bold yellow]● Pertanyaan dibatalkan oleh pengguna (Ctrl+C).[/bold yellow]\n")
            if accumulated_tokens and self.chat_manager:
                partial_resp = "".join(accumulated_tokens).strip()
                self.chat_manager.add_message(session.id, "assistant", f"{partial_resp} [Dibatalkan]")
            return
        except Exception as exc:
            self.console.print(f"\n[bold red][x] Generation error:[/bold red] {exc}\n")

    def _handle_suggested_actions(self, response_text: str) -> None:
        """Detect and offer interactive execution for Scrpcy action blocks in AI response."""
        pattern = re.compile(r"```scrpcy\r?\n(.*?)\r?\n```", re.DOTALL | re.IGNORECASE)
        matches = pattern.findall(response_text)
        if not matches:
            return

        actions = [cmd.strip() for m in matches for cmd in m.splitlines() if cmd.strip() and not cmd.strip().startswith("#")]
        if not actions:
            return

        # Deduplicate while preserving order
        unique_actions: list[str] = []
        for a in actions:
            if a not in unique_actions:
                unique_actions.append(a)

        for action in unique_actions:
            # Action Pill: Mengubah / Menulis
            if "docx " in action:
                doc_arg = action.split("docx", 1)[1].strip()
                self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Menulis[/bold bright_white] [cyan]drafts/{doc_arg}[/cyan]")
            elif "vault add " in action:
                v_arg = action.split("vault add", 1)[1].strip()
                self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Mengubah[/bold bright_white] [cyan]vault/{v_arg}[/cyan]")

            confirmed = interactive_action_confirm(action, console=self.console)
            if confirmed:
                self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Menjalankan aksi:[/bold bright_white] [cyan]{action}[/cyan]\n")
                cmd_to_run = action if action.startswith("/") else f"/{action}"
                self.handle_project_command(cmd_to_run)
            else:
                self.console.print(f"[dim]Aksi '{action}' dilewati.[/dim]\n")
