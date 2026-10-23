"""Module part of scrpcy studio."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
from pathlib import Path
import re
import shlex
import sys
import threading
import time
from typing import Any, Callable
import webbrowser

from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from scrpcy.core.chats import ChatManager, ChatSession
from scrpcy.core.config import ConfigManager, ScrpcyConfig, detect_9router_key, fetch_9router_models
from scrpcy.core.contracts import AdaptiveCriticVerdict, TaskContract, TaskContractExtractor
from scrpcy.core.critic import CritiqueReport, DevilsAdvocate
from scrpcy.core.engine import LLMEngine, estimate_tokens
from scrpcy.core.gdrive import GoogleDriveManager, extract_drive_file_id
from scrpcy.core.logger import StudioLogger, get_studio_logger
from scrpcy.core.mcp_client import WordCraftClient
from scrpcy.core.models import DocumentMetadata, ParsedDocument
from scrpcy.core.parser import DocumentParser
from scrpcy.core.polisher import AcademicPolisher, PolishResult
from scrpcy.core.project import ProjectInfo, ProjectManager
from scrpcy.core.references import append_vault_bibliography
from scrpcy.core.scout import LiteratureScout, ScoutQuery, ScoutResult
from scrpcy.core.search import PaperSearcher, SearchResult, sanitize_filename
from scrpcy.core.skills import SkillsLoader
from scrpcy.core.tasks import TaskInfo, TaskManager
from scrpcy.core.tracker import TokenTracker
from scrpcy.core.vault import VaultItem, VaultManager
from scrpcy.core.verifier import FactChecker, VerificationReport
from scrpcy.core.workspace import WorkspaceManager

from scrpcy.studio.signals import (
    SCRPCY_LOGO,
    SCRPCY_VERSION,
    SessionCloseSignal,
    SessionDetachSignal,
    _AgentProgressTracker,
)
from scrpcy.ui.formatters import render_academic_markdown
from scrpcy.ui.interactive import (
    _clear_lines,
    _format_model_display_name,
    _render_diff_inplace,
    init_windows_terminal,
    interactive_action_confirm,
    interactive_carousel_table,
    interactive_confirm,
    interactive_detail_viewer,
    interactive_help_view,
    interactive_input_prompt,
    interactive_item_browser,
    interactive_menu,
    interactive_model_selector,
    interactive_skill_selector,
    interactive_token_warning_confirm,
)


class StudioDashboardMixin:
    def run_interactive_dashboard(self) -> str | None:

        """Dashboard interactive menu without typing."""

        while self.current_project is None:

            projects = self.project_manager.list_projects()

            p_count = len(projects)



            header = Text()

            header.append("scrpcy", style="bold cyan")

            header.append(" • ", style="dim")

            header.append(f"dashboard ({p_count} registered projects)", style="dim")



            options = [

                ("projects", f"Projects ({p_count} terdaftar)"),

                ("settings", "Settings & Models"),

                ("exit", "Exit"),

            ]



            choice = interactive_menu(

                options=options,

                header=header,

                console=self.console,

                footer_hint="↑/↓ Navigate · enter Select · 1-3 Quick jump · esc Exit",

                allow_back=True,

            )



            if choice in ("exit", None):

                self.console.print("\n[bold cyan]Closing scrpcy research studio. Goodbye![/bold cyan]")

                return "exit"



            if choice == "projects":

                self.run_interactive_dashboard_projects()

            elif choice == "settings":

                self.run_interactive_dashboard_settings()



        return None

    def run_interactive_dashboard_projects(self) -> None:
        """Browse and select or create projects with zero terminal stacking."""
        while True:

            projects = self.project_manager.list_projects()

            if not projects:

                header = Text("Belum ada master project terdaftar.", style="dim")

                opts = [

                    ("new", "Create New Project"),

                    ("back", "Back to Dashboard"),

                ]

                ch = interactive_menu(opts, header=header, console=self.console)

                if ch == "new":

                    self.interactive_create_project()

                else:

                    return

            else:

                columns = [

                    {"name": "#", "justify": "right", "width": 3, "style": "dim"},

                    {"name": "Project Name", "style": "bold white", "width": 20},

                    {"name": "Location", "style": "dim", "width": 38},

                    {"name": "Model", "style": "cyan", "width": 26},

                    {"name": "Last Opened", "style": "dim", "width": 16},

                ]



                def _row_builder(idx: int, p: ProjectInfo, is_selected: bool) -> list[Any]:

                    m_label = _format_model_display_name(p.model or self.config.default_model)

                    last_str = p.last_opened[:16].replace("T", " ") if p.last_opened else ""

                    arrow = "> " if is_selected else "  "

                    p_name_text = Text()

                    p_name_text.append(arrow, style="bold cyan" if is_selected else "dim")

                    p_name_text.append(p.name, style="bold bright_white" if is_selected else "white")

                    return [

                        str(idx + 1),

                        p_name_text,

                        str(p.path)[:36],

                        m_label[:24],

                        last_str,

                    ]



                header = Text()

                header.append("Daftar Master Projects", style="bold bright_white")

                header.append(f" ({len(projects)} projects)", style="dim")



                actions = [

                    ("open", "Open Project", "enter"),

                    ("new", "New Project", "n"),

                    ("delete", "Delete Project", "d"),

                    ("back", "Back", "esc"),

                ]



                import scrpcy.cli_studio as _cli_studio
                action, idx, selected_p = _cli_studio.interactive_item_browser(

                    title=header,

                    columns=columns,

                    items=projects,

                    row_builder=_row_builder,

                    actions=actions,

                    console=self.console,

                    empty_message="Belum ada project.",

                )



                if action in ("back", "esc"):

                    return

                elif action == "open" and selected_p:

                    self.open_project(selected_p, show_banner=False)

                    return

                elif action == "new":

                    self.interactive_create_project()

                    if self.current_project:

                        return

                elif action == "delete" and selected_p:
                    confirmed = _cli_studio.interactive_confirm(
                        f"Hapus registrasi project '{selected_p.name}' dari Scrpcy?",
                        default=True,
                        yes_label="Ya, hapus project",
                        no_label="Batal",
                        console=self.console,
                    )
                    if confirmed:
                        if self.current_project and self.current_project.name == selected_p.name:
                            self.current_project = None
                            self.workspace = None
                            self.vault_manager = None
                            self.active_documents.clear()
                        removed = self.project_manager.remove_project(selected_p.name)
                        if removed:
                            self.console.print(f"[bold green][✓] Registrasi project '{selected_p.name}' berhasil dihapus dari Scrpcy.[/bold green]")
                            self.console.print(f"[dim]Catatan: File asli di '{selected_p.path}' tetap aman dan tidak dihapus dari disk.[/dim]\n")

    def interactive_create_project(self) -> None:

        """Prompt user cleanly for project name."""

        try:

            self.console.print()

            p_name = self.console.input("[bold cyan]Nama project baru: [/bold cyan]").strip()

            if not p_name:

                return

            p_info = self.project_manager.create_project(name=p_name)

            self.open_project(p_info, show_banner=False)

        except (KeyboardInterrupt, EOFError):

            self.console.print()

            return

    def run_interactive_dashboard_settings(self) -> None:

        """Dashboard settings menu."""

        while True:

            cur_m = _format_model_display_name(self.config.default_model)

            header = Text()

            header.append("Scrpcy Global Settings\n", style="bold bright_white")

            header.append(f"Default Model: {cur_m} · Citation: {self.config.citation.style}", style="dim")



            opts = [

                ("model", f"Switch Default Model ({cur_m})"),

                ("citation", f"Citation Style ({self.config.citation.style.upper()})"),

                ("keys", "Manage API Keys"),

                ("back", "Back"),

            ]



            ch = interactive_menu(opts, header=header, console=self.console)

            if ch in ("back", None):

                return

            if ch == "model":

                self.cmd_models("")

            elif ch == "citation":

                self.run_interactive_citation_settings()

            elif ch == "keys":

                self.cmd_models("list")

    def run_interactive_citation_settings(self) -> None:

        """Select citation style."""

        opts = [

            ("apa", "APA Style (American Psychological Association)"),

            ("ieee", "IEEE Style (Numbered [1], [2])"),

            ("bibtex", "BibTeX Citation Keys ([Vaswani2017])"),

            ("back", "Back"),

        ]

        ch = interactive_menu(opts, header="Pilih Gaya Sitasi:", console=self.console)

        if ch and ch != "back":

            self.cmd_settings(f"set citation.style {ch}")

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
            except (KeyboardInterrupt, EOFError, OSError):
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
            except (KeyboardInterrupt, EOFError, OSError):
                self.console.print("\n[yellow]Project creation canceled.[/yellow]\n")
                return

            if choice.lower() in ("q", "quit", "cancel"):
                self.console.print("[yellow]Project creation canceled.[/yellow]\n")
                return

            if choice == "2":
                try:
                    custom_path = self.console.input("[bold cyan]path folder tujuan[/bold cyan] [bold green]❯[/bold green] ").strip()
                except (KeyboardInterrupt, EOFError, OSError):
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
            except (KeyboardInterrupt, EOFError, OSError):
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
        except (KeyboardInterrupt, EOFError, OSError):
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
        except (KeyboardInterrupt, EOFError, OSError):
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

