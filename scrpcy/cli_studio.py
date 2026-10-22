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
from typing import Any, Callable
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
from scrpcy.core.search import PaperSearcher, SearchResult, sanitize_filename
from scrpcy.core.skills import SkillsLoader
from scrpcy.core.tasks import TaskInfo, TaskManager
from scrpcy.core.tracker import TokenTracker
from scrpcy.core.vault import VaultItem, VaultManager
from scrpcy.core.verifier import FactChecker, VerificationReport
from scrpcy.core.critic import DevilsAdvocate, CritiqueReport
from scrpcy.core.intent import IntentRouter, ResearchIntent, IntentAnalysisResult
from scrpcy.core.polisher import AcademicPolisher, PolishResult
from scrpcy.core.logger import StudioLogger, get_studio_logger
from scrpcy.core.references import append_vault_bibliography
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
    _clear_lines,
    init_windows_terminal,
    _format_model_display_name,
    interactive_action_confirm,
    interactive_confirm,
    interactive_carousel_table,
    interactive_detail_viewer,
    interactive_help_view,
    interactive_input_prompt,
    interactive_item_browser,
    interactive_menu,
    interactive_model_selector,
    interactive_skill_selector,
    interactive_token_warning_confirm,
)
from scrpcy.ui.formatters import render_academic_markdown

class SessionDetachSignal(Exception):
    """User requested to detach the active chat session to background."""
    pass


class SessionCloseSignal(Exception):
    """User requested to close/terminate the active chat session."""
    pass

class _AgentProgressTracker:
    def __init__(self, init_msg: str) -> None:
        self.text = init_msg

    def set(self, msg: str) -> None:
        self.text = msg

    def __str__(self) -> str:
        return self.text

    def __call__(self) -> str:
        return self.text



SCRPCY_LOGO = (
    "███████╗  ██████╗   ██████╗   ██████╗   ██████╗   ██╗   ██╗\n"
    "██╔════╝  ██╔════╝  ██╔══██╗  ██╔══██╗  ██╔════╝  ╚██╗ ██╔╝\n"
    "███████╗  ██║       ██████╔╝  ██████╔╝  ██║        ╚████╔╝ \n"
    "╚════██║  ██║       ██╔══██╗  ██╔════╝  ██║         ╚██╔╝  \n"
    "███████║  ╚██████╗  ██║  ██║  ██║       ╚██████╗     ██║   \n"
    "╚══════╝   ╚═════╝  ╚═╝  ╚═╝  ╚═╝        ╚═════╝     ╚═╝   "
)
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
        self.active_skill_name = "scrpcy_manual"
        self.active_documents: set[str] = set()
        self.last_search_results: list[SearchResult] = []
        self.last_scout_results: list[ScoutResult] = []
        self.conversation_history: list[dict[str, str]] = []
        self._web_server_thread: threading.Thread | None = None

        self.task_manager = TaskManager()
        self.task_manager.add_completion_callback(self._on_background_task_completed)
        self.gdrive_manager = GoogleDriveManager()
        self.logger: StudioLogger = get_studio_logger(workspace.root_dir if workspace else None)
        self.vault_manager: VaultManager | None = None
        self.fact_checker: FactChecker | None = None
        self.devils_advocate: DevilsAdvocate | None = None
        self.polisher: AcademicPolisher | None = None
        self.chat_manager: ChatManager | None = None
        self.sub_mode: str | None = None
        self.active_chat_session: ChatSession | None = None
        self._has_printed_banner: bool = False
        self._ui_events_lock = threading.Lock()
        self._pending_ui_events: list[Callable[[], None]] = []

        self.current_project: ProjectInfo | None = None
        self.workspace: WorkspaceManager | None = None
        self.parser: DocumentParser | None = None
        self.wordcraft: WordCraftClient | None = None
        self.engine: LLMEngine | None = None
        self.scout: LiteratureScout | None = None
        self.intent_router: IntentRouter | None = None

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
        self.logger = get_studio_logger(self.workspace.root_dir if self.workspace else None)
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
        existing_sessions = self.chat_manager.list_sessions()
        self.active_chat_session = existing_sessions[0] if existing_sessions else None
        if hasattr(self, "task_manager") and self.task_manager:
            self.task_manager.set_db_path(self.workspace.root_dir / "tasks.db")
        self.sub_mode = None
        self.fact_checker = FactChecker(engine=self.engine)
        self.devils_advocate = DevilsAdvocate(engine=self.engine)
        self.intent_router = IntentRouter(engine=self.engine)
        self.polisher = AcademicPolisher(engine=self.engine)
        self.active_documents.clear()
        self._sync_vault_cache()

    def _enqueue_ui_event(self, event_fn: Callable[[], None]) -> None:
        """Enqueue UI rendering function to be cleanly printed on the main thread."""
        with self._ui_events_lock:
            self._pending_ui_events.append(event_fn)

    def _flush_pending_ui_events(self, clear_prompt_fn: Callable[[], None] | None = None) -> bool:
        """Drain and execute pending UI events on main thread."""
        with self._ui_events_lock:
            if not self._pending_ui_events:
                return False
            events = list(self._pending_ui_events)
            self._pending_ui_events.clear()

        if clear_prompt_fn is not None:
            clear_prompt_fn()

        for ev in events:
            try:
                ev()
            except Exception:
                pass
        return True

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
        """Ensure there is an active chat session bound, creating session1 if none exists."""
        if not self.chat_manager:
            raise RuntimeError("No chat manager initialized.")
        if self.active_chat_session:
            return self.active_chat_session

        sessions = self.chat_manager.list_sessions()
        if sessions:
            self.active_chat_session = sessions[0]
        else:
            self.active_chat_session = self.chat_manager.create_session(
                name="session1",
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
            banner_lines = "\n".join(f" {line}" for line in SCRPCY_LOGO.strip().splitlines())
            self.console.print(f"[bold cyan]{banner_lines}[/bold cyan]\n")
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
        """Deprecated: Outlier horizontal lines are exclusively reserved for input framing."""
        pass

    def run(self) -> None:
        """Main interaction loop handling Dashboard and Project modes."""
        init_windows_terminal()
        self.print_banner()

        if sys.stdin.isatty():
            self._run_interactive_tui()
            return

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
                    prompt_lbl = f"{self.current_project.name}/chat/{sess_name}> " if self.current_project else f"chat/{sess_name}> "
                    resolved_mode = "chat"
                elif self.current_project:
                    prompt_lbl = f"{self.current_project.name}> "
                    resolved_mode = "project"
                else:
                    prompt_lbl = "> "
                    resolved_mode = "dashboard"

                ctx_lbl = self.current_project.name if self.current_project else ""

                self._flush_pending_ui_events()

                def _get_project_status_indicator() -> str | None:
                    if not hasattr(self, "task_manager") or not self.task_manager:
                        return None
                    active_tasks = [t for t in self.task_manager.list_tasks() if t.status == "running"]
                    if not active_tasks:
                        return None
                    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                    idx = int(time.time() * 8) % len(SPINNER_FRAMES)
                    frame = SPINNER_FRAMES[idx]
                    if len(active_tasks) == 1:
                        t = active_tasks[0]
                        return f"[bold cyan]{frame}[/bold cyan] [bold bright_white]{t.name}[/bold bright_white] [dim]({t.progress or 'Sedang memproses di background...'})[/dim]"
                    return f"[bold cyan]{frame}[/bold cyan] [bold bright_white]{len(active_tasks)} proses latar belakang aktif...[/bold bright_white]"

                raw_input = interactive_input_prompt(
                    console=self.console,
                    prompt_label=prompt_lbl,
                    is_project_mode=(self.current_project is not None),
                    is_chat_mode=(self.sub_mode == "chat"),
                    mode=resolved_mode,
                    context_label=ctx_lbl,
                    current_model=mod,
                    current_effort=eff,
                    current_skill=self.active_skill_name,
                    on_idle=self._flush_pending_ui_events,
                    status_line_callback=_get_project_status_indicator,
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



    # =========================================================================

    # ZERO-STACKING INTERACTIVE TUI NAVIGATION (DESIGN.md COMPLIANT)

    # =========================================================================



    def _run_interactive_tui(self) -> None:

        """Top-level controller for pure cursor-driven, zero-stacking TUI."""

        while True:

            try:

                if self.current_project is None:

                    res = self.run_interactive_dashboard()

                    if res == "exit":

                        break

                else:

                    self.run_interactive_project()

            except (KeyboardInterrupt, EOFError):

                if self.current_project:

                    self.current_project = None

                    self.workspace = None

                    self.sub_mode = None

                    continue

                else:

                    self.console.print("\n[bold cyan]Closing scrpcy research studio. Goodbye![/bold cyan]")

                    break

            except Exception as exc:

                self.console.print(f"\n[bold red][!] Error:[/bold red] {escape(str(exc))}\n")

                time.sleep(1.5)



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



                action, idx, selected_p = interactive_item_browser(

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
                    confirmed = interactive_confirm(
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



    # =========================================================================

    # PROJECT MODE TUI (skripsi)

    # =========================================================================



    def run_interactive_project(self) -> None:

        """Project root interactive menu without typing."""

        while self.current_project is not None and self.sub_mode is None:

            cached_count = len(self._get_cached_stems())

            active_count = len(self.active_documents)

            model_name = self.get_active_project_model()

            effort = self.get_active_project_effort()

            model_label = _format_model_display_name(model_name)



            active_sess, closed_sess = self.chat_manager.get_active_sessions_count() if self.chat_manager else (0, 0)

            running_tasks = len([t for t in self.task_manager.list_tasks() if t.status == "running"])



            header = Text()

            header.append("scrpcy", style="bold cyan")

            header.append(" • ", style="dim")

            header.append(self.current_project.name, style="bold bright_white")

            header.append(" • ", style="dim")

            header.append(f"{model_label} ({effort})", style="bold cyan")

            header.append(" • ", style="dim")

            header.append(f"vault: {cached_count} indexed ({active_count} active)\n", style="dim")

            header.append("[✓] Connected to Master Project: ", style="bold green")

            header.append(f"{self.current_project.name}\n", style="bold cyan")

            header.append("  Location: ", style="bold white")

            header.append(f"{self.current_project.path}\n", style="dim")

            header.append("  Papers in Vault: ", style="bold white")

            header.append(f"{cached_count} indexed ({active_count} active in context)\n", style="dim")

            header.append("  Chat Sessions: ", style="bold white")

            header.append(f"{active_sess} running, {closed_sess} closed\n", style="dim")

            header.append("  Background Tasks: ", style="bold white")

            header.append(f"{running_tasks} running", style="bold cyan" if running_tasks else "dim")



            options = [

                ("vault", f"Vault ({cached_count} berkas)"),

                ("sessions", f"Sessions ({active_sess} running, {closed_sess} closed)"),

                ("processing", f"Processing ({running_tasks} proses aktif)"),

                ("settings", f"Settings ({model_label})"),

                ("exit", "Exit / Kembali ke Dashboard"),

            ]



            choice = interactive_menu(

                options=options,

                header=header,

                console=self.console,

                footer_hint="↑/↓ Navigate · enter Select · 1-5 Quick jump · esc Dashboard",

                allow_back=True,

            )



            if choice in ("exit", None):

                self.current_project = None

                self.workspace = None

                return



            if choice == "vault":

                self.sub_mode = "vault"

                self.run_interactive_vault()

                self.sub_mode = None

            elif choice == "sessions":

                self.sub_mode = "sessions"

                self.run_interactive_sessions()

                self.sub_mode = None

            elif choice == "processing":

                self.sub_mode = "processing"

                self.run_interactive_processing()

                self.sub_mode = None

            elif choice == "settings":

                self.cmd_models("")



    # =========================================================================

    # VAULT MODE TUI

    # =========================================================================



    def run_interactive_vault(self) -> None:

        """Vault resources interactive overview and options without typing."""

        while self.sub_mode == "vault" and self.current_project:

            active_items = self.vault_manager.list_active_files() if self.vault_manager else []

            cached_count = sum(1 for it in active_items if it.is_cached)

            needs_reload = len(active_items) - cached_count

            archived_count = len(self.vault_manager.list_archived_files()) if self.vault_manager else 0

            vault_path = (self.workspace.root_dir / "vault") if self.workspace else ""



            header_table = Table(box=None, padding=(0, 2), show_header=False)

            header_table.add_column("Label", style="dim", width=18)

            header_table.add_column("Value", style="white")

            header_table.add_row("Status", "[bold cyan]Aktif[/bold cyan]")

            header_table.add_row("Folder Vault", str(vault_path))

            header_table.add_row("Total Berkas", f"{len(active_items)} berkas aktif ({cached_count} terindeks Markdown, {needs_reload} perlu reload)")

            header_table.add_row("Diarsipkan", f"{archived_count} berkas di .archive/ (tidak aktif di konteks AI)")

            header_table.add_row("Format Dokumen", "[dim]PDF, DOCX, PPTX, XLSX, TXT, HTML, MD (didukung Microsoft MarkItDown)[/dim]")



            options = [

                ("view", f"View Papers ({len(active_items)} berkas)"),

                ("add", "Add Paper (Google Drive / Find Paper / File)"),

                ("back", "Back to Project"),

            ]



            choice = interactive_menu(

                options=options,

                header=header_table,

                console=self.console,

                footer_hint="↑/↓ Navigate · enter Select · 1-3 Quick jump · esc Back",

                allow_back=True,

            )



            if choice in ("back", None):

                return



            if choice == "view":

                self.run_interactive_vault_view()

            elif choice == "add":

                self.run_interactive_vault_add()



    def run_interactive_vault_view(self) -> None:

        """Browse vault items table carousel per DESIGN.md."""

        while True:

            active_items = self.vault_manager.list_active_files() if self.vault_manager else []

            if not active_items:

                self.console.print("\n[dim]Vault masih kosong (0 berkas aktif).\n[/dim]")

                time.sleep(1.2)

                return



            columns = [

                {"name": "#", "justify": "right", "width": 3, "style": "dim"},

                {"name": "Filename", "style": "bold white", "width": 34},

                {"name": "Format", "justify": "center", "width": 8, "style": "dim"},

                {"name": "Size", "justify": "right", "width": 10, "style": "dim"},

                {"name": "Context Status", "justify": "center", "width": 16},

                {"name": "Cite Key / Title", "style": "cyan"},

            ]



            def _row_builder(idx: int, item: Any, is_selected: bool) -> list[Any]:

                is_active = item.stem in self.active_documents

                ctx_badge = "[bold cyan][Active][/bold cyan]" if is_active else "[dim][Inactive][/dim]"

                arrow = "> " if is_selected else "  "

                fname_text = Text()

                fname_text.append(arrow, style="bold cyan" if is_selected else "dim")

                fname_text.append(item.filename[:30], style="bold bright_white" if is_selected else "white")

                return [

                    str(idx + 1),

                    fname_text,

                    item.extension.upper().lstrip("."),

                    item.readable_size,

                    ctx_badge,

                    (item.title or item.cite_key)[:40],

                ]



            header = Text()

            header.append(f"Daftar Berkas Vault [{self.current_project.name}]", style="bold bright_white")

            header.append(f" • {len(active_items)} berkas", style="dim")



            actions = [

                ("toggle", "Toggle Active", "enter"),

                ("read", "Read Document", "r"),

                ("archive", "Archive / Remove", "x"),

                ("restore", "Restore", "u"),

                ("refresh", "Refresh", "f"),

                ("back", "Back", "esc"),

            ]



            action, idx, selected_item = interactive_item_browser(

                title=header,

                columns=columns,

                items=active_items,

                row_builder=_row_builder,

                actions=actions,

                console=self.console,

                empty_message="Vault kosong.",

            )



            if action in ("back", "esc"):

                return

            elif action == "toggle" and selected_item:

                if selected_item.stem in self.active_documents:

                    self.active_documents.remove(selected_item.stem)

                else:

                    self.active_documents.add(selected_item.stem)

            elif action == "read" and selected_item:

                cache_doc = self.workspace.get_cached_document(selected_item.stem) if self.workspace else None

                if cache_doc and cache_doc.exists():

                    lines = cache_doc.read_text(encoding="utf-8", errors="replace").splitlines()

                    interactive_detail_viewer(

                        title=f"Membaca Dokumen: {selected_item.filename}",

                        content_lines=lines,

                        actions=[("back", "Back", "esc")],

                        console=self.console,

                    )

            elif action == "archive" and selected_item:
                confirmed = interactive_confirm(
                    f"Arsipkan '{selected_item.filename}' ke .archive/?",
                    default=True,
                    yes_label="Ya, arsipkan",
                    no_label="Batal",
                    console=self.console,
                )
                if confirmed:
                    self.vault_manager.archive_file(selected_item.filename)
                    if selected_item.stem in self.active_documents:
                        self.active_documents.remove(selected_item.stem)
                    self.console.print(f"[bold green][✓] File '{selected_item.filename}' berhasil diarsipkan ke .archive/.[/bold green]\n")

            elif action == "restore":

                archived = self.vault_manager.list_archived_files()

                if not archived:

                    self.console.print("\n[dim]Tidak ada berkas di .archive/.\n[/dim]")

                    time.sleep(1.0)

                else:

                    arch_opts = [(f.filename, f.filename) for f in archived] + [("back", "Back")]

                    ch = interactive_menu(arch_opts, header="Pilih berkas untuk di-restore:", console=self.console)

                    if ch and ch != "back":

                        self.vault_manager.restore_file(ch)

                        self.active_documents.add(Path(ch).stem)

                        self._sync_vault_cache()

            elif action == "refresh":

                self._sync_vault_cache()



    def run_interactive_vault_add(self) -> None:

        """Add document options submenu."""

        opts = [

            ("gdrive", "Google Drive Link (URL Google Drive)"),

            ("findpaper", "Find Paper (Literature Discovery & Download)"),

            ("local", "Local File / Path (.pdf, .docx, .txt)"),

            ("back", "Back"),

        ]

        choice = interactive_menu(opts, header="Tambah Dokumen ke Vault:", console=self.console)

        if choice in ("back", None):

            return

        if choice == "gdrive":

            try:

                self.console.print()

                url = self.console.input("[bold cyan]Masukkan URL Google Drive: [/bold cyan]").strip()

                if url:

                    self.cmd_vault_gdrive(url)

            except (KeyboardInterrupt, EOFError):

                pass

        elif choice == "findpaper":

            try:

                self.console.print()

                query = self.console.input("[bold cyan]Topik atau judul paper untuk dicari: [/bold cyan]").strip()

                if query:

                    self.cmd_findpaper(query)

            except (KeyboardInterrupt, EOFError):

                pass

        elif choice == "local":

            try:

                self.console.print()

                path_str = self.console.input("[bold cyan]Path berkas lokal (.pdf, .docx, dll): [/bold cyan]").strip()

                if path_str:

                    self.cmd_vault_add(path_str)

            except (KeyboardInterrupt, EOFError):

                pass



    # =========================================================================

    # PROCESSING MODE TUI

    # =========================================================================



    def run_interactive_processing(self) -> None:

        """Processing status and task inspection without typing."""

        while self.sub_mode == "processing" and self.current_project:

            tasks = self.task_manager.list_tasks()

            running_tasks = [t for t in tasks if t.status == "running"]

            completed_tasks = [t for t in tasks if t.status == "completed"]

            failed_tasks = [t for t in tasks if t.status == "failed"]

            cancelled_tasks = [t for t in tasks if t.status == "cancelled"]



            header_table = Table(box=None, padding=(0, 2), show_header=False)

            header_table.add_column("Property", style="dim", width=18)

            header_table.add_column("Value", style="white")

            header_table.add_row("Status", "[bold cyan]Active Processing[/bold cyan]" if running_tasks else "[dim]Idle[/dim]")

            header_table.add_row("Running Tasks", f"[bold cyan]{len(running_tasks)}[/bold cyan] proses" if running_tasks else "0 proses")

            header_table.add_row("Completed Tasks", f"{len(completed_tasks)} selesai")

            header_table.add_row("Failed / Cancelled", f"{len(failed_tasks)} gagal, {len(cancelled_tasks)} dibatalkan")

            header_table.add_row("Total Riwayat", f"{len(tasks)} tasks terdaftar")



            options = [

                ("view", f"View Task List ({len(tasks)} tasks)"),

                ("back", "Back to Project"),

            ]



            choice = interactive_menu(

                options=options,

                header=header_table,

                console=self.console,

                footer_hint="↑/↓ Navigate · enter Select · 1-2 Quick jump · esc Back",

                allow_back=True,

            )



            if choice in ("back", None):

                return



            if choice == "view":

                self.run_interactive_processing_view()



    def get_sanitized_task_logs(self, task: Any) -> list[str]:
        """Return task execution logs free of raw markdown AI syntheses and chat history dumps."""
        raw_logs = list(getattr(task, "logs", [])) or [f"Task: {task.name}", f"Status: {task.status}"]
        flat_logs: list[str] = []
        for entry in raw_logs:
            entry_str = str(entry)
            if entry_str.startswith("Respons AI:") or "Respons AI:\n" in entry_str:
                continue
            for l in entry_str.splitlines():
                l_strip = l.strip()
                if l_strip.startswith("Respons AI:"):
                    continue
                flat_logs.append(l)
        if getattr(task, "error", None):
            flat_logs.append(f"Error: {task.error}")
        res_sum = getattr(task, "result_summary", None)
        if res_sum and not res_sum.startswith("#"):
            if res_sum not in "\n".join(flat_logs):
                flat_logs.append(f"Summary: {res_sum}")
        return flat_logs

    def run_interactive_processing_view(self) -> None:
        """Browse task items table per DESIGN.md."""
        while True:
            tasks = self.task_manager.list_tasks()

            columns = [
                {"name": "#", "justify": "right", "width": 3, "style": "dim"},
                {"name": "Task ID", "style": "bold cyan", "width": 10},
                {"name": "Type", "justify": "center", "style": "dim", "width": 8},
                {"name": "Task Name / Description", "style": "bold white", "width": 28},
                {"name": "Origin Session", "style": "dim cyan", "width": 16},
                {"name": "Status", "justify": "center", "width": 16},
                {"name": "Tokens", "justify": "right", "style": "dim yellow", "width": 11},
                {"name": "Duration", "justify": "right", "style": "dim", "width": 9},
            ]

            def _row_builder(idx: int, t: Any, is_selected: bool) -> list[Any]:
                if t.status in ("need_action", "waiting", "need user action") or getattr(t, "needs_action", False):
                    st_badge = "[bold yellow][Need User Action][/bold yellow]"
                elif t.status == "running":
                    st_badge = "[bold cyan][Progress][/bold cyan]"
                elif t.status == "completed":
                    st_badge = "[bold green][Finish][/bold green]"
                elif t.status == "cancelled":
                    st_badge = "[dim][Cancelled][/dim]"
                else:
                    st_badge = "[bold red][Failed][/bold red]"

                sess_name = None
                if t.session_id:
                    if self.chat_manager:
                        sess = self.chat_manager.get_session(t.session_id)
                        if sess:
                            sess_name = sess.name
                    if not sess_name:
                        sess_name = t.session_id
                    if sess_name.lower().startswith("session:"):
                        sess_name = sess_name[8:].strip()
                    elif sess_name.lower().startswith("session :"):
                        sess_name = sess_name[9:].strip()

                sess_label = sess_name[:16] if sess_name else "[global]"

                tok_count = getattr(t, "display_tokens", 0)
                tok_str = f"{tok_count:,} tok" if tok_count > 0 else "-"

                arrow = "> " if is_selected else "  "
                id_text = Text()
                id_text.append(arrow, style="bold cyan" if is_selected else "dim")
                id_text.append(t.id, style="bold cyan" if is_selected else "dim")

                return [
                    str(idx + 1),
                    id_text,
                    t.task_type.upper(),
                    t.name[:28],
                    sess_label,
                    st_badge,
                    tok_str,
                    f"{t.duration_seconds:.1f}s",
                ]



            header = Text()

            header.append(f"Background Processing Tasks [{self.current_project.name}]", style="bold bright_white")

            header.append(f" • {len(tasks)} tasks", style="dim")



            actions = [
                ("detail", "Detail & Logs", "enter"),
                ("open", "Open Session", "o"),
                ("cancel", "Interrupt Task", "x"),
                ("clear", "Clear Finished", "c"),
                ("back", "Back", "esc"),
            ]



            action, idx, selected_task = interactive_item_browser(

                title=header,

                columns=columns,

                items=tasks,

                row_builder=_row_builder,

                actions=actions,

                console=self.console,

                empty_message="Tidak ada task.",

            )



            if action in ("back", "esc"):
                return
            elif action == "open" and selected_task:
                if selected_task.session_id and self.chat_manager:
                    sess = self.chat_manager.get_session(selected_task.session_id)
                    if sess:
                        self.active_chat_session = sess
                        self.active_skill_name = sess.active_skill
                        self.sub_mode = "chat"
                        self.run_interactive_chat_session(sess)
                        self.sub_mode = "processing"
                else:
                    self.console.print("[dim]Tugas ini tidak terikat dengan sesi chat tertentu.[/dim]\n")
            elif action == "detail" and selected_task:
                def _get_live_task_logs() -> list[str]:
                    cur_t = self.task_manager.get_task(selected_task.id) or selected_task
                    return self.get_sanitized_task_logs(cur_t)

                def _get_live_task_title() -> str:
                    cur_t = self.task_manager.get_task(selected_task.id) or selected_task
                    st_badge = f"[{cur_t.status.upper()}]"
                    return f"Log Tugas [{cur_t.id}]: {cur_t.name} {st_badge}"

                def _is_task_live() -> bool:
                    cur_t = self.task_manager.get_task(selected_task.id) or selected_task
                    return cur_t.status == "running"

                detail_acts = [
                    ("cancel", "Interrupt Task", "x"),
                    ("back", "Back", "esc"),
                ]

                d_choice = interactive_detail_viewer(
                    title=f"Log Tugas [{selected_task.id}]: {selected_task.name}",
                    content_lines=_get_live_task_logs(),
                    actions=detail_acts,
                    console=self.console,
                    lines_provider=_get_live_task_logs,
                    title_provider=_get_live_task_title,
                    is_live=_is_task_live,
                )

                if d_choice == "cancel":
                    self.task_manager.cancel_task(selected_task.id)
                    self.console.print(f"[bold yellow][!] Membatalkan proses tugas '{selected_task.name}'.[/bold yellow]\n")

            elif action == "cancel" and selected_task:
                if selected_task.status == "running":
                    self.task_manager.cancel_task(selected_task.id)

            elif action == "clear":

                self.task_manager.clear_completed()



    # =========================================================================

    # SESSIONS MODE TUI (TMUX-STYLE CHAT SESSIONS)

    # =========================================================================



    def run_interactive_sessions(self) -> None:

        """TMUX-style sessions manager without typing."""

        while self.sub_mode == "sessions" and self.current_project:

            active_cnt, closed_cnt = self.chat_manager.get_active_sessions_count() if self.chat_manager else (0, 0)

            total_cnt = active_cnt + closed_cnt



            header_table = Table(box=None, padding=(0, 2), show_header=False)

            header_table.add_column("Property", style="dim", width=20)

            header_table.add_column("Value", style="white")

            header_table.add_row("Active / Running", f"[bold cyan]{active_cnt}[/bold cyan] sessions")

            header_table.add_row("Closed / Terminated", f"[dim]{closed_cnt} sessions[/dim]")

            header_table.add_row("Total Sessions", f"{total_cnt} sessions")



            options = [

                ("list", f"List Sessions ({total_cnt} total)"),

                ("new", "New Session"),

                ("back", "Back to Project"),

            ]



            choice = interactive_menu(

                options=options,

                header=header_table,

                console=self.console,

                footer_hint="↑/↓ Navigate · enter Select · 1-3 Quick jump · esc Back",

                allow_back=True,

            )



            if choice in ("back", None):

                return



            if choice == "list":

                self.run_interactive_sessions_list()

            elif choice == "new":

                self.interactive_create_chat_session()



    def run_interactive_sessions_list(self) -> None:

        """Browse sessions list and Attach/Start/Close/Delete per DESIGN.md."""

        while True:

            sessions = self.chat_manager.list_sessions() if self.chat_manager else []

            if not sessions:

                self.console.print("\n[dim]Belum ada sesi percakapan. Buat sesi baru untuk memulai.\n[/dim]")

                time.sleep(1.2)

                return



            columns = [

                {"name": "#", "justify": "right", "width": 3, "style": "dim"},

                {"name": "Session Name", "style": "bold white", "width": 24},

                {"name": "Status", "justify": "center", "width": 14},

                {"name": "Persona / Skill", "style": "cyan", "width": 20},

                {"name": "Exchanges", "justify": "center", "width": 12, "style": "dim"},

                {"name": "Last Updated", "style": "dim", "width": 16},

            ]



            def _row_builder(idx: int, s: Any, is_selected: bool) -> list[Any]:

                st = getattr(s, "status", "running")

                if st == "running":

                    st_badge = "[bold cyan][Running][/bold cyan]"

                elif st == "detached":

                    st_badge = "[dim cyan][Detached][/dim cyan]"

                else:

                    st_badge = "[dim][Closed][/dim]"



                is_curr = self.active_chat_session and s.id == self.active_chat_session.id

                arrow = "> " if is_selected else "  "

                name_text = Text()

                name_text.append(arrow, style="bold cyan" if is_selected else "dim")

                name_text.append(s.name[:20], style="bold bright_white" if is_selected else "white")

                if is_curr:

                    name_text.append(" (current)", style="dim cyan")



                last_str = s.updated_at[:16].replace("T", " ") if s.updated_at else ""

                return [

                    str(idx + 1),

                    name_text,

                    st_badge,

                    s.active_skill,

                    f"{s.message_count} msgs",

                    last_str,

                ]



            header = Text()

            header.append(f"Saved Chat Sessions [{self.current_project.name}]", style="bold bright_white")

            header.append(f" • {len(sessions)} sessions", style="dim")



            actions = [

                ("attach", "Attach / Open", "enter"),

                ("terminate", "Close / Terminate", "x"),

                ("start", "Start Session", "s"),

                ("delete", "Delete Session", "d"),

                ("back", "Back", "esc"),

            ]



            action, idx, selected_sess = interactive_item_browser(

                title=header,

                columns=columns,

                items=sessions,

                row_builder=_row_builder,

                actions=actions,

                console=self.console,

                empty_message="Tidak ada sesi.",

            )



            if action in ("back", "esc"):

                return

            elif action == "attach" and selected_sess:

                st = getattr(selected_sess, "status", "running")

                if st == "closed":

                    self.chat_manager.update_session_status(selected_sess.id, "running")

                    selected_sess.status = "running"

                self.active_chat_session = selected_sess

                self.active_skill_name = selected_sess.active_skill

                self.sub_mode = "chat"

                self.run_interactive_chat_session(selected_sess)

                self.sub_mode = "sessions"

            elif action == "start" and selected_sess:

                self.chat_manager.update_session_status(selected_sess.id, "running")

                selected_sess.status = "running"

                self.active_chat_session = selected_sess

                self.active_skill_name = selected_sess.active_skill

                self.sub_mode = "chat"

                self.run_interactive_chat_session(selected_sess)

                self.sub_mode = "sessions"

            elif action == "terminate" and selected_sess:

                self.chat_manager.update_session_status(selected_sess.id, "closed")

            elif action == "delete" and selected_sess:
                confirmed = interactive_confirm(
                    f"Hapus permanen riwayat chat '{selected_sess.name}'?",
                    default=True,
                    yes_label="Ya, hapus riwayat chat",
                    no_label="Batal",
                    console=self.console,
                )
                if confirmed:
                    self.chat_manager.delete_session(selected_sess.id)
                    if self.active_chat_session and self.active_chat_session.id == selected_sess.id:
                        self.active_chat_session = None
                    self.console.print(f"[bold green][✓] Riwayat chat '{selected_sess.name}' berhasil dihapus.[/bold green]\n")



    def interactive_create_chat_session(self) -> None:

        """Create new session and immediately attach."""

        try:

            self.console.print()

            s_name = self.console.input("[bold cyan]Nama sesi baru (tekan Enter untuk default): [/bold cyan]").strip()

            session = self.chat_manager.create_session(

                name=s_name if s_name else None,

                active_skill=self.active_skill_name,

                status="running",

            )

            self.active_chat_session = session

            self.sub_mode = "chat"

            self.run_interactive_chat_session(session)

            self.sub_mode = "sessions"

        except (KeyboardInterrupt, EOFError):

            self.console.print()

            return



    def run_interactive_chat_session(self, session: ChatSession) -> None:

        """Active Chat Session: Prompt user for messages, supports /detach and /close."""

        model_name = self.get_active_project_model()

        model_label = _format_model_display_name(model_name)

        eff = self.get_active_project_effort()



        self.console.print(f"\n[bold cyan]● Sesi Aktif: {session.name}[/bold cyan] [dim]({session.active_skill} • {model_label})[/dim]")

        # Show last few messages for orientation
        messages = self.chat_manager.get_messages(session.id, limit=4)
        for msg in messages:
            if msg.role == "user":
                self.console.print(f"\n[bold bright_white]{self.current_project.name}/chat/{session.name}>[/bold bright_white] {escape(msg.content)}")
            elif msg.role == "system":
                first_line = msg.content.strip().splitlines()[0] if msg.content.strip() else ""
                self.console.print(f"[dim cyan]ℹ {escape(first_line)}[/dim cyan]")
            else:
                self.console.print()
                render_academic_markdown(msg.content, console=self.console)

        # Show active background tasks bound to this session upon entering/re-entering
        session_tasks = self.task_manager.list_tasks_for_session(session.id) if self.task_manager else []
        running = [t for t in session_tasks if t.status == "running"]
        if running:
            self.console.print(f"\n[bold yellow]●[/bold yellow] [bold bright_white]{len(running)} proses latar belakang sedang berjalan di sesi ini:[/bold bright_white]")
            for t in running:
                self.console.print(f"  [bold cyan]●[/bold cyan] [cyan]{t.name}[/cyan] [dim]({t.progress or 'Sedang memproses...'} • ID: {t.id})[/dim]")
            self.console.print(" [dim]• Indikator proses aktif di bawah prompt. Ketik /proc untuk pantau atau /detach untuk kembali.[/dim]\n")

        SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        def _get_chat_status_indicator() -> str | None:
            if not self.task_manager:
                return None
            active_tasks = [t for t in self.task_manager.list_tasks_for_session(session.id) if t.status == "running"]
            if not active_tasks:
                return None
            idx = int(time.time() * 8) % len(SPINNER_FRAMES)
            frame = SPINNER_FRAMES[idx]
            if len(active_tasks) == 1:
                t = active_tasks[0]
                desc = t.progress or "Sedang memproses di latar belakang..."
                return f"[bold cyan]{frame}[/bold cyan] [bold bright_white]{t.name}[/bold bright_white] [dim]({desc})[/dim]"
            else:
                names = ", ".join(t.id for t in active_tasks[:2])
                return f"[bold cyan]{frame}[/bold cyan] [bold bright_white]{len(active_tasks)} proses latar belakang aktif ({names})...[/bold bright_white]"

        while self.sub_mode == "chat" and self.active_chat_session:
            # Flush any background UI events first
            self._flush_pending_ui_events()

            # Process any pending suggested actions on the main thread
            if hasattr(self, "_pending_session_actions") and self._pending_session_actions.get(session.id):
                pending_acts = self._pending_session_actions.pop(session.id)
                self._execute_action_list(pending_acts)

            prompt_lbl = f"{self.current_project.name}/chat/{session.name}> "

            ctx_lbl = f"{self.current_project.name} • {session.name}"

            try:

                raw_input = interactive_input_prompt(

                    console=self.console,

                    prompt_label=prompt_lbl,

                    is_project_mode=True,

                    is_chat_mode=True,

                    mode="chat",

                    context_label=ctx_lbl,

                    current_model=model_name,

                    current_effort=eff,

                    current_skill=self.active_skill_name,

                    on_idle=self._flush_pending_ui_events,

                    status_line_callback=_get_chat_status_indicator,

                ).strip()

            except (KeyboardInterrupt, EOFError):

                # Ctrl+C inside chat detaches to sessions menu

                self.chat_manager.update_session_status(session.id, "detached")

                self.console.print(f"\n[bold cyan]● Sesi '{session.name}' di-detach ke latar belakang.[/bold cyan]\n")

                break



            if not raw_input:

                continue



            clean_lower = raw_input.lower().strip()

            if clean_lower in ("/detach", "detach", "/back", "back"):

                self.chat_manager.update_session_status(session.id, "detached")

                self.console.print(f"\n[bold cyan]● Sesi '{session.name}' di-detach ke latar belakang.[/bold cyan]\n")

                break



            if clean_lower in ("/close", "close", "/terminate", "terminate", "/exit", "exit"):

                self.chat_manager.update_session_status(session.id, "closed")

                self.console.print(f"\n[dim]● Sesi '{session.name}' ditutup (terminated).[/dim]\n")

                break



            if raw_input in ("?", "/help"):

                interactive_help_view(

                    console=self.console,

                    current_model=model_name,

                    current_effort=eff,

                    is_project_mode=True,

                    is_chat_mode=True,

                    mode="chat",

                    context_label=ctx_lbl,

                )

                continue



            # Process prompt through chat logic
            try:
                self.handle_chat_submode_interaction(raw_input)
            except SessionDetachSignal:
                break
            except SessionCloseSignal:
                break



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
            if args:
                self.handle_vault_interaction(args)
            else:
                self.render_mode_divider()
                self.sub_mode = "vault"
                self.render_vault_landing()
        elif cmd in ("/processing", "/proc", "/p"):
            if args:
                self.handle_processing_interaction(args)
            else:
                self.render_mode_divider()
                self.sub_mode = "processing"
                self.render_processing_summary()
        elif cmd in ("/chat", "/c"):
            self.render_mode_divider()
            self.sub_mode = "chat_hub"
            if args and args not in ("chat", "c", "hub"):
                self.handle_chat_hub_interaction(args)
            else:
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
        elif cmd in ("/turns", "/turn"):
            self.cmd_turns(args)
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
        except (KeyboardInterrupt, EOFError, OSError):
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

    def _ask_ai_orchestrator_barrier(
        self,
        finished_task: Any,
        running_siblings: list[Any],
        session: Any,
    ) -> dict[str, Any]:
        """Query LLM to decide whether to wait for concurrent tasks or synthesize immediately (ensuring 1 final response)."""
        model_name = self.get_active_project_model()
        api_key = self.config.get_api_key_for_model(model_name)
        if not api_key:
            return {"decision": "WAIT", "max_wait_seconds": 20, "reason": "Menunggu proses paralel selesai"}

        sibling_lines = []
        for s in running_siblings:
            dur = getattr(s, "duration_seconds", 0.0)
            if callable(dur):
                dur = dur()
            sibling_lines.append(f"- ID: {s.id} | Nama: {s.name} | Tipe: {s.task_type} | Durasi: {dur:.1f} detik")
        siblings_str = "\n".join(sibling_lines)

        gate_prompt = (
            f"[ORCHESTRATOR THREAD COORDINATION]\n"
            f"Tugas yang baru selesai: {finished_task.name} (ID: {finished_task.id})\n"
            f"Hasil: {getattr(finished_task, 'result_summary', '') or 'Selesai'}\n\n"
            f"Masih terdapat {len(running_siblings)} tugas lain yang sedang berjalan dalam sesi '{session.name}':\n"
            f"{siblings_str}\n\n"
            f"Sebagai AI Orchestrator, Anda wajib memastikan HANYA ADA 1 RESPONS AKHIR ke pengguna.\n"
            f"Tentukan salah satu keputusan koordinasi:\n"
            f"1. 'WAIT': Tunggu proses yang sedang berjalan di atas (maks 15-30 detik lagi) agar seluruh hasil digabung ke 1 respons sintesis utuh.\n"
            f"2. 'CANCEL_AND_PROCEED': Batalkan tugas yang tersisa karena hasil yang ada sudah cukup, lalu berikan 1 respons final sekarang.\n"
            f"3. 'PROCEED': Langsung berikan 1 respons final sekarang dengan data saat ini (tugas lain tetap berjalan tanpa memicu evaluasi kedua).\n\n"
            f"Format jawaban HANYA JSON murni:\n"
            f'{{\n  "decision": "WAIT" | "CANCEL_AND_PROCEED" | "PROCEED",\n  "max_wait_seconds": 20,\n  "reason": "Alasan singkat"\n}}'
        )

        try:
            tokens: list[str] = []
            async def _run_gate() -> None:
                async for tok in self.engine.stream_query(
                    query=gate_prompt,
                    documents=[],
                    skill_name="general",
                    model=model_name,
                    max_evidence_chunks=0,
                ):
                    tokens.append(tok)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_run_gate())
            finally:
                loop.close()

            raw_resp = "".join(tokens).strip()
            clean_str = raw_resp
            if "```json" in clean_str:
                clean_str = clean_str.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in clean_str:
                clean_str = clean_str.split("```", 1)[1].split("```", 1)[0].strip()
            import json
            data = json.loads(clean_str)
            return {
                "decision": data.get("decision", "WAIT"),
                "max_wait_seconds": int(data.get("max_wait_seconds", 20)),
                "reason": data.get("reason", "Koordinasi thread"),
            }
        except Exception:
            return {"decision": "WAIT", "max_wait_seconds": 20, "reason": "Menunggu proses paralel"}

    def _ask_ai_paywall_action(self, paper: Any, user_query: str) -> dict[str, str]:
        """Ask LLM whether the official publication abstract is sufficient or if an alternative OA paper is needed."""
        model_name = self.get_active_project_model()
        api_key = self.config.get_api_key_for_model(model_name)
        if not api_key:
            return {"decision": "ABSTRACT_ENOUGH", "reason": "API key tidak tersedia, menggunakan abstrak resmi publikasi."}

        title = getattr(paper, "title", "Paper")
        year = getattr(paper, "year", "n.d.")
        authors = getattr(paper, "authors", [])
        authors_str = ", ".join(authors[:3]) if authors else "Anonim"
        abstract = getattr(paper, "abstract", "") or "Abstrak tidak tersedia."

        paywall_prompt = (
            f"[EVALUASI AKSES LITERATUR ILMIAH]\n"
            f"Fokus riset pengguna: \"{user_query}\"\n\n"
            f"Ditemukan paper yang relevan dari jaringan sitasi vault, namun naskah lengkap (Full-Text PDF) berada di balik paywall komersial:\n"
            f"- Judul: {title} ({year})\n"
            f"- Penulis: {authors_str}\n"
            f"- Abstrak Resmi:\n{abstract[:1200]}\n\n"
            f"Tentukan apakah informasi pada abstrak resmi di atas SUDAH CUKUP untuk memperkuat argumen dan menjawab fokus riset pengguna, atau sistem sebaiknya MENCARI ALTERNATIF paper Open-Access lain yang memiliki naskah lengkap.\n"
            f"Format jawaban HANYA JSON murni:\n"
            f'{{\n  "decision": "ABSTRACT_ENOUGH" | "FIND_ALTERNATIVE",\n  "reason": "Alasan akademis singkat (1 kalimat)"\n}}'
        )

        try:
            tokens: list[str] = []
            async def _run_eval() -> None:
                async for tok in self.engine.stream_query(
                    query=paywall_prompt,
                    documents=[],
                    skill_name="general",
                    model=model_name,
                    max_evidence_chunks=0,
                ):
                    tokens.append(tok)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_run_eval())
            finally:
                loop.close()

            raw_resp = "".join(tokens).strip()
            clean_str = raw_resp
            if "```json" in clean_str:
                clean_str = clean_str.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in clean_str:
                clean_str = clean_str.split("```", 1)[1].split("```", 1)[0].strip()
            import json
            data = json.loads(clean_str)
            return {
                "decision": data.get("decision", "ABSTRACT_ENOUGH"),
                "reason": data.get("reason", "Evaluasi relevansi abstrak publikasi"),
            }
        except Exception:
            return {"decision": "ABSTRACT_ENOUGH", "reason": "Menggunakan abstrak resmi publikasi"}

    def _on_background_task_completed(self, task: Any) -> None:
        """Invoked asynchronously whenever any background task finishes."""
        session_id = getattr(task, "session_id", None)
        if not session_id or not self.chat_manager:
            return

        task_status = getattr(task, "status", "")
        task_name = getattr(task, "name", "Task")
        task_id = getattr(task, "id", "")
        task_type = getattr(task, "task_type", "")
        summary = getattr(task, "result_summary", "")
        error_msg = getattr(task, "error", "")

        if task_status == "completed":
            note = (
                f"[PROSES LATAR BELAKANG SELESAI: {task_name}]\n"
                f"- ID Tugas: {task_id} ({task_type})\n"
                f"- Ringkasan Hasil: {summary or 'Selesai tanpa kendala.'}\n"
                f"- Status Vault: Dokumen dan metadata telah terindeks ke dalam vault dan siap dianalisis dalam sesi riset ini."
            )
        elif task_status == "failed":
            note = (
                f"[PROSES LATAR BELAKANG GAGAL: {task_name}]\n"
                f"- ID Tugas: {task_id}\n"
                f"- Kendala: {error_msg or 'Terjadi kesalahan eksekusi.'}\n"
                f"- Rekomendasi: Anda dapat mencoba mencari sumber alternatif atau memeriksa log dengan perintah /proc."
            )
        elif task_status == "cancelled":
            note = f"[PROSES LATAR BELAKANG DIBATALKAN: {task_name} ({task_id})]"
        else:
            return

        if task_type not in ("chat", "agent"):
            try:
                self.chat_manager.add_message(session_id, "system", note)
            except Exception:
                pass

        # If user is currently active inside this exact chat session, display visual notification pill cleanly on main thread
        if self.sub_mode == "chat" and self.active_chat_session and self.active_chat_session.id == session_id:
            def _notify_task() -> None:
                if task_status == "completed":
                    if task_type in ("chat", "agent"):
                        self.console.print(f"\n[bold cyan]●[/bold cyan] [bold bright_white]Tugas Latar Belakang Selesai:[/bold bright_white] [cyan]{task_name}[/cyan]")
                        msgs = self.chat_manager.get_messages(session_id) if self.chat_manager else []
                        latest_assistant_msg = None
                        for m in reversed(msgs):
                            if m.role == "assistant":
                                latest_assistant_msg = m
                                break
                        if latest_assistant_msg:
                            self.console.print(f"\n[bold cyan]✦ Respons AI ({self.active_skill_name}):[/bold cyan]\n")
                            render_academic_markdown(latest_assistant_msg.content, console=self.console)
                            self.console.print()
                    else:
                        self.console.print(f"\n[bold cyan]●[/bold cyan] [bold bright_white]Tugas Latar Belakang Selesai:[/bold bright_white] [cyan]{task_name}[/cyan] [dim]({summary or 'Selesai'})[/dim]")
                elif task_status == "failed":
                    self.console.print(f"\n[bold red][x] Tugas Latar Belakang Gagal:[/bold red] [cyan]{task_name}[/cyan] [dim]({error_msg})[/dim]")
            self._enqueue_ui_event(_notify_task)

        # Multi-Thread Orchestrator Barrier: ensure exactly 1 consolidated response
        if task_type != "evaluation":
            if not hasattr(self, "_evaluated_task_ids"):
                self._evaluated_task_ids = set()

            if task_id in self._evaluated_task_ids:
                return

            running_siblings = [
                t for t in self.task_manager.list_tasks_for_session(session_id)
                if t.id != task_id and t.status == "running" and t.task_type != "evaluation" and t.id not in self._evaluated_task_ids
            ]

            if running_siblings:
                session = self.chat_manager.get_session(session_id)
                decision_info = self._ask_ai_orchestrator_barrier(task, running_siblings, session)
                decision = decision_info.get("decision", "WAIT")
                wait_sec = decision_info.get("max_wait_seconds", 20)
                reason = decision_info.get("reason", "")

                if self.sub_mode == "chat" and self.active_chat_session and self.active_chat_session.id == session_id:
                    if decision == "WAIT":
                        sib_names = ", ".join(f"{s.id}" for s in running_siblings)
                        self._enqueue_ui_event(
                            lambda: self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Orchestrator Thread:[/bold bright_white] [yellow]Menunggu hasil {sib_names}[/yellow] [dim]({reason})[/dim]")
                        )
                    elif decision == "CANCEL_AND_PROCEED":
                        self._enqueue_ui_event(
                            lambda: self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Orchestrator Thread:[/bold bright_white] [yellow]Membatalkan proses lain & langsung merumuskan 1 respons terpadu[/yellow] [dim]({reason})[/dim]")
                        )
                    else:
                        self._enqueue_ui_event(
                            lambda: self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Orchestrator Thread:[/bold bright_white] [cyan]Merumuskan respons final sekarang[/cyan] [dim]({reason})[/dim]")
                        )

                if decision == "WAIT":
                    start_w = time.time()
                    while time.time() - start_w < wait_sec:
                        still_running = [
                            t for t in self.task_manager.list_tasks_for_session(session_id)
                            if t.id != task_id and t.status == "running" and t.task_type != "evaluation" and t.id not in self._evaluated_task_ids
                        ]
                        if not still_running:
                            break
                        time.sleep(1)

                    still_running = [
                        t for t in self.task_manager.list_tasks_for_session(session_id)
                        if t.id != task_id and t.status == "running" and t.task_type != "evaluation" and t.id not in self._evaluated_task_ids
                    ]
                    if still_running:
                        for s in still_running:
                            self.task_manager.cancel_task(s.id)
                            self._evaluated_task_ids.add(s.id)
                        if self.sub_mode == "chat" and self.active_chat_session and self.active_chat_session.id == session_id:
                            self._enqueue_ui_event(
                                lambda: self.console.print(f"[bold cyan]●[/bold cyan] [bold yellow]Waktu tunggu tercapai ({wait_sec}s).[/bold yellow] [dim]Membatalkan proses yang terlambat dan merumuskan respons.[/dim]")
                            )

                elif decision == "CANCEL_AND_PROCEED":
                    for s in running_siblings:
                        self.task_manager.cancel_task(s.id)
                        self._evaluated_task_ids.add(s.id)

                elif decision == "PROCEED":
                    for s in running_siblings:
                        self._evaluated_task_ids.add(s.id)

            # Mark all current tasks as evaluated for this batch
            for t in self.task_manager.list_tasks_for_session(session_id):
                if t.status in ("completed", "cancelled"):
                    self._evaluated_task_ids.add(t.id)

            # Autonomous Agent Loop: only run vault evaluation for literature acquisition / curation tasks
            # Chat tasks synthesize and answer in the chat session directly without redundant vault auditing
            if task_type in ("download", "scout", "curate", "convert", "gdrive", "vault_add"):
                try:
                    self._run_task_ai_evaluation(task, session_id)
                except Exception:
                    pass

    def _run_task_ai_evaluation(self, task: Any, session_id: str) -> None:
        """Autonomous Agent Loop: Evaluate background task results and formulate user briefing."""
        if not self.engine or not self.chat_manager:
            return

        session = self.chat_manager.get_session(session_id)
        if not session:
            return

        task_status = getattr(task, "status", "")
        task_name = getattr(task, "name", "Task")
        task_id = getattr(task, "id", "")
        task_type = getattr(task, "task_type", "")
        summary = getattr(task, "result_summary", "")
        error_msg = getattr(task, "error", "")

        # Current documents in vault
        stems = self._get_cached_stems()
        vault_summary_lines = []
        for idx, stem in enumerate(stems[:10], 1):
            meta = self._get_doc_metadata(stem) if hasattr(self, "_get_doc_metadata") else None
            title = meta.title if meta and meta.title else stem.replace("_", " ")
            year = meta.year if meta and meta.year else ""
            vault_summary_lines.append(f"{idx}. [{stem}] {title} ({year})")
        vault_str = "\n".join(vault_summary_lines) if vault_summary_lines else "(Vault kosong)"

        # Session context / recent query
        messages = self.chat_manager.get_messages(session_id)
        last_user_query = ""
        for m in reversed(messages):
            if m.role == "user":
                last_user_query = m.content
                break

        # Collect all completed tasks in this session batch for 1 unified response
        all_completed = [
            t for t in self.task_manager.list_tasks_for_session(session_id)
            if t.status == "completed" and t.task_type != "evaluation"
        ]
        tasks_batch_lines = []
        for t in all_completed[:6]:
            tasks_batch_lines.append(f"- {t.id} ({t.name}): {t.result_summary or 'Selesai'}")
        tasks_batch_str = "\n".join(tasks_batch_lines) if tasks_batch_lines else f"- {task_id} ({task_name}): {summary}"

        eval_prompt = (
            f"[EVALUASI HASIL AKUISISI LITERATUR ILMIAH TERPADU]\n"
            f"Dokumen/Literatur Baru yang Telah Diunduh/Diindeks:\n"
            f"{tasks_batch_str}\n\n"
            f"Konteks Riset Sesi ({session.name}):\n"
            f"Fokus Riset/Pertanyaan Pengguna: \"{last_user_query or session.name}\"\n\n"
            f"Daftar Dokumen di Vault Saat Ini:\n"
            f"{vault_str}\n\n"
            f"Instruksi Evaluasi Mandiri Akademis (Agent Loop):\n"
            f"1. Evaluasi kualitas dan relevansi dokumen yang diperoleh terhadap fokus riset pengguna.\n"
            f"2. Analisis kecukupan literatur: Apakah literatur di vault saat ini sudah cukup untuk menjawab fokus riset pengguna.\n"
            f"3. Berikan sintesis ilmiah komprehensif dari paper yang baru diunduh dengan sitasi in-text APA.\n"
            f"4. ATURAN KRUSIAL ANTARMUKA:\n"
            f"   - Pengguna berinteraksi melalui antarmuka chat murni dan TIDAK BISA menjalankan perintah CLI slash.\n"
            f"   - DILARANG KERAS menyuruh pengguna mengetikkan perintah slash (misal JANGAN PERNAH tulis '/scout...', '/vault read...', 'Ketik /scout...').\n"
            f"   - Jika literatur sudah memadai ([STATUS: OK]), berikan sintesis yang mantap (100% yakin) tanpa meminta penelusuran lebih lanjut.\n"
            f"   - Saran riset masa depan harus ditulis murni sebagai narasi/konsep ilmiah akademis, BUKAN sebagai sintaks perintah.\n"
            f"   - DILARANG mencantumkan perintah /scout atau /vault read di akhir respons.\n"
        )

        model_name = self.get_active_project_model()
        api_key = self.config.get_api_key_for_model(model_name)
        if not api_key:
            return

        try:
            if self.sub_mode == "chat" and self.active_chat_session and self.active_chat_session.id == session_id:
                self._enqueue_ui_event(
                    lambda: self.console.print("[bold cyan]●[/bold cyan] [bold bright_white]Berpikir[/bold bright_white] [dim]Mengevaluasi hasil proses dengan AI...[/dim]\n")
                )

            history_msgs = [{"role": m.role, "content": m.content} for m in messages[-6:]]
            tokens: list[str] = []

            async def _run_eval() -> None:
                async for tok in self.engine.stream_query(
                    query=eval_prompt,
                    documents=[],
                    skill_name=self.active_skill_name,
                    model=model_name,
                    max_evidence_chunks=0,
                    history=history_msgs,
                ):
                    tokens.append(tok)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_run_eval())
            finally:
                loop.close()

            ai_briefing = "".join(tokens).strip()
            if ai_briefing:
                ai_briefing = self._sanitize_final_response(ai_briefing)
                if self.vault_manager:
                    ai_briefing = append_vault_bibliography(
                        text=ai_briefing,
                        evidence=[],
                        vault_manager=self.vault_manager,
                        style=self.config.citation.style,
                        active_documents=self.active_documents,
                    )
                self.chat_manager.add_message(session_id, "assistant", ai_briefing)
                # Queue any suggested actions to be handled cleanly on the main thread
                pattern = re.compile(r"```scrpcy\r?\n(.*?)\r?\n```", re.DOTALL | re.IGNORECASE)
                matches = pattern.findall(ai_briefing)
                if matches:
                    banned_roots = ("/vault read", "vault read", "/vault view", "vault view", "/vault list", "vault list", "/read", "read", "/scout", "scout", "/findpaper", "findpaper", "/search", "search")
                    raw_actions = [
                        cmd.strip() for m in matches for cmd in m.splitlines()
                        if cmd.strip() and not cmd.strip().startswith("#")
                        and not any(cmd.strip().lower().startswith(b) for b in banned_roots)
                    ]
                    if raw_actions:
                        if not hasattr(self, "_pending_session_actions"):
                            self._pending_session_actions = {}
                        self._pending_session_actions.setdefault(session_id, []).extend(raw_actions)

                if self.sub_mode == "chat" and self.active_chat_session and self.active_chat_session.id == session_id:
                    def _render_eval() -> None:
                        self.console.print(f"\n[bold cyan]✦ Evaluasi AI ({self.active_skill_name}):[/bold cyan]\n")
                        render_academic_markdown(ai_briefing, console=self.console)
                        self.console.print()
                    self._enqueue_ui_event(_render_eval)
        except Exception as exc:
            self.task_manager.log(task_id, f"AI evaluation error: {exc}")

    def submit_chat_background(self, query: str) -> None:
        """Submit a research query to run completely in background task manager."""
        if not self.workspace or not self.engine:
            self.console.print("[red][x] No active project workspace loaded.[/red]\n")
            return

        session = self._ensure_active_chat_session() if self.chat_manager else None
        if not session:
            self.console.print("[red][x] No active chat session.[/red]\n")
            return

        def _chat_worker(task_id: str) -> str:
            start_t = time.time()
            self.task_manager.log(task_id, f"Pertanyaan: {query}")
            model_name = self.get_active_project_model()
            self.task_manager.log(task_id, f"Model: {model_name}")
            self.task_manager.log(task_id, f"Memulai sintesis latar belakang...")
            if self.chat_manager:
                self.chat_manager.add_message(session.id, "user", query)

            active_paths = [self.workspace.get_cache_path(f"{s}.md") for s in self.active_documents]
            existing_paths = [p for p in active_paths if p.exists()]

            history_msgs = []
            if self.chat_manager:
                for m in self.chat_manager.get_messages(session.id)[-8:]:
                    if m.role == "user" and m.content == query:
                        continue
                    history_msgs.append({"role": m.role, "content": m.content})

            tokens = []
            async def _stream() -> None:
                async for tok in self.engine.stream_query(
                    query=query,
                    documents=existing_paths,
                    skill_name=self.active_skill_name,
                    model=model_name,
                    max_evidence_chunks=self.config.citation.max_evidence_chunks,
                    history=history_msgs,
                ):
                    tokens.append(tok)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_stream())
            finally:
                loop.close()

            resp = "".join(tokens).strip()
            if resp:
                try:
                    polisher = self.polisher or AcademicPolisher(engine=self.engine)
                    p_res = polisher.polish(resp, query)
                    if p_res.applied:
                        resp = p_res.polished_text
                except Exception:
                    pass

                if self.vault_manager and resp:
                    resp = append_vault_bibliography(
                        text=resp,
                        vault_manager=self.vault_manager,
                        style=self.config.citation.style,
                    )

            elapsed_c = time.time() - start_t
            in_tok = estimate_tokens(query) + 250
            out_tok = estimate_tokens(resp)
            tot_tok = in_tok + out_tok

            if self.chat_manager and resp:
                self.chat_manager.add_message(session.id, "assistant", resp)

            self.task_manager.update_tokens(task_id, tot_tok)
            self.task_manager.log(task_id, f"Token: {in_tok:,} in · {out_tok:,} out · {tot_tok:,} total ({model_name}) • {elapsed_c:.1f}s")
            self.task_manager.log(task_id, f"Selesai! {len(resp)} karakter disimpan ke riwayat sesi.")
            return f"Sintesis selesai ({len(resp)} karakter, {tot_tok:,} total token)" if resp else "Selesai"

        task = self.task_manager.submit(
            name=f"Chat: {query[:35]}...",
            task_type="chat",
            target_fn=_chat_worker,
            session_id=session.id,
        )
        self.console.print(f"[bold green][✓] Pertanyaan dikirim ke proses latar belakang:[/bold green] [bold cyan]{task.id}[/bold cyan]")
        self.console.print(" [dim]• Anda dapat tetap mengetik, keluar (/back), atau memantau status dengan /proc.[/dim]")
        self.console.print(" [dim]• Saat selesai, AI akan otomatis mengevaluasi hasil dan menyimpan jawaban di sesi ini.[/dim]\n")

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

    def _background_scout_worker(
        self,
        task_id: str,
        query: str = "",
        topic_query: str = "",
        max_download: int = 3,
        **kwargs: Any,
    ) -> str:
        """Background agent worker that formulates criteria, searches multi-source, ranks, and downloads into vault."""
        clean_query = (query or topic_query).strip().strip("'\"")
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
        all_oa_candidates = [p for p in curated if p.has_pdf and p.pdf_url]
        tried_titles = {p.title.lower() for p in oa_papers}

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
                self.task_manager.log(task_id, f"[!] Download error on #{idx} '{paper.title[:35]}': {exc}")
                # Autonomous error recovery: find open-access alternative candidate
                alt_found = False
                for alt_paper in all_oa_candidates:
                    if alt_paper.title.lower() in tried_titles:
                        continue
                    tried_titles.add(alt_paper.title.lower())
                    self.task_manager.log(task_id, f"[→] Trying OA alternative: '{alt_paper.title[:40]}...'")
                    try:
                        dl_path = loop.run_until_complete(self.scout.download_scout_paper(alt_paper))
                        self.active_documents.add(dl_path.stem)
                        self._sync_vault_cache()
                        self.task_manager.log(task_id, f"[✓] Ingested alternative to vault: {alt_paper.title[:45]}")
                        downloaded += 1
                        alt_found = True
                        break
                    except Exception as alt_exc:
                        self.task_manager.log(task_id, f"[!] Alternative failed: {alt_exc}")
                        continue

                if not alt_found and paper.abstract:
                    self.task_manager.log(task_id, f"[→] Fallback: Ingesting abstract for '{paper.title[:35]}...'")
                    self.scout.ingest_abstract_paper(paper)
                    safe_stem = sanitize_filename(paper.title)
                    self.active_documents.add(safe_stem)
                    self._sync_vault_cache()
                    downloaded += 1
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

        sess_id = self.active_chat_session.id if (self.sub_mode == "chat" and self.active_chat_session) else None
        task = self.task_manager.submit(
            name=f"Literature Scout & Ingest: {clean_query[:40]}",
            task_type="scout",
            target_fn=self._background_scout_worker,
            query=clean_query,
            max_download=3,
            session_id=sess_id,
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
        elif cmd in ("view", "read"):
            self.cmd_view(arg)
        elif cmd == "clear":
            self.clear_screen()
        elif cmd in ("curate", "kurasi"):
            self._cmd_vault_curate(arg)
        else:
            self.console.print(f"[yellow]Unknown vault command: '{clean}'. Type 'help' to see vault commands, or 'back' to return.[/yellow]\n")

    def _cmd_vault_curate(self, criteria: str) -> None:
        """Dispatch background agentic vault curation loop."""
        if not criteria:
            self.console.print("[yellow]Usage: vault curate \"<kriteria penelitian>\"[/yellow]\n")
            return
        session = self._ensure_active_chat_session() if self.chat_manager else None
        session_id = session.id if session else None
        task_id = self.task_manager.add_task(
            description=f"Kurasi vault: {criteria[:60]}",
            session_id=session_id,
        )
        self.task_manager.add_completion_callback(task_id, self._on_background_task_completed)
        import threading
        t = threading.Thread(
            target=self._run_background_task,
            args=(task_id, self._background_vault_curation_worker, task_id, session_id, criteria),
            daemon=True,
        )
        t.start()
        self.console.print(
            f"[bold cyan]\u25cf Kurasi vault dimulai di background[/bold cyan] [dim](task {task_id[:8]})[/dim]\n"
            f"[dim]Kriteria: {criteria}[/dim]\n"
            f"[dim]Gunakan /processing untuk memantau. Hasil tersedia saat selesai.[/dim]\n"
        )

    def _background_vault_curation_worker(
        self,
        task_id: str,
        session_id: str | None,
        criteria: str,
    ) -> str:
        """Agentic loop: iterates vault papers, AI evaluates each vs criteria (token-frugal).

        Short AI answers per batch. Stores each round as a system note in the session.
        Stops when AI emits [SELESAI].
        """
        import asyncio

        tm = self.task_manager
        tm.log(task_id, f"Memulai kurasi vault: {criteria!r}")

        if not self.vault_manager or not self.engine:
            return "Vault atau engine tidak tersedia."

        stems = self._get_cached_stems()
        if not stems:
            return "Vault kosong, tidak ada yang perlu dikurasi."

        vault_index = "\n".join(f"{i+1}. {s}" for i, s in enumerate(stems))

        BATCH = 5
        accumulated: list[str] = []

        def _run(coro):
            try:
                return asyncio.run(coro)
            except RuntimeError:
                # Already running loop (shouldn't happen in thread, but safety net)
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()

        for batch_start in range(0, len(stems), BATCH):
            batch = stems[batch_start : batch_start + BATCH]
            batch_nums = list(range(batch_start + 1, batch_start + 1 + len(batch)))
            batch_lines = "\n".join(f"{n}. {s}" for n, s in zip(batch_nums, batch))
            prompt = (
                f"Kriteria: {criteria}\n\n"
                f"Seluruh vault ({len(stems)} paper):\n{vault_index}\n\n"
                f"Evaluasi (RELEVAN/ARSIP, 1 baris per paper):\n{batch_lines}"
            )
            if batch_start + BATCH >= len(stems):
                prompt += "\n\nIni batch terakhir. Akhiri jawaban dengan [SELESAI]."

            try:
                result = _run(
                    self.engine.query(
                        query=prompt,
                        documents=[],
                        skill_name="scrpcy_manual",
                    )
                )
                round_text = result.answer.strip()
            except Exception as exc:
                round_text = f"[ERROR batch {batch_start+1}] {exc}"

            accumulated.append(round_text)
            tm.log(task_id, f"Batch {batch_start+1}-{batch_start+len(batch)}: {round_text[:100]}")

            if self.chat_manager and session_id:
                note = f"[LOOP-KURASI {batch_start+1}-{batch_start+len(batch)}]\n{round_text}"
                self.chat_manager.add_message(session_id, "system", note)

            if "[SELESAI]" in round_text.upper():
                break

        full_summary = "\n\n---\n\n".join(accumulated)
        if self.chat_manager and session_id:
            final_note = (
                f"[LOOP-KURASI SELESAI] Kriteria: {criteria!r}\n\n{full_summary}"
            )
            self.chat_manager.add_message(session_id, "system", final_note)

        tm.log(task_id, f"Kurasi selesai: {len(stems)} paper dievaluasi.")
        return f"Selesai mengevaluasi {len(stems)} paper."

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
        """Add external file or URL into project vault and convert with MarkItDown."""
        if not self.vault_manager or not self.workspace:
            self.console.print("[red][x] Vault manager is not active.[/red]\n")
            return

        clean_path = path_str.strip().strip('"\'')
        if not clean_path:
            self.console.print("[yellow]Usage: add <path/to/file or url>[/yellow]\n")
            return

        if clean_path.startswith(("http://", "https://")):
            import tempfile
            import urllib.request
            import urllib.parse
            try:
                with self.console.status(f"[cyan]Downloading document from URL...[/cyan]"):
                    parsed_url = urllib.parse.urlparse(clean_path)
                    suggested_name = Path(parsed_url.path).name or "downloaded_paper.pdf"
                    if not suggested_name.lower().endswith((".pdf", ".docx", ".txt", ".md")):
                        suggested_name += ".pdf"
                    temp_dir = Path(tempfile.gettempdir())
                    dest_file = temp_dir / suggested_name
                    req = urllib.request.Request(
                        clean_path,
                        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                    )
                    with urllib.request.urlopen(req, timeout=30) as resp, open(dest_file, "wb") as out_f:
                        out_f.write(resp.read())
                    src_path = dest_file
            except Exception as exc:
                self.console.print(f"[bold red][x] Failed to download URL:[/bold red] {exc}\n")
                return
        else:
            src_path = Path(clean_path).expanduser().resolve()
            if not src_path.exists():
                self.console.print(f"[bold red][x] File not found at:[/bold red] '{src_path}'\n")
                return

        if self.sub_mode == "chat":
            task = self.task_manager.submit(
                name=f"Ingest Document: {src_path.name}",
                task_type="convert",
                target_fn=self._background_vault_add_worker,
                path_str=str(src_path),
                session_id=self.active_chat_session.id if self.active_chat_session else None,
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
        """Soft-archive one or more vault entries using snapshot resolution to prevent index-shifting."""
        if not self.vault_manager:
            return

        if not identifier:
            self.console.print("[yellow]Usage: remove <# | filename> [# ...][/yellow]\n")
            return

        clean_id = identifier.strip()
        raw_ids: list[str] = []
        if '"' in clean_id or "'" in clean_id:
            import shlex
            try:
                tokens = shlex.split(clean_id)
                for t in tokens:
                    for sub in t.split(","):
                        s = sub.strip().strip('"\'')
                        if s:
                            raw_ids.append(s)
            except Exception:
                raw_ids = [t.strip().strip('"\'') for t in re.split(r"[\s,]+", clean_id) if t.strip().strip('"\'')]
        else:
            raw_ids = [t.strip() for t in re.split(r"[\s,]+", clean_id) if t.strip()]
        if not raw_ids:
            return

        try:
            archived_items = self.vault_manager.archive_multiple(raw_ids)
            for item in archived_items:
                self.active_documents.discard(item.path.stem)
            self._sync_vault_cache()

            if archived_items:
                for item in archived_items:
                    self.console.print(f"[yellow][-] '{item.filename}' dipindahkan ke .archive/.[/yellow]")
                if len(archived_items) > 1:
                    self.console.print(f"[dim]{len(archived_items)} dokumen diarsipkan dan dikeluarkan dari konteks AI.[/dim]\n")
                else:
                    self.console.print(f"[dim]Dikeluarkan dari konteks AI aktif. Gunakan \'restore\' untuk mengembalikan.[/dim]\n")
            else:
                self.console.print(f"[bold red][x] Tidak ada dokumen yang cocok dengan ID: {identifier}[/bold red]\n")
        except Exception as exc:
            self.console.print(f"[bold red][x] Gagal mengarsipkan dokumen:[/bold red] {exc}\n")

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
            except (KeyboardInterrupt, EOFError, OSError):
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
            padding=(0, 1),
            title_style="bold bright_white",
            header_style="bold cyan",
        )
        table.add_column("#", justify="right", style="dim", width=3)
        table.add_column("Task ID", style="bold cyan", width=9)
        table.add_column("Type", justify="center", style="dim", width=7)
        table.add_column("Description", style="white", no_wrap=True)
        table.add_column("Origin Session", style="dim cyan", width=14)
        table.add_column("Status", justify="center", width=13)
        table.add_column("Tokens", justify="right", style="dim yellow", width=11, no_wrap=True)
        table.add_column("Duration", justify="right", style="dim", width=8)

        for idx, t in enumerate(tasks, 1):
            if t.status == "running":
                status_badge = "[bold cyan][Running][/bold cyan]"
            elif t.status == "completed":
                status_badge = "[bold white][Completed][/bold white]"
            elif t.status == "cancelled":
                status_badge = "[dim][Cancelled][/dim]"
            else:
                status_badge = "[bold red][Failed][/bold red]"

            sess_name = None
            if t.session_id:
                if self.chat_manager:
                    sess = self.chat_manager.get_session(t.session_id)
                    if sess:
                        sess_name = sess.name
                if not sess_name:
                    sess_name = t.session_id
                if sess_name.lower().startswith("session:"):
                    sess_name = sess_name[8:].strip()
                elif sess_name.lower().startswith("session :"):
                    sess_name = sess_name[9:].strip()
            sess_label = sess_name[:14] if sess_name else "[global]"

            tok_count = getattr(t, "display_tokens", 0)
            tok_str = f"{tok_count:,} tok" if tok_count > 0 else "-"

            duration_str = f"{t.duration_seconds:.1f}s"
            table.add_row(
                str(idx),
                t.id,
                t.task_type.upper(),
                t.name,
                sess_label,
                status_badge,
                tok_str,
                duration_str,
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
        """Route to chat hub."""
        self.render_mode_divider()
        self.sub_mode = "chat_hub"
        clean = arg.strip()
        if clean and clean not in ("chat", "c", "hub"):
            self.handle_chat_hub_interaction(clean)
        else:
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

        if sub in ("chat", "c", "hub"):
            self.cmd_chat_list_sessions()
            return

        if sub in ("list", "ls", "sessions"):
            self.cmd_chat_list_sessions()
            return

        if sub in ("new", "create"):
            if not self.chat_manager:
                return
            chat_name = param
            if not chat_name:
                try:
                    chat_name = self.console.input("[bold cyan]Nama chat (tekan Enter untuk default): [/bold cyan]").strip()
                except (KeyboardInterrupt, EOFError, OSError):
                    self.console.print()
                    return

            if not chat_name:
                existing_sessions = self.chat_manager.list_sessions()
                existing_names = {s.name.lower() for s in existing_sessions}
                idx = 1
                while f"session{idx}" in existing_names or f"chat{idx}" in existing_names:
                    idx += 1
                chat_name = f"session{idx}"

            session = self.chat_manager.create_session(
                name=chat_name,
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
                except (KeyboardInterrupt, EOFError, OSError):
                    return
            if not target:
                return
            session = self.chat_manager.get_session(target)
            if session:
                self.render_mode_divider()
                self.active_chat_session = session
                self.sub_mode = "chat"
                self.console.print(f"[bold green][✓] Masuk ke sesi percakapan:[/bold green] [bold cyan]{session.name}[/bold cyan] ({session.message_count} pesan)")

                # Display any active or recent background tasks bound to this session
                session_tasks = self.task_manager.list_tasks_for_session(session.id)
                running = [t for t in session_tasks if t.status == "running"]
                completed = [t for t in session_tasks if t.status == "completed"]
                if running:
                    self.console.print(f"[bold yellow]●[/bold yellow] [bold bright_white]{len(running)} proses latar belakang masih berjalan di sesi ini:[/bold bright_white]")
                    for t in running:
                        self.console.print(f"  [dim]•[/dim] [cyan]{t.name}[/cyan] [dim]({t.progress} • Ketik /proc untuk pantau)[/dim]")
                elif completed:
                    self.console.print(f"[bold green][✓][/bold green] [bold bright_white]Proses latar belakang terakhir di sesi ini:[/bold bright_white]")
                    for t in completed[:2]:
                        self.console.print(f"  [dim]•[/dim] [cyan]{t.name}[/cyan] [dim]({t.result_summary or 'Selesai'})[/dim]")

                self.cmd_chat_history()
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
                    target = self.console.input("Masukkan nomor atau nama chat yang ingin dihapus (contoh: '1, 2' atau 'all'): ").strip()
                except (KeyboardInterrupt, EOFError, OSError):
                    return

            if not target:
                return

            if target.lower() == "all":
                all_sessions = self.chat_manager.list_sessions()
                total = len(all_sessions)
                for s in all_sessions:
                    self.chat_manager.delete_session(s.id)
                self.active_chat_session = None
                self.console.print(f"[bold green][✓] Berhasil menghapus semua ({total}) sesi chat.[/bold green]\n")
                self.cmd_chat_list_sessions()
                return

            targets = [t.strip() for t in target.replace(",", " ").split() if t.strip()]
            deleted_names = []
            not_found = []

            for t in targets:
                session = self.chat_manager.get_session(t)
                if session:
                    self.chat_manager.delete_session(session.id)
                    deleted_names.append(session.name)
                    if self.active_chat_session and self.active_chat_session.id == session.id:
                        self.active_chat_session = None
                else:
                    not_found.append(t)

            if deleted_names:
                self.console.print(f"[bold green][✓] Berhasil menghapus {len(deleted_names)} sesi chat:[/bold green] {', '.join(deleted_names)}")
            if not_found:
                self.console.print(f"[yellow][!] Sesi tidak ditemukan:[/yellow] {', '.join(not_found)}")
            self.console.print()
            self.cmd_chat_list_sessions()
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

        # Direct number or session name shorthand (e.g. typing "1" or "session1" directly)
        if self.chat_manager and sub not in ("chat", "c", "hub"):
            session = self.chat_manager.get_session(sub)
            if session:
                self.render_mode_divider()
                self.active_chat_session = session
                self.sub_mode = "chat"
                self.console.print(f"[bold green][✓] Masuk ke sesi percakapan:[/bold green] [bold cyan]{session.name}[/bold cyan] ({session.message_count} pesan)")
                self.cmd_chat_history()
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

            if cmd in ("/detach", "detach", "/back", "back"):
                self.render_mode_divider()
                running_tasks = [t for t in self.task_manager.list_tasks_for_session(self.active_chat_session.id) if t.status == "running"] if self.active_chat_session else []
                if running_tasks:
                    self.console.print(f"[bold yellow]●[/bold yellow] [dim]{len(running_tasks)} proses latar belakang tetap berjalan di background. Anda dapat kembali ke sesi ini kapan saja dengan 'resume {self.active_chat_session.name}'.[/dim]")
                if self.active_chat_session and self.chat_manager:
                    self.chat_manager.update_session_status(self.active_chat_session.id, "detached")
                self.sub_mode = "chat_hub"
                self.console.print(f"[dim]Kembali ke menu chat: {self.current_project.name}/chat>[/dim]\n")
                return
            elif cmd in ("/close", "close", "/exit", "exit", "/quit", "quit"):
                self.render_mode_divider()
                if self.active_chat_session and self.chat_manager:
                    self.chat_manager.update_session_status(self.active_chat_session.id, "closed")
                self.sub_mode = "chat_hub"
                self.console.print(f"[dim]Sesi '{self.active_chat_session.name if self.active_chat_session else ''}' ditutup (terminated).[/dim]\n")
                return
            elif cmd in ("/bg", "/async"):
                if not arg:
                    self.console.print("[yellow]Gunakan: /bg <pertanyaan atau perintah>[/yellow]\n")
                    return
                self.submit_chat_background(arg)
                return
            elif cmd in ("/cancel", "/stop"):
                if self.active_chat_session:
                    session_tasks = [t for t in self.task_manager.list_tasks_for_session(self.active_chat_session.id) if t.status == "running"]
                    if session_tasks:
                        for t in session_tasks:
                            self.task_manager.cancel_task(t.id)
                        self.console.print(f"[bold yellow][!] Membatalkan {len(session_tasks)} proses latar belakang di sesi ini.[/bold yellow]\n")
                    else:
                        self.console.print("[dim]Tidak ada proses latar belakang yang sedang berjalan di sesi ini.[/dim]\n")
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
            elif cmd in ("/new", "/create"):
                if not self.chat_manager:
                    return
                chat_name = arg.strip()
                if not chat_name:
                    try:
                        chat_name = self.console.input("[bold cyan]Nama chat (tekan Enter untuk default): [/bold cyan]").strip()
                    except (KeyboardInterrupt, EOFError, OSError):
                        self.console.print()
                        return

                if not chat_name:
                    existing_sessions = self.chat_manager.list_sessions()
                    existing_names = {s.name.lower() for s in existing_sessions}
                    idx = 1
                    while f"session{idx}" in existing_names or f"chat{idx}" in existing_names:
                        idx += 1
                    chat_name = f"session{idx}"

                session = self.chat_manager.create_session(
                    name=chat_name,
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
                    self.cmd_chat_history()
                    self.console.print("[dim]Ketik pertanyaan riset Anda langsung di bawah ini. Gunakan '/' untuk perintah (contoh: /back).[/dim]\n")
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

        self.console.print(f"\n[bold bright_white]Saved Chat Sessions [{self.current_project.name if self.current_project else 'Active'}][/bold bright_white]")
        sessions = self.chat_manager.list_sessions()
        if not sessions:
            self.console.print("[dim]Belum ada sesi percakapan tersimpan. Ketik 'new' untuk membuat sesi baru.[/dim]\n")
            return
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
                self.console.print(f"[bold bright_white]User:[/bold bright_white] {escape(msg.content)}")
            elif msg.role == "system":
                self.render_system_message(msg.content)
            else:
                self.console.print(f"[bold cyan]AI ({self.active_chat_session.active_skill}):[/bold cyan]\n")
                render_academic_markdown(msg.content, console=self.console)
                if msg.citations:
                    self.console.print(f"[dim]Citations: {', '.join(msg.citations)}[/dim]")
            self.console.print()

    def render_system_message(self, content: str) -> None:
        """Render system notifications cleanly without raw hashes or quotes."""
        lines = content.strip().splitlines()
        header = lines[0] if lines else ""
        match_completed = re.search(r"\[PROSES LATAR BELAKANG SELESAI:\s*(.*?)\]", header, re.IGNORECASE)
        match_failed = re.search(r"\[PROSES LATAR BELAKANG GAGAL:\s*(.*?)\]", header, re.IGNORECASE)
        match_cancelled = re.search(r"\[PROSES LATAR BELAKANG DIBATALKAN:\s*(.*?)\]", header, re.IGNORECASE)
        match_curation = re.search(r"\[LOOP-KURASI(?:\s+SELESAI)?\]\s*(.*)", header, re.IGNORECASE)

        if match_completed:
            task_name = match_completed.group(1).strip()
            self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Sistem:[/bold bright_white] [bold green]✓[/bold green] [bold cyan]PROSES LATAR BELAKANG SELESAI:[/bold cyan] [cyan]{task_name}[/cyan]")
            for line in lines[1:]:
                clean = re.sub(r"^[-\*#\s]+", "", line).replace("`", "").strip()
                if clean:
                    self.console.print(f"  [dim]•[/dim] [dim]{clean}[/dim]")
        elif match_failed:
            task_name = match_failed.group(1).strip()
            self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Sistem:[/bold bright_white] [bold red]✗[/bold red] [bold red]PROSES LATAR BELAKANG GAGAL:[/bold red] [cyan]{task_name}[/cyan]")
            for line in lines[1:]:
                clean = re.sub(r"^[-\*#\s]+", "", line).replace("`", "").strip()
                if clean:
                    self.console.print(f"  [dim]•[/dim] [dim]{clean}[/dim]")
        elif match_cancelled:
            task_name = match_cancelled.group(1).strip()
            self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Sistem:[/bold bright_white] [bold yellow]● Dibatalkan:[/bold yellow] [cyan]{task_name}[/cyan]")
        elif match_curation:
            extra = match_curation.group(1).strip()
            self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Sistem:[/bold bright_white] [bold cyan]Kurasi Vault:[/bold cyan] [dim]{extra}[/dim]")
            for line in lines[1:]:
                clean = re.sub(r"^[-\*#\s]+", "", line).replace("`", "").strip()
                if clean:
                    self.console.print(f"  [dim]•[/dim] [dim]{clean}[/dim]")
        else:
            clean = re.sub(r"###\s*", "", content).replace("`", "")
            self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Sistem:[/bold bright_white] [dim]{clean}[/dim]")

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
        if arg.lower().startswith("use "):
            arg = arg[4:].strip()
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
                self.config_manager.config.default_model = chosen_model
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
            s_table.add_row("Max Agent Turns", f"{getattr(self.config, 'max_agent_turns', 15)} turns")
            s_table.add_row("Refinement Rounds", f"{getattr(self.config, 'max_refinement_turns', 8)} rounds")
            s_table.add_row("Max Output Tokens", f"{getattr(self.config, 'max_output_tokens', 8192)} tokens")
            self.console.print(s_table)
            self.console.print("[dim]Commands: settings set-citation <apa|ieee> | settings set-turns <1-50> | settings set-refinement <2-25> | settings set-max-tokens <512-65536> | settings set-workspace <path>[/dim]\n")
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
        elif sub in ("set-turns", "turns", "turn"):
            if len(parts) < 2:
                self.console.print("[yellow]Usage: settings set-turns <1-50>[/yellow]\n")
                return
            try:
                val = int(parts[1].strip())
                if not (1 <= val <= 50):
                    self.console.print("[yellow]Turns must be between 1 and 50.[/yellow]\n")
                    return
                self.config_manager.update(max_agent_turns=val)
                self.config = self.config_manager.config
                self.console.print(f"[bold green][✓] Max agent turns set to:[/bold green] [bold cyan]{val}[/bold cyan]\n")
            except ValueError:
                self.console.print("[yellow]Invalid turns value. Enter an integer between 1 and 50.[/yellow]\n")
        elif sub in ("set-max-tokens", "max-tokens", "max_tokens"):
            if len(parts) < 2:
                self.console.print("[yellow]Usage: settings set-max-tokens <512-65536>[/yellow]\n")
                return
            try:
                val = int(parts[1].strip())
                if not (512 <= val <= 65536):
                    self.console.print("[yellow]Max output tokens must be between 512 and 65536.[/yellow]\n")
                    return
                self.config_manager.update(max_output_tokens=val)
                self.config = self.config_manager.config
                if getattr(self, "engine", None):
                    self.engine.config = self.config
                self.console.print(f"[bold green][✓] Max output tokens set to:[/bold green] [bold cyan]{val}[/bold cyan]\n")
            except ValueError:
                self.console.print("[yellow]Invalid value. Enter an integer between 512 and 65536.[/yellow]\n")
        elif sub in ("set-refinement", "set-refinement-rounds", "refinement"):
            if len(parts) < 2:
                self.console.print("[yellow]Usage: settings set-refinement <2-25>[/yellow]\n")
                return
            try:
                val = int(parts[1].strip())
                if not (2 <= val <= 25):
                    self.console.print("[yellow]Refinement rounds must be between 2 and 25.[/yellow]\n")
                    return
                self.config_manager.update(max_refinement_turns=val)
                self.config = self.config_manager.config
                if getattr(self, "engine", None):
                    self.engine.config = self.config
                self.console.print(f"[bold green][✓] Max refinement rounds set to:[/bold green] [bold cyan]{val}[/bold cyan]\n")
            except ValueError:
                self.console.print("[yellow]Invalid value. Enter an integer between 2 and 25.[/yellow]\n")
        else:
            self.console.print("[yellow]Usage: settings [show | set-citation <apa|ieee> | set-turns <1-50> | set-workspace <path>][/yellow]\n")

    def cmd_config(self, arg: str) -> None:
        """Alias for backward compatibility directing to cmd_models."""
        self.cmd_models(arg)

    def cmd_turns(self, arg: str = "") -> None:
        """View or configure the maximum agent loop turns (default 15, range 1-50)."""
        clean = arg.strip()
        if not clean:
            current_turns = getattr(self.config, "max_agent_turns", 15)
            self.console.print(f"\n[bold bright_white]Agent Loop Turn Limit:[/bold bright_white] [bold cyan]{current_turns}[/bold cyan] turn (batas maksimal loop sintesis, verifikasi, dan sitasi).")
            self.console.print("[dim]Gunakan: /turns <angka 1-50> (contoh: /turns 15 atau /turns 20)[/dim]\n")
            return

        try:
            val = int(clean)
            if not (1 <= val <= 50):
                self.console.print("[yellow]Batas turn harus bernilai antara 1 sampai 50.[/yellow]\n")
                return
            self.config_manager.update(max_agent_turns=val)
            self.config = self.config_manager.config
            self.console.print(f"[bold green][✓] Batas agent turn berhasil diatur ke:[/bold green] [bold cyan]{val} turn[/bold cyan]\n")
        except ValueError:
            self.console.print(f"[yellow]Nilai turn tidak valid: '{clean}'. Masukkan angka bulat (1-50).\n[/yellow]")


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
            sess_id = self.active_chat_session.id if (self.sub_mode == "chat" and self.active_chat_session) else None
            task = self.task_manager.submit(
                name=f"WordCraft Export: {filename}",
                task_type="convert",
                target_fn=self._background_docx_worker,
                filename=filename,
                session_id=sess_id,
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

    def _detect_internal_tool_call(self, raw_resp: str) -> str | None:
        """Detect internal tool commands emitted during an agent turn."""
        if not raw_resp:
            return None

        # 1. Fenced blocks: ```scrpcy ... ``` or ``` ... ```
        fenced_match = re.search(
            r"```(?:scrpcy)?\r?\n\s*(/(?:vault\s+(?:read|view|list|remove|rm|archive)|read|view|list|remove|archive|scout|search|findpaper)[^\n`]*)\r?\n```",
            raw_resp,
            re.IGNORECASE,
        )
        if fenced_match:
            return fenced_match.group(1).strip()

        # 2. Bare command line
        for line in raw_resp.splitlines():
            clean_line = line.strip()
            if re.match(
                r"^/(?:vault\s+(?:read|view|list|remove|rm|archive)|read|view|list|remove|archive|scout|search|findpaper)\b",
                clean_line,
                re.IGNORECASE,
            ):
                return clean_line

        # 3. Inline backticks: `/vault read 1`
        inline_match = re.search(
            r"`(/(?:vault\s+(?:read|view|list|remove|rm|archive)|read|view|list|remove|archive|scout|search|findpaper)[^`]+)`",
            raw_resp,
            re.IGNORECASE,
        )
        if inline_match:
            return inline_match.group(1).strip()

        return None

    def _sanitize_final_response(self, text: str) -> str:
        """Strip leaked internal tools and ambiguous slash commands after final response."""
        if not text:
            return ""

        has_ok_signal = "[status: ok]" in text.lower()

        # 1. Clean code blocks containing internal tools
        def _clean_block(m: re.Match) -> str:
            content = m.group(1)
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            filtered = []
            for l in lines:
                is_internal = re.match(r"^/(?:vault\s+(?:read|view|list)|read|view|list)\b", l, re.IGNORECASE)
                is_scout = re.match(r"^/(?:scout|findpaper|search)\b", l, re.IGNORECASE) or re.match(r"^(?:scout|findpaper)\b", l, re.IGNORECASE)
                if is_internal or (has_ok_signal and is_scout):
                    continue
                filtered.append(l)
            if not filtered:
                return ""
            return "```scrpcy\n" + "\n".join(filtered) + "\n```"

        cleaned = re.sub(r"```(?:scrpcy)?\r?\n(.*?)\r?\n```", _clean_block, text, flags=re.DOTALL | re.IGNORECASE)

        # 2. Strip bare lines containing internal reading/listing slash commands
        cleaned = re.sub(
            r"(?m)^\s*/(?:vault\s+(?:read|view|list)|read|view|list)\b[^\n]*\r?\n?",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        if has_ok_signal:
            cleaned = re.sub(
                r"(?m)^\s*/(?:scout|findpaper|search)\b[^\n]*\r?\n?",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

        # 3. Strip lines instructing user to type slash commands
        cleaned = re.sub(
            r"(?m)^.*?(?:ketik|jalankan|gunakan|perintah|command)\s*[:\-]?\s*`?/(?:vault\s+(?:read|view|list)|read|view|list)\b[^\n]*\r?\n?",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        if has_ok_signal:
            cleaned = re.sub(
                r"(?m)^.*?(?:ketik|jalankan|gunakan|perintah|command)\s*[:\-]?\s*`?/(?:scout|findpaper|search)\b[^\n]*\r?\n?",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

        # 4. Clean up multiple blank lines
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    def _format_vault_citation_guide(self) -> str:
        """Generate formatted citation guidelines and list of active vault papers for the prompt."""
        if not self.vault_manager or not self.workspace:
            return ""
        active_items = self.vault_manager.list_active_files()
        if not active_items:
            return ""
        lines = [
            "### PANDUAN SITASI DAN REFERENSI VAULT (GAYA APA):",
            "Dokumen ilmiah yang terverifikasi ada di vault proyek saat ini:",
        ]
        for idx, item in enumerate(active_items, 1):
            cache_path = self.workspace.get_cache_path(f"{item.stem}.md")
            meta = self.vault_manager._read_frontmatter_metadata(cache_path) if cache_path.exists() else None
            authors_str = ", ".join(meta.authors[:3]) if (meta and meta.authors) else item.stem.replace("_", " ").title()
            first_auth = meta.authors[0].split(",")[0].strip().split()[-1] if (meta and meta.authors) else item.stem.split("_")[0].capitalize()
            year_str = str(meta.year) if (meta and meta.year) else "n.d."
            cite_apa = f"({first_auth} et al., {year_str})" if (meta and meta.authors and len(meta.authors) > 2) else f"({first_auth}, {year_str})"
            cite_key = meta.cite_key if (meta and meta.cite_key) else (item.cite_key or item.stem)
            lines.append(f"- [{idx}] {authors_str} ({year_str}). \"{item.title or item.stem}\". File: {item.filename} | Sitasi in-text APA: {cite_apa} atau [{cite_key}]")
        lines.extend([
            "",
            "ATURAN SITASI & INTEGRITAS REFERENSI:",
            "1. Anda WAJIB menyertakan sitasi in-text APA (e.g. (NamaPenulis, Tahun) atau (NamaPenulis et al., Tahun)) pada SETIAP klaim faktual, metodologi, dan data angka.",
            "2. HANYA gunakan referensi dari dokumen vault di atas. Dilarang mengutip referensi fiktif.",
            "3. DILARANG MENULIS DAFTAR PUSTAKA ATAU REFERENSI DI AKHIR JAWABAN. Berhentilah setelah kesimpulan/rekomendasi. Sistem Scrpcy otomatis menyusun dan menambahkan Daftar Pustaka terverifikasi dari metadata vault di akhir respon.",
            "4. Jawaban harus lengkap, komprehensif, dan mantap (100% yakin). JANGAN menyuruh pengguna melakukan pencarian atau menjalankan perintah.",
            "5. Setelah jawaban akhir disajikan, DILARANG KERAS mencantumkan perintah slash (/scout, /vault read, dll).",
        ])
        return "\n".join(lines)

    def _execute_internal_vault_tool(self, cmd_line: str) -> tuple[str, str, str]:
        """Execute an internal read or list vault tool, returning (action_label, target_label, tool_output)."""
        clean = cmd_line.strip().lstrip("/")
        parts = clean.split(maxsplit=2)
        cmd_root = parts[0].lower() if parts else ""
        sub = parts[1].lower() if len(parts) > 1 else ""
        arg = parts[2].strip() if len(parts) > 2 else ""

        if cmd_root in ("read", "view"):
            arg = sub
            sub = "read"

        if sub in ("list", "ls", ""):
            stems = self._get_cached_stems()
            if not stems:
                return "Membaca", "Vault Kosong", "Vault saat ini kosong. Belum ada paper yang terindeks."

            lines = ["Daftar dokumen di vault saat ini:"]
            for idx, stem in enumerate(stems, 1):
                lines.append(f"{idx}. {stem}")
            return "Membaca", "Daftar Paper di Vault", "\n".join(lines)

        elif sub in ("read", "view"):
            if not arg and len(parts) > 1 and parts[1].isdigit():
                arg = parts[1]
            stems = self._get_cached_stems()
            target_stem = None
            if arg.isdigit():
                idx = int(arg) - 1
                if 0 <= idx < len(stems):
                    target_stem = stems[idx]
            else:
                matching = [s for s in stems if s.lower() == arg.lower() or arg.lower() in s.lower()]
                if matching:
                    target_stem = matching[0]

            if not target_stem and self.vault_manager:
                self.vault_manager.reload_vault(force=False)
                self._sync_vault_cache()
                stems = self._get_cached_stems()
                if arg.isdigit():
                    idx = int(arg) - 1
                    if 0 <= idx < len(stems):
                        target_stem = stems[idx]
                else:
                    matching = [s for s in stems if s.lower() == arg.lower() or arg.lower() in s.lower()]
                    if matching:
                        target_stem = matching[0]

            if not target_stem or not self.workspace:
                return "Membaca", f"Dokumen '{arg}' Tidak Ditemukan", f"Dokumen '{arg}' tidak ditemukan di vault."

            try:
                text_content = self.workspace.read_from_cache(f"{target_stem}.md")
            except Exception:
                text_content = ""

            max_chars = 12000
            truncated = text_content[:max_chars]
            if len(text_content) > max_chars:
                truncated += "\n\n... (bagian dokumen selanjutnya dipotong untuk efisiensi token) ..."

            return "Membaca", f"cache/{target_stem}.md", truncated

        elif cmd_root in ("scout", "search", "findpaper") or sub in ("scout", "search", "findpaper"):
            q_raw = f"{sub} {arg}".strip() if cmd_root in ("scout", "search", "findpaper") else arg.strip()
            query_str = q_raw.strip('"\'')
            if not query_str:
                return "Mencari", "Query Kosong", "Tidak ada kata kunci pencarian yang diberikan."
            if not self.scout:
                return "Mencari", "Scout Tidak Aktif", "Fitur scout literatur tidak tersedia."
            try:
                from scrpcy.core.scout import ScoutQuery
                q_spec = ScoutQuery(keywords=query_str, summary=query_str)
                results = asyncio.run(self.scout.search_multi_source(q_spec, limit=3))
                if not results:
                    return "Mencari", f"'{query_str}' (0 hasil)", f"Tidak ditemukan paper untuk '{query_str}'."
                lines = [f"Hasil pencarian literatur untuk '{query_str}':"]
                for i, r in enumerate(results[:3], 1):
                    lines.append(f"{i}. {r.title} ({r.year or 'n.d.'}) - Authors: {', '.join(r.authors[:3])}")
                    if r.abstract:
                        lines.append(f"   Abstrak: {r.abstract[:300]}...")
                return "Mencari", f"'{query_str[:25]}'", "\n".join(lines)
            except Exception as e:
                return "Mencari", "Gagal", f"Pencarian literatur gagal: {e}"

        elif sub in ("remove", "rm", "archive"):
            if not arg and len(parts) > 1 and parts[1].isdigit():
                arg = parts[1]

            raw_ids = [t.strip() for t in re.split(r"[\s,]+", arg.strip()) if t.strip()]
            if not raw_ids:
                return "Mengarsipkan", "Tidak Ada Target", "Tidak ada ID yang diberikan."

            if not self.vault_manager:
                return "Mengarsipkan", "Gagal", "Vault manager tidak aktif."

            try:
                archived_items = self.vault_manager.archive_multiple(raw_ids)
                for item in archived_items:
                    self.active_documents.discard(item.path.stem)
                self._sync_vault_cache()

                if not archived_items:
                    return "Mengarsipkan", f"ID '{arg}' Tidak Ditemukan", f"Tidak ada dokumen di vault yang cocok dengan: {arg}"

                lines_out = [f"Dokumen '{item.filename}' berhasil diarsipkan ke .archive/." for item in archived_items]
                label = f"vault/{archived_items[0].filename}" if len(archived_items) == 1 else f"{len(archived_items)} dokumen diarsipkan"
                return "Mengarsipkan", label, "\n".join(lines_out)
            except Exception as exc:
                return "Mengarsipkan", "Gagal", f"Gagal mengarsipkan dokumen '{arg}': {exc}"

        return "Membaca", clean, f"Perintah internal {clean} dijalankan."

    def _extract_scout_query(self, query: str) -> str:
        """Extract clean salient keywords from user query for academic literature scouting."""
        stopwords = {
            "bagaimana", "apa", "apakah", "mengapa", "kenapa", "jelaskan", "jelaskanlah",
            "tolong", "bisa", "tolonglah", "uraikan", "sebutkan", "bagaimanakah",
            "how", "what", "why", "explain", "describe", "can", "could", "would",
            "the", "is", "are", "and", "or", "to", "in", "of", "on", "for", "with",
            "tentang", "mengenai", "pada", "di", "ke", "dari", "dan", "atau", "yang",
            "ini", "itu", "tersebut", "secara", "seperti", "yaitu", "adalah", "ialah",
            "carikan", "cari", "aku", "saya", "kami", "kita", "berikan", "kasih", "butuh",
            "argument", "argumen", "alasan", "dampak", "buruk", "baik", "bagus",
            "mau", "ingin", "paper", "jurnal", "artikel", "literature", "literatur",
        }
        cleaned = re.sub(r"[^\w\s-]", " ", query)
        words = cleaned.split()
        salient = [w for w in words if w.lower() not in stopwords and len(w) > 2]
        if salient:
            return " ".join(salient[:6])
        return query.strip()

    def _run_agent_turn_interactive(
        self,
        coro: Any,
        step_desc: str | Any,
        prompt_lbl: str,
        model_label: str,
        effort: str,
    ) -> Any:
        """Run an agent streaming turn with framed input outlier and persistent footer.
        
        Renders:
        ● Step indicator / spinner
        [Empty line]
        ─────────────────────────────────────────────────────────────
        skripsi/chat/testing1> █
        ─────────────────────────────────────────────────────────────
        ? for shortcuts               scrpcy_manual • Gemini 3.8 Flash • high
        """
        import shutil

        # Non-interactive fallback for CI, pipes, or tests
        desc_str = step_desc() if callable(step_desc) else str(step_desc)
        if not sys.stdin.isatty():
            with self.console.status(
                f"[bold cyan]●[/bold cyan] [bold bright_white]{desc_str}[/bold bright_white] [dim]({model_label})[/dim]",
                spinner="dots",
            ):
                return asyncio.run(coro)

        SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

        # Flush any stale keystrokes before starting the wait loop
        if sys.platform == "win32":
            import msvcrt
            while msvcrt.kbhit():
                try:
                    msvcrt.getwch()
                except Exception:
                    break

        async def _interactive_loop() -> Any:
            t = asyncio.create_task(coro)
            typed_buffer: list[str] = []
            frame_idx = 0
            last_spin = time.time()
            prev_lines_drawn = 0
            last_desc_str = ""

            def _get_layout() -> tuple[str, str]:
                cur_width = self.console.width or shutil.get_terminal_size((80, 24)).columns
                cur_div = "─" * cur_width
                left_st = "? for shortcuts · /detach to background · /close to terminate"
                right_st = f"{self.active_skill_name} • {model_label} • {effort}" if self.active_skill_name else f"{model_label} • {effort}"
                pad = max(2, cur_width - len(left_st) - len(right_st) - 2)
                cur_foot = f"{left_st}{' ' * pad}{right_st}"
                return cur_div, cur_foot

            divider, footer = _get_layout()

            try:
                while not t.done():
                    now = time.time()
                    need_redraw = False
                    if now - last_spin >= 0.08:
                        frame_idx = (frame_idx + 1) % len(SPINNER_FRAMES)
                        last_spin = now
                        need_redraw = True

                    if sys.platform == "win32":
                        import msvcrt
                        while msvcrt.kbhit():
                            ch = msvcrt.getwch()
                            need_redraw = True
                            if ch == "\x03":  # Ctrl+C
                                t.cancel()
                                raise asyncio.CancelledError("cancel")
                            elif ch == "\x1b":  # Esc
                                if typed_buffer:
                                    typed_buffer.clear()
                                else:
                                    raise SessionDetachSignal("detach")
                            elif ch in ("\r", "\n"):
                                cmd = "".join(typed_buffer).strip().lower()
                                typed_buffer.clear()
                                if cmd in ("/detach", "detach", "/back", "back"):
                                    raise SessionDetachSignal("detach")
                                elif cmd in ("/close", "close", "/terminate", "terminate", "/exit", "exit"):
                                    t.cancel()
                                    raise SessionCloseSignal("close")
                                elif cmd in ("/cancel", "cancel", "/batal", "batal", "/stop", "stop"):
                                    t.cancel()
                                    raise asyncio.CancelledError("cancel")
                            elif ch in ("\b", "\x7f"):
                                if typed_buffer:
                                    typed_buffer.pop()
                            elif ch.isprintable():
                                typed_buffer.append(ch)
                                curr = "".join(typed_buffer).strip().lower()
                                if curr in ("/detach", "detach"):
                                    raise SessionDetachSignal("detach")
                                elif curr in ("/close", "close"):
                                    t.cancel()
                                    raise SessionCloseSignal("close")
                                elif curr in ("/cancel", "cancel", "/batal", "/stop"):
                                    t.cancel()
                                    raise asyncio.CancelledError("cancel")

                    cur_desc_val = step_desc() if callable(step_desc) else str(step_desc)
                    if cur_desc_val != last_desc_str:
                        last_desc_str = cur_desc_val
                        need_redraw = True

                    if need_redraw:
                        divider, footer = _get_layout()
                        buf_str = "".join(typed_buffer)
                        frame = SPINNER_FRAMES[frame_idx]
                        prompt_line = (
                            f"\033[1;36m{prompt_lbl}\033[0m\033[1;37m{buf_str}\033[0m\033[1;37m█\033[0m"
                            if buf_str
                            else f"\033[1;36m{prompt_lbl}\033[0m\033[2;37m[Sedang memproses... Tekan Esc atau ketik /detach untuk latar belakang]\033[0m"
                        )
                        lines = [
                            f"\033[36m{frame}\033[0m {cur_desc_val}",
                            "",
                            f"\033[90m{divider}\033[0m",
                            prompt_line,
                            f"\033[90m{divider}\033[0m",
                            f"\033[90m{footer}\033[0m",
                        ]
                        if prev_lines_drawn > 0:
                            _clear_lines(prev_lines_drawn, self.console)
                        sys.stdout.write("\n".join(lines) + "\n")
                        sys.stdout.flush()
                        prev_lines_drawn = len(lines)

                    await asyncio.sleep(0.03)

                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, self.console)
                return await t

            except BaseException as exc:
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, self.console)
                if isinstance(exc, SessionDetachSignal):
                    raise
                if not t.done():
                    t.cancel()
                try:
                    await t
                except BaseException:
                    pass
                raise

        return asyncio.run(_interactive_loop())

    def handle_chat(self, query: str) -> None:
        """Process research query with evidence synthesis and real-time streaming output."""
        if not self.workspace or not self.engine:
            self.console.print("[red][x] No active project workspace loaded.[/red]\n")
            return

        session = self._ensure_active_chat_session() if self.chat_manager else None
        model_name = self.get_active_project_model()
        effort = self.get_active_project_effort()
        model_label = _format_model_display_name(model_name)

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

        # Intent Analysis & Ambiguity Disambiguation Gate
        router = self.intent_router or IntentRouter(engine=self.engine)
        intent_res = router.analyze(query)

        # Interactive clarification prompt if input is ambiguous and in interactive terminal
        if intent_res.needs_clarification and sys.stdin.isatty():
            self.console.print(f"\n[bold cyan][?][/bold cyan] [bold bright_white]{intent_res.clarification_prompt}[/bold bright_white]")
            for opt_i, opt_text in enumerate(intent_res.clarification_options, 1):
                self.console.print(f"  [cyan]{opt_i}.[/cyan] {opt_text}")
            try:
                c_pick = self.console.input("[bold cyan]Pilih opsi [1-3, default 1]: [/bold cyan]").strip()
                if c_pick == "1":
                    intent_res.intent = ResearchIntent.ARGUMENT_EXPLORATION
                    intent_res.target_skill_name = "argument_matrix"
                elif c_pick == "2":
                    intent_res.intent = ResearchIntent.LITERATURE_REVIEW
                    intent_res.target_skill_name = "scrpcy_manual"
                elif c_pick == "3":
                    intent_res.intent = ResearchIntent.FACT_VERIFICATION
                    intent_res.target_skill_name = "scrpcy_manual"
            except (KeyboardInterrupt, EOFError):
                self.console.print()

        # Display recognized intent indicator
        if intent_res.intent == ResearchIntent.ARGUMENT_EXPLORATION:
            self.console.print("[bold cyan]●[/bold cyan] [bold bright_white]Mendeteksi Niat:[/bold bright_white] [cyan]Eksplorasi Argumen Riset (Matriks Dialektika Pro/Kontra)[/cyan]")
        elif intent_res.intent == ResearchIntent.LITERATURE_REVIEW:
            self.console.print("[bold cyan]●[/bold cyan] [bold bright_white]Mendeteksi Niat:[/bold bright_white] [cyan]Tinjauan Pustaka Terstruktur (Section-by-Section Synthesis)[/cyan]")
        elif intent_res.intent == ResearchIntent.FACT_VERIFICATION:
            self.console.print("[bold cyan]●[/bold cyan] [bold bright_white]Mendeteksi Niat:[/bold bright_white] [cyan]Verifikasi Data & Fakta Empiris[/cyan]")

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

        # Autonomous Deep-Research Literature Discovery Hook (when evidence is insufficient or domain coverage is low)
        coverage_score = self.engine.evidence_synthesizer.compute_coverage_score(query, evidence) if evidence else 0.0
        if evidence and coverage_score < 0.35:
            self.console.print(f"[dim]● Catatan: Cakupan literatur vault terhadap kueri saat ini adalah {int(coverage_score * 100)}%. Evaluasi kritis akan menelusuri literatur tambahan jika diperlukan.[/dim]")

        # Pre-flight auto-scout triggers when vault has no matching evidence
        if not evidence and self.scout:
            self.console.print("[bold cyan]●[/bold cyan] [bold bright_white]Berpikir[/bold bright_white] [dim]Mengevaluasi kecukupan literatur di vault...[/dim]")
            self.console.print("[dim]● Bukti di vault belum mencukupi untuk topik ini.[/dim]")
            suggested_topic = self._extract_scout_query(query)
            self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Rencana:[/bold bright_white] [cyan]Mencari & mengunduh literatur terkait '{suggested_topic}'[/cyan]")

            try:
                user_choice = self.console.input("[bold cyan][?] Jalankan pencarian & unduh otomatis? [Y/n/ketik kata kunci baru]: [/bold cyan]").strip()
            except (KeyboardInterrupt, EOFError, OSError):
                user_choice = "n"
                self.console.print()

            if user_choice.lower() in ("n", "no", "tidak", "batal", "cancel"):
                self.console.print("[dim]● Pencarian otomatis dilewati. Melanjutkan sintesis dengan pengetahuan umum...[/dim]\n")
            else:
                target_topic = suggested_topic if user_choice.lower() in ("", "y", "yes", "ya", "ok") else user_choice
                self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Mencari[/bold bright_white] [dim]Menelusuri repositori ilmiah (OpenAlex & Semantic Scholar)...[/dim]")

                scout_progress = _AgentProgressTracker("Menelusuri repositori ilmiah (OpenAlex & Semantic Scholar)...")

                def on_scout_status(action: str, label: str) -> None:
                    scout_progress.set(f"{action}: {label}")
                    if not sys.stdin.isatty():
                        self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]{action}[/bold bright_white] [cyan]{label}[/cyan]")

                new_stems: list[str] = []
                timed_out = False
                try:
                    async def _run_scout_with_timeout() -> list[str]:
                        import inspect
                        sig = inspect.signature(self.scout.auto_scout_and_ingest)
                        scout_kwargs: dict[str, Any] = {
                            "max_papers": 2,
                            "on_status": on_scout_status,
                        }
                        if "engine" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                            scout_kwargs["engine"] = self.engine

                        return await asyncio.wait_for(
                            self.scout.auto_scout_and_ingest(
                                target_topic,
                                **scout_kwargs,
                            ),
                            timeout=35.0,
                        )

                    prompt_str = f"{self.current_project.name}/chat/{session.name}> " if (self.sub_mode == "chat" and session and self.current_project) else (f"{self.current_project.name}> " if self.current_project else "> ")
                    new_stems = self._run_agent_turn_interactive(
                        coro=_run_scout_with_timeout(),
                        step_desc=scout_progress,
                        prompt_lbl=prompt_str,
                        model_label=model_label,
                        effort=effort,
                    )
                except (asyncio.TimeoutError, TimeoutError):
                    timed_out = True
                except Exception as exc:
                    if hasattr(self, "logger") and self.logger:
                        self.logger.log_error(session.id if session else None, "AUTO_SCOUT", exc)
                    self.console.print(f"[dim]● Pencarian repositori gagal: {exc}. Melanjutkan...[/dim]\n")

                if timed_out:
                    task = self.task_manager.submit(
                        name=f"Deep Research Scout: {target_topic}",
                        task_type="scout",
                        target_fn=self._background_scout_worker,
                        query=target_topic,
                        topic_query=target_topic,
                        session_id=session.id if session else None,
                    )
                    self.console.print("[bold yellow][!] Unduhan repositori membutuhkan waktu > 20 detik.[/bold yellow]")
                    self.console.print(f"[bold green][✓] Proses dialihkan ke background:[/bold green] [bold cyan]{task.id}[/bold cyan] [dim](Ketik /proc untuk pantau)[/dim]\n")
                    initial_query_override = (
                        f"{query}\n\n"
                        f"[CATATAN SISTEM SCRPCY: Proses penelusuran & pengunduhan literatur untuk topik '{target_topic}' "
                        f"memerlukan waktu lebih dari 20 detik dan dialihkan ke latar belakang (Task ID: {task.id}).\n"
                        f"Instruksi untuk AI: Beritahu pengguna secara ringkas bahwa unduhan sedang diproses di background. "
                        f"Berikan sintesis awal atau fondasi konseptual terbaik berdasarkan pengetahuan yang ada saat ini. "
                        f"Secara proaktif sarankan fokus konseptual atau hipotesis apa yang bisa dikaji sekarang sementara tugas unduhan selesai di background (JANGAN menyuruh pengguna mengetikkan perintah slash).]"
                    )
                elif new_stems:
                    if self.vault_manager:
                        self.vault_manager.reload_vault(force=False)
                    self._sync_vault_cache()
                    for s in new_stems:
                        self.active_documents.add(s)

                    active_paths = [self.workspace.get_cache_path(f"{s}.md") for s in self.active_documents]
                    existing_paths = [p for p in active_paths if p.exists()]
                    evidence = self.engine.evidence_synthesizer.retrieve_evidence(
                        query=query,
                        documents=existing_paths,
                        max_chunks=self.config.citation.max_evidence_chunks,
                    )
                    self.console.print(f"[bold green][✓] Berhasil mengindeks {len(new_stems)} dokumen baru ke vault.[/bold green]")
                    for s in new_stems:
                        self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Membaca[/bold bright_white] [cyan]cache/{s}.md[/cyan] [dim](Dokumen siap)[/dim]")
                    self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Menganalisis[/bold bright_white] [dim]Menyintesis temuan paper baru ke dalam jawaban komprehensif...[/dim]\n")

        # 1. Action Indicator: Membaca (jika sudah ada bukti)
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

        if not session and self.chat_manager:
            session = self._ensure_active_chat_session()
        if session and self.chat_manager:
            self.chat_manager.add_message(session.id, "user", query)

        # 2. Action Indicator: Berpikir
        self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Berpikir[/bold bright_white] [dim]Menganalisis literatur dan menyusun sintesis ilmiah...[/dim]")
        self.console.print(f"\n[bold cyan]✦ {model_label}[/bold cyan] [dim]({self.active_skill_name} • {effort} effort)[/dim]")

        session_id = session.id if session else "default"
        if hasattr(self, "logger") and self.logger:
            self.logger.log_user_query(session_id, query, self.active_documents)
            if evidence:
                self.logger.log_evidence_retrieved(session_id, query, evidence)

        start_time = time.time()
        max_agent_turns = getattr(self.config, "max_agent_turns", 15)
        max_refinement_turns = getattr(self.config, "max_refinement_turns", min(8, max_agent_turns - 2))
        turn = 0
        refinement_turns = 0
        locked_supported_claims: list[str] = []
        best_draft: str = ""
        best_grounding: float = -1.0
        best_critic: int = -1
        best_composite: float = -1.0
        accumulated_input_tokens = 0
        accumulated_output_tokens = 0
        total_turns_completed = 0
        current_query = initial_query_override if "initial_query_override" in locals() and initial_query_override else query
        citation_guide = self._format_vault_citation_guide()
        if citation_guide:
            current_query = f"{citation_guide}\n\n{current_query}"
        current_docs = list(existing_paths)
        final_response = ""
        last_turn_tokens: list[str] = []
        last_v_report: VerificationReport | None = None
        last_c_report: CritiqueReport | None = None

        try:
            while turn < max_agent_turns:
                turn_tokens: list[str] = []

                # Gather previous session dialogue history
                history_msgs: list[dict[str, str]] = []
                if self.chat_manager and session:
                    raw_history = self.chat_manager.get_messages(session.id)
                    for m in raw_history[-10:]:
                        if m.role == "user" and m.content == query:
                            continue
                        history_msgs.append({"role": m.role, "content": m.content})

                progress = _AgentProgressTracker(
                    "Menghubungi AI & Menganalisis..." if turn == 0 else "Menyintesis temuan dokumen ke dalam jawaban akhir..."
                )

                async def _run_full_turn_async() -> dict[str, Any]:
                    progress.set("Menghubungi AI & Menganalisis..." if turn == 0 else "Menyintesis temuan dokumen...")
                    t_stream_s = time.time()
                    if hasattr(self, "logger") and self.logger:
                        self.logger.log_llm_stream_start(session_id, turn, model_name, str(progress))

                    effective_skill_name = self.active_skill_name
                    if self.active_skill_name == "scrpcy_manual" and intent_res.intent == ResearchIntent.ARGUMENT_EXPLORATION:
                        effective_skill_name = "argument_matrix"

                    async for token in self.engine.stream_query(
                        query=current_query,
                        documents=current_docs,
                        skill_name=effective_skill_name,
                        model=model_name,
                        max_evidence_chunks=self.config.citation.max_evidence_chunks if current_docs else 0,
                        history=history_msgs,
                    ):
                        turn_tokens.append(token)

                    t_stream_dur = time.time() - t_stream_s
                    raw_resp = "".join(turn_tokens)
                    out_tok = estimate_tokens(raw_resp)
                    if hasattr(self, "logger") and self.logger:
                        self.logger.log_llm_stream_finish(session_id, turn, model_name, out_tok, t_stream_dur, len(raw_resp))

                    # 1. Internal vault tool call check
                    tool_cmd = self._detect_internal_tool_call(raw_resp)
                    if tool_cmd and turn < max_agent_turns - 1:
                        return {"type": "internal_tool", "command": tool_cmd, "response": raw_resp}

                    # 2. Academic Polisher
                    progress.set("Menyelaraskan ritme & gaya bahasa akademis...")
                    p_resp = raw_resp
                    t_pol_s = time.time()
                    try:
                        polisher = self.polisher or AcademicPolisher(engine=self.engine)
                        p_res = await asyncio.to_thread(polisher.polish, raw_resp, query)
                        if p_res.applied:
                            p_resp = p_res.polished_text
                        if hasattr(self, "logger") and self.logger:
                            self.logger.log_polisher_result(
                                session_id=session_id,
                                applied=p_res.applied,
                                duration_s=time.time() - t_pol_s,
                                reason=p_res.reason,
                                burst_before=p_res.burstiness_score_before,
                                burst_after=p_res.burstiness_score_after,
                                citations_preserved=p_res.citations_preserved,
                            )
                    except Exception as p_err:
                        if hasattr(self, "logger") and self.logger:
                            self.logger.log_error(session_id, "POLISHER", p_err)

                    # 3. FactChecker
                    progress.set("Mengaudit keabsahan fakta terhadap dokumen vault...")
                    t_fc_s = time.time()
                    verifier = self.fact_checker or FactChecker(engine=self.engine)
                    v_rep = await asyncio.to_thread(verifier.verify_synthesis, p_resp, evidence)
                    if hasattr(self, "logger") and self.logger:
                        self.logger.log_fact_check_result(
                            session_id=session_id,
                            grounding_score=v_rep.grounding_score,
                            total_claims=v_rep.total_claims,
                            supported_claims=v_rep.supported_claims,
                            unsupported_claims=v_rep.unsupported_claims,
                            duration_s=time.time() - t_fc_s,
                            hallucination_warning=v_rep.hallucination_warning,
                        )

                    # 4. Devil's Advocate
                    progress.set("Menjalankan evaluasi kritis (Devil's Advocate)...")
                    t_cr_s = time.time()
                    critic = self.devils_advocate or DevilsAdvocate(engine=self.engine)
                    c_rep = await asyncio.to_thread(critic.critique, query, p_resp, intent_res.intent.value)
                    if hasattr(self, "logger") and self.logger:
                        self.logger.log_critic_result(
                            session_id=session_id,
                            score=c_rep.score,
                            passed=c_rep.passed,
                            duration_s=time.time() - t_cr_s,
                            recency_alert=c_rep.recency_alert,
                            needs_citation_mapping=c_rep.needs_citation_mapping,
                            weaknesses=c_rep.weaknesses,
                            summary=c_rep.critique_summary,
                        )

                    # Monotonic Claim Locking: Record verified supported claims
                    for c in v_rep.claims:
                        if c.status == "supported" and c.claim not in locked_supported_claims:
                            locked_supported_claims.append(c.claim)

                    # Best-Draft Tracking
                    composite = 0.55 * v_rep.grounding_score + 0.45 * (c_rep.score / 10.0)
                    nonlocal best_composite, best_draft, best_grounding, best_critic
                    if composite > best_composite or not best_draft:
                        best_composite = composite
                        best_draft = p_resp
                        best_grounding = v_rep.grounding_score
                        best_critic = c_rep.score

                    # Accumulate token metrics for this turn (across ALL turns including refinement iterations)
                    stream_in_est = estimate_tokens(current_query) + sum(c.token_estimate or estimate_tokens(c.text) for c in (evidence or [])) + 250
                    nonlocal accumulated_input_tokens, accumulated_output_tokens, total_turns_completed
                    accumulated_input_tokens += stream_in_est + getattr(v_rep, "input_tokens", 0) + getattr(c_rep, "input_tokens", 0)
                    accumulated_output_tokens += out_tok + getattr(v_rep, "output_tokens", 0) + getattr(c_rep, "output_tokens", 0)
                    total_turns_completed = turn + 1

                    # Check if ungrounded claims or citation mapping triggers another turn
                    needs_refine = (bool(v_rep.hallucination_warning) or bool(c_rep.needs_citation_mapping)) and turn < max_agent_turns - 1
                    if needs_refine:
                        max_refine_for_intent = 1 if intent_res.intent in (ResearchIntent.ARGUMENT_EXPLORATION, ResearchIntent.FACT_VERIFICATION) else max_refinement_turns
                        if refinement_turns >= max_refine_for_intent:
                            return {
                                "type": "epistemic_exit",
                                "best_draft": best_draft or p_resp,
                                "v_report": v_rep,
                                "c_report": c_rep,
                                "best_grounding": best_grounding,
                                "best_critic": best_critic,
                            }
                        # Priority: If Devil's Advocate flags missing citations or knowledge gaps,
                        # run citation mapping & scout search to ingest new literature!
                        if c_rep.needs_citation_mapping:
                            return {
                                "type": "needs_citation_mapping",
                                "report": c_rep,
                                "v_report": v_rep,
                                "response": p_resp,
                            }
                        if v_rep.hallucination_warning:
                            return {
                                "type": "ungrounded",
                                "report": v_rep,
                                "c_report": c_rep,
                                "response": p_resp,
                            }

                    # 5. Automatic Bibliography
                    final_text = p_resp
                    if self.vault_manager and final_text:
                        progress.set("Menyusun daftar pustaka vault terverifikasi...")
                        final_text = await asyncio.to_thread(
                            append_vault_bibliography,
                            text=final_text,
                            evidence=evidence,
                            vault_manager=self.vault_manager,
                            style=self.config.citation.style,
                            active_documents=self.active_documents,
                        )
                        if hasattr(self, "logger") and self.logger:
                            self.logger.log_bibliography_appended(
                                session_id=session_id,
                                style=self.config.citation.style,
                                entries_count=len(evidence),
                                filenames=[c.source for c in evidence],
                            )

                    return {
                        "type": "final",
                        "response": final_text,
                        "v_report": v_rep,
                        "c_report": c_rep,
                    }

                prompt_str = f"{self.current_project.name}/chat/{session.name}> " if (self.sub_mode == "chat" and session and self.current_project) else (f"{self.current_project.name}> " if self.current_project else "> ")
                turn_coro = _run_full_turn_async()
                try:
                    turn_result = self._run_agent_turn_interactive(
                        coro=turn_coro,
                        step_desc=progress,
                        prompt_lbl=prompt_str,
                        model_label=model_label,
                        effort=effort,
                    )
                except BaseException:
                    try:
                        turn_coro.close()
                    except Exception:
                        pass
                    raise
                last_turn_tokens = turn_tokens

                if turn_result["type"] == "internal_tool":
                    cmd_line = turn_result["command"].strip()
                    is_archive = any(k in cmd_line.lower() for k in ("remove", "rm", "archive"))
                    is_scout = any(k in cmd_line.lower() for k in ("scout", "search", "findpaper"))
                    if is_archive:
                        spin_desc = "Mengarsipkan dokumen..."
                    elif is_scout:
                        spin_desc = "Mencari literatur ilmiah..."
                    else:
                        spin_desc = "Membaca dokumen dari vault..."
                    with self.console.status(
                        f"[bold cyan]●[/bold cyan] [bold bright_white]{spin_desc}[/bold bright_white]",
                        spinner="dots",
                    ):
                        action_label, target_label, tool_output = self._execute_internal_vault_tool(cmd_line)

                    # Status Indicator: Membaca/Mengarsipkan -> Menganalisis
                    if is_archive:
                        sub_label = "(Mengarsipkan paper yang tidak relevan...)"
                    elif is_scout:
                        sub_label = "(Menelusuri repositori akademik...)"
                    else:
                        sub_label = "(Membaca teks dari vault...)"
                    self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]{action_label}[/bold bright_white] [cyan]{target_label}[/cyan] [dim]{sub_label}[/dim]")
                    self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Menganalisis[/bold bright_white] [dim]Menyintesis temuan dokumen ke dalam jawaban akhir...[/dim]\n")

                    current_query = (
                        f"### HASIL EKSEKUSI ALAT ({cmd_line}):\n"
                        f"Aksi: {action_label}\n"
                        f"Target: {target_label}\n"
                        f"```markdown\n{tool_output}\n```\n\n"
                        f"Pertanyaan riset asli pengguna: \"{query}\"\n"
                        f"Berdasarkan informasi di atas, silakan lanjutkan sintesis akademis mendalam 100% yakin dan berikan jawaban akhir tanpa menyuruh pengguna menjalankan perintah slash."
                    )
                    current_docs = []
                    turn += 1
                    continue

                elif turn_result["type"] == "epistemic_exit":
                    self.console.print("[bold yellow]●[/bold yellow] [bold bright_white]Epistemic Convergence[/bold bright_white] [dim]Batas 3 iterasi perbaikan tercapai. Memfinalisasi draf terbaik dengan catatan dialektis...[/dim]\n")
                    raw_final = turn_result.get("best_draft", "")
                    dialectic_caveat = (
                        "\n\n> [!NOTE]\n"
                        "> **Catatan Analisis Kritis**: Sintesis ini mempertahankan tinjauan dialektis dari literatur yang tersedia. "
                        "Terdapat variasi metodologis antara temuan empiris dokumen dengan implikasi kebijakan di lapangan."
                    )
                    final_response = self._sanitize_final_response(raw_final) + dialectic_caveat
                    if self.vault_manager and final_response:
                        final_response = append_vault_bibliography(
                            text=final_response,
                            evidence=evidence,
                            vault_manager=self.vault_manager,
                            style=self.config.citation.style,
                            active_documents=self.active_documents,
                        )
                    last_v_report = turn_result.get("v_report")
                    last_c_report = turn_result.get("c_report")
                    break

                elif turn_result["type"] == "ungrounded":
                    refinement_turns += 1
                    v_rep = turn_result["report"]
                    self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Memverifikasi Fakta[/bold bright_white] [yellow]{v_rep.hallucination_warning}[/yellow]")
                    unsupported_claims = [c.claim for c in v_rep.claims if c.status == "unsupported"]
                    if unsupported_claims:
                        locked_sec = ""
                        if locked_supported_claims:
                            locked_sec = (
                                "### KLAIM TERVERIFIKASI SEBELUMNYA (WAJIB DIPERTAHANKAN VERBATIM):\n"
                                "Pernyataan-pernyataan berikut telah terverifikasi valid terhadap dokumen vault. Pertahankan:\n"
                                + "\n".join(f"- {c}" for c in locked_supported_claims[:6]) + "\n\n"
                            )
                        current_query = (
                            f"{locked_sec}"
                            f"### VERIFIKASI FAKTA (CLAIM GROUNDING AUDIT):\n"
                            f"Beberapa klaim berikut tidak ditemukan dasarnya pada literatur vault:\n"
                            + "\n".join(f"- {c}" for c in unsupported_claims) +
                            f"\n\nPertanyaan riset asli: \"{query}\"\n"
                            f"Instruksi: Pertahankan seluruh klaim terverifikasi di atas. Pangkas atau sesuaikan klaim yang belum didukung agar 100% didukung oleh bukti teks dokumen vault."
                        )
                        turn += 1
                        continue

                elif turn_result["type"] == "needs_citation_mapping":
                    refinement_turns += 1
                    c_rep = turn_result["report"]
                    self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Devil's Advocate[/bold bright_white] [yellow]Skor {c_rep.score}/10: {c_rep.critique_summary}[/yellow]")
                    stems = self._get_cached_stems()
                    mapped_papers = []
                    newly_ingested_stems = []

                    # 1. Directed Gap Search via Typed Critique Queries
                    directed_queries = list(getattr(c_rep, "directed_search_queries", []))
                    v_rep = turn_result.get("v_report")
                    if v_rep and getattr(v_rep, "claims", None):
                        unsupported = [c.claim for c in v_rep.claims if c.status == "unsupported"]
                        for u_cl in unsupported[:2]:
                            clean_words = [w for w in re.findall(r"\b\w{4,}\b", u_cl) if w.lower() not in ("yang", "pada", "dari", "untuk", "dengan", "dalam", "adalah", "bahwa", "tidak")]
                            if len(clean_words) >= 2:
                                gap_q = " ".join(clean_words[:5])
                                if gap_q not in directed_queries:
                                    directed_queries.append(gap_q)

                    if directed_queries and self.scout:
                        for target_gap_q in directed_queries[:2]:
                            self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Directed Gap Search[/bold bright_white] [cyan]Menelusuri literatur spesifik penambal celah: '{target_gap_q}'...[/cyan]")
                            try:
                                sq = ScoutQuery(keywords=target_gap_q, limit=2)
                                candidates = asyncio.run(self.scout.search_multi_source(sq, limit=2))
                                if candidates:
                                    existing_titles = {p.title for p in mapped_papers}
                                    mapped_papers.extend([p for p in candidates if p.title not in existing_titles])
                            except Exception:
                                pass

                    # 2. Reference Mapping (Vault citation network)
                    if stems and self.scout:
                        self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Reference Mapping[/bold bright_white] [cyan]Menelusuri jaringan sitasi dokumen vault...[/cyan]")
                        try:
                            t_map_s = time.time()
                            cit_papers = asyncio.run(self.scout.map_vault_citation_network(stems, limit=2))
                            if cit_papers:
                                existing_titles = {p.title for p in mapped_papers}
                                mapped_papers.extend([p for p in cit_papers if p.title not in existing_titles])
                            if hasattr(self, "logger") and self.logger:
                                self.logger.log_citation_mapping(
                                    session_id=session_id,
                                    stems=stems,
                                    found_papers=[p.title for p in mapped_papers],
                                    duration_s=time.time() - t_map_s,
                                )
                        except Exception:
                            pass

                    # Auto-download mapped papers or evaluate paywall fallback
                    if mapped_papers and self.scout:
                        for idx, p in enumerate(mapped_papers, 1):
                            safe_stem = sanitize_filename(p.title)
                            downloaded = False
                            if p.has_pdf and p.pdf_url:
                                self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Mengunduh Sitasi Terhubung[/bold bright_white] [cyan][{idx}/{len(mapped_papers)}] {p.title[:50]}...[/cyan] [dim](Open-Access PDF)[/dim]")
                                try:
                                    asyncio.run(self.scout.download_scout_paper(p))
                                    newly_ingested_stems.append(safe_stem)
                                    downloaded = True
                                except Exception:
                                    downloaded = False

                            if not downloaded:
                                # Journal is paywalled or download failed
                                self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Paywall Terdeteksi[/bold bright_white] [yellow][{idx}/{len(mapped_papers)}] {p.title[:50]}...[/yellow] [dim](Naskah berbayar)[/dim]")
                                self.console.print(f"[dim]● Mengevaluasi kecukupan abstrak resmi dengan AI...[/dim]")
                                decision = self._ask_ai_paywall_action(p, query)

                                if decision.get("decision") == "FIND_ALTERNATIVE":
                                    self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Mencari Alternatif Open-Access[/bold bright_white] [cyan]{p.title[:45]}...[/cyan] [dim]({decision.get('reason', '')})[/dim]")
                                    alt_oa = []
                                    try:
                                        sq = ScoutQuery(keywords=f"{p.title} open access", is_oa_only=True)
                                        alt_candidates = asyncio.run(self.scout.search_multi_source(sq, limit=5))
                                        alt_oa = [c for c in alt_candidates if c.has_pdf and c.pdf_url]
                                    except Exception:
                                        pass

                                    alt_downloaded = False
                                    if alt_oa:
                                        alt_paper = alt_oa[0]
                                        self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Mengunduh Alternatif Open-Access[/bold bright_white] [cyan]{alt_paper.title[:50]}... (PDF)[/cyan]")
                                        try:
                                            asyncio.run(self.scout.download_scout_paper(alt_paper))
                                            newly_ingested_stems.append(sanitize_filename(alt_paper.title))
                                            alt_downloaded = True
                                        except Exception:
                                            alt_downloaded = False

                                    if not alt_downloaded:
                                        self.console.print(f"[dim]● Alternatif full-text tidak tersedia, mengindeks abstrak resmi publikasi.[/dim]")
                                        self.scout.ingest_abstract_paper(p)
                                        newly_ingested_stems.append(safe_stem)
                                else:
                                    self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Abstrak Memadai[/bold bright_white] [green]Mengindeks abstrak resmi paper ke vault[/green] [dim]({decision.get('reason', '')})[/dim]")
                                    self.scout.ingest_abstract_paper(p)
                                    newly_ingested_stems.append(safe_stem)

                    if newly_ingested_stems:
                        if self.vault_manager:
                            self.vault_manager.reload_vault(force=False)
                        self._sync_vault_cache()
                        for s in newly_ingested_stems:
                            self.active_documents.add(s)

                        active_paths = [self.workspace.get_cache_path(f"{s}.md") for s in self.active_documents]
                        existing_paths = [p for p in active_paths if p.exists()]
                        current_docs = list(existing_paths)
                        evidence = self.engine.evidence_synthesizer.retrieve_evidence(
                            query=query,
                            documents=current_docs,
                            max_chunks=self.config.citation.max_evidence_chunks,
                        )
                        citation_guide = self._format_vault_citation_guide()
                        self.console.print(f"[bold green][✓] Berhasil mengindeks {len(newly_ingested_stems)} dokumen sitasi baru ke vault.[/bold green]")
                        for s in newly_ingested_stems:
                            self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Membaca[/bold bright_white] [cyan]cache/{s}.md[/cyan] [dim](Siap untuk sintesis)[/dim]")
                        self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Menganalisis[/bold bright_white] [dim]Menyintesis argumen ilmiah terpadu dengan literatur baru...[/dim]\n")
                        max_agent_turns = max(max_agent_turns, turn + 2)

                    lines_map = [f"- {p.title} ({p.year or 'n.d.'}) [{p.relevance_reason}]" for p in mapped_papers]
                    extra_ref = ""
                    if lines_map:
                        extra_ref = "### HASIL REFERENCE MAPPING (JARINGAN SITASI VAULT TERUNDUH):\nPaper terhubung yang baru diindeks:\n" + "\n".join(lines_map) + "\n\n"

                    weaknesses_str = ", ".join(c_rep.weaknesses) if c_rep.weaknesses else "Perkuat landasan argumen dan sertakan batasan metodologis."
                    counter_str = ("\nPotensi sanggahan: " + ", ".join(c_rep.counter_arguments)) if c_rep.counter_arguments else ""
                    recency_str = f"\nCatatan kebaruan: {c_rep.recency_alert}" if c_rep.recency_alert else ""

                    locked_sec = ""
                    if locked_supported_claims:
                        locked_sec = (
                            "### KLAIM TERVERIFIKASI SEBELUMNYA (WAJIB DIPERTAHANKAN VERBATIM):\n"
                            "Pernyataan-pernyataan berikut telah terverifikasi valid terhadap dokumen vault. Pertahankan:\n"
                            + "\n".join(f"- {c}" for c in locked_supported_claims[:6]) + "\n\n"
                        )

                    unsupported_sec = ""
                    if v_rep and getattr(v_rep, "claims", None):
                        unsupported = [c.claim for c in v_rep.claims if c.status == "unsupported"]
                        if unsupported:
                            unsupported_sec = (
                                "### KLAIM YANG MEMBUTUHKAN BUKTI DARI LITERATUR BARU:\n"
                                "Klaim berikut sebelumnya belum memiliki bukti di vault. Manfaatkan literatur baru yang baru diindeks di atas untuk memverifikasi dan memperkuatnya:\n"
                                + "\n".join(f"- {c}" for c in unsupported[:5]) + "\n\n"
                            )

                    current_query = (
                        f"{citation_guide}\n\n"
                        f"{locked_sec}"
                        f"{extra_ref}### EVALUASI DEVIL'S ADVOCATE (SKOR {c_rep.score}/10):\n"
                        f"Catatan kelemahan: {weaknesses_str}{counter_str}{recency_str}\n\n"
                        f"{unsupported_sec}"
                        f"Pertanyaan riset asli: \"{query}\"\n"
                        f"Instruksi: Pertahankan seluruh klaim terverifikasi di atas. Mentailor ulang sintesis untuk mengatasi catatan kelemahan di atas dengan memanfaatkan bukti literatur baru yang telah diindeks di vault, sertakan sitasi in-text APA pada setiap klaim, dan berikan jawaban yang komprehensif serta mantap (100% yakin)."
                    )
                    turn += 1
                    continue

                else:
                    raw_final = turn_result.get("response", "")
                    final_response = self._sanitize_final_response(raw_final)
                    last_v_report = turn_result.get("v_report")
                    last_c_report = turn_result.get("c_report")
                    break

            elapsed = time.time() - start_time

            # 3. Clean academic Markdown with LaTeX math beautification
            render_academic_markdown(final_response, console=self.console)
            self.console.print()

            # 4. Accurate cumulative token consumption telemetry across all turns
            evidence_tokens = sum(c.token_estimate or estimate_tokens(c.text) for c in evidence)
            fallback_in = estimate_tokens(query) + 250 + evidence_tokens
            fallback_out = estimate_tokens(final_response)
            input_tokens = max(accumulated_input_tokens, fallback_in)
            output_tokens = max(accumulated_output_tokens, fallback_out)
            total_tokens = input_tokens + output_tokens
            style_name = self.config.citation.style.upper()

            # Reuse already evaluated reports from the turn pipeline (Zero duplicate LLM calls)
            v_score = last_v_report.grounding_score if last_v_report else 1.0
            c_score = last_c_report.score if last_c_report else 8
            grounding_badge = f"Grounding: {int(v_score * 100)}%"
            critic_badge = f"Critic: {c_score}/10"
            actual_turns = max(total_turns_completed, turn + 1)
            turns_badge = f"{actual_turns} turn" if actual_turns == 1 else f"{actual_turns} turns"

            # Permanent telemetry footer stored in chat database for persistent retrieval
            footer_badge_md = (
                f"\n\n---\n"
                f"*[dim]Tokens: {input_tokens:,} in · {output_tokens:,} out · {total_tokens:,} total ({model_label}) • {turns_badge} • {grounding_badge} • {critic_badge} • {elapsed:.1f}s • {style_name} style[/dim]*"
            )
            final_response_persisted = f"{final_response}{footer_badge_md}"

            self.console.print(
                f"[dim]── Tokens: {input_tokens:,} in · {output_tokens:,} out · {total_tokens:,} total ({model_label}) • {turns_badge} • {grounding_badge} • {critic_badge} • {elapsed:.1f}s • {style_name} style ──[/dim]\n"
            )

            if hasattr(self, "logger") and self.logger:
                self.logger.log_response_rendered(
                    session_id=session_id,
                    total_input_tokens=input_tokens,
                    total_output_tokens=output_tokens,
                    elapsed_total_s=elapsed,
                    model_label=model_label,
                    grounding_score=v_score,
                    critic_score=c_score,
                )

            if self.chat_manager:
                self.chat_manager.add_message(session.id, "assistant", final_response_persisted)
            self.conversation_history.append({
                "query": query,
                "response": final_response_persisted,
            })

            # 5. Suggested actions & Action Indicator: Mengubah / Menulis
            self._handle_suggested_actions(final_response)

        except SessionDetachSignal:
            if session and self.chat_manager:
                self.chat_manager.update_session_status(session.id, "detached")

            # Handoff query to background TaskManager so AI continues running
            def _bg_worker(task_id: str) -> str:
                self.task_manager.log(task_id, f"Pertanyaan: {query}")
                self.task_manager.log(task_id, f"Model: {model_label} (effort: {effort})")
                self.task_manager.log(task_id, f"Melanjutkan sintesis di background untuk sesi '{session.name if session else ''}'...")
                bg_tokens = list(last_turn_tokens) if last_turn_tokens else []
                async def _bg_stream() -> None:
                    async for tok in self.engine.stream_query(
                        query=current_query,
                        documents=current_docs,
                        skill_name=self.active_skill_name,
                        model=model_name,
                        max_evidence_chunks=0,
                        history=history_msgs,
                    ):
                        bg_tokens.append(tok)

                b_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(b_loop)
                try:
                    b_loop.run_until_complete(_bg_stream())
                finally:
                    b_loop.close()

                final_text = "".join(bg_tokens).strip()
                if final_text:
                    try:
                        polisher = self.polisher or AcademicPolisher(engine=self.engine)
                        p_res = polisher.polish(final_text, query)
                        if p_res.applied:
                            final_text = p_res.polished_text
                    except Exception:
                        pass

                    if self.vault_manager:
                        final_text = append_vault_bibliography(
                            text=final_text,
                            evidence=evidence,
                            vault_manager=self.vault_manager,
                            style=self.config.citation.style,
                        )

                if self.chat_manager and session and final_text:
                    self.chat_manager.add_message(session.id, "assistant", final_text)
                    self.chat_manager.update_session_status(session.id, "detached")

                ev_tokens = sum(c.token_estimate or estimate_tokens(c.text) for c in evidence) if evidence else 0
                in_tok = estimate_tokens(query) + 250 + ev_tokens
                out_tok = estimate_tokens(final_text)
                tot_tok = in_tok + out_tok
                elapsed_bg = time.time() - start_time

                self.task_manager.update_tokens(task_id, tot_tok)
                self.task_manager.log(task_id, f"Token: {in_tok:,} in · {out_tok:,} out · {tot_tok:,} total ({model_label}) • {elapsed_bg:.1f}s")
                self.task_manager.log(task_id, f"Selesai! {len(final_text)} karakter disimpan ke riwayat sesi.")
                return f"Sintesis selesai ({len(final_text)} karakter, {tot_tok:,} total token)" if final_text else "Selesai"

            task = self.task_manager.submit(
                name=f"Chat ({session.name if session else 'session'}): {query[:30]}...",
                task_type="agent",
                target_fn=_bg_worker,
                session_id=session.id if session else None,
            )

            self.console.print(f"\n[bold cyan]● Sesi '{session.name if session else ''}' di-detach ke latar belakang.[/bold cyan]")
            self.console.print(f"[dim]● Proses analisis dialihkan ke background (Task: {task.id}). Pantau di menu Processing atau buka kembali sesi kapan saja.[/dim]\n")
            raise

        except SessionCloseSignal:
            if session and self.chat_manager:
                self.chat_manager.update_session_status(session.id, "closed")
            self.console.print(f"\n[dim]● Sesi '{session.name if session else ''}' ditutup (terminated).\n[/dim]")
            raise

        except (KeyboardInterrupt, asyncio.CancelledError):
            self.console.print("\n\n[bold yellow]● Pertanyaan dibatalkan oleh pengguna (Ctrl+C).[/bold yellow]\n")
            # Stay inside active chat session; do not corrupt history with [Dibatalkan]
            return
        except Exception as exc:
            self.console.print(f"\n[bold red][x] Generation error:[/bold red] {exc}\n")

    def _execute_action_list(self, actions: list[str]) -> None:
        """Present interactive confirmation for a list of Scrpcy commands on the main thread."""
        if not actions:
            return

        # Deduplicate while preserving order
        unique_actions: list[str] = []
        for a in actions:
            if a not in unique_actions:
                unique_actions.append(a)

        # Coalesce multiple separate /vault remove lines into a single bulk command
        remove_args: list[str] = []
        filtered_actions: list[str] = []
        for a in unique_actions:
            clean_a = a.strip()
            if any(clean_a.startswith(k) for k in ("/vault remove", "vault remove", "/vault rm", "vault rm", "/vault archive", "vault archive")):
                parts = clean_a.split(maxsplit=2)
                if len(parts) > 2:
                    clean_arg = parts[2].replace(",", " ")
                    tokens = [t.strip().strip('"\'') for t in clean_arg.split() if t.strip().strip('"\'')]
                    remove_args.extend(tokens)
                elif len(parts) == 2 and parts[1].strip().strip('"\'').isdigit():
                    remove_args.append(parts[1].strip().strip('"\''))
            else:
                filtered_actions.append(a)

        if remove_args:
            deduped_args = list(dict.fromkeys(remove_args))
            filtered_actions.append(f"/vault remove {' '.join(deduped_args)}")

        unique_actions = list(dict.fromkeys(filtered_actions))

        for action in unique_actions:
            # Action Pill: Mengubah / Menulis
            if "docx " in action:
                doc_arg = action.split("docx", 1)[1].strip()
                self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Menulis[/bold bright_white] [cyan]drafts/{doc_arg}[/cyan]")
            elif "vault add " in action:
                v_arg = action.split("vault add", 1)[1].strip()
                self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Mengubah[/bold bright_white] [cyan]vault/{v_arg}[/cyan]")
            elif any(k in action for k in ("vault remove", "vault rm", "vault archive")):
                _rm_parts = action.split(maxsplit=2)
                v_arg = _rm_parts[2].strip() if len(_rm_parts) > 2 else ""
                ids_display = [t.strip().strip('"\'') for t in v_arg.replace(",", " ").split() if t.strip().strip('"\'')]
                if len(ids_display) > 1:
                    self.console.print(
                        f"[bold cyan]●[/bold cyan] [bold bright_white]Mengarsipkan[/bold bright_white] "
                        f"[cyan]{len(ids_display)} dokumen -> vault/.archive/[/cyan] "
                        f"[dim]({', '.join(ids_display)})[/dim]"
                    )
                else:
                    self.console.print(
                        f"[bold cyan]●[/bold cyan] [bold bright_white]Mengarsipkan[/bold bright_white] "
                        f"[cyan]vault/{v_arg} -> vault/.archive/[/cyan]"
                    )
            elif action.strip() in ("/clear", "clear"):
                self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Membersihkan[/bold bright_white] [cyan]Konteks chat aktif[/cyan]")

            choice_res = interactive_action_confirm(action, console=self.console)
            if isinstance(choice_res, tuple):
                action_choice, user_note = choice_res
            else:
                action_choice = "execute" if choice_res else "skip"
                user_note = None

            if action_choice == "execute":
                self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Menjalankan aksi:[/bold bright_white] [cyan]{action}[/cyan]\n")
                if action.strip() in ("/clear", "clear"):
                    self.clear_screen()
                else:
                    cmd_to_run = action if action.startswith("/") else f"/{action}"
                    self.handle_project_command(cmd_to_run)
            elif action_choice == "write_in" and user_note:
                self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Masukan pengguna dikirim ke AI:[/bold bright_white] [italic]{user_note}[/italic]\n")
                feed_query = (
                    f"Pengguna meninjau saran aksi Anda `{action}` dan memberikan arahan/masukan berikut:\n"
                    f"\"{user_note}\"\n\n"
                    f"Silakan sesuaikan analisis, strategi investigasi, atau rencana kurasi Anda berdasarkan masukan tersebut."
                )
                self.handle_chat(feed_query)
                break  # stop further actions to let AI adapt to user guidance
            else:
                self.console.print(f"[dim]Aksi '{action}' dilewati.[/dim]\n")

    def _handle_suggested_actions(self, response_text: str) -> None:
        """Detect and offer interactive execution for Scrpcy action blocks in AI response."""
        pattern = re.compile(r"```scrpcy\r?\n(.*?)\r?\n```", re.DOTALL | re.IGNORECASE)
        matches = pattern.findall(response_text)
        if not matches:
            return

        has_ok_signal = "[status: ok]" in response_text.lower()
        banned_internal_roots = [
            "/vault read", "vault read", "/vault view", "vault view",
            "/vault list", "vault list", "/read", "read", "/view", "view",
            "/list", "list",
        ]
        if has_ok_signal:
            banned_internal_roots.extend([
                "/scout", "scout", "/findpaper", "findpaper", "/search", "search",
            ])

        actions = [
            cmd.strip() for m in matches for cmd in m.splitlines()
            if cmd.strip() and not cmd.strip().startswith("#")
            and not any(cmd.strip().lower().startswith(b) for b in banned_internal_roots)
        ]
        if not actions:
            return
        self._execute_action_list(actions)
