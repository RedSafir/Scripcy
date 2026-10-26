"""Module part of scripcy studio."""

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

from scripcy.core.chats import ChatManager, ChatSession, SessionEvent, SessionRuntime
from scripcy.core.config import ConfigManager, ScripcyConfig, detect_9router_key, fetch_9router_models
from scripcy.core.contracts import AdaptiveCriticVerdict, TaskContract, TaskContractExtractor
from scripcy.core.critic import CritiqueReport, DevilsAdvocate
from scripcy.core.engine import LLMEngine, estimate_tokens
from scripcy.core.gdrive import GoogleDriveManager, extract_drive_file_id
from scripcy.core.logger import StudioLogger, get_studio_logger
from scripcy.core.mcp_client import WordCraftClient
from scripcy.core.models import DocumentMetadata, ParsedDocument
from scripcy.core.parser import DocumentParser
from scripcy.core.polisher import AcademicPolisher, PolishResult
from scripcy.core.project import ProjectInfo, ProjectManager
from scripcy.core.references import append_vault_bibliography
from scripcy.core.scout import LiteratureScout, ScoutQuery, ScoutResult
from scripcy.core.search import PaperSearcher, SearchResult, sanitize_filename
from scripcy.core.skills import SkillsLoader
from scripcy.core.tasks import TaskInfo, TaskManager
from scripcy.core.tracker import TokenTracker
from scripcy.core.vault import VaultItem, VaultManager
from scripcy.core.verifier import FactChecker, VerificationReport
from scripcy.core.web_scout import WebScout
from scripcy.core.workspace import WorkspaceManager

from scripcy.studio.signals import (
    SCRIPCY_LOGO,
    SCRIPCY_VERSION,
    SessionCloseSignal,
    SessionDetachSignal,
    _AgentProgressTracker,
)
from scripcy.ui.formatters import render_academic_markdown
from scripcy.ui.interactive import (
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


class StudioBase:
    def __init__(
        self,
        workspace: WorkspaceManager | None = None,
        config: ScripcyConfig | None = None,
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
        self.active_skill_name = "scripcy_manual"
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
        self._vault_sync_lock = threading.RLock()
        self._pending_ui_events: list[Callable[[], None]] = []
        self._pending_user_input: str = ""
        self._transient_toasts: list[dict[str, Any]] = []
        self._known_task_states: dict[str, str] = {}

        self.current_project: ProjectInfo | None = None
        self.workspace: WorkspaceManager | None = None
        self.parser: DocumentParser | None = None
        self.wordcraft: WordCraftClient | None = None
        self.engine: LLMEngine | None = None
        self.scout: LiteratureScout | None = None
        self.web_scout: WebScout | None = None
        self.contract_extractor: TaskContractExtractor | None = None

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
        self.web_scout = WebScout(workspace=self.workspace)
        self.vault_manager = VaultManager(workspace=self.workspace)
        self.chat_manager = ChatManager(self.workspace.root_dir / "chats.db")
        existing_sessions = self.chat_manager.list_sessions()
        self.active_chat_session = existing_sessions[0] if existing_sessions else None
        if hasattr(self, "task_manager") and self.task_manager:
            self.task_manager.set_db_path(self.workspace.root_dir / "tasks.db")
        self.sub_mode = None
        self.fact_checker = FactChecker(engine=self.engine)
        self.devils_advocate = DevilsAdvocate(engine=self.engine)
        self.contract_extractor = TaskContractExtractor(engine=self.engine)
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
        with self._vault_sync_lock:
            return [p.stem for p in self.workspace.list_cached_documents()]

    def _sync_vault_cache(self, session_id: str | None = None) -> list[str]:
        """Thread-safely index available markdown caches into project and session runtimes."""
        with self._vault_sync_lock:
            cached_files = self._get_cached_stems()
            for stem in cached_files:
                self.active_documents.add(stem)
            if self.chat_manager:
                self.chat_manager.sync_all_runtimes_documents(cached_files)
                if session_id:
                    self.chat_manager.get_runtime(session_id, initial_documents=cached_files)
            return cached_files

    def get_session_runtime(self, session_id: str | None = None) -> SessionRuntime | None:
        """Return isolated SessionRuntime for the target or active session."""
        if not self.chat_manager:
            return None
        target_id = session_id or (self.active_chat_session.id if self.active_chat_session else None)
        if not target_id:
            return None
        return self.chat_manager.get_runtime(target_id, initial_documents=self.active_documents)

    def get_session_active_documents(self, session_id: str | None = None) -> set[str]:
        """Return isolated set of active document stems for a session (falling back to workspace set)."""
        rt = self.get_session_runtime(session_id)
        if rt is not None:
            docs = rt.get_documents()
            if docs or not self.active_documents:
                return docs
            rt.add_documents(self.active_documents)
            return rt.get_documents()
        with self._vault_sync_lock:
            return set(self.active_documents)

    def add_session_active_documents(self, stems: list[str] | set[str], session_id: str | None = None) -> None:
        """Thread-safely register newly ingested document stems into session and studio state."""
        if not stems:
            return
        with self._vault_sync_lock:
            for s in stems:
                if s:
                    self.active_documents.add(s)
            rt = self.get_session_runtime(session_id)
            if rt is not None:
                rt.add_documents(stems)

    def is_session_foreground(self, session_id: str | None, session_obj: ChatSession | None = None) -> bool:
        """Return True ONLY if `session_id` is currently attached and rendered in the foreground chat view."""
        if session_obj is not None and getattr(session_obj, "status", "running") == "detached":
            return False
        if not session_id:
            return self.sub_mode in (None, "chat")
        if self.chat_manager:
            persisted = self.chat_manager.get_session(session_id)
            if persisted and persisted.status == "detached":
                return False
        if self.active_chat_session and self.active_chat_session.id == session_id:
            if getattr(self.active_chat_session, "status", "running") == "detached":
                return False
            return self.sub_mode in (None, "chat")
        return False

    def emit_session_output(
        self,
        session_id: str | None,
        message: str,
        session_obj: ChatSession | None = None,
    ) -> None:
        """UI Arbitrator: print to console if session is attached in foreground; otherwise spool to session buffer."""
        if self.is_session_foreground(session_id, session_obj=session_obj):
            self.console.print(message)
        else:
            rt = self.get_session_runtime(session_id)
            if rt is not None:
                rt.append_output(message)

    def print_banner(self) -> None:
        """Render the large ASCII art logo at most once."""
        if not self._has_printed_banner:
            banner_lines = "\n".join(f" {line}" for line in SCRIPCY_LOGO.strip().splitlines())
            self.console.print(f"[bold cyan]{banner_lines}[/bold cyan]\n")
            self.console.print(f" [dim]{SCRIPCY_VERSION} • Local-First Academic Research Studio[/dim]\n")
            self._has_printed_banner = True

    def print_dashboard_banner(self) -> None:
        """Render minimal borderless dashboard status without command suggestion blocks."""
        projects = self.project_manager.list_projects()
        p_count = len(projects)
        self.console.print(
            f" [bold cyan]scripcy[/bold cyan] [dim]•[/dim] [dim]dashboard ({p_count} projects)[/dim]"
        )

    def render_mode_divider(self) -> None:
        """Deprecated: Outlier horizontal lines are exclusively reserved for input framing."""
        pass

    def add_transient_toast(self, message: str, level: str = "info", duration_seconds: float = 3.5) -> None:
        """Add non-intrusive transient notification displayed above active input prompt."""
        self._transient_toasts.append({
            "message": message,
            "level": level,
            "created_at": time.monotonic(),
            "duration": duration_seconds,
        })

    def get_live_status_indicator(self, session_id: str | None = None) -> str | None:
        """Return seamless transient status and background worker indicators for the active prompt."""
        now = time.monotonic()

        # 1. State change detection: auto-generate toasts for background tasks
        if hasattr(self, "task_manager") and self.task_manager:
            try:
                all_tasks = self.task_manager.list_tasks()
                for t in all_tasks:
                    prev_st = self._known_task_states.get(t.id)
                    if prev_st == "running" and t.status != "running":
                        if t.status == "completed":
                            sum_txt = f" [dim]({t.result_summary})[/dim]" if getattr(t, "result_summary", None) else ""
                            self.add_transient_toast(
                                f"[bold green]✓ Selesai:[/bold green] [bold white]{t.name}[/bold white]{sum_txt}",
                                level="success",
                                duration_seconds=3.5,
                            )
                        elif t.status == "failed":
                            err_txt = f" [dim]({t.error})[/dim]" if getattr(t, "error", None) else ""
                            self.add_transient_toast(
                                f"[bold red][!] Gagal:[/bold red] [bold white]{t.name}[/bold white]{err_txt}",
                                level="error",
                                duration_seconds=4.0,
                            )
                        elif t.status == "cancelled":
                            self.add_transient_toast(
                                f"[dim][x] Dibatalkan: {t.name}[/dim]",
                                level="warning",
                                duration_seconds=3.0,
                            )
                    self._known_task_states[t.id] = t.status
            except Exception:
                pass

        # 2. Prune expired toasts
        self._transient_toasts = [
            t for t in self._transient_toasts
            if now - t["created_at"] < t.get("duration", 3.5)
        ]

        toast_line = None
        if self._transient_toasts:
            toast_line = self._transient_toasts[-1]["message"]

        # 3. Active running / queued tasks indicator
        task_line = None
        if hasattr(self, "task_manager") and self.task_manager:
            try:
                if session_id:
                    active_tasks = [
                        t for t in self.task_manager.list_tasks_for_session(session_id)
                        if getattr(t, "is_active", t.status == "running")
                    ]
                else:
                    active_tasks = [
                        t for t in self.task_manager.list_tasks()
                        if getattr(t, "is_active", t.status == "running")
                    ]

                if active_tasks:
                    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                    frame = SPINNER_FRAMES[int(now * 8) % len(SPINNER_FRAMES)]
                    if len(active_tasks) == 1:
                        t = active_tasks[0]
                        desc = t.progress or "Memproses di background..."
                        task_line = f"[bold cyan]{frame}[/bold cyan] [bold bright_white]{t.name}[/bold bright_white] [dim]({desc})[/dim]"
                    else:
                        running_cnt = sum(1 for t in active_tasks if t.status == "running")
                        waiting_cnt = len(active_tasks) - running_cnt
                        if waiting_cnt > 0:
                            task_line = f"[bold cyan]{frame}[/bold cyan] [bold bright_white]{running_cnt} proses berjalan, {waiting_cnt} menunggu antrean dependensi...[/bold bright_white]"
                        else:
                            task_line = f"[bold cyan]{frame}[/bold cyan] [bold bright_white]{len(active_tasks)} proses latar belakang aktif...[/bold bright_white]"
            except Exception:
                pass

        # 4. If in chat session, check if OTHER sessions have running tasks
        other_session_notice = None
        if session_id and hasattr(self, "task_manager") and self.task_manager and not task_line:
            try:
                other_tasks = [
                    t for t in self.task_manager.list_tasks()
                    if getattr(t, "is_active", t.status == "running") and t.session_id != session_id
                ]
                if other_tasks:
                    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                    frame = SPINNER_FRAMES[int(now * 8) % len(SPINNER_FRAMES)]
                    other_session_notice = f"[dim cyan]{frame} {len(other_tasks)} background task aktif di sesi lain (/proc)[/dim cyan]"
            except Exception:
                pass

        parts = []
        if toast_line:
            parts.append(toast_line)
        if task_line:
            parts.append(task_line)
        elif other_session_notice:
            parts.append(other_session_notice)

        if not parts:
            return None
        return "\n".join(parts)

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

                init_buf = self._pending_user_input
                self._pending_user_input = ""

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
                    status_line_callback=self.get_live_status_indicator,
                    initial_buffer=init_buf,
                ).strip()

                if not raw_input:
                    continue

                if self.sub_mode is None and raw_input in ("/exit", "/quit", "exit", "quit", "q"):
                    self.console.print("\n[bold cyan]Closing scripcy research studio. Goodbye![/bold cyan]")
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

                    self.console.print("\n[bold cyan]Closing scripcy research studio. Goodbye![/bold cyan]")

                    break

            except Exception as exc:

                self.console.print(f"\n[bold red][!] Error:[/bold red] {escape(str(exc))}\n")

                time.sleep(1.5)

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

            header.append("scripcy", style="bold cyan")

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

                self.run_interactive_settings()

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
            f" [bold cyan]scripcy[/bold cyan] [dim]•[/dim] [bold bright_white]{self.current_project.name}[/bold bright_white]  "
            f"[dim]•[/dim]  [bold cyan]{model_label}[/bold cyan] [dim]({effort})[/dim]  "
            f"[dim]•[/dim]  [dim]vault: {cached_count} indexed ({active_count} active)[/dim]"
        )

    def clear_screen(self) -> None:
        """Clear the terminal screen and reprint logo."""
        os.system("cls" if os.name == "nt" else "clear")
        init_windows_terminal()
        self.print_banner()

