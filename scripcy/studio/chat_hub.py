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

from scripcy.core.chats import ChatManager, ChatSession
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


class StudioChatHubMixin:
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
        if not sys.stdin.isatty():
            self.cmd_chat_list_sessions()
            return

        while True:
            self._reconcile_session_lifecycle()
            sessions = self.chat_manager.list_sessions() if self.chat_manager else []
            if not sessions:
                self.console.print("\n[dim]Belum ada sesi percakapan. Buat sesi baru untuk memulai.\n[/dim]")
                time.sleep(1.2)
                return

            columns = [
                {"name": "#", "justify": "right", "width": 3, "style": "dim"},
                {"name": "Session Name", "style": "bold white", "width": 30},
                {"name": "Status", "justify": "center", "width": 16},
                {"name": "Exchanges", "justify": "center", "width": 12, "style": "dim"},
                {"name": "Last Updated", "style": "dim", "width": 18},
            ]

            def _row_builder(idx: int, s: Any, is_selected: bool) -> list[Any]:
                st = getattr(s, "status", "running")
                active_tasks = (
                    self.task_manager.list_active_tasks_for_session(s.id)
                    if hasattr(self, "task_manager") and self.task_manager
                    else []
                )
                if st == "running":
                    st_badge = f"[bold cyan][Running • {len(active_tasks)}][/bold cyan]" if active_tasks else "[bold cyan][Running][/bold cyan]"
                elif st == "detached":
                    st_badge = f"[bold yellow][Detached • {len(active_tasks)}][/bold yellow]" if active_tasks else "[dim cyan][Detached][/dim cyan]"
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
                    f"{s.message_count} msgs",
                    last_str,
                ]

            header = Text()
            header.append(f"Saved Chat Sessions [{self.current_project.name}]", style="bold bright_white")
            header.append(f" • {len(sessions)} sessions", style="dim")

            actions = [
                ("attach", "Attach / Open", "enter"),
                ("new", "New Session", "n"),
                ("terminate", "Close / Terminate", "x"),
                ("delete", "Delete Session", "d"),
                ("back", "Back", "esc/q"),
            ]

            action, idx, selected_sess = interactive_item_browser(
                title=header,
                columns=columns,
                items=sessions,
                row_builder=_row_builder,
                actions=actions,
                console=self.console,
                empty_message="Tidak ada sesi percakapan.",
                items_provider=self.chat_manager.list_sessions if self.chat_manager else None,
                poll_interval=0.5,
            )

            if action in ("back", "esc"):
                return
            elif action == "new":
                self.interactive_create_chat_session()
            elif action == "attach" and selected_sess:
                st = getattr(selected_sess, "status", "running")
                if st in ("closed", "detached"):
                    self.chat_manager.update_session_status(selected_sess.id, "running")
                    selected_sess.status = "running"
                self.active_chat_session = selected_sess
                self.active_skill_name = selected_sess.active_skill
                self.sub_mode = "chat"
                self.run_interactive_chat_session(selected_sess)
                self.sub_mode = None
                self._reconcile_session_lifecycle(selected_sess.id)

            elif action == "start" and selected_sess:
                self.chat_manager.update_session_status(selected_sess.id, "running")
                selected_sess.status = "running"
                self.active_chat_session = selected_sess
                self.active_skill_name = selected_sess.active_skill
                self.sub_mode = "chat"
                self.run_interactive_chat_session(selected_sess)
                self.sub_mode = "sessions"
                self._reconcile_session_lifecycle(selected_sess.id)

            elif action == "terminate" and selected_sess:
                self.chat_manager.update_session_status(selected_sess.id, "closed")
                if hasattr(self, "task_manager") and self.task_manager:
                    self.task_manager.cancel_session_tasks(selected_sess.id)

            elif action == "delete" and selected_sess:
                confirmed = interactive_confirm(
                    f"Hapus permanen riwayat chat '{selected_sess.name}'?",
                    default=True,
                    yes_label="Ya, hapus riwayat chat",
                    no_label="Batal",
                    console=self.console,
                )
                if confirmed:
                    if hasattr(self, "task_manager") and self.task_manager:
                        self.task_manager.cancel_session_tasks(selected_sess.id)
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

    def _drain_and_render_session_buffer(self, session_id: str) -> None:
        """Flush any spooled background notifications recorded while the session was detached."""
        runtime = self.get_session_runtime(session_id)
        if runtime is not None:
            buffered_lines = runtime.drain_output_buffer()
            for line in buffered_lines:
                self.console.print(line)

    def run_interactive_chat_session(self, session: ChatSession) -> None:
        """Active Chat Session: Prompt user for messages, supports /detach and /close."""
        if self.chat_manager and getattr(session, "status", "running") != "running":
            self.chat_manager.update_session_status(session.id, "running")
            session.status = "running"

        model_name = self.get_active_project_model()
        model_label = _format_model_display_name(model_name)
        eff = self.get_active_project_effort()
        skill_lbl = f" • {session.active_skill}" if session.active_skill and session.active_skill != "scripcy_manual" else ""
        self.console.print(f"\n[bold cyan]● Sesi Aktif: {session.name}[/bold cyan] [dim]({model_label} • {eff} effort{skill_lbl})[/dim]")

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

        # Flush any spooled output that occurred while session was detached
        self._drain_and_render_session_buffer(session.id)

        # Show active background tasks bound to this session upon entering/re-entering
        session_tasks = self.task_manager.list_tasks_for_session(session.id) if self.task_manager else []
        running = [t for t in session_tasks if getattr(t, "is_active", t.status == "running")]
        if running:
            self.console.print(f"\n[bold yellow]●[/bold yellow] [bold bright_white]{len(running)} proses latar belakang sedang aktif di sesi ini:[/bold bright_white]")
            for t in running:
                st_tag = " [Waiting Deps]" if t.status == "waiting_deps" else ""
                self.console.print(f"  [bold cyan]●[/bold cyan] [cyan]{t.name}{st_tag}[/cyan] [dim]({t.progress or 'Sedang memproses...'} • ID: {t.id})[/dim]")
            self.console.print(" [dim]• Indikator proses aktif di bawah prompt. Ketik /proc untuk pantau atau /detach untuk kembali.[/dim]\n")

        while self.sub_mode == "chat" and self.active_chat_session:
            # Flush any background UI events and session buffer first
            self._flush_pending_ui_events()
            self._drain_and_render_session_buffer(session.id)

            # Process any pending suggested actions on the main thread
            if hasattr(self, "_pending_session_actions") and self._pending_session_actions.get(session.id):
                pending_acts = self._pending_session_actions.pop(session.id)
                self._execute_action_list(pending_acts)

            prompt_lbl = f"{self.current_project.name}/chat/{session.name}> "
            ctx_lbl = f"{self.current_project.name} • {session.name}"

            try:
                init_buf = self._pending_user_input
                self._pending_user_input = ""

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
                    status_line_callback=lambda: self.get_live_status_indicator(session.id),
                    initial_buffer=init_buf,
                ).strip()

            except (KeyboardInterrupt, EOFError):
                # Ctrl+C inside chat exits to sessions menu; auto-reap if no active tasks
                self.sub_mode = "chat_hub"
                self._reconcile_session_lifecycle(session.id)
                running_tasks = self.task_manager.list_active_tasks_for_session(session.id) if hasattr(self, "task_manager") and self.task_manager else []
                if running_tasks:
                    self.console.print(f"\n[bold cyan]● Sesi '{session.name}' di-detach ke latar belakang ({len(running_tasks)} background task berjalan).[/bold cyan]\n")
                else:
                    self.console.print(f"\n[dim]● Sesi '{session.name}' ditutup otomatis (tidak ada proses latar belakang).[/dim]\n")
                break

            if not raw_input:
                continue

            clean_lower = raw_input.lower().strip()

            if clean_lower in ("/detach", "detach", "/back", "back"):
                self.sub_mode = "chat_hub"
                self._reconcile_session_lifecycle(session.id)
                running_tasks = self.task_manager.list_active_tasks_for_session(session.id) if hasattr(self, "task_manager") and self.task_manager else []
                if running_tasks:
                    self.console.print(f"\n[bold cyan]● Sesi '{session.name}' di-detach ke latar belakang ({len(running_tasks)} background task berjalan).[/bold cyan]\n")
                else:
                    self.console.print(f"\n[dim]● Sesi '{session.name}' ditutup otomatis (tidak ada proses latar belakang).[/dim]\n")
                break

            if clean_lower in ("/close", "close", "/terminate", "terminate", "/exit", "exit"):
                self.chat_manager.update_session_status(session.id, "closed")
                session.status = "closed"
                if hasattr(self, "task_manager") and self.task_manager:
                    self.task_manager.cancel_session_tasks(session.id)

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

            prev_sess_id = self.active_chat_session.id if self.active_chat_session else None
            session = self.chat_manager.create_session(
                name=chat_name,
                active_skill=self.active_skill_name,
            )
            self.render_mode_divider()
            self.active_chat_session = session
            self.sub_mode = "chat"
            self.chat_manager.update_session_status(session.id, "running")
            session.status = "running"
            if prev_sess_id:
                self._reconcile_session_lifecycle(prev_sess_id)
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
                prev_sess_id = self.active_chat_session.id if self.active_chat_session else None
                if getattr(session, "status", "running") != "running":
                    self.chat_manager.update_session_status(session.id, "running")
                    session.status = "running"
                self.render_mode_divider()
                self.active_chat_session = session
                self.sub_mode = "chat"
                if prev_sess_id and prev_sess_id != session.id:
                    self._reconcile_session_lifecycle(prev_sess_id)
                self.console.print(f"[bold green][✓] Masuk ke sesi percakapan:[/bold green] [bold cyan]{session.name}[/bold cyan] ({session.message_count} pesan)")

                # Display any active or recent background tasks bound to this session
                session_tasks = self.task_manager.list_tasks_for_session(session.id)
                running = [t for t in session_tasks if getattr(t, "is_active", t.status == "running")]
                completed = [t for t in session_tasks if t.status == "completed"]
                if running:
                    self.console.print(f"[bold yellow]●[/bold yellow] [bold bright_white]{len(running)} proses latar belakang masih berjalan di sesi ini:[/bold bright_white]")
                    for t in running:
                        self.console.print(f"  [dim]•[/dim] [cyan]{t.name}[/cyan] [dim]({t.progress} • Ketik /proc untuk pantau)[/dim]")
                elif completed:
                    self.console.print(f"[bold green][✓][/bold green] [bold bright_white]Proses latar belakang terakhir di sesi ini:[/bold bright_white]")
                    for t in completed[:2]:
                        self.console.print(f"  [dim]•[/dim] [cyan]{t.name}[/cyan] [dim]({t.result_summary or 'Selesai'})[/dim]")

                self._drain_and_render_session_buffer(session.id)
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
                    if hasattr(self, "task_manager") and self.task_manager:
                        self.task_manager.cancel_session_tasks(s.id)
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
                    if hasattr(self, "task_manager") and self.task_manager:
                        self.task_manager.cancel_session_tasks(session.id)
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
                if getattr(session, "status", "running") != "running":
                    self.chat_manager.update_session_status(session.id, "running")
                    session.status = "running"
                self.render_mode_divider()
                self.active_chat_session = session
                self.sub_mode = "chat"
                self.console.print(f"[bold green][✓] Masuk ke sesi percakapan:[/bold green] [bold cyan]{session.name}[/bold cyan] ({session.message_count} pesan)")
                self._drain_and_render_session_buffer(session.id)
                self.cmd_chat_history()
                self.console.print("[dim]Ketik pertanyaan riset Anda langsung di bawah ini. Gunakan '/' untuk perintah (contoh: /back).[/dim]\n")
                return

        self.console.print(f"[yellow]Perintah tidak dikenal: '{clean}'. Ketik '?' atau '/' untuk melihat perintah yang tersedia.[/yellow]\n")

    def cmd_chat_list_sessions(self) -> None:
        """Render table of all chat sessions stored in this project."""
        if not self.chat_manager:
            return

        self._reconcile_session_lifecycle()
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
                ai_lbl = f"AI ({self.active_chat_session.active_skill})" if self.active_chat_session and self.active_chat_session.active_skill and self.active_chat_session.active_skill != "scripcy_manual" else "AI"
                self.console.print(f"[bold cyan]{ai_lbl}:[/bold cyan]\n")
                render_academic_markdown(msg.content, console=self.console)
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
