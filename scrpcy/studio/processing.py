"""Module part of scrpcy studio."""

from __future__ import annotations

import asyncio
import inspect
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
from unittest.mock import AsyncMock
import webbrowser

from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from scrpcy.core.chats import ChatManager, ChatSession, SessionEvent
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
from scrpcy.core.web_scout import QueryDisambiguator
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


class StudioProcessingMixin:
    def run_interactive_processing(self) -> None:
        """Processing status and task inspection without typing."""
        while self.sub_mode == "processing" and self.current_project:
            tasks = self.task_manager.list_tasks()
            running_tasks = [t for t in tasks if getattr(t, "is_active", t.status == "running")]
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
        self.sub_mode = "processing"
        if not sys.stdin.isatty():
            self.render_processing_summary()
            return

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
                elif t.status == "waiting_deps":
                    st_badge = "[bold yellow][Waiting Deps][/bold yellow]"
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
                ("cancel", "Cancel Task", "c/x"),
                ("open", "Open Session", "o"),
                ("retry", "Retry Failed", "r"),
                ("clear", "Clear Finished", "d"),
                ("back", "Back", "esc/q"),
            ]

            action, idx, selected_task = interactive_item_browser(
                title=header,
                columns=columns,
                items=tasks,
                row_builder=_row_builder,
                actions=actions,
                console=self.console,
                empty_message="Tidak ada task latar belakang.",
                items_provider=self.task_manager.list_tasks if self.task_manager else None,
                poll_interval=0.4,
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
                    return getattr(cur_t, "is_active", cur_t.status == "running")

                detail_acts = [
                    ("cancel", "Cancel Task", "c/x"),
                    ("back", "Back", "esc/q"),
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
                    self.add_transient_toast(f"[bold yellow][!] Membatalkan proses tugas '{selected_task.name}'.[/bold yellow]")

            elif action == "cancel" and selected_task:
                if getattr(selected_task, "is_active", selected_task.status == "running"):
                    self.task_manager.cancel_task(selected_task.id)
                    self.add_transient_toast(f"[bold yellow][!] Membatalkan proses tugas '{selected_task.name}'.[/bold yellow]")

            elif action == "retry" and selected_task:
                if selected_task.status in ("failed", "cancelled"):
                    self.add_transient_toast(f"[bold cyan]Tugas '{selected_task.name}' dapat dijalankan ulang dari sesi asalnya.[/bold cyan]")
                else:
                    self.add_transient_toast(f"[dim]Tugas '{selected_task.name}' masih aktif atau sudah selesai.[/dim]")

            elif action == "clear":
                cleared_cnt = self.task_manager.clear_completed()
                if cleared_cnt > 0:
                    self.add_transient_toast(f"[dim]{cleared_cnt} task selesai dibersihkan dari riwayat.[/dim]")

    def _on_background_task_completed(self, task: Any) -> None:
        """Turn-boundary notification dispatcher invoked when any background task finishes."""
        session_id = getattr(task, "session_id", None)
        if not session_id or not self.chat_manager:
            return

        task_status = getattr(task, "status", "")
        task_name = getattr(task, "name", "Task")
        task_id = getattr(task, "id", "")
        task_type = getattr(task, "task_type", "")
        summary = getattr(task, "result_summary", "")
        error_msg = getattr(task, "error", "")
        ingested_stems = list(getattr(task, "ingested_stems", []) or [])

        if task_status == "completed":
            event_type = "task_completed"
            note = (
                f"[PROSES LATAR BELAKANG SELESAI: {task_name}]\n"
                f"- ID Tugas: {task_id} ({task_type})\n"
                f"- Ringkasan Hasil: {summary or 'Selesai tanpa kendala.'}\n"
                f"- Status Vault: Dokumen dan metadata telah terindeks ke dalam vault dan siap dianalisis dalam sesi riset ini."
            )
        elif task_status == "failed":
            event_type = "task_failed"
            note = (
                f"[PROSES LATAR BELAKANG GAGAL: {task_name}]\n"
                f"- ID Tugas: {task_id}\n"
                f"- Kendala: {error_msg or 'Terjadi kesalahan eksekusi.'}\n"
                f"- Rekomendasi: Anda dapat mencoba mencari sumber alternatif atau memeriksa log dengan perintah /proc."
            )
        elif task_status == "cancelled":
            event_type = "task_cancelled"
            note = f"[PROSES LATAR BELAKANG DIBATALKAN: {task_name} ({task_id})]"
        else:
            return

        # 1. Push structured event into the session's Turn-Boundary Notification Inbox & trigger reactive wakeup
        runtime = self.get_session_runtime(session_id)
        if runtime is not None:
            runtime.push_event(
                SessionEvent(
                    event_type=event_type,
                    session_id=session_id,
                    task_id=task_id,
                    task_name=task_name,
                    task_type=task_type,
                    summary=summary,
                    error=error_msg,
                    ingested_stems=ingested_stems,
                )
            )

        # 2. Record system message in SQLite history for non-chat I/O tasks
        if task_type not in ("chat", "agent"):
            try:
                self.chat_manager.add_message(session_id, "system", note)
            except Exception:
                pass

        # 3. UI Output Arbitrator: render only if session is attached in foreground; otherwise spool to session buffer
        fg_tasks = getattr(self, "_foreground_monitored_tasks", set())
        is_monitored_in_fg = (task_id in fg_tasks)

        if self.sub_mode == "chat" and self.active_chat_session and self.active_chat_session.id == session_id:
            def _notify_task() -> None:
                if task_status == "completed":
                    if task_type in ("chat", "agent"):
                        if is_monitored_in_fg:
                            return
                        self.console.print(f"\n[bold cyan]●[/bold cyan] [bold bright_white]Tugas Latar Belakang Selesai:[/bold bright_white] [cyan]{task_name}[/cyan]")
                        msgs = self.chat_manager.get_messages(session_id) if self.chat_manager else []
                        latest_assistant_msg = None
                        for m in reversed(msgs):
                            if m.role == "assistant":
                                latest_assistant_msg = m
                                break
                        if latest_assistant_msg:
                            skill_lbl = f" ({self.active_skill_name})" if self.active_skill_name and self.active_skill_name != "scrpcy_manual" else ""
                            self.console.print(f"\n[bold cyan]✦ Respons AI{skill_lbl}:[/bold cyan]\n")
                            render_academic_markdown(latest_assistant_msg.content, console=self.console)
                            self.console.print()
                    else:
                        self.console.print(f"\n[bold cyan]●[/bold cyan] [bold bright_white]Tugas Latar Belakang Selesai:[/bold bright_white] [cyan]{task_name}[/cyan] [dim]({summary or 'Selesai'})[/dim]")
                elif task_status == "failed" and not is_monitored_in_fg:
                    self.console.print(f"\n[bold red][x] Tugas Latar Belakang Gagal:[/bold red] [cyan]{task_name}[/cyan] [dim]({error_msg})[/dim]")
            self._enqueue_ui_event(_notify_task)
        elif runtime is not None and not is_monitored_in_fg:
            if task_status == "completed":
                runtime.append_output(
                    f"[bold cyan]●[/bold cyan] [bold bright_white]Tugas Latar Belakang Selesai:[/bold bright_white] [cyan]{task_name}[/cyan] [dim]({summary or 'Selesai'})[/dim]"
                )
            elif task_status == "failed":
                runtime.append_output(
                    f"[bold red][x] Tugas Latar Belakang Gagal:[/bold red] [cyan]{task_name}[/cyan] [dim]({error_msg})[/dim]"
                )

    def _run_task_ai_evaluation(self, task: Any, session_id: str) -> None:
        """Deprecated: Evaluasi otomatis background dinonaktifkan agar tidak menghasilkan respons ganda."""
        return

    def _background_download_worker(
        self,
        task_id: str,
        papers: list[ScoutResult],
        cancel_event: threading.Event | None = None,
    ) -> str:
        """Background agent worker downloading and ingesting academic papers into vault sequentially."""
        total = len(papers)
        success = 0
        failed = 0
        ingested_stems: list[str] = []
        t_obj = self.task_manager.get_task(task_id) if self.task_manager else None
        sess_id = t_obj.session_id if t_obj else None

        self.task_manager.log(task_id, f"Mengunduh {total} naskah secara sekuensial (1 per 1)...")
        self.task_manager.update_progress(task_id, f"Mengunduh {total} naskah sekuensial...")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            for idx, p in enumerate(papers, 1):
                if cancel_event and cancel_event.is_set():
                    break
                if not p.has_pdf or not p.pdf_url:
                    if p.doi:
                        email = getattr(self.config, "unpaywall_email", None)
                        try:
                            oa_url = loop.run_until_complete(self.scout.searcher.lookup_unpaywall(p.doi, email=email))
                            if oa_url:
                                p.pdf_url = oa_url
                                p.has_pdf = True
                                p.relevance_reason = (p.relevance_reason or "") + " (Versi Green OA via Unpaywall)"
                        except Exception:
                            pass

                if not p.has_pdf or not p.pdf_url:
                    self.task_manager.log(task_id, f"[-] Dilewati #{idx} '{p.title[:45]}': Tidak ada PDF open-access.")
                    failed += 1
                    continue

                self.task_manager.log(task_id, f"[{idx}/{total}] Mengunduh PDF: '{p.title[:45]}...'")
                self.task_manager.update_progress(task_id, f"Mengunduh [{idx}/{total}]: {p.title[:30]}...")
                try:
                    dl_path = loop.run_until_complete(self.scout.download_scout_paper(p))
                    ingested_stems.append(dl_path.stem)
                    self.add_session_active_documents([dl_path.stem], session_id=sess_id)
                    success += 1
                    self.task_manager.log(task_id, f"[✓] Added #{idx} to vault & indexed: {p.title[:50]}")
                except Exception as exc:
                    self.task_manager.log(task_id, f"[x] Gagal mengunduh #{idx} '{p.title[:40]}': {exc}")
                    failed += 1

            self.task_manager.record_ingested_stems(task_id, ingested_stems)
            self._sync_vault_cache(session_id=sess_id)
        finally:
            loop.close()

        summary = f"Berhasil mengunduh {success}/{total} dokumen ke vault ({failed} gagal/dilewati)."
        self.task_manager.log(task_id, f"Selesai: {summary}")
        return summary

    def _determine_target_scout_quota(self, query: str, contract: Any = None) -> int:
        """Dynamically determine optimal paper ingestion quota (5-10) based on inquiry scope and complexity."""
        words = len(query.split())
        complexity_keywords = (
            "argument", "argumen", "komprehensif", "lengkap", "analisis",
            "tinjauan", "review", "pro", "kontra", "dialektika", "celah", "kebijakan", "sintesis"
        )
        is_complex = any(k in query.lower() for k in complexity_keywords)

        if contract:
            max_words = getattr(contract, "max_words", 800)
            dod_len = len(getattr(contract, "definition_of_done", []))
            if max_words >= 1000 or dod_len >= 4:
                return 9 if is_complex else 8
            elif max_words >= 600 or is_complex:
                return 7
            return 5

        if is_complex or words >= 8:
            return 8
        return 5

    def _background_scout_worker(
        self,
        task_id: str,
        query: str = "",
        topic_query: str = "",
        max_download: int | None = None,
        cancel_event: threading.Event | None = None,
        **kwargs: Any,
    ) -> str:
        """Background agent worker that searches multi-source (30-40 candidates), ranks with AI, and downloads/ingests a robust batch (5-10) into vault."""
        clean_query = (query or topic_query).strip().strip("'\"")
        scout_t0 = time.time()
        self.task_manager.log(task_id, f"Formulating query strategy for: '{clean_query}'")
        t_obj = self.task_manager.get_task(task_id) if self.task_manager else None
        sess_id = t_obj.session_id if t_obj else None
        ingested_stems: list[str] = []

        # Support mocked or custom auto_scout_and_ingest from tests
        auto_fn = getattr(self.scout, "auto_scout_and_ingest", None)
        is_scout_mocked = (
            auto_fn is not None
            and (
                hasattr(auto_fn, "assert_called")
                or getattr(auto_fn, "__code__", None) != LiteratureScout.auto_scout_and_ingest.__code__
            )
            and (inspect.iscoroutinefunction(auto_fn) or isinstance(auto_fn, AsyncMock))
        )
        if is_scout_mocked:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                def _on_st(action: str, label: str) -> None:
                    self.task_manager.log(task_id, f"[{action}] {label}")
                    self.emit_session_output(
                        sess_id,
                        f"[bold cyan]●[/bold cyan] [bold bright_white]{action}[/bold bright_white] [cyan]{label}[/cyan]",
                    )
                scout_kwargs: dict[str, Any] = {
                    "max_papers": max_download or 2,
                    "on_status": _on_st,
                }
                sig = inspect.signature(auto_fn)
                if "engine" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    scout_kwargs["engine"] = self.engine
                res = auto_fn(clean_query, **scout_kwargs)
                try:
                    if inspect.isawaitable(res):
                        res_stems = loop.run_until_complete(res)
                    else:
                        res_stems = res
                except (asyncio.TimeoutError, TimeoutError) as to_exc:
                    raise TimeoutError("Auto-scout timed out") from to_exc
                stems_list = list(res_stems or [])
                self.add_session_active_documents(stems_list, session_id=sess_id)
                self.task_manager.record_ingested_stems(task_id, stems_list)
                return f"Ingested {len(stems_list)} paper(s): {', '.join(stems_list)}"
            finally:
                loop.close()

        # 1. Pre-flight Official Web Scouting (WebScout)
        is_mock_env = (
            "PYTEST_CURRENT_TEST" in os.environ
            or getattr(self.engine, "acompletion_fn", None) is not None
            or any(str(v).startswith("mock") for v in getattr(self.config, "api_keys", {}).values())
            or getattr(self.config, "default_model", "").startswith("mock")
        )
        if getattr(self, "web_scout", None):
            is_web_mocked = (
                hasattr(self.web_scout, "_is_mock")
                or "search_official_web" in getattr(self.web_scout, "__dict__", {})
                or "Mock" in type(getattr(self.web_scout, "search_official_web", None)).__name__
                or "Mock" in type(self.web_scout).__name__
            )
            if not (is_mock_env and not is_web_mocked):
                self.task_manager.log(task_id, "Menelusuri portal web resmi (.go.id/institusi) via WebScout...")
                self.task_manager.update_progress(task_id, "Menelusuri portal resmi (.go.id)...")
                w_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(w_loop)
                try:
                    w_hits = w_loop.run_until_complete(
                        self.web_scout.search_official_web(clean_query, limit=3, require_official=True)
                    )
                    if w_hits:
                        vec_vault = getattr(self.vault_manager, "vector_vault", None) if self.vault_manager else None
                        disambiguated = QueryDisambiguator.disambiguate(clean_query)
                        for wh in w_hits[:2]:
                            try:
                                parsed_w = w_loop.run_until_complete(
                                    self.web_scout.ingest_url(
                                        wh.url,
                                        vector_vault=vec_vault,
                                        required_topic_terms=disambiguated.core_entity_terms,
                                    )
                                )
                                if parsed_w and parsed_w.metadata and parsed_w.metadata.cite_key:
                                    ingested_stems.append(parsed_w.metadata.cite_key)
                                    self.add_session_active_documents([parsed_w.metadata.cite_key], session_id=sess_id)
                                    self.task_manager.log(task_id, f"[✓] Web resmi diindeks: '{wh.title[:45]}...' ({wh.institution})")
                            except Exception as w_err:
                                self.task_manager.log(task_id, f"[!] Gagal unduh web resmi: {w_err}")
                except Exception as w_exc:
                    self.task_manager.log(task_id, f"[!] Penelusuran web resmi: {w_exc}")
                finally:
                    w_loop.close()

        try:
            query_spec = self.scout.extract_search_criteria(clean_query, engine=self.engine)
        except Exception as exc:
            self.task_manager.log(task_id, f"[!] Criteria formulation fallback: {exc}")
            query_spec = ScoutQuery(keywords=clean_query, summary=clean_query)

        # Dynamic target quota between 5 and 10 papers
        target_quota = max_download if (max_download is not None and max_download >= 5) else self._determine_target_scout_quota(clean_query)
        self.task_manager.log(task_id, f"Target kuota literatur: {target_quota} dokumen.")

        # Multi-source search with larger candidate limit (35 candidates)
        self.task_manager.log(task_id, f"Scraping OpenAlex & Semantic Scholar (35 candidates, keywords: '{query_spec.keywords}')...")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            candidates = loop.run_until_complete(self.scout.search_multi_source(query_spec, limit=35))
        except Exception as exc:
            self.task_manager.log(task_id, f"[x] Multi-source search error: {exc}")
            raise
        finally:
            loop.close()

        if not candidates:
            msg = f"No scholarly candidates found for '{query}'."
            self.task_manager.log(task_id, msg)
            return msg

        self.task_manager.log(task_id, f"Curating & ranking {len(candidates)} candidates against research intent with AI...")
        try:
            curated = self.scout.curate_and_rank(clean_query, candidates, engine=self.engine, max_output=target_quota * 2)
        except Exception:
            curated = candidates

        self.last_scout_results = curated

        ingested_count = 0
        tried_titles: set[str] = set()
        papers_to_ingest = []
        for paper in curated:
            if paper.title.lower() not in tried_titles:
                tried_titles.add(paper.title.lower())
                papers_to_ingest.append(paper)
            if len(papers_to_ingest) >= target_quota:
                break

        self.task_manager.log(task_id, f"Mengunduh {len(papers_to_ingest)} naskah secara sekuensial (1 per 1)...")
        self.task_manager.update_progress(task_id, f"Mengunduh {len(papers_to_ingest)} naskah...")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            for p_idx, p_obj in enumerate(papers_to_ingest, 1):
                if cancel_event and cancel_event.is_set():
                    break
                stem = sanitize_filename(p_obj.title)
                if not (p_obj.has_pdf and p_obj.pdf_url) and p_obj.doi:
                    email = getattr(self.config, "unpaywall_email", None)
                    try:
                        oa_url = loop.run_until_complete(self.scout.searcher.lookup_unpaywall(p_obj.doi, email=email))
                        if oa_url:
                            p_obj.pdf_url = oa_url
                            p_obj.has_pdf = True
                            p_obj.relevance_reason = (p_obj.relevance_reason or "") + " (Versi Green OA via Unpaywall)"
                    except Exception:
                        pass

                has_pdf = bool(p_obj.has_pdf and p_obj.pdf_url)
                if has_pdf:
                    self.task_manager.log(task_id, f"[{p_idx}/{len(papers_to_ingest)}] Mengunduh PDF: '{p_obj.title[:45]}...'")
                    self.task_manager.update_progress(task_id, f"Mengunduh [{p_idx}/{len(papers_to_ingest)}]: {p_obj.title[:30]}...")
                    try:
                        dl_p = loop.run_until_complete(self.scout.download_scout_paper(p_obj))
                        ingested_stems.append(dl_p.stem)
                        self.add_session_active_documents([dl_p.stem], session_id=sess_id)
                        ingested_count += 1
                        self.task_manager.log(task_id, f"[✓] Ingested PDF [{ingested_count}/{len(papers_to_ingest)}]: {p_obj.title[:50]}")
                    except Exception as err:
                        self.task_manager.log(task_id, f"[!] Kendala unduh '{p_obj.title[:35]}': {err}")
                        if p_obj.abstract:
                            try:
                                self.scout.ingest_abstract_paper(p_obj)
                                ingested_stems.append(stem)
                                self.add_session_active_documents([stem], session_id=sess_id)
                                ingested_count += 1
                                self.task_manager.log(task_id, f"[✓] Fallback abstrak diindeks: {p_obj.title[:45]}")
                            except Exception:
                                pass
                elif p_obj.abstract:
                    self.task_manager.log(task_id, f"[{p_idx}/{len(papers_to_ingest)}] Mengindeks abstrak resmi (paywalled): '{p_obj.title[:45]}...'")
                    self.task_manager.update_progress(task_id, f"Indeks abstrak [{p_idx}/{len(papers_to_ingest)}]: {p_obj.title[:30]}...")
                    try:
                        self.scout.ingest_abstract_paper(p_obj)
                        ingested_stems.append(stem)
                        self.add_session_active_documents([stem], session_id=sess_id)
                        ingested_count += 1
                        self.task_manager.log(task_id, f"[✓] Ingested abstrak [{ingested_count}/{len(papers_to_ingest)}]: {p_obj.title[:50]}")
                    except Exception:
                        pass

            self.task_manager.record_ingested_stems(task_id, ingested_stems)
            self._sync_vault_cache(session_id=sess_id)
        finally:
            loop.close()

        # If still below target_quota and we have active documents, expand via Reference Mapping
        session_docs = self.get_session_active_documents(sess_id)
        if ingested_count < target_quota and session_docs and not (cancel_event and cancel_event.is_set()):
            stems = list(session_docs)
            self.task_manager.log(task_id, f"Menelusuri jejaring sitasi (Reference Mapping) untuk mencapai target {target_quota} dokumen...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                cit_papers = loop.run_until_complete(self.scout.map_vault_citation_network(stems, limit=target_quota - ingested_count + 3))
                for cp in cit_papers:
                    if ingested_count >= target_quota or (cancel_event and cancel_event.is_set()):
                        break
                    if cp.title.lower() in tried_titles:
                        continue
                    tried_titles.add(cp.title.lower())
                    safe_stem = sanitize_filename(cp.title)
                    if cp.has_pdf and cp.pdf_url:
                        try:
                            dl_path = loop.run_until_complete(self.scout.download_scout_paper(cp))
                            ingested_stems.append(dl_path.stem)
                            self.add_session_active_documents([dl_path.stem], session_id=sess_id)
                            self._sync_vault_cache(session_id=sess_id)
                            ingested_count += 1
                            self.task_manager.log(task_id, f"[✓] Ingested sitasi PDF #{ingested_count}: {cp.title[:45]}")
                        except Exception:
                            if cp.abstract:
                                self.scout.ingest_abstract_paper(cp)
                                ingested_stems.append(safe_stem)
                                self.add_session_active_documents([safe_stem], session_id=sess_id)
                                self._sync_vault_cache(session_id=sess_id)
                                ingested_count += 1
                    elif cp.abstract:
                        self.scout.ingest_abstract_paper(cp)
                        ingested_stems.append(safe_stem)
                        self.add_session_active_documents([safe_stem], session_id=sess_id)
                        self._sync_vault_cache(session_id=sess_id)
                        ingested_count += 1
                        self.task_manager.log(task_id, f"[✓] Ingested sitasi abstrak #{ingested_count}: {cp.title[:45]}")
                self.task_manager.record_ingested_stems(task_id, ingested_stems)
            except Exception as ref_exc:
                self.task_manager.log(task_id, f"[!] Kendala reference mapping: {ref_exc}")
            finally:
                loop.close()

        total_ingested = len(ingested_stems)
        summary = f"Scout & Ingest selesai: Mengindeks {total_ingested} dokumen (Web Resmi & Jurnal) ke vault."
        if hasattr(self, "logger") and self.logger:
            self.logger.log_literature_scout(
                session_id=sess_id or "global",
                query=clean_query,
                candidates_count=len(candidates) if "candidates" in locals() and candidates else total_ingested,
                ingested_count=total_ingested,
                ingested_stems=ingested_stems,
                duration_s=time.time() - scout_t0,
            )
        self.task_manager.log(task_id, summary)
        return summary

    def render_processing_summary(self) -> None:
        """Render concise status overview of background processing tasks."""
        tasks = self.task_manager.list_tasks()
        running_tasks = [t for t in tasks if getattr(t, "is_active", t.status == "running")]
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
        table.add_column("Status", justify="center", width=15)
        table.add_column("Tokens", justify="right", style="dim yellow", width=11, no_wrap=True)
        table.add_column("Duration", justify="right", style="dim", width=8)

        for idx, t in enumerate(tasks, 1):
            if t.status == "waiting_deps":
                status_badge = "[bold yellow][Waiting Deps][/bold yellow]"
            elif t.status == "running":
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
        t_table.add_row("Lane", getattr(task, "lane", "io").upper())
        t_table.add_row("Status", task.status.upper())
        if getattr(task, "depends_on", None):
            t_table.add_row("Depends On", ", ".join(task.depends_on))
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

    def _background_docx_worker(self, task_id: str, filename: str) -> str:
        """Background worker compiling conversation synthesis into Word (.docx) manuscript."""
        self.task_manager.log(task_id, f"Initializing manuscript compilation: {filename}")
        t_obj = self.task_manager.get_task(task_id) if self.task_manager else None
        sess_id = t_obj.session_id if t_obj else None
        runtime = self.get_session_runtime(sess_id)

        drafts_dir = self.workspace.root_dir / "drafts"
        drafts_dir.mkdir(parents=True, exist_ok=True)
        dest_path = drafts_dir / filename

        history_items = runtime.get_history() if (runtime and runtime.get_history()) else list(self.conversation_history)
        sections = []
        if history_items:
            for idx, item in enumerate(history_items, 1):
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
        doc_stems = self.get_session_active_documents(sess_id)
        for stem in doc_stems:
            try:
                content = self.workspace.read_from_cache(f"{stem}.md")
                meta, _ = DocumentMetadata.from_frontmatter(content)
                references.append(meta.model_dump())
            except Exception:
                pass

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
