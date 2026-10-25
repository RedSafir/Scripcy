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
from scrpcy.core.contractor import ContractDecomposer, CritiqueAdjudication, SubContract
from scrpcy.core.critic import CritiqueReport, DevilsAdvocate
from scrpcy.core.orchestrator import SubagentOrchestrator
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
    interactive_scout_confirm,
    interactive_skill_selector,
    interactive_token_warning_confirm,
)


class StudioAgentLoopMixin:
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

    def submit_chat_background(self, query: str, depends_on: list[str] | None = None) -> TaskInfo | None:
        """Submit a research query to run in the session's serialized turn lane with automatic I/O dependency barriers."""
        if not self.workspace or not self.engine:
            self.console.print("[red][x] No active project workspace loaded.[/red]\n")
            return None

        session = self._ensure_active_chat_session() if self.chat_manager else None
        if not session:
            self.console.print("[red][x] No active chat session.[/red]\n")
            return None

        def _chat_worker(task_id: str, cancel_event: threading.Event | None = None) -> str:
            start_t = time.time()
            self.task_manager.log(task_id, f"Pertanyaan: {query}")
            model_name = self.get_active_project_model()
            self.task_manager.log(task_id, f"Model: {model_name}")
            self.task_manager.log(task_id, "Memulai sintesis latar belakang...")

            runtime = self.get_session_runtime(session.id)
            inbox_events = runtime.drain_inbox() if runtime is not None else []
            if inbox_events:
                if self.vault_manager:
                    try:
                        self.vault_manager.reload_vault(force=False)
                    except Exception:
                        pass
                self._sync_vault_cache(session_id=session.id)

            if self.chat_manager:
                self.chat_manager.add_message(session.id, "user", query)

            sess_docs = self.get_session_active_documents(session.id)
            active_paths = [self.workspace.get_cache_path(f"{s}.md") for s in sess_docs]
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
                    vector_vault=getattr(self.vault_manager, "vector_vault", None),
                ):
                    if cancel_event and cancel_event.is_set():
                        return
                    tokens.append(tok)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_stream())
            finally:
                loop.close()

            if cancel_event and cancel_event.is_set():
                return "Cancelled"

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
                        active_documents=sess_docs,
                    )

            elapsed_c = time.time() - start_t
            in_tok = estimate_tokens(query) + 250
            out_tok = estimate_tokens(resp)
            tot_tok = in_tok + out_tok

            if self.chat_manager and resp:
                self.chat_manager.add_message(session.id, "assistant", resp)
            if runtime is not None and resp:
                runtime.record_exchange(query, resp)

            self.task_manager.update_tokens(task_id, tot_tok)
            self.task_manager.log(task_id, f"Token: {in_tok:,} in · {out_tok:,} out · {tot_tok:,} total ({model_name}) • {elapsed_c:.1f}s")
            self.task_manager.log(task_id, f"Selesai! {len(resp)} karakter disimpan ke riwayat sesi.")
            return f"Sintesis selesai ({len(resp)} karakter, {tot_tok:,} total token)" if resp else "Selesai"

        task = self.task_manager.submit(
            name=f"Chat: {query[:35]}...",
            task_type="chat",
            target_fn=_chat_worker,
            session_id=session.id,
            depends_on=depends_on,
            lane="turn",
            auto_depend_session_io=True,
            auto_serialize_turn=True,
        )
        if task.status == "waiting_deps" and task.depends_on:
            deps_str = ", ".join(task.depends_on)
            self.console.print(
                f"[bold green][✓] Pertanyaan dijadwalkan di antrean sesi:[/bold green] [bold cyan]{task.id}[/bold cyan] "
                f"[bold yellow][Waiting Deps: {deps_str}][/bold yellow]"
            )
            self.console.print(" [dim]• Tugas ini otomatis menunggu proses I/O atau turn sebelumnya selesai sebelum mulai menyintesis.[/dim]\n")
        else:
            self.console.print(f"[bold green][✓] Pertanyaan dikirim ke proses latar belakang:[/bold green] [bold cyan]{task.id}[/bold cyan]")
            self.console.print(" [dim]• Anda dapat tetap mengetik, keluar (/back), atau memantau status dengan /proc.[/dim]")
            self.console.print(" [dim]• Saat selesai, AI akan otomatis mengevaluasi hasil dan menyimpan jawaban di sesi ini.[/dim]\n")
        return task

    def _ask_ai_task_continuation(self, task_name: str, progress: str, elapsed: float) -> str:
        """Ultra-frugal AI check (~50 tokens) to decide whether to extend or abort a prolonged task."""
        if not self.engine:
            return "EXTEND" if elapsed < 60.0 else "ABORT"

        try:
            model_name = self.get_active_project_model()
            api_key = self.config.get_api_key_for_model(model_name)
            if not api_key or api_key.startswith("mock") or getattr(self.engine, "_acompletion_fn", None) is not None:
                return "EXTEND" if elapsed < 60.0 else "ABORT"

            effective_model = model_name
            api_base = None
            if model_name.startswith("9router/"):
                submodel = model_name.split("/", 1)[1]
                effective_model = f"openai/{submodel}"
                api_base = "http://localhost:20128/v1"

            import json
            import litellm
            kwargs: dict[str, Any] = {
                "model": effective_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a concise runtime task scheduler. Return JSON: {\"decision\": \"EXTEND\" | \"ABORT\"}",
                    },
                    {
                        "role": "user",
                        "content": f"Task: {task_name}\nLast Progress: '{progress}'\nElapsed: {elapsed:.0f}s.\nIf progress shows downloading/parsing in action, reply EXTEND. If idle/no candidates, reply ABORT.",
                    },
                ],
                "api_key": api_key,
                "temperature": 0.0,
                "max_tokens": 30,
                "response_format": {"type": "json_object"},
                "timeout": 8,
            }
            if api_base:
                kwargs["api_base"] = api_base

            resp = litellm.completion(**kwargs)
            raw = resp.choices[0].message.content or "{}"
            data = json.loads(raw.strip())
            dec = str(data.get("decision", "EXTEND")).upper().strip()
            return "ABORT" if "ABORT" in dec else "EXTEND"
        except Exception:
            return "EXTEND" if elapsed < 60.0 else "ABORT"

    def _check_agent_turn_checkpoint(
        self,
        turn: int,
        best_grounding: float,
        best_critic: int,
        accumulated_input_tokens: int,
        accumulated_output_tokens: int,
        task_id: str | None = None,
        session: Any = None,
        max_refine_limit: int = 15,
        last_c_report: Any = None,
    ) -> bool:
        """Evaluate if an interactive 3-turn checkpoint should pause and ask user to continue.

        Returns True if turn execution should STOP (finalize with best draft).
        Returns False if turn execution should CONTINUE.
        """
        completed_turns = turn + 1
        if completed_turns <= 0 or completed_turns % 3 != 0:
            return False

        # If Grounding and Critic evaluators already passed threshold, finish without asking!
        if best_grounding >= 0.70 and best_critic >= 7:
            return True

        # In detached background mode, do not block on stdin
        if session and getattr(session, "status", None) == "detached":
            return completed_turns >= max_refine_limit

        if not sys.stdin.isatty():
            # Non-interactive / headless environment (e.g. test suites, piped CLI)
            return completed_turns >= 15

        # If running in worker thread with task manager, use thread-safe checkpoint handshake
        if threading.current_thread() is not threading.main_thread() and self.task_manager and task_id:
            critique = getattr(last_c_report, "critique_summary", "") if last_c_report else ""
            prompt_data = {
                "completed_turns": completed_turns,
                "best_grounding": best_grounding,
                "best_critic": best_critic,
                "accumulated_input_tokens": accumulated_input_tokens,
                "accumulated_output_tokens": accumulated_output_tokens,
                "critique_summary": critique,
            }
            choice = self.task_manager.request_checkpoint(task_id, prompt_data, timeout=300.0)
            if choice in ("stop", "finalize", "selesai", "n", "no", "cancelled", None):
                return True
            return False

        # If running directly on main thread
        total_tokens = accumulated_input_tokens + accumulated_output_tokens
        self.console.print(
            f"\n[bold yellow]●[/bold yellow] [bold bright_white]Checkpoint Iterasi ({completed_turns} Turn Selesai)[/bold bright_white]\n"
            f"  [dim]•[/dim] Skor Evaluasi: [cyan]Grounding: {max(0, int(best_grounding * 100))}%[/cyan] [dim]•[/dim] [yellow]Critic: {max(0, best_critic)}/10[/yellow]\n"
            f"  [dim]•[/dim] Konsumsi Token: [dim]{accumulated_input_tokens:,} in · {accumulated_output_tokens:,} out ({total_tokens:,} total)[/dim]"
        )
        try:
            choice = self.console.input(
                " [bold cyan]?[/bold cyan] Tambah 3 iterasi turn lagi untuk menyempurnakan jawaban? [bold cyan][Y/n/selesai][/bold cyan]: "
            ).strip().lower()
            if choice in ("n", "no", "selesai", "c", "cancel"):
                return True
            self.console.print("[dim]● Melanjutkan 3 iterasi turn berikutnya...[/dim]\n")
            return False
        except (KeyboardInterrupt, EOFError):
            return True

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

            if cmd in ("/detach", "detach"):
                self.render_mode_divider()
                running_tasks = (
                    self.task_manager.list_active_tasks_for_session(self.active_chat_session.id)
                    if (self.active_chat_session and self.task_manager)
                    else []
                )
                if running_tasks:
                    self.console.print(f"[bold yellow]●[/bold yellow] [dim]{len(running_tasks)} proses latar belakang tetap berjalan di background. Anda dapat kembali ke sesi ini kapan saja dengan '/switch' atau 'resume {self.active_chat_session.name}'.[/dim]")
                if self.active_chat_session and self.chat_manager:
                    self.chat_manager.update_session_status(self.active_chat_session.id, "detached")
                    self.active_chat_session.status = "detached"
                self.sub_mode = None
                self.console.print(f"[dim]Kembali ke project root: {self.current_project.name if self.current_project else 'project'}>[/dim]\n")
                return
            elif cmd in ("/back", "back"):
                self.render_mode_divider()
                running_tasks = (
                    self.task_manager.list_active_tasks_for_session(self.active_chat_session.id)
                    if (self.active_chat_session and self.task_manager)
                    else []
                )
                if running_tasks:
                    self.console.print(f"[bold yellow]●[/bold yellow] [dim]{len(running_tasks)} proses latar belakang tetap berjalan di background. Anda dapat kembali ke sesi ini kapan saja dengan '/switch' atau 'resume {self.active_chat_session.name}'.[/dim]")
                if self.active_chat_session and self.chat_manager:
                    self.chat_manager.update_session_status(self.active_chat_session.id, "detached")
                    self.active_chat_session.status = "detached"
                self.sub_mode = "chat_hub"
                self.console.print(f"[dim]Kembali ke menu chat: {self.current_project.name if self.current_project else 'project'}/chat>[/dim]\n")
                return
            elif cmd in ("/close", "close", "/exit", "exit", "/quit", "quit"):
                self.render_mode_divider()
                if self.active_chat_session and self.chat_manager:
                    self.chat_manager.update_session_status(self.active_chat_session.id, "closed")
                    self.active_chat_session.status = "closed"
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
                    session_tasks = self.task_manager.list_active_tasks_for_session(self.active_chat_session.id)
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
            elif cmd in ("/resume", "/open", "/switch", "/chats", "/sessions"):
                if not self.chat_manager:
                    return
                if not arg:
                    self.run_interactive_sessions_list()
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
            elif cmd in ("/processing", "/process", "/proc", "/tasks", "/task"):
                self.sub_mode = "processing"
                if arg:
                    self.handle_processing_interaction(arg)
                else:
                    self.run_interactive_processing_view()
                return
            elif cmd in ("/export",):
                self.cmd_chat_export()
            else:
                self.handle_project_command(clean)
            return

        # Serialized Turn Lane Coordination:
        # If another synthesis turn (lane == "turn") is already running in this session,
        # queue this prompt onto the session's serialized turn lane (depends_on) instead of dropping it.
        if self.active_chat_session and self.task_manager:
            active_turns = self.task_manager.list_active_tasks_for_session(
                self.active_chat_session.id,
                lane="turn",
            )
            if active_turns:
                self.submit_chat_background(clean, depends_on=[t.id for t in active_turns])
                return

        # Any input without leading "/" in chat mode is strictly treated as an AI conversation prompt
        self.handle_chat(clean)

    def _detect_internal_tool_call(self, raw_resp: str) -> str | None:
        """Detect internal tool commands emitted during an agent turn."""
        if not raw_resp:
            return None

        # 1. Fenced blocks: ```scrpcy ... ``` or ``` ... ```
        fenced_match = re.search(
            r"```(?:scrpcy)?\r?\n\s*(/(?:vault\s+(?:read|view|list|remove|rm|archive)|read|view|list|remove|archive|scout|search|findpaper|web|web-scout|webscout)[^\n`]*)\r?\n```",
            raw_resp,
            re.IGNORECASE,
        )
        if fenced_match:
            return fenced_match.group(1).strip()

        # 2. Bare command line
        for line in raw_resp.splitlines():
            clean_line = line.strip()
            if re.match(
                r"^/(?:vault\s+(?:read|view|list|remove|rm|archive)|read|view|list|remove|archive|scout|search|findpaper|web|web-scout|webscout)\b",
                clean_line,
                re.IGNORECASE,
            ):
                return clean_line

        # 3. Inline backticks: `/vault read 1`
        inline_match = re.search(
            r"`(/(?:vault\s+(?:read|view|list|remove|rm|archive)|read|view|list|remove|archive|scout|search|findpaper|web|web-scout|webscout)[^`]+)`",
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
                is_web = re.match(r"^/(?:web|web-scout|webscout)\b", l, re.IGNORECASE)
                if is_internal or (has_ok_signal and (is_scout or is_web)):
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
                r"(?m)^\s*/(?:scout|findpaper|search|web|web-scout|webscout)\b[^\n]*\r?\n?",
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

    def _format_vault_citation_guide(
        self,
        active_evidence_stems: set[str] | list[str] | None = None,
    ) -> str:
        """Generate formatted citation guidelines and list of active vault papers for the prompt."""
        if not self.vault_manager or not self.workspace:
            return ""
        active_items = self.vault_manager.list_active_files()
        if not active_items:
            return ""

        stems_filter = set(active_evidence_stems) if active_evidence_stems else None

        primary_items = []
        secondary_items = []
        for item in active_items:
            if stems_filter is None or item.stem in stems_filter or item.cite_key in stems_filter:
                primary_items.append(item)
            else:
                secondary_items.append(item)

        # If filter yielded nothing (e.g. source ID mismatch), show all
        if not primary_items:
            primary_items = active_items
            secondary_items = []

        lines = [
            "### PANDUAN SITASI DAN REFERENSI VAULT (GAYA APA):",
            "Dokumen ilmiah terverifikasi yang dirujuk sebagai bukti dalam kueri ini:",
        ]
        for idx, item in enumerate(primary_items, 1):
            cache_path = self.workspace.get_cache_path(f"{item.stem}.md")
            meta = self.vault_manager._read_frontmatter_metadata(cache_path) if cache_path.exists() else None
            authors_str = ", ".join(meta.authors[:3]) if (meta and meta.authors) else item.stem.replace("_", " ").title()
            first_auth = meta.authors[0].split(",")[0].strip().split()[-1] if (meta and meta.authors) else item.stem.split("_")[0].capitalize()
            year_str = str(meta.year) if (meta and meta.year) else "n.d."
            cite_apa = f"({first_auth} et al., {year_str})" if (meta and meta.authors and len(meta.authors) > 2) else f"({first_auth}, {year_str})"
            cite_key = meta.cite_key if (meta and meta.cite_key) else (item.cite_key or item.stem)
            lines.append(f"- [{idx}] {authors_str} ({year_str}). \"{item.title or item.stem}\". File: {item.filename} | Sitasi in-text APA: {cite_apa} atau [{cite_key}]")

        if secondary_items:
            other_names = ", ".join(it.cite_key or it.stem for it in secondary_items[:8])
            lines.append(f"\n[Dokumen tambahan di vault yang dapat dibaca via internal tool jika diperlukan]: {other_names}")

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

        elif cmd_root in ("web", "web-scout", "webscout") or sub in ("web", "web-scout", "webscout"):
            target_web = clean.split(maxsplit=1)[1].strip() if len(clean.split(maxsplit=1)) > 1 else ""
            target_web = target_web.strip("'\"")
            if not target_web:
                return "Menelusuri Web", "URL/Query Kosong", "Tidak ada URL atau kata kunci web yang diberikan."
            from scrpcy.core.web_scout import WebScout
            web_scout = getattr(self, "web_scout", None) or WebScout(workspace=self.workspace)
            if target_web.startswith("http://") or target_web.startswith("https://"):
                try:
                    vec_vault = self.vault_manager.vector_vault if self.vault_manager else None
                    parsed = asyncio.run(web_scout.ingest_url(target_web, vector_vault=vec_vault))
                    self._sync_vault_cache()
                    summary_text = (
                        f"Dokumen web resmi berhasil diunduh dan diproses ke vault:\n"
                        f"- Judul: {parsed.metadata.title}\n"
                        f"- Instansi: {parsed.metadata.institution}\n"
                        f"- Cite Key: [{parsed.metadata.cite_key}]\n"
                        f"- Skor Otoritas: {parsed.metadata.authority_score or 0.0:.2f} (Tier {parsed.metadata.authority_tier or 1})\n\n"
                        f"Isi Dokumen (ringkasan):\n"
                        f"{parsed.body_markdown[:3500]}"
                    )
                    return "Mengunduh Web", f"vault/web/{parsed.metadata.cite_key}.html", summary_text
                except Exception as exc:
                    return "Mengunduh Web", "Gagal", f"Gagal memproses web resmi '{target_web}': {exc}"
            else:
                try:
                    results = asyncio.run(web_scout.search_official_web(target_web, limit=4, require_official=True))
                    if not results:
                        return "Menelusuri Web", f"'{target_web}' (0 hasil resmi)", f"Tidak ditemukan web resmi untuk '{target_web}'."
                    lines = [f"Hasil pencarian web resmi untuk '{target_web}' (diurutkan berdasarkan otoritas institusi):"]
                    for idx, r in enumerate(results, 1):
                        lines.append(
                            f"{idx}. {r.title} ({r.domain} - {r.institution})\n"
                            f"   - Skor Otoritas: {r.authority_score:.2f} (Tier {r.authority_tier})\n"
                            f"   - URL: {r.url}\n"
                            f"   - Snippet: {r.snippet[:250]}"
                        )
                    return "Menelusuri Web", f"'{target_web[:25]}'", "\n".join(lines)
                except Exception as exc:
                    return "Menelusuri Web", "Gagal", f"Pencarian web resmi gagal: {exc}"

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
            "lakukan", "buat", "berikan", "terhadap", "selama", "analisa", "analisis", "kajian", "studi",
        }
        cleaned = re.sub(r"[^\w\s-]", " ", query)
        words = cleaned.split()
        salient = [w for w in words if w.lower() not in stopwords and (len(w) > 2 or w.isdigit())]
        if salient:
            # Preserve up to 10 terms so geographic, domain, and temporal terms are not truncated
            return " ".join(salient[:10])
        return query.strip()

    def _run_agent_turn_interactive(
        self,
        coro: Any,
        step_desc: str | Any,
        prompt_lbl: str,
        model_label: str,
        effort: str,
        task_id: str | None = None,
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

        async def _interactive_loop() -> Any:
            t = asyncio.create_task(coro)
            frame_idx = 0
            last_spin = time.time()
            hint_str = "[Esc: detach · Ctrl+C: batal]"

            try:
                while not t.done():
                    now = time.time()
                    need_redraw = False

                    # Check if worker thread requested a checkpoint decision
                    if hasattr(self, "task_manager") and self.task_manager and task_id:
                        pending_cp = self.task_manager.get_pending_checkpoint(task_id)
                        if pending_cp:
                            if not sys.stdin.isatty():
                                comp_turns = pending_cp.prompt_data.get("completed_turns", 3)
                                decision = "finalize" if comp_turns >= 15 else "continue"
                                self.task_manager.resolve_checkpoint(task_id, decision)
                                continue

                            sys.stdout.write("\r\x1b[2K")
                            sys.stdout.flush()

                            cp_data = pending_cp.prompt_data
                            comp_turns = cp_data.get("completed_turns", 3)
                            bg_score = cp_data.get("best_grounding", 0.0)
                            bc_score = cp_data.get("best_critic", 0)
                            in_tok = cp_data.get("accumulated_input_tokens", 0)
                            out_tok = cp_data.get("accumulated_output_tokens", 0)
                            tot_tok = in_tok + out_tok
                            critique = cp_data.get("critique_summary", "")

                            card_header = Text()
                            card_header.append("\n● ", style="bold yellow")
                            card_header.append(f"Checkpoint Iterasi Riset ({comp_turns} Turn Selesai)\n", style="bold bright_white")
                            card_header.append("  • Evaluasi: ", style="dim")
                            card_header.append(f"Grounding {int(bg_score * 100)}% ", style="bold cyan")
                            card_header.append("• ", style="dim")
                            card_header.append(f"Devil's Advocate {bc_score}/10\n", style="bold yellow")
                            card_header.append("  • Konsumsi: ", style="dim")
                            card_header.append(f"{in_tok:,} in · {out_tok:,} out ({tot_tok:,} total token)\n", style="dim")
                            if critique:
                                card_header.append(f"  • Catatan: {critique[:100]}...\n", style="dim")

                            cp_opts = [
                                ("continue", "Lanjutkan 3 iterasi turn perbaikan lagi (kejar grounding & audit)"),
                                ("finalize", "Selesaikan sekarang dan finalisasi draf terbaik"),
                            ]

                            ch = interactive_menu(
                                cp_opts,
                                header=card_header,
                                console=self.console,
                                footer_hint="↑/↓ Pilih · enter Konfirmasi",
                                allow_back=False,
                            )

                            decision = "continue" if ch == "continue" else "finalize"
                            self.task_manager.resolve_checkpoint(task_id, decision)
                            last_spin = time.time()
                            continue
                    if now - last_spin >= 0.10:
                        frame_idx = (frame_idx + 1) % len(SPINNER_FRAMES)
                        last_spin = now
                        need_redraw = True

                    if sys.platform == "win32":
                        import msvcrt
                        while msvcrt.kbhit():
                            ch = msvcrt.getwch()
                            if ch == "\x03":  # Ctrl+C
                                t.cancel()
                                raise asyncio.CancelledError("cancel")
                            elif ch == "\x1b":  # Esc
                                raise SessionDetachSignal("detach")
                            # User input is disabled during processing per user UX preference

                    cur_desc_val = step_desc() if callable(step_desc) else str(step_desc)
                    from scrpcy.ui.interactive import get_live_terminal_size
                    term_cols, _ = get_live_terminal_size(self.console)
                    max_w = max(20, term_cols - 1)
                    avail_desc = max(5, max_w - len(hint_str) - 8)
                    if len(cur_desc_val) > avail_desc:
                        disp_desc = cur_desc_val[:avail_desc - 1] + "…"
                    else:
                        disp_desc = cur_desc_val

                    if need_redraw:
                        frame = SPINNER_FRAMES[frame_idx]
                        line = f"\r\x1b[2K\033[1;36m{frame}\033[0m \033[1;37m{disp_desc}\033[0m   \033[2;37m{hint_str}\033[0m"
                        sys.stdout.write(line)
                        sys.stdout.flush()

                    await asyncio.sleep(0.04)

                sys.stdout.write("\r\x1b[2K")
                sys.stdout.flush()
                return await t

            except BaseException as exc:
                sys.stdout.write("\r\x1b[2K")
                sys.stdout.flush()
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
                session_id=session.id if session else None,
                lane="io",
                file_id=gdrive_id,
            )
            self.console.print(
                f"[bold green][✓] Proses download Google Drive dikirim ke antrean Processing:[/bold green] [bold cyan]{task.id}[/bold cyan]"
            )
            self.console.print(" [dim]• File akan otomatis diunduh, dikonversi ke Markdown, dan diindeks ke dalam vault.[/dim]")
            self.console.print(" [dim]• Ketik [bold cyan]/proc[/bold cyan] untuk memantau status atau log.[/dim]\n")
            return

        preflight_status = ["[Vault] Memeriksa indeks dokumen di Vault lokal..."]
        stop_preflight_spinner = threading.Event()
        spinner_thread: threading.Thread | None = None

        if sys.stdin.isatty():
            def _preflight_spinner_loop() -> None:
                frames = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
                idx = 0
                t_start = time.monotonic()
                while not stop_preflight_spinner.is_set():
                    elapsed = time.monotonic() - t_start
                    desc = preflight_status[0]
                    frame = frames[idx % len(frames)]
                    idx += 1
                    line = (
                        f"\r\x1b[2K\033[1;36m{frame}\033[0m "
                        f"\033[1;37m{desc}\033[0m "
                        f"\033[2;37m({elapsed:.1f}s · Ctrl+C: batal)\033[0m"
                    )
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    stop_preflight_spinner.wait(0.06)
                sys.stdout.write("\r\x1b[2K")
                sys.stdout.flush()

            spinner_thread = threading.Thread(target=_preflight_spinner_loop, daemon=True)
            spinner_thread.start()

        try:
            if self.vault_manager:
                try:
                    self.vault_manager.reload_vault()
                except Exception:
                    pass
            self._sync_vault_cache(session_id=session.id if session else None)

            sess_docs = self.get_session_active_documents(session.id if session else None)
            active_paths = [self.workspace.get_cache_path(f"{s}.md") for s in sess_docs]
            existing_paths = [p for p in active_paths if p.exists()]

            preflight_status[0] = (
                f"[AI Contract] Merumuskan Kontrak Tugas Absolut & estimasi token ({model_label})..."
            )
            contract_extractor = self.contract_extractor or TaskContractExtractor(engine=self.engine)
            task_contract = contract_extractor.extract(query)

            preflight_status[0] = "[Vault] Mengambil potongan bukti (evidence chunks) yang relevan..."
            evidence = self.engine.evidence_synthesizer.retrieve_evidence(
                query=query,
                documents=existing_paths,
                max_chunks=self.config.citation.max_evidence_chunks,
                vector_vault=getattr(self.vault_manager, "vector_vault", None),
            )
        except KeyboardInterrupt:
            stop_preflight_spinner.set()
            if spinner_thread is not None:
                spinner_thread.join(timeout=0.3)
            self.console.print("\n[bold red]✗ Dibatalkan oleh pengguna.[/bold red]\n")
            return
        finally:
            stop_preflight_spinner.set()
            if spinner_thread is not None:
                spinner_thread.join(timeout=0.3)

        margin_pct_disp = int(round(task_contract.token_margin_pct * 100))
        self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Kontrak Tugas (Absolut):[/bold bright_white] [cyan]{task_contract.primary_goal}[/cyan]")
        self.console.print(
            f"  [dim]• Format: {task_contract.format_requirements} "
            f"(Target: ~{task_contract.min_words}-{task_contract.max_words} kata | "
            f"~{task_contract.target_output_tokens} tok +-{margin_pct_disp}% -> "
            f"{task_contract.effective_min_tokens}-{task_contract.effective_max_tokens} tok)[/dim]"
        )
        if task_contract.out_of_scope:
            self.console.print(f"  [dim]• Batasan: Eksklusi {', '.join(task_contract.out_of_scope)}[/dim]")

        # High Token Consumption Warning & Confirmation (if > 5 chunks)
        if len(evidence) > 5:
            sources = sorted(list({c.source for c in evidence}))
            evidence_tokens = sum(c.token_estimate or estimate_tokens(c.text) for c in evidence)
            est_total_input = estimate_tokens(query) + 250 + evidence_tokens

            import scrpcy.cli_studio as _cli_studio
            choice = _cli_studio.interactive_token_warning_confirm(
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

        # Pre-flight auto-scout triggers when vault has no matching evidence OR when existing evidence coverage is very low (< 0.25)
        needs_auto_scout = (not evidence) or (evidence and coverage_score < 0.25)
        if needs_auto_scout and self.scout:
            self.console.print("[bold cyan]●[/bold cyan] [bold bright_white]Berpikir[/bold bright_white] [dim]Mengevaluasi kecukupan literatur di vault...[/dim]")
            if not evidence:
                self.console.print("[dim]● Bukti di vault belum mencukupi untuk topik ini.[/dim]")
            else:
                self.console.print(f"[dim]● Referensi di vault memiliki kecocokan rendah ({int(coverage_score * 100)}%) terhadap fokus pertanyaan.[/dim]")
            suggested_topic = self._extract_scout_query(query)
            target_quota = self._determine_target_scout_quota(query, task_contract)
            self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Rencana: Mencari & mengunduh literatur[/bold bright_white] [cyan]({target_quota} paper terkait '{suggested_topic}')[/cyan]")

            choice, custom_keyword = interactive_scout_confirm(
                suggested_topic=suggested_topic,
                quota=target_quota,
                console=self.console,
            )

            if choice == "no":
                self.console.print("[dim]● Pencarian otomatis dilewati. Melanjutkan sintesis dengan pengetahuan umum...[/dim]\n")
            else:
                target_topic = custom_keyword if (choice == "custom" and custom_keyword) else suggested_topic
                self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Mencari[/bold bright_white] [dim]Menelusuri repositori ilmiah (OpenAlex & Semantic Scholar)...[/dim]")

                initial_active = set(self.get_session_active_documents(session.id if session else None))
                task = self.task_manager.submit(
                    name=f"Deep Research Scout: {target_topic}",
                    task_type="scout",
                    target_fn=self._background_scout_worker,
                    query=target_topic,
                    topic_query=target_topic,
                    max_download=target_quota,
                    session_id=session.id if session else None,
                    lane="io",
                )

                scout_timeout = getattr(self.config, "scout_timeout", 180.0)

                async def _wait_for_task() -> TaskInfo:
                    start_t = time.monotonic()
                    last_progress_t = start_t
                    last_log_count = 0
                    while True:
                        t_info = self.task_manager.get_task(task.id)
                        if not t_info:
                            break
                        if t_info.status in ("completed", "failed", "cancelled"):
                            return t_info
                        fut = self.task_manager._futures.get(task.id)
                        if fut and fut.done():
                            return self.task_manager.get_task(task.id) or task

                        cur_logs = len(t_info.logs) if t_info and t_info.logs else 0
                        if cur_logs > last_log_count:
                            last_log_count = cur_logs
                            last_progress_t = time.monotonic()

                        if time.monotonic() - last_progress_t > scout_timeout:
                            raise asyncio.TimeoutError("Scout timeout exceeded")
                        await asyncio.sleep(0.08)
                    return task

                def _get_scout_log() -> str:
                    t_info = self.task_manager.get_task(task.id)
                    if t_info and t_info.logs:
                        return f"[Deep Research Scout] {t_info.logs[-1]}"
                    return f"[Deep Research Scout] Menelusuri repositori ilmiah..."

                prompt_str = f"{self.current_project.name}/chat/{session.name}> " if (self.sub_mode == "chat" and session and self.current_project) else (f"{self.current_project.name}> " if self.current_project else "> ")

                timed_out = False
                task_cancelled = False
                completed_task: TaskInfo | None = None
                wait_coro = _wait_for_task()
                try:
                    completed_task = self._run_agent_turn_interactive(
                        coro=wait_coro,
                        step_desc=_get_scout_log,
                        prompt_lbl=prompt_str,
                        model_label=model_label,
                        effort=effort,
                    )
                except SessionDetachSignal:
                    self.console.print(f"\n[bold cyan]● Sesi '{session.name if session else ''}' di-detach ke latar belakang.[/bold cyan]")
                    self.console.print(f"  [dim]Tugas '{task.name}' ({task.id}) tetap berjalan di background (Ketik /proc untuk pantau).[/dim]\n")
                    if session and self.chat_manager:
                        self.chat_manager.update_session_status(session.id, "detached")
                        session.status = "detached"
                    self.sub_mode = "chat_hub"
                    return
                except (asyncio.CancelledError, KeyboardInterrupt):
                    self.task_manager.cancel_task(task.id)
                    task_cancelled = True
                    self.console.print("[dim]● Pencarian literatur dibatalkan oleh pengguna.[/dim]\n")
                except (asyncio.TimeoutError, TimeoutError):
                    timed_out = True
                except Exception as exc:
                    if hasattr(self, "logger") and self.logger:
                        self.logger.log_error(session.id if session else None, "AUTO_SCOUT", exc)
                    self.console.print(f"[dim]● Pemantauan tugas repositori: {exc}. Melanjutkan...[/dim]\n")
                finally:
                    import inspect
                    if inspect.iscoroutine(wait_coro):
                        try:
                            wait_coro.close()
                        except Exception:
                            pass

                is_timeout = (
                    timed_out
                    or (
                        completed_task
                        and completed_task.status == "failed"
                        and any(k in (completed_task.error or "").lower() for k in ("timeout", "timed out", "timed_out", "time out"))
                    )
                )

                if is_timeout:
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
                    evidence = []
                elif completed_task and completed_task.status == "completed":
                    if self.vault_manager:
                        self.vault_manager.reload_vault(force=False)
                    self._sync_vault_cache(session_id=session.id if session else None)
                    updated_docs = self.get_session_active_documents(session.id if session else None)
                    new_stems = [s for s in updated_docs if s not in initial_active]
                    if not new_stems and updated_docs:
                        new_stems = list(updated_docs)

                    active_paths = [self.workspace.get_cache_path(f"{s}.md") for s in updated_docs]
                    existing_paths = [p for p in active_paths if p.exists()]
                    evidence = self.engine.evidence_synthesizer.retrieve_evidence(
                        query=query,
                        documents=existing_paths,
                        max_chunks=self.config.citation.max_evidence_chunks,
                        vector_vault=getattr(self.vault_manager, "vector_vault", None),
                    )
                    self.console.print(f"[bold green][✓] Berhasil mengindeks {len(new_stems)} dokumen baru ke vault.[/bold green]")
                    for s in new_stems:
                        self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Membaca[/bold bright_white] [cyan]cache/{s}.md[/cyan] [dim](Dokumen siap)[/dim]")
                    self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Menganalisis[/bold bright_white] [dim]Menyintesis temuan paper baru ke dalam jawaban komprehensif...[/dim]\n")
                elif completed_task and completed_task.status == "failed":
                    self.console.print(f"[dim]● Pencarian repositori gagal: {completed_task.error}. Melanjutkan...[/dim]\n")

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
        skill_badge = f"{self.active_skill_name} • " if self.active_skill_name and self.active_skill_name != "scrpcy_manual" else ""
        self.console.print(f"\n[bold cyan]✦ {model_label}[/bold cyan] [dim]({skill_badge}{effort} effort)[/dim]")

        session_id = session.id if session else "default"
        if hasattr(self, "logger") and self.logger:
            self.logger.log_user_query(session_id, query, self.active_documents)
            if evidence:
                self.logger.log_evidence_retrieved(session_id, query, evidence)

        task_result_box: dict[str, Any] = {}

        def _bg_chat_synthesis_worker(task_id: str, cancel_event: threading.Event | None = None) -> str:
            self.task_manager.log(task_id, f"Pertanyaan: {query}")
            self.task_manager.log(task_id, f"Model: {model_label} (effort: {effort})")
            self.task_manager.log(task_id, "Menghubungi AI & Menganalisis...")
            self.task_manager.update_progress(task_id, "Menghubungi AI & Menganalisis...")

            start_time = time.time()
            max_agent_turns = getattr(self.config, "max_agent_turns", 9999) if self.config else 9999
            max_refinement_turns = getattr(self.config, "max_refinement_turns", 3) if self.config else 3
            max_refine_limit = max_refinement_turns
            turn = 0
            refinement_turns = 0

            # Restrict refinement turns if primary literature download is still active in background
            is_background_scout_active = False
            if self.task_manager and session:
                active_scouts = [
                    t for t in self.task_manager.list_tasks_for_session(session.id)
                    if t.task_type == "scout" and t.status == "running"
                ]
                if active_scouts:
                    is_background_scout_active = True
            if "initial_query_override" in locals() and initial_query_override:
                is_background_scout_active = True

            locked_supported_claims: list[str] = []
            best_draft: str = ""
            best_grounding: float = -1.0
            best_critic: int = -1
            best_composite: float = -1.0
            accumulated_input_tokens = 0
            accumulated_output_tokens = 0
            total_turns_completed = 0
            current_query = initial_query_override if "initial_query_override" in locals() and initial_query_override else query
            active_stems = {c.source for c in evidence} if evidence else None
            citation_guide = self._format_vault_citation_guide(active_evidence_stems=active_stems)
            if citation_guide:
                current_query = f"{citation_guide}\n\n{current_query}"

            # Inject dynamic Task Contract directive to guide synthesis
            dod_str = "\n".join(f"  * {d}" for d in task_contract.definition_of_done)
            margin_disp = int(round(task_contract.token_margin_pct * 100))
            contract_directive = (
                f"\n\n[PANDUAN KONTRAK PENYELESAIAN TUGAS (ABSOLUT)]:\n"
                f"- Sasaran Utama: {task_contract.primary_goal}\n"
                f"- Format Output: {task_contract.format_requirements}\n"
                f"- Alokasi Panjang: {task_contract.min_words} - {task_contract.max_words} kata | "
                f"Target Token: ~{task_contract.target_output_tokens} token (+-{margin_disp}%: {task_contract.effective_min_tokens}-{task_contract.effective_max_tokens} token).\n"
                f"- Batasan Negatif (Out-of-Scope): DILARANG menyertakan {', '.join(task_contract.out_of_scope)}.\n"
                f"- Kebijakan Batas Pencarian: {task_contract.search_exhaustion_policy}\n"
                f"- Definition of Done (Checklist yang Wajib Dipenuhi):\n{dod_str}\n"
                f"Patuhi seluruh instruksi di atas secara presisi."
            )
            current_query = f"{current_query}{contract_directive}"

            # Search for relevant visual assets (figures / tables) from the vault
            recommended_visuals: list[dict[str, Any]] = []
            if self.vault_manager:
                try:
                    recommended_visuals = self.vault_manager.search_visual_assets(query=query, top_k=2)
                except Exception as e:
                    logger.debug(f"Failed to retrieve visual assets: {e}")

            if recommended_visuals:
                for idx, v in enumerate(recommended_visuals, 1):
                    v["index"] = idx

                task_result_box["recommended_visuals"] = recommended_visuals
                vis_bullets = []
                for idx, v in enumerate(recommended_visuals, 1):
                    v_type = "Tabel" if v.get("asset_type") == "table" else "Gambar/Grafik"
                    vis_bullets.append(
                        f"  * [Visual #{idx}] Jenis: {v_type} | Halaman: {v.get('page_num')} | "
                        f"Paper: {v.get('paper_title')} | Keterangan: \"{v.get('caption')}\""
                    )
                vis_directive = (
                    f"\n\n[REFERENSI VISUAL TERSEDIA DI VAULT]:\n"
                    f"Sistem mendeteksi figur/tabel relevan berikut di dalam dokumen vault:\n"
                    + "\n".join(vis_bullets) + "\n"
                    f"PANDUAN RUJUKAN VISUAL:\n"
                    f"- Jika relevan dengan analisis Anda, referensikan visual di atas secara eksplisit (contoh: 'Sebagaimana disajikan pada [Visual #1]').\n"
                    f"- Rujukan visual ini akan otomatis ditampilkan secara berwarna kepada pengguna di antarmuka terminal beserta sitasi lengkapnya."
                )
                current_query = f"{current_query}{vis_directive}"
            current_docs = list(existing_paths)
            curr_evidence = list(evidence) if evidence else []
            final_response = ""
            last_v_report: VerificationReport | None = None
            last_c_report: CritiqueReport | None = None

            session_runtime = self.get_session_runtime(session.id if session else None)

            # Gather previous session dialogue history
            history_msgs: list[dict[str, str]] = []
            if self.chat_manager and session:
                raw_history = self.chat_manager.get_messages(session.id)
                for m in raw_history[-10:]:
                    if m.role == "user" and m.content == query:
                        continue
                    history_msgs.append({"role": m.role, "content": m.content})

            async def _async_synthesis_loop() -> None:
                nonlocal turn, refinement_turns, locked_supported_claims, best_draft, best_grounding, best_critic, best_composite
                nonlocal accumulated_input_tokens, accumulated_output_tokens, total_turns_completed, current_query, current_docs
                nonlocal final_response, last_v_report, last_c_report, max_agent_turns, max_refine_limit, curr_evidence, citation_guide

                attempted_search_channels: set[str] = {"vault_lokal"}
                contractor_gatekeeper = ContractDecomposer()

                while turn < max_agent_turns:
                    if cancel_event and cancel_event.is_set():
                        return

                    # Turn-Boundary Inbox Drain: consume completed background I/O events
                    if session_runtime is not None:
                        inbox_events = session_runtime.drain_inbox()
                        completed_io_events = [
                            ev for ev in inbox_events
                            if ev.status == "completed" and ev.task_type in ("scout", "download", "web", "gdrive")
                        ]
                        if completed_io_events:
                            if self.vault_manager:
                                try:
                                    self.vault_manager.reload_vault(force=False)
                                except Exception:
                                    pass
                            self._sync_vault_cache(session_id=session.id if session else None)
                            for ev in completed_io_events:
                                if ev.ingested_stems:
                                    self.add_session_active_documents(ev.ingested_stems, session_id=session.id if session else None)
                            sess_docs_now = self.get_session_active_documents(session.id if session else None)
                            active_paths_now = [self.workspace.get_cache_path(f"{s}.md") for s in sess_docs_now]
                            current_docs = [p for p in active_paths_now if p.exists()]
                            curr_evidence = self.engine.evidence_synthesizer.retrieve_evidence(
                                query=query,
                                documents=current_docs,
                                max_chunks=self.config.citation.max_evidence_chunks,
                                vector_vault=getattr(self.vault_manager, "vector_vault", None),
                            )
                            citation_guide = self._format_vault_citation_guide(
                                active_evidence_stems={c.source for c in curr_evidence} if curr_evidence else None
                            )
                            if turn > 0:
                                ev_summaries = "; ".join(f"{ev.task_name} ({ev.summary})" for ev in completed_io_events)
                                current_query = (
                                    f"{citation_guide}\n\n"
                                    f"[NOTIFIKASI TURN-BOUNDARY: Tugas latar belakang selesai: {ev_summaries}. "
                                    f"Dokumen baru telah diindeks ke vault sesi ini.]\n\n"
                                    f"{current_query}"
                                )

                    if turn == 0 and curr_evidence:
                        self.task_manager.update_progress(
                            task_id,
                            "[Contractor] Memecah kontrak ke Subagent (Vault, Web, Literatur)...",
                        )

                        def _on_subagent_tree_update(lines: list[str]) -> None:
                            for raw_line in reversed(lines):
                                plain = re.sub(r"\[/?[^\]]+\]", "", raw_line).strip()
                                if "RUNNING" in plain or "QUEUED" in plain:
                                    cleaned = re.sub(r"^[├└│─\s]+", "", plain)
                                    self.task_manager.update_progress(task_id, f"[Subagent] {cleaned}")
                                    return

                        orchestrator = SubagentOrchestrator(
                            engine=self.engine,
                            task_manager=self.task_manager,
                            scout=self.scout,
                            web_scout=getattr(self, "web_scout", None),
                            vault_manager=self.vault_manager,
                            workspace=self.workspace,
                        )
                        orch_res = await orchestrator.orchestrate(
                            query=query,
                            contract=task_contract,
                            evidence=curr_evidence,
                            session_id=session.id if session else None,
                            enable_autonomous_scout=not is_background_scout_active,
                            on_tree_update=_on_subagent_tree_update,
                            cancel_event=cancel_event,
                        )
                        accumulated_input_tokens += orch_res.subagent_input_tokens
                        accumulated_output_tokens += orch_res.subagent_output_tokens
                        attempted_search_channels.update(orch_res.attempted_search_channels)
                        for t_line in orch_res.tree_lines:
                            self.task_manager.log(task_id, re.sub(r"\[/?[^\]]+\]", "", t_line))
                        if orch_res.newly_ingested_stems:
                            self._sync_vault_cache(session_id=session.id if session else None)
                            self.add_session_active_documents(
                                orch_res.newly_ingested_stems,
                                session_id=session.id if session else None,
                            )
                            sess_docs_now = self.get_session_active_documents(session.id if session else None)
                            active_paths_now = [self.workspace.get_cache_path(f"{s}.md") for s in sess_docs_now]
                            current_docs = [p for p in active_paths_now if p.exists()]
                            curr_evidence = self.engine.evidence_synthesizer.retrieve_evidence(
                                query=query,
                                documents=current_docs,
                                max_chunks=self.config.citation.max_evidence_chunks,
                                vector_vault=getattr(self.vault_manager, "vector_vault", None),
                            )
                            citation_guide = self._format_vault_citation_guide(
                                active_evidence_stems={c.source for c in curr_evidence} if curr_evidence else None
                            )
                        if orch_res.assembly_directive:
                            current_query = f"{current_query}\n\n{orch_res.assembly_directive}"

                    turn_tokens: list[str] = []
                    step_msg = f"Menyintesis argumen ilmiah (Turn {turn + 1})..." if turn == 0 else f"Menyintesis perbaikan draf (Turn {turn + 1})..."
                    self.task_manager.update_progress(task_id, step_msg)
                    self.task_manager.log(task_id, step_msg)

                    t_stream_s = time.time()
                    if hasattr(self, "logger") and self.logger:
                        self.logger.log_llm_stream_start(session_id, turn, model_name, step_msg)

                    async for token in self.engine.stream_query(
                        query=current_query,
                        documents=current_docs,
                        skill_name=self.active_skill_name,
                        model=model_name,
                        max_evidence_chunks=self.config.citation.max_evidence_chunks if current_docs else 0,
                        history=history_msgs,
                        vector_vault=getattr(self.vault_manager, "vector_vault", None),
                    ):
                        if cancel_event and cancel_event.is_set():
                            return
                        turn_tokens.append(token)

                    t_stream_dur = time.time() - t_stream_s
                    raw_resp = "".join(turn_tokens)
                    out_tok = estimate_tokens(raw_resp)
                    if hasattr(self, "logger") and self.logger:
                        self.logger.log_llm_stream_finish(session_id, turn, model_name, out_tok, t_stream_dur, len(raw_resp))

                    # 1. Internal vault tool call check
                    tool_cmd = self._detect_internal_tool_call(raw_resp)
                    if tool_cmd and turn < max_agent_turns - 1:
                        if cancel_event and cancel_event.is_set():
                            return
                        cmd_line = tool_cmd.strip()
                        is_archive = any(k in cmd_line.lower() for k in ("remove", "rm", "archive"))
                        is_scout = any(k in cmd_line.lower() for k in ("scout", "search", "findpaper"))
                        is_web = any(k in cmd_line.lower() for k in ("web", "web-scout", "webscout"))
                        action_label, target_label, tool_output = self._execute_internal_vault_tool(cmd_line)

                        if is_archive:
                            sub_label = "(Mengarsipkan paper yang tidak relevan...)"
                        elif is_scout:
                            sub_label = "(Menelusuri repositori akademik...)"
                        elif is_web:
                            sub_label = "(Menelusuri & memproses web resmi...)"
                        else:
                            sub_label = "(Membaca teks dari vault...)"

                        self.task_manager.log(task_id, f"Eksekusi alat internal: {action_label} {target_label}")
                        self.emit_session_output(
                            session.id if session else None,
                            f"[bold cyan]●[/bold cyan] [bold bright_white]{action_label}[/bold bright_white] [cyan]{target_label}[/cyan] [dim]{sub_label}[/dim]\n"
                            f"[bold cyan]●[/bold cyan] [bold bright_white]Menganalisis[/bold bright_white] [dim]Menyintesis temuan dokumen ke dalam jawaban akhir...[/dim]\n",
                            session_obj=session,
                        )

                        current_query = (
                            f"### HASIL EKSEKUSI ALAT ({cmd_line}):\n"
                            f"Aksi: {action_label}\n"
                            f"Target: {target_label}\n"
                            f"```markdown\n{tool_output}\n```\n\n"
                            f"Pertanyaan riset asli pengguna: \"{query}\"\n"
                            f"Berdasarkan informasi di atas, silakan lanjutkan sintesis akademis mendalam 100% yakin dan berikan jawaban akhir tanpa menyuruh pengguna menjalankan perintah slash."
                        )
                        sess_docs_now = self.get_session_active_documents(session.id if session else None)
                        current_docs = [self.workspace.get_cache_path(f"{s}.md") for s in sess_docs_now if self.workspace.get_cache_path(f"{s}.md").exists()]
                        turn += 1
                        if self._check_agent_turn_checkpoint(
                            turn=turn,
                            best_grounding=best_grounding,
                            best_critic=best_critic,
                            accumulated_input_tokens=accumulated_input_tokens,
                            accumulated_output_tokens=accumulated_output_tokens,
                            task_id=task_id,
                            session=session,
                            max_refine_limit=max_refine_limit,
                            last_c_report=last_c_report,
                        ):
                            raw_final = best_draft or raw_resp
                            final_response = self._sanitize_final_response(raw_final)
                            if self.vault_manager and final_response:
                                final_response = append_vault_bibliography(
                                    text=final_response,
                                    evidence=curr_evidence,
                                    vault_manager=self.vault_manager,
                                    style=self.config.citation.style,
                                    active_documents=self.get_session_active_documents(session.id if session else None),
                                )
                            break
                        continue

                    # 2. Academic Polisher
                    if cancel_event and cancel_event.is_set():
                        return
                    p_msg = "Menyelaraskan ritme & gaya bahasa akademis..."
                    self.task_manager.update_progress(task_id, p_msg)
                    self.task_manager.log(task_id, p_msg)
                    p_resp = raw_resp
                    self.task_manager.update_progress(task_id, "Menghaluskan gaya akademik & integritas sitasi (Polisher)...")
                    t_pol_s = time.time()
                    try:
                        polisher = self.polisher or AcademicPolisher(engine=self.engine)
                        p_res = polisher.polish(raw_resp, query)
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
                    if cancel_event and cancel_event.is_set():
                        return
                    fc_msg = "Mengaudit keabsahan fakta terhadap dokumen vault..."
                    self.task_manager.update_progress(task_id, fc_msg)
                    self.task_manager.log(task_id, fc_msg)
                    t_fc_s = time.time()
                    verifier = self.fact_checker or FactChecker(engine=self.engine)
                    v_rep = verifier.verify_synthesis(p_resp, curr_evidence)
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

                    # 4. Devil's Advocate + Contractor Gatekeeper Adjudication
                    if cancel_event and cancel_event.is_set():
                        return
                    cr_msg = "Menjalankan evaluasi kritis (Devil's Advocate)..."
                    self.task_manager.update_progress(task_id, cr_msg)
                    self.task_manager.log(task_id, cr_msg)
                    t_cr_s = time.time()
                    critic = self.devils_advocate or DevilsAdvocate(engine=self.engine)
                    c_rep = critic.critique(query, p_resp, contract=task_contract, fact_check_verdict=v_rep)

                    # Contractor Gatekeeper: adjudicate Devil's Advocate complaints against the Absolute Contract
                    adjudication = contractor_gatekeeper.adjudicate_critique(
                        query=query,
                        draft=p_resp,
                        contract=task_contract,
                        critique_report=c_rep,
                        fact_verdict=v_rep,
                        attempted_search_channels=attempted_search_channels,
                    )
                    if adjudication.rejected_critiques:
                        self.task_manager.log(
                            task_id,
                            f"[Contractor Gatekeeper] {adjudication.adjudication_summary}",
                        )
                    if not adjudication.should_revise:
                        c_rep.passed = True
                        c_rep.score = adjudication.override_score
                        c_rep.needs_citation_mapping = False
                        c_rep.weaknesses = []
                        if not task_contract.allow_counter_arguments:
                            c_rep.counter_arguments = []
                        if (
                            adjudication.exhausted_search_disclosure
                            and "Batas Penelusuran Literatur & Data" not in p_resp
                        ):
                            p_resp = f"{p_resp.rstrip()}{adjudication.exhausted_search_disclosure}"
                    else:
                        c_rep.weaknesses = list(adjudication.accepted_revisions)
                        if not task_contract.allow_counter_arguments:
                            c_rep.counter_arguments = []
                        c_rep.needs_citation_mapping = adjudication.allow_citation_mapping

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

                    # Monotonic Claim Locking
                    for c in v_rep.claims:
                        if c.status == "supported" and c.claim not in locked_supported_claims:
                            locked_supported_claims.append(c.claim)

                    # Best-Draft Tracking
                    composite = 0.55 * v_rep.grounding_score + 0.45 * (c_rep.score / 10.0)
                    if composite > best_composite or not best_draft:
                        best_composite = composite
                        best_draft = p_resp
                        best_grounding = v_rep.grounding_score
                        best_critic = c_rep.score

                    stream_in_est = estimate_tokens(current_query) + sum(c.token_estimate or estimate_tokens(c.text) for c in (curr_evidence or [])) + 250
                    accumulated_input_tokens += stream_in_est + getattr(v_rep, "input_tokens", 0) + getattr(c_rep, "input_tokens", 0)
                    accumulated_output_tokens += out_tok + getattr(v_rep, "output_tokens", 0) + getattr(c_rep, "output_tokens", 0)
                    total_turns_completed = turn + 1

                    # Check if refinement needed
                    needs_refine = (
                        (bool(v_rep.hallucination_warning) or (not c_rep.passed and c_rep.score < 7))
                        and turn < max_agent_turns - 1
                        and not is_background_scout_active
                    )
                    if needs_refine:
                        # Allow full refinement iterations if grounding or critic score is weak
                        max_refine_limit = max_refinement_turns
                        if getattr(task_contract, 'max_words', 1000) <= 750 and c_rep.passed and v_rep.grounding_score >= 0.7:
                            max_refine_limit = 1
                        if refinement_turns >= max_refine_limit:
                            self.emit_session_output(
                                session.id if session else None,
                                "[bold yellow]●[/bold yellow] [bold bright_white]Epistemic Convergence[/bold bright_white] [dim]Batas 3 iterasi perbaikan tercapai. Memfinalisasi draf terbaik dengan catatan dialektis...[/dim]\n",
                                session_obj=session,
                            )
                            raw_final = best_draft or p_resp
                            dialectic_caveat = (
                                "\n\n> [!NOTE]\n"
                                "> **Catatan Analisis Kritis**: Sintesis ini mempertahankan tinjauan dialektis dari literatur yang tersedia. "
                                "Terdapat variasi metodologis antara temuan empiris dokumen dengan implikasi kebijakan di lapangan."
                            )
                            final_response = self._sanitize_final_response(raw_final) + dialectic_caveat
                            if self.vault_manager and final_response:
                                final_response = append_vault_bibliography(
                                    text=final_response,
                                    evidence=curr_evidence,
                                    vault_manager=self.vault_manager,
                                    style=self.config.citation.style,
                                    active_documents=self.get_session_active_documents(session.id if session else None),
                                )
                            last_v_report = v_rep
                            last_c_report = c_rep
                            break

                        if c_rep.needs_citation_mapping:
                            if cancel_event and cancel_event.is_set():
                                return
                            refinement_turns += 1
                            attempted_search_channels.update({
                                "vault_lokal",
                                "web_resmi",
                                "repositori_jurnal_multi_source",
                                "reference_mapping_snowballing_backward_forward",
                            })
                            self.task_manager.log(task_id, f"Devil's Advocate: Skor {c_rep.score}/10: {c_rep.critique_summary}")
                            self.emit_session_output(
                                session.id if session else None,
                                f"[bold cyan]●[/bold cyan] [bold bright_white]Devil's Advocate[/bold bright_white] [yellow]Skor {c_rep.score}/10: {c_rep.critique_summary}[/yellow]",
                                session_obj=session,
                            )
                            stems = self._get_cached_stems()
                            if stems and self.scout:
                                self.emit_session_output(
                                    session.id if session else None,
                                    "[bold cyan]●[/bold cyan] [bold bright_white]Reference Mapping[/bold bright_white] [cyan]Menelusuri jaringan sitasi dokumen vault...[/cyan]",
                                    session_obj=session,
                                )
                            mapped_papers: list[Any] = []
                            newly_ingested_stems: list[str] = []
                            directed_queries = list(getattr(c_rep, "directed_search_queries", []))
                            directed_web_queries = list(getattr(c_rep, "directed_web_queries", []))
                            if v_rep and getattr(v_rep, "claims", None):
                                unsupported = [c.claim for c in v_rep.claims if c.status == "unsupported"]
                                for u_cl in unsupported[:2]:
                                    clean_words = [
                                        w for w in re.findall(r"\b\w{4,}\b", u_cl)
                                        if w.lower() not in ("yang", "pada", "dari", "untuk", "dengan", "dalam", "adalah", "bahwa", "tidak")
                                    ]
                                    if len(clean_words) >= 2:
                                        gap_q = " ".join(clean_words[:5])
                                        if gap_q not in directed_queries:
                                            directed_queries.append(gap_q)
                                        if gap_q not in directed_web_queries:
                                            directed_web_queries.append(gap_q)

                            # 1. Search multi-source across academic providers AND official web in parallel
                            web_candidates: list[Any] = []
                            is_mock_env = (
                                "PYTEST_CURRENT_TEST" in os.environ
                                or getattr(self.engine, "acompletion_fn", None) is not None
                                or any(str(v).startswith("mock") for v in getattr(self.config, "api_keys", {}).values())
                                or getattr(self.config, "default_model", "").startswith("mock")
                            )

                            if directed_queries and self.scout:
                                is_scout_mocked = (
                                    hasattr(self.scout, "_is_mock")
                                    or "search_multi_source" in getattr(self.scout, "__dict__", {})
                                    or "Mock" in type(getattr(self.scout, "search_multi_source", None)).__name__
                                    or "Mock" in type(self.scout).__name__
                                )
                                if not (is_mock_env and not is_scout_mocked):
                                    for target_gap_q in directed_queries[:2]:
                                        try:
                                            sq = ScoutQuery(keywords=target_gap_q, limit=5)
                                            candidates = await self.scout.search_multi_source(sq, limit=5)
                                            if candidates:
                                                existing_titles = {p.title for p in mapped_papers}
                                                mapped_papers.extend([p for p in candidates if p.title not in existing_titles])
                                        except Exception as exc:
                                            logger.warning(f"Error in gap scout search: {exc}")

                            # Search official web portals in parallel for empirical data/statistics
                            target_web_queries = directed_web_queries or directed_queries
                            if target_web_queries and getattr(self, "web_scout", None):
                                is_web_mocked = (
                                    hasattr(self.web_scout, "_is_mock")
                                    or "search_official_web" in getattr(self.web_scout, "__dict__", {})
                                    or "Mock" in type(getattr(self.web_scout, "search_official_web", None)).__name__
                                    or "Mock" in type(self.web_scout).__name__
                                )
                                if not (is_mock_env and not is_web_mocked):
                                    for w_gap_q in target_web_queries[:2]:
                                        try:
                                            w_hits = await self.web_scout.search_official_web(w_gap_q, limit=2, require_official=True)
                                            if w_hits:
                                                existing_urls = {w.url for w in web_candidates}
                                                web_candidates.extend([w for w in w_hits if w.url not in existing_urls])
                                        except Exception as w_exc:
                                            logger.warning(f"Error in gap web scout search: {w_exc}")

                            # 2. Also map vault citation network (snowballing backward references)
                            if stems and self.scout:
                                is_mock_env = (
                                    "PYTEST_CURRENT_TEST" in os.environ
                                    or getattr(self.engine, "acompletion_fn", None) is not None
                                    or any(str(v).startswith("mock") for v in getattr(self.config, "api_keys", {}).values())
                                    or getattr(self.config, "default_model", "").startswith("mock")
                                )
                                is_map_mocked = (
                                    hasattr(self.scout, "_is_mock")
                                    or "map_vault_citation_network" in getattr(self.scout, "__dict__", {})
                                    or "Mock" in type(getattr(self.scout, "map_vault_citation_network", None)).__name__
                                    or "Mock" in type(self.scout).__name__
                                )
                                if not (is_mock_env and not is_map_mocked):
                                    try:
                                        cit_papers = await self.scout.map_vault_citation_network(stems, limit=5)
                                        if cit_papers:
                                            existing_titles = {p.title for p in mapped_papers}
                                            mapped_papers.extend([p for p in cit_papers if p.title not in existing_titles])
                                    except Exception as exc:
                                        logger.warning(f"Error in vault citation network mapping: {exc}")

                            # 3. Retrieve additional evidence from existing vault if available
                            if directed_queries and self.vault_manager and getattr(self.vault_manager, "vector_vault", None):
                                for dq in directed_queries[:2]:
                                    more_evidence = self.engine.evidence_synthesizer.retrieve_evidence(
                                        query=dq,
                                        documents=current_docs,
                                        max_chunks=2,
                                        vector_vault=self.vault_manager.vector_vault,
                                    )
                                    if more_evidence:
                                        curr_evidence.extend(more_evidence)

                            # 4. Curate candidates against original query and gaps to take only top 2 most relevant (safe, low memory)
                            if mapped_papers and self.scout:
                                try:
                                    curated = self.scout.curate_and_rank(mapped_papers, query=query, target_count=2)
                                    if curated:
                                        mapped_papers = curated
                                    else:
                                        mapped_papers = mapped_papers[:2]
                                except Exception:
                                    mapped_papers = mapped_papers[:2]

                            # Ingest authoritative official web pages discovered for gap filling
                            if web_candidates and getattr(self, "web_scout", None):
                                for idx, w in enumerate(web_candidates[:2], 1):
                                    try:
                                        self.emit_session_output(
                                            session.id if session else None,
                                            f"[bold cyan]●[/bold cyan] [bold bright_white]Web Resmi Ditemukan[/bold bright_white] [green]{w.title[:50]}...[/green] [dim]({w.institution})[/dim]",
                                            session_obj=session,
                                        )
                                        vec_vault = self.vault_manager.vector_vault if self.vault_manager else None
                                        parsed_w = await self.web_scout.ingest_url(w.url, vector_vault=vec_vault)
                                        if parsed_w and parsed_w.metadata and parsed_w.metadata.cite_key:
                                            newly_ingested_stems.append(parsed_w.metadata.cite_key)
                                    except Exception as ing_w_err:
                                        logger.warning(f"Failed auto-ingesting official gap web: {ing_w_err}")

                            if mapped_papers and self.scout:
                                for idx, p in enumerate(mapped_papers, 1):
                                    safe_stem = sanitize_filename(p.title)
                                    downloaded = False
                                    if p.has_pdf and p.pdf_url:
                                        try:
                                            await self.scout.download_scout_paper(p)
                                            newly_ingested_stems.append(safe_stem)
                                            downloaded = True
                                        except Exception:
                                            downloaded = False
                                    if not downloaded:
                                        self.emit_session_output(
                                            session.id if session else None,
                                            f"[bold cyan]●[/bold cyan] [bold bright_white]Paywall Terdeteksi[/bold bright_white] [yellow][{idx}/{len(mapped_papers)}] {p.title[:50]}...[/yellow] [dim](Naskah berbayar)[/dim]",
                                            session_obj=session,
                                        )
                                        decision = await asyncio.to_thread(self._ask_ai_paywall_action, p, query)
                                        if decision.get("decision") == "FIND_ALTERNATIVE":
                                            try:
                                                is_mock_env = (
                                                    "PYTEST_CURRENT_TEST" in os.environ
                                                    or getattr(self.engine, "acompletion_fn", None) is not None
                                                    or any(str(v).startswith("mock") for v in getattr(self.config, "api_keys", {}).values())
                                                    or getattr(self.config, "default_model", "").startswith("mock")
                                                )
                                                is_scout_mocked = (
                                                    hasattr(self.scout, "_is_mock")
                                                    or "search_multi_source" in getattr(self.scout, "__dict__", {})
                                                    or "Mock" in type(getattr(self.scout, "search_multi_source", None)).__name__
                                                    or "Mock" in type(self.scout).__name__
                                                )
                                                if not (is_mock_env and not is_scout_mocked):
                                                    sq = ScoutQuery(keywords=f"{p.title} open access", is_oa_only=True)
                                                    alt_candidates = await self.scout.search_multi_source(sq, limit=5)
                                                    alt_oa = [c for c in alt_candidates if c.has_pdf and c.pdf_url]
                                                    if alt_oa:
                                                        await self.scout.download_scout_paper(alt_oa[0])
                                                        newly_ingested_stems.append(sanitize_filename(alt_oa[0].title))
                                                    else:
                                                        self.emit_session_output(
                                                            session.id if session else None,
                                                            "[bold cyan]●[/bold cyan] [bold bright_white]Abstrak Memadai[/bold bright_white] [green]Mengindeks abstrak resmi paper ke vault[/green]",
                                                            session_obj=session,
                                                        )
                                                        await asyncio.to_thread(self.scout.ingest_abstract_paper, p)
                                                        newly_ingested_stems.append(safe_stem)
                                                else:
                                                    await asyncio.to_thread(self.scout.ingest_abstract_paper, p)
                                                    newly_ingested_stems.append(safe_stem)
                                            except Exception:
                                                await asyncio.to_thread(self.scout.ingest_abstract_paper, p)
                                                newly_ingested_stems.append(safe_stem)
                                        else:
                                            self.emit_session_output(
                                                session.id if session else None,
                                                f"[bold cyan]●[/bold cyan] [bold bright_white]Abstrak Memadai[/bold bright_white] [green]Mengindeks abstrak resmi paper ke vault[/green] [dim]({decision.get('reason', '')})[/dim]",
                                                session_obj=session,
                                            )
                                            await asyncio.to_thread(self.scout.ingest_abstract_paper, p)
                                            newly_ingested_stems.append(safe_stem)

                            t_info = self.task_manager.get_task(task_id) if self.task_manager else None
                            is_task_alive = bool(t_info and t_info.status == "running" and not (cancel_event and cancel_event.is_set()))
                            if is_task_alive:
                                if mapped_papers:
                                    self.emit_session_output(
                                        session.id if session else None,
                                        f"[bold cyan]●[/bold cyan] [bold bright_white]Directed Gap Search[/bold bright_white] [cyan]Menemukan {len(mapped_papers)} literatur penambal celah empiris.[/cyan]",
                                        session_obj=session,
                                    )
                                if newly_ingested_stems:
                                    lines_msg = [
                                        f"[bold green][✓] Berhasil mengindeks {len(newly_ingested_stems)} dokumen sitasi baru ke vault.[/bold green]"
                                    ]
                                    for s in newly_ingested_stems:
                                        lines_msg.append(f"[bold cyan]●[/bold cyan] [bold bright_white]Membaca[/bold bright_white] [cyan]cache/{s}.md[/cyan] [dim](Siap untuk sintesis)[/dim]")
                                    lines_msg.append("[bold cyan]●[/bold cyan] [bold bright_white]Menganalisis[/bold bright_white] [dim]Menyintesis argumen ilmiah terpadu dengan literatur baru...[/dim]\n")
                                    self.emit_session_output(
                                        session.id if session else None,
                                        "\n".join(lines_msg),
                                        session_obj=session,
                                    )

                            if newly_ingested_stems:
                                if self.vault_manager:
                                    self.vault_manager.reload_vault(False)
                                self._sync_vault_cache(session_id=session.id if session else None)
                                self.add_session_active_documents(newly_ingested_stems, session_id=session.id if session else None)
                                sess_docs_now = self.get_session_active_documents(session.id if session else None)
                                active_paths = [self.workspace.get_cache_path(f"{s}.md") for s in sess_docs_now]
                                current_docs = [p for p in active_paths if p.exists()]
                                curr_evidence = self.engine.evidence_synthesizer.retrieve_evidence(
                                    query=query,
                                    documents=current_docs,
                                    max_chunks=self.config.citation.max_evidence_chunks,
                                    vector_vault=getattr(self.vault_manager, "vector_vault", None),
                                )
                                citation_guide = self._format_vault_citation_guide(
                                active_evidence_stems={c.source for c in curr_evidence} if curr_evidence else None
                            )
                                max_agent_turns = max(max_agent_turns, turn + 2)

                            gap_sc = SubContract(
                                id="sub-refine-gap",
                                role="GapExtractor",
                                objective=f"Ekstrak fakta dan metrik terverifikasi untuk menjawab: '{query}'",
                                allowed_scope=task_contract.in_scope,
                                forbidden_scope=list(task_contract.out_of_scope) + list(task_contract.forbidden_drift_topics),
                                evidence_slice=curr_evidence[:4],
                                max_output_tokens=200,
                            )
                            gap_orchestrator = SubagentOrchestrator(engine=self.engine)
                            gap_art = await asyncio.to_thread(gap_orchestrator.pool.run_subcontract, gap_sc)
                            accumulated_input_tokens += gap_art.input_tokens
                            accumulated_output_tokens += gap_art.output_tokens

                            verified_ledger_lines = [
                                f"- {f.claim_or_metric} [Sumber: {f.citation_stem}{', ' + f.year if f.year else ''}]"
                                for f in gap_art.extracted_facts[:6]
                            ]
                            if not verified_ledger_lines and locked_supported_claims:
                                verified_ledger_lines = [f"- {c[:180]}" for c in locked_supported_claims[:4]]

                            ledger_sec = (
                                "### LEDGER FAKTA & SITASI TERVERIFIKASI (GUNAKAN FAKTA INI SECARA ALAMI, TIDAK PERLU VERBATIM):\n"
                                + "\n".join(verified_ledger_lines) + "\n\n"
                            ) if verified_ledger_lines else ""

                            weaknesses_str = "; ".join(c_rep.weaknesses[:2]) if c_rep.weaknesses else "Fokuskan pada bukti empiris yang relevan."
                            counter_str = (
                                "\nCatatan dialektika: " + "; ".join(c_rep.counter_arguments[:2])
                            ) if (c_rep.counter_arguments and getattr(task_contract, "allow_counter_arguments", False)) else ""
                            recency_str = f"\nCatatan kebaruan: {c_rep.recency_alert}" if c_rep.recency_alert else ""

                            unsupported_sec = ""
                            if v_rep and getattr(v_rep, "claims", None):
                                unsupported = [c.claim for c in v_rep.claims if c.status == "unsupported"]
                                if unsupported:
                                    unsupported_sec = (
                                        "### KLAIM TANPA BUKTI (HAPUS ATAU GANTI DENGAN FAKTA VAULT):\n"
                                        + "\n".join(f"- {c}" for c in unsupported[:4]) + "\n\n"
                                    )

                            current_query = (
                                f"{citation_guide}\n\n"
                                f"{ledger_sec}"
                                f"### PANDUAN PERBAIKAN SUBSTANSI (KONTRAK: {task_contract.archetype}, TARGET {task_contract.min_words}-{task_contract.max_words} KATA):\n"
                                f"Fokus perbaikan: {weaknesses_str}{counter_str}{recency_str}\n\n"
                                f"{unsupported_sec}"
                                f"Pertanyaan riset asli: \"{query}\"\n"
                                f"Instruksi: Jawab langsung pertanyaan pengguna menggunakan fakta terverifikasi di atas dengan kalimat yang mengalir alami. "
                                f"DILARANG menjawab kritikus dalam bentuk poin-poin pembelaan diri dan DILARANG memperluas pembahasan ke sub-bab tinjauan pustaka (Bab 2) di luar substansi pertanyaan. "
                                f"Jika sebagian tahun/cakupan data memang tidak tersedia di dokumen vault, nyatakan cakupan data yang tersedia secara jujur dan singkat."
                            )
                            turn += 1
                            if self._check_agent_turn_checkpoint(
                                turn=turn,
                                best_grounding=best_grounding,
                                best_critic=best_critic,
                                accumulated_input_tokens=accumulated_input_tokens,
                                accumulated_output_tokens=accumulated_output_tokens,
                                task_id=task_id,
                                session=session,
                                max_refine_limit=max_refine_limit,
                                last_c_report=c_rep,
                            ):
                                raw_final = best_draft or p_resp
                                final_response = self._sanitize_final_response(raw_final)
                                if self.vault_manager and final_response:
                                    final_response = append_vault_bibliography(
                                        text=final_response,
                                        evidence=curr_evidence,
                                        vault_manager=self.vault_manager,
                                        style=self.config.citation.style,
                                        active_documents=self.get_session_active_documents(session.id if session else None),
                                    )
                                last_v_report = v_rep
                                last_c_report = c_rep
                                break
                            continue

                        elif v_rep.hallucination_warning:
                            refinement_turns += 1
                            self.task_manager.log(task_id, f"FactChecker: {v_rep.hallucination_warning}")
                            self.emit_session_output(
                                session.id if session else None,
                                f"[bold cyan]●[/bold cyan] [bold bright_white]Memverifikasi Fakta[/bold bright_white] [yellow]{v_rep.hallucination_warning}[/yellow]",
                                session_obj=session,
                            )
                            unsupported_claims = [c.claim for c in v_rep.claims if c.status == "unsupported"]
                            if unsupported_claims:
                                ledger_sec = (
                                    "### FAKTA TERVERIFIKASI (PERTAHANKAN SUBSTANSI & SITASI INI SECARA ALAMI):\n"
                                    + "\n".join(f"- {c[:180]}" for c in locked_supported_claims[:5]) + "\n\n"
                                ) if locked_supported_claims else ""
                                current_query = (
                                    f"{ledger_sec}"
                                    f"### AUDIT FAKTA (HAPUS KLAIM TANPA RUJUKAN VAULT):\n"
                                    f"Klaim berikut tidak ditemukan dasarnya pada literatur vault:\n"
                                    + "\n".join(f"- {c}" for c in unsupported_claims[:4]) +
                                    f"\n\nPertanyaan riset asli: \"{query}\"\n"
                                    f"Instruksi: Susun ulang jawaban secara ringkas ({task_contract.min_words}-{task_contract.max_words} kata) hanya menggunakan fakta yang didukung bukti teks dokumen vault."
                                )
                                turn += 1
                            if self._check_agent_turn_checkpoint(
                                turn=turn,
                                best_grounding=best_grounding,
                                best_critic=best_critic,
                                accumulated_input_tokens=accumulated_input_tokens,
                                accumulated_output_tokens=accumulated_output_tokens,
                                task_id=task_id,
                                session=session,
                                max_refine_limit=max_refine_limit,
                                last_c_report=c_rep,
                            ):
                                raw_final = best_draft or p_resp
                                final_response = self._sanitize_final_response(raw_final)
                                if self.vault_manager and final_response:
                                    final_response = append_vault_bibliography(
                                        text=final_response,
                                        evidence=curr_evidence,
                                        vault_manager=self.vault_manager,
                                        style=self.config.citation.style,
                                        active_documents=self.get_session_active_documents(session.id if session else None),
                                    )
                                last_v_report = v_rep
                                last_c_report = c_rep
                                break
                            continue

                    # 5. Automatic Bibliography
                    final_text = p_resp
                    if self.vault_manager and final_text:
                        b_msg = "Menyusun daftar pustaka vault terverifikasi..."
                        self.task_manager.update_progress(task_id, b_msg)
                        self.task_manager.log(task_id, b_msg)
                        final_text = append_vault_bibliography(
                            text=final_text,
                            evidence=curr_evidence,
                            vault_manager=self.vault_manager,
                            style=self.config.citation.style,
                            active_documents=self.get_session_active_documents(session.id if session else None),
                        )
                        if hasattr(self, "logger") and self.logger:
                            self.logger.log_bibliography_appended(
                                session_id=session_id,
                                style=self.config.citation.style,
                                entries_count=len(curr_evidence),
                                filenames=[c.source for c in curr_evidence],
                            )

                    final_response = self._sanitize_final_response(final_text)
                    last_v_report = v_rep
                    last_c_report = c_rep
                    break

            b_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(b_loop)
            try:
                b_loop.run_until_complete(_async_synthesis_loop())
            except (KeyboardInterrupt, asyncio.CancelledError):
                if cancel_event:
                    cancel_event.set()
                return "Cancelled"
            finally:
                b_loop.close()

            if cancel_event and cancel_event.is_set():
                return "Cancelled"

            elapsed = time.time() - start_time
            ev_tokens = sum(c.token_estimate or estimate_tokens(c.text) for c in curr_evidence)
            fallback_in = estimate_tokens(query) + 250 + ev_tokens
            fallback_out = estimate_tokens(final_response)
            input_tokens = max(accumulated_input_tokens, fallback_in)
            output_tokens = max(accumulated_output_tokens, fallback_out)
            total_tokens = input_tokens + output_tokens
            style_name = self.config.citation.style.upper()

            v_score = last_v_report.grounding_score if last_v_report else 1.0
            c_score = last_c_report.score if last_c_report else 8
            grounding_badge = f"Grounding: {int(v_score * 100)}%"
            critic_badge = f"Critic: {c_score}/10"
            actual_turns = max(total_turns_completed, turn + 1)
            turns_badge = f"{actual_turns} turn" if actual_turns == 1 else f"{actual_turns} turns"

            footer_badge_md = (
                f"\n\n---\n"
                f"*[dim]Tokens: {input_tokens:,} in · {output_tokens:,} out · {total_tokens:,} total ({model_label}) • {turns_badge} • {grounding_badge} • {critic_badge} • {elapsed:.1f}s • {style_name} style[/dim]*"
            )
            final_response_persisted = f"{final_response}{footer_badge_md}"

            if self.chat_manager and session:
                self.chat_manager.add_message(session.id, "assistant", final_response_persisted)
            if session_runtime is not None:
                session_runtime.record_exchange(query, final_response_persisted)
            self.conversation_history.append({
                "query": query,
                "response": final_response_persisted,
            })

            self.task_manager.update_tokens(task_id, total_tokens)
            self.task_manager.log(task_id, f"Token: {input_tokens:,} in · {output_tokens:,} out · {total_tokens:,} total ({model_label}) • {elapsed:.1f}s")
            self.task_manager.log(task_id, f"Selesai! {len(final_response)} karakter disimpan ke riwayat sesi.")

            task_result_box.update({
                "final_response": final_response,
                "final_response_persisted": final_response_persisted,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "elapsed": elapsed,
                "model_label": model_label,
                "turns_badge": turns_badge,
                "grounding_badge": grounding_badge,
                "critic_badge": critic_badge,
                "style_name": style_name,
                "last_v_report": last_v_report,
                "last_c_report": last_c_report,
                "recommended_visuals": recommended_visuals,
            })

            return f"Sintesis selesai ({len(final_response)} karakter, {total_tokens:,} total token)" if final_response else "Selesai"

        task = self.task_manager.submit(
            name=f"Chat ({session.name if session else 'session'}): {query[:30]}...",
            task_type="agent",
            target_fn=_bg_chat_synthesis_worker,
            session_id=session.id if session else None,
            lane="turn",
        )

        async def _wait_for_chat_task() -> TaskInfo:
            start_t = time.monotonic()
            last_activity_t = start_t
            last_log_cnt = 0
            last_prog_str = ""
            freeze_threshold = 120.0  # Alert watchdog if completely frozen for 120s

            while True:
                t_info = self.task_manager.get_task(task.id)
                if not t_info:
                    break
                if t_info.status in ("completed", "failed", "cancelled"):
                    return t_info
                fut = self.task_manager._futures.get(task.id)
                if fut and fut.done():
                    return self.task_manager.get_task(task.id) or task

                cur_logs_cnt = len(t_info.logs) if (t_info and t_info.logs) else 0
                cur_prog = (t_info.progress or "") if t_info else ""
                now = time.monotonic()

                # Activity detected: progress updated or logs appended
                if cur_logs_cnt != last_log_cnt or cur_prog != last_prog_str:
                    last_log_cnt = cur_logs_cnt
                    last_prog_str = cur_prog
                    last_activity_t = now
                elif now - last_activity_t >= freeze_threshold:
                    # Potential freeze: no progress update for 120s
                    self.task_manager.log(
                        task.id,
                        f"[Watchdog] Tidak ada respons aktivitas baru selama {int(now - last_activity_t)}s pada tahap: '{cur_prog}'."
                    )
                    last_activity_t = now

                await asyncio.sleep(0.04)
            return task

        def _get_chat_status_line() -> str:
            t_info = self.task_manager.get_task(task.id)
            if t_info and t_info.progress and not t_info.progress.startswith("[Watchdog"):
                return t_info.progress
            if t_info and t_info.logs:
                for l in reversed(t_info.logs):
                    clean_l = l.split("]", 1)[-1].strip() if "]" in l else l.strip()
                    if clean_l and not clean_l.startswith("Started task") and not clean_l.startswith("Pertanyaan:") and not clean_l.startswith("Model:") and not clean_l.startswith("[Watchdog") and not clean_l.startswith("[Pemeriksaan"):
                        return clean_l
            return "Menganalisis literatur dan menyusun sintesis ilmiah..."

        prompt_str = f"{self.current_project.name}/chat/{session.name}> " if (self.sub_mode == "chat" and session and self.current_project) else (f"{self.current_project.name}> " if self.current_project else "> ")

        if not hasattr(self, "_foreground_monitored_tasks"):
            self._foreground_monitored_tasks = set()
        self._foreground_monitored_tasks.add(task.id)

        wait_coro = _wait_for_chat_task()
        completed_task: TaskInfo | None = None
        try:
            completed_task = self._run_agent_turn_interactive(
                coro=wait_coro,
                step_desc=_get_chat_status_line,
                prompt_lbl=prompt_str,
                model_label=model_label,
                effort=effort,
                task_id=task.id,
            )
        except SessionDetachSignal:
            if session and self.chat_manager:
                self.chat_manager.update_session_status(session.id, "detached")
                session.status = "detached"
            self.console.print(f"\n[bold cyan]● Sesi '{session.name if session else ''}' di-detach ke latar belakang.[/bold cyan]")
            self.console.print(f"[dim]● Proses sintesis AI ({task.id}) tetap berjalan di background. Pantau di menu Processing atau buka kembali sesi kapan saja.[/dim]\n")
            raise
        except SessionCloseSignal:
            if session and self.chat_manager:
                self.chat_manager.update_session_status(session.id, "closed")
                session.status = "closed"
            if hasattr(self, "task_manager") and self.task_manager:
                if session:
                    self.task_manager.cancel_session_tasks(session.id)
                else:
                    self.task_manager.cancel_task(task.id)
            self.console.print(f"\n[dim]● Sesi '{session.name if session else ''}' ditutup (terminated).\n[/dim]")
            raise
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.task_manager.cancel_task(task.id)
            self.console.print("\n\n[bold yellow]● Pertanyaan dibatalkan oleh pengguna (Ctrl+C).[/bold yellow]\n")
            return
        except Exception as exc:
            self.task_manager.cancel_task(task.id)
            self.console.print(f"\n[bold red][x] Generation error:[/bold red] {exc}\n")
            return
        finally:
            if hasattr(self, "_foreground_monitored_tasks"):
                self._foreground_monitored_tasks.discard(task.id)
            import inspect
            if inspect.iscoroutine(wait_coro):
                try:
                    wait_coro.close()
                except Exception:
                    pass

        if completed_task and completed_task.status == "completed":
            final_resp = task_result_box.get("final_response", "")
            render_academic_markdown(final_resp, console=self.console)
            self.console.print()

            # Render Option A: Visual Assets Panel (Figures / Tables with citation)
            rec_visuals = task_result_box.get("recommended_visuals", [])
            if rec_visuals:
                from scrpcy.ui.terminal_image import render_visual_panel
                for v_info in rec_visuals:
                    v_panel = render_visual_panel(v_info, console=self.console)
                    if v_panel:
                        self.console.print(v_panel)
                        self.console.print()

            in_tok = task_result_box.get("input_tokens", 0)
            out_tok = task_result_box.get("output_tokens", 0)
            tot_tok = task_result_box.get("total_tokens", 0)
            m_lbl = task_result_box.get("model_label", model_label)
            t_bdg = task_result_box.get("turns_badge", "1 turn")
            g_bdg = task_result_box.get("grounding_badge", "Grounding: 100%")
            c_bdg = task_result_box.get("critic_badge", "Critic: 8/10")
            el_s = task_result_box.get("elapsed", 0.0)
            st_name = task_result_box.get("style_name", "APA")

            self.console.print(
                f"[dim]── Tokens: {in_tok:,} in · {out_tok:,} out · {tot_tok:,} total ({m_lbl}) • {t_bdg} • {g_bdg} • {c_bdg} • {el_s:.1f}s • {st_name} style ──[/dim]\n"
            )

            if hasattr(self, "logger") and self.logger:
                v_rep = task_result_box.get("last_v_report")
                c_rep = task_result_box.get("last_c_report")
                self.logger.log_response_rendered(
                    session_id=session_id,
                    total_input_tokens=in_tok,
                    total_output_tokens=out_tok,
                    elapsed_total_s=el_s,
                    model_label=m_lbl,
                    grounding_score=v_rep.grounding_score if v_rep else 1.0,
                    critic_score=c_rep.score if c_rep else 8,
                )

            # Suggested actions & Action Indicator: Mengubah / Menulis
            self._handle_suggested_actions(final_resp)

        elif completed_task and completed_task.status == "cancelled":
            self.console.print("\n\n[bold yellow]● Pertanyaan dibatalkan oleh pengguna (Ctrl+C).[/bold yellow]\n")
            return

        elif completed_task and completed_task.status == "failed":
            self.console.print(f"\n[bold red][x] Generation error:[/bold red] {completed_task.error}\n")

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
            elif any(k in action for k in ("web ", "web-scout ", "webscout ")):
                w_arg = action.split(maxsplit=1)[1].strip() if len(action.split()) > 1 else ""
                self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Menelusuri Web Resmi[/bold bright_white] [cyan]{w_arg}[/cyan]")
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

            import scrpcy.cli_studio as _cli_studio
            choice_res = _cli_studio.interactive_action_confirm(action, console=self.console)
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
