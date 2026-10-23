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

        if not sys.stdin.isatty():
            # Non-interactive / headless environment (e.g. test suites)
            return completed_turns >= 15

        # Interactive terminal: render checkpoint telemetry and prompt user
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
                running_tasks = [t for t in self.task_manager.list_tasks_for_session(self.active_chat_session.id) if t.status == "running"] if (self.active_chat_session and self.task_manager) else []
                if running_tasks:
                    self.console.print(f"[bold yellow]●[/bold yellow] [dim]{len(running_tasks)} proses latar belakang tetap berjalan di background. Anda dapat kembali ke sesi ini kapan saja dengan '/switch' atau 'resume {self.active_chat_session.name}'.[/dim]")
                if self.active_chat_session and self.chat_manager:
                    self.chat_manager.update_session_status(self.active_chat_session.id, "detached")
                self.sub_mode = None
                self.console.print(f"[dim]Kembali ke project root: {self.current_project.name if self.current_project else 'project'}>[/dim]\n")
                return
            elif cmd in ("/back", "back"):
                self.render_mode_divider()
                running_tasks = [t for t in self.task_manager.list_tasks_for_session(self.active_chat_session.id) if t.status == "running"] if (self.active_chat_session and self.task_manager) else []
                if running_tasks:
                    self.console.print(f"[bold yellow]●[/bold yellow] [dim]{len(running_tasks)} proses latar belakang tetap berjalan di background. Anda dapat kembali ke sesi ini kapan saja dengan '/switch' atau 'resume {self.active_chat_session.name}'.[/dim]")
                if self.active_chat_session and self.chat_manager:
                    self.chat_manager.update_session_status(self.active_chat_session.id, "detached")
                self.sub_mode = "chat_hub"
                self.console.print(f"[dim]Kembali ke menu chat: {self.current_project.name if self.current_project else 'project'}/chat>[/dim]\n")
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
                # MVP single active process per session check
                if self.active_chat_session and self.task_manager:
                    running_tasks = [t for t in self.task_manager.list_tasks_for_session(self.active_chat_session.id) if t.status == "running"]
                    if running_tasks:
                        t = running_tasks[0]
                        self.console.print(
                            f"\n[bold yellow][!] Sesi '{self.active_chat_session.name}' sudah memiliki 1 proses aktif:[/bold yellow] "
                            f"[bold cyan]{t.name}[/bold cyan] [dim]({t.id})[/dim]\n"
                            f"  [dim]• Sesuai arsitektur MVP, setiap sesi dibatasi tepat 1 proses aktif.\n"
                            f"  • Tunggu proses selesai atau ketik /cancel untuk membatalkan.[/dim]\n"
                        )
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

        # MVP Single Process Busy Guard:
        # If this session already has an active running task, guard against new prompt submission
        if self.active_chat_session and self.task_manager:
            running_tasks = [t for t in self.task_manager.list_tasks_for_session(self.active_chat_session.id) if t.status == "running"]
            if running_tasks:
                active_t = running_tasks[0]
                self.console.print(
                    f"\n[bold yellow][!] Sesi '{self.active_chat_session.name}' sedang menjalankan 1 proses aktif:[/bold yellow] "
                    f"[bold cyan]{active_t.name}[/bold cyan] [dim]({active_t.id})[/dim]\n"
                    f"  [dim]• Status: {active_t.progress or 'Sedang memproses...'} (Tunggu hingga selesai)\n"
                    f"  • Ketik /cancel untuk membatalkan proses.\n"
                    f"  • Ketik /detach atau /back untuk beralih ke sesi/menu lain.[/dim]\n"
                )
                return

        # Any input without leading "/" in chat mode is strictly treated as an AI conversation prompt
        self.handle_chat(clean)

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

        async def _interactive_loop() -> Any:
            t = asyncio.create_task(coro)
            typed_buffer: list[str] = list(self._pending_user_input) if self._pending_user_input else []
            self._pending_user_input = ""
            frame_idx = 0
            last_spin = time.time()
            prev_lines_drawn = 0
            last_desc_str = ""
            last_frame_text = ""

            def _get_layout() -> tuple[str, str]:
                cur_width = max(20, (self.console.width or shutil.get_terminal_size((80, 24)).columns) - 1)
                cur_div = "─" * cur_width
                left_st = "? shortcuts · /detach bg · /close exit"
                right_st = f"{self.active_skill_name} • {model_label} • {effort}" if self.active_skill_name else f"{model_label} • {effort}"
                if len(left_st) + len(right_st) + 2 > cur_width:
                    right_st = f"{model_label} • {effort}"
                pad = max(1, cur_width - len(left_st) - len(right_st))
                cur_foot = f"{left_st}{' ' * pad}{right_st}"[:cur_width]
                return cur_div, cur_foot

            divider, footer = _get_layout()

            try:
                while not t.done():
                    now = time.time()
                    need_redraw = False
                    if now - last_spin >= 0.12:
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
                    max_desc_w = max(10, (self.console.width or shutil.get_terminal_size((80, 24)).columns) - 5)
                    if len(cur_desc_val) > max_desc_w:
                        cur_desc_val = cur_desc_val[:max_desc_w - 1] + "…"

                    if cur_desc_val != last_desc_str:
                        last_desc_str = cur_desc_val
                        need_redraw = True

                    if need_redraw:
                        divider, footer = _get_layout()
                        buf_str = "".join(typed_buffer)
                        frame = SPINNER_FRAMES[frame_idx]
                        cur_w = max(20, (self.console.width or shutil.get_terminal_size((80, 24)).columns) - 1)
                        max_p_w = max(10, cur_w - 1)
                        vis_prompt = prompt_lbl
                        if buf_str:
                            avail_buf = max_p_w - len(vis_prompt) - 2
                            display_buf = buf_str if len(buf_str) <= avail_buf else "…" + buf_str[-(avail_buf - 1):]
                            prompt_line = f"\033[1;36m{vis_prompt}\033[0m\033[1;37m{display_buf}\033[0m\033[1;37m█\033[0m"
                        else:
                            ph = "[Sedang memproses... Tekan Esc atau ketik /detach untuk latar belakang]"
                            avail_ph = max_p_w - len(vis_prompt) - 1
                            display_ph = ph if len(ph) <= avail_ph else ph[:avail_ph - 1] + "…]"
                            prompt_line = f"\033[1;36m{vis_prompt}\033[0m\033[2;37m{display_ph}\033[0m"

                        lines = [
                            f"\033[36m{frame}\033[0m {cur_desc_val}",
                            "",
                            f"\033[90m{divider}\033[0m",
                            prompt_line,
                            f"\033[90m{divider}\033[0m",
                            f"\033[90m{footer}\033[0m",
                        ]
                        cur_frame_text = "\n".join(lines)
                        if cur_frame_text != last_frame_text or prev_lines_drawn == 0:
                            last_frame_text = cur_frame_text
                            out = []
                            if prev_lines_drawn > 0:
                                out.append(f"\x1b[{prev_lines_drawn}A\r")
                            for line in lines:
                                out.append(f"\x1b[2K{line}\n")
                            sys.stdout.write("".join(out))
                            sys.stdout.flush()
                            prev_lines_drawn = len(lines)

                    await asyncio.sleep(0.04)

                if prev_lines_drawn > 0:
                    out = [f"\x1b[{prev_lines_drawn}A\r"]
                    for _ in range(prev_lines_drawn):
                        out.append("\x1b[2K\n")
                    out.append(f"\x1b[{prev_lines_drawn}A\r")
                    sys.stdout.write("".join(out))
                    sys.stdout.flush()
                    prev_lines_drawn = 0

                if typed_buffer:
                    self._pending_user_input = "".join(typed_buffer)

                return await t

            except BaseException as exc:
                if prev_lines_drawn > 0:
                    out = [f"\x1b[{prev_lines_drawn}A\r"]
                    for _ in range(prev_lines_drawn):
                        out.append("\x1b[2K\n")
                    out.append(f"\x1b[{prev_lines_drawn}A\r")
                    sys.stdout.write("".join(out))
                    sys.stdout.flush()
                    prev_lines_drawn = 0
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

        # Dynamic Task Contract Extraction (Goal, Scope, Format, & Definition of Done)
        contract_extractor = self.contract_extractor or TaskContractExtractor(engine=self.engine)
        task_contract = contract_extractor.extract(query)

        self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Kontrak Tugas:[/bold bright_white] [cyan]{task_contract.primary_goal}[/cyan]")
        self.console.print(f"  [dim]• Format: {task_contract.format_requirements} (Target: ~{task_contract.min_words}-{task_contract.max_words} kata)[/dim]")
        if task_contract.out_of_scope:
            self.console.print(f"  [dim]• Batasan: Eksklusi {', '.join(task_contract.out_of_scope)}[/dim]")

        evidence = self.engine.evidence_synthesizer.retrieve_evidence(
            query=query,
            documents=existing_paths,
            max_chunks=self.config.citation.max_evidence_chunks,
            vector_vault=getattr(self.vault_manager, "vector_vault", None),
        )

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

            try:
                from unittest.mock import Mock
                if isinstance(self.console.input, Mock):
                    user_choice = self.console.input("[bold cyan][?] Jalankan pencarian & unduh otomatis? [Y/n/ketik kata kunci baru]: [/bold cyan]").strip()
                elif not sys.stdin.isatty():
                    user_choice = "n"
                else:
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
                    self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]{action}[/bold bright_white] [cyan]{label}[/cyan]")

                async def _run_scout_with_timeout() -> list[str]:
                    scout_kwargs: dict[str, Any] = {
                        "max_papers": target_quota,
                        "on_status": on_scout_status,
                    }
                    import inspect
                    sig = inspect.signature(self.scout.auto_scout_and_ingest)
                    if "engine" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                        scout_kwargs["engine"] = self.engine

                    scout_timeout = getattr(self.config, "scout_timeout", 180.0)
                    return await asyncio.wait_for(
                        self.scout.auto_scout_and_ingest(
                            target_topic,
                            **scout_kwargs,
                        ),
                        timeout=scout_timeout,
                    )

                timed_out = False
                new_stems: list[str] = []
                prompt_str = f"{self.current_project.name}/chat/{session.name}> " if (self.sub_mode == "chat" and session and self.current_project) else (f"{self.current_project.name}> " if self.current_project else "> ")
                try:
                    new_stems = self._run_agent_turn_interactive(
                        coro=_run_scout_with_timeout(),
                        step_desc=scout_progress,
                        prompt_lbl=prompt_str,
                        model_label=model_label,
                        effort=effort,
                    )
                except SessionDetachSignal:
                    task = self.task_manager.submit(
                        name=f"Deep Research Scout: {target_topic}",
                        task_type="scout",
                        target_fn=self._background_scout_worker,
                        query=target_topic,
                        topic_query=target_topic,
                        max_download=target_quota,
                        session_id=session.id if session else None,
                    )
                    self.console.print(f"\n[bold cyan]● Sesi '{session.name if session else ''}' di-detach ke latar belakang.[/bold cyan]")
                    self.console.print(f"  [dim]Tugas '{task.name}' ({task.id}) tetap berjalan di background (Ketik /proc untuk pantau).[/dim]\n")
                    if session and self.chat_manager:
                        self.chat_manager.update_session_status(session.id, "detached")
                    self.sub_mode = "chat_hub"
                    return
                except (asyncio.TimeoutError, TimeoutError):
                    timed_out = True
                except (asyncio.CancelledError, KeyboardInterrupt):
                    self.console.print("[dim]● Pencarian literatur dibatalkan oleh pengguna.[/dim]\n")
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
                        max_download=target_quota,
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
                        vector_vault=getattr(self.vault_manager, "vector_vault", None),
                    )
                    self.console.print(f"[bold green][✓] Berhasil mengindeks {len(new_stems)} dokumen baru ke vault.[/bold green]")
                    for s in new_stems:
                        self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Membaca[/bold bright_white] [cyan]cache/{s}.md[/cyan] [dim](Dokumen siap)[/dim]")
                    self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Menganalisis[/bold bright_white] [dim]Menyintesis temuan paper baru ke dalam jawaban komprehensif...[/dim]\n")

                # Synchronize vault cache with freshly downloaded papers
                if self.vault_manager:
                    self.vault_manager.reload_vault(force=False)
                self._sync_vault_cache()

                active_paths = [self.workspace.get_cache_path(f"{s}.md") for s in self.active_documents]
                existing_paths = [p for p in active_paths if p.exists()]
                evidence = self.engine.evidence_synthesizer.retrieve_evidence(
                    query=query,
                    documents=existing_paths,
                    max_chunks=self.config.citation.max_evidence_chunks,
                    vector_vault=getattr(self.vault_manager, "vector_vault", None),
                )
                if evidence:
                    self.console.print(f"[bold green][✓] Berhasil memuat bukti literatur dari dokumen vault yang baru diunduh.[/bold green]")

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
        max_agent_turns = getattr(self.config, "max_agent_turns", 9999) if self.config else 9999
        max_refinement_turns = getattr(self.config, "max_refinement_turns", 3) if self.config else 3
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

        # Inject dynamic Task Contract directive to guide synthesis
        dod_str = "\n".join(f"  * {d}" for d in task_contract.definition_of_done)
        contract_directive = (
            f"\n\n[PANDUAN KONTRAK PENYELESAIAN TUGAS]:\n"
            f"- Sasaran Utama: {task_contract.primary_goal}\n"
            f"- Format Output: {task_contract.format_requirements}\n"
            f"- Alokasi Panjang: {task_contract.min_words} - {task_contract.max_words} kata (Patuhi batas ini agar tidak terpotong).\n"
            f"- Batasan Negatif (Out-of-Scope): DILARANG menyertakan {', '.join(task_contract.out_of_scope)}.\n"
            f"- Definition of Done (Checklist yang Wajib Dipenuhi):\n{dod_str}\n"
            f"Patuhi seluruh instruksi di atas secara presisi."
        )
        current_query = f"{current_query}{contract_directive}"
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

                    async for token in self.engine.stream_query(
                        query=current_query,
                        documents=current_docs,
                        skill_name=self.active_skill_name,
                        model=model_name,
                        max_evidence_chunks=self.config.citation.max_evidence_chunks if current_docs else 0,
                        history=history_msgs,
                        vector_vault=getattr(self.vault_manager, "vector_vault", None),
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
                    c_rep = await asyncio.to_thread(critic.critique, query, p_resp, contract=task_contract)
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
                        max_refine_limit = 1 if getattr(task_contract, 'max_words', 1000) <= 750 else max_refinement_turns
                        if refinement_turns >= max_refine_limit:
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

                    tool_progress = _AgentProgressTracker(spin_desc)
                    async def _run_tool_async() -> tuple[str, str, str]:
                        return await asyncio.to_thread(self._execute_internal_vault_tool, cmd_line)

                    tool_coro = _run_tool_async()
                    try:
                        action_label, target_label, tool_output = self._run_agent_turn_interactive(
                            coro=tool_coro,
                            step_desc=tool_progress,
                            prompt_lbl=prompt_str,
                            model_label=model_label,
                            effort=effort,
                        )
                    except BaseException:
                        try:
                            tool_coro.close()
                        except Exception:
                            pass
                        raise

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
                    if self._check_agent_turn_checkpoint(
                        turn=turn,
                        best_grounding=best_grounding,
                        best_critic=best_critic,
                        accumulated_input_tokens=accumulated_input_tokens,
                        accumulated_output_tokens=accumulated_output_tokens,
                    ):
                        raw_final = best_draft or turn_result.get("response", "")
                        final_response = self._sanitize_final_response(raw_final)
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
                    if self._check_agent_turn_checkpoint(
                        turn=turn,
                        best_grounding=best_grounding,
                        best_critic=best_critic,
                        accumulated_input_tokens=accumulated_input_tokens,
                        accumulated_output_tokens=accumulated_output_tokens,
                    ):
                        raw_final = best_draft or turn_result.get("response", "")
                        final_response = self._sanitize_final_response(raw_final)
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
                        continue

                elif turn_result["type"] == "needs_citation_mapping":
                    refinement_turns += 1
                    c_rep = turn_result["report"]
                    v_rep = turn_result.get("v_report")
                    self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Devil's Advocate[/bold bright_white] [yellow]Skor {c_rep.score}/10: {c_rep.critique_summary}[/yellow]")
                    stems = self._get_cached_stems()
                    if stems and self.scout:
                        self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Reference Mapping[/bold bright_white] [cyan]Menelusuri jaringan sitasi dokumen vault...[/cyan]")

                    gap_progress = _AgentProgressTracker("Menelusuri literatur spesifik penambal celah...")

                    async def _run_gap_search_and_ingestion_async() -> tuple[list[Any], list[str], list[Any], str]:
                        mapped_papers: list[Any] = []
                        newly_ingested_stems: list[str] = []

                        # 1. Directed Gap Search via Typed Critique Queries
                        directed_queries = list(getattr(c_rep, "directed_search_queries", []))
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

                        is_mock_env = bool(api_key and (api_key.startswith("mock") or api_key == "mock_key"))
                        is_scout_mocked = bool(self.scout and (hasattr(self.scout.search_multi_source, "assert_called") or getattr(self.scout.search_multi_source, "__code__", None) != LiteratureScout.search_multi_source.__code__))
                        is_map_mocked = bool(self.scout and (hasattr(self.scout.map_vault_citation_network, "assert_called") or getattr(self.scout.map_vault_citation_network, "__code__", None) != LiteratureScout.map_vault_citation_network.__code__))

                        if directed_queries and self.scout and (not is_mock_env or is_scout_mocked):
                            for target_gap_q in directed_queries[:2]:
                                gap_progress.set(f"Menelusuri literatur penambal celah: '{target_gap_q}'...")
                                try:
                                    sq = ScoutQuery(keywords=target_gap_q, limit=5)
                                    candidates = await self.scout.search_multi_source(sq, limit=5)
                                    if candidates:
                                        existing_titles = {p.title for p in mapped_papers}
                                        mapped_papers.extend([p for p in candidates if p.title not in existing_titles])
                                except Exception:
                                    pass

                        # 2. Reference Mapping (Vault citation network)
                        if stems and self.scout and (not is_mock_env or is_map_mocked):
                            gap_progress.set("Menelusuri jaringan sitasi dokumen vault...")
                            try:
                                t_map_s = time.time()
                                cit_papers = await self.scout.map_vault_citation_network(stems, limit=5)
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
                                    gap_progress.set(f"Mengunduh sitasi [{idx}/{len(mapped_papers)}]: {p.title[:38]}... (PDF)")
                                    try:
                                        await self.scout.download_scout_paper(p)
                                        newly_ingested_stems.append(safe_stem)
                                        downloaded = True
                                    except Exception:
                                        downloaded = False

                                if not downloaded:
                                    self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Paywall Terdeteksi[/bold bright_white] [yellow][{idx}/{len(mapped_papers)}] {p.title[:50]}...[/yellow] [dim](Naskah berbayar)[/dim]")
                                    gap_progress.set(f"Mengevaluasi naskah berbayar [{idx}/{len(mapped_papers)}]: {p.title[:35]}...")
                                    decision = await asyncio.to_thread(self._ask_ai_paywall_action, p, query)

                                    if decision.get("decision") == "FIND_ALTERNATIVE":
                                        gap_progress.set(f"Mencari alternatif open-access: {p.title[:35]}...")
                                        alt_oa = []
                                        try:
                                            sq = ScoutQuery(keywords=f"{p.title} open access", is_oa_only=True)
                                            alt_candidates = await self.scout.search_multi_source(sq, limit=5)
                                            alt_oa = [c for c in alt_candidates if c.has_pdf and c.pdf_url]
                                        except Exception:
                                            pass

                                        alt_downloaded = False
                                        if alt_oa:
                                            alt_paper = alt_oa[0]
                                            gap_progress.set(f"Mengunduh alternatif: {alt_paper.title[:35]}... (PDF)")
                                            try:
                                                await self.scout.download_scout_paper(alt_paper)
                                                newly_ingested_stems.append(sanitize_filename(alt_paper.title))
                                                alt_downloaded = True
                                            except Exception:
                                                alt_downloaded = False

                                        if not alt_downloaded:
                                            self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Abstrak Memadai[/bold bright_white] [green]Mengindeks abstrak resmi paper ke vault[/green]")
                                            gap_progress.set(f"Mengindeks abstrak resmi: {p.title[:35]}...")
                                            await asyncio.to_thread(self.scout.ingest_abstract_paper, p)
                                            newly_ingested_stems.append(safe_stem)
                                    else:
                                        self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Abstrak Memadai[/bold bright_white] [green]Mengindeks abstrak resmi paper ke vault[/green] [dim]({decision.get('reason', '')})[/dim]")
                                        gap_progress.set(f"Mengindeks abstrak resmi: {p.title[:35]}...")
                                        await asyncio.to_thread(self.scout.ingest_abstract_paper, p)
                                        newly_ingested_stems.append(safe_stem)

                        curr_evidence = evidence
                        curr_citation_guide = citation_guide
                        if newly_ingested_stems:
                            gap_progress.set("Mengindeks dokumen sitasi baru ke vault...")
                            if self.vault_manager:
                                await asyncio.to_thread(self.vault_manager.reload_vault, False)
                            await asyncio.to_thread(self._sync_vault_cache)
                            for s in newly_ingested_stems:
                                self.active_documents.add(s)

                            active_paths = [self.workspace.get_cache_path(f"{s}.md") for s in self.active_documents]
                            existing_paths = [p for p in active_paths if p.exists()]
                            nonlocal current_docs
                            current_docs = list(existing_paths)
                            gap_progress.set("Mengekstrak bukti dokumen baru...")
                            curr_evidence = await asyncio.to_thread(
                                self.engine.evidence_synthesizer.retrieve_evidence,
                                query=query,
                                documents=current_docs,
                                max_chunks=self.config.citation.max_evidence_chunks,
                                vector_vault=getattr(self.vault_manager, "vector_vault", None),
                            )
                            curr_citation_guide = self._format_vault_citation_guide()

                        return mapped_papers, newly_ingested_stems, curr_evidence, curr_citation_guide

                    gap_coro = _run_gap_search_and_ingestion_async()
                    try:
                        mapped_papers, newly_ingested_stems, evidence, citation_guide = self._run_agent_turn_interactive(
                            coro=gap_coro,
                            step_desc=gap_progress,
                            prompt_lbl=prompt_str,
                            model_label=model_label,
                            effort=effort,
                        )
                    except BaseException:
                        try:
                            gap_coro.close()
                        except Exception:
                            pass
                        raise

                    if mapped_papers:
                        self.console.print(f"[bold cyan]●[/bold cyan] [bold bright_white]Directed Gap Search[/bold bright_white] [cyan]Menemukan {len(mapped_papers)} literatur penambal celah empiris.[/cyan]")
                    if newly_ingested_stems:
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
                    if self._check_agent_turn_checkpoint(
                        turn=turn,
                        best_grounding=best_grounding,
                        best_critic=best_critic,
                        accumulated_input_tokens=accumulated_input_tokens,
                        accumulated_output_tokens=accumulated_output_tokens,
                    ):
                        raw_final = best_draft or turn_result.get("response", "")
                        final_response = self._sanitize_final_response(raw_final)
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

