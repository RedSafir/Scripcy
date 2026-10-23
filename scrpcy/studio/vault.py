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


class StudioVaultMixin:
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
        tables_count = len(self.vault_manager.list_tables())
        images_count = len(self.vault_manager.list_images())

        v_table.add_row("Status", "[bold cyan]Aktif[/bold cyan]")
        v_table.add_row("Folder Vault", str(vault_path))
        v_table.add_row("Total Berkas", f"{len(active_items)} berkas ({cached_count} terindeks Markdown, {needs_reload} perlu reload)")
        v_table.add_row("Koleksi Paper", f"{len(active_items)} berkas di vault/papers/ ({cached_count} terindeks Markdown, {needs_reload} perlu reload)")
        v_table.add_row("Tabel Terstruktur", f"{tables_count} tabel terindeks di vault/tables/")
        v_table.add_row("Gambar & Grafik", f"{images_count} visual diagram di vault/images/")
        v_table.add_row("Diarsipkan", f"{archived_count} berkas di .archive/ (tidak aktif di konteks AI)")
        v_table.add_row("Format Dokumen", "[dim]PDF, DOCX, PPTX, XLSX, TXT, HTML, MD (didukung Microsoft MarkItDown)[/dim]")
        self.console.print(v_table)
        self.console.print()

    def render_vault_view(self) -> None:
        """Render active vault items (papers, tables, images) following DESIGN.md."""
        if not self.vault_manager:
            return
        items = self.vault_manager.list_active_files()
        tables = self.vault_manager.list_tables()
        images = self.vault_manager.list_images()

        if not items and not tables and not images:
            self.console.print("\n[dim]Vault masih kosong (0 paper, 0 tabel, 0 gambar).[/dim]")
            self.console.print("[dim]Gunakan 'add <path>', 'findpaper <topik>', atau 'gdrive' untuk menambahkan berkas.[/dim]\n")
            return

        self.console.print(f"\n[bold bright_white]Koleksi Paper Vault [{self.current_project.name if self.current_project else 'Vault'}][/bold bright_white] [dim](vault/papers/ • {len(items)} paper)[/dim]")
        p_table = Table(
            box=None,
            padding=(0, 2),
            show_header=True,
            header_style="bold cyan",
        )
        p_table.add_column("#", justify="right", style="dim", width=3)
        p_table.add_column("Filename", style="bold white")
        p_table.add_column("Format", justify="center", width=8, style="dim")
        p_table.add_column("Size", justify="right", width=10, style="dim")
        p_table.add_column("Markdown Status", justify="center", width=18)
        p_table.add_column("Cite Key / Title", style="cyan")

        for idx, item in enumerate(items, 1):
            md_status = "[bold cyan][Ready / Cached][/bold cyan]" if item.is_cached else "[dim][Needs Reload][/dim]"
            p_table.add_row(
                str(idx),
                item.filename,
                item.extension.upper().lstrip("."),
                item.readable_size,
                md_status,
                item.title or item.cite_key,
            )
        self.console.print(p_table)

        if tables:
            self.console.print(f"\n[bold bright_white]Tabel Terstruktur Akademis[/bold bright_white] [dim](vault/tables/ • {len(tables)} tabel)[/dim]")
            t_table = Table(
                box=None,
                padding=(0, 2),
                show_header=True,
                header_style="bold cyan",
            )
            t_table.add_column("#", justify="right", style="dim", width=3)
            t_table.add_column("Paper Sumber", style="bold white")
            t_table.add_column("Halaman", justify="center", width=8, style="dim")
            t_table.add_column("Format", justify="center", width=10, style="dim")
            t_table.add_column("Metode", justify="center", width=18)
            t_table.add_column("Keterangan", style="cyan")

            for t_idx, tab in enumerate(tables, 1):
                method_badge = "[cyan]Multimodal Vision[/cyan]" if tab.get("is_vision_fallback") else "[bold cyan]PyMuPDF Tabular[/bold cyan]"
                fmt = "MD + CSV" if tab.get("csv_path") else "Markdown"
                caption = (tab.get("caption") or "-")[:50]
                t_table.add_row(
                    str(t_idx),
                    tab.get("paper_id", "")[:28],
                    str(tab.get("page_num", 1)),
                    fmt,
                    method_badge,
                    caption,
                )
            self.console.print(t_table)

        if images:
            self.console.print(f"\n[bold bright_white]Gambar & Visual Diagram Akademis[/bold bright_white] [dim](vault/images/ • {len(images)} gambar informatif)[/dim]")
            i_table = Table(
                box=None,
                padding=(0, 2),
                show_header=True,
                header_style="bold cyan",
            )
            i_table.add_column("#", justify="right", style="dim", width=3)
            i_table.add_column("File Gambar", style="bold white")
            i_table.add_column("Paper Sumber", style="dim", width=25)
            i_table.add_column("Dimensi", justify="center", width=12, style="dim")
            i_table.add_column("Analisis Multimodal AI", style="cyan")

            for i_idx, img in enumerate(images, 1):
                f_name = Path(img.get("file_path", "")).name
                dim = f"{img.get('width', 0)}x{img.get('height', 0)}"
                analysis = (img.get("ai_analysis") or img.get("caption") or "Diagram informatif")[:55]
                i_table.add_row(
                    str(i_idx),
                    f_name,
                    img.get("paper_id", "")[:23],
                    dim,
                    analysis,
                )
            self.console.print(i_table)

        archived_count = len(self.vault_manager.list_archived_files())
        archive_hint = f" ({archived_count} files in .archive/)" if archived_count else ""
        self.console.print(
            f"\n[dim]Ketik 'view <#>' untuk baca paper, 'view table <#>' untuk buka tabel, 'remove <#>' untuk arsip, atau 'back' untuk kembali{archive_hint}.[/dim]\n"
        )

    def _cmd_view_table(self, arg: str) -> None:
        """View extracted table markdown or CSV."""
        if not self.vault_manager:
            return
        tables = self.vault_manager.list_tables()
        if not tables:
            self.console.print("[dim]Belum ada tabel terindeks di vault/tables/.[/dim]\n")
            return
        if not arg or not arg.isdigit():
            self.console.print(f"[yellow]Usage: /view table <1..{len(tables)}>[/yellow]\n")
            return
        idx = int(arg) - 1
        if not (0 <= idx < len(tables)):
            self.console.print(f"[red][x] Tabel #{arg} tidak ditemukan (total: {len(tables)}).[/red]\n")
            return
        tab = tables[idx]
        md_p = Path(tab.get("md_path", ""))
        self.console.print(f"\n[bold bright_white]Tabel #{arg} [{tab.get('paper_id')} - Halaman {tab.get('page_num')}][/bold bright_white]")
        if tab.get("caption"):
            self.console.print(f"[dim]Keterangan: {tab.get('caption')}[/dim]")
        if md_p.exists():
            content = md_p.read_text(encoding="utf-8")
            self.console.print(f"\n{content}\n")
        else:
            self.console.print(f"[dim]Berkas tabel: {md_p}[/dim]\n")

    def _cmd_view_image(self, arg: str) -> None:
        """View extracted informative figure metadata and multimodal AI analysis."""
        if not self.vault_manager:
            return
        images = self.vault_manager.list_images()
        if not images:
            self.console.print("[dim]Belum ada gambar/diagram terindeks di vault/images/.[/dim]\n")
            return
        if not arg or not arg.isdigit():
            self.console.print(f"[yellow]Usage: /view image <1..{len(images)}>[/yellow]\n")
            return
        idx = int(arg) - 1
        if not (0 <= idx < len(images)):
            self.console.print(f"[red][x] Gambar #{arg} tidak ditemukan (total: {len(images)}).[/red]\n")
            return
        img = images[idx]
        img_p = Path(img.get("file_path", ""))
        self.console.print(f"\n[bold bright_white]Gambar #{arg} [{img_p.name}][/bold bright_white]")
        self.console.print(f"[dim]Paper Sumber : {img.get('paper_id')} (Halaman {img.get('page_num')})[/dim]")
        self.console.print(f"[dim]Dimensi      : {img.get('width')} x {img.get('height')} px[/dim]")
        self.console.print(f"[dim]File Path    : {img_p}[/dim]")
        if img.get("caption"):
            self.console.print(f"[dim]Caption Teks : {img.get('caption')}[/dim]")
        if img.get("ai_analysis"):
            self.console.print(f"\n[bold cyan]Interpretasi Visual Multimodal AI:[/bold cyan]\n{img.get('ai_analysis')}\n")

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

