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


class StudioCommandsMixin:
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
        elif cmd in ("/processing",):
            self.render_mode_divider()
            self.sub_mode = "processing"
            if args:
                self.handle_processing_interaction(args)
            else:
                self.render_processing_summary()
        elif cmd in ("/process", "/proc", "/p", "/tasks", "/task"):
            if args:
                self.handle_processing_interaction(args)
            else:
                self.run_interactive_processing_view()
        elif cmd in ("/chats", "/sessions", "/session"):
            if args:
                self.handle_chat_hub_interaction(args)
            else:
                self.run_interactive_sessions_list()
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
        """Inspect paper, table, or figure image in the vault."""
        if not arg:
            self.console.print("[yellow]Usage: /view <# | stem> atau /view table <#> atau /view image <#>[/yellow]\n")
            return

        clean = arg.strip()
        parts = clean.split(maxsplit=1)
        sub = parts[0].lower()
        sub_arg = parts[1].strip() if len(parts) > 1 else ""

        if sub in ("table", "tables", "t"):
            self._cmd_view_table(sub_arg)
            return
        elif sub in ("image", "images", "img", "i"):
            self._cmd_view_image(sub_arg)
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
                import scrpcy.cli_studio as _cli_studio
                selector_fn = getattr(_cli_studio, 'interactive_model_selector', interactive_model_selector)
                chosen = selector_fn(
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

