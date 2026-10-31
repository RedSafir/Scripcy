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
from scripcy.core.graph_rag import GraphRAGBuilder
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

logger = logging.getLogger(__name__)

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


class StudioVaultMixin:
    def run_interactive_vault(self) -> None:
        """Top-level vault entry: simple 3-option menu for Downloads, User Input, and Back."""
        while self.sub_mode == "vault" and self.current_project:
            options = [
                ("downloads", "Downloads"),
                ("user_input", "User Input"),
                ("back", "Kembali ke Proyek"),
            ]
            choice = interactive_menu(
                options=options,
                header=Text(f"Pilih Repositori Vault [{self.current_project.name}]:", style="bold bright_white"),
                console=self.console,
                footer_hint="↑/↓ Navigasi · enter Pilih · 1-3 Lompat cepat · esc Kembali",
                allow_back=True,
            )

            if choice in ("back", None):
                return
            elif choice == "downloads":
                self.run_interactive_vault_hub(scope="downloads")
            elif choice == "user_input":
                self.run_interactive_vault_hub(scope="user")

    def run_interactive_vault_hub(self, scope: str = "downloads") -> None:
        """Vault resources interactive overview for the chosen tier (downloads or user)."""
        scope_title = "Downloads" if scope == "downloads" else "User Input"
        while self.sub_mode == "vault" and self.current_project:
            # If user_input, auto-scan USER INPUT HERE first
            if scope == "user" and self.vault_manager:
                try:
                    staging = self.vault_manager.scan_local_staging()
                    for item in staging:
                        if not item.get("is_duplicate"):
                            self.vault_manager.process_local_item(item, action="keep_both")
                except Exception:
                    pass

            health = self.vault_manager.check_vault_health(category=scope) if self.vault_manager else {
                "status": "active",
                "status_display": "[bold green]Aktif[/bold green]",
                "total_files": 0,
                "warning_details": [],
                "error_details": [],
                "category_health": {},
            }

            header_table = Table(box=None, padding=(0, 2), show_header=False)
            header_table.add_column("Label", style="dim", width=18)
            header_table.add_column("Value", style="white")

            header_table.add_row("Status", health["status_display"])
            header_table.add_row("Total Berkas", str(health["total_files"]))
            header_table.add_row("Terakhir Update", str(health.get("last_updated", "-")))

            if health.get("warning_details"):
                warn_text = "\n".join(health["warning_details"])
                header_table.add_row("Peringatan", f"[yellow]{warn_text}[/yellow]")
            if health.get("error_details"):
                err_text = "\n".join(health["error_details"])
                header_table.add_row("Error", f"[red]{err_text}[/red]")

            options = [
                ("list_berkas", f"List Berkas ({health['total_files']} berkas)"),
                ("graph", "Lihat Graph Knowledge Network (GraphRAG)"),
                ("history", "Riwayat Vault (Log Aktivitas & Sinkronisasi)"),
                ("add", "Tambah Berkas"),
                ("update", "Update Berkas"),
                ("back", "Kembali ke Menu Vault"),
            ]

            choice = interactive_menu(
                options=options,
                header=header_table,
                console=self.console,
                footer_hint="↑/↓ Navigasi · enter Pilih · 1-6 Lompat cepat · esc Kembali",
                allow_back=True,
            )

            if choice in ("back", None):
                return

            if choice == "list_berkas":
                if scope == "downloads":
                    self.run_interactive_vault_categories(scope="downloads")
                else:
                    self.run_interactive_vault_view(category="user")
            elif choice == "graph":
                self.cmd_vault_graph()
            elif choice == "history":
                self.run_interactive_vault_history()
            elif choice == "add":
                self.run_interactive_vault_add()
            elif choice == "update":
                self.cmd_vault_update()

    def run_interactive_vault_categories(self, scope: str = "downloads") -> None:
        """Browse categorized vault sections with category-specific issue indicators."""
        while self.sub_mode == "vault" and self.current_project:
            health = self.vault_manager.check_vault_health(category=scope) if self.vault_manager else {
                "total_files": 0,
                "category_health": {},
            }

            cat_health = health.get("category_health", {})
            papers_h = cat_health.get("papers", {"total": 0, "warning": 0, "error": 0})
            web_h = cat_health.get("web", {"total": 0, "warning": 0, "error": 0})
            tables_h = cat_health.get("tables", {"total": 0, "warning": 0, "error": 0})
            images_h = cat_health.get("images", {"total": 0, "warning": 0, "error": 0})

            def _build_cat_label(name: str, count: int, unit: str, warn: int, err: int) -> str:
                lbl = f"{name:<38} • {count} {unit}"
                sub_lines = []
                if warn > 0:
                    sub_lines.append(f"    [yellow]peringatan[/yellow]                               {warn} berkas")
                if err > 0:
                    sub_lines.append(f"    [red]error[/red]                                    {err} berkas")
                if sub_lines:
                    lbl += "\n" + "\n".join(sub_lines)
                return lbl

            options = [
                ("papers", _build_cat_label("Downloads (Naskah & Jurnal Ilmiah)", papers_h["total"], "berkas", papers_h["warning"], papers_h["error"])),
                ("web", _build_cat_label("Web (Arsip & Portal Resmi)", web_h["total"], "berkas", web_h["warning"], web_h["error"])),
                ("tables", _build_cat_label("Tables (Tabel Data Terstruktur)", tables_h["total"], "tabel", tables_h["warning"], tables_h["error"])),
                ("images", _build_cat_label("Images (Visual, Grafik & Diagram)", images_h["total"], "gambar", images_h["warning"], images_h["error"])),
                ("graph", "Graph Knowledge Network (Visualisasi Relasi Antar-Dokumen)"),
                ("search", f"Cari Berkas (Pencarian Cepat Seluruh Pustaka) • {health['total_files']} berkas"),
                ("back", "Kembali ke Menu Vault"),
            ]

            header = Text(f"Kategori Berkas Downloads [{self.current_project.name}]:", style="bold bright_white")

            choice = interactive_menu(
                options=options,
                header=header,
                console=self.console,
                footer_hint="↑/↓ Navigasi · enter Pilih · 1-7 Lompat cepat · esc Kembali",
                allow_back=True,
            )

            if choice in ("back", None):
                return

            if choice == "graph":
                self.cmd_vault_graph()
            elif choice == "search":
                self.run_interactive_vault_search()
            else:
                self.run_interactive_vault_view(category=choice)

    def run_interactive_vault_view(self, category: str = "papers") -> None:
        """Browse vault items table carousel per DESIGN.md."""
        if category == "tables":
            while True:
                tables = self.vault_manager.list_tables() if self.vault_manager else []
                if not tables:
                    self.console.print("\n[dim]Belum ada tabel terstruktur di vault/tables/.\n[/dim]")
                    time.sleep(1.2)
                    return

                columns = [
                    {"name": "#", "justify": "right", "width": 3, "style": "dim"},
                    {"name": "Nama Tabel", "style": "bold white", "width": 30},
                    {"name": "Naskah Asal", "style": "cyan", "width": 24},
                    {"name": "Baris", "justify": "right", "width": 8, "style": "dim"},
                    {"name": "Kolom", "justify": "right", "width": 8, "style": "dim"},
                    {"name": "Status", "justify": "center", "width": 12},
                ]

                def _table_row_builder(idx: int, item: Any, is_selected: bool) -> list[Any]:
                    arrow = "> " if is_selected else "  "
                    t_name = Text()
                    t_name.append(arrow, style="bold cyan" if is_selected else "dim")
                    tbl_name = str(item.get("id") or Path(item.get("csv_path", "")).stem or item.get("table_id") or item.get("name") or "tabel")
                    t_name.append(tbl_name[:26], style="bold bright_white" if is_selected else "white")
                    return [
                        str(idx + 1),
                        t_name,
                        str(item.get("paper_id") or "")[:22],
                        str(item.get("rows") or "-"),
                        str(item.get("cols") or "-"),
                        "[bold green][Siap][/bold green]",
                    ]

                header = Text(f"Tabel Terstruktur [{self.current_project.name}] • {len(tables)} tabel", style="bold bright_white")
                actions = [
                    ("view", "Lihat Tabel", "enter"),
                    ("detail", "Detail", "d"),
                    ("back", "Kembali", "esc"),
                ]
                action, idx, selected_item = interactive_item_browser(
                    title=header,
                    columns=columns,
                    items=tables,
                    row_builder=_table_row_builder,
                    actions=actions,
                    console=self.console,
                    empty_message="Tidak ada tabel.",
                )
                if action in ("back", "esc") or not selected_item:
                    return
                elif action in ("view", "detail") and selected_item:
                    content = ""
                    md_path = selected_item.get("md_path")
                    csv_path = selected_item.get("csv_path")
                    if md_path and Path(md_path).exists():
                        try:
                            content = Path(md_path).read_text(encoding="utf-8")
                        except Exception:
                            pass
                    elif csv_path and Path(csv_path).exists():
                        try:
                            content = Path(csv_path).read_text(encoding="utf-8")
                        except Exception:
                            pass
                    if not content:
                        content = selected_item.get("markdown") or selected_item.get("csv") or json.dumps(selected_item, indent=2)

                    tbl_stem = str(selected_item.get("id") or selected_item.get("table_id") or "tabel")
                    paper_pid = str(selected_item.get("paper_id") or "")
                    details_header = [
                        f"ID Tabel        : {tbl_stem}",
                        f"Naskah Asal     : {paper_pid}",
                        f"Lokasi CSV      : {csv_path or f'vault/tables/{paper_pid}/'}",
                        f"Lokasi Markdown : {md_path or f'vault/tables/{paper_pid}/'}",
                        f"Keterangan      : {selected_item.get('caption') or '-'}",
                        "",
                        "--- Pratinjau Konten Tabel ---",
                    ]
                    full_lines = details_header + content.splitlines()
                    interactive_detail_viewer(
                        title=f"Pratinjau Tabel: {tbl_stem}",
                        content_lines=full_lines,
                        actions=[("back", "Kembali", "esc")],
                        console=self.console,
                    )
            return

        if category == "images":
            while True:
                images = self.vault_manager.list_images() if self.vault_manager else []
                if not images:
                    self.console.print("\n[dim]Belum ada gambar / diagram visual di vault/images/.\n[/dim]")
                    time.sleep(1.2)
                    return

                columns = [
                    {"name": "#", "justify": "right", "width": 3, "style": "dim"},
                    {"name": "Nama Berkas / ID", "style": "bold white", "width": 28},
                    {"name": "Naskah Asal", "style": "cyan", "width": 22},
                    {"name": "Dimensi", "justify": "center", "width": 12, "style": "dim"},
                    {"name": "Keterangan / Caption", "style": "white", "width": 34},
                ]

                def _img_row_builder(idx: int, item: Any, is_selected: bool) -> list[Any]:
                    arrow = "> " if is_selected else "  "
                    f_name = Text()
                    f_name.append(arrow, style="bold cyan" if is_selected else "dim")
                    img_file = Path(item.get("file_path", "")).name if item.get("file_path") else ""
                    img_id = str(item.get("id") or item.get("image_id") or img_file or "gambar")
                    f_name.append(img_id[:24], style="bold bright_white" if is_selected else "white")

                    dim = f"{item.get('width', 0)}x{item.get('height', 0)}" if item.get("width") else "-"

                    return [
                        str(idx + 1),
                        f_name,
                        str(item.get("paper_id") or "")[:20],
                        dim,
                        str(item.get("caption") or item.get("description") or "-")[:32],
                    ]

                header = Text(f"Gambar & Diagram [{self.current_project.name}] • {len(images)} gambar", style="bold bright_white")
                actions = [
                    ("view", "Lihat Detail", "enter"),
                    ("detail", "Detail", "d"),
                    ("back", "Kembali", "esc"),
                ]
                action, idx, selected_item = interactive_item_browser(
                    title=header,
                    columns=columns,
                    items=images,
                    row_builder=_img_row_builder,
                    actions=actions,
                    console=self.console,
                    empty_message="Tidak ada gambar.",
                )
                if action in ("back", "esc") or not selected_item:
                    return
                elif action in ("view", "detail") and selected_item:
                    f_path = selected_item.get("file_path") or selected_item.get("path") or ""
                    img_id = selected_item.get("id") or selected_item.get("image_id") or Path(f_path).stem
                    paper_pid = selected_item.get("paper_id") or ""
                    dim = f"{selected_item.get('width', 0)}x{selected_item.get('height', 0)}" if selected_item.get("width") else "-"
                    lines = [
                        f"ID Gambar       : {img_id}",
                        f"Naskah Asal     : {paper_pid}",
                        f"Jalur Berkas    : {f_path or f'vault/images/{paper_pid}/'}",
                        f"Dimensi         : {dim}",
                        "",
                        "Caption / Keterangan:",
                        selected_item.get("caption") or selected_item.get("description") or "(Tidak ada keterangan)",
                        "",
                        "Analisis Visual AI:",
                        selected_item.get("ai_analysis") or "(Belum ada analisis visual)",
                    ]
                    interactive_detail_viewer(
                        title=f"Detail Visual: {img_id}",
                        content_lines=lines,
                        actions=[("back", "Kembali", "esc")],
                        console=self.console,
                    )
            return

        cat_title = "User Input" if category in ("user", "user_input") else ("Downloads" if category in ("papers", "downloads") else ("Web" if category == "web" else "Semua Berkas"))
        while True:
            active_items = self.vault_manager.list_active_files(category=category if category != "all" else None) if self.vault_manager else []
            if not active_items:
                self.console.print(f"\n[dim]Kategori {cat_title} masih kosong (0 berkas aktif).\n[/dim]")
                time.sleep(1.2)
                return

            columns = [
                {"name": "#", "justify": "right", "width": 3, "style": "dim"},
                {"name": "Judul", "style": "bold white", "width": 38},
                {"name": "Konteks", "justify": "center", "width": 14},
                {"name": "Status", "justify": "center", "width": 16},
                {"name": "Tabel", "justify": "right", "width": 7, "style": "dim"},
                {"name": "Visual", "justify": "right", "width": 7, "style": "dim"},
            ]

            def _row_builder(idx: int, item: Any, is_selected: bool) -> list[Any]:
                is_active = item.stem in self.active_documents
                ctx_badge = "[bold cyan][Aktif][/bold cyan]" if is_active else "[dim][Nonaktif][/dim]"

                if item.has_error:
                    status_badge = "[bold red][Error][/bold red]"
                    item_style = "bold red"
                elif getattr(item, "is_duplicate", False):
                    status_badge = "[bold yellow][Duplikat][/bold yellow]"
                    item_style = "bold yellow"
                elif not item.is_cached:
                    status_badge = "[bold cyan][Perlu Update][/bold cyan]"
                    item_style = "bold bright_white" if is_selected else "white"
                else:
                    status_badge = "[bold green][Siap][/bold green]"
                    item_style = "bold bright_white" if is_selected else "white"

                arrow = "> " if is_selected else "  "
                title_text = Text()
                title_text.append(arrow, style="bold cyan" if is_selected else "dim")
                title_text.append((item.title or item.stem)[:36], style=item_style)

                return [
                    str(idx + 1),
                    title_text,
                    ctx_badge,
                    status_badge,
                    str(item.tables_count),
                    str(item.visuals_count),
                ]

            header = Text()
            header.append(f"Daftar Berkas {cat_title} [{self.current_project.name}]", style="bold bright_white")
            header.append(f" • {len(active_items)} berkas", style="dim")

            actions = [
                ("toggle", "Ganti Aktif", "enter"),
                ("detail", "Detail Metadata", "d"),
                ("archive", "Arsipkan", "x"),
                ("restore", "Pulihkan", "u"),
                ("refresh", "Muat Ulang", "f"),
                ("back", "Kembali", "esc"),
            ]

            action, idx, selected_item = interactive_item_browser(
                title=header,
                columns=columns,
                items=active_items,
                row_builder=_row_builder,
                actions=actions,
                console=self.console,
                empty_message=f"Kategori {cat_title} kosong.",
            )

            if action in ("back", "esc"):
                return
            elif action == "toggle" and selected_item:
                if selected_item.stem in self.active_documents:
                    self.active_documents.remove(selected_item.stem)
                else:
                    self.active_documents.add(selected_item.stem)
            elif action == "detail" and selected_item:
                paper_stem = selected_item.stem
                tbl_folder = f"vault/tables/{paper_stem}/"
                img_folder = f"vault/images/{paper_stem}/"

                try:
                    rel_orig = selected_item.path.relative_to(self.workspace.root_dir)
                    orig_loc = str(rel_orig).replace("\\", "/")
                except Exception:
                    orig_loc = str(selected_item.path).replace("\\", "/")

                cache_file = self.workspace.get_cache_path(f"{selected_item.stem}.md") if self.workspace else None
                if cache_file and cache_file.exists():
                    try:
                        rel_cache = cache_file.relative_to(self.workspace.root_dir)
                        md_loc = str(rel_cache).replace("\\", "/")
                    except Exception:
                        md_loc = str(cache_file).replace("\\", "/")
                    md_display = f"{md_loc} (Terindeks)"
                else:
                    md_loc = f"cache/{selected_item.stem}.md"
                    md_display = f"{md_loc} (Belum terindeks)"

                lines = [
                    f"Judul             : {selected_item.title or selected_item.stem}",
                    f"Penulis           : {', '.join(selected_item.authors) if selected_item.authors else '-'}",
                    f"Tahun             : {selected_item.year or '-'}",
                    f"Cite Key          : {selected_item.cite_key or '-'}",
                    f"DOI / URL         : {selected_item.doi or selected_item.url or '-'}",
                    f"Kategori          : {'Papers (Naskah & Jurnal Ilmiah)' if selected_item.category == 'papers' else 'Web (Arsip & Portal Resmi)'}",
                    f"Nama Berkas       : {selected_item.filename}",
                    f"Ukuran            : {selected_item.readable_size} ({selected_item.size_bytes:,} bytes)",
                    f"Terakhir Update   : {selected_item.updated_at or '-'}",
                    f"Lokasi Berkas Asli: {orig_loc}",
                    f"Lokasi Berkas MD  : {md_display}",
                    f"Tabel Terindeks   : {selected_item.tables_count} tabel ({tbl_folder})",
                    f"Visual Terindeks  : {selected_item.visuals_count} gambar / diagram ({img_folder})",
                ]
                interactive_detail_viewer(
                    title=f"Metadata Berkas: {selected_item.filename}",
                    content_lines=lines,
                    actions=[("back", "Kembali", "esc")],
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
                    self.console.print(f"[bold green][✓] Berkas '{selected_item.filename}' berhasil diarsipkan ke .archive/.[/bold green]\n")

            elif action == "restore":
                archived = self.vault_manager.list_archived_files()
                if not archived:
                    self.console.print("\n[dim]Tidak ada berkas di .archive/.\n[/dim]")
                    time.sleep(1.0)
                else:
                    arch_opts = [(f.filename, f.filename) for f in archived] + [("back", "Kembali")]
                    ch = interactive_menu(arch_opts, header="Pilih berkas untuk dipulihkan:", console=self.console)
                    if ch and ch != "back":
                        self.vault_manager.restore_file(ch)
                        self.active_documents.add(Path(ch).stem)
                        self._sync_vault_cache()

            elif action == "refresh":
                self.cmd_vault_update()

    def run_interactive_vault_search(self) -> None:
        """Search and browse all vault items across all categories with interactive filter."""
        all_items = self.vault_manager.list_active_files() if self.vault_manager else []
        if not all_items:
            self.console.print("\n[dim]Belum ada berkas di dalam Vault untuk dicari.\n[/dim]")
            time.sleep(1.2)
            return

        query = ""
        current_category = "all"
        while self.sub_mode == "vault" and self.current_project:
            # Filter items by category and query
            filtered = all_items
            if current_category != "all":
                filtered = [it for it in filtered if it.category == current_category]
            if query.strip():
                q_low = query.strip().lower()
                filtered = [
                    it for it in filtered
                    if q_low in it.title.lower()
                    or q_low in it.stem.lower()
                    or q_low in it.filename.lower()
                    or q_low in it.cite_key.lower()
                    or any(q_low in a.lower() for a in it.authors)
                ]

            columns = [
                {"name": "#", "justify": "right", "width": 3, "style": "dim"},
                {"name": "Judul", "style": "bold white", "width": 36},
                {"name": "Kategori", "justify": "center", "width": 10, "style": "cyan"},
                {"name": "Konteks", "justify": "center", "width": 12},
                {"name": "Status", "justify": "center", "width": 14},
                {"name": "Tabel", "justify": "right", "width": 7, "style": "dim"},
                {"name": "Visual", "justify": "right", "width": 7, "style": "dim"},
            ]

            def _search_row_builder(idx: int, item: Any, is_selected: bool) -> list[Any]:
                is_active = item.stem in self.active_documents
                ctx_badge = "[bold cyan][Aktif][/bold cyan]" if is_active else "[dim][Nonaktif][/dim]"

                if item.has_error:
                    status_badge = "[bold red][Error][/bold red]"
                    item_style = "bold red"
                elif getattr(item, "is_duplicate", False):
                    status_badge = "[bold yellow][Duplikat][/bold yellow]"
                    item_style = "bold yellow"
                elif not item.is_cached:
                    status_badge = "[bold cyan][Perlu Update][/bold cyan]"
                    item_style = "bold bright_white" if is_selected else "white"
                else:
                    status_badge = "[bold green][Siap][/bold green]"
                    item_style = "bold bright_white" if is_selected else "white"

                arrow = "> " if is_selected else "  "
                title_text = Text()
                title_text.append(arrow, style="bold cyan" if is_selected else "dim")
                title_text.append((item.title or item.stem)[:34], style=item_style)

                cat_badge = item.category.title()

                return [
                    str(idx + 1),
                    title_text,
                    cat_badge,
                    ctx_badge,
                    status_badge,
                    str(item.tables_count),
                    str(item.visuals_count),
                ]

            cat_pills = []
            categories = [("all", "Semua"), ("papers", "Papers"), ("web", "Web"), ("tables", "Tables"), ("images", "Images")]
            for cat_key, cat_name in categories:
                if current_category == cat_key:
                    cat_pills.append(f"[bold cyan][● {cat_name}][/bold cyan]")
                else:
                    cat_pills.append(f"[dim][○ {cat_name}][/dim]")
            pills_str = "  ".join(cat_pills)

            header = Table(box=None, padding=(0, 0), show_header=False)
            header.add_column("Col")
            header.add_row(Text(f"Cari Berkas Vault [{self.current_project.name}] • {len(filtered)} berkas", style="bold bright_white"))
            header.add_row(Text.from_markup(f"Filter Kategori : {pills_str}"))
            header.add_row(Text.from_markup(f"Kata Kunci      : [bold white]{query}[/bold white][bold cyan]_[/bold cyan]"))

            actions = [
                ("toggle", "Ganti Aktif", "enter"),
                ("detail", "Detail Metadata", "d"),
                ("search", "Ubah Kata Kunci", "s"),
                ("filter", "Ganti Filter", "f"),
                ("archive", "Arsipkan", "x"),
                ("back", "Kembali", "esc"),
            ]

            action, idx, selected_item = interactive_item_browser(
                title=header,
                columns=columns,
                items=filtered,
                row_builder=_search_row_builder,
                actions=actions,
                console=self.console,
                empty_message=f"Tidak ada berkas yang cocok dengan filter / kata kunci '{query}'.",
            )

            if action in ("back", "esc"):
                return
            elif action == "filter":
                cat_keys = ["all", "papers", "web", "tables", "images"]
                curr_idx = cat_keys.index(current_category) if current_category in cat_keys else 0
                current_category = cat_keys[(curr_idx + 1) % len(cat_keys)]
            elif action == "search":
                new_q = interactive_input_prompt(
                    console=self.console,
                    prompt_label="Cari > ",
                    initial_buffer=query,
                    default=query,
                )
                query = (new_q or "").strip()
            elif action == "toggle" and selected_item:
                if selected_item.stem in self.active_documents:
                    self.active_documents.remove(selected_item.stem)
                else:
                    self.active_documents.add(selected_item.stem)
            elif action == "detail" and selected_item:
                paper_stem = selected_item.stem
                tbl_folder = f"vault/tables/{paper_stem}/"
                img_folder = f"vault/images/{paper_stem}/"

                try:
                    rel_orig = selected_item.path.relative_to(self.workspace.root_dir)
                    orig_loc = str(rel_orig).replace("\\", "/")
                except Exception:
                    orig_loc = str(selected_item.path).replace("\\", "/")

                cache_file = self.workspace.get_cache_path(f"{selected_item.stem}.md") if self.workspace else None
                if cache_file and cache_file.exists():
                    try:
                        rel_cache = cache_file.relative_to(self.workspace.root_dir)
                        md_loc = str(rel_cache).replace("\\", "/")
                    except Exception:
                        md_loc = str(cache_file).replace("\\", "/")
                    md_display = f"{md_loc} (Terindeks)"
                else:
                    md_loc = f"cache/{selected_item.stem}.md"
                    md_display = f"{md_loc} (Belum terindeks)"

                lines = [
                    f"Judul             : {selected_item.title or selected_item.stem}",
                    f"Penulis           : {', '.join(selected_item.authors) if selected_item.authors else '-'}",
                    f"Tahun             : {selected_item.year or '-'}",
                    f"Cite Key          : {selected_item.cite_key or '-'}",
                    f"DOI / URL         : {selected_item.doi or selected_item.url or '-'}",
                    f"Kategori          : {'Papers (Naskah & Jurnal Ilmiah)' if selected_item.category == 'papers' else 'Web (Arsip & Portal Resmi)'}",
                    f"Nama Berkas       : {selected_item.filename}",
                    f"Ukuran            : {selected_item.readable_size} ({selected_item.size_bytes:,} bytes)",
                    f"Terakhir Update   : {selected_item.updated_at or '-'}",
                    f"Lokasi Berkas Asli: {orig_loc}",
                    f"Lokasi Berkas MD  : {md_display}",
                    f"Tabel Terindeks   : {selected_item.tables_count} tabel ({tbl_folder})",
                    f"Visual Terindeks  : {selected_item.visuals_count} gambar / diagram ({img_folder})",
                ]
                interactive_detail_viewer(
                    title=f"Metadata Berkas: {selected_item.filename}",
                    content_lines=lines,
                    actions=[("back", "Kembali", "esc")],
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
                    all_items = self.vault_manager.list_active_files()
                    self.console.print(f"[bold green][✓] Berkas '{selected_item.filename}' berhasil diarsipkan ke .archive/.[/bold green]\n")

    def run_interactive_vault_add(self) -> None:
        """Add document options submenu."""
        opts = [
            ("local", "Berkas Lokal (Folder local/ atau Jalur Berkas)"),
            ("findpaper", "Cari Journal Ilmiah"),
            ("findwebsite", "Cari Website"),
            ("gdrive", "Tambah Dari Google Drive"),
            ("back", "Kembali"),
        ]

        choice = interactive_menu(
            opts,
            header=Text("Tambah Berkas ke Vault:", style="bold bright_white"),
            console=self.console,
            footer_hint="↑/↓ Navigasi · enter Pilih · 1-5 Lompat cepat · esc Kembali",
        )

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
            self.run_interactive_vault_find_paper()

        elif choice == "findwebsite":
            self.run_interactive_vault_find_web()

        elif choice == "local":
            try:
                self.console.print()
                path_str = self.console.input("[bold cyan]Jalur berkas lokal atau tekan Enter untuk scan folder local/: [/bold cyan]").strip()
                if path_str:
                    self.cmd_vault_add(path_str)
                else:
                    self.cmd_vault_update()
            except (KeyboardInterrupt, EOFError):
                pass

    def run_interactive_vault_find_paper(self, initial_query: str = "") -> None:
        """Search academic literature and select papers to ingest into the vault via paginated carousel."""
        if not self.workspace or not getattr(self, "scout", None):
            self.console.print("[red][x] Buka atau buat project terlebih dahulu sebelum mencari journal.[/red]\n")
            return

        query = initial_query.strip().strip("'\"")
        if not query:
            try:
                self.console.print()
                query = self.console.input("[bold cyan]Topik atau judul journal ilmiah untuk dicari: [/bold cyan]").strip()
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[yellow]Pencarian journal dibatalkan.[/yellow]\n")
                return

        if not query or query.lower() in ("q", "quit", "cancel", "exit", "batal"):
            self.console.print("[yellow]Pencarian journal dibatalkan.[/yellow]\n")
            return

        with self.console.status("[cyan]Mengekstrak parameter dan menelusuri OpenAlex, Semantic Scholar & ArXiv...[/cyan]"):
            try:
                query_spec = self.scout.extract_search_criteria(query, engine=self.engine)
                candidates = asyncio.run(self.scout.search_multi_source(query_spec, limit=25))
            except Exception as exc:
                self.console.print(f"[bold red][x] Pencarian journal gagal:[/bold red] {exc}\n")
                return

        if not candidates:
            self.console.print(f"[yellow][!] Tidak ditemukan journal ilmiah yang sesuai untuk topik '{query}'.[/yellow]\n")
            return

        columns = [
            {"name": "#", "justify": "right", "style": "dim", "width": 3},
            {"name": "Sumber", "justify": "center", "width": 12},
            {"name": "Judul Journal", "style": "bold white", "width": 36, "overflow": "fold"},
            {"name": "Penulis", "style": "dim", "width": 18, "overflow": "fold"},
            {"name": "Tahun", "justify": "center", "width": 6, "style": "dim"},
            {"name": "Akses", "justify": "center", "width": 13},
            {"name": "Intisari / Relevansi", "style": "dim", "overflow": "fold"},
        ]

        rows = []
        for idx, item in enumerate(candidates, 1):
            if item.source == "openalex":
                source_badge = "[bold cyan][OpenAlex][/bold cyan]"
            elif item.source == "ai_recommended":
                source_badge = "[bold cyan][Seminal][/bold cyan]"
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
                item.relevance_reason or "Sesuai parameter pencarian.",
            ])

        dl_choice = interactive_carousel_table(
            title="Katalog Hasil Pencarian Journal",
            columns=columns,
            rows=rows,
            console=self.console,
            prompt_label="download > ",
            help_text="Navigasi dengan ↑/↓ lalu tekan space untuk centang, atau ketik nomor (contoh: '1, 3' / 'all', 'q' untuk batal):",
            item_label="journal",
        )

        if not dl_choice or dl_choice.lower() in ("q", "quit", "cancel", "n", "no", "batal"):
            self.console.print("[dim]Pencarian selesai tanpa mengunduh berkas.[/dim]\n")
            return

        to_download_indices = []
        if dl_choice.lower() == "all":
            to_download_indices = list(range(len(candidates)))
        else:
            for part in dl_choice.replace(",", " ").split():
                if part.isdigit():
                    num = int(part) - 1
                    if 0 <= num < len(candidates):
                        to_download_indices.append(num)

        if not to_download_indices:
            self.console.print("[dim]Tidak ada nomor journal valid yang dipilih.[/dim]\n")
            return

        selected_papers = [candidates[i] for i in to_download_indices]

        ingested_count = 0
        failed_count = 0
        ingested_files: list[str] = []

        with self.console.status(f"[cyan]Mengunduh dan memproses {len(selected_papers)} journal terpilih ke Vault...[/cyan]"):
            for p in selected_papers:
                try:
                    if p.has_pdf and p.pdf_url:
                        dl_path = asyncio.run(self.scout.download_scout_paper(p))
                        ingested_files.append(dl_path.name)
                        ingested_count += 1
                    elif p.abstract:
                        self.scout.ingest_abstract_paper(p)
                        safe_stem = sanitize_filename(p.title)
                        ingested_files.append(f"{safe_stem}.md")
                        ingested_count += 1
                    else:
                        failed_count += 1
                except Exception as dl_err:
                    logger.warning(f"Failed to ingest paper {p.title}: {dl_err}")
                    failed_count += 1

        if self.vault_manager:
            self.vault_manager.reload_vault(force=False)
            if ingested_files:
                self.vault_manager.record_history(
                    action="tambah_journal",
                    description=f"Mengunduh {ingested_count} journal ilmiah dari pencarian '{query}'",
                    files_affected=ingested_files,
                    details={"query": query, "selected_count": len(selected_papers)},
                )
        self._sync_vault_cache()

        self.console.print()
        if ingested_count > 0:
            self.console.print(
                f"[bold green][✓] Berhasil menambahkan {ingested_count} journal ke Vault.[/bold green]"
            )
            for f in ingested_files[:5]:
                self.console.print(f"  [cyan]•[/cyan] [white]{f}[/white]")
            if len(ingested_files) > 5:
                self.console.print(f"  [dim]... dan {len(ingested_files) - 5} berkas lainnya.[/dim]")
        if failed_count > 0:
            self.console.print(f"[yellow][!] {failed_count} journal tidak dapat diunduh (paywalled / akses ditolak).[/yellow]")
        self.console.print()

    def run_interactive_vault_find_web(self, initial_target: str = "") -> None:
        """Search official web documents and select pages to ingest into the vault via paginated carousel."""
        if not self.workspace:
            self.console.print("[red][x] Buka atau buat project terlebih dahulu sebelum mencari website.[/red]\n")
            return

        target = initial_target.strip().strip("'\"")
        if not target:
            try:
                self.console.print()
                target = self.console.input("[bold cyan]URL atau kata kunci dokumen web resmi: [/bold cyan]").strip()
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[yellow]Pencarian website dibatalkan.[/yellow]\n")
                return

        if not target or target.lower() in ("q", "quit", "cancel", "exit", "batal"):
            self.console.print("[yellow]Pencarian website dibatalkan.[/yellow]\n")
            return

        from scripcy.core.web_scout import WebScout, OfficialSourceRejectedError, WebScoutError
        web_scout = getattr(self, "web_scout", None) or WebScout(workspace=self.workspace)
        vec_vault = self.vault_manager.vector_vault if self.vault_manager else None

        # 1. Direct URL Ingestion
        if target.startswith("http://") or target.startswith("https://"):
            self.console.print(f"\n[cyan]Memverifikasi dan memproses sumber resmi:[/cyan] [bold]{target}[/bold]")
            with self.console.status("[cyan]Mengevaluasi otoritas, mengekstrak visual substantif, dan mengonversi Markdown...[/cyan]"):
                try:
                    parsed = asyncio.run(web_scout.ingest_url(target, vector_vault=vec_vault))
                except OfficialSourceRejectedError as rej_err:
                    self.console.print(f"\n[bold red][x] Sumber Ditolak:[/bold red] {rej_err}\n")
                    return
                except Exception as exc:
                    self.console.print(f"\n[bold red][x] Gagal memproses URL web:[/bold red] {exc}\n")
                    return

            if self.vault_manager:
                self.vault_manager.reload_vault(force=False)
                self.vault_manager.record_history(
                    action="tambah_web",
                    description=f"Menambahkan dokumen web resmi dari URL '{target}'",
                    files_affected=[str(parsed.file_path.name)],
                    details={"url": target, "title": parsed.metadata.title},
                )
            self._sync_vault_cache()

            self.console.print("\n[bold green][✓] Dokumen web resmi berhasil dimasukkan ke Vault:[/bold green]")
            t = Table(box=None, show_header=False, padding=(0, 2))
            t.add_column("Key", style="dim", width=16)
            t.add_column("Value", style="bright_white")
            t.add_row("Judul", parsed.metadata.title)
            t.add_row("Instansi", parsed.metadata.institution or "Resmi")
            t.add_row("Cite Key", f"[bold cyan][{parsed.metadata.cite_key}][/bold cyan]")
            t.add_row("Skor Otoritas", f"{parsed.metadata.authority_score or 0.0:.2f} (Tier {parsed.metadata.authority_tier or 1})")
            t.add_row("Snapshot File", str(parsed.file_path.name))
            t.add_row("Cache Markdown", str(parsed.cache_path.name if parsed.cache_path else ""))
            self.console.print(t)
            self.console.print()
            return

        # 2. Query Search Flow with Carousel
        self.console.print(f"\n[cyan]Mencari dokumen web resmi untuk topik:[/cyan] [bold]{target}[/bold]")
        with self.console.status("[cyan]Menelusuri portal web pemerintah, akademik, dan institusi resmi...[/cyan]"):
            try:
                results = asyncio.run(web_scout.search_official_web(target, limit=18, require_official=True))
            except Exception as exc:
                self.console.print(f"\n[bold red][x] Pencarian web gagal:[/bold red] {exc}\n")
                return

        if not results:
            self.console.print(f"[yellow][!] Tidak ditemukan dokumen web resmi yang memenuhi kriteria otoritas untuk query '{target}'.[/yellow]\n")
            return

        columns = [
            {"name": "#", "justify": "right", "style": "dim", "width": 3},
            {"name": "Instansi / Domain", "style": "green", "width": 24, "overflow": "fold"},
            {"name": "Judul Dokumen Web", "style": "bold white", "width": 38, "overflow": "fold"},
            {"name": "Otoritas", "justify": "center", "style": "cyan", "width": 10},
            {"name": "Snippet / Ringkasan", "style": "dim", "overflow": "fold"},
        ]

        rows = []
        for idx, res in enumerate(results, 1):
            inst_label = res.institution or res.domain
            tier_label = f"Tier {res.authority_tier}" if getattr(res, "authority_tier", None) else f"{res.authority_score:.2f}"
            rows.append([
                str(idx),
                inst_label,
                res.title,
                tier_label,
                res.snippet or "Sumber portal resmi pemerintah / akademik.",
            ])

        choice = interactive_carousel_table(
            title="Katalog Dokumen Web Resmi",
            columns=columns,
            rows=rows,
            console=self.console,
            prompt_label="ingest > ",
            help_text="Navigasi dengan ↑/↓ lalu tekan space untuk centang, atau ketik nomor (contoh: '1, 3' / 'all', 'q' untuk batal):",
            item_label="dokumen web",
        )

        if not choice or choice.lower() in ("q", "quit", "cancel", "n", "no", "batal"):
            self.console.print("[dim]Pencarian selesai tanpa menyimpan dokumen web.[/dim]\n")
            return

        to_ingest_indices = []
        if choice.lower() == "all":
            to_ingest_indices = list(range(len(results)))
        else:
            for part in choice.replace(",", " ").split():
                if part.isdigit():
                    num = int(part) - 1
                    if 0 <= num < len(results):
                        to_ingest_indices.append(num)

        if not to_ingest_indices:
            self.console.print("[dim]Tidak ada nomor dokumen valid yang dipilih.[/dim]\n")
            return

        selected_web = [results[i] for i in to_ingest_indices]
        ingested_count = 0
        failed_count = 0
        ingested_files: list[str] = []

        with self.console.status(f"[cyan]Mengunduh dan memproses {len(selected_web)} dokumen web resmi ke Vault...[/cyan]"):
            for item in selected_web:
                try:
                    parsed = asyncio.run(web_scout.ingest_url(item.url, vector_vault=vec_vault))
                    ingested_files.append(str(parsed.file_path.name))
                    ingested_count += 1
                except Exception as ing_err:
                    logger.warning(f"Failed to ingest web document {item.url}: {ing_err}")
                    failed_count += 1

        if self.vault_manager:
            self.vault_manager.reload_vault(force=False)
            if ingested_files:
                self.vault_manager.record_history(
                    action="tambah_web",
                    description=f"Menyimpan {ingested_count} dokumen web resmi dari pencarian '{target}'",
                    files_affected=ingested_files,
                    details={"query": target, "selected_count": len(selected_web)},
                )
        self._sync_vault_cache()

        self.console.print()
        if ingested_count > 0:
            self.console.print(
                f"[bold green][✓] Berhasil menambahkan {ingested_count} dokumen web resmi ke Vault.[/bold green]"
            )
            for f in ingested_files[:5]:
                self.console.print(f"  [cyan]•[/cyan] [white]{f}[/white]")
            if len(ingested_files) > 5:
                self.console.print(f"  [dim]... dan {len(ingested_files) - 5} berkas lainnya.[/dim]")
        if failed_count > 0:
            self.console.print(f"[yellow][!] {failed_count} dokumen web gagal diproses.[/yellow]")
        self.console.print()

    def render_vault_landing(self) -> None:
        """Render concise status overview of vault resources per minimalist wireframe."""
        if not self.vault_manager or not self.workspace:
            return

        health = self.vault_manager.check_vault_health()
        self.console.print(f"\n[bold bright_white]Vault Berkas [{self.current_project.name if self.current_project else 'Aktif'}][/bold bright_white]")
        v_table = Table(box=None, padding=(0, 2), show_header=False)
        v_table.add_column("Label", style="dim", width=18)
        v_table.add_column("Value", style="white")

        v_table.add_row("Status", health["status_display"])
        v_table.add_row("Total Berkas", str(health["total_files"]))

        if health.get("warning_details"):
            warn_text = "\n".join(health["warning_details"])
            v_table.add_row("Peringatan", f"[yellow]{warn_text}[/yellow]")
        if health.get("error_details"):
            err_text = "\n".join(health["error_details"])
            v_table.add_row("Error", f"[red]{err_text}[/red]")

        self.console.print(v_table)
        self.console.print()

    def render_vault_view(self) -> None:
        """Render active vault items (documents, resources, tables, images) following DESIGN.md."""
        if not self.vault_manager:
            return
        items = self.vault_manager.list_active_files()
        tables = self.vault_manager.list_tables()
        images = self.vault_manager.list_images()

        if not items and not tables and not images:
            self.console.print("\n[dim]Vault masih kosong (0 resource, 0 tabel, 0 gambar).[/dim]")
            self.console.print("[dim]Gunakan 'add <path>', 'findpaper <topik>', atau 'gdrive' untuk menambahkan berkas.[/dim]\n")
            return

        self.console.print(f"\n[bold bright_white]Koleksi Resource Vault [{self.current_project.name if self.current_project else 'Vault'}][/bold bright_white] [dim]({len(items)} berkas)[/dim]")
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
            if item.has_error:
                fn_str = f"[bold red]{item.filename}[/bold red]"
                title_str = f"[bold red]{item.title or item.cite_key}[/bold red]"
                md_status = "[bold red][Error / Gagal][/bold red]"
            elif getattr(item, "is_duplicate", False):
                fn_str = f"[bold yellow]{item.filename}[/bold yellow]"
                title_str = f"[bold yellow]{item.title or item.cite_key}[/bold yellow]"
                md_status = "[bold yellow][Duplikat][/bold yellow]"
            elif item.is_cached:
                fn_str = f"[bold white]{item.filename}[/bold white]"
                title_str = item.title or item.cite_key
                md_status = "[bold cyan][Ready / Cached][/bold cyan]"
            else:
                fn_str = f"[white]{item.filename}[/white]"
                title_str = item.title or item.cite_key
                md_status = "[dim][Needs Reload][/dim]"

            p_table.add_row(
                str(idx),
                fn_str,
                item.extension.upper().lstrip("."),
                item.readable_size,
                md_status,
                title_str,
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
            t_table.add_column("Sumber Berkas", style="bold white")
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
            i_table.add_column("Sumber Berkas", style="dim", width=25)
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
            f"\n[dim]Ketik 'view <#>' untuk baca berkas, 'view table <#>' untuk buka tabel, 'remove <#>' untuk arsip, atau 'back' untuk kembali{archive_hint}.[/dim]\n"
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
        from scripcy.ui.terminal_image import render_visual_panel
        v_dict = dict(img)
        v_dict["index"] = idx + 1
        v_dict["asset_type"] = "image"
        v_panel = render_visual_panel(v_dict, console=self.console)
        if v_panel:
            self.console.print()
            self.console.print(v_panel)
            self.console.print()
        else:
            self.console.print(f"\n[bold bright_white]Gambar #{arg} [{img_p.name}][/bold bright_white]")
            self.console.print(f"[dim]Paper Sumber : {img.get('paper_id')} (Halaman {img.get('page_num')})[/dim]")
            self.console.print(f"[dim]Dimensi      : {img.get('width')} x {img.get('height')} px[/dim]")
            self.console.print(f"[dim]File Path    : {img_p}[/dim]")
            if img.get("caption"):
                self.console.print(f"[dim]Caption Teks : {img.get('caption')}[/dim]")
            if img.get("ai_analysis"):
                self.console.print(f"\n[bold cyan]Interpretasi Visual Multimodal AI:[/bold cyan]\n{img.get('ai_analysis')}\n")

    def cmd_vault_reindex_visuals(self) -> None:
        """Purge junk images/tables and re-extract informative academic figures."""
        if not self.vault_manager:
            self.console.print("[red][x] Tidak ada vault aktif.[/red]\n")
            return
        model_name = self.get_active_project_model()
        with self.console.status("[bold cyan]Membersihkan dan mengindeks ulang visual ilmiah di vault...[/bold cyan]"):
            res = self.vault_manager.clean_and_reindex_visuals(engine=self.engine, model_name=model_name)
        self.console.print(f"[bold green][✓] Pembersihan visual tuntas: {res.get('papers', 0)} paper diproses, {res.get('images', 0)} gambar informatif terindeks, {res.get('tables', 0)} tabel terindeks.[/bold green]\n")

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
            self.run_interactive_vault_find_paper(arg)
        elif cmd in ("web", "findweb", "findwebsite"):
            self.run_interactive_vault_find_web(arg)
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
        elif cmd in ("graph", "network", "plot"):
            self.cmd_vault_graph(arg)
        elif cmd in ("reindex-visuals", "clean-visuals", "reindex_visuals"):
            self.cmd_vault_reindex_visuals()
        else:
            self.console.print(f"[yellow]Unknown vault command: '{clean}'. Type 'help' to see vault commands, or 'back' to return.[/yellow]\n")

    def cmd_vault_graph(self, arg: str = "") -> None:
        """Render and open interactive GraphRAG knowledge network in default browser."""
        if not self.workspace:
            self.console.print("[red][x] Buka atau pilih proyek terlebih dahulu untuk memvisualisasikan graph.[/red]\n")
            return

        with self.console.status("[cyan]Menganalisis entitas dan menyusun GraphRAG network...[/cyan]"):
            builder = GraphRAGBuilder(workspace=self.workspace, vault_manager=self.vault_manager)
            graph_data = builder.build_graph(query=arg.strip())
            graph_dir = self.workspace.vault_dir / "graphs"
            graph_dir.mkdir(parents=True, exist_ok=True)
            output_file = graph_dir / "rag_network.html"
            builder.render_interactive_html(
                output_path=output_file,
                graph_data=graph_data,
                title=f"Scripcy GraphRAG - {self.current_project.name if self.current_project else 'Vault'}",
            )

        n_count = len(graph_data.get("nodes", []))
        e_count = len(graph_data.get("links", []))
        self.console.print(
            f"[bold green][\u2713] Berhasil merender GraphRAG Network ({n_count} node, {e_count} relasi):[/bold green]\n"
            f"[dim]File: {output_file}[/dim]\n"
        )
        try:
            webbrowser.open(output_file.as_uri())
            self.console.print("[cyan]\u25cf Membuka visualisasi interaktif di browser...[/cyan]\n")
        except Exception as e:
            self.console.print(f"[yellow]Tidak dapat membuka browser otomatis: {e}[/yellow]\n")

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
                        skill_name="scripcy_manual",
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

    def resolve_duplicate_item(self, item_info: dict[str, Any]) -> str:
        """Interactive cursor-driven dialog to resolve a duplicate file adhering to DESIGN.md."""
        fname = item_info["filename"]
        match_loc = str(item_info.get("existing_match", "-")).replace("\\", "/")
        try:
            rel_match = Path(match_loc).relative_to(self.workspace.root_dir)
            match_disp = str(rel_match).replace("\\", "/")
        except Exception:
            match_disp = match_loc

        header = Table(box=None, padding=(0, 1), show_header=False)
        header.add_column("Col")
        header.add_row(Text(f"Peringatan Duplikasi: '{fname}'", style="bold yellow"))
        header.add_row(Text(f"Berkas serupa sudah ada di: {match_disp}", style="dim"))
        header.add_row(Text("Pilih tindakan yang diinginkan:", style="bold bright_white"))

        stem = Path(fname).stem
        ext = Path(fname).suffix
        dup_opts = [
            ("keep_both", f"Simpan keduanya (Ganti nama baru: '{stem}_salinan{ext}')"),
            ("overwrite", f"Timpa berkas lama (Gantikan dengan versi baru dari local/)"),
            ("discard", f"Abaikan berkas baru (Hapus berkas duplikat dari local/)"),
        ]

        choice = interactive_menu(
            options=dup_opts,
            header=header,
            console=self.console,
            footer_hint="↑/↓ Navigasi · enter Pilih · 1-3 Lompat cepat · ctrl+c / esc Batalkan",
            allow_back=True,
        )
        if choice in ("back", None):
            return "discard"
        return choice

    def run_interactive_context_questionnaire(
        self,
        group_items: list[dict[str, Any]],
        folder_context: str,
    ) -> str | None:
        """Interactive cursor-driven questionnaire to clarify experiment context adhering to DESIGN.md."""
        file_names = [it["filename"] for it in group_items]
        clean_ctx = folder_context.replace(" > ", " / ") if folder_context else "Root (Umum)"

        options_text: list[str] = []
        if folder_context:
            options_text.append(f"Hasil pengujian parameter: {clean_ctx}")
            options_text.append(f"Data eksperimen kelompok: {clean_ctx}")
            options_text.append(f"Sampel kontrol atau baseline: {clean_ctx}")
        else:
            options_text.append("Dataset dan dokumentasi pengujian mandiri")
            options_text.append("Sampel data eksperimental laboratorium")

        # Smart hypothesis generation using LLM engine if active (manifest only, no token leak)
        if hasattr(self, "engine") and self.engine and getattr(self, "config", None):
            try:
                manifest_prompt = (
                    f"Diberikan daftar berkas penelitian: {', '.join(file_names[:6])} di dalam folder '{clean_ctx}'.\n"
                    "Buat 2 opsi hipotesis konteks penelitian yang sangat ringkas (maksimal 8 kata per opsi) dalam Bahasa Indonesia.\n"
                    "Format keluaran hanya 2 baris, tanpa nomor atau bullet."
                )
                ai_reply = self.engine.generate_text(manifest_prompt)
                ai_lines = [l.strip().lstrip("1234567890.- ") for l in (ai_reply or "").splitlines() if l.strip()]
                if len(ai_lines) >= 2:
                    options_text = ai_lines[:3]
            except Exception:
                pass

        menu_options = [(opt, opt) for opt in options_text]
        menu_options.append(("custom", "Tuliskan konteks eksperimen sendiri (Input Bebas)..."))

        header = Table(box=None, padding=(0, 1), show_header=False)
        header.add_column("Col")
        header.add_row(Text(f"Identifikasi Konteks Berkas ({len(file_names)} berkas di folder '{clean_ctx}'):", style="bold bright_white"))
        header.add_row(Text(f"Berkas: {', '.join(file_names[:4])}" + (f" (+{len(file_names)-4} lainnya)" if len(file_names) > 4 else ""), style="dim"))
        header.add_row(Text(""))
        header.add_row(Text(f'Pertanyaan AI: "Bagaimana berkas pada kelompok \'{clean_ctx}\' ini sebaiknya dikelompokkan?"', style="bold cyan"))

        choice = interactive_menu(
            options=menu_options,
            header=header,
            console=self.console,
            footer_hint="↑/↓ Navigasi · enter Pilih · 1-4 Lompat cepat · ctrl+c / esc Batalkan",
            allow_back=True,
        )

        if choice in ("back", None):
            return None

        if choice == "custom":
            try:
                custom_text = interactive_input_prompt(
                    console=self.console,
                    prompt_label="Konteks > ",
                    initial_buffer=clean_ctx,
                    default=clean_ctx,
                )
                return (custom_text or clean_ctx).strip()
            except (KeyboardInterrupt, Exception):
                return None

        return choice

    def run_interactive_vault_history(self) -> None:
        """Browse vault history and audit log carousel per DESIGN.md."""
        entries = self.vault_manager.list_history() if self.vault_manager else []
        if not entries:
            self.console.print("\n[dim]Belum ada riwayat aktivitas di dalam Vault.\n[/dim]")
            time.sleep(1.2)
            return

        columns = [
            {"name": "#", "justify": "right", "width": 3, "style": "dim"},
            {"name": "Waktu", "style": "dim", "width": 19},
            {"name": "Aktivitas", "style": "bold cyan", "width": 22},
            {"name": "Keterangan", "style": "white", "width": 36},
            {"name": "Berkas", "justify": "right", "width": 10, "style": "dim"},
        ]

        def _history_row_builder(idx: int, item: Any, is_selected: bool) -> list[Any]:
            arrow = "> " if is_selected else "  "
            act_text = Text()
            act_text.append(arrow, style="bold cyan" if is_selected else "dim")
            act_name = str(item.get("action", "-")).replace("_", " ").title()
            act_text.append(act_name[:20], style="bold bright_white" if is_selected else "white")

            files_count = len(item.get("files_affected", []))
            return [
                str(idx + 1),
                str(item.get("timestamp", "-")),
                act_text,
                str(item.get("description", "-"))[:34],
                f"{files_count} item",
            ]

        header = Text(f"Riwayat Aktivitas Vault [{self.current_project.name}] • {len(entries)} catatan", style="bold bright_white")
        actions = [
            ("view", "Lihat Rincian", "enter"),
            ("back", "Kembali", "esc"),
        ]

        while True:
            action, idx, selected_item = interactive_item_browser(
                title=header,
                columns=columns,
                items=entries,
                row_builder=_history_row_builder,
                actions=actions,
                console=self.console,
                empty_message="Tidak ada catatan riwayat.",
            )

            if action in ("back", "esc") or not selected_item:
                return

            if action == "view" and selected_item:
                affected = selected_item.get("files_affected", [])
                lines = [
                    f"Waktu Eksekusi : {selected_item.get('timestamp', '-')}",
                    f"Jenis Aktivitas: {str(selected_item.get('action', '-')).replace('_', ' ').title()}",
                    f"Keterangan     : {selected_item.get('description', '-')}",
                    "",
                    f"Berkas Terdampak ({len(affected)} berkas):",
                ]
                for f in affected:
                    lines.append(f"  • {f}")
                details = selected_item.get("details", {})
                if details:
                    lines.append("")
                    lines.append("Rincian Tambahan:")
                    for k, v in details.items():
                        lines.append(f"  {k}: {v}")

                interactive_detail_viewer(
                    title=f"Rincian Aktivitas: {selected_item.get('action', 'Riwayat')}",
                    content_lines=lines,
                    actions=[("back", "Kembali", "esc")],
                    console=self.console,
                )

    def cmd_vault_update(self) -> None:
        """Scan folder vault dan konversi berkas baru/berubah ke Markdown."""
        if not self.vault_manager or not self.workspace:
            self.console.print("[red][x] Manajer vault tidak aktif.[/red]\n")
            return

        # 1. Check local staging area
        local_items = self.vault_manager.scan_local_staging()
        if local_items:
            self.console.print(f"\n[bold cyan]Ditemukan {len(local_items)} berkas di folder local/.[/bold cyan]")

            # Resolve duplicates first
            duplicate_actions: dict[str, str] = {}
            for item in local_items:
                if item.get("is_duplicate"):
                    dup_choice = self.resolve_duplicate_item(item)
                    duplicate_actions[item["rel_path"]] = dup_choice

            # Group items by folder context
            from collections import defaultdict
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for item in local_items:
                if duplicate_actions.get(item["rel_path"]) == "discard":
                    continue
                groups[item["folder_context"]].append(item)

            # Interactive context questionnaire per group
            group_contexts: dict[str, str] = {}
            for f_ctx, g_items in groups.items():
                ctx_result = self.run_interactive_context_questionnaire(g_items, f_ctx)
                if ctx_result is None:
                    self.console.print("\n[dim]Sesi kuesioner dibatalkan. Menggunakan nama folder sebagai konteks.[/dim]\n")
                    group_contexts[f_ctx] = f_ctx
                else:
                    group_contexts[f_ctx] = ctx_result

            # Move and ingest items
            processed_files: list[str] = []
            for item in local_items:
                act = duplicate_actions.get(item["rel_path"], "keep_both")
                ctx = group_contexts.get(item["folder_context"], item["folder_context"])
                dest = self.vault_manager.process_local_item(
                    item_info=item,
                    action=act,
                    context_override=ctx,
                )
                if dest:
                    processed_files.append(item["filename"])

            if processed_files:
                self.vault_manager.record_history(
                    action="sortir_mandiri",
                    description=f"Memindahkan & menyortir {len(processed_files)} berkas dari local/ ke Vault",
                    files_affected=processed_files,
                    details={"grup_konteks": group_contexts},
                )
                self.console.print(f"[bold green][✓] {len(processed_files)} berkas dari local/ berhasil disortir dan dipindahkan ke Vault.[/bold green]\n")

        with self.console.status("[cyan]Memindai vault dan memperbarui berkas dengan MarkItDown...[/cyan]"):
            total, converted = self.vault_manager.reload_vault(force=False)
            self._sync_vault_cache()

        self.vault_manager.record_history(
            action="update_berkas",
            description=f"Sinkronisasi vault: {total} total berkas, {converted} diperbarui",
            files_affected=[],
            details={"total": total, "converted": converted},
        )

        self.console.print(f"[bold green][✓] Pembaruan berkas selesai:[/bold green] {total} total berkas, [bold cyan]{converted} berkas diperbarui[/bold cyan] ke cache Markdown.")
        self.console.print(f"[dim]Konteks sitasi aktif diperbarui ({len(self.active_documents)} berkas aktif).[/dim]\n")

    cmd_vault_reload = cmd_vault_update

    def _background_vault_add_worker(self, task_id: str, path_str: str) -> str:
        """Background worker ingesting local document and converting to Markdown."""
        src_path = Path(path_str).expanduser().resolve()
        self.task_manager.log(task_id, f"Ingesting '{src_path.name}' into vault...")
        try:
            item = self.vault_manager.add_resource(src_path)
            t_info = self.task_manager.get_task(task_id)
            sess_id = t_info.session_id if t_info else None
            self.add_session_active_documents([item.path.stem], session_id=sess_id)
            self.task_manager.record_ingested_stems(task_id, [item.path.stem])
            self._sync_vault_cache(session_id=sess_id)
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
            removed_stems = [item.path.stem for item in archived_items]
            for stem in removed_stems:
                self.active_documents.discard(stem)
            if self.chat_manager and removed_stems:
                self.chat_manager.remove_documents_from_all_runtimes(removed_stems)
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
            t_info = self.task_manager.get_task(task_id)
            sess_id = t_info.session_id if t_info else None
            self.add_session_active_documents([downloaded.stem], session_id=sess_id)
            self.task_manager.record_ingested_stems(task_id, [downloaded.stem])
            self._sync_vault_cache(session_id=sess_id)
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
            client_secrets = Path.home() / ".scripcy" / "credentials.json"
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

        sess_id = self.active_chat_session.id if (self.sub_mode == "chat" and self.active_chat_session) else None
        task = self.task_manager.submit(
            name=f"Google Drive Ingest: {file_id[:12]}...",
            task_type="gdrive",
            target_fn=self._background_gdrive_worker,
            session_id=sess_id,
            lane="io",
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
            self.console.print("[dim]Project vault is currently empty. Use [bold cyan]/search[/bold cyan] or [bold cyan]/parse <file>[/bold cyan] to add resources.[/dim]\n")
            return

        self.console.print(f"\n[bold bright_white]Indexed Documents & Resources ({len(stems)} total, {len(self.active_documents)} active)[/bold bright_white]")
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
        self.console.print("[dim]Use [bold cyan]/active <#|stem>[/bold cyan] to toggle documents in/out of context.[/dim]\n")

