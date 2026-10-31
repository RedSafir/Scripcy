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


class StudioDashboardMixin:
    def run_interactive_dashboard(self) -> str | None:

        """Dashboard interactive menu without typing."""

        while self.current_project is None:

            projects = self.project_manager.list_projects()

            p_count = len(projects)



            header = Text()
            header.append("scripcy", style="bold cyan")
            header.append(" • ", style="dim")
            header.append(f"dashboard ({p_count} proyek terdaftar)", style="dim")

            options = [
                ("projects", f"Proyek ({p_count} terdaftar)"),
                ("settings", "Pengaturan & Model"),
                ("exit", "Keluar"),
            ]

            choice = interactive_menu(
                options=options,
                header=header,
                console=self.console,
                footer_hint="↑/↓ Navigasi · enter Pilih · 1-3 Lompat cepat · esc Keluar",
                allow_back=True,
            )

            if choice in ("exit", None):
                self.console.print("\n[bold cyan]Menutup studio riset Scripcy. Sampai jumpa![/bold cyan]")
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
                    {"name": "Terakhir Dibuka", "style": "dim", "width": 16},
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
                header.append("Daftar Master Proyek", style="bold bright_white")
                header.append(f" ({len(projects)} proyek)", style="dim")

                actions = [
                    ("open", "Buka Proyek", "enter"),
                    ("new", "Proyek Baru", "n"),
                    ("delete", "Hapus Proyek", "d"),
                    ("back", "Kembali", "esc"),
                ]

                import scripcy.cli_studio as _cli_studio
                action, idx, selected_p = _cli_studio.interactive_item_browser(
                    title=header,
                    columns=columns,
                    items=projects,
                    row_builder=_row_builder,
                    actions=actions,
                    console=self.console,
                    empty_message="Belum ada proyek.",
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
                        f"Hapus registrasi project '{selected_p.name}' dari Scripcy?",
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
                            self.console.print(f"[bold green][✓] Registrasi project '{selected_p.name}' berhasil dihapus dari Scripcy.[/bold green]")
                            self.console.print(f"[dim]Catatan: File asli di '{selected_p.path}' tetap aman dan tidak dihapus dari disk.[/dim]\n")

    def interactive_create_project(self) -> None:

        """Prompt user cleanly for project name."""

        try:

            self.console.print()

            p_name = self.console.input("[bold cyan]Nama project baru: [/bold cyan]").strip()

            if not p_name:

                return

            with self.console.status(f"[bold cyan]Menyiapkan struktur master project '{p_name}'...[/bold cyan]"):
                p_info = self.project_manager.create_project(name=p_name)
                self.open_project(p_info, show_banner=False)

        except (KeyboardInterrupt, EOFError):

            self.console.print()

            return

    def run_interactive_settings(self) -> None:
        """Unified Antigravity-style interactive settings hub for Dashboard and Project modes."""
        while True:
            cur_m_raw = self.get_active_project_model() if self.current_project else self.config.default_model
            cur_m = _format_model_display_name(cur_m_raw)
            cur_eff = self.get_active_project_effort() if self.current_project else getattr(self.config, "default_effort_tier", "medium")
            cur_cit = self.config.citation.style.upper()

            header = Text()
            header.append("scripcy", style="bold cyan")
            header.append(" • ", style="dim")
            header.append("Pengaturan & Konfigurasi\n", style="bold bright_white")
            header.append(f"Model: {cur_m} ({cur_eff}) · Sitasi: {cur_cit} · Pencarian: OpenAlex (Utama)", style="dim")

            low_cfg = getattr(self.config, "effort_tier_low", None)
            med_cfg = getattr(self.config, "effort_tier_medium", None)
            ext_cfg = getattr(self.config, "effort_tier_extra", None)
            low_s = f"{low_cfg.min_sources}-{low_cfg.max_sources}" if low_cfg else "0-1"
            med_s = f"{med_cfg.min_sources}-{med_cfg.max_sources}" if med_cfg else "1-3"
            ext_s = f"{ext_cfg.min_sources}-{ext_cfg.max_sources}" if ext_cfg else "3-7"

            opts = [
                ("model", f"Model & Penalaran AI ({cur_m} · upaya {cur_eff})"),
                ("federation", "Pencarian Akademik & Provider (OpenAlex, S2, CrossRef, Unpaywall)"),
                ("citation", f"Format Sitasi & Bibliografi ({cur_cit})"),
                ("tier", f"Effort Tiers & Kuota Jurnal (Low: {low_s} · Med: {med_s} · Extra: {ext_s})"),
                ("keys", "Manajemen API Key (Kredensial BYOK)"),
                ("back", "Kembali ke " + (f"Proyek '{self.current_project.name}'" if self.current_project else "Dashboard")),
            ]

            ch = interactive_menu(
                opts,
                header=header,
                console=self.console,
                footer_hint="↑/↓ Navigasi · enter Pilih · 1-6 Lompat cepat · esc Kembali",
                allow_back=True,
            )

            if ch in ("back", None):
                return
            if ch == "model":
                self._run_interactive_model_settings()
            elif ch == "federation":
                self._run_interactive_federation_settings()
            elif ch == "citation":
                self.run_interactive_citation_settings()
            elif ch in ("tier", "agent"):
                self._run_interactive_tier_settings()
            elif ch == "keys":
                self._run_interactive_keys_settings()

    def run_interactive_dashboard_settings(self) -> None:
        """Alias for backward compatibility."""
        self.run_interactive_settings()

    def _run_interactive_model_settings(self) -> None:
        """Model selection and reasoning effort settings."""
        while True:
            cur_m_raw = self.get_active_project_model() if self.current_project else self.config.default_model
            cur_m_label = _format_model_display_name(cur_m_raw)
            cur_eff = self.get_active_project_effort() if self.current_project else "high"

            header = Text()
            header.append("Pengaturan Model & Penalaran AI\n", style="bold bright_white")
            header.append(f"Model Aktif: {cur_m_label} · Tingkat Penalaran: {cur_eff}", style="dim")

            opts = [
                ("switch", f"Ganti Model (Saat ini: {cur_m_label})"),
                ("effort", f"Ubah Upaya Penalaran (Saat ini: {cur_eff})"),
                ("back", "Kembali ke Pengaturan"),
            ]

            ch = interactive_menu(opts, header=header, console=self.console, footer_hint="↑/↓ Navigasi · enter Pilih · 1-3 Lompat cepat · esc Kembali", allow_back=True)
            if ch in ("back", None):
                return

            if ch == "switch":
                r9_base = self.config.api_bases.get("9router", "http://localhost:20128/v1")
                live_r9_models = fetch_9router_models(r9_base)
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

                chosen = interactive_model_selector(
                    current_model=cur_m_raw,
                    current_effort=cur_eff,
                    available_models=all_options,
                    console=self.console,
                )
                if chosen:
                    chosen_m, chosen_eff = chosen
                    if self.current_project:
                        self.project_manager.set_project_model(self.current_project.name, chosen_m)
                        self.project_manager.set_project_effort(self.current_project.name, chosen_eff)
                        self.current_project.model = chosen_m
                    else:
                        self.config.default_model = chosen_m
                        self.config.save()
                    if self.engine:
                        self.engine.default_model = chosen_m

            elif ch == "effort":
                eff_opts = [
                    ("low", "low    - Respons cepat, ideal untuk eksplorasi dan draf ringkas"),
                    ("medium", "medium - Penalaran standar untuk sintesis akademis seimbang"),
                    ("high", "high   - Penalaran mendalam untuk masalah kompleks (default)"),
                    ("back", "Back"),
                ]
                eff_ch = interactive_menu(eff_opts, header="Pilih Level Reasoning Effort:", console=self.console, allow_back=True)
                if eff_ch and eff_ch != "back":
                    if self.current_project:
                        self.project_manager.set_project_effort(self.current_project.name, eff_ch)
                    else:
                        self.config.reasoning_effort = eff_ch
                        self.config.save()

    def _run_interactive_federation_settings(self) -> None:
        """Academic Search & Literature Federation toggles and configurations."""
        while True:
            s2_en = getattr(self.config, "enable_semantic_scholar", True)
            cr_en = getattr(self.config, "enable_crossref", True)
            oc_en = getattr(self.config, "enable_opencitations", False)
            up_email = self.config.unpaywall_email or "researcher@scripcy.org (default)"

            header = Text()
            header.append("Academic Search & Literature Federation\n", style="bold bright_white")
            header.append("Atur provider pencarian paper, toggle status, dan konfigurasi email Unpaywall.", style="dim")

            s2_lbl = "[bold cyan]Active[/bold cyan]" if s2_en else "[dim]Disabled[/dim]"
            cr_lbl = "[bold cyan]Active[/bold cyan]" if cr_en else "[dim]Disabled[/dim]"
            oc_lbl = "[bold cyan]Active[/bold cyan]" if oc_en else "[dim]Disabled[/dim]"

            opts = [
                ("s2_toggle", f"Semantic Scholar : {s2_lbl} (Tekan Enter untuk toggle)"),
                ("cr_toggle", f"CrossRef API     : {cr_lbl} (Tekan Enter untuk toggle · Polite mailto)"),
                ("oc_toggle", f"OpenCitations    : {oc_lbl} (Tekan Enter untuk toggle)"),
                ("email", f"Unpaywall Email  : {up_email} (Ubah email Green OA)"),
                ("openalex_info", "OpenAlex         : [bold cyan]Utama / Primary[/bold cyan] (Selalu aktif)"),
                ("arxiv_info", "ArXiv Guard      : [bold cyan]Active[/bold cyan] (Khusus CS/AI/Math)"),
                ("back", "Back to Settings"),
            ]

            ch = interactive_menu(opts, header=header, console=self.console, allow_back=True)
            if ch in ("back", None):
                return

            if ch == "s2_toggle":
                new_val = not s2_en
                self.config_manager.update(enable_semantic_scholar=new_val)
                self.config = self.config_manager.config
            elif ch == "cr_toggle":
                new_val = not cr_en
                self.config_manager.update(enable_crossref=new_val)
                self.config = self.config_manager.config
            elif ch == "oc_toggle":
                new_val = not oc_en
                self.config_manager.update(enable_opencitations=new_val)
                self.config = self.config_manager.config
            elif ch == "email":
                new_email = interactive_input_prompt(
                    console=self.console,
                    prompt_label="Unpaywall Email > ",
                    initial_buffer=self.config.unpaywall_email or "",
                ).strip()
                if new_email and "@" in new_email:
                    self.config_manager.update(unpaywall_email=new_email)
                    self.config = self.config_manager.config

    def run_interactive_citation_settings(self) -> None:
        """Select citation style and bibliography options."""
        while True:
            cur_style = self.config.citation.style.lower()
            inc_doi = getattr(self.config.citation, "include_doi", True)

            header = Text()
            header.append("Citation & Bibliography Style\n", style="bold bright_white")
            header.append(f"Format sitasi aktif: {cur_style.upper()} · Sertakan DOI: {'Ya' if inc_doi else 'Tidak'}", style="dim")

            opts = [
                ("apa", f"APA Style (American Psychological Association) {'[bold cyan](Active)[/bold cyan]' if cur_style == 'apa' else ''}"),
                ("ieee", f"IEEE Style (Numbered [1], [2]) {'[bold cyan](Active)[/bold cyan]' if cur_style == 'ieee' else ''}"),
                ("bibtex", rf"BibTeX Citation Keys (\[Vaswani2017]) {'[bold cyan](Active)[/bold cyan]' if cur_style == 'bibtex' else ''}"),
                ("doi_toggle", f"Sertakan DOI di Bibliografi: {'[bold cyan]Ya[/bold cyan]' if inc_doi else '[dim]Tidak[/dim]'} (Tekan Enter)"),
                ("back", "Back to Settings"),
            ]

            ch = interactive_menu(opts, header=header, console=self.console, allow_back=True)
            if ch in ("back", None):
                return

            if ch in ("apa", "ieee", "bibtex"):
                self.config.citation.style = ch
                self.config.save()
            elif ch == "doi_toggle":
                self.config.citation.include_doi = not inc_doi
                self.config.save()

    def _run_interactive_tier_settings(self) -> None:
        """Configure academic search effort tiers and journal quotas (low, medium, extra)."""
        while True:
            cur_eff = self.get_active_project_effort() if self.current_project else getattr(self.config, "default_effort_tier", "medium")
            low_cfg = getattr(self.config, "effort_tier_low", None)
            med_cfg = getattr(self.config, "effort_tier_medium", None)
            ext_cfg = getattr(self.config, "effort_tier_extra", None)

            low_min = low_cfg.min_sources if low_cfg else 0
            low_max = low_cfg.max_sources if low_cfg else 1
            med_min = med_cfg.min_sources if med_cfg else 1
            med_max = med_cfg.max_sources if med_cfg else 3
            ext_min = ext_cfg.min_sources if ext_cfg else 3
            ext_max = ext_cfg.max_sources if ext_cfg else 7

            header = Text()
            header.append("Effort Tiers & Kuota Sumber Jurnal\n", style="bold bright_white")
            header.append(f"Effort Aktif: {cur_eff.upper()} · Atur batas minimal & maksimal jurnal untuk tiap tingkat pencarian.", style="dim")

            def _mark(t: str) -> str:
                return " [bold green](Aktif)[/bold green]" if cur_eff.lower() == t else ""

            opts = [
                ("low", f"Low Effort Tier    : Min {low_min} s/d Max {low_max} jurnal (Deduktif/QA ringkas){_mark('low')}"),
                ("medium", f"Medium Effort Tier : Min {med_min} s/d Max {med_max} jurnal (QA Faktual/Standar){_mark('medium')}"),
                ("extra", f"Extra Effort Tier  : Min {ext_min} s/d Max {ext_max} jurnal (Tinjauan Pustaka/Komparasi){_mark('extra')}"),
                ("switch_active", f"Ganti Effort Aktif : {cur_eff.upper()} (Tekan untuk mengubah default)"),
                ("back", "Back to Settings"),
            ]

            ch = interactive_menu(opts, header=header, console=self.console, allow_back=True)
            if ch in ("back", None):
                return

            if ch == "switch_active":
                sw_opts = [
                    ("low", f"LOW    - Min {low_min} s/d Max {low_max} jurnal (Eksplorasi cepat & deduktif)"),
                    ("medium", f"MEDIUM - Min {med_min} s/d Max {med_max} jurnal (Standar empiris terarah)"),
                    ("extra", f"EXTRA  - Min {ext_min} s/d Max {ext_max} jurnal (Riset komparatif & literatur mendalam)"),
                    ("back", "Back"),
                ]
                sw_ch = interactive_menu(sw_opts, header="Pilih Default Effort Tier:", console=self.console, allow_back=True)
                if sw_ch and sw_ch != "back":
                    if self.current_project:
                        self.project_manager.set_project_effort(self.current_project.name, sw_ch)
                    self.config_manager.update(default_effort_tier=sw_ch)
                    self.config = self.config_manager.config
            elif ch in ("low", "medium", "extra"):
                self._configure_single_tier(ch)

    def _configure_single_tier(self, tier: str) -> None:
        """Sub-menu to configure min and max sources for a specific effort tier."""
        tier_names = {
            "low": "LOW Effort (Penalaran Konseptual & Deduktif)",
            "medium": "MEDIUM Effort (QA Faktual & Riset Standar)",
            "extra": "EXTRA Effort (Tinjauan Pustaka Komprehensif)",
        }
        title = tier_names.get(tier, f"{tier.upper()} Effort")

        while True:
            t_cfg = getattr(self.config, f"effort_tier_{tier}", None)
            cur_min = t_cfg.min_sources if t_cfg else (0 if tier == "low" else (1 if tier == "medium" else 3))
            cur_max = t_cfg.max_sources if t_cfg else (1 if tier == "low" else (3 if tier == "medium" else 7))

            header = Text()
            header.append(f"Konfigurasi Kuota: {title}\n", style="bold bright_white")
            header.append(f"Saat ini: Min {cur_min} s/d Max {cur_max} jurnal", style="dim")

            if tier == "low":
                presets = [
                    ("p1", "Preset: Min 0 s/d Max 1 jurnal (Default Scripcy)"),
                    ("p2", "Preset: Min 0 s/d Max 2 jurnal (Fleksibel)"),
                    ("p3", "Preset: Min 1 s/d Max 2 jurnal (Wajib minimal 1 paper)"),
                ]
            elif tier == "medium":
                presets = [
                    ("p1", "Preset: Min 1 s/d Max 3 jurnal (Default Standar)"),
                    ("p2", "Preset: Min 2 s/d Max 4 jurnal (Lebih komprehensif)"),
                    ("p3", "Preset: Min 2 s/d Max 5 jurnal (Multi-source kuat)"),
                ]
            else:
                presets = [
                    ("p1", "Preset: Min 3 s/d Max 7 jurnal (Default Extra)"),
                    ("p2", "Preset: Min 4 s/d Max 8 jurnal (Mendalam)"),
                    ("p3", "Preset: Min 5 s/d Max 10 jurnal (Ekstensif)"),
                ]

            opts = [
                ("min", f"Ubah Batas Minimal Jurnal (Saat ini: {cur_min})"),
                ("max", f"Ubah Batas Maksimal Jurnal (Saat ini: {cur_max})"),
                *presets,
                ("back", "Back"),
            ]

            sub_ch = interactive_menu(opts, header=header, console=self.console, allow_back=True)
            if sub_ch in ("back", None):
                return

            if sub_ch == "min":
                val_str = interactive_input_prompt(
                    console=self.console,
                    prompt_label=f"Batas Minimal Jurnal ({tier.upper()}) > ",
                    initial_buffer=str(cur_min),
                ).strip()
                if val_str.isdigit():
                    new_min = int(val_str)
                    new_max = max(cur_max, new_min)
                    self.config_manager.update_effort_tier(tier, min_sources=new_min, max_sources=new_max)
                    self.config = self.config_manager.config
                    if hasattr(self, "engine") and self.engine:
                        self.engine.config = self.config
                    if hasattr(self, "contract_extractor") and self.contract_extractor:
                        self.contract_extractor.engine = self.engine
            elif sub_ch == "max":
                val_str = interactive_input_prompt(
                    console=self.console,
                    prompt_label=f"Batas Maksimal Jurnal ({tier.upper()}) > ",
                    initial_buffer=str(cur_max),
                ).strip()
                if val_str.isdigit():
                    new_max = int(val_str)
                    new_min = min(cur_min, new_max)
                    self.config_manager.update_effort_tier(tier, min_sources=new_min, max_sources=new_max)
                    self.config = self.config_manager.config
                    if hasattr(self, "engine") and self.engine:
                        self.engine.config = self.config
                    if hasattr(self, "contract_extractor") and self.contract_extractor:
                        self.contract_extractor.engine = self.engine
            elif sub_ch == "p1":
                p_min, p_max = (0, 1) if tier == "low" else ((1, 3) if tier == "medium" else (3, 7))
                self.config_manager.update_effort_tier(tier, min_sources=p_min, max_sources=p_max)
                self.config = self.config_manager.config
                if hasattr(self, "engine") and self.engine:
                    self.engine.config = self.config
                if hasattr(self, "contract_extractor") and self.contract_extractor:
                    self.contract_extractor.engine = self.engine
            elif sub_ch == "p2":
                p_min, p_max = (0, 2) if tier == "low" else ((2, 4) if tier == "medium" else (4, 8))
                self.config_manager.update_effort_tier(tier, min_sources=p_min, max_sources=p_max)
                self.config = self.config_manager.config
                if hasattr(self, "engine") and self.engine:
                    self.engine.config = self.config
                if hasattr(self, "contract_extractor") and self.contract_extractor:
                    self.contract_extractor.engine = self.engine
            elif sub_ch == "p3":
                p_min, p_max = (1, 2) if tier == "low" else ((2, 5) if tier == "medium" else (5, 10))
                self.config_manager.update_effort_tier(tier, min_sources=p_min, max_sources=p_max)
                self.config = self.config_manager.config
                if hasattr(self, "engine") and self.engine:
                    self.engine.config = self.config
                if hasattr(self, "contract_extractor") and self.contract_extractor:
                    self.contract_extractor.engine = self.engine

    def _run_interactive_agent_limits_settings(self) -> None:
        """Manage agent limits: max agent turns, refinement turns, output tokens."""
        while True:
            header = Text()
            header.append("Agent Limits & Execution Bounds\n", style="bold bright_white")
            opts = [
                ("turns", f"Max Agent Turns : {self.config.max_agent_turns}"),
                ("tokens", f"Max Output Tokens : {self.config.max_output_tokens}"),
                ("back", "Back"),
            ]
            ch = interactive_menu(opts, header=header, console=self.console, allow_back=True)
            if ch in ("back", None):
                return
            if ch == "turns":
                turns_opts = [("15", "15 turns"), ("25", "25 turns"), ("35", "35 turns"), ("back", "Back")]
                t_ch = interactive_menu(turns_opts, header=header, console=self.console, allow_back=True)
                if t_ch and t_ch.isdigit():
                    self.config.max_agent_turns = int(t_ch)
                    self.config_manager.save()
            elif ch == "tokens":
                tok_opts = [("4096", "4k tokens"), ("8192", "8k tokens"), ("16384", "16k tokens"), ("back", "Back")]
                t_ch = interactive_menu(tok_opts, header=header, console=self.console, allow_back=True)
                if t_ch and t_ch.isdigit():
                    self.config.max_output_tokens = int(t_ch)
                    self.config_manager.save()

    def _run_interactive_keys_settings(self) -> None:
        """Manage API keys for LLM and Academic Search providers."""
        while True:
            header = Text()
            header.append("API Keys & Credentials (BYOK)\n", style="bold bright_white")
            header.append("Disimpan secara lokal di mesin Anda (~/.scripcy/config.json).", style="dim")

            providers = [
                ("gemini", "Google Gemini"),
                ("anthropic", "Anthropic Claude"),
                ("openai", "OpenAI"),
                ("deepseek", "DeepSeek"),
                ("groq", "Groq"),
                ("9router", "9Router Gateway"),
                ("openalex", "OpenAlex"),
                ("semantic_scholar", "Semantic Scholar"),
                ("opencitations", "OpenCitations"),
            ]

            opts = []
            for p_id, p_name in providers:
                val = self.config.get_api_key(p_id)
                if val:
                    masked = f"{val[:3]}...{val[-4:]}" if len(val) > 8 else "***"
                    status = f"[bold cyan]Configured ({masked})[/bold cyan]"
                else:
                    status = "[dim]Not Set[/dim]"
                opts.append((p_id, f"{p_name:<20} : {status}"))
            opts.append(("back", "Back to Settings"))

            ch = interactive_menu(opts, header=header, console=self.console, allow_back=True)
            if ch in ("back", None):
                return

            prov_name = dict(providers).get(ch, ch)
            prompt_str = f"API Key {prov_name} (kosongkan untuk batal, 'clear' untuk hapus) > "
            new_key = interactive_input_prompt(
                console=self.console,
                prompt_label=prompt_str,
                initial_buffer="",
            ).strip()

            if new_key.lower() == "clear":
                self.config.set_api_key(ch, "")
                self.config.save()
            elif new_key:
                self.config.set_api_key(ch, new_key)
                self.config.save()

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
        elif cmd in ("skills", "skill", "/skills", "/skill"):
            self.console.print("[yellow][!] Perintah /skill telah ditiadakan.[/yellow]")
            self.console.print("[dim]Scripcy kini beroperasi secara terintegrasi menggunakan Autonomous Multi-Agent Academic System.[/dim]\n")
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
        table.add_row("Vault Resources", f"{vault_count} documents in vault")
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
            default_dir = (self.project_manager.default_projects_dir / clean_name).resolve()

            self.console.print()
            self.console.print("[bold bright_white]Pilih lokasi penyimpanan master project:[/bold bright_white]")
            self.console.print(f"  [bold cyan][1][/bold cyan] Folder projects/ (default) : [dim]{default_dir}[/dim]")
            self.console.print(f"  [bold cyan][2][/bold cyan] Folder lain / kustom       : [dim]Tentukan path sendiri (misal: D:\\Riset\\{clean_name})[/dim]")
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
                # Default option: dedicated projects/ directory
                custom_path = str(default_dir)

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
        self.console.print(f"    Resources in Vault: [white]{cached_count} indexed[/white]\n")

    def dashboard_delete_project(self, args: list[str]) -> None:
        """Remove a project from the Scripcy registry without deleting physical files."""
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
                f"Are you sure you want to remove project '[bold cyan]{project_info.name}[/bold cyan]' from Scripcy? (y/N): "
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

        with self.console.status(f"[bold cyan]Menghapus project '{project_info.name}' dari registry...[/bold cyan]"):
            removed = self.project_manager.remove_project(project_info.name)
        if removed:
            self.console.print(f"[bold green][✓] Project '{project_info.name}' removed from Scripcy.[/bold green]")
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
        """Remove project from Scripcy registry, closing it if currently active."""
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
                f"Are you sure you want to remove {prompt_label} from Scripcy? (y/N): "
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

        with self.console.status(f"[bold cyan]Menghapus project '{project_info.name}' dari registry...[/bold cyan]"):
            removed = self.project_manager.remove_project(project_info.name)
        if removed:
            self.console.print(f"[bold green][✓] Project '{project_info.name}' removed from Scripcy.[/bold green]")
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

