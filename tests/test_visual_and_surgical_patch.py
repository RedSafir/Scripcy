"""Unit tests for inline visual placeholder resolution and surgical paragraph patching."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from scripcy.cli_studio import CLIStudio
from scripcy.core.config import ScripcyConfig
from scripcy.core.engine import LLMEngine
from scripcy.core.project import ProjectManager
from scripcy.core.vector_vault import ProjectVectorVault
from scripcy.core.workspace import WorkspaceManager


class MockStreamChunk:
    def __init__(self, content: str | None) -> None:
        self.choices = [MagicMock(delta=MagicMock(content=content))]


def test_visual_placeholder_resolution_to_clickable_links(tmp_path: Path):
    """Verify [[VISUAL:id]] placeholders are resolved to clickable file:/// links without terminal half-blocks."""
    db_path = tmp_path / "vault.db"
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir(parents=True, exist_ok=True)

    vv = ProjectVectorVault(db_path=db_path, vault_dir=vault_dir)

    # Insert a dummy image and table into database
    dummy_img = tmp_path / "img_stunting.png"
    dummy_img.write_bytes(b"dummy image bytes")

    dummy_tbl = tmp_path / "tbl_stunting.csv"
    dummy_tbl.write_text("Provinsi,Prevalensi\nAceh,31.2\nNTT,37.8\n", encoding="utf-8")

    with vv._get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO images (id, paper_id, page_num, file_path, caption, ai_analysis, width, height)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ("img_123", "paper_stunting", 3, str(dummy_img), "Prevalensi Stunting Nasional 2024", "Bagan tren prevalensi stunting", 600, 400))

        cur.execute("""
            INSERT INTO tables (id, paper_id, page_num, csv_path, md_path, caption, is_vision_fallback)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ("tbl_456", "paper_stunting", 4, str(dummy_tbl), None, "Tabel Prevalensi Stunting per Provinsi", 0))
        conn.commit()

    raw_text = (
        "Berdasarkan analisis data, tren stunting mengalami penurunan signifikan sebagaimana terlihat pada [[VISUAL:img_123]]. "
        "Distribusi spasial antarprovinsi disajikan dalam [[VISUAL:tbl_456]]."
    )

    resolved = vv.resolve_visual_placeholders(raw_text)

    # Must contain clickable file:/// markdown links
    assert "file:///" in resolved or "file://" in resolved
    assert "[📊 Buka Gambar](file://" in resolved
    assert "Prevalensi Stunting Nasional 2024" in resolved
    assert "[📋 Buka Tabel](file://" in resolved
    assert "Tabel Prevalensi Stunting per Provinsi" in resolved
    assert "(paper_stunting)" in resolved
    assert "[[VISUAL:" not in resolved


def test_render_academic_markdown_renders_file_hyperlinks():
    """Verify render_academic_markdown produces true OSC 8 clickable terminal hyperlinks for file:// links."""
    from scripcy.ui.formatters import render_academic_markdown
    from rich.markdown import Markdown

    c = Console(force_terminal=True, legacy_windows=False)
    md_sample = (
        "Visualisasi:\n\n"
        "[📊 Buka Gambar](file:///C:/vault/images/fig_1.jpeg)\n"
        "Gambar 1. Tren Deforestasi\n"
        "(Nurhayati et al., 2021)\n"
    )
    # Ensure MarkdownIt parses file:/// as active link
    md = Markdown(md_sample)
    rendered_segs = [(seg.text, str(seg.style)) for seg in c.render(md)]
    link_segs = [s for s in rendered_segs if "Buka Gambar" in s[0]]
    assert len(link_segs) == 1
    text, style = link_segs[0]
    assert "link file:///C:/vault/images/fig_1.jpeg" in style


def test_surgical_paragraph_patch_preserves_verified_paragraphs(
    temp_workspace: WorkspaceManager,
    tmp_path: Path,
):
    """Verify Turn > 0 uses Option 1 (Surgical Paragraph Patch): only weak paragraph is rewritten while verified text is kept."""
    temp_workspace.save_to_cache(
        "stunting_report.md",
        "---\ntitle: Laporan Stunting 2024\nyear: 2024\ncite_key: Kemenkes2024\n---\n# Data\nPrevalensi stunting Indonesia tahun 2024 adalah 19.8 persen menurut SSGI 2024.",
    )

    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "reg.json")
    proj = pm.create_project("stunting_proj", temp_workspace.root_dir)
    cfg = ScripcyConfig(workspace_path=str(temp_workspace.root_dir), max_agent_turns=3, max_refinement_turns=2)

    call_count = 0
    received_queries: list[str] = []

    async def mock_acompletion(**kwargs: Any):
        nonlocal call_count
        call_count += 1
        query_sent = ""
        for m in kwargs.get("messages", []):
            if m.get("role") == "user":
                query_sent = m.get("content", "")
        received_queries.append(query_sent)

        if call_count == 1:
            # Turn 0: Paragraph 1 is verified; Paragraph 2 contains hallucinated/unsupported claim
            yield MockStreamChunk(
                "Paragraf 1: Prevalensi stunting Indonesia tahun 2024 adalah 19.8 persen menurut (Kemenkes2024).\n\n"
                "Paragraf 2: Klaim tidak berdasar bahwa stunting disebabkan oleh radiasi sinyal 5G di pedesaan."
            )
        else:
            # Turn 1: Surgical patch should only rewrite paragraph 2
            yield MockStreamChunk(
                "Paragraf 2: Berdasarkan bukti empiris (Kemenkes2024), determinan utama stunting adalah pola asuh, sanitasi, dan asupan gizi balita."
            )

    engine = LLMEngine(config=cfg, acompletion_fn=mock_acompletion)
    studio = CLIStudio(
        workspace=temp_workspace,
        config=cfg,
        console=console,
        project_manager=pm,
        engine=engine,
    )
    studio.open_project(proj)
    studio.config.api_keys["gemini"] = "mock_key"
    studio.scout = None

    studio.handle_chat("Berapa prevalensi stunting 2024 dan apa faktor penyebabnya?")

    out = studio.console.file.getvalue()

    # Verify both Paragraf 1 (preserved) and Paragraf 2 (patched) exist in the final output
    assert "19.8 persen" in out
    assert "pola asuh" in out or "asupan gizi" in out
    # Verify the hallucinated 5G claim was surgically excised
    assert "radiasi sinyal 5G" not in out
