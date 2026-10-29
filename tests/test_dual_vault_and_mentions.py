import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from scripcy.core.workspace import WorkspaceManager
from scripcy.core.vault import VaultManager, VaultItem
from scripcy.core.vector_vault import ProjectVectorVault
from scripcy.core.engine import EvidenceChunk, EvidenceSynthesizer
from scripcy.core.references import resolve_vault_reference_entries, append_vault_bibliography
from scripcy.ui.interactive import interactive_input_prompt, get_mention_suggestions


def test_workspace_directory_structure(tmp_path: Path):
    proj_dir = tmp_path / "my_project"
    ws = WorkspaceManager(proj_dir)
    ws.ensure_directories()

    assert ws.downloads_dir.is_dir()
    assert ws.downloads_dir.name == "Downloads"
    assert ws.user_dir.is_dir()
    assert ws.user_dir.name == "user"
    assert ws.user_input_dir.is_dir()
    assert ws.user_input_dir.name == "USER INPUT HERE"


def test_vault_staging_and_category_split(tmp_path: Path):
    proj_dir = tmp_path / "study_proj"
    ws = WorkspaceManager(proj_dir)
    ws.ensure_directories()
    vm = VaultManager(ws)

    # 1. Put user experiment file in USER INPUT HERE
    user_note = ws.user_input_dir / "experiment_notes.txt"
    user_note.write_text("Catatan eksperimen loss", encoding="utf-8")

    # 2. Put literature paper in vault/Downloads
    paper = ws.downloads_dir / "smith2023.txt"
    paper.write_text("Smith et al. (2023) finding...", encoding="utf-8")

    # Staging scan
    staged = vm.scan_local_staging()
    assert len(staged) == 1
    assert staged[0]["filename"] == "experiment_notes.txt"
    assert staged[0]["category"] == "user"

    # Active items scoped listing
    vm.reload_vault()
    all_items = vm.list_active_files()
    assert len(all_items) >= 1

    download_items = vm.list_active_files(category="downloads")
    user_items = vm.list_active_files(category="user")
    assert any(it.filename == "smith2023.txt" for it in download_items)
    assert not any(it.filename == "smith2023.txt" for it in user_items)


def test_user_category_excluded_from_formal_bibliography(tmp_path: Path):
    proj_dir = tmp_path / "test_ref"
    ws = WorkspaceManager(proj_dir)
    ws.ensure_directories()
    vm = VaultManager(ws)

    # Literature paper
    paper = ws.downloads_dir / "vaswani2017.txt"
    paper.write_text("Attention is all you need.", encoding="utf-8")

    # User note / experimental data in user_dir
    user_data = ws.user_dir / "my_experimental_results.txt"
    user_data.write_text("Experiment results: Accuracy 98%", encoding="utf-8")

    vm.reload_vault()

    # Evidence from both literature and user
    evidence = [
        EvidenceChunk(chunk_id="c1", source="vaswani2017.txt", text="Attention is all you need.", category="papers"),
        EvidenceChunk(chunk_id="c2", source="my_experimental_results.txt", text="Accuracy 98%", category="user"),
    ]

    synth = EvidenceSynthesizer()
    prompt = synth.format_context_prompt(chunks=evidence)

    # Check distinct prompt sections
    assert "### DATA RISET PRIMER PENELITI (FOKUS ANALISIS UTAMA):" in prompt
    assert "my_experimental_results.txt" in prompt
    assert "### BUKTI LITERATUR ILMIAH (RUJUKAN PEMBANDING & SITASI):" in prompt
    assert "vaswani2017.txt" in prompt

    # Check bibliography generation excludes user experimental files
    draft_answer = "Berdasarkan pengujian, akurasi mencapai 98% sesuai dengan temuan (Vaswani, 2017)."
    bib_text = append_vault_bibliography(
        text=draft_answer,
        evidence=evidence,
        vault_manager=vm,
    )
    # vaswani should be cited in bibliography, user experimental result must NOT
    assert "vaswani2017.txt" in bib_text
    assert "my_experimental_results.txt" not in bib_text


def test_at_mention_suggestions(tmp_path: Path):
    # Setup candidate directories
    user_input_dir = tmp_path / "USER INPUT HERE"
    user_input_dir.mkdir(parents=True)
    (user_input_dir / "training_loss.png").write_bytes(b"image")

    downloads_dir = tmp_path / "vault" / "Downloads"
    downloads_dir.mkdir(parents=True)
    (downloads_dir / "bert_paper.pdf").write_bytes(b"pdf")

    # Typing @ should suggest all
    all_sug = get_mention_suggestions("analisis @", base_dir=tmp_path)
    assert len(all_sug) == 2
    names = [s[0] for s in all_sug]
    assert "@training_loss.png" in names
    assert "@bert_paper.pdf" in names

    # Typing @tra should filter to training_loss.png
    filt_sug = get_mention_suggestions("analisis grafik @tra", base_dir=tmp_path)
    assert len(filt_sug) == 1
    assert filt_sug[0][0] == "@training_loss.png"
    assert "PNG" in filt_sug[0][1]
    assert "USER INPUT HERE" in filt_sug[0][1]

