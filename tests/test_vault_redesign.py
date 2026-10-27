"""Unit tests for the redesigned Vault with dynamic health status, local staging, and categorization."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from scripcy.cli_studio import CLIStudio
from scripcy.core.config import ScripcyConfig
from scripcy.core.project import ProjectManager
from scripcy.core.vault import VaultManager
from scripcy.core.workspace import WorkspaceManager


@pytest.fixture
def test_vault_workspace(tmp_path: Path) -> WorkspaceManager:
    return WorkspaceManager(root_dir=tmp_path / "study_project")


def test_vault_health_statuses(test_vault_workspace: WorkspaceManager) -> None:
    vm = VaultManager(workspace=test_vault_workspace)
    assert vm.local_dir.is_dir()

    # 1. Clean vault -> Active
    health_clean = vm.check_vault_health()
    assert health_clean["status"] == "active"
    assert "Aktif" in health_clean["status_display"]
    assert health_clean["total_files"] == 0
    assert len(health_clean["warning_details"]) == 0
    assert len(health_clean["error_details"]) == 0

    # 2. File in local staging needs update -> need_update
    (vm.local_dir / "draft.txt").write_text("Hello draft", encoding="utf-8")
    health_update = vm.check_vault_health()
    assert health_update["status"] == "need_update"
    assert "Perlu Update" in health_update["status_display"]
    assert health_update["total_files"] == 1
    assert "belum di-update" in health_update["warning_details"][0]

    # Convert to markdown -> Active
    vm.reload_vault()
    assert vm.check_vault_health()["status"] == "active"

    # 3. Duplicate file -> warning
    (vm.papers_dir / "draft.txt").write_text("Duplicate in papers", encoding="utf-8")
    vm.reload_vault()
    health_dup = vm.check_vault_health()
    assert health_dup["status"] == "warning"
    assert "Peringatan" in health_dup["status_display"]
    assert "duplikat" in health_dup["warning_details"][0]

    # 4. Unsupported file -> error (takes highest precedence)
    (vm.local_dir / "malicious.exe").write_bytes(b"MZ123")
    health_err = vm.check_vault_health()
    assert health_err["status"] == "error"
    assert "Error" in health_err["status_display"]
    assert "tidak didukung" in health_err["error_details"][0]


def test_vault_category_health_breakdown(test_vault_workspace: WorkspaceManager) -> None:
    vm = VaultManager(workspace=test_vault_workspace)

    # Put a paper in papers/
    (vm.papers_dir / "paper1.txt").write_text("Academic content", encoding="utf-8")
    # Put a web file in web/
    (vm.web_dir / "article1.html").write_text("<html>Article</html>", encoding="utf-8")
    # Put an unsupported file in papers/
    (vm.papers_dir / "paper_bad.bin").write_bytes(b"binary")

    health = vm.check_vault_health()
    cat_h = health["category_health"]

    assert cat_h["papers"]["total"] == 2
    assert cat_h["papers"]["error"] == 1
    assert cat_h["web"]["total"] == 1
    assert cat_h["web"]["error"] == 0


def test_render_vault_landing_minimalist(test_vault_workspace: WorkspaceManager, tmp_path: Path) -> None:
    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "reg.json")
    proj = pm.create_project("test_proj", test_vault_workspace.root_dir)
    cfg = ScripcyConfig(workspace_path=str(test_vault_workspace.root_dir))

    studio = CLIStudio(
        workspace=test_vault_workspace,
        config=cfg,
        console=console,
        project_manager=pm,
    )
    studio.open_project(proj)

    studio.render_vault_landing()
    out = console.file.getvalue()

    # Must contain Status and Total Berkas
    assert "Status" in out
    assert "Total Berkas" in out
    assert "Aktif" in out

    # Must NOT contain Folder paths, diarsipkan, or format documents per user request
    assert "Folder Vault" not in out
    assert "Folder Local" not in out
    assert "Diarsipkan" not in out
    assert "Format Dokumen" not in out


def test_vault_add_menu_labels(test_vault_workspace: WorkspaceManager, tmp_path: Path) -> None:
    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "reg.json")
    proj = pm.create_project("test_proj", test_vault_workspace.root_dir)
    cfg = ScripcyConfig(workspace_path=str(test_vault_workspace.root_dir))

    studio = CLIStudio(
        workspace=test_vault_workspace,
        config=cfg,
        console=console,
        project_manager=pm,
    )
    studio.open_project(proj)
    studio.sub_mode = "vault"

    captured_opts = []

    def mock_interactive_menu(options, **kwargs):
        nonlocal captured_opts
        captured_opts = options
        return "back"

    with patch("scripcy.studio.vault.interactive_menu", side_effect=mock_interactive_menu):
        studio.run_interactive_vault_add()

    labels = [lbl for _, lbl in captured_opts]
    assert any("Cari Journal Ilmiah" in l for l in labels)
    assert any("Cari Website" in l for l in labels)
    assert any("Tambah Dari Google Drive" in l for l in labels)
    assert any("Berkas Lokal" in l for l in labels)


def test_vault_last_updated_and_metadata_counts(test_vault_workspace: WorkspaceManager) -> None:
    vm = VaultManager(workspace=test_vault_workspace)
    # Check default last updated on clean vault
    h_clean = vm.check_vault_health()
    assert h_clean["last_updated"] == "-"

    # Add a file
    p_file = vm.papers_dir / "paper_test.txt"
    p_file.write_text("Title: Test Paper\nContent", encoding="utf-8")

    h_with_file = vm.check_vault_health()
    assert h_with_file["last_updated"] != "-"

    files = vm.list_active_files()
    assert len(files) == 1
    item = files[0]
    assert item.updated_at != ""
    assert item.tables_count == 0
    assert item.visuals_count == 0


def test_workspace_get_cached_document(test_vault_workspace: WorkspaceManager) -> None:
    wm = test_vault_workspace
    doc_path = wm.save_to_cache("sample.md", "# Heading\nSample body")
    assert doc_path.exists()

    # Query with stem
    found = wm.get_cached_document("sample")
    assert found is not None
    assert found.exists()
    assert found.name == "sample.md"

    # Query with full filename
    found2 = wm.get_cached_document("sample.md")
    assert found2 is not None
    assert found2.exists()


def test_vault_search_category_menu(test_vault_workspace: WorkspaceManager, tmp_path: Path) -> None:
    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "reg.json")
    proj = pm.create_project("test_proj", test_vault_workspace.root_dir)
    cfg = ScripcyConfig(workspace_path=str(test_vault_workspace.root_dir))

    studio = CLIStudio(
        workspace=test_vault_workspace,
        config=cfg,
        console=console,
        project_manager=pm,
    )
    studio.open_project(proj)
    studio.sub_mode = "vault"

    captured_opts = []

    def mock_interactive_menu(options, **kwargs):
        nonlocal captured_opts
        captured_opts = options
        return "back"

    with patch("scripcy.studio.vault.interactive_menu", side_effect=mock_interactive_menu):
        studio.run_interactive_vault_categories()

    keys = [k for k, _ in captured_opts]
    labels = [lbl for _, lbl in captured_opts]

    # Must contain 'search' (Cari Berkas) and NOT 'all' (Semua Berkas)
    assert "search" in keys
    assert "all" not in keys
    assert any("Cari Berkas" in l for l in labels)


def test_vault_search_keyword_input_with_default(test_vault_workspace: WorkspaceManager, tmp_path: Path) -> None:
    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "reg.json")
    proj = pm.create_project("test_proj", test_vault_workspace.root_dir)
    cfg = ScripcyConfig(workspace_path=str(test_vault_workspace.root_dir))

    # Add a dummy file to vault so list_active_files() is not empty
    vm = VaultManager(workspace=test_vault_workspace)
    dummy_paper = vm.papers_dir / "climate_study.txt"
    dummy_paper.write_text("Title: Climate Study\nAbstract: Data on forests.", encoding="utf-8")

    studio = CLIStudio(
        workspace=test_vault_workspace,
        config=cfg,
        console=console,
        project_manager=pm,
    )
    studio.open_project(proj)
    studio.sub_mode = "vault"

    # Sequence of actions in search table: first 'search', then 'back'
    action_sequence = [("search", 0, None), ("back", 0, None)]

    def mock_table_browser(*args, **kwargs):
        if action_sequence:
            return action_sequence.pop(0)
        return ("back", 0, None)

    with patch("scripcy.studio.vault.interactive_item_browser", side_effect=mock_table_browser), \
         patch("scripcy.studio.vault.interactive_input_prompt", return_value="climate change") as mock_prompt:
        studio.run_interactive_vault_search()
        mock_prompt.assert_called_once()
        _, kw = mock_prompt.call_args
        assert kw.get("default") == "" or kw.get("initial_buffer") == ""


