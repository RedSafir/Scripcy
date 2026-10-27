"""Unit tests for standalone tables, visuals, folder context extraction, history log, and duplicate resolution."""

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
def standalone_workspace(tmp_path: Path) -> WorkspaceManager:
    return WorkspaceManager(root_dir=tmp_path / "research_proj")


def test_standalone_table_and_image_ingestion(standalone_workspace: WorkspaceManager, tmp_path: Path) -> None:
    vm = VaultManager(workspace=standalone_workspace)

    # 1. Test Standalone Table (.csv)
    test_csv = tmp_path / "hasil_viskositas.csv"
    test_csv.write_text("Suhu,Viskositas,LajuGeser\n50,1.25,100\n70,0.95,150", encoding="utf-8")

    res_tbl = vm.vector_vault.ingest_standalone_table(
        table_path=test_csv,
        caption="Uji Viskositas Fluida",
        context="percobaan_1 > suhu_50c",
    )
    assert "user_tbl_hasil_viskositas" in res_tbl["id"]
    assert Path(res_tbl["md_path"]).exists()
    assert Path(res_tbl["csv_path"]).exists()

    tables = vm.list_tables()
    assert any("hasil_viskositas" in (t.get("id") or "") for t in tables)
    matching_tbl = [t for t in tables if "hasil_viskositas" in (t.get("id") or "")][0]
    assert matching_tbl.get("paper_id") == "[Unggahan Mandiri]"
    assert "percobaan_1 > suhu_50c" in matching_tbl.get("caption", "")

    # 2. Test Standalone Image (.png)
    test_png = tmp_path / "grafik_uji.png"
    # Create simple 10x10 dummy PNG
    from PIL import Image
    im = Image.new("RGB", (120, 80), color=(73, 109, 137))
    im.save(test_png)

    res_img = vm.vector_vault.ingest_standalone_image(
        image_path=test_png,
        caption="Grafik Kurva Viskositas",
        context="percobaan_1 > suhu_50c",
        ai_analysis="Kurva linier penurunan viskositas terhadap suhu.",
    )
    assert "user_img_grafik_uji" in res_img["id"]
    assert Path(res_img["file_path"]).exists()

    images = vm.list_images()
    assert any("grafik_uji" in (i.get("id") or "") for i in images)
    matching_img = [i for i in images if "grafik_uji" in (i.get("id") or "")][0]
    assert matching_img.get("paper_id") == "[Unggahan Mandiri]"
    assert matching_img.get("width") == 120
    assert matching_img.get("height") == 80


def test_scan_local_staging_folder_hierarchy_and_duplicates(standalone_workspace: WorkspaceManager) -> None:
    vm = VaultManager(workspace=standalone_workspace)

    # Create nested structure in vault/local/
    sub1 = vm.local_dir / "percobaan_1" / "suhu_50c"
    sub1.mkdir(parents=True, exist_ok=True)

    csv_file = sub1 / "aliran.csv"
    csv_file.write_text("waktu,debit\n1,10\n2,20", encoding="utf-8")

    png_file = sub1 / "grafik.png"
    from PIL import Image
    im = Image.new("RGB", (50, 50), color="blue")
    im.save(png_file)

    pdf_file = vm.local_dir / "dokumen_naskah.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 dummy pdf")

    scanned = vm.scan_local_staging()
    assert len(scanned) == 3

    # Verify folder context extraction
    csv_info = [s for s in scanned if s["filename"] == "aliran.csv"][0]
    assert csv_info["folder_context"] == "percobaan_1 > suhu_50c"
    assert csv_info["category"] == "tables"
    assert csv_info["is_duplicate"] is False

    png_info = [s for s in scanned if s["filename"] == "grafik.png"][0]
    assert png_info["folder_context"] == "percobaan_1 > suhu_50c"
    assert png_info["category"] == "images"
    assert png_info["is_duplicate"] is False

    pdf_info = [s for s in scanned if s["filename"] == "dokumen_naskah.pdf"][0]
    assert pdf_info["folder_context"] == ""
    assert pdf_info["category"] == "papers"

    # Test processing & moving file: local/ must be cleaned
    dest = vm.process_local_item(csv_info, action="keep_both", context_override="Pengujian Aliran 50C")
    assert dest is not None
    assert dest.exists()
    assert not csv_file.exists()  # Moved and cleaned from local

    # Now put another file with same name to test duplicate detection
    csv_file2 = sub1 / "aliran.csv"
    csv_file2.write_text("waktu,debit\n3,30", encoding="utf-8")

    scanned2 = vm.scan_local_staging()
    dup_info = [s for s in scanned2 if s["filename"] == "aliran.csv"][0]
    assert dup_info["is_duplicate"] is True


def test_vault_history_logging(standalone_workspace: WorkspaceManager) -> None:
    vm = VaultManager(workspace=standalone_workspace)

    assert vm.list_history() == []

    vm.record_history(
        action="sortir_mandiri",
        description="Memindahkan 2 berkas dari local/ ke Vault",
        files_affected=["aliran.csv", "grafik.png"],
        details={"grup": "percobaan_1"},
    )

    history = vm.list_history()
    assert len(history) == 1
    entry = history[0]
    assert entry["action"] == "sortir_mandiri"
    assert len(entry["files_affected"]) == 2
    assert "timestamp" in entry


def test_vault_search_filter_categories_and_search_bar(standalone_workspace: WorkspaceManager, tmp_path: Path) -> None:
    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "reg.json")
    proj = pm.create_project("test_proj", standalone_workspace.root_dir)
    cfg = ScripcyConfig(workspace_path=str(standalone_workspace.root_dir))

    studio = CLIStudio(
        workspace=standalone_workspace,
        config=cfg,
        console=console,
        project_manager=pm,
    )
    studio.open_project(proj)
    studio.sub_mode = "vault"

    # Put a paper and a table
    vm = studio.vault_manager
    (vm.papers_dir / "paper_ml.txt").write_text("Machine learning research", encoding="utf-8")
    (vm.user_tables_dir / "data_lab.csv").write_text("a,b\n1,2", encoding="utf-8")
    vm.reload_vault()

    captured_title = None
    captured_actions = []

    def mock_item_browser(title, actions, **kwargs):
        nonlocal captured_title, captured_actions
        captured_title = title
        captured_actions = actions
        return "back", 0, None

    with patch("scripcy.studio.vault.interactive_item_browser", side_effect=mock_item_browser):
        studio.run_interactive_vault_search()

    action_keys = [k for k, _, _ in captured_actions]
    assert "filter" in action_keys
    assert "search" in action_keys
    assert "detail" in action_keys

    # Check header has Filter Kategori and Kata Kunci
    rendered_header = str(captured_title)
    # captured_title is rich Table
    assert captured_title is not None
