"""Comprehensive End-to-End Headless Vault Test Suite.

Tests all Vault features via the CLI headless interface:
1. vault add (URL & local file with context)
2. vault update (nested folder context staging in vault/local/, duplicate handling, cleanup)
3. vault list (human-readable table & JSON, category filter)
4. vault search (keyword & hybrid semantic)
5. vault status (health check & JSON)
6. vault history (audit logging & JSON)
7. vault remove (archiving) & vault archived
8. vault restore (restoring from archive)
9. vault reindex (visual & table vector reindexing)
10. vault tables & vault images (inventory inspection)
"""

import json
from pathlib import Path
from unittest.mock import patch
import pytest
from typer.testing import CliRunner

from scripcy.cli import app
from scripcy.core.project import ProjectManager
from scripcy.core.vault import VaultManager
from scripcy.core.workspace import WorkspaceManager

runner = CliRunner()

# Valid 1x1 RGBA PNG bytes
VALID_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15c4\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xa74d\xda\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.fixture
def e2e_vault_env(tmp_path: Path):
    """Setup an isolated workspace and project for headless testing."""
    reg_file = tmp_path / "registry.json"
    pm = ProjectManager(registry_file=reg_file)
    proj_dir = tmp_path / "headless_experiment_vault"
    proj_dir.mkdir(parents=True, exist_ok=True)
    proj = pm.create_project("e2e_headless_proj", proj_dir)
    ws = WorkspaceManager(proj_dir)
    vm = VaultManager(workspace=ws)
    return pm, proj, ws, vm


def test_e2e_full_headless_vault_workflow(e2e_vault_env, tmp_path: Path):
    pm, proj, ws, vm = e2e_vault_env
    proj_name = proj.name

    with patch("scripcy.cli_vault.ProjectManager", return_value=pm):
        # ---------------------------------------------------------------------
        # 1. TEST VAULT ADD (LOCAL FILE WITH CONTEXT)
        # ---------------------------------------------------------------------
        local_paper = tmp_path / "quantum_annealing.txt"
        local_paper.write_text(
            "Title: Quantum Annealing Benchmarks\n"
            "Authors: Alice Walker, Bob Stone\n"
            "Abstract: Systematic study of thermal fluctuations in quantum annealing systems.\n",
            encoding="utf-8",
        )

        res_add_local = runner.invoke(app, [
            "vault", "add", str(local_paper),
            "-p", proj_name,
            "--context", "Percobaan Fisika Kuantum 2026",
            "--json",
        ])
        assert res_add_local.exit_code == 0
        data_add = json.loads(res_add_local.output)
        assert data_add["status"] == "success"
        assert data_add["filename"] == "quantum_annealing.txt"
        assert data_add["context"] == "Percobaan Fisika Kuantum 2026"

        # ---------------------------------------------------------------------
        # 2. TEST VAULT ADD (URL DOWNLOAD)
        # ---------------------------------------------------------------------
        # Downloading official Iris sample dataset CSV directly via URL
        iris_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        res_add_url = runner.invoke(app, [
            "vault", "add", iris_url,
            "-p", proj_name,
            "--context", "Dataset Pembanding Bunga Iris",
            "--json",
        ])
        assert res_add_url.exit_code == 0
        data_url = json.loads(res_add_url.output)
        assert data_url["status"] == "success"
        assert "iris.csv" in data_url["filename"].lower()

        # ---------------------------------------------------------------------
        # 3. TEST LOCAL STAGING INGESTION (vault/local/ with subfolders)
        # ---------------------------------------------------------------------
        # Create nested experiment structure
        exp1_dir = vm.local_dir / "percobaan1" / "suhu_50c"
        exp1_dir.mkdir(parents=True, exist_ok=True)

        exp1_csv = exp1_dir / "pengukuran_sensor.csv"
        exp1_csv.write_text("waktu_detik,suhu_celsius,tekanan_kpa\n10,50.1,101.3\n20,50.3,101.4\n", encoding="utf-8")

        exp1_img = exp1_dir / "grafik_distribusi.png"
        exp1_img.write_bytes(VALID_PNG_BYTES)

        exp2_dir = vm.local_dir / "percobaan2"
        exp2_dir.mkdir(parents=True, exist_ok=True)
        exp2_txt = exp2_dir / "catatan_reaksi.txt"
        exp2_txt.write_text("Title: Catatan Observasi Reaksi Kimia\nKatalisator ditambahkan.", encoding="utf-8")

        # Run vault update with duplicate handling
        res_update = runner.invoke(app, [
            "vault", "update",
            "-p", proj_name,
            "--on-duplicate", "keep-both",
            "--json",
        ])
        assert res_update.exit_code == 0
        data_update = json.loads(res_update.output)
        assert data_update["status"] == "success"
        assert len(data_update["local_processed"]) == 3
        assert "pengukuran_sensor.csv" in data_update["local_processed"]
        assert "grafik_distribusi.png" in data_update["local_processed"]
        assert "catatan_reaksi.txt" in data_update["local_processed"]

        # Verify staging folders were cleaned up
        assert not exp1_csv.exists()
        assert not exp1_img.exists()
        assert not exp1_dir.exists()

        # ---------------------------------------------------------------------
        # 4. TEST VAULT LIST (Categories & Context Verification)
        # ---------------------------------------------------------------------
        # List all
        res_list = runner.invoke(app, ["vault", "list", "-p", proj_name, "--json"])
        assert res_list.exit_code == 0
        all_items = json.loads(res_list.output)
        filenames = [it["filename"] for it in all_items]
        assert "quantum_annealing.txt" in filenames
        assert "catatan_reaksi.txt" in filenames

        # Verify folder context was inherited automatically
        item_reaksi = next(it for it in all_items if it["filename"] == "catatan_reaksi.txt")
        assert "percobaan2" in item_reaksi["context"]

        # List tables category
        res_list_tables = runner.invoke(app, ["vault", "list", "-p", proj_name, "-c", "tables", "--json"])
        assert res_list_tables.exit_code == 0
        tbl_items = json.loads(res_list_tables.output)
        tbl_names = [t["filename"] for t in tbl_items]
        assert any("pengukuran_sensor" in n or "iris" in n for n in tbl_names)

        # List human-readable table (verify Rich rendering doesn't crash)
        res_list_table_ui = runner.invoke(app, ["vault", "list", "-p", proj_name])
        assert res_list_table_ui.exit_code == 0
        assert "Daftar Berkas Vault" in res_list_table_ui.output

        # ---------------------------------------------------------------------
        # 5. TEST VAULT TABLES & IMAGES DIRECT INVENTORY
        # ---------------------------------------------------------------------
        res_tables = runner.invoke(app, ["vault", "tables", "-p", proj_name, "--json"])
        assert res_tables.exit_code == 0
        tables_data = json.loads(res_tables.output)
        assert isinstance(tables_data, list)
        assert len(tables_data) >= 1

        res_images = runner.invoke(app, ["vault", "images", "-p", proj_name, "--json"])
        assert res_images.exit_code == 0
        images_data = json.loads(res_images.output)
        assert isinstance(images_data, list)
        assert len(images_data) >= 1

        # ---------------------------------------------------------------------
        # 6. TEST VAULT SEARCH (Keyword and Semantic Hybrid Vector)
        # ---------------------------------------------------------------------
        # Keyword search
        res_search_kw = runner.invoke(app, ["vault", "search", "quantum", "-p", proj_name, "--json"])
        assert res_search_kw.exit_code == 0
        kw_matches = json.loads(res_search_kw.output)
        assert len(kw_matches) >= 1
        assert "quantum" in kw_matches[0]["title"].lower() or "quantum" in kw_matches[0]["id"].lower()

        # Semantic hybrid search
        res_search_sem = runner.invoke(app, ["vault", "search", "thermal fluctuations", "-p", proj_name, "--semantic", "--json"])
        assert res_search_sem.exit_code == 0
        sem_matches = json.loads(res_search_sem.output)
        assert isinstance(sem_matches, list)

        # ---------------------------------------------------------------------
        # 7. TEST VAULT STATUS & HEALTH
        # ---------------------------------------------------------------------
        res_status = runner.invoke(app, ["vault", "status", "-p", proj_name, "--json"])
        assert res_status.exit_code == 0
        status_data = json.loads(res_status.output)
        assert status_data["status"] in ("active", "healthy")
        assert status_data["total_files"] >= 4
        assert status_data.get("error_count", 0) == 0
        assert status_data["last_updated"] != "-"

        # ---------------------------------------------------------------------
        # 8. TEST VAULT HISTORY
        # ---------------------------------------------------------------------
        res_hist = runner.invoke(app, ["vault", "history", "-p", proj_name, "--json"])
        assert res_hist.exit_code == 0
        history_data = json.loads(res_hist.output)
        assert len(history_data) >= 2
        actions = [h.get("action") for h in history_data]
        assert "tambah_berkas" in actions or "sortir_mandiri" in actions

        # ---------------------------------------------------------------------
        # 9. TEST ARCHIVE & RESTORE WORKFLOW
        # ---------------------------------------------------------------------
        # Archive quantum_annealing.txt
        res_rem = runner.invoke(app, ["vault", "remove", "quantum_annealing.txt", "-p", proj_name, "--json"])
        assert res_rem.exit_code == 0
        data_rem = json.loads(res_rem.output)
        assert data_rem["status"] == "archived"

        # Verify in archived list
        res_arch = runner.invoke(app, ["vault", "archived", "-p", proj_name, "--json"])
        assert res_arch.exit_code == 0
        arch_items = json.loads(res_arch.output)
        assert any(a["filename"] == "quantum_annealing.txt" for a in arch_items)

        # Verify removed from active list
        res_list_after_rem = runner.invoke(app, ["vault", "list", "-p", proj_name, "--json"])
        active_filenames = [it["filename"] for it in json.loads(res_list_after_rem.output)]
        assert "quantum_annealing.txt" not in active_filenames

        # Restore quantum_annealing.txt
        res_restore = runner.invoke(app, ["vault", "restore", "quantum_annealing.txt", "-p", proj_name, "--json"])
        assert res_restore.exit_code == 0
        data_restore = json.loads(res_restore.output)
        assert data_restore["status"] == "restored"

        # Verify back in active list
        res_list_after_res = runner.invoke(app, ["vault", "list", "-p", proj_name, "--json"])
        restored_filenames = [it["filename"] for it in json.loads(res_list_after_res.output)]
        assert "quantum_annealing.txt" in restored_filenames

        # ---------------------------------------------------------------------
        # 10. TEST REINDEX
        # ---------------------------------------------------------------------
        res_reindex = runner.invoke(app, ["vault", "reindex", "-p", proj_name, "--json"])
        assert res_reindex.exit_code == 0
        reindex_data = json.loads(res_reindex.output)
        assert reindex_data["status"] == "success"
        assert "counts" in reindex_data
