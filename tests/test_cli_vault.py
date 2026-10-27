"""Tests for headless Scripcy Vault CLI commands."""

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from scripcy.cli import app
from scripcy.core.project import ProjectManager
from scripcy.core.vault import VaultManager
from scripcy.core.workspace import WorkspaceManager

runner = CliRunner()


def setup_test_vault_project(tmp_path: Path):
    reg_file = tmp_path / "registry.json"
    pm = ProjectManager(registry_file=reg_file)
    proj_dir = tmp_path / "research_proj"
    proj_dir.mkdir(parents=True, exist_ok=True)
    proj = pm.create_project("my_proj", proj_dir)
    ws = WorkspaceManager(proj_dir)
    vm = VaultManager(workspace=ws)
    return pm, proj, ws, vm


def test_vault_list_empty_and_json(tmp_path: Path):
    pm, proj, ws, vm = setup_test_vault_project(tmp_path)

    with patch("scripcy.cli_vault.ProjectManager", return_value=pm):
        res = runner.invoke(app, ["vault", "list", "-p", "my_proj"])
        assert res.exit_code == 0
        assert "Tidak ada berkas di Vault" in res.output

        res_json = runner.invoke(app, ["vault", "list", "-p", "my_proj", "--json"])
        assert res_json.exit_code == 0
        data = json.loads(res_json.output)
        assert isinstance(data, list)
        assert len(data) == 0


def test_vault_add_local_file_and_list(tmp_path: Path):
    pm, proj, ws, vm = setup_test_vault_project(tmp_path)

    # Create dummy local file
    sample_file = tmp_path / "climate_study.txt"
    sample_file.write_text("Title: Forest and Climate\nAbstract: Effect on trees.", encoding="utf-8")

    with patch("scripcy.cli_vault.ProjectManager", return_value=pm):
        res_add = runner.invoke(app, [
            "vault", "add", str(sample_file),
            "-p", "my_proj",
            "--context", "Percobaan Lapangan A",
            "--json",
        ])
        assert res_add.exit_code == 0
        add_data = json.loads(res_add.output)
        assert add_data["status"] == "success"
        assert add_data["filename"] == "climate_study.txt"

        # List should now show 1 file
        res_list = runner.invoke(app, ["vault", "list", "-p", "my_proj", "--json"])
        assert res_list.exit_code == 0
        list_data = json.loads(res_list.output)
        assert len(list_data) == 1
        assert list_data[0]["filename"] == "climate_study.txt"
        assert list_data[0]["context"] == "Percobaan Lapangan A"


def test_vault_update_from_local_staging_with_folder_context(tmp_path: Path):
    pm, proj, ws, vm = setup_test_vault_project(tmp_path)

    # Place file in vault/local/percobaan1/suhu_50c/
    exp_dir = vm.local_dir / "percobaan1" / "suhu_50c"
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_file = exp_dir / "sensor_log.txt"
    exp_file.write_text("Sensor Log Suhu 50C\nData: 100", encoding="utf-8")

    with patch("scripcy.cli_vault.ProjectManager", return_value=pm):
        res_up = runner.invoke(app, ["vault", "update", "-p", "my_proj", "--json"])
        assert res_up.exit_code == 0
        up_data = json.loads(res_up.output)
        assert up_data["status"] == "success"
        assert "sensor_log.txt" in up_data["local_processed"]

        # Check list to verify folder context was applied
        res_list = runner.invoke(app, ["vault", "list", "-p", "my_proj", "--json"])
        list_data = json.loads(res_list.output)
        item = next(it for it in list_data if it["filename"] == "sensor_log.txt")
        assert "percobaan1 > suhu_50c" in item["context"]


def test_vault_search_keyword_and_semantic(tmp_path: Path):
    pm, proj, ws, vm = setup_test_vault_project(tmp_path)

    p1 = vm.papers_dir / "wildfire_regimes.txt"
    p1.write_text("Title: Wildfire Regimes\nContent on fires and climate.", encoding="utf-8")
    vm.reload_vault()

    with patch("scripcy.cli_vault.ProjectManager", return_value=pm):
        # Keyword search
        res = runner.invoke(app, ["vault", "search", "wildfire", "-p", "my_proj", "--json"])
        assert res.exit_code == 0
        data = json.loads(res.output)
        assert len(data) >= 1
        assert "wildfire" in data[0]["title"].lower() or "wildfire" in data[0]["id"].lower()

        # Semantic search
        res_sem = runner.invoke(app, ["vault", "search", "wildfire", "-p", "my_proj", "--semantic", "--json"])
        assert res_sem.exit_code == 0
        sem_data = json.loads(res_sem.output)
        assert isinstance(sem_data, list)


def test_vault_status_and_history(tmp_path: Path):
    pm, proj, ws, vm = setup_test_vault_project(tmp_path)

    with patch("scripcy.cli_vault.ProjectManager", return_value=pm):
        # Status
        res_stat = runner.invoke(app, ["vault", "status", "-p", "my_proj", "--json"])
        assert res_stat.exit_code == 0
        stat_data = json.loads(res_stat.output)
        assert "status" in stat_data
        assert "total_files" in stat_data

        # History
        vm.record_history("test_action", "Aktivitas uji coba", files_affected=["sample.txt"])
        res_hist = runner.invoke(app, ["vault", "history", "-p", "my_proj", "--json"])
        assert res_hist.exit_code == 0
        hist_data = json.loads(res_hist.output)
        assert len(hist_data) >= 1
        assert hist_data[0]["action"] == "test_action"


def test_vault_remove_and_restore(tmp_path: Path):
    pm, proj, ws, vm = setup_test_vault_project(tmp_path)

    f = vm.papers_dir / "remove_target.txt"
    f.write_text("Title: Will be removed", encoding="utf-8")
    vm.reload_vault()

    with patch("scripcy.cli_vault.ProjectManager", return_value=pm):
        # Remove / Archive
        res_rem = runner.invoke(app, ["vault", "remove", "remove_target.txt", "-p", "my_proj", "--json"])
        assert res_rem.exit_code == 0
        rem_data = json.loads(res_rem.output)
        assert rem_data["status"] == "archived"

        # Check archived list
        res_arch = runner.invoke(app, ["vault", "archived", "-p", "my_proj", "--json"])
        assert res_arch.exit_code == 0
        arch_data = json.loads(res_arch.output)
        assert any(a["filename"] == "remove_target.txt" for a in arch_data)

        # Restore
        res_res = runner.invoke(app, ["vault", "restore", "remove_target.txt", "-p", "my_proj", "--json"])
        assert res_res.exit_code == 0
        res_data = json.loads(res_res.output)
        assert res_data["status"] == "restored"


def test_vault_reindex_tables_images(tmp_path: Path):
    pm, proj, ws, vm = setup_test_vault_project(tmp_path)

    with patch("scripcy.cli_vault.ProjectManager", return_value=pm):
        # Reindex
        res_idx = runner.invoke(app, ["vault", "reindex", "-p", "my_proj", "--json"])
        assert res_idx.exit_code == 0
        idx_data = json.loads(res_idx.output)
        assert idx_data["status"] == "success"
        assert "counts" in idx_data

        # Tables and images list
        res_tbl = runner.invoke(app, ["vault", "tables", "-p", "my_proj", "--json"])
        assert res_tbl.exit_code == 0
        assert isinstance(json.loads(res_tbl.output), list)

        res_img = runner.invoke(app, ["vault", "images", "-p", "my_proj", "--json"])
        assert res_img.exit_code == 0
        assert isinstance(json.loads(res_img.output), list)
