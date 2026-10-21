"""Tests: bulk vault remove and background curation loop (updated for archive_multiple)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest

from scrpcy.core.vault import VaultItem


def _make_studio(tmp_path: Path):
    """Build a minimal CLIStudio-like mock for testing vault helpers."""
    from scrpcy.cli_studio import CLIStudio
    from scrpcy.core.config import ScrpcyConfig

    config = ScrpcyConfig(data_dir=tmp_path)
    studio = CLIStudio.__new__(CLIStudio)
    studio.config = config
    studio.console = MagicMock()
    studio.vault_manager = MagicMock()
    studio.active_documents = set()
    studio._vault_stem_cache = []
    studio.workspace = None
    studio.engine = None
    studio.chat_manager = None
    studio.task_manager = MagicMock()
    studio._sync_vault_cache = MagicMock()
    studio._get_cached_stems = MagicMock(return_value=[])
    return studio


def _item(name: str) -> VaultItem:
    return VaultItem(
        filename=name,
        path=Path(name),
        extension=".pdf",
        size_bytes=1000,
        is_archived=True,
        is_cached=False,
        title=name.replace(".pdf", ""),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. cmd_vault_remove: uses archive_multiple
# ──────────────────────────────────────────────────────────────────────────────

class TestCmdVaultRemoveBulk:
    def test_single_id_still_works(self, tmp_path):
        studio = _make_studio(tmp_path)
        studio.vault_manager.archive_multiple.return_value = [_item("paper_a.pdf")]

        studio.cmd_vault_remove("1")

        studio.vault_manager.archive_multiple.assert_called_once_with(["1"])
        studio._sync_vault_cache.assert_called_once()

    def test_space_separated_bulk(self, tmp_path):
        studio = _make_studio(tmp_path)
        studio.vault_manager.archive_multiple.return_value = [
            _item("paper_a.pdf"),
            _item("paper_b.pdf"),
            _item("paper_c.pdf"),
        ]

        studio.cmd_vault_remove("1 3 5")

        studio.vault_manager.archive_multiple.assert_called_once_with(["1", "3", "5"])
        studio._sync_vault_cache.assert_called_once()

    def test_comma_separated_bulk(self, tmp_path):
        studio = _make_studio(tmp_path)
        studio.vault_manager.archive_multiple.return_value = [
            _item("p1.pdf"),
            _item("p2.pdf"),
        ]

        studio.cmd_vault_remove("2,4")

        studio.vault_manager.archive_multiple.assert_called_once_with(["2", "4"])

    def test_mixed_separator(self, tmp_path):
        studio = _make_studio(tmp_path)
        studio.vault_manager.archive_multiple.return_value = [
            _item("p1.pdf"),
            _item("p2.pdf"),
            _item("p3.pdf"),
        ]

        studio.cmd_vault_remove("1, 2, 3")

        studio.vault_manager.archive_multiple.assert_called_once_with(["1", "2", "3"])

    def test_handles_empty_result(self, tmp_path):
        studio = _make_studio(tmp_path)
        studio.vault_manager.archive_multiple.return_value = []

        studio.cmd_vault_remove("99")

        studio.vault_manager.archive_multiple.assert_called_once_with(["99"])
        studio._sync_vault_cache.assert_called_once()


# ──────────────────────────────────────────────────────────────────────────────
# 2. _execute_internal_vault_tool: bulk remove branch
# ──────────────────────────────────────────────────────────────────────────────

class TestInternalVaultToolBulk:
    def test_bulk_remove_returns_label(self, tmp_path):
        studio = _make_studio(tmp_path)
        studio.vault_manager.archive_multiple.return_value = [
            _item("paper_a.pdf"),
            _item("paper_c.pdf"),
        ]

        action_label, target_label, output = studio._execute_internal_vault_tool("/vault remove 1 3")

        assert action_label == "Mengarsipkan"
        assert "2 dokumen" in target_label
        assert "paper_a.pdf" in output
        assert "paper_c.pdf" in output

    def test_bulk_with_not_found(self, tmp_path):
        studio = _make_studio(tmp_path)
        studio.vault_manager.archive_multiple.return_value = []

        action_label, target_label, output = studio._execute_internal_vault_tool("/vault remove 99")

        assert "Tidak Ditemukan" in target_label
        assert "99" in output


# ──────────────────────────────────────────────────────────────────────────────
# 3. VaultManager.archive_multiple real integration (atomic zero-shift test)
# ──────────────────────────────────────────────────────────────────────────────

def test_archive_multiple_atomic_zero_shift(tmp_path):
    from scrpcy.core.workspace import WorkspaceManager
    from scrpcy.core.vault import VaultManager

    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    vault_dir = proj_dir / "vault"
    vault_dir.mkdir()

    # Create 4 test files
    f1 = vault_dir / "earth_1.txt"
    f2 = vault_dir / "ai_ml_2.txt"
    f3 = vault_dir / "earth_3.txt"
    f4 = vault_dir / "marketing_4.txt"
    for f in (f1, f2, f3, f4):
        f.write_text("dummy content", encoding="utf-8")

    ws = WorkspaceManager(proj_dir)
    vm = VaultManager(ws)

    # Active files initially: [ai_ml_2, earth_1, earth_3, marketing_4] or sorted order
    active = vm.list_active_files()
    assert len(active) == 4

    # Target non-earth by 1-based index based on initial snapshot:
    # Say we remove index 1 and 4
    target_names = [active[0].filename, active[3].filename]
    archived = vm.archive_multiple(["1", "4"])

    assert len(archived) == 2
    assert [a.filename for a in archived] == target_names

    # Check remaining
    remaining = [f.filename for f in vm.list_active_files()]
    assert len(remaining) == 2
    assert active[0].filename not in remaining
    assert active[3].filename not in remaining
    assert active[1].filename in remaining
    assert active[2].filename in remaining


# ──────────────────────────────────────────────────────────────────────────────
# 4. Background curation loop: _background_vault_curation_worker
# ──────────────────────────────────────────────────────────────────────────────

class TestBackgroundCurationLoop:
    def test_empty_vault_returns_early(self, tmp_path):
        studio = _make_studio(tmp_path)
        studio._get_cached_stems.return_value = []
        studio.vault_manager = MagicMock()
        studio.engine = MagicMock()

        result = studio._background_vault_curation_worker("task-1", None, "topik bumi")

        assert "kosong" in result.lower()

    def test_loop_stores_notes_in_session(self, tmp_path):
        studio = _make_studio(tmp_path)
        studio._get_cached_stems.return_value = ["paper_a", "paper_b"]
        studio.vault_manager = MagicMock()
        chat_mgr = MagicMock()
        studio.chat_manager = chat_mgr

        mock_result = MagicMock()
        mock_result.answer = "1. RELEVAN\n2. ARSIP\n[SELESAI]"
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=mock_result)
        studio.engine = mock_engine
        studio.task_manager.log = MagicMock()

        result = studio._background_vault_curation_worker("task-1", "session-abc", "topik bumi")

        assert chat_mgr.add_message.called
        calls = chat_mgr.add_message.call_args_list
        session_ids = [c.args[0] for c in calls]
        assert all(sid == "session-abc" for sid in session_ids)

        final_call = calls[-1]
        assert "LOOP-KURASI SELESAI" in final_call.args[2]
        assert "selesai" in result.lower()

    def test_loop_stops_on_selesai_sentinel(self, tmp_path):
        studio = _make_studio(tmp_path)
        studio._get_cached_stems.return_value = [f"paper_{i}" for i in range(12)]
        studio.vault_manager = MagicMock()
        studio.chat_manager = MagicMock()
        studio.task_manager.log = MagicMock()

        call_count = 0

        async def _query(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            m = MagicMock()
            m.answer = "Evaluasi selesai. [SELESAI]"
            return m

        studio.engine = MagicMock()
        studio.engine.query = _query

        studio._background_vault_curation_worker("task-1", "sess", "bumi")

        assert call_count == 1

    def test_no_engine_returns_error(self, tmp_path):
        studio = _make_studio(tmp_path)
        studio._get_cached_stems.return_value = ["paper_a"]
        studio.vault_manager = MagicMock()
        studio.engine = None

        result = studio._background_vault_curation_worker("task-1", None, "bumi")
        assert "tidak tersedia" in result.lower()
