"""Unit tests for VaultManager, MarkItDown conversion, and soft-archiving."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripcy.core.vault import VaultManager
from scripcy.core.workspace import WorkspaceManager


@pytest.fixture
def test_vault_manager(tmp_path: Path) -> VaultManager:
    """Fixture providing an isolated VaultManager."""
    ws = WorkspaceManager(root_dir=tmp_path / "study_vault")
    return VaultManager(workspace=ws)


class TestVaultManager:
    """Tests for vault file discovery, format conversion, and archiving."""

    def test_directory_initialization(self, test_vault_manager: VaultManager) -> None:
        assert test_vault_manager.vault_dir.is_dir()
        assert test_vault_manager.archive_dir.is_dir()

    def test_add_and_convert_text_file(self, test_vault_manager: VaultManager, tmp_path: Path) -> None:
        source_file = tmp_path / "research_notes.txt"
        source_file.write_text("# Chapter 1 Introduction\n\nQualitative methodology notes.", encoding="utf-8")

        item = test_vault_manager.add_resource(source_file)
        assert item.filename == "research_notes.txt"
        assert item.extension == ".txt"
        assert item.is_cached is True

        # Check cache was created with frontmatter
        cache_path = test_vault_manager.workspace.get_cache_path("research_notes.md")
        assert cache_path.exists()
        content = cache_path.read_text(encoding="utf-8")
        assert "research_notes" in content
        assert "Qualitative methodology notes" in content

    def test_soft_remove_to_archive_and_restore(
        self,
        test_vault_manager: VaultManager,
        tmp_path: Path,
    ) -> None:
        # Create file in vault
        doc = test_vault_manager.vault_dir / "draft_review.txt"
        doc.write_text("Draft review content", encoding="utf-8")

        # Convert to markdown
        test_vault_manager.convert_file(doc)
        cache_path = test_vault_manager.workspace.get_cache_path("draft_review.md")
        assert cache_path.exists()
        assert len(test_vault_manager.list_active_files()) == 1
        assert len(test_vault_manager.list_archived_files()) == 0

        # Archive (soft-remove)
        archived = test_vault_manager.archive_resource(1)
        assert archived.filename == "draft_review.txt"
        assert not (test_vault_manager.vault_dir / "draft_review.txt").exists()
        assert (test_vault_manager.archive_dir / "draft_review.txt").exists()
        assert not cache_path.exists()  # Unlinked from active context

        assert len(test_vault_manager.list_active_files()) == 0
        assert len(test_vault_manager.list_archived_files()) == 1

        # Restore from archive
        restored = test_vault_manager.restore_resource(1)
        assert restored.filename == "draft_review.txt"
        assert (test_vault_manager.vault_dir / "draft_review.txt").exists()
        assert not (test_vault_manager.archive_dir / "draft_review.txt").exists()
        assert cache_path.exists()  # Re-converted & re-indexed

        assert len(test_vault_manager.list_active_files()) == 1
        assert len(test_vault_manager.list_archived_files()) == 0

    def test_reload_vault(self, test_vault_manager: VaultManager) -> None:
        # Drop two raw files manually into vault
        (test_vault_manager.vault_dir / "notes_a.txt").write_text("Notes A", encoding="utf-8")
        (test_vault_manager.vault_dir / "notes_b.txt").write_text("Notes B", encoding="utf-8")

        total, converted = test_vault_manager.reload_vault(force=False)
        assert total == 2
        assert converted == 2

        # Second reload without changes should not re-convert
        total2, converted2 = test_vault_manager.reload_vault(force=False)
        assert total2 == 2
        assert converted2 == 0


    def test_archive_multiple_with_quotes_and_prefix(self, test_vault_manager: VaultManager) -> None:
        (test_vault_manager.vault_dir / "The_biogeochemical_cycle_of_iron_in_the_ocean.txt").write_text("Iron ocean", encoding="utf-8")
        (test_vault_manager.vault_dir / "Development_of_the_MIROC-ES2L_Earth_system_model_and_the_evaluation_of_biogeochemical_cycles.txt").write_text("MIROC model", encoding="utf-8")

        test_vault_manager.reload_vault(force=False)
        assert len(test_vault_manager.list_active_files()) == 2

        # Test with quoted identifiers and prefix matching
        targets = [
            '"The_biogeochemical_cycle_of_iron_in_the_ocean"',
            '"Development_of_the_MIROC-ES2L_Earth_system_model_and_the_evaluation_of_biogeoche"',
        ]
        archived = test_vault_manager.archive_multiple(targets)
        assert len(archived) == 2
        assert len(test_vault_manager.list_active_files()) == 0
        assert len(test_vault_manager.list_archived_files()) == 2

    def test_local_staging_and_health_check(self, test_vault_manager: VaultManager) -> None:
        assert test_vault_manager.local_dir.is_dir()

        # Drop a file into local/
        local_file = test_vault_manager.local_dir / "draft_paper.pdf"
        local_file.write_bytes(b"%PDF-1.4 simulated pdf")

        health = test_vault_manager.check_vault_health()
        assert health["total_files"] == 1
        assert health["status"] == "need_update"
        assert "belum di-update" in health["warning_details"][0]
        assert health["category_health"]["user"]["total"] == 1

        # Re-index / convert
        test_vault_manager.reload_vault()
        health_after = test_vault_manager.check_vault_health()
        assert health_after["status"] == "active"
        assert len(health_after["warning_details"]) == 0

    def test_duplicate_and_unsupported_file_detection(self, test_vault_manager: VaultManager) -> None:
        # 1. Unsupported file
        bad_file = test_vault_manager.local_dir / "script.exe"
        bad_file.write_bytes(b"MZbinary")

        health = test_vault_manager.check_vault_health()
        assert health["status"] == "error"
        assert health["error_count"] >= 1
        assert "tidak didukung" in health["error_details"][0]

        bad_file.unlink()

        # 2. Duplicate file
        (test_vault_manager.papers_dir / "sample.txt").write_text("Test content", encoding="utf-8")
        (test_vault_manager.local_dir / "sample.txt").write_text("Test duplicate content", encoding="utf-8")
        test_vault_manager.reload_vault()

        health_dup = test_vault_manager.check_vault_health()
        assert health_dup["status"] == "warning"
        assert health_dup["duplicate_count"] >= 1
        assert "duplikat" in health_dup["warning_details"][0]

        # Verify VaultItem is_duplicate flag
        active_items = test_vault_manager.list_active_files()
        dup_items = [it for it in active_items if it.is_duplicate]
        assert len(dup_items) >= 1
        assert dup_items[0].is_duplicate is True

        # 3. Test has_error flag on invalid file
        bad_file2 = test_vault_manager.papers_dir / "unsupported.bin"
        bad_file2.write_bytes(b"invalid binary")
        items_with_err = test_vault_manager.list_active_files()
        err_items = [it for it in items_with_err if it.has_error]
        assert len(err_items) >= 1
        assert err_items[0].has_error is True
