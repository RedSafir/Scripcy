"""Unit tests for VaultManager, MarkItDown conversion, and soft-archiving."""

from __future__ import annotations

from pathlib import Path

import pytest

from scrpcy.core.vault import VaultManager
from scrpcy.core.workspace import WorkspaceManager


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
