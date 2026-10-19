"""Tests for the WorkspaceManager directory and cache handling."""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from scrpcy.core.workspace import (
    DEFAULT_WORKSPACE_ENV,
    DEFAULT_WORKSPACE_RELATIVE,
    WorkspaceManager,
)
from scrpcy.exceptions import WorkspaceError


def test_default_workspace_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DEFAULT_WORKSPACE_ENV, raising=False)
    workspace = WorkspaceManager(auto_create=False)
    expected_root = (Path.home() / DEFAULT_WORKSPACE_RELATIVE).resolve()
    assert workspace.root_dir == expected_root
    assert workspace.cache_dir == expected_root / "cache"
    assert workspace.vault_dir == expected_root / "vault"


def test_custom_workspace_path(tmp_path: Path) -> None:
    custom_dir = tmp_path / "custom_workspace"
    workspace = WorkspaceManager(root_dir=custom_dir, auto_create=True)
    assert workspace.root_dir == custom_dir.resolve()
    assert workspace.cache_dir.is_dir()
    assert workspace.vault_dir.is_dir()


def test_workspace_environment_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_dir = tmp_path / "env_workspace"
    monkeypatch.setenv(DEFAULT_WORKSPACE_ENV, str(env_dir))
    workspace = WorkspaceManager(auto_create=True)
    assert workspace.root_dir == env_dir.resolve()
    assert workspace.cache_dir.is_dir()


def test_cache_save_and_read(temp_workspace: WorkspaceManager) -> None:
    filename = "document.md"
    content = "# Research Notes\nSample body content."
    saved_path = temp_workspace.save_to_cache(filename, content)

    assert saved_path.is_file()
    assert saved_path == temp_workspace.get_cache_path(filename)
    assert temp_workspace.cache_exists(filename) is True

    read_content = temp_workspace.read_from_cache(filename)
    assert read_content == content


def test_read_nonexistent_cache_raises_error(temp_workspace: WorkspaceManager) -> None:
    with pytest.raises(FileNotFoundError):
        temp_workspace.read_from_cache("missing_file.md")


def test_list_cached_documents(temp_workspace: WorkspaceManager) -> None:
    temp_workspace.save_to_cache("doc_b.md", "content b")
    temp_workspace.save_to_cache("doc_a.md", "content a")
    temp_workspace.save_to_cache("notes.txt", "text note")

    md_docs = temp_workspace.list_cached_documents("*.md")
    assert len(md_docs) == 2
    assert md_docs[0].name == "doc_a.md"
    assert md_docs[1].name == "doc_b.md"


def test_clear_cache(temp_workspace: WorkspaceManager) -> None:
    temp_workspace.save_to_cache("doc1.md", "content 1")
    temp_workspace.save_to_cache("doc2.md", "content 2")
    assert len(temp_workspace.list_cached_documents()) == 2

    temp_workspace.clear_cache()
    assert len(temp_workspace.list_cached_documents()) == 0
    assert temp_workspace.cache_dir.is_dir()
