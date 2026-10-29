"""Workspace directory management for scripcy local-first storage."""

from __future__ import annotations

import os
from pathlib import Path
import shutil

from scripcy.exceptions import WorkspaceError

DEFAULT_WORKSPACE_ENV = "SCRIPCY_WORKSPACE"
DEFAULT_WORKSPACE_RELATIVE = Path(".scripcy") / "workspace"


class WorkspaceManager:
    """Manages the local workspace filesystem, subdirectories, and document caches."""

    def __init__(
        self,
        root_dir: str | Path | None = None,
        auto_create: bool = True,
    ) -> None:
        """Initialize workspace directory manager.

        Args:
            root_dir: Base directory for workspace. If None, checks the
                SCRIPCY_WORKSPACE environment variable, defaulting to ~/.scripcy/workspace.
            auto_create: If True, automatically creates workspace directories.
        """
        if root_dir is not None:
            resolved_root = Path(root_dir)
        elif os.environ.get(DEFAULT_WORKSPACE_ENV):
            resolved_root = Path(os.environ[DEFAULT_WORKSPACE_ENV])
        else:
            resolved_root = Path.home() / DEFAULT_WORKSPACE_RELATIVE

        self.root_dir = resolved_root.expanduser().resolve()
        self.cache_dir = self.root_dir / "cache"
        self.vault_dir = self.root_dir / "vault"
        self.downloads_dir = self.vault_dir / "Downloads"
        self.papers_dir = self.downloads_dir  # Alias for downloads
        self.user_dir = self.vault_dir / "user"
        self.user_input_dir = self.root_dir / "USER INPUT HERE"
        self.local_dir = self.user_input_dir  # Alias for user input intake
        self.web_dir = self.vault_dir / "web"
        self.images_dir = self.vault_dir / "images"
        self.tables_dir = self.vault_dir / "tables"
        self.annotated_dir = self.vault_dir / "annotated"

        if auto_create:
            self.ensure_directories()

    def ensure_directories(self) -> None:
        """Create workspace root, cache, and segregated vault directories if they do not exist."""
        try:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.vault_dir.mkdir(parents=True, exist_ok=True)
            self.downloads_dir.mkdir(parents=True, exist_ok=True)
            self.user_dir.mkdir(parents=True, exist_ok=True)
            self.user_input_dir.mkdir(parents=True, exist_ok=True)
            self.web_dir.mkdir(parents=True, exist_ok=True)
            self.images_dir.mkdir(parents=True, exist_ok=True)
            self.tables_dir.mkdir(parents=True, exist_ok=True)
            self.annotated_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise WorkspaceError(
                f"Failed to create workspace directories at '{self.root_dir}': {exc}"
            ) from exc

    def get_cache_path(self, filename: str | Path) -> Path:
        """Return the absolute path for a file inside the cache directory.

        Args:
            filename: Target file name or path relative to cache.
        """
        name = Path(filename).name
        return self.cache_dir / name

    def save_to_cache(
        self,
        filename: str | Path,
        content: str,
        encoding: str = "utf-8",
    ) -> Path:
        """Save text content to a cached file.

        Args:
            filename: Name of the cached file (e.g. document.md).
            content: Text content to write.
            encoding: Text encoding, default UTF-8.

        Returns:
            The Path to the saved file.
        """
        self.ensure_directories()
        target_path = self.get_cache_path(filename)
        try:
            target_path.write_text(content, encoding=encoding)
            return target_path
        except OSError as exc:
            raise WorkspaceError(
                f"Failed to write cache file '{target_path}': {exc}"
            ) from exc

    def read_from_cache(
        self,
        filename: str | Path,
        encoding: str = "utf-8",
    ) -> str:
        """Read text content from a cached file.

        Args:
            filename: Name of the cached file.
            encoding: Text encoding, default UTF-8.

        Returns:
            The text contents of the cached file.
        """
        target_path = self.get_cache_path(filename)
        if not target_path.is_file():
            raise FileNotFoundError(f"Cache file not found at '{target_path}'")
        try:
            return target_path.read_text(encoding=encoding)
        except OSError as exc:
            raise WorkspaceError(
                f"Failed to read cache file '{target_path}': {exc}"
            ) from exc

    def cache_exists(self, filename: str | Path) -> bool:
        """Check whether a specific file exists in the cache."""
        return self.get_cache_path(filename).is_file()

    def get_cached_document(self, filename_or_stem: str | Path) -> Path | None:
        """Return Path to cached markdown document if it exists, checking both direct name and .md stem."""
        p = Path(filename_or_stem)
        candidate1 = self.get_cache_path(p.name)
        if candidate1.is_file():
            return candidate1
        candidate2 = self.get_cache_path(f"{p.stem}.md")
        if candidate2.is_file():
            return candidate2
        return candidate2

    def list_cached_documents(self, pattern: str = "*.md") -> list[Path]:
        """List cached document paths matching a given glob pattern."""
        if not self.cache_dir.is_dir():
            return []
        return sorted(self.cache_dir.glob(pattern))

    def clear_cache(self) -> None:
        """Remove all files within the cache directory."""
        if self.cache_dir.is_dir():
            try:
                for item in self.cache_dir.iterdir():
                    if item.is_file() or item.is_symlink():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
            except OSError as exc:
                raise WorkspaceError(
                    f"Failed to clear cache directory '{self.cache_dir}': {exc}"
                ) from exc

    def __repr__(self) -> str:
        return f"WorkspaceManager(root_dir='{self.root_dir}')"
