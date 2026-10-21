"""Vault resource manager with Microsoft MarkItDown conversion and soft-archiving."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
from typing import Any

from pydantic import BaseModel, Field

from scrpcy.core.models import DocumentMetadata, ParsedDocument
from scrpcy.core.parser import DocumentParser
from scrpcy.core.workspace import WorkspaceManager

logger = logging.getLogger(__name__)

# Try importing MarkItDown
try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False


class VaultItem(BaseModel):
    """Represents a physical file in the project vault."""

    filename: str
    path: Path
    extension: str
    size_bytes: int
    is_archived: bool = False
    is_cached: bool = False
    title: str = ""
    cite_key: str = ""

    @property
    def stem(self) -> str:
        """Return filename stem without extension."""
        return self.path.stem

    @property
    def readable_size(self) -> str:
        """Return formatted file size."""
        size = self.size_bytes
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"


class VaultManager:
    """Manages active and archived vault files, format conversion, and markdown caching."""

    def __init__(self, workspace: WorkspaceManager) -> None:
        self.workspace = workspace
        self.vault_dir = workspace.vault_dir
        self.archive_dir = self.vault_dir / ".archive"
        self.pdf_parser = DocumentParser(workspace=workspace)
        self.markitdown = MarkItDown() if MARKITDOWN_AVAILABLE else None
        self.ensure_directories()

    def ensure_directories(self) -> None:
        """Ensure vault and archive directories exist."""
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def list_active_files(self) -> list[VaultItem]:
        """List all non-hidden, active files inside vault/."""
        self.ensure_directories()
        items: list[VaultItem] = []
        for p in sorted(self.vault_dir.iterdir()):
            if p.is_file() and not p.name.startswith("."):
                cache_file = self.workspace.get_cache_path(f"{p.stem}.md")
                meta = self._read_frontmatter_metadata(cache_file) if cache_file.exists() else None
                items.append(VaultItem(
                    filename=p.name,
                    path=p,
                    extension=p.suffix.lower(),
                    size_bytes=p.stat().st_size,
                    is_archived=False,
                    is_cached=cache_file.exists(),
                    title=meta.title if meta else p.stem,
                    cite_key=meta.cite_key if meta else "",
                ))
        return items

    def list_archived_files(self) -> list[VaultItem]:
        """List all soft-removed files inside vault/.archive/."""
        self.ensure_directories()
        items: list[VaultItem] = []
        if not self.archive_dir.is_dir():
            return items

        for p in sorted(self.archive_dir.iterdir()):
            if p.is_file() and not p.name.startswith("."):
                items.append(VaultItem(
                    filename=p.name,
                    path=p,
                    extension=p.suffix.lower(),
                    size_bytes=p.stat().st_size,
                    is_archived=True,
                    is_cached=False,
                    title=p.stem,
                ))
        return items

    def _read_frontmatter_metadata(self, cache_path: Path) -> DocumentMetadata | None:
        """Extract metadata frontmatter from cached markdown if available."""
        try:
            from scrpcy.core.models import FRONTMATTER_PATTERN
            import yaml
            text = cache_path.read_text(encoding="utf-8")
            match = FRONTMATTER_PATTERN.match(text)
            if match:
                data = yaml.safe_load(match.group(1)) or {}
                return DocumentMetadata.model_validate(data)
        except Exception:
            pass
        return None

    def convert_file(self, file_path: Path, force: bool = False) -> Path:
        """Convert any vault document into Markdown and save to cache/."""
        cache_path = self.workspace.get_cache_path(f"{file_path.stem}.md")
        if cache_path.exists() and not force:
            if cache_path.stat().st_mtime >= file_path.stat().st_mtime:
                return cache_path

        # 1. Academic PDF Parser path
        if file_path.suffix.lower() == ".pdf":
            try:
                parsed = self.pdf_parser.parse(file_path, save_to_cache=True)
                return parsed.cache_path
            except Exception as pdf_err:
                logger.warning(f"Academic PDF parser failed for {file_path.name}, falling back: {pdf_err}")

        # 2. MarkItDown path for all document types (Word, PPT, Excel, HTML, PDF fallback)
        if self.markitdown:
            try:
                res = self.markitdown.convert(str(file_path))
                raw_md = res.text_content
            except Exception as mid_err:
                logger.warning(f"MarkItDown failed for {file_path.name}: {mid_err}")
                raw_md = f"# {file_path.name}\n\n[Warning: Binary content conversion unavailable]"
        else:
            # Fallback if text or raw format
            try:
                raw_md = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                raw_md = f"# {file_path.name}\n\n[Binary file]"

        # Create standard frontmatter
        meta = DocumentMetadata(
            title=file_path.stem.replace("_", " ").title(),
            cite_key=file_path.stem.lower().replace(" ", "_"),
        )
        meta.ensure_cite_key()
        full_content = f"{meta.to_frontmatter()}\n{raw_md}"
        return self.workspace.save_to_cache(f"{file_path.stem}.md", full_content)

    def reload_vault(self, force: bool = False) -> tuple[int, int]:
        """Scan all active vault files, convert missing/updated to markdown.
        
        Returns:
            Tuple of (total_files, converted_files).
        """
        active_files = self.list_active_files()
        converted_count = 0

        for item in active_files:
            cache_path = self.workspace.get_cache_path(f"{item.path.stem}.md")
            if force or not cache_path.exists() or cache_path.stat().st_mtime < item.path.stat().st_mtime:
                self.convert_file(item.path, force=force)
                converted_count += 1

        return len(active_files), converted_count

    def add_resource(self, source_path: str | Path) -> VaultItem:
        """Copy an external file into vault/ and convert it to markdown."""
        src = Path(source_path).expanduser().resolve()
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(f"Source file not found at '{src}'")

        dest = self.vault_dir / src.name
        # If dest already exists, generate a safe unique name
        if dest.exists() and dest != src:
            stem = src.stem
            ext = src.suffix
            counter = 1
            while dest.exists():
                dest = self.vault_dir / f"{stem}_{counter}{ext}"
                counter += 1

        if dest != src:
            shutil.copy2(src, dest)

        # Convert to markdown
        self.convert_file(dest, force=True)

        return VaultItem(
            filename=dest.name,
            path=dest,
            extension=dest.suffix.lower(),
            size_bytes=dest.stat().st_size,
            is_archived=False,
            is_cached=True,
            title=dest.stem,
        )

    def archive_multiple(self, identifiers: list[str | int]) -> list[VaultItem]:
        """Soft-archive multiple resources atomically using initial active snapshot (prevents index-shifting)."""
        active_files = self.list_active_files()
        targets: list[VaultItem] = []
        seen_filenames: set[str] = set()

        for ident in identifiers:
            target: VaultItem | None = None
            ident_clean = str(ident).strip().strip('"\'')
            if isinstance(ident, int) or ident_clean.isdigit():
                idx = int(ident_clean) - 1
                if 0 <= idx < len(active_files):
                    target = active_files[idx]
            else:
                clean_ident = ident_clean.lower()
                for f in active_files:
                    f_name = f.filename.lower()
                    f_stem = f.path.stem.lower()
                    if (
                        f_name == clean_ident
                        or f_stem == clean_ident
                        or f_stem.startswith(clean_ident)
                        or clean_ident.startswith(f_stem)
                        or (len(clean_ident) >= 10 and clean_ident in f_stem)
                    ):
                        target = f
                        break
            if target and target.filename not in seen_filenames:
                targets.append(target)
                seen_filenames.add(target.filename)

        archived_items: list[VaultItem] = []
        for t in targets:
            dest = self.archive_dir / t.filename
            if t.path.exists():
                shutil.move(str(t.path), str(dest))
                cache_path = self.workspace.get_cache_path(f"{t.path.stem}.md")
                if cache_path.exists():
                    try:
                        cache_path.unlink()
                    except OSError:
                        pass
                archived_items.append(
                    VaultItem(
                        filename=dest.name,
                        path=dest,
                        extension=dest.suffix.lower(),
                        size_bytes=dest.stat().st_size,
                        is_archived=True,
                        is_cached=False,
                        title=dest.stem,
                    )
                )
        return archived_items

    def archive_resource(self, identifier: str | int) -> VaultItem:
        """Soft-remove a file from active project by moving it to .archive/."""
        active_files = self.list_active_files()
        target: VaultItem | None = None

        ident_clean = str(identifier).strip().strip('"\'')
        if isinstance(identifier, int) or ident_clean.isdigit():
            idx = int(ident_clean) - 1
            if 0 <= idx < len(active_files):
                target = active_files[idx]
        else:
            clean_ident = ident_clean.lower()
            for f in active_files:
                f_name = f.filename.lower()
                f_stem = f.path.stem.lower()
                if (
                    f_name == clean_ident
                    or f_stem == clean_ident
                    or f_stem.startswith(clean_ident)
                    or clean_ident.startswith(f_stem)
                    or (len(clean_ident) >= 10 and clean_ident in f_stem)
                ):
                    target = f
                    break

        if not target:
            raise FileNotFoundError(f"Active vault file '{identifier}' not found.")

        dest = self.archive_dir / target.filename
        shutil.move(str(target.path), str(dest))

        # Unlink cached markdown so AI context excludes it
        cache_path = self.workspace.get_cache_path(f"{target.path.stem}.md")
        if cache_path.exists():
            try:
                cache_path.unlink()
            except OSError:
                pass

        return VaultItem(
            filename=dest.name,
            path=dest,
            extension=dest.suffix.lower(),
            size_bytes=dest.stat().st_size,
            is_archived=True,
            is_cached=False,
            title=dest.stem,
        )

    def restore_resource(self, identifier: str | int) -> VaultItem:
        """Restore an archived file back to active vault/ and re-convert to markdown."""
        archived_files = self.list_archived_files()
        target: VaultItem | None = None

        if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
            idx = int(identifier) - 1
            if 0 <= idx < len(archived_files):
                target = archived_files[idx]
        else:
            clean_ident = str(identifier).strip().lower()
            for f in archived_files:
                if f.filename.lower() == clean_ident or f.path.stem.lower() == clean_ident:
                    target = f
                    break

        if not target:
            raise FileNotFoundError(f"Archived vault file '{identifier}' not found.")

        dest = self.vault_dir / target.filename
        shutil.move(str(target.path), str(dest))

        # Re-convert to markdown
        self.convert_file(dest, force=True)

        return VaultItem(
            filename=dest.name,
            path=dest,
            extension=dest.suffix.lower(),
            size_bytes=dest.stat().st_size,
            is_archived=False,
            is_cached=True,
            title=dest.stem,
        )
