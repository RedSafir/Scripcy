"""Vault resource manager with Microsoft MarkItDown conversion and soft-archiving."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Any

from pydantic import BaseModel, Field

from scripcy.core.models import DocumentMetadata, ParsedDocument
from scripcy.core.parser import DocumentParser
from scripcy.core.workspace import WorkspaceManager
from scripcy.core.vector_vault import ProjectVectorVault

logger = logging.getLogger(__name__)

# MarkItDown is imported lazily on demand to avoid heavy Azure SDK startup penalty
MARKITDOWN_AVAILABLE = True

TABLE_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".svg", ".gif", ".bmp", ".tiff"}
WEB_EXTENSIONS = {".html", ".htm", ".mhtml"}

SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".doc", ".ppt",
    ".txt", ".md", ".html", ".htm", ".csv", ".json", ".xml", ".tsv",
    ".png", ".jpg", ".jpeg", ".webp", ".svg", ".gif"
}


class VaultItem(BaseModel):
    """Represents a physical file in the project vault."""

    filename: str
    path: Path
    extension: str
    size_bytes: int
    is_archived: bool = False
    is_cached: bool = False
    title: str = ""
    context: str = ""
    cite_key: str = ""
    category: str = "papers"
    is_supported: bool = True
    has_error: bool = False
    is_duplicate: bool = False
    error_message: str = ""
    updated_at: str = ""
    tables_count: int = 0
    visuals_count: int = 0
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str = ""
    url: str = ""

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
        self.papers_dir = getattr(workspace, "papers_dir", self.vault_dir / "papers")
        self.web_dir = getattr(workspace, "web_dir", self.vault_dir / "web")
        self.local_dir = getattr(workspace, "local_dir", self.vault_dir / "local")
        self.user_tables_dir = self.vault_dir / "tables" / "user"
        self.user_images_dir = self.vault_dir / "images" / "user"
        self.history_file = self.vault_dir / ".history.jsonl"
        self.pdf_parser = DocumentParser(workspace=workspace)
        self._markitdown = None
        self.ensure_directories()
        self.vector_vault = ProjectVectorVault(
            db_path=self.vault_dir / "project_vault.db",
            vault_dir=self.vault_dir,
        )

    @property
    def markitdown(self) -> Any:
        if self._markitdown is None:
            try:
                from markitdown import MarkItDown
                self._markitdown = MarkItDown()
            except ImportError:
                self._markitdown = False
        return self._markitdown if self._markitdown is not False else None

    @markitdown.setter
    def markitdown(self, val: Any) -> None:
        self._markitdown = val

    def ensure_directories(self) -> None:
        """Ensure vault, archive, and local staging directories exist."""
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.user_tables_dir.mkdir(parents=True, exist_ok=True)
        self.user_images_dir.mkdir(parents=True, exist_ok=True)

    def record_history(
        self,
        action: str,
        description: str,
        files_affected: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record an audit trail event in vault/.history.jsonl."""
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "description": description,
            "files_affected": files_affected or [],
            "details": details or {},
        }
        try:
            with open(self.history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug(f"Failed to record vault history: {e}")

    def list_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """List audit history entries from newest to oldest."""
        if not self.history_file.exists():
            return []
        entries = []
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except Exception:
                            pass
        except Exception:
            return []
        return list(reversed(entries))[:limit]

    def scan_local_staging(self) -> list[dict[str, Any]]:
        """Scan all files and subfolders in vault/local/, detecting folder context, category, and duplicate status."""
        self.ensure_directories()
        results: list[dict[str, Any]] = []
        if not self.local_dir.exists():
            return results

        active_items = self.list_active_files()
        existing_filenames = {
            it.filename.lower(): it.path
            for it in active_items
            if self.local_dir not in it.path.parents and it.path.parent != self.local_dir
        }

        for p in sorted(self.local_dir.rglob("*")):
            if p.is_file() and not p.name.startswith(".") and not p.name.endswith(".db"):
                try:
                    rel_p = p.relative_to(self.local_dir)
                    subfolder_parts = rel_p.parent.parts
                    folder_context = " > ".join(subfolder_parts) if subfolder_parts else ""
                    subfolder_path = Path(*subfolder_parts) if subfolder_parts else Path("")
                except Exception:
                    folder_context = ""
                    subfolder_path = Path("")

                ext = p.suffix.lower()
                if ext in TABLE_EXTENSIONS:
                    category = "tables"
                    target_dir = self.user_tables_dir / subfolder_path
                elif ext in IMAGE_EXTENSIONS:
                    category = "images"
                    target_dir = self.user_images_dir / subfolder_path
                elif ext in WEB_EXTENSIONS:
                    category = "web"
                    target_dir = self.web_dir / subfolder_path
                else:
                    category = "papers"
                    target_dir = self.papers_dir / subfolder_path

                target_file = target_dir / p.name
                is_duplicate = target_file.exists() or (p.name.lower() in existing_filenames)
                existing_match = target_file if target_file.exists() else existing_filenames.get(p.name.lower())

                results.append({
                    "path": p,
                    "filename": p.name,
                    "rel_path": str(rel_p).replace("\\", "/"),
                    "folder_context": folder_context,
                    "subfolder_path": subfolder_path,
                    "category": category,
                    "target_dir": target_dir,
                    "target_file": target_file,
                    "is_duplicate": is_duplicate,
                    "existing_match": existing_match,
                    "size_bytes": p.stat().st_size,
                })

        return results

    def process_local_item(
        self,
        item_info: dict[str, Any],
        action: str = "keep_both",
        context_override: str = "",
        caption_override: str = "",
    ) -> Path | None:
        """Move and ingest a single local staging file into its designated vault folder."""
        src_path: Path = item_info["path"]
        if not src_path.exists():
            return None

        category: str = item_info["category"]
        target_dir: Path = item_info["target_dir"]
        target_dir.mkdir(parents=True, exist_ok=True)

        final_name = src_path.name
        if item_info.get("is_duplicate"):
            if action == "discard":
                try:
                    src_path.unlink()
                except Exception:
                    pass
                return None
            elif action == "keep_both":
                stem = src_path.stem
                ext = src_path.suffix
                counter = 1
                while (target_dir / f"{stem}_salinan{counter}{ext}").exists():
                    counter += 1
                final_name = f"{stem}_salinan{counter}{ext}"
            elif action == "overwrite":
                pass

        dest_file = target_dir / final_name
        ctx = context_override or item_info.get("folder_context", "")
        cap = caption_override or src_path.stem

        if category == "tables":
            if hasattr(self, "vector_vault"):
                self.vector_vault.ingest_standalone_table(
                    table_path=src_path,
                    caption=cap,
                    context=ctx,
                    target_dir=target_dir,
                )
            if src_path.resolve() != dest_file.resolve() and src_path.exists():
                shutil.move(str(src_path), str(dest_file))
        elif category == "images":
            if hasattr(self, "vector_vault"):
                self.vector_vault.ingest_standalone_image(
                    image_path=src_path,
                    caption=cap,
                    context=ctx,
                    target_dir=target_dir,
                )
            if src_path.resolve() != dest_file.resolve() and src_path.exists():
                shutil.move(str(src_path), str(dest_file))
        else:
            if src_path.resolve() != dest_file.resolve():
                shutil.move(str(src_path), str(dest_file))
            try:
                self.convert_file(dest_file, force=True)
            except Exception:
                pass

        try:
            parent = src_path.parent
            if parent != self.local_dir and parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        except Exception:
            pass

        return dest_file

    def list_active_files(self, category: str | None = None) -> list[VaultItem]:
        """List all non-hidden, active paper files inside vault directories."""
        self.ensure_directories()
        items: list[VaultItem] = []
        seen_paths: set[Path] = set()

        # Pre-aggregate table and image counts from vector vault
        tables_by_paper: dict[str, int] = {}
        images_by_paper: dict[str, int] = {}
        try:
            if hasattr(self, "vector_vault"):
                inventory = self.vector_vault.list_inventory()
                for tbl in inventory.get("tables", []):
                    pid = str(tbl.get("paper_id") or "").lower().strip()
                    if pid:
                        tables_by_paper[pid] = tables_by_paper.get(pid, 0) + 1
                for img in inventory.get("images", []):
                    pid = str(img.get("paper_id") or "").lower().strip()
                    if pid:
                        images_by_paper[pid] = images_by_paper.get(pid, 0) + 1
        except Exception:
            pass

        candidate_dirs = [
            self.workspace.papers_dir,
            getattr(self.workspace, "web_dir", self.vault_dir / "web"),
            self.user_tables_dir,
            self.user_images_dir,
            self.local_dir,
            self.vault_dir,
        ]
        for cdir in candidate_dirs:
            if not cdir.exists():
                continue
            generator = cdir.iterdir() if cdir == self.vault_dir else cdir.rglob("*")
            for p in sorted(generator):
                if p.is_file() and not p.name.startswith(".") and not p.name.endswith(".db"):
                    if hasattr(self, "user_tables_dir") and self.user_tables_dir in p.parents:
                        if p.suffix.lower() == ".md" and any((p.parent / f"{p.stem}{ext}").exists() for ext in TABLE_EXTENSIONS):
                            continue
                    if p.resolve() in seen_paths:
                        continue
                    seen_paths.add(p.resolve())
                    cache_file = self.workspace.get_cache_path(f"{p.stem}.md")
                    meta = self._read_frontmatter_metadata(cache_file) if cache_file.exists() else None

                    # Determine item category
                    if (hasattr(self.workspace, "web_dir") and self.workspace.web_dir in p.parents) or p.suffix.lower() in WEB_EXTENSIONS:
                        item_category = "web"
                    elif p.suffix.lower() in TABLE_EXTENSIONS:
                        item_category = "tables"
                    elif p.suffix.lower() in IMAGE_EXTENSIONS:
                        item_category = "images"
                    elif (hasattr(self.workspace, "papers_dir") and self.workspace.papers_dir in p.parents) or p.suffix.lower() == ".pdf":
                        item_category = "papers"
                    else:
                        item_category = "papers"

                    is_supported = p.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
                    has_error = not is_supported or p.stat().st_size == 0
                    err_msg = ""
                    if not is_supported:
                        err_msg = f"Format {p.suffix} tidak didukung"
                    elif p.stat().st_size == 0:
                        err_msg = "Berkas kosong (0 byte)"
                    elif cache_file.exists():
                        try:
                            sample = cache_file.read_text(encoding="utf-8", errors="ignore")[:300]
                            if "[Warning: Binary content conversion unavailable]" in sample:
                                has_error = True
                                err_msg = "Gagal konversi MarkItDown"
                        except Exception:
                            pass

                    if item_category in ("tables", "images"):
                        is_cached = True
                    else:
                        is_cached = cache_file.exists() and cache_file.stat().st_mtime >= p.stat().st_mtime
                    mtime_ts = p.stat().st_mtime
                    updated_at_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime_ts))

                    p_keys = [p.stem.lower().strip(), p.name.lower().strip()]
                    if meta and meta.cite_key:
                        p_keys.append(meta.cite_key.lower().strip())

                    tbl_count = 0
                    for k in p_keys:
                        if k in tables_by_paper:
                            tbl_count = max(tbl_count, tables_by_paper[k])

                    img_count = 0
                    for k in p_keys:
                        if k in images_by_paper:
                            img_count = max(img_count, images_by_paper[k])

                    authors_list = meta.authors if meta and meta.authors else []
                    year_val = meta.year if meta else None
                    doi_val = meta.doi or "" if meta else ""

                    rel_ctx = ""
                    if hasattr(self, "user_tables_dir") and self.user_tables_dir in p.parents and p.parent != self.user_tables_dir:
                        rel_ctx = " > ".join(p.parent.relative_to(self.user_tables_dir).parts)
                    elif hasattr(self, "user_images_dir") and self.user_images_dir in p.parents and p.parent != self.user_images_dir:
                        rel_ctx = " > ".join(p.parent.relative_to(self.user_images_dir).parts)
                    elif self.local_dir in p.parents and p.parent != self.local_dir:
                        rel_ctx = " > ".join(p.parent.relative_to(self.local_dir).parts)
                    elif hasattr(self.workspace, "papers_dir") and self.workspace.papers_dir in p.parents and p.parent != self.workspace.papers_dir:
                        rel_ctx = " > ".join(p.parent.relative_to(self.workspace.papers_dir).parts)
                    elif meta and getattr(meta, "context", None):
                        rel_ctx = meta.context

                    items.append(VaultItem(
                        filename=p.name,
                        path=p,
                        extension=p.suffix.lower(),
                        size_bytes=p.stat().st_size,
                        is_archived=False,
                        is_cached=is_cached,
                        title=meta.title if (meta and meta.title) else p.stem,
                        context=rel_ctx,
                        cite_key=meta.cite_key if meta else "",
                        category=item_category,
                        is_supported=is_supported,
                        has_error=has_error,
                        error_message=err_msg,
                        updated_at=updated_at_str,
                        tables_count=tbl_count,
                        visuals_count=img_count,
                        authors=authors_list,
                        year=year_val,
                        doi=doi_val,
                    ))

        # Detect and flag duplicate items by cite_key or stem
        seen_keys: dict[str, list[VaultItem]] = {}
        for it in items:
            key = (it.cite_key or it.stem).lower().strip()
            seen_keys.setdefault(key, []).append(it)

        for group in seen_keys.values():
            if len(group) > 1:
                for it in group[1:]:
                    it.is_duplicate = True

        if category and category != "all":
            return [it for it in items if it.category == category]
        return items

    def check_vault_health(self) -> dict[str, Any]:
        """Check status and health anomalies across all vault items and categories."""
        active_files = self.list_active_files()
        tables = self.list_tables()
        images = self.list_images()

        last_updated = "-"
        if active_files:
            latest_mtime = max(it.path.stat().st_mtime for it in active_files)
            last_updated = time.strftime("%Y-%m-%d %H:%M", time.localtime(latest_mtime))

        unindexed = [it for it in active_files if not it.is_cached and not it.has_error and it.category not in ("tables", "images")]
        errors = [it for it in active_files if it.has_error]

        # Duplicate detection by stem or cite_key
        seen_keys: dict[str, list[VaultItem]] = {}
        for it in active_files:
            key = (it.cite_key or it.stem).lower().strip()
            seen_keys.setdefault(key, []).append(it)

        duplicate_items = [it for group in seen_keys.values() if len(group) > 1 for it in group[1:]]

        papers_items = [it for it in active_files if it.category == "papers"]
        web_items = [it for it in active_files if it.category == "web"]

        def _cat_stats(cat_items: list[VaultItem]) -> dict[str, Any]:
            c_unindexed = sum(1 for it in cat_items if not it.is_cached and not it.has_error)
            c_dup = sum(1 for it in cat_items if it in duplicate_items)
            c_err = sum(1 for it in cat_items if it.has_error)
            return {
                "total": len(cat_items),
                "warning": c_unindexed + c_dup,
                "error": c_err,
                "items": cat_items,
            }

        cat_health = {
            "papers": _cat_stats(papers_items),
            "web": _cat_stats(web_items),
            "tables": {"total": len(tables), "warning": 0, "error": 0, "items": tables},
            "images": {"total": len(images), "warning": 0, "error": 0, "items": images},
        }

        warning_details: list[str] = []
        if unindexed:
            warning_details.append(f"{len(unindexed)} berkas belum di-update (ketik 'update')")
        if duplicate_items:
            warning_details.append(f"{len(duplicate_items)} berkas terdeteksi duplikat")

        error_details: list[str] = []
        if errors:
            error_details.append(f"{len(errors)} berkas gagal diproses / format tidak didukung")

        if errors:
            status = "error"
            status_display = "[bold red]Error[/bold red]"
        elif duplicate_items:
            status = "warning"
            status_display = "[bold yellow]Peringatan[/bold yellow]"
        elif unindexed:
            status = "need_update"
            status_display = "[bold cyan]Perlu Update[/bold cyan]"
        else:
            status = "active"
            status_display = "[bold green]Aktif[/bold green]"

        return {
            "status": status,
            "status_display": status_display,
            "total_files": len(active_files),
            "last_updated": last_updated,
            "unindexed_count": len(unindexed),
            "duplicate_count": len(duplicate_items),
            "error_count": len(errors),
            "category_health": cat_health,
            "warning_details": warning_details,
            "error_details": error_details,
            "active_items": active_files,
        }

    def list_images(self) -> list[dict[str, Any]]:
        """List filtered informative images from project vector vault."""
        try:
            return self.vector_vault.list_inventory().get("images", [])
        except Exception:
            return []

    def list_tables(self) -> list[dict[str, Any]]:
        """List structured tables extracted into vault/tables/."""
        try:
            return self.vector_vault.list_inventory().get("tables", [])
        except Exception:
            return []

    def search_visual_assets(self, query: str, top_k: int = 2) -> list[dict[str, Any]]:
        """Search and rank relevant academic figures and tables in the vault."""
        try:
            return self.vector_vault.search_visual_assets(query, top_k=top_k)
        except Exception as e:
            logger.debug(f"search_visual_assets error: {e}")
            return []

    def clean_and_reindex_visuals(self, engine: Any = None, model_name: str | None = None) -> dict[str, int]:
        """Purge junk images/tables and re-extract clean figures and tables for all papers."""
        try:
            return self.vector_vault.reindex_all_visuals(engine=engine, model_name=model_name)
        except Exception as e:
            logger.warning(f"clean_and_reindex_visuals error: {e}")
            return {"papers": 0, "images": 0, "tables": 0}

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
            from scripcy.core.models import FRONTMATTER_REGEX
            import yaml
            text = cache_path.read_text(encoding="utf-8")
            match = FRONTMATTER_REGEX.match(text)
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
                try:
                    self.vector_vault.ingest_paper_pdf(
                        pdf_path=file_path,
                        paper_id=file_path.stem,
                        title=file_path.stem.replace("_", " ").title(),
                    )
                except Exception as vec_err:
                    logger.debug(f"Vector vault ingestion skipped/failed for {file_path.name}: {vec_err}")

                parsed = self.pdf_parser.parse(file_path, save_to_cache=True)
                return parsed.cache_path
            except Exception as pdf_err:
                logger.warning(f"Academic PDF parser failed for {file_path.name}, falling back: {pdf_err}")

        # 2. Direct read for plain text and markdown
        if file_path.suffix.lower() in (".txt", ".md"):
            try:
                raw_md = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                raw_md = f"# {file_path.name}"
            from scripcy.core.models import FRONTMATTER_REGEX
            if FRONTMATTER_REGEX.match(raw_md):
                return self.workspace.save_to_cache(f"{file_path.stem}.md", raw_md)
        # 3. MarkItDown path for all document types (Word, PPT, Excel, HTML, PDF fallback)
        elif self.markitdown:
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

        # Create standard frontmatter with heuristic metadata
        meta = self.pdf_parser._heuristic_extract_metadata(raw_md, file_path)
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

    # Backward-compatible convenience aliases
    archive_file = archive_resource
    restore_file = restore_resource
