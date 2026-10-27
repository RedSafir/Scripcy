"""Headless Command-Line Interface for Scripcy Vault."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text
import typer

from scripcy.core.project import ProjectInfo, ProjectManager
from scripcy.core.vault import VaultItem, VaultManager
from scripcy.core.workspace import WorkspaceManager

console = Console()

vault_app = typer.Typer(
    name="vault",
    help="Kelola vault riset, berkas dokumen, tabel, dan visual secara headless.",
    no_args_is_help=True,
)


def resolve_vault(
    project_name: str | None = None,
) -> tuple[ProjectInfo | None, WorkspaceManager, VaultManager]:
    """Resolve target project, workspace manager, and vault manager for headless execution."""
    pm = ProjectManager()
    target_project: ProjectInfo | None = None

    if project_name:
        for p in pm.list_projects():
            if p.name.lower() == project_name.lower():
                target_project = p
                break
        if not target_project:
            cand_path = Path(project_name).expanduser().resolve()
            if cand_path.is_dir():
                ws = WorkspaceManager(cand_path)
                vm = VaultManager(workspace=ws)
                return None, ws, vm
            console.print(f"[bold red][x] Proyek '{project_name}' tidak ditemukan.[/bold red]")
            raise typer.Exit(code=1)
    else:
        # Check current working directory
        cwd = Path.cwd().resolve()
        if (cwd / "vault").is_dir() or (cwd / ".scripcy").exists():
            ws = WorkspaceManager(cwd)
            vm = VaultManager(workspace=ws)
            return None, ws, vm

        # Check active project in registry
        target_project = pm.get_active_project()
        if not target_project:
            projects = pm.list_projects()
            if projects:
                target_project = projects[0]

    if target_project:
        ws = WorkspaceManager(Path(target_project.path))
        vm = VaultManager(workspace=ws)
        return target_project, ws, vm

    # Fallback to current working directory
    ws = WorkspaceManager(Path.cwd().resolve())
    vm = VaultManager(workspace=ws)
    return None, ws, vm


# =============================================================================
# VAULT SUBCOMMANDS
# =============================================================================

@vault_app.command("list")
def vault_list(
    category: str = typer.Option("all", "--category", "-c", help="Filter kategori: all, papers, web, tables, images"),
    project: str | None = typer.Option(None, "--project", "-p", help="Nama atau direktori proyek riset"),
    as_json: bool = typer.Option(False, "--json", help="Tampilkan keluaran dalam format JSON terstruktur"),
) -> None:
    """Tampilkan daftar seluruh berkas yang tersimpan di Vault aktif."""
    _, ws, vm = resolve_vault(project)
    cat_filter = None if category == "all" else category.lower()
    items = vm.list_active_files(category=cat_filter)

    if as_json:
        data = []
        for it in items:
            data.append({
                "id": it.stem,
                "title": it.title,
                "filename": it.filename,
                "category": it.category,
                "context": it.context,
                "status": "error" if it.has_error else ("ready" if it.is_cached else "needs_update"),
                "is_cached": it.is_cached,
                "has_error": it.has_error,
                "tables_count": it.tables_count,
                "visuals_count": it.visuals_count,
                "size_bytes": it.size_bytes,
                "readable_size": it.readable_size,
                "updated_at": it.updated_at,
                "path": str(it.path),
            })
        sys.stdout.write(json.dumps(data, indent=2) + "\n")
        return

    if not items:
        console.print(f"[dim]Tidak ada berkas di Vault (filter: {category}).[/dim]")
        return

    table = Table(
        title=f"Daftar Berkas Vault ({len(items)} berkas · Kategori: {category})",
        title_style="bold cyan",
        border_style="cyan",
        header_style="bold bright_cyan",
    )
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Judul", style="bold white", min_width=25)
    table.add_column("Kategori", justify="center", style="cyan", width=10)
    table.add_column("Konteks", justify="center", width=14)
    table.add_column("Status", justify="center", width=14)
    table.add_column("Tabel", justify="right", style="dim", width=7)
    table.add_column("Visual", justify="right", style="dim", width=7)
    table.add_column("Update", justify="center", style="dim", width=12)

    for idx, it in enumerate(items, 1):
        if it.has_error:
            status_badge = "[bold red]Error[/bold red]"
        elif not it.is_cached:
            status_badge = "[bold cyan]Perlu Update[/bold cyan]"
        else:
            status_badge = "[bold green]Siap[/bold green]"

        ctx_badge = it.context if it.context else "-"
        table.add_row(
            str(idx),
            (it.title or it.stem)[:36],
            it.category.title(),
            ctx_badge[:14],
            status_badge,
            str(it.tables_count),
            str(it.visuals_count),
            it.updated_at or "-",
        )

    console.print(table)


@vault_app.command("add")
def vault_add(
    source: str = typer.Argument(..., help="Path berkas lokal atau URL dokumen (PDF/DOCX/TXT/MD)"),
    category: str | None = typer.Option(None, "--category", "-c", help="Kategori target: papers atau web"),
    context: str = typer.Option("", "--context", help="Konteks eksperimen / deskripsi berkas"),
    project: str | None = typer.Option(None, "--project", "-p", help="Nama atau direktori proyek riset"),
    as_json: bool = typer.Option(False, "--json", help="Tampilkan keluaran dalam format JSON terstruktur"),
) -> None:
    """Tambahkan berkas lokal atau unduh dokumen dari URL ke dalam Vault."""
    _, ws, vm = resolve_vault(project)
    clean_src = source.strip().strip('"\'')

    if clean_src.startswith(("http://", "https://")):
        try:
            parsed_url = urllib.parse.urlparse(clean_src)
            suggested_name = Path(parsed_url.path).name or "downloaded_paper.pdf"
            if not suggested_name.lower().endswith((".pdf", ".docx", ".txt", ".md", ".html", ".csv")):
                suggested_name += ".pdf"
            temp_dir = Path(tempfile.gettempdir())
            dest_file = temp_dir / suggested_name
            req = urllib.request.Request(
                clean_src,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp, open(dest_file, "wb") as out_f:
                out_f.write(resp.read())
            src_path = dest_file
        except Exception as exc:
            if as_json:
                sys.stdout.write(json.dumps({"status": "error", "message": f"Download failed: {exc}"}, indent=2) + "\n")
                raise typer.Exit(code=1)
            console.print(f"[bold red][x] Gagal mengunduh URL:[/bold red] {exc}")
            raise typer.Exit(code=1)
    else:
        src_path = Path(clean_src).expanduser().resolve()
        if not src_path.exists():
            if as_json:
                sys.stdout.write(json.dumps({"status": "error", "message": f"File not found: {src_path}"}, indent=2) + "\n")
                raise typer.Exit(code=1)
            console.print(f"[bold red][x] Berkas tidak ditemukan:[/bold red] '{src_path}'")
            raise typer.Exit(code=1)

    try:
        item = vm.add_resource(src_path)
        if context:
            item.context = context
            cache_file = ws.get_cache_path(f"{item.path.stem}.md")
            if cache_file.exists():
                try:
                    import re
                    txt = cache_file.read_text(encoding="utf-8")
                    if txt.startswith("---"):
                        txt = re.sub(r"^---\n", f"---\ncontext: {context}\n", txt, count=1)
                        cache_file.write_text(txt, encoding="utf-8")
                except Exception:
                    pass

        vm.record_history(
            action="tambah_berkas",
            description=f"Menambahkan {item.filename} ({item.readable_size}) ke Vault secara headless",
            files_affected=[item.filename],
            details={"source": clean_src, "category": item.category, "context": context},
        )
    except Exception as exc:
        if as_json:
            sys.stdout.write(json.dumps({"status": "error", "message": str(exc)}, indent=2) + "\n")
            raise typer.Exit(code=1)
        console.print(f"[bold red][x] Gagal menambahkan berkas ke Vault:[/bold red] {exc}")
        raise typer.Exit(code=1)

    if as_json:
        res = {
            "status": "success",
            "id": item.stem,
            "filename": item.filename,
            "category": item.category,
            "context": item.context,
            "size": item.readable_size,
            "path": str(item.path),
        }
        sys.stdout.write(json.dumps(res, indent=2) + "\n")
        return

    console.print(f"[bold green][✓] Berkas berhasil ditambahkan ke Vault:[/bold green] [bold cyan]{item.filename}[/bold cyan]")
    console.print(f"    Kategori: {item.category.title()} | Format: {item.extension.upper().lstrip('.')} | Ukuran: {item.readable_size}")


@vault_app.command("update")
def vault_update(
    on_duplicate: str = typer.Option("keep-both", "--on-duplicate", help="Strategi duplikat berkas local/: keep-both, overwrite, skip"),
    force: bool = typer.Option(False, "--force", "-f", help="Paksa konversi ulang semua berkas markdown cache"),
    project: str | None = typer.Option(None, "--project", "-p", help="Nama atau direktori proyek riset"),
    as_json: bool = typer.Option(False, "--json", help="Tampilkan keluaran dalam format JSON terstruktur"),
) -> None:
    """Pindai staging local/, urai konteks subfolder, dan sinkronisasi cache Markdown."""
    _, ws, vm = resolve_vault(project)

    # Map on_duplicate strategy
    dup_mode = on_duplicate.lower().replace("-", "_")
    if dup_mode in ("skip", "discard"):
        dup_act = "discard"
    elif dup_mode == "overwrite":
        dup_act = "overwrite"
    else:
        dup_act = "keep_both"

    # 1. Scan local staging
    local_items = vm.scan_local_staging()
    processed_files: list[str] = []

    if local_items:
        for item in local_items:
            act = dup_act if item.get("is_duplicate") else "keep_both"
            ctx = item.get("folder_context", "")
            dest = vm.process_local_item(
                item_info=item,
                action=act,
                context_override=ctx,
            )
            if dest:
                processed_files.append(item["filename"])

        if processed_files:
            vm.record_history(
                action="sortir_mandiri",
                description=f"Menyortir & memindahkan {len(processed_files)} berkas dari local/ ke Vault (headless)",
                files_affected=processed_files,
                details={"duplicate_strategy": dup_act},
            )

    # 2. Reload and convert
    total, converted = vm.reload_vault(force=force)

    vm.record_history(
        action="update_berkas",
        description=f"Sinkronisasi vault headless: {total} total berkas, {converted} diperbarui",
        files_affected=[],
        details={"total": total, "converted": converted, "force": force},
    )

    if as_json:
        res = {
            "status": "success",
            "total_files": total,
            "converted_files": converted,
            "local_processed": processed_files,
        }
        sys.stdout.write(json.dumps(res, indent=2) + "\n")
        return

    if processed_files:
        console.print(f"[bold green][✓] {len(processed_files)} berkas dari local/ berhasil disortir dan dipindahkan ke Vault.[/bold green]")
    console.print(f"[bold green][✓] Sinkronisasi Vault selesai:[/bold green] {total} total berkas, [bold cyan]{converted} diperbarui[/bold cyan] ke cache.")


@vault_app.command("search")
def vault_search(
    query: str = typer.Argument(..., help="Kata kunci pencarian atau query semantik"),
    category: str = typer.Option("all", "--category", "-c", help="Filter kategori: all, papers, web, tables, images"),
    semantic: bool = typer.Option(False, "--semantic", "-s", help="Gunakan hybrid semantic vector search"),
    limit: int = typer.Option(10, "--limit", "-l", help="Jumlah maksimal hasil yang ditampilkan"),
    project: str | None = typer.Option(None, "--project", "-p", help="Nama atau direktori proyek riset"),
    as_json: bool = typer.Option(False, "--json", help="Tampilkan keluaran dalam format JSON terstruktur"),
) -> None:
    """Cari dokumen, tabel, atau visual di dalam Vault (substring atau hybrid vector)."""
    _, ws, vm = resolve_vault(project)

    results: list[dict[str, Any]] = []

    if semantic:
        hybrid_res = vm.vector_vault.hybrid_search_rrf(query=query, top_k=limit)
        for r in hybrid_res:
            results.append({
                "id": r.chunk_id,
                "title": r.paper_id,
                "type": getattr(r, "chunk_type", "chunk"),
                "score": round(r.rrf_score, 4),
                "snippet": r.content[:200],
                "caption": getattr(r, "caption", ""),
                "page_num": getattr(r, "page_num", 1),
                "tables": getattr(r, "tables", []),
                "images": getattr(r, "images", []),
            })
    else:
        cat_filter = None if category == "all" else category.lower()
        active = vm.list_active_files(category=cat_filter)
        q_low = query.strip().lower()
        matches = [
            it for it in active
            if q_low in it.title.lower()
            or q_low in it.stem.lower()
            or q_low in it.filename.lower()
            or q_low in it.cite_key.lower()
            or any(q_low in a.lower() for a in it.authors)
        ][:limit]

        for m in matches:
            results.append({
                "id": m.stem,
                "title": m.title or m.stem,
                "category": m.category,
                "context": m.context,
                "filename": m.filename,
                "tables_count": m.tables_count,
                "visuals_count": m.visuals_count,
            })

    if as_json:
        sys.stdout.write(json.dumps(results, indent=2) + "\n")
        return

    if not results:
        console.print(f"[dim]Tidak ditemukan hasil pencarian untuk '{query}'.[/dim]")
        return

    table = Table(
        title=f"Hasil Pencarian Vault: '{query}' ({'Vektor Semantik' if semantic else 'Kata Kunci'})",
        title_style="bold cyan",
        border_style="cyan",
        header_style="bold bright_cyan",
    )
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Judul / Target", style="bold white", min_width=25)
    table.add_column("Tipe", justify="center", style="cyan", width=12)
    table.add_column("Detail / Snippet", style="white")

    for idx, r in enumerate(results, 1):
        if semantic:
            detail = r.get("caption") or r.get("snippet", "")
            table.add_row(str(idx), r["title"][:34], r["type"], detail[:60] + "...")
        else:
            table.add_row(
                str(idx),
                r["title"][:34],
                r["category"].title(),
                f"Konteks: {r.get('context') or '-'} · Tabel: {r.get('tables_count', 0)} · Visual: {r.get('visuals_count', 0)}",
            )

    console.print(table)


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert Pydantic models and Path objects to serializable primitives."""
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "model_dump"):
        return _sanitize_for_json(obj.model_dump())
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


@vault_app.command("status")
def vault_status(
    project: str | None = typer.Option(None, "--project", "-p", help="Nama atau direktori proyek riset"),
    as_json: bool = typer.Option(False, "--json", help="Tampilkan keluaran dalam format JSON terstruktur"),
) -> None:
    """Tampilkan status kesehatan Vault, breakdown kategori berkas, dan waktu update."""
    _, ws, vm = resolve_vault(project)
    health = vm.check_vault_health()

    if as_json:
        sys.stdout.write(json.dumps(_sanitize_for_json(health), indent=2) + "\n")
        return

    table = Table(
        title="Status Kesehatan & Inventaris Vault",
        title_style="bold cyan",
        border_style="cyan",
        header_style="bold bright_cyan",
    )
    table.add_column("Metrik / Kategori", style="bold white", width=26)
    table.add_column("Nilai / Rincian", style="white")

    st_val = health.get("status", "active")
    st_color = "bold green" if st_val in ("active", "healthy") else ("bold yellow" if st_val == "warning" else "bold red")
    table.add_row("Status Kesehatan", f"[{st_color}]{st_val.upper()}[/{st_color}]")
    table.add_row("Terakhir Update", f"[bold cyan]{health.get('last_updated', '-')}[/bold cyan]")
    table.add_row("Total Berkas Aktif", str(health.get("total_files", 0)))

    cat_h = health.get("category_health", {})
    papers_stat = cat_h.get("papers", {})
    web_stat = cat_h.get("web", {})
    tables_stat = cat_h.get("tables", {})
    images_stat = cat_h.get("images", {})

    table.add_row("Papers (Jurnal/Artikel)", f"{papers_stat.get('total', 0)} berkas")
    table.add_row("Web Resources", f"{web_stat.get('total', 0)} berkas")
    table.add_row("Tabel Terindeks", f"{tables_stat.get('total', 0)} tabel")
    table.add_row("Visual & Gambar", f"{images_stat.get('total', 0)} gambar")
    table.add_row("Berkas Bermasalah", str(health.get("error_count", 0)))

    console.print(table)


@vault_app.command("history")
def vault_history(
    limit: int = typer.Option(20, "--limit", "-l", help="Jumlah log aktivitas yang ditampilkan"),
    project: str | None = typer.Option(None, "--project", "-p", help="Nama atau direktori proyek riset"),
    as_json: bool = typer.Option(False, "--json", help="Tampilkan keluaran dalam format JSON terstruktur"),
) -> None:
    """Tampilkan riwayat log aktivitas, penambahan, dan sinkronisasi Vault."""
    _, ws, vm = resolve_vault(project)
    entries = vm.list_history(limit=limit)

    if as_json:
        sys.stdout.write(json.dumps(entries, indent=2) + "\n")
        return

    if not entries:
        console.print("[dim]Belum ada riwayat aktivitas yang tercatat di Vault.[/dim]")
        return

    table = Table(
        title=f"Riwayat Aktivitas Vault ({len(entries)} entri terakhir)",
        title_style="bold cyan",
        border_style="cyan",
        header_style="bold bright_cyan",
    )
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Waktu", style="dim", width=19)
    table.add_column("Aksi", style="bold cyan", width=16)
    table.add_column("Deskripsi Aktivitas", style="white")

    for idx, e in enumerate(entries, 1):
        ts = e.get("timestamp", "-")
        act = e.get("action", "-")
        desc = e.get("description", "-")
        table.add_row(str(idx), ts[:19], act, desc)

    console.print(table)


@vault_app.command("remove")
def vault_remove(
    identifier: str = typer.Argument(..., help="Nama berkas atau stem dokumen yang akan diarsipkan"),
    project: str | None = typer.Option(None, "--project", "-p", help="Nama atau direktori proyek riset"),
    as_json: bool = typer.Option(False, "--json", help="Tampilkan keluaran dalam format JSON terstruktur"),
) -> None:
    """Arsipkan dokumen dari Vault aktif ke folder .archive/."""
    _, ws, vm = resolve_vault(project)
    try:
        archived = vm.archive_resource(identifier)
        vm.record_history(
            action="arsipkan_berkas",
            description=f"Mengarsipkan {archived.filename} dari Vault aktif (headless)",
            files_affected=[archived.filename],
        )
    except Exception as exc:
        if as_json:
            sys.stdout.write(json.dumps({"status": "error", "message": str(exc)}, indent=2) + "\n")
            raise typer.Exit(code=1)
        console.print(f"[bold red][x] Gagal mengarsipkan berkas:[/bold red] {exc}")
        raise typer.Exit(code=1)

    if as_json:
        sys.stdout.write(json.dumps({"status": "archived", "filename": archived.filename}, indent=2) + "\n")
        return

    console.print(f"[bold green][✓] Berkas '{archived.filename}' berhasil dipindahkan ke Arsip.[/bold green]")


@vault_app.command("restore")
def vault_restore(
    identifier: str = typer.Argument(..., help="Nama berkas atau stem dokumen di arsip yang akan dipulihkan"),
    project: str | None = typer.Option(None, "--project", "-p", help="Nama atau direktori proyek riset"),
    as_json: bool = typer.Option(False, "--json", help="Tampilkan keluaran dalam format JSON terstruktur"),
) -> None:
    """Pulihkan berkas yang diarsipkan kembali ke Vault aktif."""
    _, ws, vm = resolve_vault(project)
    try:
        restored = vm.restore_resource(identifier)
        vm.record_history(
            action="pulihkan_berkas",
            description=f"Memulihkan {restored.filename} dari arsip ke Vault aktif (headless)",
            files_affected=[restored.filename],
        )
    except Exception as exc:
        if as_json:
            sys.stdout.write(json.dumps({"status": "error", "message": str(exc)}, indent=2) + "\n")
            raise typer.Exit(code=1)
        console.print(f"[bold red][x] Gagal memulihkan berkas dari arsip:[/bold red] {exc}")
        raise typer.Exit(code=1)

    if as_json:
        sys.stdout.write(json.dumps({"status": "restored", "filename": restored.filename}, indent=2) + "\n")
        return

    console.print(f"[bold green][✓] Berkas '{restored.filename}' berhasil dipulihkan ke Vault aktif.[/bold green]")


@vault_app.command("archived")
def vault_archived(
    project: str | None = typer.Option(None, "--project", "-p", help="Nama atau direktori proyek riset"),
    as_json: bool = typer.Option(False, "--json", help="Tampilkan keluaran dalam format JSON terstruktur"),
) -> None:
    """Tampilkan daftar seluruh berkas yang berada di dalam folder arsip."""
    _, ws, vm = resolve_vault(project)
    arch = vm.list_archived_files()

    if as_json:
        data = [{"filename": it.filename, "size": it.readable_size, "path": str(it.path)} for it in arch]
        sys.stdout.write(json.dumps(data, indent=2) + "\n")
        return

    if not arch:
        console.print("[dim]Folder arsip kosong.[/dim]")
        return

    table = Table(
        title=f"Daftar Berkas Terarsip ({len(arch)} berkas)",
        title_style="bold cyan",
        border_style="cyan",
        header_style="bold bright_cyan",
    )
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Nama Berkas", style="bold white")
    table.add_column("Ukuran", justify="right", style="dim", width=12)

    for idx, it in enumerate(arch, 1):
        table.add_row(str(idx), it.filename, it.readable_size)

    console.print(table)


@vault_app.command("reindex")
def vault_reindex(
    project: str | None = typer.Option(None, "--project", "-p", help="Nama atau direktori proyek riset"),
    as_json: bool = typer.Option(False, "--json", help="Tampilkan keluaran dalam format JSON terstruktur"),
) -> None:
    """Bangun ulang indeks visual dan vektor untuk seluruh gambar & tabel Vault."""
    _, ws, vm = resolve_vault(project)
    counts = vm.clean_and_reindex_visuals()

    vm.record_history(
        action="reindex_visual",
        description=f"Reindex visual & tabel: {counts.get('tables', 0)} tabel, {counts.get('images', 0)} gambar (headless)",
        files_affected=[],
        details=counts,
    )

    if as_json:
        sys.stdout.write(json.dumps({"status": "success", "counts": counts}, indent=2) + "\n")
        return

    console.print("[bold green][✓] Pembangunan ulang indeks visual selesai:[/bold green]")
    console.print(f"    • Tabel terindeks  : [bold cyan]{counts.get('tables', 0)}[/bold cyan]")
    console.print(f"    • Gambar terindeks : [bold cyan]{counts.get('images', 0)}[/bold cyan]")


@vault_app.command("tables")
def vault_tables(
    project: str | None = typer.Option(None, "--project", "-p", help="Nama atau direktori proyek riset"),
    as_json: bool = typer.Option(False, "--json", help="Tampilkan keluaran dalam format JSON terstruktur"),
) -> None:
    """Tampilkan daftar seluruh tabel yang terindeks di dalam Vault."""
    _, ws, vm = resolve_vault(project)
    tables = vm.list_tables()

    if as_json:
        sys.stdout.write(json.dumps(tables, indent=2) + "\n")
        return

    if not tables:
        console.print("[dim]Belum ada tabel yang terindeks di Vault.[/dim]")
        return

    table = Table(
        title=f"Daftar Tabel Terindeks ({len(tables)} tabel)",
        title_style="bold cyan",
        border_style="cyan",
        header_style="bold bright_cyan",
    )
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Nama / Caption", style="bold white", min_width=25)
    table.add_column("Sumber Dokumen", style="cyan", width=22)
    table.add_column("Konteks", style="dim", width=18)

    for idx, t in enumerate(tables, 1):
        cap = t.get("caption") or t.get("filename", "-")
        p_id = t.get("paper_id") or "-"
        ctx = t.get("context") or "-"
        table.add_row(str(idx), cap[:36], p_id[:20], ctx[:16])

    console.print(table)


@vault_app.command("images")
def vault_images(
    project: str | None = typer.Option(None, "--project", "-p", help="Nama atau direktori proyek riset"),
    as_json: bool = typer.Option(False, "--json", help="Tampilkan keluaran dalam format JSON terstruktur"),
) -> None:
    """Tampilkan daftar seluruh gambar dan visual yang terindeks di dalam Vault."""
    _, ws, vm = resolve_vault(project)
    images = vm.list_images()

    if as_json:
        sys.stdout.write(json.dumps(images, indent=2) + "\n")
        return

    if not images:
        console.print("[dim]Belum ada gambar yang terindeks di Vault.[/dim]")
        return

    table = Table(
        title=f"Daftar Visual & Gambar Terindeks ({len(images)} gambar)",
        title_style="bold cyan",
        border_style="cyan",
        header_style="bold bright_cyan",
    )
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Nama / Caption", style="bold white", min_width=25)
    table.add_column("Sumber Dokumen", style="cyan", width=22)
    table.add_column("Konteks", style="dim", width=18)

    for idx, img in enumerate(images, 1):
        cap = img.get("caption") or img.get("filename", "-")
        p_id = img.get("paper_id") or "-"
        ctx = img.get("context") or "-"
        table.add_row(str(idx), cap[:36], p_id[:20], ctx[:16])

    console.print(table)
