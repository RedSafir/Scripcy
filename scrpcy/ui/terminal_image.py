"""Universal Terminal Image Renderer using ANSI 24-bit TrueColor Half-Blocks.

Renders images in full 24-bit RGB color across all terminal emulators
(Windows Terminal, Windows Console Host, PowerShell, CMD, VSCode, Git Bash,
Linux, macOS, SSH) with 1:1 pixel aspect-ratio correction and zero external C dependencies.
"""

from __future__ import annotations

import io
import math
import sys
from pathlib import Path
from typing import Any

from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


def init_virtual_terminal() -> None:
    """Initialize Windows Console VT processing and UTF-8 output encoding."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        # Enable UTF-8 code page on Windows
        try:
            kernel32.SetConsoleOutputCP(65001)
            kernel32.SetConsoleCP(65001)
        except Exception:
            pass

        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            if hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

        # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        target_mode = 0x0001 | 0x0002 | 0x0004 | 0x0008
        for handle_id in (-11, -12):  # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
            h = kernel32.GetStdHandle(handle_id)
            if h and h != -1:
                mode = ctypes.c_ulong()
                if kernel32.GetConsoleMode(h, ctypes.byref(mode)):
                    kernel32.SetConsoleMode(h, mode.value | target_mode)
    except Exception:
        pass


def render_image_to_ansi(
    image_path: str | Path | bytes,
    max_cols: int = 56,
    max_rows: int = 32,
) -> str:
    """Render an image into a 24-bit TrueColor Half-Block ANSI string.

    A monospace terminal character cell is roughly 1:2 (width:height).
    The unicode upper half block '▀' (U+2580) splits a cell vertically into two square pixels:
    - Foreground color sets the top half-pixel.
    - Background color sets the bottom half-pixel.
    This produces a 1:1 physical pixel aspect ratio in all terminals.
    """
    init_virtual_terminal()

    if isinstance(image_path, (str, Path)):
        p = Path(image_path)
        if not p.exists():
            return f"[Gambar tidak ditemukan: {p}]"
        img = Image.open(p)
    elif isinstance(image_path, bytes):
        img = Image.open(io.BytesIO(image_path))
    else:
        return "[Format gambar tidak didukung]"

    try:
        # Handle transparency by composite over dark background
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            bg = Image.new("RGB", img.size, (18, 18, 18))
            rgba = img.convert("RGBA")
            bg.paste(rgba, mask=rgba.split()[3])
            img = bg
        else:
            img = img.convert("RGB")

        w_orig, h_orig = img.size
        if w_orig == 0 or h_orig == 0:
            return ""

        ar = w_orig / float(h_orig)
        target_w = max(10, min(max_cols, w_orig))
        # Each terminal line contains 2 vertical pixel rows
        target_h = max(2, int(round(target_w / ar)))

        # Constrain max height (number of text lines = target_h // 2)
        if (target_h // 2) > max_rows:
            target_h = max_rows * 2
            target_w = max(10, int(round(target_h * ar)))

        # Ensure even height for exact top/bottom pairs
        if target_h % 2 != 0:
            target_h += 1

        resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        pixels = resized.load()

        lines: list[str] = []
        # Step 2 rows per character cell
        for y in range(0, target_h, 2):
            row_parts: list[str] = []
            prev_fg: tuple[int, int, int] | None = None
            prev_bg: tuple[int, int, int] | None = None

            for x in range(target_w):
                top_r, top_g, top_b = pixels[x, y]
                bot_r, bot_g, bot_b = pixels[x, y + 1] if (y + 1) < target_h else (0, 0, 0)

                cur_fg = (top_r, top_g, top_b)
                cur_bg = (bot_r, bot_g, bot_b)

                # Run-length caching: emit escape sequences only when RGB changes
                fg_code = f"\x1b[38;2;{top_r};{top_g};{top_b}m" if cur_fg != prev_fg else ""
                bg_code = f"\x1b[48;2;{bot_r};{bot_g};{bot_b}m" if cur_bg != prev_bg else ""

                if fg_code:
                    prev_fg = cur_fg
                if bg_code:
                    prev_bg = cur_bg

                row_parts.append(f"{fg_code}{bg_code}▀")

            row_parts.append("\x1b[0m")
            lines.append("".join(row_parts))

        return "\n".join(lines)
    finally:
        try:
            img.close()
        except Exception:
            pass


def render_visual_panel(
    visual_info: dict[str, Any],
    console: Console | None = None,
    max_cols: int = 56,
) -> Panel | None:
    """Wrap visual asset (image or table) into a clean, publication-ready Rich panel.

    Includes explicit paper citation, page number, caption, and ANSI rendering.
    """
    init_virtual_terminal()
    file_path_str = visual_info.get("file_path") or visual_info.get("csv_path") or ""
    file_path = Path(file_path_str) if file_path_str else None

    paper_id = visual_info.get("paper_id", "Dokumen Vault")
    paper_title = visual_info.get("paper_title") or paper_id
    page_num = visual_info.get("page_num", 1)
    caption = visual_info.get("caption") or f"Visual Halaman {page_num}"
    ai_analysis = visual_info.get("ai_analysis") or ""
    asset_type = visual_info.get("asset_type", "image")

    content_text = Text()

    # If it's an image, render ANSI truecolor half-blocks
    if asset_type == "image" and file_path and file_path.exists():
        ansi_art = render_image_to_ansi(file_path, max_cols=max_cols)
        content_text.append(Text.from_ansi(ansi_art))
        content_text.append("\n\n")
    elif asset_type == "table":
        md_p = Path(visual_info.get("md_path") or "")
        if md_p.exists():
            content_text.append(f"{md_p.read_text(encoding='utf-8')[:800]}\n\n")

    # Metadata & Academic Citation
    content_text.append("● Sumber Paper : ", style="bold cyan")
    content_text.append(f"{paper_title} ", style="bold bright_white")
    content_text.append(f"[{paper_id}]\n", style="dim")

    width = visual_info.get("width")
    height = visual_info.get("height")
    if width and height:
        content_text.append("● Dimensi      : ", style="bold cyan")
        content_text.append(f"{width} x {height} px\n", style="white")

    content_text.append("● Lokasi       : ", style="bold cyan")
    content_text.append(f"Halaman {page_num}\n", style="white")

    content_text.append("● Keterangan   : ", style="bold cyan")
    content_text.append(f"{caption}\n", style="white")

    if ai_analysis and "JUNK_IMAGE" not in ai_analysis and "Gambar ilmiah terindeks" not in ai_analysis:
        content_text.append("● Analisis Data: ", style="bold cyan")
        content_text.append(f"{ai_analysis}\n", style="italic bright_white")

    if file_path and file_path.exists():
        content_text.append(f"Berkas: {file_path}", style="dim")

    panel = Panel(
        content_text,
        title=f"[bold bright_cyan]Gambar #{visual_info.get('index', 1)} • Rujukan Visual Ilmiah #{visual_info.get('index', 1)}[/bold bright_cyan]",
        subtitle="[dim]vault/images • TrueColor Half-Block[/dim]",
        border_style="cyan",
        expand=False,
    )
    return panel
