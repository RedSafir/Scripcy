"""Interactive Antigravity-style TUI selectors for models and skills."""

from __future__ import annotations

import sys
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text


def init_windows_terminal() -> None:
    """Initialize Windows Console VT processing and UTF-8 encoding."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        try:
            kernel32.SetConsoleOutputCP(65001)
            kernel32.SetConsoleCP(65001)
        except Exception:
            pass

        target_mode = 0x0001 | 0x0002 | 0x0004 | 0x0008
        for handle_id in (-11, -12):
            h = kernel32.GetStdHandle(handle_id)
            if h and h != -1:
                mode = ctypes.c_ulong()
                if kernel32.GetConsoleMode(h, ctypes.byref(mode)):
                    kernel32.SetConsoleMode(h, mode.value | target_mode)

        GENERIC_READ = 0x80000000
        GENERIC_WRITE = 0x40000000
        FILE_SHARE_WRITE = 2
        OPEN_EXISTING = 3
        h_con = kernel32.CreateFileW(
            "CONOUT$",
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_WRITE,
            None,
            OPEN_EXISTING,
            0,
            None,
        )
        if h_con and h_con != -1:
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(h_con, ctypes.byref(mode)):
                kernel32.SetConsoleMode(h_con, mode.value | target_mode)
            kernel32.CloseHandle(h_con)
    except Exception:
        pass


init_windows_terminal()


def _clear_lines(num_lines: int, console: Console | None = None) -> None:
    """Cross-platform zero-flicker line clear: erases N lines upward without stacking."""
    if num_lines <= 0:
        return
    sys.stdout.write(f"\x1b[{num_lines}A\r\x1b[J")
    sys.stdout.flush()


def _read_key() -> str:
    """Read a single keypress or escape sequence cross-platform."""
    if sys.platform == "win32":
        import msvcrt

        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            # Special arrow or function key
            code = msvcrt.getwch()
            if code == "H":
                return "UP"
            elif code == "P":
                return "DOWN"
            elif code == "K":
                return "LEFT"
            elif code == "M":
                return "RIGHT"
            return "SPECIAL"
        elif ch == "\r" or ch == "\n":
            return "ENTER"
        elif ch == "\t":
            return "TAB"
        elif ch == "\x1b":
            return "ESC"
        elif ch == "\x08" or ch == "\x7f":
            return "BACKSPACE"
        return ch
    else:
        # Basic POSIX fallback
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                # Check for escape sequence
                next1 = sys.stdin.read(1)
                if next1 == "[":
                    next2 = sys.stdin.read(1)
                    if next2 == "A":
                        return "UP"
                    elif next2 == "B":
                        return "DOWN"
                    elif next2 == "D":
                        return "LEFT"
                    elif next2 == "C":
                        return "RIGHT"
                return "ESC"
            elif ch in ("\r", "\n"):
                return "ENTER"
            elif ch == "\t":
                return "TAB"
            elif ch in ("\x7f", "\x08"):
                return "BACKSPACE"
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _format_model_display_name(model_id: str) -> str:
    """Format model identifier into human-friendly Antigravity-style label."""
    clean = model_id
    if clean.startswith("9router/"):
        raw = clean[len("9router/"):]
        if raw.startswith("ag/"):
            raw = raw[3:]
        parts = raw.replace("-", " ").replace("_", " ").split()
        capitalized = " ".join(p.capitalize() if not p.isupper() else p for p in parts)
        return f"{capitalized} (9Router)"
    elif clean.startswith("gemini/"):
        raw = clean[len("gemini/"):]
        return raw.replace("-", " ").title()
    elif clean.startswith("anthropic/"):
        raw = clean[len("anthropic/"):]
        return raw.replace("-", " ").title()
    elif clean.startswith("openai/"):
        raw = clean[len("openai/"):]
        return raw.upper() if "gpt" in raw else raw.title()
    elif clean.startswith("deepseek/"):
        raw = clean[len("deepseek/"):]
        return raw.replace("-", " ").title()
    return clean


format_model_display_name = _format_model_display_name


EFFORT_LEVELS = ["low", "medium", "high"]
EFFORT_DESCRIPTIONS = {
    "low": "Fastest responses for quick exploration & basic summarization",
    "medium": "Balanced reasoning for standard academic synthesis",
    "high": "Deepest reasoning for complex problems – slower but strongest",
}


def interactive_model_selector(
    current_model: str,
    current_effort: str,
    available_models: list[str],
    console: Console | None = None,
) -> tuple[str, str] | None:
    """Interactive Antigravity CLI model switcher with cursor navigation, search filter, and effort slider.

    Returns:
        (chosen_model_id, chosen_effort) or None if cancelled via Esc.
    """
    if not sys.stdin.isatty():
        return None

    c = console or Console(force_terminal=True)
    effort = current_effort.lower() if current_effort in EFFORT_LEVELS else "high"
    effort_idx = EFFORT_LEVELS.index(effort)

    # Ensure current_model is in the list
    all_models = list(available_models)
    if current_model and current_model not in all_models:
        all_models.insert(0, current_model)

    search_query = ""
    cursor_idx = 0

    # Initial cursor position on current_model if present
    for idx, m in enumerate(all_models):
        if m == current_model:
            cursor_idx = idx
            break

    def get_filtered() -> list[str]:
        if not search_query.strip():
            return all_models
        q = search_query.strip().lower()
        return [m for m in all_models if q in m.lower() or q in _format_model_display_name(m).lower()]

    # Hide cursor during rendering if possible
    c.show_cursor(False)
    prev_lines_drawn = 0
    try:
        while True:
            filtered = get_filtered()
            if not filtered:
                cursor_idx = 0
            else:
                cursor_idx = max(0, min(cursor_idx, len(filtered) - 1))

            # Build UI frame
            lines: list[Text] = []
            lines.append(Text("Switch Model", style="bold white"))
            lines.append(Text(""))

            # Search bar
            search_line = Text("  Search: ", style="dim")
            search_line.append(search_query, style="bold cyan")
            search_line.append("█", style="blink bold white")
            lines.append(search_line)
            lines.append(Text(""))

            # Visible window of models (max 7 items)
            window_size = 7
            start_window = max(0, min(cursor_idx - 3, len(filtered) - window_size))
            visible = filtered[start_window : start_window + window_size]

            if not filtered:
                lines.append(Text("  [No models match search]", style="dim yellow"))
            else:
                for idx_offset, m in enumerate(visible):
                    real_idx = start_window + idx_offset
                    is_cursor = real_idx == cursor_idx
                    is_current = m == current_model

                    display_name = _format_model_display_name(m)
                    item_text = Text()

                    if is_cursor:
                        item_text.append("> ", style="bold cyan")
                        item_text.append(display_name, style="bold bright_white")
                    else:
                        item_text.append("  ", style="dim")
                        item_text.append(display_name, style="white")

                    if is_current:
                        pad = max(1, 38 - len(display_name))
                        item_text.append(" " * pad)
                        item_text.append("(current)", style="dim cyan")

                    lines.append(item_text)

            lines.append(Text(""))

            # Effort slider
            # Effort  ◀ ────●────●────◎────▶
            #              low   medium  high
            #         Deepest reasoning for complex problems – slower but strongest
            active_effort = EFFORT_LEVELS[effort_idx]
            bullet_low = "◎" if active_effort == "low" else "●"
            bullet_med = "◎" if active_effort == "medium" else "●"
            bullet_high = "◎" if active_effort == "high" else "●"

            slider_line = Text("  Effort  ", style="bold white")
            slider_line.append("◀ ────", style="dim")
            slider_line.append(bullet_low, style="bold cyan" if active_effort == "low" else "dim")
            slider_line.append("────", style="dim")
            slider_line.append(bullet_med, style="bold cyan" if active_effort == "medium" else "dim")
            slider_line.append("────", style="dim")
            slider_line.append(bullet_high, style="bold cyan" if active_effort == "high" else "dim")
            slider_line.append("──── ▶", style="dim")
            lines.append(slider_line)

            labels_line = Text("               ")
            labels_line.append("low", style="bold cyan" if active_effort == "low" else "dim")
            labels_line.append("     ")
            labels_line.append("medium", style="bold cyan" if active_effort == "medium" else "dim")
            labels_line.append("    ")
            labels_line.append("high", style="bold cyan" if active_effort == "high" else "dim")
            lines.append(labels_line)

            desc = EFFORT_DESCRIPTIONS[active_effort]
            desc_line = Text(f"          {desc}", style="dim italic")
            lines.append(desc_line)
            lines.append(Text(""))

            # Keybindings footer
            footer = Text("  Keyboard: ", style="dim")
            footer.append("↑/↓", style="bold cyan")
            footer.append(" Navigate  ", style="dim")
            footer.append("←/→", style="bold cyan")
            footer.append(" Effort  ", style="dim")
            footer.append("enter", style="bold green")
            footer.append(" Select  ", style="dim")
            footer.append("esc", style="bold yellow")
            footer.append(" Go Back", style="dim")
            lines.append(footer)
            lines.append(Text(""))

            # Redraw in place without scrolling
            if prev_lines_drawn > 0:
                _clear_lines(prev_lines_drawn, c)

            for line in lines:
                c.print(line)
            sys.stdout.flush()
            prev_lines_drawn = len(lines)

            # Wait for key
            key = _read_key()

            if key == "ESC":
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)
                return None
            elif key == "ENTER":
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)
                chosen_m = filtered[cursor_idx] if filtered else current_model
                return chosen_m, active_effort
            elif key == "UP":
                cursor_idx = max(0, cursor_idx - 1)
            elif key == "DOWN":
                cursor_idx = min(len(filtered) - 1, cursor_idx + 1)
            elif key == "LEFT":
                effort_idx = max(0, effort_idx - 1)
            elif key == "RIGHT":
                effort_idx = min(len(EFFORT_LEVELS) - 1, effort_idx + 1)
            elif key == "BACKSPACE":
                if search_query:
                    search_query = search_query[:-1]
            elif len(key) == 1 and key.isprintable():
                search_query += key
                cursor_idx = 0

    finally:
        c.show_cursor(True)


def interactive_skill_selector(
    current_skill: str,
    available_skills: list[Any],
    console: Console | None = None,
) -> str | None:
    """Interactive Antigravity CLI skill switcher with cursor navigation and description display.

    Returns:
        chosen_skill_name or None if cancelled.
    """
    if not sys.stdin.isatty():
        return None

    c = console or Console(force_terminal=True)
    search_query = ""
    cursor_idx = 0

    # Locate current index
    for idx, s in enumerate(available_skills):
        if getattr(s, "name", "") == current_skill:
            cursor_idx = idx
            break

    def get_filtered() -> list[Any]:
        if not search_query.strip():
            return available_skills
        q = search_query.strip().lower()
        return [
            s
            for s in available_skills
            if q in getattr(s, "name", "").lower()
            or q in getattr(s, "description", "").lower()
        ]

    c.show_cursor(False)
    prev_lines_drawn = 0
    try:
        while True:
            filtered = get_filtered()
            if not filtered:
                cursor_idx = 0
            else:
                cursor_idx = max(0, min(cursor_idx, len(filtered) - 1))

            lines: list[Text] = []
            lines.append(Text("Switch Research Persona / Skill", style="bold white"))
            lines.append(Text(""))

            search_line = Text("  Search: ", style="dim")
            search_line.append(search_query, style="bold cyan")
            search_line.append("█", style="blink bold white")
            lines.append(search_line)
            lines.append(Text(""))

            for idx, s in enumerate(filtered):
                is_cursor = idx == cursor_idx
                s_name = getattr(s, "name", str(s))
                is_current = s_name == current_skill

                item_text = Text()
                if is_cursor:
                    item_text.append("> ", style="bold cyan")
                    item_text.append(s_name, style="bold bright_white")
                else:
                    item_text.append("  ", style="dim")
                    item_text.append(s_name, style="white")

                if is_current:
                    pad = max(1, 32 - len(s_name))
                    item_text.append(" " * pad)
                    item_text.append("(current)", style="dim cyan")

                lines.append(item_text)

            lines.append(Text(""))

            # Active skill description
            if filtered and 0 <= cursor_idx < len(filtered):
                active_s = filtered[cursor_idx]
                desc = getattr(active_s, "description", "").strip() or "Custom research persona."
                lines.append(Text(f"  Description: {desc}", style="dim italic"))
                lines.append(Text(""))

            # Keybindings footer
            footer = Text("  Keyboard: ", style="dim")
            footer.append("↑/↓", style="bold cyan")
            footer.append(" Navigate  ", style="dim")
            footer.append("enter", style="bold green")
            footer.append(" Select  ", style="dim")
            footer.append("esc", style="bold yellow")
            footer.append(" Go Back", style="dim")
            lines.append(footer)
            lines.append(Text(""))

            # Redraw in place without scrolling
            if prev_lines_drawn > 0:
                _clear_lines(prev_lines_drawn, c)

            for line in lines:
                c.print(line)
            sys.stdout.flush()
            prev_lines_drawn = len(lines)

            key = _read_key()

            if key == "ESC":
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)
                return None
            elif key == "ENTER":
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)
                if filtered:
                    return getattr(filtered[cursor_idx], "name", str(filtered[cursor_idx]))
                return current_skill
            elif key == "UP":
                cursor_idx = max(0, cursor_idx - 1)
            elif key == "DOWN":
                cursor_idx = min(len(filtered) - 1, cursor_idx + 1)
            elif key == "BACKSPACE":
                if search_query:
                    search_query = search_query[:-1]
            elif len(key) == 1 and key.isprintable():
                search_query += key
                cursor_idx = 0
    finally:
        c.show_cursor(True)


def interactive_action_confirm(action_text: str, console: Console | None = None) -> tuple[str, str | None]:
    """Prompt user with 3 options: Execute, Write-in advice/feedback, or Skip."""
    c = console or Console()

    # Enable ANSI Virtual Terminal processing on Windows
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass

    # Non-interactive fallback
    if not sys.stdin.isatty():
        c.print(f"[bold cyan]✦ Aksi Scrpcy Disarankan AI:[/bold cyan] [bold green]{action_text}[/bold green]")
        try:
            line = c.input("  [bold bright_white]Jalankan aksi ini sekarang? (y/n/saran):[/bold bright_white] ").strip()
            if line.lower() in ("", "y", "yes", "ya", "ok", "1"):
                return ("execute", None)
            elif line.lower() in ("n", "no", "tidak", "skip", "0"):
                return ("skip", None)
            else:
                return ("write_in", line)
        except Exception:
            return ("skip", None)

    options = [
        ("Ya, jalankan aksi ini", "execute"),
        ("Beri masukan / saran tambahan untuk AI (Write-in)", "write_in"),
        ("Tidak, lewati aksi ini", "skip"),
    ]
    cursor_idx = 0
    prev_lines_drawn = 0

    c.show_cursor(False)
    try:
        while True:
            lines_out: list[Text] = []
            header = Text("✦ Aksi Scrpcy Disarankan AI: ", style="bold cyan")
            header.append(action_text, style="bold bright_white")
            lines_out.append(header)

            for idx, (label, val) in enumerate(options):
                is_cursor = idx == cursor_idx
                item_text = Text()
                if is_cursor:
                    item_text.append("  > ", style="bold cyan")
                    marker_style = "bold green" if val == "execute" else ("bold yellow" if val == "write_in" else "bold red")
                    item_text.append("[x] ", style=marker_style)
                    item_text.append(label, style="bold bright_white")
                else:
                    item_text.append("    ", style="dim")
                    item_text.append("[ ] ", style="dim")
                    item_text.append(label, style="dim white")
                lines_out.append(item_text)

            footer = Text("  Navigasi: ", style="dim")
            footer.append("↑/↓", style="bold cyan")
            footer.append(" Pilih  ", style="dim")
            footer.append("Enter", style="bold green")
            footer.append(" Konfirmasi  ", style="dim")
            footer.append("Esc/n", style="bold yellow")
            footer.append(" Batal", style="dim")
            lines_out.append(footer)

            # Clean redraw: erase previous lines completely
            if prev_lines_drawn > 0:
                _clear_lines(prev_lines_drawn, c)

            for line in lines_out:
                c.print(line)
            sys.stdout.flush()
            prev_lines_drawn = len(lines_out)

            key = _read_key()
            if key in ("ESC", "n", "N", "q", "Q"):
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)
                return ("skip", None)
            elif key == "ENTER":
                choice = options[cursor_idx][1]
                # Cleanly erase options from screen so chat stays pristine
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)
                if choice == "write_in":
                    c.show_cursor(True)
                    c.print(f"[bold cyan]● Berikan masukan / instruksi tambahan untuk AI:[/bold cyan]")
                    try:
                        user_advice = c.input("  > ").strip()
                    except Exception:
                        user_advice = ""
                    return ("write_in", user_advice if user_advice else None)
                return (choice, None)
            elif key in ("UP", "LEFT"):
                cursor_idx = (cursor_idx - 1) % len(options)
            elif key in ("DOWN", "RIGHT", "TAB"):
                cursor_idx = (cursor_idx + 1) % len(options)
            elif key in ("y", "Y"):
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)
                return ("execute", None)
    finally:
        c.show_cursor(True)

def interactive_token_warning_confirm(
    chunk_count: int,
    doc_count: int,
    token_estimate: int,
    console: Console | None = None,
) -> str:
    """Display interactive warning for high token / large chunk operations with in-place cursor options.

    Returns:
        'proceed': Run full multi-document analysis with all chunks
        'limit': Restrict to standard 5 chunks for token conservation
        'cancel': Abort question and return to prompt
    """
    c = console or Console(force_terminal=True)
    if not sys.stdin.isatty():
        return "proceed"

    options: list[tuple[str, str, str]] = [
        (
            f"Lanjutkan analisis penuh ({chunk_count} chunks, ~{token_estimate:,} token)",
            "proceed",
            "Membaca seluruh dokumen untuk sintesis komprehensif",
        ),
        (
            "Batasi ke mode hemat (5 chunks teratas)",
            "limit",
            "Menghemat token dengan hanya menganalisis kutipan paling relevan",
        ),
        (
            "Batalkan pertanyaan",
            "cancel",
            "Kembali ke prompt tanpa memanggil model LLM",
        ),
    ]

    cursor_idx = 0
    prev_lines_drawn = 0

    c.show_cursor(False)
    try:
        while True:
            lines: list[Text] = []

            # Warning header
            header = Text("! Peringatan Konsumsi Token: ", style="bold yellow")
            header.append(
                f"Permintaan ini akan memproses {chunk_count} chunks dari {doc_count} dokumen (est. ~{token_estimate:,} token).",
                style="bold bright_white",
            )
            lines.append(header)

            sub_hint = Text("  Pilih tindakan untuk melanjutkan:", style="dim")
            lines.append(sub_hint)

            for idx, (label, val, desc) in enumerate(options):
                is_cursor = idx == cursor_idx
                item_text = Text()
                if is_cursor:
                    item_text.append("  > ", style="bold cyan")
                    item_text.append(f"{label:<52}", style="bold bright_white")
                    item_text.append(f"({desc})", style="dim")
                else:
                    item_text.append("    [ ] ", style="dim")
                    item_text.append(f"{label:<52}", style="dim white")
                    item_text.append(f"({desc})", style="dim")
                lines.append(item_text)

            footer = Text("  Navigasi: ", style="dim")
            footer.append("↑/↓", style="bold cyan")
            footer.append(" Pilih  ", style="dim")
            footer.append("Enter", style="bold green")
            footer.append(" Konfirmasi  ", style="dim")
            footer.append("Esc", style="bold yellow")
            footer.append(" Batal", style="dim")
            lines.append(footer)

            # Redraw in place without c.clear()
            if prev_lines_drawn > 0:
                _clear_lines(prev_lines_drawn, c)

            for line in lines:
                c.print(line)
            sys.stdout.flush()
            prev_lines_drawn = len(lines)

            key = _read_key()
            if key in ("ESC", "q", "Q"):
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)
                return "cancel"
            elif key == "ENTER":
                val = options[cursor_idx][1]
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)
                return val
            elif key in ("UP", "LEFT"):
                cursor_idx = (cursor_idx - 1) % len(options)
            elif key in ("DOWN", "RIGHT", "TAB"):
                cursor_idx = (cursor_idx + 1) % len(options)
    finally:
        c.show_cursor(True)


# =============================================================================
# SLASH COMMAND DEFINITIONS & AUTOCOMPLETE CATALOGS
# =============================================================================

SLASH_COMMANDS_DASHBOARD: list[tuple[str, str]] = [
    ("/projects", "List registered research projects (alias: /list)"),
    ("/new", "Create and connect a master project folder (/new <name>)"),
    ("/open", "Open an existing project folder (/open <#|name>)"),
    ("/delete", "Remove project from Scrpcy (/delete <#|name>)"),
    ("/models", "Manage LLM models, API keys, 9Router, and tokens"),
    ("/skills", "List available AI persona skills"),
    ("/settings", "Manage workspace and citation preferences"),
    ("/clear", "Clear terminal screen"),
    ("/help", "Open interactive help and keyboard shortcuts"),
    ("/exit", "Exit Scrpcy research studio"),
]

SLASH_COMMANDS_PROJECT: list[tuple[str, str]] = [
    ("/search", "Search literature on Semantic Scholar, ArXiv, OpenAlex"),
    ("/findpaper", "Search & scout literature on OpenAlex/S2"),
    ("/scout", "Intelligent literature scout pipeline & AI curation"),
    ("/vault", "Manage vault documents and soft-archive"),
    ("/docx", "Compile synthesized research into Microsoft Word (.docx)"),
    ("/chat", "Manage multi-session conversation history (/chat new|list)"),
    ("/skill", "Switch active research persona (/skill <name|#>)"),
    ("/models", "Manage LLM models, API keys, 9Router, and tokens"),
    ("/processing", "View and track background tasks and downloads"),
    ("/gdrive", "Ingest research paper directly from Google Drive"),
    ("/active", "Toggle active context papers for AI queries"),
    ("/rename", "Rename the current project (/rename <new_name>)"),
    ("/tokens", "Inspect project token usage statistics"),
    ("/status", "Lihat model aktif dan konsumsi token proyek saat ini"),
    ("/settings", "Manage workspace and citation settings"),
    ("/web", "Launch 3-panel Web Studio in browser"),
    ("/dashboard", "Return to Dashboard mode (alias: /close, /back)"),
    ("/clear", "Clear terminal screen"),
    ("/help", "Open interactive help and keyboard shortcuts"),
    ("/exit", "Exit Scrpcy research studio"),
]

SLASH_COMMANDS_CHAT: list[tuple[str, str]] = [
    ("/back", "Kembali ke menu daftar chat (skripsi/chat>)"),
    ("/status", "Lihat model aktif dan konsumsi token proyek saat ini"),
    ("/new", "Buat sesi percakapan baru (/new [nama])"),
    ("/resume", "Pindah ke sesi percakapan lain (/resume <#|nama>)"),
    ("/history", "Lihat riwayat percakapan sesi ini"),
    ("/clear", "Bersihkan pesan dalam sesi percakapan ini"),
    ("/export", "Ekspor percakapan sesi ini ke markdown"),
    ("/help", "Buka panduan perintah chat (alias: ?)"),
]

SLASH_COMMANDS_CHAT_HUB: list[tuple[str, str]] = [
    ("/list", "Tampilkan daftar sesi percakapan yang ada (alias: /ls)"),
    ("/new", "Buat sesi chat baru (/new [nama])"),
    ("/open", "Buka atau lanjutkan sesi percakapan (/open <#|nama>, alias: /resume)"),
    ("/delete", "Hapus satu atau banyak sesi (/delete <1, 2|all>, alias: /rm)"),
    ("/clear", "Bersihkan tampilan layar terminal"),
    ("/help", "Tampilkan panduan perintah chat hub (alias: ?)"),
    ("/back", "Kembali ke prompt project utama"),
]

SLASH_COMMANDS_VAULT: list[tuple[str, str]] = [
    ("/list", "Tampilkan seluruh daftar berkas di vault & status markdown (alias: /ls)"),
    ("/reload", "Pindai folder & konversi semua file baru/diedit ke Markdown"),
    ("/add", "Salin dokumen dari komputer ke vault dan langsung dikonversi"),
    ("/findpaper", "Cari, kurasi & unduh paper akademis (OpenAlex & S2)"),
    ("/gdrive", "Unduh file Google Drive langsung ke vault via link"),
    ("/remove", "Arsipkan berkas dari project aktif ke .archive/"),
    ("/archived", "Lihat daftar berkas yang sedang diarsipkan di .archive/"),
    ("/restore", "Kembalikan berkas arsip ke vault aktif"),
    ("/view", "Lihat pratinjau teks Markdown hasil konversi di terminal"),
    ("/clear", "Bersihkan tampilan layar terminal"),
    ("/help", "Tampilkan panduan perintah vault (alias: ?)"),
    ("/back", "Kembali ke prompt project utama"),
]

SLASH_COMMANDS_PROCESSING: list[tuple[str, str]] = [
    ("/list", "Tampilkan daftar detail semua background tasks (alias: /ls)"),
    ("/status", "Tampilkan ringkasan status background tasks"),
    ("/detail", "Lihat rincian lengkap task (/detail <#|id>)"),
    ("/log", "Lihat log aktivitas background task (/log <#|id>)"),
    ("/cancel", "Batalkan task yang sedang berjalan (/cancel <#|id>)"),
    ("/clear", "Bersihkan riwayat task selesai atau dibatalkan"),
    ("/help", "Tampilkan panduan perintah processing (alias: ?)"),
    ("/back", "Kembali ke prompt project utama"),
]

SUBCOMMAND_SUGGESTIONS: dict[str, list[tuple[str, str]]] = {
    "/vault": [
        ("/vault list", "List all documents and cached papers in vault"),
        ("/vault add", "Add local paper or document to vault (/vault add <file>)"),
        ("/vault view", "Read paper summary or text (/vault view <#>)"),
        ("/vault archive", "Move paper to archive (/vault archive <#>)"),
    ],
    "/models": [
        ("/models list", "List configured models, API keys, and providers"),
        ("/models use", "Switch active LLM model (/models use <model>)"),
        ("/models 9router", "Configure 9Router local gateway proxy"),
        ("/models tokens", "Inspect global and project token consumption"),
        ("/models set-key", "Store API key per-provider (/models set-key <provider> <key>)"),
    ],
    "/chat": [
        ("/chat", "Buka menu hub sesi chat riset (skripsi/chat>)"),
        ("/chat list", "Lihat seluruh sesi percakapan yang tersimpan"),
        ("/chat new", "Buat sesi percakapan baru"),
        ("/chat resume", "Buka/lanjutkan sesi percakapan yang sudah ada"),
        ("/chat delete", "Hapus satu atau banyak sesi percakapan (contoh: 1, 2 atau all)"),
    ],
    "/settings": [
        ("/settings show", "Show current workspace and citation settings"),
        ("/settings set-citation", "Change citation style preference (apa | ieee)"),
    ],
    "/skill": [
        ("/skill scrpcy_manual", "Scrpcy AI Workbench Manual & autonomous research tool usage"),
    ],
}


def interactive_help_view(
    console: Console | None = None,
    current_model: str = "gemini/gemini-2.5-pro",
    current_effort: str = "high",
    is_project_mode: bool = True,
    is_chat_mode: bool = False,
    mode: str | None = None,
    context_label: str = "",
) -> None:
    """Antigravity-style simple available commands view."""
    c = console or Console(force_terminal=True)
    width = c.width or 80

    if mode == "vault":
        commands_list = SLASH_COMMANDS_VAULT
        title = f"Available Commands [{context_label or 'Vault'}]"
    elif mode == "processing":
        commands_list = SLASH_COMMANDS_PROCESSING
        title = f"Available Commands [{context_label or 'Processing'}]"
    elif mode == "chat_hub":
        commands_list = SLASH_COMMANDS_CHAT_HUB
        title = f"Available Commands [{context_label or 'Chat'}]"
    elif mode == "chat" or (mode is None and is_chat_mode):
        commands_list = SLASH_COMMANDS_CHAT
        title = f"Available Commands [{context_label or 'Chat Session'}]"
    elif mode == "dashboard":
        commands_list = SLASH_COMMANDS_DASHBOARD
        title = f"Available Commands [{context_label or 'Dashboard'}]"
    elif mode == "project" or (mode is None and is_project_mode):
        commands_list = SLASH_COMMANDS_PROJECT
        title = f"Available Commands [{context_label or 'Project'}]"
    else:
        commands_list = SLASH_COMMANDS_DASHBOARD
        title = "Available Commands [Dashboard]"

    if not sys.stdin.isatty():
        c.print(f"\n[bold bright_white]{escape(title)}[/bold bright_white]")
        table = Table(box=None, padding=(0, 2), show_header=False)
        table.add_column("Command", style="bold cyan", width=20)
        table.add_column("Description", style="white")
        for cmd, desc in commands_list:
            table.add_row(cmd, desc)
        c.print(table)
        c.print()
        return

    cursor_idx = 0
    c.show_cursor(False)
    prev_lines_drawn = 0
    try:
        while True:
            cursor_idx = max(0, min(cursor_idx, len(commands_list) - 1))
            lines: list[Text] = []

            lines.append(Text(title, style="bold bright_white"))
            lines.append(Text(""))

            for idx, (k, desc) in enumerate(commands_list):
                is_cursor = idx == cursor_idx
                row = Text()
                if is_cursor:
                    row.append("> ", style="bold cyan")
                    row.append(f"{k:<22}", style="bold bright_white")
                    row.append(desc, style="bold white")
                else:
                    row.append("  ", style="dim")
                    row.append(f"{k:<22}", style="dim")
                    row.append(desc, style="dim")
                lines.append(row)

            lines.append(Text(f"[{cursor_idx + 1}-{len(commands_list)} of {len(commands_list)} items]", style="dim"))
            lines.append(Text(""))

            # Footer
            footer = Text("↑/↓ Navigate · esc Close", style="dim")
            m_label = _format_model_display_name(current_model)
            right_status = f"{m_label} · {current_effort}"
            padding_len = max(2, width - len(footer.plain) - len(right_status) - 2)
            footer.append(" " * padding_len)
            footer.append(right_status, style="dim")
            lines.append(footer)

            # Redraw in place without scrolling
            if prev_lines_drawn > 0:
                _clear_lines(prev_lines_drawn, c)

            for line in lines:
                c.print(line)
            sys.stdout.flush()
            prev_lines_drawn = len(lines)

            key = _read_key()
            if key in ("ESC", "q", "Q", "ENTER"):
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)
                return
            elif key == "UP":
                cursor_idx = max(0, cursor_idx - 1)
            elif key == "DOWN":
                cursor_idx = min(len(commands_list) - 1, cursor_idx + 1)
    finally:
        c.show_cursor(True)


def interactive_input_prompt(
    console: Console | None = None,
    prompt_label: str = "> ",
    is_project_mode: bool = True,
    is_chat_mode: bool = False,
    mode: str | None = None,
    context_label: str = "",
    current_model: str = "gemini/gemini-2.5-pro",
    current_effort: str = "high",
    current_skill: str = "scrpcy_manual",
) -> str:
    """Prompt user with live slash command autocomplete and interactive help."""
    c = console or Console(force_terminal=True)
    width = c.width or 80

    # Non-interactive fallback (tests, pipes, CI)
    if not sys.stdin.isatty():
        res = c.input(f"[bold cyan]{prompt_label}[/bold cyan]").strip()
        return res

    if mode == "vault":
        commands_list = SLASH_COMMANDS_VAULT
    elif mode == "processing":
        commands_list = SLASH_COMMANDS_PROCESSING
    elif mode == "chat_hub":
        commands_list = SLASH_COMMANDS_CHAT_HUB
    elif mode == "chat" or (mode is None and is_chat_mode):
        commands_list = SLASH_COMMANDS_CHAT
    elif mode == "dashboard":
        commands_list = SLASH_COMMANDS_DASHBOARD
    elif mode == "project" or (mode is None and is_project_mode):
        commands_list = SLASH_COMMANDS_PROJECT
    else:
        commands_list = SLASH_COMMANDS_DASHBOARD

    buffer = ""
    suggestion_cursor = 0
    prev_lines_drawn = 0

    def get_suggestions(buf: str) -> list[tuple[str, str]]:
        clean = buf.strip()
        if not clean.startswith("/"):
            return []

        # Check if user typed a command with trailing space for subcommands
        for cmd_prefix, sub_list in SUBCOMMAND_SUGGESTIONS.items():
            if clean == cmd_prefix or clean.startswith(cmd_prefix + " "):
                sub_query = clean[len(cmd_prefix):].strip().lower()
                if not sub_query:
                    return sub_list
                return [item for item in sub_list if sub_query in item[0].lower() or sub_query in item[1].lower()]

        # Filter root slash commands
        q = clean.lower()
        return [item for item in commands_list if item[0].lower().startswith(q) or q in item[0].lower()]

    c.show_cursor(False)
    prev_lines_drawn = 0
    try:
        while True:
            # Check if user entered help shortcut "?"
            if buffer.strip() == "?":
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)
                    prev_lines_drawn = 0
                interactive_help_view(
                    console=c,
                    current_model=current_model,
                    current_effort=current_effort,
                    is_project_mode=is_project_mode,
                    is_chat_mode=is_chat_mode,
                    mode=mode,
                    context_label=context_label,
                )
                buffer = ""
                suggestion_cursor = 0

            suggestions = get_suggestions(buffer)
            if not suggestions:
                suggestion_cursor = 0
            else:
                suggestion_cursor = max(0, min(suggestion_cursor, len(suggestions) - 1))

            # Build lines to render with outlier framing & persistent footer
            lines_to_render: list[Text] = []
            divider_line = Text("─" * max(10, width - 1), style="grey23")

            # 1. Top Outlier Line
            lines_to_render.append(divider_line)

            # 2. Input line
            input_row = Text()
            input_row.append(prompt_label, style="bold cyan")
            input_row.append(buffer, style="bold bright_white")
            input_row.append("█", style="blink bold white")
            lines_to_render.append(input_row)

            # 3. Suggestions dropdown if typing a slash command
            if suggestions:
                max_display = 6
                display_slice = suggestions[:max_display]
                for idx, (cmd_name, cmd_desc) in enumerate(display_slice):
                    is_sel = idx == suggestion_cursor
                    s_row = Text()
                    if is_sel:
                        s_row.append("> ", style="bold cyan")
                        s_row.append(f"{cmd_name:<22}", style="bold bright_white")
                        s_row.append(cmd_desc, style="bold white")
                    else:
                        s_row.append("  ", style="dim")
                        s_row.append(f"{cmd_name:<22}", style="dim")
                        s_row.append(cmd_desc, style="dim")
                    lines_to_render.append(s_row)

                if len(suggestions) > max_display:
                    rem = len(suggestions) - max_display
                    lines_to_render.append(Text(f"  ↓ {rem} more", style="dim"))

            # 4. Bottom Outlier Line
            lines_to_render.append(divider_line)

            # 5. Persistent Footer: left '? for shortcuts' and right status
            if suggestions:
                s_footer = Text("↑/↓ Navigate · enter Select · tab Complete · esc Cancel · ? for shortcuts", style="dim")
            elif mode == "chat" or (mode is None and is_chat_mode):
                s_footer = Text("? for shortcuts · /detach to background · /close to terminate", style="dim")
            else:
                s_footer = Text("? for shortcuts", style="dim")

            m_label = _format_model_display_name(current_model)
            if context_label and (mode == "chat" or is_chat_mode):
                right_st = f"{context_label} • {m_label}"
            elif current_skill:
                right_st = f"{current_skill} • {m_label} • {current_effort}"
            else:
                right_st = f"{m_label} • {current_effort}"

            pad_w = max(2, width - len(s_footer.plain) - len(right_st) - 2)
            s_footer.append(" " * pad_w)
            s_footer.append(right_st, style="dim")
            lines_to_render.append(s_footer)

            # Redraw in place
            if prev_lines_drawn > 0:
                _clear_lines(prev_lines_drawn, c)

            for line in lines_to_render:
                c.print(line)
            sys.stdout.flush()
            prev_lines_drawn = len(lines_to_render)

            # Read keystroke
            key = _read_key()

            if key == "ENTER":
                if suggestions:
                    chosen = suggestions[suggestion_cursor][0]
                    # If chosen has subcommands and buffer was just prefix, complete with space
                    if chosen in SUBCOMMAND_SUGGESTIONS and buffer.strip() == chosen:
                        buffer = chosen + " "
                        suggestion_cursor = 0
                        continue
                    elif buffer.strip() != chosen and not buffer.strip().endswith(" "):
                        buffer = chosen
                        if chosen in SUBCOMMAND_SUGGESTIONS:
                            buffer = chosen + " "
                            suggestion_cursor = 0
                            continue

                # Final redraw without dropdown
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)

                final_row = Text()
                final_row.append(prompt_label, style="bold cyan")
                final_row.append(buffer, style="bold bright_white")
                c.print(final_row)
                return buffer.strip()

            elif key == "TAB":
                if suggestions:
                    chosen = suggestions[suggestion_cursor][0]
                    buffer = chosen + " "
                    suggestion_cursor = 0
                continue

            elif key == "UP":
                if suggestions:
                    suggestion_cursor = max(0, suggestion_cursor - 1)
                continue

            elif key == "DOWN":
                if suggestions:
                    suggestion_cursor = min(len(suggestions) - 1, suggestion_cursor + 1)
                continue

            elif key == "ESC":
                if suggestions:
                    suggestion_cursor = 0
                    buffer = ""
                continue

            elif key == "BACKSPACE":
                if buffer:
                    buffer = buffer[:-1]
                    suggestion_cursor = 0
                continue

            elif len(key) == 1 and key.isprintable():
                buffer += key
                suggestion_cursor = 0
    finally:
        c.show_cursor(True)


def interactive_carousel_table(
    title: str,
    columns: list[dict[str, Any]],
    rows: list[list[str]],
    console: Console | None = None,
    page_size: int | None = None,
    prompt_label: str = "download > ",
    help_text: str = "Ketik nomor paper yang ingin diunduh ke vault project (contoh: '1, 2' atau 'all', 'q' untuk batal):",
    item_label: str = "papers",
) -> str:
    """Interactive paginated table carousel with horizontal slide navigation (←/→) and action prompt."""
    import math

    c = console or Console(force_terminal=True)
    total_items = len(rows)
    if total_items == 0:
        return ""

    term_height = c.height or 24
    if page_size is not None and page_size > 0:
        effective_page_size = page_size
    else:
        # Dynamic calculation fitting terminal height:
        # Reserve ~9 lines for header, table headers/padding, navigation bar, help text, and prompt line
        # Assume ~2 terminal lines per wrapped row
        available_lines = max(6, term_height - 9)
        effective_page_size = max(3, min(8, available_lines // 2))

    total_slides = max(1, math.ceil(total_items / effective_page_size))
    current_slide = 0

    # Non-interactive fallback (tests, CI, piped stdin)
    if not sys.stdin.isatty():
        start_idx = 0
        end_idx = min(total_items, effective_page_size)
        c.print(f"\n[bold bright_white]{title} ({total_items} {item_label})[/bold bright_white]  [dim]•[/dim]  [bold cyan][ Slide 1/{total_slides} ][/bold cyan] [dim](Showing #{start_idx + 1}–{end_idx})[/dim]")
        table = Table(box=None, padding=(0, 2), show_header=True, header_style="bold cyan")
        for col in columns:
            table.add_column(
                col.get("name", ""),
                justify=col.get("justify", "left"),
                style=col.get("style"),
                width=col.get("width"),
                overflow=col.get("overflow", "ellipsis"),
            )
        for r in rows[start_idx:end_idx]:
            table.add_row(*r)
        c.print(table)
        if help_text:
            c.print(f"[dim]{help_text}[/dim]")
        try:
            return c.input(f"[bold cyan]{prompt_label}[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            return ""

    input_buffer = ""
    prev_lines_drawn = 0
    c.show_cursor(False)
    try:
        while True:
            start_idx = current_slide * effective_page_size
            end_idx = min(total_items, start_idx + effective_page_size)
            slice_rows = rows[start_idx:end_idx]

            # 1. Header
            header_text = Text()
            header_text.append(title, style="bold bright_white")
            header_text.append(f" ({total_items} {item_label})", style="bold bright_white")
            header_text.append("   •   ", style="dim")
            header_text.append(f"[ Slide {current_slide + 1}/{total_slides} ]", style="bold cyan")
            header_text.append(f" (Showing #{start_idx + 1}–{end_idx})", style="dim")

            # 2. Table
            table = Table(box=None, padding=(0, 2), show_header=True, header_style="bold cyan")
            for col in columns:
                table.add_column(
                    col.get("name", ""),
                    justify=col.get("justify", "left"),
                    style=col.get("style"),
                    width=col.get("width"),
                    overflow=col.get("overflow", "ellipsis"),
                )
            for r in slice_rows:
                table.add_row(*r)

            # 3. Navigation Controls Bar
            nav_bar = Text()
            if total_slides > 1:
                nav_bar.append("←", style="bold cyan")
                nav_bar.append(" Prev Slide   ", style="dim")
                nav_bar.append("→", style="bold cyan")
                nav_bar.append(" Next Slide   ", style="dim")
            nav_bar.append("all", style="bold cyan")
            nav_bar.append(" Ingest All   ", style="dim")
            nav_bar.append("esc / q", style="bold yellow")
            nav_bar.append(" Cancel", style="dim")

            # 4. Help text
            help_t = Text(help_text, style="dim") if help_text else None

            # 5. Input prompt
            prompt_t = Text()
            prompt_t.append(prompt_label, style="bold cyan")
            prompt_t.append(input_buffer, style="bold bright_white")
            prompt_t.append("█", style="blink bold white")

            # Render to buffer and count exact lines
            with c.capture() as capture:
                c.print()
                c.print(header_text)
                c.print(table)
                c.print(nav_bar)
                if help_t:
                    c.print(help_t)
                c.print(prompt_t)

            rendered = capture.get()
            lines_count = len(rendered.splitlines())

            # Redraw in place without scrolling
            if prev_lines_drawn > 0:
                _clear_lines(prev_lines_drawn, c)

            sys.stdout.write(rendered)
            sys.stdout.flush()
            prev_lines_drawn = lines_count

            # Wait for key
            key = _read_key()

            if key == "LEFT":
                current_slide = max(0, current_slide - 1)
            elif key == "RIGHT":
                current_slide = min(total_slides - 1, current_slide + 1)
            elif key == "ENTER":
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)
                # Print static final state cleanly
                c.print()
                c.print(header_text)
                c.print(table)
                final_row = Text()
                final_row.append(prompt_label, style="bold cyan")
                final_row.append(input_buffer, style="bold bright_white")
                c.print(final_row)
                return input_buffer.strip()
            elif key in ("ESC", "\x03"):
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)
                return ""
            elif key == "BACKSPACE":
                if input_buffer:
                    input_buffer = input_buffer[:-1]
            elif len(key) == 1 and key.isprintable():
                if not input_buffer and key in ("q", "Q"):
                    if prev_lines_drawn > 0:
                        _clear_lines(prev_lines_drawn, c)
                    return ""
                input_buffer += key
    finally:
        c.show_cursor(True)



def interactive_menu(
    options: list[tuple[str, str]],
    header: Any | None = None,
    cursor_idx: int = 0,
    console: Console | None = None,
    footer_hint: str = "↑/↓ Navigate · enter Select · 1-N Quick jump · esc Back",
    allow_back: bool = True,
) -> str | None:
    """Antigravity-style in-place interactive menu selector without terminal stacking."""
    if not options:
        return None

    c = console or Console(force_terminal=True)
    if not sys.stdin.isatty():
        if header:
            c.print(header)
        return options[0][0]

    cursor = max(0, min(cursor_idx, len(options) - 1))
    prev_lines_drawn = 0
    c.show_cursor(False)
    try:
        while True:
            lines: list[Any] = []
            if header is not None:
                lines.append(header)
                lines.append(Text(""))

            for idx, (k, label) in enumerate(options, 1):
                is_sel = (idx - 1) == cursor
                opt_row = Text()
                if is_sel:
                    opt_row.append("> ", style="bold cyan")
                    opt_row.append(f"[{idx}] ", style="bold cyan")
                    opt_row.append(label, style="bold bright_white")
                else:
                    opt_row.append("  ", style="dim")
                    opt_row.append(f"[{idx}] ", style="dim")
                    opt_row.append(label, style="white")
                lines.append(opt_row)

            lines.append(Text(""))
            # STRICTLY NO HORIZONTAL LINE PER DESIGN.MD AND USER SPECIFICATION
            lines.append(Text(footer_hint, style="dim"))

            # Exact terminal line count via capture
            with c.capture() as capture:
                for item in lines:
                    c.print(item)
            rendered_lines = len(capture.get().splitlines())

            if prev_lines_drawn > 0:
                _clear_lines(prev_lines_drawn, c)

            for item in lines:
                c.print(item)
            sys.stdout.flush()
            prev_lines_drawn = rendered_lines

            key = _read_key()
            if key == "UP":
                cursor = (cursor - 1) % len(options)
            elif key == "DOWN":
                cursor = (cursor + 1) % len(options)
            elif key == "ENTER":
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)
                return options[cursor][0]
            elif key in ("ESC", "\x03", "q", "Q"):
                if allow_back:
                    if prev_lines_drawn > 0:
                        _clear_lines(prev_lines_drawn, c)
                    return None
            elif len(key) == 1 and key.isdigit():
                d = int(key) - 1
                if 0 <= d < len(options):
                    if prev_lines_drawn > 0:
                        _clear_lines(prev_lines_drawn, c)
                    return options[d][0]
    finally:
        c.show_cursor(True)


def interactive_item_browser(
    title: Any | None,
    columns: list[dict[str, Any]],
    items: list[Any],
    row_builder: Any,
    actions: list[tuple[str, str, str]],
    page_size: int = 7,
    console: Console | None = None,
    empty_message: str = "Tidak ada item.",
) -> tuple[str, int, Any | None]:
    """Interactive slide/carousel item browser with actions per DESIGN.md."""
    import math

    c = console or Console(force_terminal=True)
    total_items = len(items)

    if not sys.stdin.isatty():
        if total_items == 0:
            return ("back", -1, None)
        return (actions[0][0] if actions else "select", 0, items[0])

    if total_items == 0:
        c.print(f"\n[dim]{empty_message}[/dim]\n")
        return ("back", -1, None)

    total_slides = max(1, math.ceil(total_items / page_size))
    current_slide = 0
    cursor_idx = 0
    prev_lines_drawn = 0

    c.show_cursor(False)
    try:
        while True:
            start_idx = current_slide * page_size
            end_idx = min(total_items, start_idx + page_size)
            slice_items = items[start_idx:end_idx]

            cursor_idx = max(0, min(cursor_idx, len(slice_items) - 1))
            absolute_cursor = start_idx + cursor_idx

            cap_console = Console(record=True, width=c.width or 80)
            if title:
                if isinstance(title, str):
                    cap_console.print(title)
                else:
                    cap_console.print(title)
                cap_console.print()

            table = Table(box=None, padding=(0, 2), show_header=True, header_style="bold cyan")
            for col in columns:
                table.add_column(
                    col.get("name", ""),
                    justify=col.get("justify", "left"),
                    style=col.get("style"),
                    width=col.get("width"),
                    overflow=col.get("overflow", "ellipsis"),
                )

            for i, it in enumerate(slice_items):
                is_selected = i == cursor_idx
                row_vals = row_builder(start_idx + i, it, is_selected)
                table.add_row(*row_vals)

            cap_console.print(table)
            cap_console.print()

            # Actions / Navigation bar
            action_bar = Text()
            if total_slides > 1:
                action_bar.append(f"[Slide {current_slide + 1}/{total_slides}] ", style="bold cyan")
                action_bar.append("←/→ Page · ", style="dim")
            action_bar.append("↑/↓ Navigate · ", style="dim")

            for a_key, a_label, a_key_hint in actions:
                action_bar.append(f"[{a_key_hint}] ", style="bold cyan")
                action_bar.append(f"{a_label}   ", style="dim")

            
            cap_console.print(action_bar)

            rendered = cap_console.export_text()
            lines_count = rendered.count("\n")

            if prev_lines_drawn > 0:
                _clear_lines(prev_lines_drawn, c)

            sys.stdout.write(rendered)
            sys.stdout.flush()
            prev_lines_drawn = lines_count

            key = _read_key()
            if key == "UP":
                cursor_idx = (cursor_idx - 1) % len(slice_items)
            elif key == "DOWN":
                cursor_idx = (cursor_idx + 1) % len(slice_items)
            elif key == "LEFT" and total_slides > 1:
                current_slide = (current_slide - 1) % total_slides
                cursor_idx = 0
            elif key == "RIGHT" and total_slides > 1:
                current_slide = (current_slide + 1) % total_slides
                cursor_idx = 0
            elif key in ("ESC", "\x03"):
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)
                return ("back", absolute_cursor, items[absolute_cursor] if items else None)
            elif key == "ENTER":
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)
                # Default enter action is first action
                first_act = actions[0][0] if actions else "select"
                return (first_act, absolute_cursor, items[absolute_cursor])
            else:
                # Check action shortcut match
                k_clean = key.lower()
                matched_action = None
                for a_key, a_label, a_key_hint in actions:
                    if k_clean == a_key_hint.lower():
                        matched_action = a_key
                        break
                if matched_action:
                    if prev_lines_drawn > 0:
                        _clear_lines(prev_lines_drawn, c)
                    return (matched_action, absolute_cursor, items[absolute_cursor])
    finally:
        c.show_cursor(True)


def interactive_detail_viewer(
    title: str,
    content_lines: list[str],
    actions: list[tuple[str, str, str]] | None = None,
    console: Console | None = None,
) -> str:
    """Interactive log / document viewer with scroll support per DESIGN.md."""
    c = console or Console(force_terminal=True)
    if not sys.stdin.isatty():
        c.print(f"\n[bold cyan]=== {title} ===[/bold cyan]")
        for l in content_lines[:20]:
            c.print(l)
        return "back"

    acts = actions or [("back", "Back", "esc")]
    view_height = max(5, (c.height or 24) - 8)
    scroll_top = 0
    total_lines = len(content_lines)
    prev_lines_drawn = 0

    c.show_cursor(False)
    try:
        while True:
            cap_console = Console(record=True, width=c.width or 80)
            cap_console.print(f"[bold bright_white]{title}[/bold bright_white] [dim]({total_lines} lines)[/dim]")
            cap_console.print()

            visible = content_lines[scroll_top : scroll_top + view_height]
            if not visible:
                cap_console.print("[dim](No log output)[/dim]")
            else:
                for line in visible:
                    cap_console.print(Text(line[:(c.width or 80) - 2], style="white"))

            # Fill remaining space
            for _ in range(view_height - len(visible)):
                cap_console.print()

            

            action_bar = Text()
            if total_lines > view_height:
                action_bar.append(f"[Line {scroll_top + 1}-{min(total_lines, scroll_top + view_height)}/{total_lines}] ", style="bold cyan")
                action_bar.append("↑/↓ Scroll · ", style="dim")
            for a_key, a_label, a_key_hint in acts:
                action_bar.append(f"[{a_key_hint}] ", style="bold cyan")
                action_bar.append(f"{a_label}   ", style="dim")

            cap_console.print(action_bar)
            rendered = cap_console.export_text()
            lines_count = rendered.count("\n")

            if prev_lines_drawn > 0:
                _clear_lines(prev_lines_drawn, c)

            sys.stdout.write(rendered)
            sys.stdout.flush()
            prev_lines_drawn = lines_count

            key = _read_key()
            if key == "UP":
                scroll_top = max(0, scroll_top - 1)
            elif key == "DOWN":
                scroll_top = max(0, min(total_lines - view_height, scroll_top + 1))
            elif key in ("ESC", "\x03", "q", "Q"):
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)
                return "back"
            else:
                k_clean = key.lower()
                for a_key, a_label, a_key_hint in acts:
                    if k_clean == a_key_hint.lower():
                        if prev_lines_drawn > 0:
                            _clear_lines(prev_lines_drawn, c)
                        return a_key
    finally:
        c.show_cursor(True)
