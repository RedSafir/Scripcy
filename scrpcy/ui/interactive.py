"""Interactive Antigravity-style TUI selectors for models and skills."""

from __future__ import annotations

import collections
import sys
import time
from typing import Any, Callable

from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

import math
import os
import re
import shutil

from rich.cells import cell_len

_KEY_BUFFER: collections.deque[str] = collections.deque()
_LAST_OBSERVED_TERM_SIZE: tuple[int, int] | None = None
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


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

        # Enable VT processing (0x0001 | 0x0002 | 0x0004) and explicitly CLEAR
        # DISABLE_NEWLINE_AUTO_RETURN (0x0008) so '\n' always returns cursor to column 0.
        target_mode = 0x0001 | 0x0002 | 0x0004
        for handle_id in (-11, -12):
            h = kernel32.GetStdHandle(handle_id)
            if h and h != -1:
                mode = ctypes.c_ulong()
                if kernel32.GetConsoleMode(h, ctypes.byref(mode)):
                    kernel32.SetConsoleMode(h, (mode.value | target_mode) & ~0x0008)

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
                kernel32.SetConsoleMode(h_con, (mode.value | target_mode) & ~0x0008)
            kernel32.CloseHandle(h_con)
    except Exception:
        pass


init_windows_terminal()


def get_live_terminal_size(console: Console | None = None) -> tuple[int, int]:
    """Query live terminal (columns, rows) directly from OS console without stale caching."""
    if console is not None:
        explicit_w = getattr(console, "_width", None)
        explicit_h = getattr(console, "_height", None)
        if explicit_w is not None:
            return (max(20, int(explicit_w)), max(8, int(explicit_h or 24)))

    if sys.platform == "win32":
        try:
            import ctypes

            class _COORD(ctypes.Structure):
                _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]

            class _SMALL_RECT(ctypes.Structure):
                _fields_ = [
                    ("Left", ctypes.c_short),
                    ("Top", ctypes.c_short),
                    ("Right", ctypes.c_short),
                    ("Bottom", ctypes.c_short),
                ]

            class _CSBI(ctypes.Structure):
                _fields_ = [
                    ("dwSize", _COORD),
                    ("dwCursorPosition", _COORD),
                    ("wAttributes", ctypes.c_ushort),
                    ("srWindow", _SMALL_RECT),
                    ("dwMaximumWindowSize", _COORD),
                ]

            kernel32 = ctypes.windll.kernel32
            csbi = _CSBI()
            for handle_id in (-11, -12):
                h = kernel32.GetStdHandle(handle_id)
                if h and h != -1 and kernel32.GetConsoleScreenBufferInfo(h, ctypes.byref(csbi)):
                    win_cols = csbi.srWindow.Right - csbi.srWindow.Left + 1
                    win_rows = csbi.srWindow.Bottom - csbi.srWindow.Top + 1
                    buf_cols = csbi.dwSize.X
                    cols = min(win_cols, buf_cols) if (win_cols > 0 and buf_cols > 0) else (win_cols or buf_cols)
                    if cols > 0 and win_rows > 0:
                        return (max(20, int(cols)), max(8, int(win_rows)))
        except Exception:
            pass

    for fd in (1, 2, 0):
        try:
            sz = os.get_terminal_size(fd)
            if sz.columns > 0 and sz.lines > 0:
                return (max(20, int(sz.columns)), max(8, int(sz.lines)))
        except OSError:
            continue

    fallback_sz = shutil.get_terminal_size((80, 24))
    return (max(20, int(fallback_sz.columns)), max(8, int(fallback_sz.lines)))


def _visible_cell_len(line: str) -> int:
    """Calculate visible terminal cell width of an ANSI-rendered line."""
    plain = _ANSI_ESCAPE_RE.sub("", line)
    return cell_len(plain)


def _adapt_table_columns_to_width(
    columns: list[dict[str, Any]],
    term_cols: int,
) -> tuple[list[dict[str, Any]], tuple[int, int]]:
    """Dynamically scale and reflow table columns and cell padding to fit live terminal width."""
    if not columns:
        return [], (0, 1)

    avail = max(24, term_cols - 1)
    pad_x = 1 if avail < 90 else 2
    col_count = len(columns)
    total_pad = col_count * (pad_x * 2)
    content_budget = max(col_count * 3, avail - total_pad)

    base_widths: list[int] = []
    flex_indices: list[int] = []
    for idx, col in enumerate(columns):
        w = col.get("width")
        name_len = max(3, len(str(col.get("name", ""))))
        if w is not None and int(w) > 0:
            bw = int(w)
            base_widths.append(bw)
            if bw >= 18:
                flex_indices.append(idx)
        else:
            base_widths.append(max(16, name_len + 6))
            flex_indices.append(idx)

    if not flex_indices and len(columns) > 1:
        widest_idx = max(range(len(base_widths)), key=lambda i: base_widths[i])
        flex_indices.append(widest_idx)

    total_base = sum(base_widths)
    adapted_widths = list(base_widths)

    if total_base > content_budget:
        # Proportionally shrink columns wider than 4 cells so the table never overflows
        shrinkable_indices = [i for i, w in enumerate(adapted_widths) if w > 4]
        shrinkable_sum = sum(adapted_widths[i] for i in shrinkable_indices)
        fixed_sum = total_base - shrinkable_sum
        target_shrink_sum = max(len(shrinkable_indices) * 4, content_budget - fixed_sum)
        if shrinkable_sum > 0:
            ratio = target_shrink_sum / shrinkable_sum
            for i in shrinkable_indices:
                min_col_w = 3 if i == 0 else 5
                adapted_widths[i] = max(min_col_w, int(adapted_widths[i] * ratio))
    elif total_base < content_budget and flex_indices:
        # Distribute extra width on wide terminals to primary text columns
        extra = min(48, content_budget - total_base)
        per_flex = extra // len(flex_indices)
        for i in flex_indices:
            adapted_widths[i] += per_flex

    adapted_cols: list[dict[str, Any]] = []
    for idx, col in enumerate(columns):
        c_copy = dict(col)
        c_copy["width"] = adapted_widths[idx]
        if c_copy.get("overflow") == "fold" and avail < 95:
            c_copy["overflow"] = "ellipsis"
        adapted_cols.append(c_copy)

    return adapted_cols, (0, pad_x)


def _clear_lines(num_lines: int, console: Console | None = None) -> None:
    """Cross-platform zero-flicker line clear: erases N lines upward without blanking below."""
    if num_lines <= 0:
        return
    out = [f"\x1b[{num_lines}A\r"]
    for _ in range(num_lines):
        out.append("\r\x1b[2K\n")
    out.append(f"\x1b[{num_lines}A\r")
    target = getattr(console, "file", None) or sys.stdout
    target.write("".join(out))
    target.flush()


def _render_diff_inplace(
    console: Console,
    lines: list[Any],
    last_rendered_text: str,
    prev_lines_drawn: int,
) -> tuple[str, int]:
    """Seamless Differential In-Place Terminal Rendering.

    Overwrites the previous terminal output line-by-line without blanking or clearing
    the screen below (no `\\x1b[J`), completely eliminating flicker. Performs dirty-checking:
    if rendered output matches `last_rendered_text`, performs zero I/O writes.
    Clamps rendering width to terminal columns - 1 to prevent auto-margin-wrap row displacement,
    and automatically clears reflowed physical rows when the terminal window is resized.

    Returns:
        tuple[str, int]: (new_rendered_text, new_lines_drawn_count)
    """
    term_cols, _ = get_live_terminal_size(console)
    max_w = max(20, term_cols - 1)

    prev_term_cols = getattr(console, "_scrpcy_last_term_cols", None)
    prev_line_widths: list[int] = getattr(console, "_scrpcy_last_line_widths", [])
    resized = prev_term_cols is not None and prev_term_cols != term_cols

    # Preserve raw _width / _height (None in interactive mode) so console.width is never frozen!
    orig_raw_w = getattr(console, "_width", None)
    orig_raw_h = getattr(console, "_height", None)
    console._width = max_w
    try:
        with console.capture() as capture:
            for item in lines:
                console.print(item, crop=True, no_wrap=True)
        raw_str = capture.get()
    finally:
        console._width = orig_raw_w
        console._height = orig_raw_h

    rendered_lines = raw_str.splitlines()
    cur_text = "\n".join(rendered_lines)

    # 1. State Dirty Checking: if identical and terminal size unchanged, do zero I/O
    if not resized and cur_text == last_rendered_text and prev_lines_drawn > 0:
        return last_rendered_text, prev_lines_drawn

    # 2. If terminal was resized while a frame was on screen, clear reflowed physical rows first
    if resized and prev_lines_drawn > 0:
        if term_cols < (prev_term_cols or term_cols) and prev_line_widths:
            reflowed_rows = sum(
                max(1, math.ceil(w / max(1, term_cols))) for w in prev_line_widths
            )
            rows_to_clear = max(prev_lines_drawn, reflowed_rows)
        else:
            rows_to_clear = prev_lines_drawn
        _clear_lines(rows_to_clear, console)
        prev_lines_drawn = 0

    # 3. In-Place Overwriting:
    out = []
    if prev_lines_drawn > 0:
        out.append(f"\x1b[{prev_lines_drawn}A\r")

    for line in rendered_lines:
        out.append(f"\r\x1b[2K{line}\n")

    if len(rendered_lines) < prev_lines_drawn:
        excess = prev_lines_drawn - len(rendered_lines)
        for _ in range(excess):
            out.append("\r\x1b[2K\n")
        out.append(f"\x1b[{excess}A\r")

    target = getattr(console, "file", None) or sys.stdout
    target.write("".join(out))
    target.flush()

    try:
        setattr(console, "_scrpcy_last_term_cols", term_cols)
        setattr(
            console,
            "_scrpcy_last_line_widths",
            [_visible_cell_len(line) for line in rendered_lines],
        )
    except Exception:
        pass

    return cur_text, len(rendered_lines)


def _read_key_immediate() -> str:
    """Blocking read of a single ready keypress or escape sequence cross-platform."""
    if sys.platform == "win32":
        import msvcrt

        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
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
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
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


def _poll_key() -> str | None:
    """Non-blocking read of a single keypress, returns None if no key pressed."""
    if _KEY_BUFFER:
        return _KEY_BUFFER.popleft()

    if sys.platform == "win32":
        import msvcrt

        if not msvcrt.kbhit():
            return None
        return _read_key_immediate()
    else:
        import select

        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            return _read_key_immediate()
        return None


def _read_key(timeout: float | None = None) -> str | None:
    """Read a single keypress or escape sequence, waking up on terminal resize."""
    global _LAST_OBSERVED_TERM_SIZE
    if _KEY_BUFFER:
        return _KEY_BUFFER.popleft()

    start = time.monotonic()
    while True:
        if _KEY_BUFFER:
            return _KEY_BUFFER.popleft()
        k = _poll_key()
        if k is not None:
            return k
        cur_size = get_live_terminal_size()
        if _LAST_OBSERVED_TERM_SIZE is None:
            _LAST_OBSERVED_TERM_SIZE = cur_size
        elif cur_size != _LAST_OBSERVED_TERM_SIZE:
            _LAST_OBSERVED_TERM_SIZE = cur_size
            return "__RESIZE__"
        if timeout is not None and (time.monotonic() - start >= timeout):
            return None
        time.sleep(0.015)



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
    last_rendered_text = ""
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

            # In-place differential redraw
            last_rendered_text, prev_lines_drawn = _render_diff_inplace(
                c, lines, last_rendered_text, prev_lines_drawn
            )

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
    last_rendered_text = ""
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

            # In-place differential redraw
            last_rendered_text, prev_lines_drawn = _render_diff_inplace(
                c, lines, last_rendered_text, prev_lines_drawn
            )

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



def interactive_confirm(
    prompt_text: str,
    default: bool = True,
    yes_label: str = "Ya, lanjutkan",
    no_label: str = "Batal",
    console: Console | None = None,
) -> bool:
    """Prompt user with a clean, focused 2-option confirmation dialog (Yes / No).

    Used for standard direct user actions (e.g. deleting a project, archiving a file,
    deleting a chat session) without unnecessary AI framing or write-in prompts.
    """
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
        c.print(f"[bold cyan]✦ Konfirmasi:[/bold cyan] [bold bright_white]{prompt_text}[/bold bright_white]")
        try:
            suffix = "(Y/n)" if default else "(y/N)"
            line = c.input(f"  {yes_label} / {no_label} {suffix}: ").strip().lower()
            if not line:
                return default
            if line in ("y", "yes", "ya", "ok", "1"):
                return True
            return False
        except Exception:
            return default

    options = [
        (yes_label, True),
        (no_label, False),
    ]
    cursor_idx = 0 if default else 1
    prev_lines_drawn = 0
    last_rendered_text = ""

    c.show_cursor(False)
    try:
        while True:
            lines_out: list[Text] = []
            header = Text("✦ Konfirmasi: ", style="bold cyan")
            header.append(prompt_text, style="bold bright_white")
            lines_out.append(header)

            for idx, (label, val) in enumerate(options):
                is_cursor = idx == cursor_idx
                item_text = Text()
                if is_cursor:
                    item_text.append("  > ", style="bold cyan")
                    marker_style = "bold green" if val else "bold yellow"
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

            last_rendered_text, prev_lines_drawn = _render_diff_inplace(
                c, lines_out, last_rendered_text, prev_lines_drawn
            )

            key = _read_key()
            if key in ("ESC", "n", "N", "q", "Q"):
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)
                return False
            elif key in ("y", "Y"):
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)
                return True
            elif key == "ENTER":
                choice = options[cursor_idx][1]
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)
                return choice
            elif key in ("UP", "LEFT", "DOWN", "RIGHT", "TAB"):
                cursor_idx = (cursor_idx + 1) % len(options)
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
    last_rendered_text = ""

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

            last_rendered_text, prev_lines_drawn = _render_diff_inplace(
                c, lines_out, last_rendered_text, prev_lines_drawn
            )

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


def interactive_scout_confirm(
    suggested_topic: str,
    quota: int = 7,
    console: Console | None = None,
) -> tuple[str, str | None]:
    """Interactive 3-option confirmation for literature scout conforming to DESIGN.md.

    Options:
        [1] Ya: Run autonomous literature search and ingest
        [2] Tolak: Decline and continue synthesis with general knowledge
        [3] Input User: Enter custom keywords for scouting

    Returns:
        tuple[str, str | None]: (action, custom_keyword)
            where action in ('yes', 'no', 'custom')
    """
    c = console or Console()

    # Enable ANSI Virtual Terminal processing on Windows
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass

    # Non-interactive / mock fallback for tests and automated pipelines
    from unittest.mock import Mock
    is_mocked = isinstance(getattr(c, "input", None), Mock)
    if is_mocked or not sys.stdin.isatty():
        try:
            line = c.input("[bold cyan][?] Jalankan pencarian & unduh otomatis? [Y/n/ketik kata kunci baru]: [/bold cyan]").strip()
            if line.lower() in ("", "y", "yes", "ya", "ok", "1"):
                return ("yes", suggested_topic)
            elif line.lower() in ("n", "no", "tidak", "skip", "batal", "cancel", "0", "2"):
                return ("no", None)
            elif line == "3":
                c.print("[bold cyan]● Masukkan kata kunci pencarian kustom:[/bold cyan]")
                custom = c.input("  > ").strip()
                return ("custom", custom if custom else suggested_topic)
            else:
                return ("custom", line)
        except Exception:
            return ("no", None)

    options = [
        (f"Ya, jalankan pencarian & unduh otomatis ({quota} paper: '{suggested_topic}')", "yes"),
        ("Tolak, lanjutkan sintesis dengan pengetahuan umum", "no"),
        ("Input kata kunci baru (pencarian kustom)", "custom"),
    ]
    cursor_idx = 0
    prev_lines_drawn = 0
    last_rendered_text = ""

    c.show_cursor(False)
    try:
        while True:
            lines_out: list[Text] = []
            header = Text("● Rencana: Mencari & mengunduh literatur ", style="bold cyan")
            header.append(f"({quota} paper terkait '{suggested_topic}')", style="bold bright_white")
            lines_out.append(header)
            lines_out.append(Text("  Pilih tindakan untuk melanjutkan:", style="dim"))

            for idx, (label, val) in enumerate(options):
                is_cursor = idx == cursor_idx
                item_text = Text()
                num_tag = f"[{idx + 1}] "
                if is_cursor:
                    item_text.append("  > ", style="bold cyan")
                    marker_style = "bold green" if val == "yes" else ("bold red" if val == "no" else "bold yellow")
                    item_text.append(num_tag, style=marker_style)
                    item_text.append(label, style="bold bright_white")
                else:
                    item_text.append("    ", style="dim")
                    item_text.append(num_tag, style="dim")
                    item_text.append(label, style="dim white")
                lines_out.append(item_text)

            footer = Text("  Navigasi: ", style="dim")
            footer.append("↑/↓ / 1-3", style="bold cyan")
            footer.append(" Pilih  ", style="dim")
            footer.append("Enter", style="bold green")
            footer.append(" Konfirmasi  ", style="dim")
            footer.append("Esc/n", style="bold yellow")
            footer.append(" Tolak", style="dim")
            lines_out.append(footer)

            last_rendered_text, prev_lines_drawn = _render_diff_inplace(
                c, lines_out, last_rendered_text, prev_lines_drawn
            )

            key = _read_key()
            if key in ("ESC", "n", "N", "q", "Q"):
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)
                return ("no", None)
            elif key in ("y", "Y"):
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)
                return ("yes", suggested_topic)
            elif key in ("1", "2", "3"):
                choice_idx = int(key) - 1
                choice = options[choice_idx][1]
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)
                if choice == "yes":
                    return ("yes", suggested_topic)
                elif choice == "no":
                    return ("no", None)
                elif choice == "custom":
                    c.show_cursor(True)
                    c.print("[bold cyan]● Masukkan kata kunci pencarian kustom:[/bold cyan]")
                    try:
                        user_input = c.input("  > ").strip()
                    except Exception:
                        user_input = ""
                    if user_input:
                        return ("custom", user_input)
                    return ("no", None)
            elif key == "ENTER":
                choice = options[cursor_idx][1]
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)
                if choice == "yes":
                    return ("yes", suggested_topic)
                elif choice == "no":
                    return ("no", None)
                elif choice == "custom":
                    c.show_cursor(True)
                    c.print("[bold cyan]● Masukkan kata kunci pencarian kustom:[/bold cyan]")
                    try:
                        user_input = c.input("  > ").strip()
                    except Exception:
                        user_input = ""
                    if user_input:
                        return ("custom", user_input)
                    return ("no", None)
            elif key in ("UP", "LEFT"):
                cursor_idx = (cursor_idx - 1) % len(options)
            elif key in ("DOWN", "RIGHT", "TAB"):
                cursor_idx = (cursor_idx + 1) % len(options)
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
    last_rendered_text = ""

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

            last_rendered_text, prev_lines_drawn = _render_diff_inplace(
                c, lines, last_rendered_text, prev_lines_drawn
            )

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
    ("/chats", "Daftar dan switcher sesi percakapan interaktif (alias: /sessions)"),
    ("/sessions", "Manajemen sesi percakapan interaktif (TMUX-style)"),
    ("/skill", "Switch active research persona (/skill <name|#>)"),
    ("/models", "Manage LLM models, API keys, 9Router, and tokens"),
    ("/processing", "View and track background tasks and downloads (alias: /process)"),
    ("/process", "View and track background tasks and downloads (alias: /tasks)"),
    ("/tasks", "View and track background tasks and downloads"),
    ("/gdrive", "Ingest research paper directly from Google Drive"),
    ("/active", "Toggle active context papers for AI queries"),
    ("/rename", "Rename the current project (/rename <new_name>)"),
    ("/tokens", "Inspect project token usage statistics"),
    ("/status", "Lihat model aktif dan konsumsi token proyek saat ini"),
    ("/settings", "Manage workspace and citation settings"),
    ("/turns", "Atur batas iterasi loop AI agent (1-50 turn)"),
    ("/web", "Launch 3-panel Web Studio in browser"),
    ("/dashboard", "Return to Dashboard mode (alias: /close, /back)"),
    ("/clear", "Clear terminal screen"),
    ("/help", "Open interactive help and keyboard shortcuts"),
    ("/exit", "Exit Scrpcy research studio"),
]

SLASH_COMMANDS_CHAT: list[tuple[str, str]] = [
    ("/back", "Kembali ke menu daftar chat (skripsi/chat>)"),
    ("/detach", "Lepas sesi ke background (tmux detach, tugas tetap berjalan)"),
    ("/switch", "Pindah sesi chat via dialog interaktif (alias: /chats, /sessions)"),
    ("/chats", "Daftar dan switcher sesi percakapan interaktif"),
    ("/sessions", "Manajemen sesi percakapan interaktif (TMUX-style)"),
    ("/process", "Pantau dan kelola proses latar belakang (alias: /tasks)"),
    ("/tasks", "Pantau dan kelola proses latar belakang (alias: /process)"),
    ("/cancel", "Batalkan proses AI atau task yang sedang berjalan di sesi ini"),
    ("/skill", "Ganti persona riset aktif (/skill <nama|#>)"),
    ("/status", "Lihat model aktif dan konsumsi token proyek saat ini"),
    ("/new", "Buat sesi percakapan baru (/new [nama])"),
    ("/resume", "Pindah ke sesi percakapan lain (/resume <#|nama>)"),
    ("/history", "Lihat riwayat percakapan sesi ini"),
    ("/clear", "Bersihkan pesan dalam sesi percakapan ini"),
    ("/export", "Ekspor percakapan sesi ini ke markdown"),
    ("/turns", "Atur batas iterasi loop AI agent (1-50 turn)"),
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
    last_rendered_text = ""
    try:
        while True:
            width, _ = get_live_terminal_size(c)
            cursor_idx = max(0, min(cursor_idx, len(commands_list) - 1))
            lines: list[Text] = []

            lines.append(Text(title, style="bold bright_white"))
            lines.append(Text(""))

            cmd_col_w = 16 if width < 72 else 22
            for idx, (k, desc) in enumerate(commands_list):
                is_cursor = idx == cursor_idx
                row = Text()
                if is_cursor:
                    row.append("> ", style="bold cyan")
                    row.append(f"{k:<{cmd_col_w}}", style="bold bright_white")
                    row.append(desc, style="bold white")
                else:
                    row.append("  ", style="dim")
                    row.append(f"{k:<{cmd_col_w}}", style="dim")
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

            # In-place differential redraw
            last_rendered_text, prev_lines_drawn = _render_diff_inplace(
                c, lines, last_rendered_text, prev_lines_drawn
            )

            key = _read_key()
            if key == "__RESIZE__":
                continue
            elif key in ("ESC", "q", "Q", "ENTER"):
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
    on_idle: Callable[[Callable[[], None]], bool] | None = None,
    status_line_callback: Callable[[], str | None] | None = None,
    initial_buffer: str = "",
) -> str:
    """Prompt user with live slash command autocomplete and interactive help."""
    c = console or Console(force_terminal=True)

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

    buffer = initial_buffer or ""
    suggestion_cursor = 0
    prev_lines_drawn = 0
    last_rendered_text = ""

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
    try:
        while True:
            width, _ = get_live_terminal_size(c)
            max_line_w = max(20, width - 1)

            # Check if user entered help shortcut "?"
            if buffer.strip() == "?":
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)
                    prev_lines_drawn = 0
                    last_rendered_text = ""
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
                width, _ = get_live_terminal_size(c)
                max_line_w = max(20, width - 1)

            suggestions = get_suggestions(buffer)
            if not suggestions:
                suggestion_cursor = 0
            else:
                suggestion_cursor = max(0, min(suggestion_cursor, len(suggestions) - 1))

            # Build lines to render with outlier framing & persistent footer
            lines_to_render: list[Text] = []

            # 0. Live Status Line (e.g. background task progress / spinner)
            status_text = status_line_callback() if status_line_callback is not None else None
            if status_text:
                for st_line in status_text.splitlines():
                    lines_to_render.append(Text.from_markup(st_line))
                lines_to_render.append(Text(""))

            divider_line = Text("─" * max(10, max_line_w), style="grey23")

            # 1. Top Outlier Line
            lines_to_render.append(divider_line)

            # 2. Input line (wraps cleanly across multiple lines if user types past terminal width)
            full_input_len = cell_len(prompt_label) + cell_len(buffer) + 1
            if full_input_len <= max_line_w:
                input_row = Text()
                input_row.append(prompt_label, style="bold cyan")
                input_row.append(buffer, style="bold bright_white")
                input_row.append("█", style="blink bold white")
                lines_to_render.append(input_row)
            else:
                first_cap = max(1, max_line_w - cell_len(prompt_label))
                first_chunk = buffer[:first_cap]
                rem_buf = buffer[first_cap:]
                first_row = Text()
                first_row.append(prompt_label, style="bold cyan")
                first_row.append(first_chunk, style="bold bright_white")
                if not rem_buf and cell_len(first_row.plain) < max_line_w:
                    first_row.append("█", style="blink bold white")
                    lines_to_render.append(first_row)
                else:
                    lines_to_render.append(first_row)
                    chunk_w = max(10, max_line_w)
                    chunks = [rem_buf[i : i + chunk_w] for i in range(0, len(rem_buf), chunk_w)] or [""]
                    for c_idx, ch_str in enumerate(chunks):
                        cont_row = Text(ch_str, style="bold bright_white")
                        if c_idx == len(chunks) - 1:
                            if cell_len(cont_row.plain) < max_line_w:
                                cont_row.append("█", style="blink bold white")
                                lines_to_render.append(cont_row)
                            else:
                                lines_to_render.append(cont_row)
                                lines_to_render.append(Text("█", style="blink bold white"))
                        else:
                            lines_to_render.append(cont_row)

            # 3. Suggestions dropdown if typing a slash command
            if suggestions:
                max_display = 6
                display_slice = suggestions[:max_display]
                cmd_col_w = 16 if width < 72 else 22
                for idx, (cmd_name, cmd_desc) in enumerate(display_slice):
                    is_sel = idx == suggestion_cursor
                    s_row = Text()
                    if is_sel:
                        s_row.append("> ", style="bold cyan")
                        s_row.append(f"{cmd_name:<{cmd_col_w}}", style="bold bright_white")
                        s_row.append(cmd_desc, style="bold white")
                    else:
                        s_row.append("  ", style="dim")
                        s_row.append(f"{cmd_name:<{cmd_col_w}}", style="dim")
                        s_row.append(cmd_desc, style="dim")
                    lines_to_render.append(s_row)

                if len(suggestions) > max_display:
                    rem = len(suggestions) - max_display
                    lines_to_render.append(Text(f"  ↓ {rem} more", style="dim"))

            # 4. Bottom Outlier Line
            lines_to_render.append(divider_line)

            # 5. Responsive Persistent Footer: adapts left hints & right status across terminal widths
            m_label = _format_model_display_name(current_model)
            if context_label and (mode == "chat" or is_chat_mode):
                right_st = f"{context_label} • {m_label}"
            elif current_skill and current_skill != "scrpcy_manual":
                right_st = f"{current_skill} • {m_label} • {current_effort}"
            else:
                right_st = f"{m_label} • {current_effort}"

            if suggestions:
                left_full = "↑/↓ Navigate · enter Select · tab Complete · esc Cancel · ? for shortcuts"
                left_compact = "↑/↓ · enter · tab · esc"
            elif mode == "chat" or (mode is None and is_chat_mode):
                left_full = "? for shortcuts · /detach to background · /close to terminate"
                left_compact = "? shortcuts · /detach · /close"
            else:
                left_full = "? for shortcuts"
                left_compact = "? shortcuts"

            if len(left_full) + len(right_st) + 3 <= max_line_w:
                left_chosen = left_full
                right_chosen = right_st
            elif len(left_compact) + len(right_st) + 3 <= max_line_w:
                left_chosen = left_compact
                right_chosen = right_st
            elif len(left_compact) + len(m_label) + 3 <= max_line_w:
                left_chosen = left_compact
                right_chosen = m_label
            else:
                left_chosen = "? shortcuts"
                avail_r = max(0, max_line_w - len(left_chosen) - 3)
                right_chosen = m_label[:avail_r] if avail_r > 4 else ""

            s_footer = Text(left_chosen, style="dim")
            if right_chosen:
                pad_w = max(2, max_line_w - len(left_chosen) - len(right_chosen) - 1)
                s_footer.append(" " * pad_w)
                s_footer.append(right_chosen, style="dim")
            lines_to_render.append(s_footer)

            # In-place differential redraw
            last_rendered_text, prev_lines_drawn = _render_diff_inplace(
                c, lines_to_render, last_rendered_text, prev_lines_drawn
            )

            # Read keystroke (with idle callback if provided)
            if on_idle is not None:
                key = None
                last_spin_time = time.monotonic()
                while key is None:
                    key = _read_key(timeout=0.08)
                    if key is not None:
                        break
                    def _clear_prompt_fn() -> None:
                        nonlocal prev_lines_drawn, last_rendered_text
                        if prev_lines_drawn > 0:
                            _clear_lines(prev_lines_drawn, c)
                            prev_lines_drawn = 0
                            last_rendered_text = ""

                    if on_idle(_clear_prompt_fn):
                        # Something was printed by idle handler; break out to redraw prompt
                        break

                    now = time.monotonic()
                    if status_line_callback is not None and (now - last_spin_time >= 0.12):
                        last_spin_time = now
                        new_status = status_line_callback()
                        if new_status != status_text:
                            # Status or spinner frame changed; break out to redraw prompt in place
                            break

                if key is None or key == "__RESIZE__":
                    # Idle handler, status update, or terminal resize requested prompt redraw
                    continue
            else:
                key = _read_key()
                if key == "__RESIZE__":
                    continue

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
                elif not buffer:
                    if prev_lines_drawn > 0:
                        _clear_lines(prev_lines_drawn, c)
                    return "/detach" if (mode == "chat" or is_chat_mode) else ""
                else:
                    buffer = ""
                continue

            elif key == "":
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)
                raise KeyboardInterrupt()

            elif key == "BACKSPACE":
                if buffer:
                    buffer = buffer[:-1]
                    suggestion_cursor = 0
                continue

            elif len(key) == 1 and key.isprintable():
                buffer += key
                if sys.platform == "win32":
                    import msvcrt
                    while msvcrt.kbhit():
                        ch = msvcrt.getwch()
                        if ch in ("\x00", "\xe0"):
                            code = msvcrt.getwch()
                            if code == "H":
                                _KEY_BUFFER.append("UP")
                            elif code == "P":
                                _KEY_BUFFER.append("DOWN")
                            elif code == "K":
                                _KEY_BUFFER.append("LEFT")
                            elif code == "M":
                                _KEY_BUFFER.append("RIGHT")
                            else:
                                _KEY_BUFFER.append("SPECIAL")
                            break
                        elif ch in ("\r", "\n"):
                            _KEY_BUFFER.append("ENTER")
                            break
                        elif ch == "\t":
                            _KEY_BUFFER.append("TAB")
                            break
                        elif ch == "\x1b":
                            _KEY_BUFFER.append("ESC")
                            break
                        elif ch in ("\x08", "\x7f"):
                            _KEY_BUFFER.append("BACKSPACE")
                            break
                        elif ch == "\x03":
                            _KEY_BUFFER.append("\x03")
                            break
                        elif ch.isprintable():
                            buffer += ch
                        else:
                            break
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

    term_cols, term_height = get_live_terminal_size(c)
    if page_size is not None and page_size > 0:
        effective_page_size = page_size
    else:
        available_lines = max(6, term_height - 9)
        effective_page_size = max(3, min(8, available_lines // 2))

    total_slides = max(1, math.ceil(total_items / effective_page_size))
    current_slide = 0

    # Non-interactive fallback (tests, CI, piped stdin)
    if not sys.stdin.isatty():
        start_idx = 0
        end_idx = min(total_items, effective_page_size)
        c.print(f"\n[bold bright_white]{title} ({total_items} {item_label})[/bold bright_white]  [dim]•[/dim]  [bold cyan][ Slide 1/{total_slides} ][/bold cyan] [dim](Showing #{start_idx + 1}–{end_idx})[/dim]")
        adapted_cols, pad_tuple = _adapt_table_columns_to_width(columns, term_cols)
        table = Table(box=None, padding=pad_tuple, show_header=True, header_style="bold cyan")
        for col in adapted_cols:
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
    last_rendered_text = ""
    c.show_cursor(False)
    try:
        while True:
            term_cols, term_height = get_live_terminal_size(c)
            if page_size is not None and page_size > 0:
                effective_page_size = page_size
            else:
                available_lines = max(6, term_height - 9)
                effective_page_size = max(3, min(8, available_lines // 2))
            total_slides = max(1, math.ceil(total_items / effective_page_size))
            current_slide = max(0, min(current_slide, total_slides - 1))

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
            adapted_cols, pad_tuple = _adapt_table_columns_to_width(columns, term_cols)
            table = Table(box=None, padding=pad_tuple, show_header=True, header_style="bold cyan")
            for col in adapted_cols:
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

            lines_to_render = [header_text, table, nav_bar]
            if help_t:
                lines_to_render.append(help_t)
            lines_to_render.append(prompt_t)

            # In-place differential redraw
            last_rendered_text, prev_lines_drawn = _render_diff_inplace(
                c, lines_to_render, last_rendered_text, prev_lines_drawn
            )

            # Wait for key
            key = _read_key()
            if key == "__RESIZE__":
                continue

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
                if sys.platform == "win32":
                    import msvcrt
                    while msvcrt.kbhit():
                        ch = msvcrt.getwch()
                        if ch in ("\x00", "\xe0"):
                            code = msvcrt.getwch()
                            if code == "K":
                                _KEY_BUFFER.append("LEFT")
                            elif code == "M":
                                _KEY_BUFFER.append("RIGHT")
                            elif code == "H":
                                _KEY_BUFFER.append("UP")
                            elif code == "P":
                                _KEY_BUFFER.append("DOWN")
                            break
                        elif ch in ("\r", "\n"):
                            _KEY_BUFFER.append("ENTER")
                            break
                        elif ch == "\x1b":
                            _KEY_BUFFER.append("ESC")
                            break
                        elif ch in ("\x08", "\x7f"):
                            _KEY_BUFFER.append("BACKSPACE")
                            break
                        elif ch == "\x03":
                            _KEY_BUFFER.append("\x03")
                            break
                        elif ch.isprintable():
                            input_buffer += ch
                        else:
                            break
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
    last_rendered_text = ""
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
                else:
                    opt_row.append("  ", style="dim")
                    opt_row.append(f"[{idx}] ", style="dim")

                base_style = "bold bright_white" if is_sel else "white"
                if isinstance(label, Text):
                    lbl_text = label.copy()
                elif isinstance(label, str) and "[" in label and "]" in label:
                    try:
                        lbl_text = Text.from_markup(label, style=base_style)
                    except Exception:
                        lbl_text = Text(label, style=base_style)
                else:
                    lbl_text = Text(str(label), style=base_style)

                opt_row.append_text(lbl_text)
                lines.append(opt_row)

            lines.append(Text(""))
            # STRICTLY NO HORIZONTAL LINE PER DESIGN.MD AND USER SPECIFICATION
            lines.append(Text(footer_hint, style="dim"))

            # Differential in-place redraw
            last_rendered_text, prev_lines_drawn = _render_diff_inplace(
                c, lines, last_rendered_text, prev_lines_drawn
            )

            key = _read_key()
            if key == "__RESIZE__":
                continue
            elif key == "UP":
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
    items_provider: Any | None = None,
    poll_interval: float | None = 0.4,
) -> tuple[str, int, Any | None]:
    """Interactive slide/carousel item browser with actions per DESIGN.md."""
    import math

    c = console or Console(force_terminal=True)
    if items_provider is not None:
        try:
            items = items_provider() or []
        except Exception:
            pass
    total_items = len(items)

    if not sys.stdin.isatty():
        if total_items == 0:
            return ("back", -1, None)
        return (actions[0][0] if actions else "select", 0, items[0])

    total_slides = max(1, math.ceil(total_items / page_size)) if total_items > 0 else 1
    current_slide = 0
    cursor_idx = 0
    prev_lines_drawn = 0
    last_rendered_text = ""

    c.show_cursor(False)
    try:
        while True:
            term_cols, term_rows = get_live_terminal_size(c)
            effective_page_size = max(2, min(page_size, term_rows - 8))

            if items_provider is not None:
                try:
                    items = items_provider() or []
                    total_items = len(items)
                except Exception:
                    pass

            total_slides = max(1, math.ceil(total_items / effective_page_size)) if total_items > 0 else 1
            current_slide = max(0, min(current_slide, total_slides - 1))

            start_idx = current_slide * effective_page_size
            end_idx = min(total_items, start_idx + effective_page_size)
            slice_items = items[start_idx:end_idx]

            adapted_cols, pad_tuple = _adapt_table_columns_to_width(columns, term_cols)

            if total_items == 0:
                lines: list[Any] = []
                if title:
                    lines.append(title if not isinstance(title, str) else Text(title))
                    lines.append(Text(""))

                table = Table(box=None, padding=pad_tuple, show_header=True, header_style="bold cyan")
                for col in adapted_cols:
                    table.add_column(
                        col.get("name", ""),
                        justify=col.get("justify", "left"),
                        style=col.get("style"),
                        width=col.get("width"),
                        overflow=col.get("overflow", "ellipsis"),
                    )
                empty_row = ["-"] * len(adapted_cols)
                target_col = 3 if len(empty_row) > 3 else (1 if len(empty_row) > 1 else 0)
                empty_row[target_col] = f"[dim]{empty_message}[/dim]"
                table.add_row(*empty_row)
                lines.append(table)
                lines.append(Text(""))

                action_bar = Text()
                action_bar.append("[esc/q] ", style="bold cyan")
                action_bar.append("Back", style="dim")
                lines.append(action_bar)

                last_rendered_text, prev_lines_drawn = _render_diff_inplace(
                    c, lines, last_rendered_text, prev_lines_drawn
                )

                if items_provider is not None:
                    poll_dur = poll_interval if poll_interval is not None else 0.4
                    key = _read_key(timeout=poll_dur)
                else:
                    key = _read_key()

                if key == "__RESIZE__":
                    continue

                if key in ("ESC", "\x03", "q", "Q", "ENTER", "back", "Back"):
                    if prev_lines_drawn > 0:
                        _clear_lines(prev_lines_drawn, c)
                    return ("back", -1, None)
                continue

            cursor_idx = max(0, min(cursor_idx, len(slice_items) - 1))
            absolute_cursor = start_idx + cursor_idx

            lines = []
            if title:
                lines.append(title if not isinstance(title, str) else Text(title))
                lines.append(Text(""))

            table = Table(box=None, padding=pad_tuple, show_header=True, header_style="bold cyan")
            for col in adapted_cols:
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

            lines.append(table)
            lines.append(Text(""))

            # Actions / Navigation bar
            action_bar = Text()
            if total_slides > 1:
                action_bar.append(f"[Slide {current_slide + 1}/{total_slides}] ", style="bold cyan")
                action_bar.append("←/→ Page · ", style="dim")
            action_bar.append("↑/↓ Navigate · ", style="dim")

            for a_key, a_label, a_key_hint in actions:
                action_bar.append(f"[{a_key_hint}] ", style="bold cyan")
                action_bar.append(f"{a_label}   ", style="dim")

            lines.append(action_bar)

            last_rendered_text, prev_lines_drawn = _render_diff_inplace(
                c, lines, last_rendered_text, prev_lines_drawn
            )

            if items_provider is not None:
                poll_dur = poll_interval if poll_interval is not None else 0.4
                key = _read_key(timeout=poll_dur)
            else:
                key = _read_key()

            if key is None or key == "__RESIZE__":
                continue

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
            elif key in ("ESC", "\x03", "q", "Q"):
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
                # Check action shortcut match (support multi-key hints like "c/x")
                k_clean = key.lower()
                matched_action = None
                for a_key, a_label, a_key_hint in actions:
                    hints = [h.strip().lower() for h in a_key_hint.split("/")]
                    if k_clean in hints:
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
    content_lines: list[str] | None = None,
    actions: list[tuple[str, str, str]] | None = None,
    console: Console | None = None,
    lines_provider: Any | None = None,
    title_provider: Any | None = None,
    is_live: Any | None = None,
) -> str:
    """Interactive log / document viewer with scroll and live streaming support per DESIGN.md."""
    import time

    c = console or Console(force_terminal=True)
    initial_lines = lines_provider() if lines_provider else (content_lines or [])
    if not sys.stdin.isatty():
        c.print(f"\n[bold cyan]=== {title} ===[/bold cyan]")
        for l in initial_lines[:20]:
            c.print(l)
        return "back"

    acts = actions or [("back", "Back", "esc")]
    view_height = max(5, (c.height or 24) - 8)
    auto_scroll = True
    scroll_top = 0
    prev_lines_drawn = 0
    last_rendered_text = ""

    c.show_cursor(False)
    try:
        while True:
            term_cols, term_rows = get_live_terminal_size(c)
            view_height = max(5, term_rows - 8)
            term_w = max(20, term_cols - 1)

            cur_lines = lines_provider() if lines_provider else (content_lines or [])
            cur_title = title_provider() if title_provider else title
            total_lines = len(cur_lines)

            if auto_scroll:
                scroll_top = max(0, total_lines - view_height)
            else:
                scroll_top = max(0, min(scroll_top, max(0, total_lines - view_height)))

            lines: list[Any] = []
            header_t = Text(no_wrap=True, overflow="ellipsis")
            header_t.append(cur_title, style="bold bright_white")
            header_t.append(f" ({total_lines} lines)", style="dim")
            if lines_provider:
                currently_running = is_live() if is_live else False
                if currently_running:
                    header_t.append("   •   ", style="dim")
                    header_t.append("[● Live Streaming]", style="bold cyan")
                else:
                    header_t.append("   •   ", style="dim")
                    header_t.append("[✓ Finished]", style="bold green")
            lines.append(header_t)
            lines.append(Text(""))

            visible = cur_lines[scroll_top : scroll_top + view_height]
            if not visible:
                lines.append(Text("(No log output)", style="dim"))
            else:
                for line in visible:
                    lines.append(Text(line[:term_w - 2], style="white", no_wrap=True, overflow="ellipsis"))

            # Fill remaining space
            for _ in range(view_height - len(visible)):
                lines.append(Text(""))

            action_bar = Text(no_wrap=True, overflow="ellipsis")
            if total_lines > view_height:
                action_bar.append(f"[Line {scroll_top + 1}-{min(total_lines, scroll_top + view_height)}/{total_lines}] ", style="bold cyan")
                action_bar.append("↑/↓ Scroll · ", style="dim")
                if not auto_scroll and lines_provider:
                    action_bar.append("G Follow · ", style="dim")
            for a_key, a_label, a_key_hint in acts:
                action_bar.append(f"[{a_key_hint}] ", style="bold cyan")
                action_bar.append(f"{a_label}   ", style="dim")

            lines.append(action_bar)

            last_rendered_text, prev_lines_drawn = _render_diff_inplace(
                c, lines, last_rendered_text, prev_lines_drawn
            )

            if lines_provider:
                key = _read_key(timeout=0.25)
            else:
                key = _read_key()

            if key is None or key == "__RESIZE__":
                continue

            if key == "UP":
                scroll_top = max(0, scroll_top - 1)
                auto_scroll = False
            elif key == "DOWN":
                scroll_top = min(max(0, total_lines - view_height), scroll_top + 1)
                if scroll_top >= max(0, total_lines - view_height):
                    auto_scroll = True
            elif key in ("g", "G"):
                auto_scroll = True
                scroll_top = max(0, total_lines - view_height)
            elif key in ("ESC", "\x03", "q", "Q"):
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)
                return "back"
            elif key == "ENTER":
                if prev_lines_drawn > 0:
                    _clear_lines(prev_lines_drawn, c)
                return acts[0][0] if acts else "back"
            else:
                k_clean = key.lower()
                for a_key, a_label, a_key_hint in acts:
                    if k_clean == a_key_hint.lower():
                        if prev_lines_drawn > 0:
                            _clear_lines(prev_lines_drawn, c)
                        return a_key
    finally:
        c.show_cursor(True)
