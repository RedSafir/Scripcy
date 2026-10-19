"""Interactive Antigravity-style TUI selectors for models and skills."""

from __future__ import annotations

import sys
from typing import Any

from rich.console import Console
from rich.text import Text


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

            # Clear screen or clear block
            c.clear()
            for line in lines:
                c.print(line)

            # Wait for key
            key = _read_key()

            if key == "ESC":
                c.clear()
                return None
            elif key == "ENTER":
                c.clear()
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

            c.clear()
            for line in lines:
                c.print(line)

            key = _read_key()

            if key == "ESC":
                c.clear()
                return None
            elif key == "ENTER":
                c.clear()
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


def interactive_action_confirm(action_text: str, console: Console | None = None) -> bool:
    """Prompt the user with an Antigravity-style UP/DOWN arrow selector for action execution."""
    c = console or Console()

    # If non-interactive (e.g. redirected stdin, pipes, or unit tests)
    if not sys.stdin.isatty():
        c.print(f"[bold cyan]✦ Aksi Scrpcy Disarankan AI:[/bold cyan] [bold green]{action_text}[/bold green]")
        try:
            line = c.input("  [bold bright_white]Jalankan aksi ini sekarang?[/bold bright_white] ").strip().lower()
            return line in ("", "y", "yes", "ya", "ok", "1")
        except Exception:
            return False

    options = [
        ("Ya, jalankan di background", True),
        ("Tidak, lewati aksi ini", False),
    ]
    cursor_idx = 0

    c.show_cursor(False)
    try:
        while True:
            lines: list[Text] = []
            header = Text("✦ Aksi Scrpcy Disarankan AI: ", style="bold cyan")
            header.append(action_text, style="bold green")
            lines.append(header)
            lines.append(Text(""))

            for idx, (label, val) in enumerate(options):
                is_cursor = idx == cursor_idx
                item_text = Text()
                if is_cursor:
                    item_text.append("  > ", style="bold cyan")
                    item_text.append(f"[{'x' if val else ' '}] ", style="bold green" if val else "bold red")
                    item_text.append(label, style="bold bright_white")
                else:
                    item_text.append("    ", style="dim")
                    item_text.append(f"[{' ' if val else ' '}] ", style="dim")
                    item_text.append(label, style="dim white")
                lines.append(item_text)

            lines.append(Text(""))
            footer = Text("  Navigasi: ", style="dim")
            footer.append("↑/↓", style="bold cyan")
            footer.append(" Pilih  ", style="dim")
            footer.append("Enter", style="bold green")
            footer.append(" Konfirmasi  ", style="dim")
            footer.append("Esc/n", style="bold yellow")
            footer.append(" Batal", style="dim")
            lines.append(footer)

            c.clear()
            for line in lines:
                c.print(line)

            key = _read_key()
            if key in ("ESC", "n", "N", "q", "Q"):
                c.clear()
                return False
            elif key == "ENTER":
                c.clear()
                return options[cursor_idx][1]
            elif key in ("UP", "DOWN", "LEFT", "RIGHT", "TAB"):
                cursor_idx = 1 - cursor_idx
            elif key in ("y", "Y"):
                c.clear()
                return True
    finally:
        c.show_cursor(True)
