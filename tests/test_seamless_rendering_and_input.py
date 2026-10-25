"""Unit tests for Seamless Differential In-Place Rendering & Zero Keystroke Loss."""

import io
import sys
from unittest.mock import MagicMock, patch
from rich.console import Console
from rich.text import Text

from scrpcy.ui.interactive import (
    _render_diff_inplace,
    interactive_input_prompt,
    _clear_lines,
    _KEY_BUFFER,
)


def test_render_diff_inplace_no_flicker_and_no_erase_below() -> None:
    c = Console(force_terminal=True, width=80)
    lines_frame1 = [Text("Hello World 1"), Text("Line 2")]

    fake_stdout = io.StringIO()
    with patch("sys.stdout", fake_stdout):
        last_text, count = _render_diff_inplace(c, lines_frame1, "", 0)

    raw_output = fake_stdout.getvalue()
    # Must NEVER contain \x1b[J (blanking below screen)
    assert "\x1b[J" not in raw_output
    # Must use \x1b[2K for line-by-line overwrite
    assert "\x1b[2K" in raw_output
    assert count == 2
    assert "Hello World 1" in last_text


def test_render_diff_inplace_dirty_checking_zero_io() -> None:
    c = Console(force_terminal=True, width=80)
    lines = [Text("Static Header"), Text("Static Status")]

    # First render
    fake_stdout = io.StringIO()
    with patch("sys.stdout", fake_stdout):
        last_text, count = _render_diff_inplace(c, lines, "", 0)

    # Second render with identical content
    fake_stdout_second = io.StringIO()
    with patch("sys.stdout", fake_stdout_second):
        new_text, new_count = _render_diff_inplace(c, lines, last_text, count)

    # Zero bytes written to stdout because content is unchanged
    assert fake_stdout_second.getvalue() == ""
    assert new_text == last_text
    assert new_count == count


def test_render_diff_inplace_reclaims_excess_lines() -> None:
    c = Console(force_terminal=True, width=80)
    lines_frame1 = [Text("Line 1"), Text("Line 2"), Text("Line 3"), Text("Line 4")]
    lines_frame2 = [Text("Line 1 updated"), Text("Line 2 updated")]

    fake_stdout1 = io.StringIO()
    with patch("sys.stdout", fake_stdout1):
        last_text, count = _render_diff_inplace(c, lines_frame1, "", 0)

    fake_stdout2 = io.StringIO()
    with patch("sys.stdout", fake_stdout2):
        new_text, new_count = _render_diff_inplace(c, lines_frame2, last_text, count)

    output = fake_stdout2.getvalue()
    # Must move up 4 lines
    assert "\x1b[4A\r" in output
    # Must clear excess lines
    assert new_count == 2


def test_interactive_input_prompt_initial_buffer() -> None:
    c = Console(force_terminal=True, width=80)
    # Simulate user immediately pressing ENTER
    with patch("sys.stdin.isatty", return_value=True):
        with patch("scrpcy.ui.interactive._read_key", return_value="ENTER"):
            result = interactive_input_prompt(
                console=c,
                prompt_label="test> ",
                initial_buffer="lanjutkan bab 2",
            )
            assert result == "lanjutkan bab 2"


def test_key_buffer_queue_preservation() -> None:
    _KEY_BUFFER.clear()
    _KEY_BUFFER.append("a")
    _KEY_BUFFER.append("ENTER")

    from scrpcy.ui.interactive import _read_key

    k1 = _read_key()
    k2 = _read_key()
    assert k1 == "a"
    assert k2 == "ENTER"


def test_render_diff_inplace_does_not_freeze_console_width() -> None:
    """Verify _render_diff_inplace never freezes console._width when initially None."""
    c = Console(force_terminal=True)
    assert c._width is None

    fake_stdout = io.StringIO()
    with patch("sys.stdout", fake_stdout):
        with patch("scrpcy.ui.interactive.get_live_terminal_size", return_value=(120, 30)):
            last_text, count = _render_diff_inplace(c, [Text("Line 1")], "", 0)

    # console._width must remain None so subsequent resizes are never ignored
    assert c._width is None
    assert count == 1


def test_render_diff_inplace_clears_reflowed_rows_on_terminal_shrink() -> None:
    """Verify shrinking terminal width clears reflowed physical rows before redrawing."""
    c = Console(force_terminal=True)
    fake_stdout1 = io.StringIO()
    # Frame 1 at 120 columns: 4 lines where 3 lines are 119 chars wide
    wide_divider = Text("─" * 119)
    prompt_line = Text("karhutla/chat/karhutla> c█")
    footer_line = Text("? for shortcuts" + (" " * 80) + "karhutla • Gemini 3.8")
    frame1 = [wide_divider, prompt_line, wide_divider, footer_line]

    with patch("sys.stdout", fake_stdout1):
        with patch("scrpcy.ui.interactive.get_live_terminal_size", return_value=(120, 30)):
            last_text, count1 = _render_diff_inplace(c, frame1, "", 0)
    assert count1 == 4

    # Frame 2 after user drags CMD window narrower to 80 columns:
    # The three 119-char lines wrapped into 2 rows each (2 + 1 + 2 + 2 = 7 physical rows).
    narrow_divider = Text("─" * 79)
    narrow_footer = Text("? shortcuts · /detach · /close" + (" " * 25) + "karhutla • Gemini")
    frame2 = [narrow_divider, prompt_line, narrow_divider, narrow_footer]

    fake_stdout2 = io.StringIO()
    with patch("sys.stdout", fake_stdout2):
        with patch("scrpcy.ui.interactive.get_live_terminal_size", return_value=(80, 30)):
            last_text2, count2 = _render_diff_inplace(c, frame2, last_text, count1)

    out2 = fake_stdout2.getvalue()
    # Must clear all 7 reflowed physical rows upward (\x1b[7A\r)
    assert "\x1b[7A\r" in out2
    assert count2 == 4


def test_interactive_input_prompt_keystrokes_and_live_resize() -> None:
    """Verify typing multiple characters and resizing CMD updates in-place without stacking."""
    c = Console(force_terminal=True)
    fake_stdout = io.StringIO()
    sizes = [
        (120, 30),  # initial render
        (120, 30),  # 'c'
        (120, 30),  # 'a'
        (80, 24),   # '__RESIZE__' to 80 cols
        (80, 24),   # 'r'
        (80, 24),   # 'BACKSPACE'
        (80, 24),   # 'ENTER'
    ]
    size_iter = iter(sizes)

    def fake_live_size(_con=None):
        return next(size_iter, (80, 24))

    keys = ["c", "a", "__RESIZE__", "r", "BACKSPACE", "ENTER"]
    with patch("sys.stdin.isatty", return_value=True):
        with patch("sys.stdout", fake_stdout):
            with patch("scrpcy.ui.interactive.get_live_terminal_size", side_effect=fake_live_size):
                with patch("scrpcy.ui.interactive._read_key", side_effect=keys):
                    res = interactive_input_prompt(
                        console=c,
                        prompt_label="karhutla/chat/karhutla> ",
                        is_chat_mode=True,
                        mode="chat",
                        context_label="karhutla • karhutla",
                        on_idle=lambda _clear: False,
                    )

    assert res == "ca"
    raw = fake_stdout.getvalue()
    # Every line overwrite must begin with carriage return + line erase (\r\x1b[2K)
    assert "\r\x1b[2K" in raw
    # When resized from 120 to 80, the 7 reflowed rows must be cleared
    assert "\x1b[7A\r" in raw
