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
