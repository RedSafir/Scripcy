"""Unit tests for interactive_carousel_table in scripcy.ui.interactive."""

from unittest.mock import patch
from rich.console import Console

from scripcy.ui.interactive import interactive_carousel_table


def test_carousel_empty_rows() -> None:
    c = Console(force_terminal=True)
    res = interactive_carousel_table("Test Table", [{"name": "Col"}], [], console=c)
    assert res == ""


def test_carousel_non_interactive_fallback() -> None:
    c = Console(force_terminal=True)
    columns = [{"name": "#"}, {"name": "Title"}]
    rows = [[str(i), f"Paper {i}"] for i in range(1, 15)]

    with patch("sys.stdin.isatty", return_value=False):
        with patch.object(c, "input", return_value="1, 2"):
            res = interactive_carousel_table(
                "Literature Results",
                columns=columns,
                rows=rows,
                console=c,
                page_size=5,
            )
            assert res == "1, 2"


def test_carousel_interactive_navigation_and_input() -> None:
    c = Console(force_terminal=True)
    columns = [{"name": "#"}, {"name": "Title"}]
    rows = [[str(i), f"Paper {i}"] for i in range(1, 15)]

    # Simulate keys: RIGHT (slide 2), '1', '0', ENTER
    simulated_keys = ["RIGHT", "1", "0", "ENTER"]
    with patch("sys.stdin.isatty", return_value=True):
        with patch("scripcy.ui.interactive._read_key", side_effect=simulated_keys):
            res = interactive_carousel_table(
                "Literature Results",
                columns=columns,
                rows=rows,
                console=c,
                page_size=5,
            )
            assert res == "10"


def test_carousel_interactive_cancel_q() -> None:
    c = Console(force_terminal=True)
    columns = [{"name": "#"}, {"name": "Title"}]
    rows = [[str(i), f"Paper {i}"] for i in range(1, 6)]

    # Immediate 'q' on empty buffer cancels
    with patch("sys.stdin.isatty", return_value=True):
        with patch("scripcy.ui.interactive._read_key", return_value="q"):
            res = interactive_carousel_table(
                "Literature Results",
                columns=columns,
                rows=rows,
                console=c,
            )
            assert res == ""


def test_carousel_interactive_cancel_esc() -> None:
    c = Console(force_terminal=True)
    columns = [{"name": "#"}, {"name": "Title"}]
    rows = [[str(i), f"Paper {i}"] for i in range(1, 6)]

    with patch("sys.stdin.isatty", return_value=True):
        with patch("scripcy.ui.interactive._read_key", return_value="ESC"):
            res = interactive_carousel_table(
                "Literature Results",
                columns=columns,
                rows=rows,
                console=c,
            )
            assert res == ""
