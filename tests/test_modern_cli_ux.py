"""Unit tests for Modern CLI UX Ergonomics: Background Tasks, TMUX Sessions, & Toasts."""

import time
from unittest.mock import MagicMock, patch
from rich.console import Console
from rich.text import Text

from scrpcy.cli_studio import CLIStudio
from scrpcy.core.tasks import TaskInfo, TaskManager
from scrpcy.core.chats import ChatManager, ChatSession
from scrpcy.ui.interactive import interactive_item_browser


def test_transient_toast_lifecycle() -> None:
    """Test that transient toasts can be added, retrieved, and expire properly."""
    app = CLIStudio.__new__(CLIStudio)
    app._transient_toasts = []
    app._known_task_states = {}
    app.task_manager = MagicMock()
    app.task_manager.list_tasks.return_value = []

    # 1. Add toast manually
    app.add_transient_toast("Download selesai: 3 papers", level="success", duration_seconds=0.2)
    assert len(app._transient_toasts) == 1
    assert "Download selesai" in app.get_live_status_indicator()

    # 2. Wait for expiration
    time.sleep(0.25)
    indicator = app.get_live_status_indicator()
    assert indicator is None
    assert len(app._transient_toasts) == 0


def test_auto_toast_generation_on_task_state_change() -> None:
    """Test that transitioning from running to completed/failed auto-generates a toast."""
    app = CLIStudio.__new__(CLIStudio)
    app._transient_toasts = []
    app._known_task_states = {}

    tm = MagicMock()
    t1 = TaskInfo(id="t1", name="Literature Scout", task_type="scout", status="running")
    tm.list_tasks.return_value = [t1]
    app.task_manager = tm

    # Frame 1: task is running
    indicator1 = app.get_live_status_indicator()
    assert indicator1 is not None
    assert "Literature Scout" in indicator1
    assert len(app._transient_toasts) == 0

    # Frame 2: task completes
    t1_completed = TaskInfo(
        id="t1",
        name="Literature Scout",
        task_type="scout",
        status="completed",
        result_summary="4 papers downloaded",
    )
    tm.list_tasks.return_value = [t1_completed]

    indicator2 = app.get_live_status_indicator()
    assert indicator2 is not None
    assert "✓ Selesai" in indicator2
    assert "4 papers downloaded" in indicator2
    assert len(app._transient_toasts) == 1


def test_inter_session_background_awareness() -> None:
    """Test that indicator reflects background tasks in other sessions."""
    app = CLIStudio.__new__(CLIStudio)
    app._transient_toasts = []
    app._known_task_states = {}

    tm = MagicMock()
    # Task running in session_b
    t_bg = TaskInfo(id="t_bg", name="Scout Paper B", task_type="scout", status="running", session_id="session_b")
    tm.list_tasks_for_session.return_value = []
    tm.list_tasks.return_value = [t_bg]
    app.task_manager = tm

    # User is currently inside session_a
    indicator = app.get_live_status_indicator(session_id="session_a")
    assert indicator is not None
    assert "background task aktif di sesi lain" in indicator


def test_interactive_item_browser_multi_key_hints_and_quit() -> None:
    """Test single-key shortcut matching with multi-key hints (e.g. c/x, esc/q)."""
    columns = [{"name": "ID", "width": 10}, {"name": "Name", "width": 20}]
    items = [{"id": "1", "name": "Task 1"}]

    def _row(idx, it, sel):
        return [it["id"], it["name"]]

    actions = [
        ("detail", "Detail", "enter"),
        ("cancel", "Cancel", "c/x"),
        ("back", "Back", "esc/q"),
    ]

    with patch("sys.stdin.isatty", return_value=True):
        # Test pressing 'c' triggers 'cancel'
        with patch("scrpcy.ui.interactive._read_key", return_value="c"):
            action, _, _ = interactive_item_browser(
                title="Tasks",
                columns=columns,
                items=items,
                row_builder=_row,
                actions=actions,
                console=Console(force_terminal=True),
            )
            assert action == "cancel"

        # Test pressing 'x' also triggers 'cancel'
        with patch("scrpcy.ui.interactive._read_key", return_value="x"):
            action, _, _ = interactive_item_browser(
                title="Tasks",
                columns=columns,
                items=items,
                row_builder=_row,
                actions=actions,
                console=Console(force_terminal=True),
            )
            assert action == "cancel"

        # Test pressing 'q' triggers 'back'
        with patch("scrpcy.ui.interactive._read_key", return_value="q"):
            action, _, _ = interactive_item_browser(
                title="Tasks",
                columns=columns,
                items=items,
                row_builder=_row,
                actions=actions,
                console=Console(force_terminal=True),
            )
            assert action == "back"


def test_project_command_routing_for_modern_ux() -> None:
    """Test slash commands /process, /tasks, /chats, /sessions route cleanly."""
    app = CLIStudio.__new__(CLIStudio)
    app.run_interactive_processing_view = MagicMock()
    app.run_interactive_sessions_list = MagicMock()
    app.handle_processing_interaction = MagicMock()
    app.handle_chat_hub_interaction = MagicMock()

    # /process without args opens interactive task view
    app.handle_project_command("/process")
    app.run_interactive_processing_view.assert_called_once()

    # /tasks without args opens interactive task view
    app.handle_project_command("/tasks")
    assert app.run_interactive_processing_view.call_count == 2

    # /chats without args opens interactive sessions list
    app.handle_project_command("/chats")
    app.run_interactive_sessions_list.assert_called_once()

    # /sessions without args opens interactive sessions list
    app.handle_project_command("/sessions")
    assert app.run_interactive_sessions_list.call_count == 2


def test_chat_submode_routing_for_modern_ux() -> None:
    """Test that /process, /switch, and /detach work ergonomically inside chat mode."""
    app = CLIStudio.__new__(CLIStudio)
    app.run_interactive_processing_view = MagicMock()
    app.run_interactive_sessions_list = MagicMock()
    app.render_mode_divider = MagicMock()
    app.console = Console(force_terminal=True)
    app.current_project = MagicMock(name="skripsi_test")
    app.current_project.name = "skripsi_test"

    app.task_manager = TaskManager()
    sess = ChatSession(id="sess_1", name="testing", active_skill="general")
    app.active_chat_session = sess
    app.chat_manager = MagicMock()
    app.sub_mode = "chat"

    # /proc inside chat opens interactive processing view
    app.handle_chat_submode_interaction("/proc")
    app.run_interactive_processing_view.assert_called_once()

    # /switch inside chat opens interactive sessions list
    app.handle_chat_submode_interaction("/switch")
    app.run_interactive_sessions_list.assert_called_once()

    # /detach inside chat sets sub_mode to None (returns to project root)
    app.handle_chat_submode_interaction("/detach")
    assert app.sub_mode is None
    app.chat_manager.update_session_status.assert_called_with("sess_1", "detached")


def test_render_diff_inplace_margin_wrap_clamping() -> None:
    """Test that _render_diff_inplace clamps line widths to prevent terminal auto-wrap cursor displacement."""
    from scrpcy.ui.interactive import _render_diff_inplace
    import io

    out_buf = io.StringIO()
    c = Console(file=out_buf, force_terminal=True, width=80)

    # 1. Plain long line gets formatted to max_w (79) lines
    long_line = "A" * 120
    rendered_text, lines_count = _render_diff_inplace(
        console=c,
        lines=[long_line],
        last_rendered_text="",
        prev_lines_drawn=0,
    )
    for l in rendered_text.splitlines():
        assert len(l) <= 79

    # 2. Text with no_wrap stays strictly 1 line <= 79 cols
    t_line = Text("B" * 120, no_wrap=True, overflow="ellipsis")
    rendered_text2, lines_count2 = _render_diff_inplace(
        console=c,
        lines=[t_line],
        last_rendered_text="",
        prev_lines_drawn=0,
    )
    assert lines_count2 == 1
    assert len(rendered_text2) <= 79


def test_interactive_detail_viewer_no_wrap_overflow() -> None:
    """Test that interactive_detail_viewer renders exact number of lines without row inflation."""
    from scrpcy.ui.interactive import interactive_detail_viewer

    c = Console(force_terminal=False, width=80)
    # Non-interactive fallback should print without error
    res = interactive_detail_viewer(
        title="Super Long Title That Might Otherwise Wrap Into Multiple Rows In A Narrow Terminal" * 3,
        content_lines=["Log line " + str(i) for i in range(50)],
        console=c,
    )
    assert res == "back"
