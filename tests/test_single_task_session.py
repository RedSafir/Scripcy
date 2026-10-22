"""Unit tests for single-task session guard, non-blocking detach, and AI continuation."""

import threading
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from scrpcy.cli_studio import CLIStudio
from scrpcy.core.chats import ChatSession
from scrpcy.core.tasks import TaskInfo, TaskManager
from scrpcy.core.workspace import WorkspaceManager


@pytest.fixture
def temp_studio(tmp_path: Path) -> CLIStudio:
    ws_dir = tmp_path / "test_ws"
    ws_dir.mkdir(parents=True)
    ws = WorkspaceManager(root_dir=ws_dir)
    ws.ensure_directories()
    studio = CLIStudio(workspace=ws)
    return studio


def test_busy_guard_rejects_new_prompt_when_task_running(temp_studio: CLIStudio, capsys: pytest.CaptureFixture) -> None:
    session = temp_studio.chat_manager.create_session("test_session")
    temp_studio.active_chat_session = session

    hold_evt = threading.Event()
    try:
        task = temp_studio.task_manager.submit(
            name="Deep Scout Simulation",
            task_type="scout",
            target_fn=lambda cancel_event: cancel_event.wait(5.0),
            session_id=session.id,
        )
        t = temp_studio.task_manager.get_task(task.id)
        assert t.status == "running"

        # Attempt to submit a new question
        with patch.object(temp_studio, "handle_chat") as mock_handle_chat:
            temp_studio.handle_chat_submode_interaction("pertanyaan baru tentang gambut")
            # Must NOT call handle_chat because session is busy
            mock_handle_chat.assert_not_called()

        # Verify active task is tracked
        active_tasks = [tk for tk in temp_studio.task_manager.list_tasks_for_session(session.id) if tk.status == "running"]
        assert len(active_tasks) == 1
    finally:
        temp_studio.task_manager.cancel_task(task.id)


def test_busy_guard_allows_cancel_command(temp_studio: CLIStudio) -> None:
    session = temp_studio.chat_manager.create_session("test_session")
    temp_studio.active_chat_session = session

    task = temp_studio.task_manager.submit(
        name="Heavy Job",
        task_type="scout",
        target_fn=lambda cancel_event: cancel_event.wait(5.0),
        session_id=session.id,
    )
    t = temp_studio.task_manager.get_task(task.id)
    assert t.status == "running"

    # User issues /cancel
    temp_studio.handle_chat_submode_interaction("/cancel")

    # Task must now be cancelled
    t_after = temp_studio.task_manager.get_task(task.id)
    assert t_after.status == "cancelled"


def test_busy_guard_rejects_second_bg_task(temp_studio: CLIStudio) -> None:
    session = temp_studio.chat_manager.create_session("test_session")
    temp_studio.active_chat_session = session

    try:
        task = temp_studio.task_manager.submit(
            name="Existing Job",
            task_type="scout",
            target_fn=lambda cancel_event: cancel_event.wait(5.0),
            session_id=session.id,
        )
        t = temp_studio.task_manager.get_task(task.id)
        assert t.status == "running"

        with patch.object(temp_studio, "submit_chat_background") as mock_submit:
            temp_studio.handle_chat_submode_interaction("/bg tugas kedua")
            mock_submit.assert_not_called()
    finally:
        temp_studio.task_manager.cancel_task(task.id)


def test_ai_task_continuation_fallback(temp_studio: CLIStudio) -> None:
    # Heuristic fallback when engine has no real API key
    res_under_60 = temp_studio._ask_ai_task_continuation("Scout Paper", "Downloading...", elapsed=40.0)
    assert res_under_60 == "EXTEND"

    res_over_60 = temp_studio._ask_ai_task_continuation("Scout Paper", "Idle...", elapsed=65.0)
    assert res_over_60 == "ABORT"
