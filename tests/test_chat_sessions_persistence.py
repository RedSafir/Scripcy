"""Unit tests for persistent chat sessions, background task binding, and session resumption."""

from __future__ import annotations

import io
from pathlib import Path
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from scripcy.cli_studio import CLIStudio
from scripcy.core.chats import ChatManager
from scripcy.core.config import ScripcyConfig
from scripcy.core.engine import LLMEngine
from scripcy.core.project import ProjectManager
from scripcy.core.tasks import TaskInfo, TaskManager
from scripcy.core.workspace import WorkspaceManager


class MockStreamChunk:
    def __init__(self, content: str | None) -> None:
        self.choices = [MagicMock(delta=MagicMock(content=content))]


def test_task_manager_session_id_and_callback() -> None:
    """Verify TaskManager tracks session_id and invokes completion callbacks."""
    tm = TaskManager(max_workers=2)
    completed_tasks: list[TaskInfo] = []

    tm.add_completion_callback(lambda t: completed_tasks.append(t))

    def sample_worker() -> str:
        return "Work done"

    task = tm.submit(
        name="Session Bound Task",
        task_type="convert",
        target_fn=sample_worker,
        session_id="sess-123",
    )

    assert task.session_id == "sess-123"

    # Wait for completion
    time.sleep(0.15)
    tasks_for_sess = tm.list_tasks_for_session("sess-123")
    assert len(tasks_for_sess) == 1
    assert tasks_for_sess[0].status == "completed"
    assert len(completed_tasks) == 1
    assert completed_tasks[0].id == task.id

    tm.shutdown(wait=False)


def test_background_task_completion_injects_system_note(
    temp_workspace: WorkspaceManager,
    tmp_path: Path,
) -> None:
    """When a background task bound to a session completes, Scripcy logs a system note into SQLite chats.db."""
    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "reg.json")
    proj = pm.create_project("thesis", temp_workspace.root_dir)
    cfg = ScripcyConfig(workspace_path=str(temp_workspace.root_dir))
    engine = LLMEngine(config=cfg)

    studio = CLIStudio(
        workspace=temp_workspace,
        config=cfg,
        console=console,
        project_manager=pm,
        engine=engine,
    )
    studio.open_project(proj)

    session = studio.chat_manager.create_session("analysis_session")
    studio.active_chat_session = session
    studio.sub_mode = "chat"

    def background_job() -> str:
        time.sleep(0.05)
        return "3 papers ingested successfully"

    task = studio.task_manager.submit(
        name="Literature Discovery",
        task_type="scout",
        target_fn=background_job,
        session_id=session.id,
    )

    # Wait for worker completion and callback execution
    messages = []
    for _ in range(30):
        messages = studio.chat_manager.get_messages(session.id)
        if len(messages) >= 1:
            break
        time.sleep(0.05)
    assert len(messages) >= 1
    sys_msg = messages[-1]
    assert sys_msg.role == "system"
    assert "PROSES LATAR BELAKANG SELESAI" in sys_msg.content
    assert "Literature Discovery" in sys_msg.content
    assert "3 papers ingested successfully" in sys_msg.content


def test_exit_chat_while_task_running(
    temp_workspace: WorkspaceManager,
    tmp_path: Path,
) -> None:
    """When user exits chat mode (/back), running tasks continue in the background and a reminder is shown."""
    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "reg.json")
    proj = pm.create_project("thesis", temp_workspace.root_dir)
    cfg = ScripcyConfig(workspace_path=str(temp_workspace.root_dir))
    engine = LLMEngine(config=cfg)

    studio = CLIStudio(
        workspace=temp_workspace,
        config=cfg,
        console=console,
        project_manager=pm,
        engine=engine,
    )
    studio.open_project(proj)

    session = studio.chat_manager.create_session("deep_dive")
    studio.active_chat_session = session
    studio.sub_mode = "chat"

    def long_task(cancel_event=None) -> str:
        time.sleep(0.3)
        return "Long task finished"

    task = studio.task_manager.submit(
        name="Heavy Downloader",
        task_type="download",
        target_fn=long_task,
        session_id=session.id,
    )

    # User types /back to exit chat
    studio.handle_chat_submode_interaction("/back")

    output = studio.console.file.getvalue()
    assert studio.sub_mode == "chat_hub"
    assert "proses latar belakang tetap berjalan di background" in output
    assert "deep_dive" in output

    # Task is still running or completing in background
    assert task.status in ("running", "completed")

    # Wait for it to finish and verify it logs into the session
    messages = []
    for _ in range(30):
        time.sleep(0.05)
        messages = studio.chat_manager.get_messages(session.id)
        if any("Heavy Downloader" in m.content for m in messages):
            break
    assert any("Heavy Downloader" in m.content for m in messages)


def test_resume_chat_shows_bound_tasks_and_passes_history(
    temp_workspace: WorkspaceManager,
    tmp_path: Path,
) -> None:
    """When resuming a chat session, completed/running tasks are shown, and history is passed to LLM."""
    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "reg.json")
    proj = pm.create_project("thesis", temp_workspace.root_dir)
    cfg = ScripcyConfig(workspace_path=str(temp_workspace.root_dir))

    passed_history = []

    async def mock_acompletion(**kwargs: Any):
        messages = kwargs.get("messages", [])
        for m in messages:
            passed_history.append((m.get("role"), m.get("content")))
        yield MockStreamChunk("Tentu, mari kita bahas temuan dari paper tersebut.")

    engine = LLMEngine(config=cfg, acompletion_fn=mock_acompletion)

    studio = CLIStudio(
        workspace=temp_workspace,
        config=cfg,
        console=console,
        project_manager=pm,
        engine=engine,
    )
    studio.open_project(proj)
    studio.config.api_keys["gemini"] = "mock_key"

    session = studio.chat_manager.create_session("ml_research")
    studio.chat_manager.add_message(session.id, "user", "Apa keuntungan transformer?")
    studio.chat_manager.add_message(session.id, "assistant", "Transformer memungkinkan komputasi paralel.")
    studio.chat_manager.add_message(
        session.id,
        "system",
        "### [PROSES LATAR BELAKANG SELESAI: Paper Scout]\n- Hasil: 2 paper baru di vault.",
    )

    # Simulate resuming from chat hub
    studio.sub_mode = "chat_hub"
    studio.handle_chat_hub_interaction("resume ml_research")

    output = studio.console.file.getvalue()
    assert studio.sub_mode == "chat"
    assert "Masuk ke sesi percakapan:" in output
    assert "ml_research" in output
    assert "User: Apa keuntungan transformer?" in output
    assert ("AI:" in output or "AI (" in output)
    assert "Sistem:" in output
    assert "PROSES LATAR BELAKANG SELESAI" in output

    # Now ask follow-up question
    studio.handle_chat("Jelaskan detail mekanisme self-attention berdasarkan paper tersebut")

    # Verify history was passed to the LLM
    roles = [r for r, _ in passed_history]
    assert "system" in roles
    assert "user" in roles
    assert "assistant" in roles
    contents = " ".join([c for _, c in passed_history])
    assert "Transformer memungkinkan komputasi paralel" in contents
    assert "PROSES LATAR BELAKANG SELESAI" in contents
    assert "Jelaskan detail mekanisme self-attention" in contents
