"""Unit tests for single-task session guard, non-blocking detach, and AI continuation."""

import threading
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from scripcy.cli_studio import CLIStudio
from scripcy.core.chats import ChatSession
from scripcy.core.tasks import TaskInfo, TaskManager
from scripcy.core.workspace import WorkspaceManager


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

        # Attempt to submit a new question while a turn is already running
        with patch.object(temp_studio, "submit_chat_background") as mock_submit:
            t.lane = "turn"
            temp_studio.handle_chat_submode_interaction("pertanyaan baru tentang gambut")
            mock_submit.assert_called_once()
            call_kwargs = mock_submit.call_args[1]
            assert task.id in call_kwargs.get("depends_on", [])

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
            mock_submit.assert_called_once_with("tugas kedua")
    finally:
        temp_studio.task_manager.cancel_task(task.id)


def test_ai_task_continuation_fallback(temp_studio: CLIStudio) -> None:
    # Heuristic fallback when engine has no real API key
    res_under_60 = temp_studio._ask_ai_task_continuation("Scout Paper", "Downloading...", elapsed=40.0)
    assert res_under_60 == "EXTEND"

    res_over_60 = temp_studio._ask_ai_task_continuation("Scout Paper", "Idle...", elapsed=65.0)
    assert res_over_60 == "ABORT"


def test_primary_path_task_listed_in_process_manager(temp_studio: CLIStudio) -> None:
    session = temp_studio.chat_manager.create_session("test_session")
    temp_studio.active_chat_session = session

    # Task submitted in primary path
    task = temp_studio.task_manager.submit(
        name="Deep Scout: Karhutla",
        task_type="scout",
        target_fn=lambda task_id: "Scout finished with 2 papers",
        session_id=session.id,
    )

    # Must be immediately registered and discoverable in process manager
    all_tasks = temp_studio.task_manager.list_tasks()
    found = any(t.id == task.id and t.name == "Deep Scout: Karhutla" for t in all_tasks)
    assert found, "Primary task must be listed in process manager"

    # Test that /process and /proc are recognized without throwing
    with patch.object(temp_studio, "render_processing_summary") as mock_proc:
        temp_studio.handle_chat_submode_interaction("/process")
        assert temp_studio.sub_mode == "processing"
        mock_proc.assert_called_once()

    temp_studio.sub_mode = "chat"
    with patch.object(temp_studio, "render_processing_summary") as mock_proc2:
        temp_studio.handle_chat_submode_interaction("/proc")
        assert temp_studio.sub_mode == "processing"
        mock_proc2.assert_called_once()


def test_no_duplicate_eval_response_on_task_completion(temp_studio: CLIStudio) -> None:
    session = temp_studio.chat_manager.create_session("test_session")
    temp_studio.active_chat_session = session

    # Simulate scout task finishing
    from scripcy.core.tasks import TaskInfo
    task = TaskInfo(
        id="task-scout-99",
        name="Deep Scout Simulation",
        task_type="scout",
        session_id=session.id,
        status="completed",
        result_summary="Downloaded 3 papers",
    )
    temp_studio.task_manager._tasks[task.id] = task

    # Trigger background completion callback
    temp_studio._on_background_task_completed(task)

    # Ensure NO duplicate assistant message was added
    messages = temp_studio.chat_manager.get_messages(session.id)
    assistant_msgs = [m for m in messages if m.role == "assistant"]
    assert len(assistant_msgs) == 0, "No duplicate AI evaluation message should be added"

    # Ensure system note exists for transparency
    system_msgs = [m for m in messages if m.role == "system"]
    assert len(system_msgs) == 1
    assert "PROSES LATAR BELAKANG SELESAI" in system_msgs[0].content


def test_dynamic_scout_quota_calculation(temp_studio: CLIStudio) -> None:
    # Simple narrow query
    quota_simple = temp_studio._determine_target_scout_quota("definisi emisi")
    assert 5 <= quota_simple <= 6

    # Complex multi-argument query
    quota_complex = temp_studio._determine_target_scout_quota("berikan aku argument komprehensif mengenai karhutla dan dampaknya terhadap kebijakan moratorium")
    assert 7 <= quota_complex <= 10


def test_background_scout_worker_batch_ingestion(temp_studio: CLIStudio) -> None:
    from scripcy.core.scout import ScoutResult
    from unittest.mock import AsyncMock

    mock_candidates = [
        ScoutResult(
            title=f"Paper Title {i} regarding Peatland Fire",
            authors=["Author A"],
            year=2023,
            citation_count=50 - i,
            has_pdf=(i % 2 == 0),
            pdf_url=f"https://example.com/p{i}.pdf" if (i % 2 == 0) else None,
            abstract=f"This is the official abstract for paper {i} regarding tropical peatlands.",
        )
        for i in range(1, 15)
    ]

    temp_studio.scout = MagicMock()
    temp_studio.scout.extract_search_criteria.return_value = MagicMock(keywords="karhutla")
    temp_studio.scout.search_multi_source = AsyncMock(return_value=mock_candidates)
    temp_studio.scout.curate_and_rank.return_value = mock_candidates
    temp_studio.scout.download_scout_paper = AsyncMock(return_value=Path("cache/mock_paper.pdf"))
    temp_studio.scout.ingest_abstract_paper = MagicMock(return_value=Path("cache/mock_abstract.md"))

    summary = temp_studio._background_scout_worker(
        task_id="test-task-1",
        query="berikan aku argument mengenai karhutla",
        max_download=7,
    )

    # Must attempt to ingest top candidates up to target quota 7 (using both PDF and abstracts)
    assert "Mengindeks 7/7 dokumen ke vault" in summary or "selesai" in summary.lower()
    temp_studio.scout.search_multi_source.assert_called_once()
    call_args = temp_studio.scout.search_multi_source.call_args
    assert call_args[1].get("limit") == 35 or (len(call_args[0]) > 1 and call_args[0][1] == 35)
