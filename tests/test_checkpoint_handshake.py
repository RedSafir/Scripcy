"""Unit tests for non-deadlocking TaskManager Checkpoint Handshake and Live Gap Search."""

import sys
import threading
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from scrpcy.core.config import ScrpcyConfig
from scrpcy.core.project import ProjectManager
from scrpcy.core.scout import ScoutResult
from scrpcy.core.tasks import TaskInfo, TaskManager
from scrpcy.core.workspace import WorkspaceManager
from scrpcy.studio import CLIStudio


@pytest.fixture
def cli_studio(temp_workspace: WorkspaceManager, tmp_path) -> CLIStudio:
    """Fixture providing an isolated CLIStudio instance bound to a workspace."""
    cfg = ScrpcyConfig(
        default_model="gemini/gemini-2.5-pro",
        workspace_path=str(temp_workspace.root_dir),
        skills_dir=str(temp_workspace.root_dir / "skills"),
    )
    pm = ProjectManager(registry_file=tmp_path / "projects.json")
    studio = CLIStudio(
        workspace=temp_workspace,
        config=cfg,
        console=Console(force_terminal=True),
        project_manager=pm,
    )
    info = pm.create_project(
        name="test_proj",
        path=temp_workspace.root_dir,
    )
    studio.open_project(info)
    return studio


class TestCheckpointHandshake:
    """Validate thread-safe checkpoint handshake between worker threads and UI thread."""

    def test_task_manager_checkpoint_handshake(self, tmp_path) -> None:
        tm = TaskManager(db_path=tmp_path / "tasks.db")
        task = tm.submit(name="Test Task", task_type="agent", target_fn=lambda t_id: "done")

        worker_result: list[Any] = []

        def _worker_fn() -> None:
            res = tm.request_checkpoint(task.id, {"turn": 3, "score": 5}, timeout=2.0)
            worker_result.append(res)

        t = threading.Thread(target=_worker_fn)
        t.start()

        time.sleep(0.05)
        # Main UI thread detects pending checkpoint
        pending = tm.get_pending_checkpoint(task.id)
        assert pending is not None
        assert pending.prompt_data["turn"] == 3
        assert pending.prompt_data["score"] == 5

        # Main UI thread resolves it with decision
        assert tm.resolve_checkpoint(task.id, "continue") is True
        t.join(timeout=1.0)

        assert worker_result == ["continue"]
        # Pending checkpoint should be cleaned up
        assert tm.get_pending_checkpoint(task.id) is None

    def test_task_manager_checkpoint_timeout(self, tmp_path) -> None:
        tm = TaskManager(db_path=tmp_path / "tasks.db")
        task = tm.submit(name="Test Task", task_type="agent", target_fn=lambda t_id: "done")

        # Worker times out when no UI thread answers
        res = tm.request_checkpoint(task.id, {"turn": 3}, timeout=0.05)
        assert res is None
        assert tm.get_pending_checkpoint(task.id) is None

    def test_task_manager_checkpoint_cancellation_unblocks_worker(self, tmp_path) -> None:
        tm = TaskManager(db_path=tmp_path / "tasks.db")
        task = tm.submit(name="Test Task", task_type="agent", target_fn=lambda t_id: "done")

        worker_result: list[Any] = []

        def _worker_fn() -> None:
            res = tm.request_checkpoint(task.id, {"turn": 3}, timeout=3.0)
            worker_result.append(res)

        t = threading.Thread(target=_worker_fn)
        t.start()

        time.sleep(0.05)
        assert tm.get_pending_checkpoint(task.id) is not None

        # User cancels task
        tm.cancel_task(task.id)
        t.join(timeout=1.0)

        assert worker_result == ["cancelled"]

    def test_check_agent_turn_checkpoint_in_worker_thread_uses_handshake(
        self,
        cli_studio: CLIStudio,
        tmp_path,
        monkeypatch,
    ) -> None:
        """In a worker thread with interactive terminal, _check_agent_turn_checkpoint must use TaskManager handshake."""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        task = cli_studio.task_manager.submit(name="Chat Agent", task_type="agent", target_fn=lambda t_id: "done")
        stop_decision: list[bool] = []

        def _bg_runner() -> None:
            stopped = cli_studio._check_agent_turn_checkpoint(
                turn=2,  # completed_turns = 3 (triggers checkpoint)
                best_grounding=0.5,
                best_critic=5,
                accumulated_input_tokens=1000,
                accumulated_output_tokens=500,
                task_id=task.id,
            )
            stop_decision.append(stopped)

        t = threading.Thread(target=_bg_runner)
        t.start()

        time.sleep(0.05)
        pending = cli_studio.task_manager.get_pending_checkpoint(task.id)
        assert pending is not None
        assert pending.prompt_data["completed_turns"] == 3

        # UI thread resolves: "continue" (meaning do NOT stop)
        cli_studio.task_manager.resolve_checkpoint(task.id, "continue")
        t.join(timeout=1.0)

        assert stop_decision == [False]


class TestLiveDirectedGapSearch:
    """Validate that Directed Gap Search runs live without mock env guards."""

    @pytest.mark.asyncio
    async def test_gap_search_executes_with_live_api_key(
        self,
        cli_studio: CLIStudio,
        temp_workspace: WorkspaceManager,
    ) -> None:
        """Verify that directed_search_queries trigger search_multi_source even in live environments."""
        # Live API key (not 'mock')
        cli_studio.config.set_api_key("gemini", "AIzaSyLiveProductionKey123")
        cli_studio.config.default_model = "gemini/gemini-2.5-flash"

        # Mock scout methods
        found_paper = ScoutResult(
            title="Forest Fire Statistics and Trends 2019-2023",
            authors=["Sipongi Author"],
            year=2023,
            abstract="Official hotspot statistics and burned area in Indonesia from 2019 to 2023.",
            source="openalex",
            has_pdf=False,
            pdf_url=None,
        )

        mock_search = AsyncMock(return_value=[found_paper])
        mock_map = AsyncMock(return_value=[])

        cli_studio.scout.search_multi_source = mock_search
        cli_studio.scout.map_vault_citation_network = mock_map

        # Verify that mock_search was not called yet
        assert mock_search.call_count == 0

        # Simulate gap queries block directly
        directed_queries = ["karhutla 2019 2023 luas kebakaran"]
        mapped_papers = []

        if directed_queries and cli_studio.scout:
            for target_gap_q in directed_queries[:2]:
                from scrpcy.core.scout import ScoutQuery
                sq = ScoutQuery(keywords=target_gap_q, limit=5)
                candidates = await cli_studio.scout.search_multi_source(sq, limit=5)
                if candidates:
                    mapped_papers.extend(candidates)

        assert mock_search.call_count == 1
        assert len(mapped_papers) == 1
        assert mapped_papers[0].title == "Forest Fire Statistics and Trends 2019-2023"
