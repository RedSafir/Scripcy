import asyncio
import io
import pytest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from rich.console import Console
from scrpcy.cli_studio import CLIStudio
from scrpcy.core.config import ScrpcyConfig
from scrpcy.core.engine import LLMEngine
from scrpcy.core.project import ProjectInfo, ProjectManager
from scrpcy.core.tasks import TaskManager
from scrpcy.core.workspace import WorkspaceManager


class MockStreamChunk:
    def __init__(self, content: str | None) -> None:
        self.choices = [MagicMock(delta=MagicMock(content=content))]


def test_background_scout_prevents_cascading_refinement_and_fake_evidence(
    tmp_path: Path,
) -> None:
    """Ensure that if an auto-scout task is active or times out to background:
    1. It does NOT claim new evidence was loaded from unrelated cache.
    2. The chat synthesis worker does not run runaway refinement turns.
    """
    ws = WorkspaceManager(root_dir=tmp_path / "test_ws")
    ws.ensure_directories()

    # Create an old unrelated file in cache
    old_cache = ws.cache_dir / "Unrelated_Paper_Old.md"
    old_cache.write_text("# Old Paper\nUnrelated content about algorithms.", encoding="utf-8")

    cfg = ScrpcyConfig(workspace_path=str(ws.root_dir))
    cfg.api_keys["gemini"] = "mock_key"
    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "reg1.json")
    proj = pm.create_project("test_proj1", ws.root_dir)

    async def mock_acompletion(**kwargs: Any):
        yield MockStreamChunk("Sintesis konseptual awal sementara data diunduh.")

    engine = LLMEngine(config=cfg, acompletion_fn=mock_acompletion)

    studio = CLIStudio(
        workspace=ws,
        config=cfg,
        console=console,
        project_manager=pm,
        engine=engine,
    )
    studio.open_project(proj)

    # Simulate scout that times out to background
    async def mock_timeout_scout(*args, **kwargs):
        raise asyncio.TimeoutError()

    studio.scout.auto_scout_and_ingest = mock_timeout_scout

    with patch.object(studio.console, "input", return_value="y"):
        studio.handle_chat("carikan data karhutla terkini")

    out_text = console.file.getvalue()
    # Should warn that download is backgrounded
    assert "Unduhan repositori membutuhkan waktu > 20 detik" in out_text
    # MUST NOT falsely claim that new literature from vault was loaded
    assert "Berhasil memuat bukti literatur dari dokumen vault yang baru diunduh" not in out_text


def test_chat_error_cancels_background_task(tmp_path: Path) -> None:
    """Ensure that if chat synthesis times out or raises an error, the worker task is cancelled."""
    ws = WorkspaceManager(root_dir=tmp_path / "test_ws2")
    ws.ensure_directories()

    cfg = ScrpcyConfig(workspace_path=str(ws.root_dir))
    cfg.api_keys["gemini"] = "mock_key"
    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "reg2.json")
    proj = pm.create_project("test_proj2", ws.root_dir)

    async def mock_acompletion(**kwargs: Any):
        yield MockStreamChunk("Sintesis test.")

    engine = LLMEngine(config=cfg, acompletion_fn=mock_acompletion)

    studio = CLIStudio(
        workspace=ws,
        config=cfg,
        console=console,
        project_manager=pm,
        engine=engine,
    )
    studio.open_project(proj)

    # Force _wait_for_chat_task to raise TimeoutError
    with patch.object(studio, "_run_agent_turn_interactive", side_effect=asyncio.TimeoutError("Chat synthesis timeout exceeded")):
        studio.handle_chat("pertanyaan uji coba")

    # The task should be cancelled and Generation error printed
    tasks = studio.task_manager.list_tasks()
    assert len(tasks) > 0
    assert tasks[-1].status == "cancelled"

    out_text = console.file.getvalue()
    assert "Generation error: Chat synthesis timeout exceeded" in out_text
