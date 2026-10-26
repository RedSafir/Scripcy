"""Unit tests for background processing offloading (gdrive, docx, chat actions)."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

from rich.console import Console

from scripcy.cli_studio import CLIStudio
from scripcy.core.config import ScripcyConfig
from scripcy.core.project import ProjectManager
from scripcy.core.workspace import WorkspaceManager


def test_gdrive_dispatches_to_processing(
    temp_workspace: WorkspaceManager,
    tmp_path: Path,
) -> None:
    """Ensure /gdrive submits background task to task manager and doesn't block."""
    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "test_reg.json")
    proj = pm.create_project("test_proj", temp_workspace.root_dir)
    cfg = ScripcyConfig(workspace_path=str(temp_workspace.root_dir))
    studio = CLIStudio(
        workspace=temp_workspace,
        config=cfg,
        console=console,
        project_manager=pm,
    )
    studio.open_project(proj)

    gdrive_url = "https://drive.google.com/file/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/view"
    with patch.object(studio.task_manager, "submit", wraps=studio.task_manager.submit) as mock_submit:
        with patch.object(studio.gdrive_manager, "download_public_file") as mock_dl:
            mock_file = temp_workspace.vault_dir / "sample.pdf"
            mock_file.write_bytes(b"%PDF-1.4 sample")
            mock_dl.return_value = mock_file

            studio.cmd_vault_gdrive(gdrive_url)

            assert mock_submit.called
            call_kwargs = mock_submit.call_args[1]
            assert call_kwargs.get("task_type") == "gdrive"
            assert "Google Drive" in call_kwargs.get("name")


def test_docx_dispatches_to_processing_in_chat(
    temp_workspace: WorkspaceManager,
    tmp_path: Path,
) -> None:
    """Ensure /docx when in chat mode submits a background task to processing."""
    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "test_reg.json")
    proj = pm.create_project("test_proj", temp_workspace.root_dir)
    cfg = ScripcyConfig(workspace_path=str(temp_workspace.root_dir))
    studio = CLIStudio(
        workspace=temp_workspace,
        config=cfg,
        console=console,
        project_manager=pm,
    )
    studio.open_project(proj)
    studio.sub_mode = "chat"

    with patch.object(studio.task_manager, "submit", wraps=studio.task_manager.submit) as mock_submit:
        studio.cmd_docx("my_manuscript.docx")

        assert mock_submit.called
        call_kwargs = mock_submit.call_args[1]
        assert call_kwargs.get("task_type") == "convert"
        assert "my_manuscript.docx" in call_kwargs.get("name")


def test_chat_detects_gdrive_link_and_dispatches(
    temp_workspace: WorkspaceManager,
    tmp_path: Path,
) -> None:
    """Ensure posting a Google Drive link in chat dispatches directly to task manager."""
    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "test_reg.json")
    proj = pm.create_project("test_proj", temp_workspace.root_dir)
    cfg = ScripcyConfig(workspace_path=str(temp_workspace.root_dir))
    studio = CLIStudio(
        workspace=temp_workspace,
        config=cfg,
        console=console,
        project_manager=pm,
    )
    studio.open_project(proj)
    studio.sub_mode = "chat"

    prompt = "tolong download file ini https://drive.google.com/file/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/view"
    with patch.object(studio.task_manager, "submit", wraps=studio.task_manager.submit) as mock_submit:
        studio.handle_chat(prompt)

        assert mock_submit.called
        call_kwargs = mock_submit.call_args[1]
        assert call_kwargs.get("task_type") == "gdrive"
