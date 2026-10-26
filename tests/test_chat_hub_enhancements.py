"""Unit tests for chat hub enhancements (landing, multi-delete, default naming, resume history)."""

import io
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from scripcy.cli_studio import CLIStudio
from scripcy.core.config import ScripcyConfig
from scripcy.core.project import ProjectManager
from scripcy.core.workspace import WorkspaceManager


def test_chat_command_lands_in_chat_hub(
    temp_workspace: WorkspaceManager,
    tmp_path: Path,
) -> None:
    """Entering /chat or chat lands in skripsi/chat> hub and does not create/jump to a new session."""
    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "test_reg.json")
    proj = pm.create_project("skripsi", temp_workspace.root_dir)
    cfg = ScripcyConfig(workspace_path=str(temp_workspace.root_dir))
    studio = CLIStudio(
        workspace=temp_workspace,
        config=cfg,
        console=console,
        project_manager=pm,
    )
    studio.open_project(proj)

    # Execute 'chat' command
    studio.handle_project_interaction("chat")
    assert studio.sub_mode == "chat_hub"

    # Execute '/chat' command
    studio.sub_mode = None
    studio.handle_project_interaction("/chat")
    assert studio.sub_mode == "chat_hub"


def test_new_chat_name_prompt_and_default(
    temp_workspace: WorkspaceManager,
    tmp_path: Path,
) -> None:
    """Creating a new chat prompts for name, defaulting to session1, session2, etc. when blank."""
    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "test_reg.json")
    proj = pm.create_project("skripsi", temp_workspace.root_dir)
    cfg = ScripcyConfig(workspace_path=str(temp_workspace.root_dir))
    studio = CLIStudio(
        workspace=temp_workspace,
        config=cfg,
        console=console,
        project_manager=pm,
    )
    studio.open_project(proj)
    studio.sub_mode = "chat_hub"

    # User presses Enter without name -> defaults to session1
    with patch.object(studio.console, "input", return_value=""):
        studio.handle_chat_hub_interaction("new")
        assert studio.sub_mode == "chat"
        assert studio.active_chat_session.name == "session1"

    # Back to hub and create another one with blank name -> defaults to session2
    studio.sub_mode = "chat_hub"
    with patch.object(studio.console, "input", return_value=""):
        studio.handle_chat_hub_interaction("new")
        assert studio.sub_mode == "chat"
        assert studio.active_chat_session.name == "session2"

    # Create with specific custom name
    studio.sub_mode = "chat_hub"
    studio.handle_chat_hub_interaction("new Bab1_Pendahuluan")
    assert studio.sub_mode == "chat"
    assert studio.active_chat_session.name == "Bab1_Pendahuluan"


def test_delete_multiple_sessions(
    temp_workspace: WorkspaceManager,
    tmp_path: Path,
) -> None:
    """Delete multiple sessions simultaneously via comma/space separated list or 'all'."""
    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "test_reg.json")
    proj = pm.create_project("skripsi", temp_workspace.root_dir)
    cfg = ScripcyConfig(workspace_path=str(temp_workspace.root_dir))
    studio = CLIStudio(
        workspace=temp_workspace,
        config=cfg,
        console=console,
        project_manager=pm,
    )
    studio.open_project(proj)

    # Seed 3 sessions
    s1 = studio.chat_manager.create_session("session1")
    s2 = studio.chat_manager.create_session("session2")
    s3 = studio.chat_manager.create_session("session3")
    assert len(studio.chat_manager.list_sessions()) == 3

    # Delete 1 and 2 at once
    studio.sub_mode = "chat_hub"
    studio.handle_chat_hub_interaction(f"delete {s1.name}, {s2.name}")

    remaining = studio.chat_manager.list_sessions()
    assert len(remaining) == 1
    assert remaining[0].name == "session3"

    # Seed another session and test 'delete all'
    studio.chat_manager.create_session("session4")
    assert len(studio.chat_manager.list_sessions()) == 2

    studio.handle_chat_hub_interaction("delete all")
    assert len(studio.chat_manager.list_sessions()) == 0


def test_resume_displays_chat_history(
    temp_workspace: WorkspaceManager,
    tmp_path: Path,
) -> None:
    """Resuming/opening a chat session displays previous dialogue history."""
    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "test_reg.json")
    proj = pm.create_project("skripsi", temp_workspace.root_dir)
    cfg = ScripcyConfig(workspace_path=str(temp_workspace.root_dir))
    studio = CLIStudio(
        workspace=temp_workspace,
        config=cfg,
        console=console,
        project_manager=pm,
    )
    studio.open_project(proj)

    sess = studio.chat_manager.create_session("Bab1")
    studio.chat_manager.add_message(sess.id, "user", "Apa itu Transformer?")
    studio.chat_manager.add_message(sess.id, "assistant", "### Definisi\n\n**Transformer** adalah arsitektur deep learning dengan formula $E = mc^2$.")

    studio.sub_mode = "chat_hub"
    studio.handle_chat_hub_interaction("resume Bab1")

    output = studio.console.file.getvalue()
    assert studio.sub_mode == "chat"
    assert "Masuk ke sesi percakapan: Bab1" in output
    assert "Apa itu Transformer?" in output
    assert "Transformer" in output
    assert "Definisi" in output
    # Ensure raw markdown formatting artifacts like raw '###' and raw '$E' are properly parsed/rendered
    assert "### Definisi" not in output
