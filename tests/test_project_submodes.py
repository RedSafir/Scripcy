"""Unit tests for Scrpcy hierarchical sub-modes: Vault, Processing, and Chat."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console

from scrpcy.cli_studio import CLIStudio
from scrpcy.core.config import ScrpcyConfig
from scrpcy.core.gdrive import extract_drive_file_id
from scrpcy.core.project import ProjectManager
from scrpcy.core.workspace import WorkspaceManager


@pytest.fixture
def test_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=True, color_system=None, width=120)


@pytest.fixture
def cli_studio_submodes(
    temp_workspace: WorkspaceManager,
    test_console: Console,
    tmp_path: Path,
) -> CLIStudio:
    """Fixture providing CLIStudio with an active project."""
    pm = ProjectManager(registry_file=tmp_path / "test_reg.json")
    proj = pm.create_project("skripsi", temp_workspace.root_dir)
    cfg = ScrpcyConfig(workspace_path=str(temp_workspace.root_dir))
    studio = CLIStudio(
        workspace=temp_workspace,
        config=cfg,
        console=test_console,
        project_manager=pm,
    )
    studio.open_project(proj)
    return studio


class TestSubmodeNavigation:
    """Tests for entering, navigating, and exiting sub-modes."""

    def test_enter_and_exit_vault_submode(self, cli_studio_submodes: CLIStudio) -> None:
        assert cli_studio_submodes.sub_mode is None

        # Enter vault
        cli_studio_submodes.handle_project_interaction("vault")
        assert cli_studio_submodes.sub_mode == "vault"

        output = cli_studio_submodes.console.file.getvalue()
        assert "Vault Resources" in output

        # Exit back to root (in vault, commands work with or without '/')
        cli_studio_submodes.handle_project_interaction("back")
        assert cli_studio_submodes.sub_mode is None

    def test_enter_and_exit_processing_submode(self, cli_studio_submodes: CLIStudio) -> None:
        cli_studio_submodes.handle_project_interaction("processing")
        assert cli_studio_submodes.sub_mode == "processing"

        output = cli_studio_submodes.console.file.getvalue()
        assert "Processing Status" in output

        # Exit back to root (in processing, commands work with or without '/')
        cli_studio_submodes.handle_project_interaction("back")
        assert cli_studio_submodes.sub_mode is None

    def test_enter_and_exit_chat_submode(self, cli_studio_submodes: CLIStudio) -> None:
        cli_studio_submodes.handle_project_interaction("chat new tesis_ai")
        assert cli_studio_submodes.sub_mode == "chat"
        assert cli_studio_submodes.active_chat_session is not None
        assert cli_studio_submodes.active_chat_session.name == "tesis_ai"

        output = cli_studio_submodes.console.file.getvalue()
        assert "tesis_ai" in output

        # Exit to chat hub via /back
        cli_studio_submodes.handle_project_interaction("/back")
        assert cli_studio_submodes.sub_mode == "chat_hub"

        # Exit back to root via back
        cli_studio_submodes.handle_project_interaction("back")
        assert cli_studio_submodes.sub_mode is None


class TestVaultSubmodeOperations:
    """Tests for actions inside skripsi/vault>."""

    def test_vault_add_and_archive_flow(
        self,
        cli_studio_submodes: CLIStudio,
        tmp_path: Path,
    ) -> None:
        cli_studio_submodes.sub_mode = "vault"

        # Create sample file and add to vault
        sample = tmp_path / "chapter1.txt"
        sample.write_text("Introduction to AI Research", encoding="utf-8")

        cli_studio_submodes.handle_project_interaction(f"add {sample}")
        output = cli_studio_submodes.console.file.getvalue()
        assert "Added resource to vault" in output

        # List vault
        cli_studio_submodes.handle_project_interaction("list")
        output2 = cli_studio_submodes.console.file.getvalue()
        assert "chapter1.txt" in output2

        # Soft-remove from active project (moves to .archive/)
        cli_studio_submodes.handle_project_interaction("remove 1")
        output3 = cli_studio_submodes.console.file.getvalue()
        assert "moved to .archive/" in output3

        # View archived
        cli_studio_submodes.handle_project_interaction("archived")
        output4 = cli_studio_submodes.console.file.getvalue()
        assert "chapter1.txt" in output4

        # Restore from archive
        cli_studio_submodes.handle_project_interaction("restore 1")
        output5 = cli_studio_submodes.console.file.getvalue()
        assert "restored to active vault" in output5


class TestProcessingSubmodeOperations:
    """Tests for background task tracking inside skripsi/processing>."""

    def test_processing_task_lifecycle(self, cli_studio_submodes: CLIStudio) -> None:
        cli_studio_submodes.sub_mode = "processing"

        # Register a task
        t = cli_studio_submodes.task_manager.submit("Scout Task", "scout", lambda: "Found 5 papers")
        cli_studio_submodes.task_manager.log(t.id, "Query formulation done")

        # View list
        cli_studio_submodes.handle_project_interaction("list")
        output = cli_studio_submodes.console.file.getvalue()
        assert t.id in output
        assert "Scout Task" in output

        # View detail
        cli_studio_submodes.handle_project_interaction(f"detail {t.id}")
        output_detail = cli_studio_submodes.console.file.getvalue()
        assert "Task Detail:" in output_detail

        # View log
        cli_studio_submodes.handle_project_interaction(f"log {t.id}")
        output_log = cli_studio_submodes.console.file.getvalue()
        assert "Query formulation done" in output_log


class TestChatSubmodeOperations:
    """Tests for chat sessions inside skripsi/chat1>."""

    def test_chat_multi_sessions(self, cli_studio_submodes: CLIStudio) -> None:
        # Create session 1
        cli_studio_submodes.handle_chat_root_command("new chat1")
        assert cli_studio_submodes.active_chat_session.name == "chat1"

        # Create session 2
        cli_studio_submodes.handle_chat_root_command("new chat2")
        assert cli_studio_submodes.active_chat_session.name == "chat2"

        # Resume session 1
        cli_studio_submodes.handle_chat_root_command("resume chat1")
        assert cli_studio_submodes.active_chat_session.name == "chat1"

    def test_chat_enter_and_back_in_run_loop(self, cli_studio_submodes: CLIStudio) -> None:
        from unittest.mock import patch
        with patch.object(
            cli_studio_submodes.console,
            "input",
            side_effect=["chat new chat_run_test", "/back", "back", "exit"],
        ):
            cli_studio_submodes.run()

        output = cli_studio_submodes.console.file.getvalue()
        assert "Sesi chat baru dibuat: chat_run_test" in output
        assert "Kembali ke menu chat: skripsi/chat>" in output
        assert "Returned to project root: skripsi>" in output
        assert "Closing scrpcy research studio" in output
        assert cli_studio_submodes.sub_mode is None

    def test_chat_keyboard_interrupt_dismisses_submode(self, cli_studio_submodes: CLIStudio) -> None:
        from unittest.mock import patch
        cli_studio_submodes.sub_mode = "chat"
        with patch.object(
            cli_studio_submodes.console,
            "input",
            side_effect=[KeyboardInterrupt(), KeyboardInterrupt(), "exit"],
        ):
            cli_studio_submodes.run()

        assert cli_studio_submodes.sub_mode is None
        output = cli_studio_submodes.console.file.getvalue()
        assert "Returned to chat hub: skripsi/chat>" in output
        assert "Returned to project root: skripsi>" in output


class TestProjectRename:
    """Tests for project renaming."""

    def test_cmd_rename_project(self, cli_studio_submodes: CLIStudio) -> None:
        assert cli_studio_submodes.current_project.name == "skripsi"

        cli_studio_submodes.handle_project_interaction("rename tesis_final")
        assert cli_studio_submodes.current_project.name == "tesis_final"

        output = cli_studio_submodes.console.file.getvalue()
        assert "successfully renamed to" in output
        assert "tesis_final" in output


class TestGDriveLinkExtraction:
    """Tests for Google Drive URL parsing."""

    def test_extract_file_id(self) -> None:
        url1 = "https://drive.google.com/file/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/view"
        assert extract_drive_file_id(url1) == "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"

        url2 = "https://drive.google.com/open?id=1AbCdEfGhIjKlMnOpQrStUvWxYz12345"
        assert extract_drive_file_id(url2) == "1AbCdEfGhIjKlMnOpQrStUvWxYz12345"


class TestRefinedUIAndModeFlows:
    """Tests for refined vault landing, chat hub, processing summary, and help simplicity."""

    def test_vault_landing_shows_only_resources(self, cli_studio_submodes: CLIStudio) -> None:
        cli_studio_submodes.console.file.seek(0)
        cli_studio_submodes.console.file.truncate(0)

        cli_studio_submodes.render_vault_landing()
        out = cli_studio_submodes.console.file.getvalue()

        assert "Vault Resources" in out
        assert "Folder Vault" in out
        assert "Total Berkas" in out
        # Command guide must NOT be printed on landing
        assert "Perintah Vault & Panduan Penggunaan:" not in out
        assert "Tip: Ketik 'list'" not in out

    def test_chat_hub_full_flow(self, cli_studio_submodes: CLIStudio) -> None:
        # 1. Entering /chat enters chat_hub
        cli_studio_submodes.handle_project_interaction("/chat")
        assert cli_studio_submodes.sub_mode == "chat_hub"

        # 2. Creating a new chat from chat hub enters chat session
        cli_studio_submodes.handle_project_interaction("new bab1_pendahuluan")
        assert cli_studio_submodes.sub_mode == "chat"
        assert cli_studio_submodes.active_chat_session is not None
        assert cli_studio_submodes.active_chat_session.name == "bab1_pendahuluan"

        # 3. /back returns to chat_hub
        cli_studio_submodes.handle_project_interaction("/back")
        assert cli_studio_submodes.sub_mode == "chat_hub"

        # 4. In chat hub, can delete chat session
        cli_studio_submodes.handle_project_interaction("delete bab1_pendahuluan")
        session = cli_studio_submodes.chat_manager.get_session("bab1_pendahuluan")
        assert session is None

        # 5. back from chat hub returns to project root
        cli_studio_submodes.handle_project_interaction("back")
        assert cli_studio_submodes.sub_mode is None

    def test_processing_summary_and_detailed_list(self, cli_studio_submodes: CLIStudio) -> None:
        # Register a task
        cli_studio_submodes.task_manager.submit("Unit Test Worker", "test", lambda: "done")

        cli_studio_submodes.console.file.seek(0)
        cli_studio_submodes.console.file.truncate(0)

        # Opening processing shows summary, not detailed list
        cli_studio_submodes.handle_project_interaction("processing")
        assert cli_studio_submodes.sub_mode == "processing"
        out_summary = cli_studio_submodes.console.file.getvalue()
        assert "Processing Status" in out_summary
        assert "Running Tasks" in out_summary
        assert "Total Riwayat" in out_summary
        assert "Unit Test Worker" not in out_summary

        # Typing 'list' shows detailed table
        cli_studio_submodes.handle_project_interaction("list")
        out_list = cli_studio_submodes.console.file.getvalue()
        assert "Background Processing Tasks" in out_list
        assert "Unit Test Worker" in out_list

        cli_studio_submodes.handle_project_interaction("back")
        assert cli_studio_submodes.sub_mode is None

    def test_help_view_simple_available_commands(self) -> None:
        import io
        from rich.console import Console
        from scrpcy.ui.interactive import interactive_help_view

        buf = io.StringIO()
        c = Console(file=buf, force_terminal=True, color_system=None)

        interactive_help_view(console=c, mode="dashboard")
        dashboard_out = buf.getvalue()
        assert "Available Commands [Dashboard]" in dashboard_out
        assert "/projects" in dashboard_out
        assert "Scrpcy CLI" not in dashboard_out
        assert "General Workspace" not in dashboard_out
        assert "Keyboard Shortcuts" not in dashboard_out

        buf.seek(0)
        buf.truncate(0)
        interactive_help_view(console=c, mode="vault", context_label="skripsi")
        vault_out = buf.getvalue()
        assert "Available Commands [skripsi]" in vault_out
        assert "/reload" in vault_out
        assert "/add" in vault_out
        assert "Scrpcy CLI" not in vault_out


        bare_id = "1AbCdEfGhIjKlMnOpQrStUvWxYz12345"
        assert extract_drive_file_id(bare_id) == "1AbCdEfGhIjKlMnOpQrStUvWxYz12345"
