"""Comprehensive unit tests for the Dashboard and Project CLI research studio."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from scripcy.cli_studio import CLIStudio, SCRIPCY_LOGO, SCRIPCY_VERSION
from scripcy.core.config import ConfigManager, ScripcyConfig
from scripcy.core.models import DocumentMetadata
from scripcy.core.project import ProjectInfo, ProjectManager
from scripcy.core.search import SearchResult
from scripcy.core.workspace import WorkspaceManager


@pytest.fixture
def test_console() -> Console:
    """Fixture providing an in-memory capturing console."""
    return Console(file=io.StringIO(), force_terminal=True, color_system=None)


@pytest.fixture
def project_manager(tmp_path: Path) -> ProjectManager:
    """Fixture providing an isolated project registry manager."""
    registry_file = tmp_path / "test_projects.json"
    return ProjectManager(registry_file=registry_file)


@pytest.fixture
def cli_studio(
    temp_workspace: WorkspaceManager,
    test_console: Console,
    project_manager: ProjectManager,
) -> CLIStudio:
    """Fixture providing an isolated CLIStudio instance bound to a workspace."""
    cfg = ScripcyConfig(
        default_model="gemini/gemini-2.5-pro",
        workspace_path=str(temp_workspace.root_dir),
        skills_dir=str(temp_workspace.root_dir / "skills"),
    )
    studio = CLIStudio(
        workspace=temp_workspace,
        config=cfg,
        console=test_console,
        project_manager=project_manager,
    )
    # Simulate an active master project for project-level tests
    info = project_manager.create_project(
        name="test_proj",
        path=temp_workspace.root_dir,
    )
    studio.open_project(info)
    return studio


class TestBigAsciiBanner:
    """Tests for large stylized ASCII logo and version rendering."""

    def test_big_ascii_logo_structure(self) -> None:
        assert "██" in SCRIPCY_LOGO
        assert SCRIPCY_VERSION.startswith("v")

    def test_banner_rendering(self, test_console: Console, project_manager: ProjectManager) -> None:
        studio = CLIStudio(console=test_console, project_manager=project_manager)
        studio.print_banner()

        output = test_console.file.getvalue()
        assert SCRIPCY_VERSION in output
        assert "Academic Research Studio" in output


class TestDashboardFlow:
    """Tests for initial Dashboard mode, project creation, and system status."""

    def test_dashboard_help(self, test_console: Console, project_manager: ProjectManager) -> None:
        studio = CLIStudio(console=test_console, project_manager=project_manager)
        studio.print_dashboard_help()
        output = test_console.file.getvalue()
        assert "Available Commands [Dashboard]" in output
        assert "/projects" in output
        assert "/new" in output
        assert "/open" in output
        assert "/models" in output
        assert "/settings" in output
        assert "/help" in output

    def test_dashboard_status(self, test_console: Console, project_manager: ProjectManager, tmp_path: Path) -> None:
        cfg = ScripcyConfig(default_model="gemini/gemini-2.5-pro")
        studio = CLIStudio(console=test_console, config=cfg, project_manager=project_manager)
        
        # When no projects registered
        studio.dashboard_status()
        output_empty = test_console.file.getvalue()
        assert "No research projects registered yet" in output_empty

        # When project registered
        proj_dir = tmp_path / "thesis_alpha"
        proj_dir.mkdir(parents=True, exist_ok=True)
        project_manager.create_project("thesis_alpha", proj_dir)

        studio.dashboard_status()
        output = test_console.file.getvalue()
        assert "Research Project Status: thesis_alpha" in output
        assert "Project Token Usage" in output

    def test_dashboard_new_and_open_project(
        self,
        tmp_path: Path,
        test_console: Console,
        project_manager: ProjectManager,
    ) -> None:
        studio = CLIStudio(console=test_console, project_manager=project_manager)
        proj_dir = tmp_path / "my_master_thesis"

        # Create new project
        studio.dashboard_new_project(["my_thesis", str(proj_dir)])
        assert studio.current_project is not None
        assert studio.current_project.name == "my_thesis"
        assert proj_dir.is_dir()
        assert (proj_dir / "vault").is_dir()
        assert (proj_dir / "cache").is_dir()
        assert (proj_dir / "drafts").is_dir()
        assert (proj_dir / "logs").is_dir()

        # Switch back to dashboard
        studio.cmd_return_dashboard()
        assert studio.current_project is None

        # Reopen project by name
        studio.dashboard_open_project(["my_thesis"])
        assert studio.current_project is not None
        assert studio.current_project.name == "my_thesis"

    def test_dashboard_list_projects(
        self,
        tmp_path: Path,
        test_console: Console,
        project_manager: ProjectManager,
    ) -> None:
        studio = CLIStudio(console=test_console, project_manager=project_manager)
        # Initially empty
        studio.dashboard_list_projects()
        output_empty = test_console.file.getvalue()
        assert "No projects registered yet" in output_empty

        # Create projects
        project_manager.create_project("alpha", tmp_path / "alpha")
        project_manager.create_project("beta", tmp_path / "beta")

        studio.dashboard_list_projects()
        output = test_console.file.getvalue()
        assert "Registered Research Projects" in output
        assert "alpha" in output
        assert "beta" in output


class TestProjectCommands:
    """Tests for project-level slash commands within master project folder."""

    def test_cmd_vault_empty(self, cli_studio: CLIStudio) -> None:
        cli_studio.cmd_vault()
        output = cli_studio.console.file.getvalue()
        assert "Project vault is currently empty" in output

    def test_cmd_vault_populated(self, cli_studio: CLIStudio, temp_workspace: WorkspaceManager) -> None:
        meta = DocumentMetadata(
            title="Attention Is All You Need",
            authors=["Ashish Vaswani"],
            year=2017,
            cite_key="vaswani2017attention",
        )
        temp_workspace.save_to_cache("attention.md", f"{meta.to_frontmatter()}\nTransformer content.")
        cli_studio._sync_vault_cache()

        cli_studio.cmd_vault()
        output = cli_studio.console.file.getvalue()
        assert "Indexed Research Papers" in output
        assert "vaswani2017" in output
        assert "Attention Is" in output
        assert "All You Need" in output

    def test_cmd_active_toggle(self, cli_studio: CLIStudio, temp_workspace: WorkspaceManager) -> None:
        temp_workspace.save_to_cache("doc1.md", "---\ntitle: Doc 1\n---\nContent 1")
        temp_workspace.save_to_cache("doc2.md", "---\ntitle: Doc 2\n---\nContent 2")
        cli_studio._sync_vault_cache()

        cli_studio.cmd_active("clear")
        assert len(cli_studio.active_documents) == 0

        cli_studio.cmd_active("doc1")
        assert "doc1" in cli_studio.active_documents

        cli_studio.cmd_active("doc1")
        assert "doc1" not in cli_studio.active_documents

        cli_studio.cmd_active("1")
        assert len(cli_studio.active_documents) == 1

        cli_studio.cmd_active("all")
        assert len(cli_studio.active_documents) == 2

    def test_cmd_skills_deprecated(self, cli_studio: CLIStudio) -> None:
        cli_studio.cmd_skills()
        output = cli_studio.console.file.getvalue()
        assert "Perintah /skills telah ditiadakan" in output

    def test_cmd_skill_deprecated(self, cli_studio: CLIStudio) -> None:
        cli_studio.cmd_skill("custom_reviewer")
        output = cli_studio.console.file.getvalue()
        assert "Perintah /skill telah ditiadakan" in output

    def test_dashboard_skills_deprecated(self, test_console: Console, project_manager: ProjectManager) -> None:
        studio = CLIStudio(console=test_console, project_manager=project_manager)
        studio.handle_dashboard_command("skills")
        output = test_console.file.getvalue()
        assert "Perintah /skill telah ditiadakan" in output

    def test_project_slash_skill_routing_deprecated(self, cli_studio: CLIStudio) -> None:
        cli_studio.handle_project_command("/skills")
        output = cli_studio.console.file.getvalue()
        assert "Perintah /skill telah ditiadakan" in output

    def test_cmd_model_and_style(self, cli_studio: CLIStudio) -> None:
        cli_studio.cmd_model("openai/gpt-4o")
        assert cli_studio.get_active_project_model() == "openai/gpt-4o"

        cli_studio.cmd_style("ieee")
        assert cli_studio.config.citation.style == "ieee"

    def test_cmd_models_and_settings(self, cli_studio: CLIStudio) -> None:
        # Test models list
        cli_studio.cmd_models("list")
        output = cli_studio.console.file.getvalue()
        assert "Model & Provider" in output

        # Test set-key
        cli_studio.cmd_models("set-key gemini test-key-secret-999")
        assert cli_studio.config.api_keys["gemini"] == "test-key-secret-999"

        # Test 9Router setup
        cli_studio.cmd_models("9router")
        assert cli_studio.config.default_model.startswith("9router/")
        assert "9router" in cli_studio.config.api_bases

        # Test tokens
        cli_studio.cmd_models("tokens")
        tok_out = cli_studio.console.file.getvalue()
        assert "Token Consumption" in tok_out

        # Test settings show and set-citation
        cli_studio.cmd_settings("show")
        set_out = cli_studio.console.file.getvalue()
        assert "Workspace & Citation Settings" in set_out

        cli_studio.cmd_settings("set-citation apa")
        assert cli_studio.config.citation.style == "apa"

    def test_cmd_config_compatibility_alias(self, cli_studio: CLIStudio) -> None:
        cli_studio.cmd_config("show")
        output = cli_studio.console.file.getvalue()
        assert "Model & Provider" in output

    def test_cmd_models_cancel_does_not_dump_tables(self, cli_studio: CLIStudio, monkeypatch: pytest.MonkeyPatch) -> None:
        """When user cancels model selector via Esc (None), it should return cleanly without dumping 4 tables."""
        import sys
        import scripcy.cli_studio as _cli_studio
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(_cli_studio, "interactive_model_selector", lambda **kwargs: None)
        monkeypatch.setattr("scripcy.studio.commands.interactive_model_selector", lambda **kwargs: None)
        monkeypatch.setattr("scripcy.studio.commands.fetch_9router_models", lambda base: [])
        cli_studio.console.file = io.StringIO()

        cli_studio.cmd_models("")
        out = cli_studio.console.file.getvalue()
        # Must NOT dump inventory tables on cancel
        assert "scripcy Model & Provider Inventory" not in out
        assert "Panduan Perintah 'models'" not in out

    def test_interactive_settings_hub_navigation(self, cli_studio: CLIStudio, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify unified interactive settings hub navigation and sub-menu dispatch."""
        calls = []

        def mock_interactive_menu(options, **kwargs):
            first_key = options[0][0]
            calls.append(first_key)
            return "back"

        monkeypatch.setattr("scripcy.studio.dashboard.interactive_menu", mock_interactive_menu)
        cli_studio.run_interactive_settings()

        assert "model" in calls

    def test_interactive_federation_settings_toggle(self, cli_studio: CLIStudio, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify 1-click toggle of federation providers without typing commands."""
        # Current state: True
        assert getattr(cli_studio.config, "enable_semantic_scholar", True) is True

        turn = 0
        def mock_menu(options, **kwargs):
            nonlocal turn
            turn += 1
            if turn == 1:
                return "s2_toggle"
            return "back"

        monkeypatch.setattr("scripcy.studio.dashboard.interactive_menu", mock_menu)
        cli_studio._run_interactive_federation_settings()

        # Should be toggled to False
        assert cli_studio.config.enable_semantic_scholar is False

    def test_interactive_agent_limits_settings(self, cli_studio: CLIStudio, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify interactive adjustment of max turns and output tokens."""
        turn = 0
        def mock_menu(options, **kwargs):
            nonlocal turn
            turn += 1
            if turn == 1:
                return "turns"
            elif turn == 2:
                return "25"
            return "back"

        monkeypatch.setattr("scripcy.studio.dashboard.interactive_menu", mock_menu)
        cli_studio._run_interactive_agent_limits_settings()

        assert cli_studio.config.max_agent_turns == 25

    def test_interactive_menu_renders_rich_markup_without_raw_tags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify rich markup in option labels is parsed into ANSI styles, never printed as raw tags."""
        import sys
        import io
        from rich.console import Console
        from scripcy.ui.interactive import interactive_menu

        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        # Mock single ENTER keypress
        monkeypatch.setattr("scripcy.ui.interactive._read_key", lambda: "ENTER")

        out_buf = io.StringIO()
        con = Console(file=out_buf, color_system="truecolor", force_terminal=True)

        options = [
            ("gemini", "Google Gemini : [bold cyan]Configured (tes...-999)[/bold cyan]"),
            ("claude", "Anthropic Claude : [dim]Not Set[/dim]"),
        ]

        chosen = interactive_menu(options, console=con)
        assert chosen == "gemini"

        output = out_buf.getvalue()
        # Raw markup tags must NOT be printed literally
        assert "[bold cyan]" not in output
        assert "[/bold cyan]" not in output
        assert "[dim]" not in output
        assert "[/dim]" not in output

    def test_cmd_docx_export(self, cli_studio: CLIStudio, temp_workspace: WorkspaceManager) -> None:
        meta = DocumentMetadata(
            title="Transformer Review",
            authors=["Vaswani"],
            year=2017,
            cite_key="vaswani2017",
        )
        temp_workspace.save_to_cache("trans.md", f"{meta.to_frontmatter()}\nBody.")
        cli_studio._sync_vault_cache()

        cli_studio.conversation_history.append({
            "query": "Explain Transformers",
            "response": "Transformers rely solely on attention mechanisms.",
        })

        out_docx = "cli_export_test.docx"
        cli_studio.cmd_docx(out_docx)

        drafts_dir = temp_workspace.root_dir / "drafts"
        docx_path = drafts_dir / out_docx
        assert docx_path.exists()
        assert docx_path.stat().st_size > 0

    def test_cmd_search_and_download(
        self,
        cli_studio: CLIStudio,
        temp_workspace: WorkspaceManager,
        synthetic_pdf_factory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sample_results = [
            SearchResult(
                title="Attention Is All You Need",
                authors=["Ashish Vaswani"],
                year=2017,
                pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
            )
        ]

        async def mock_search(*args: Any, **kwargs: Any):
            return sample_results

        pdf_file = synthetic_pdf_factory(filename="arxiv_mock.pdf")

        async def mock_download(*args: Any, **kwargs: Any):
            return pdf_file

        monkeypatch.setattr(cli_studio, "_execute_search", mock_search)
        monkeypatch.setattr(cli_studio, "_execute_download", mock_download)

        cli_studio.cmd_search("attention")
        assert len(cli_studio.last_search_results) == 1

        cli_studio.cmd_download("1")
        assert len(temp_workspace.list_cached_documents()) > 0

    def test_handle_chat_without_api_key(self, cli_studio: CLIStudio) -> None:
        cli_studio.config.api_keys.clear()
        cli_studio.handle_chat("What is self-attention?")
        output = cli_studio.console.file.getvalue()
        assert "API Key Required" in output or "No API key configured" in output


class TestProjectManagerClass:
    """Direct tests for the ProjectManager model and file operations."""

    def test_create_and_list_projects(self, project_manager: ProjectManager, tmp_path: Path) -> None:
        p1 = project_manager.create_project("paper1", tmp_path / "paper1", description="Paper 1")
        assert p1.name == "paper1"
        assert (tmp_path / "paper1" / "vault").is_dir()
        assert (tmp_path / "paper1" / "drafts").is_dir()
        assert (tmp_path / "paper1" / ".scripcy" / "project.json").is_file()

        projects = project_manager.list_projects()
        assert len(projects) == 1
        assert projects[0].name == "paper1"

    def test_open_by_path_auto_registers(self, project_manager: ProjectManager, tmp_path: Path) -> None:
        foreign_folder = tmp_path / "unregistered_folder"
        foreign_folder.mkdir()

        proj = project_manager.open_project(foreign_folder)
        assert proj.resolved_path == foreign_folder.resolve()
        assert (foreign_folder / "vault").is_dir()
        assert (foreign_folder / "drafts").is_dir()

        # Check it is now in registry
        all_proj = project_manager.list_projects()
        assert any(p.name == proj.name for p in all_proj)

    def test_remove_project(self, project_manager: ProjectManager, tmp_path: Path) -> None:
        p = project_manager.create_project("remove_me", tmp_path / "remove_me")
        assert len(project_manager.list_projects()) == 1

        removed = project_manager.remove_project("remove_me")
        assert removed is True
        assert len(project_manager.list_projects()) == 0
        assert (tmp_path / "remove_me").is_dir()  # Physical directory preserved

    def test_get_project_by_name_and_index(self, project_manager: ProjectManager, tmp_path: Path) -> None:
        p1 = project_manager.create_project("proj_alpha", tmp_path / "alpha")
        p2 = project_manager.create_project("proj_beta", tmp_path / "beta")

        found_by_name = project_manager.get_project("proj_alpha")
        assert found_by_name is not None
        assert found_by_name.name == "proj_alpha"

        found_by_idx = project_manager.get_project(1)
        assert found_by_idx is not None
        assert found_by_idx.name in ("proj_alpha", "proj_beta")

        assert project_manager.get_project("non_existent") is None
        assert project_manager.get_project(99) is None

    def test_remove_project_by_index(self, project_manager: ProjectManager, tmp_path: Path) -> None:
        project_manager.create_project("target_proj", tmp_path / "target")
        assert len(project_manager.list_projects()) == 1

        removed = project_manager.remove_project(1)
        assert removed is True
        assert len(project_manager.list_projects()) == 0


class TestProjectDeletionFlow:
    """Tests for CLI project deletion in Dashboard and Project modes."""

    def test_dashboard_delete_project_confirmed(
        self,
        test_console: Console,
        project_manager: ProjectManager,
        tmp_path: Path,
    ) -> None:
        proj_dir = tmp_path / "my_study"
        project_manager.create_project("my_study", proj_dir)
        studio = CLIStudio(console=test_console, project_manager=project_manager)

        with patch.object(studio.console, "input", side_effect=["y"]):
            studio.handle_dashboard_command("delete my_study")

        output = test_console.file.getvalue()
        assert "removed from Scripcy" in output
        assert "remain safe and untouched on disk" in output
        assert len(project_manager.list_projects()) == 0
        assert proj_dir.is_dir()

    def test_dashboard_delete_project_by_index(
        self,
        test_console: Console,
        project_manager: ProjectManager,
        tmp_path: Path,
    ) -> None:
        proj_dir = tmp_path / "study_one"
        project_manager.create_project("study_one", proj_dir)
        studio = CLIStudio(console=test_console, project_manager=project_manager)

        with patch.object(studio.console, "input", side_effect=["y"]):
            studio.handle_dashboard_command("delete 1")

        output = test_console.file.getvalue()
        assert "removed from Scripcy" in output
        assert len(project_manager.list_projects()) == 0
        assert proj_dir.is_dir()

    def test_dashboard_delete_project_cancelled(
        self,
        test_console: Console,
        project_manager: ProjectManager,
        tmp_path: Path,
    ) -> None:
        project_manager.create_project("keep_me", tmp_path / "keep_me")
        studio = CLIStudio(console=test_console, project_manager=project_manager)

        with patch.object(studio.console, "input", side_effect=["n"]):
            studio.handle_dashboard_command("delete keep_me")

        output = test_console.file.getvalue()
        assert "Deletion canceled." in output
        assert len(project_manager.list_projects()) == 1

    def test_dashboard_delete_project_empty_registry(
        self,
        test_console: Console,
        project_manager: ProjectManager,
    ) -> None:
        studio = CLIStudio(console=test_console, project_manager=project_manager)
        studio.handle_dashboard_command("delete nonexistent")
        output = test_console.file.getvalue()
        assert "No registered projects found to delete" in output

    def test_dashboard_delete_project_not_found(
        self,
        test_console: Console,
        project_manager: ProjectManager,
        tmp_path: Path,
    ) -> None:
        project_manager.create_project("real_project", tmp_path / "real_project")
        studio = CLIStudio(console=test_console, project_manager=project_manager)
        studio.handle_dashboard_command("delete nonexistent")
        output = test_console.file.getvalue()
        assert "not found in registry" in output

    def test_cmd_delete_active_project_in_project_mode(
        self,
        test_console: Console,
        project_manager: ProjectManager,
        tmp_path: Path,
    ) -> None:
        proj_dir = tmp_path / "active_study"
        info = project_manager.create_project("active_study", proj_dir)
        studio = CLIStudio(console=test_console, project_manager=project_manager)
        studio.open_project(info)

        assert studio.current_project is not None
        assert studio.current_project.name == "active_study"

        with patch.object(studio.console, "input", side_effect=["y"]):
            studio.handle_project_command("/delete")

        output = test_console.file.getvalue()
        assert "active_study" in output
        assert "removed from Scripcy" in output
        assert "Returned to Dashboard" in output
        assert studio.current_project is None
        assert studio.workspace is None
        assert len(project_manager.list_projects()) == 0
        assert proj_dir.is_dir()

    def test_cmd_delete_active_project_cancelled(
        self,
        test_console: Console,
        project_manager: ProjectManager,
        tmp_path: Path,
    ) -> None:
        info = project_manager.create_project("cancel_study", tmp_path / "cancel_study")
        studio = CLIStudio(console=test_console, project_manager=project_manager)
        studio.open_project(info)

        with patch.object(studio.console, "input", side_effect=["n"]):
            studio.handle_project_command("/delete")

        output = test_console.file.getvalue()
        assert "Deletion canceled." in output
        assert studio.current_project is not None
        assert len(project_manager.list_projects()) == 1


class TestNewProjectWizardFlow:
    """Tests for interactive 'new' project creation wizard."""

    def test_wizard_custom_path_success(
        self,
        test_console: Console,
        project_manager: ProjectManager,
        tmp_path: Path,
    ) -> None:
        custom_folder = tmp_path / "custom_location"
        studio = CLIStudio(console=test_console, project_manager=project_manager)

        with patch.object(
            studio.console,
            "input",
            side_effect=["my_wizard_study", "2", str(custom_folder)],
        ):
            studio.handle_dashboard_command("new")

        output = test_console.file.getvalue()
        assert "New Project Wizard" in output
        assert "my_wizard_study" in output
        assert "initialized and connected" in output
        assert studio.current_project is not None
        assert studio.current_project.name == "my_wizard_study"
        assert custom_folder.is_dir()
        assert (custom_folder / "vault").is_dir()

    def test_wizard_current_dir_success(
        self,
        test_console: Console,
        project_manager: ProjectManager,
    ) -> None:
        studio = CLIStudio(console=test_console, project_manager=project_manager)

        with patch.object(
            studio.console,
            "input",
            side_effect=["quick_study", "1"],
        ):
            studio.handle_dashboard_command("new")

        output = test_console.file.getvalue()
        assert "quick_study" in output
        assert studio.current_project is not None
        assert studio.current_project.name == "quick_study"

    def test_wizard_cancel_at_name(
        self,
        test_console: Console,
        project_manager: ProjectManager,
    ) -> None:
        studio = CLIStudio(console=test_console, project_manager=project_manager)

        with patch.object(studio.console, "input", side_effect=["q"]):
            studio.handle_dashboard_command("new")

        output = test_console.file.getvalue()
        assert "Project creation canceled." in output
        assert studio.current_project is None
        assert len(project_manager.list_projects()) == 0

    def test_wizard_cancel_at_location(
        self,
        test_console: Console,
        project_manager: ProjectManager,
    ) -> None:
        studio = CLIStudio(console=test_console, project_manager=project_manager)

        with patch.object(studio.console, "input", side_effect=["valid_name", "q"]):
            studio.handle_dashboard_command("new")

        output = test_console.file.getvalue()
        assert "Project creation canceled." in output
        assert studio.current_project is None
        assert len(project_manager.list_projects()) == 0

    def test_handle_suggested_actions_confirm(
        self,
        cli_studio: CLIStudio,
    ) -> None:
        executed_cmds: list[str] = []
        with patch.object(cli_studio, "handle_project_command", side_effect=lambda cmd: executed_cmds.append(cmd)):
            with patch.object(cli_studio.console, "input", side_effect=["y"]):
                response = (
                    "Berdasarkan analisis, kita perlu paper tambahan:\n"
                    "```scripcy\n"
                    "findpaper machine learning 2023\n"
                    "```\n"
                    "Silakan jalankan aksi di atas."
                )
                cli_studio._handle_suggested_actions(response)

        assert len(executed_cmds) == 1
        assert executed_cmds[0] == "/findpaper machine learning 2023"
        output = cli_studio.console.file.getvalue()
        assert "Aksi Scripcy Disarankan AI" in output
        assert "Menjalankan aksi" in output

    def test_handle_suggested_actions_decline(
        self,
        cli_studio: CLIStudio,
    ) -> None:
        executed_cmds: list[str] = []
        with patch.object(cli_studio, "handle_project_command", side_effect=lambda cmd: executed_cmds.append(cmd)):
            with patch.object(cli_studio.console, "input", side_effect=["n"]):
                response = (
                    "Tinjauan selesai:\n"
                    "```scripcy\n"
                    "docx bab1.docx\n"
                    "```\n"
                )
                cli_studio._handle_suggested_actions(response)

        assert len(executed_cmds) == 0
        output = cli_studio.console.file.getvalue()
        assert "Aksi Scripcy Disarankan AI" in output
        assert "dilewati" in output

    def test_chat_submode_slash_commands_and_plain_queries(
        self,
        cli_studio: CLIStudio,
    ) -> None:
        cli_studio.sub_mode = "chat"
        executed_args: list[str] = []
        with patch.object(cli_studio, "cmd_findpaper", side_effect=lambda arg: executed_args.append(arg)):
            with patch.object(cli_studio, "handle_chat") as mock_chat:
                # 1. Slash command inside chat mode -> routes to command handler, never sent to LLM
                cli_studio.handle_chat_submode_interaction("/findpaper deep learning 2024")
                assert not mock_chat.called
                assert len(executed_args) == 1
                assert executed_args[0] == "deep learning 2024"

                # 2. Plain text without slash -> strictly treated as query, never intercepts command
                cli_studio.handle_chat_submode_interaction("findpaper deep learning 2024")
                assert mock_chat.called
                mock_chat.assert_called_once_with("findpaper deep learning 2024")

    def test_project_root_plain_text_prompts_to_enter_chat_mode(
        self,
        cli_studio: CLIStudio,
    ) -> None:
        cli_studio.sub_mode = None
        with patch.object(cli_studio, "handle_chat") as mock_chat:
            cli_studio.handle_project_root_interaction("Bagaimana cara kerja transformer?")
            assert not mock_chat.called
            output = cli_studio.console.file.getvalue()
            assert "Perintah tidak dikenal:" in output
            assert "chat" in output

    def test_project_root_commands_work_without_slash(
        self,
        cli_studio: CLIStudio,
    ) -> None:
        cli_studio.sub_mode = None
        # 1. status without slash
        cli_studio.handle_project_root_interaction("status")
        output = cli_studio.console.file.getvalue()
        assert "Status Project:" in output
        assert "Project Model" in output
        assert "Project Token Usage" in output

        # 2. chat without slash enters chat hub
        cli_studio.handle_project_root_interaction("chat")
        assert cli_studio.sub_mode == "chat_hub"

    def test_chat_mode_plain_text_like_back_or_exit_sent_to_ai(
        self,
        cli_studio: CLIStudio,
    ) -> None:
        cli_studio.sub_mode = "chat"
        with patch.object(cli_studio, "handle_chat") as mock_chat:
            # Plain "back" without slash MUST be treated as an AI query, not exit
            cli_studio.handle_chat_submode_interaction("back")
            assert mock_chat.called
            mock_chat.assert_called_with("back")
            assert cli_studio.sub_mode == "chat"

            # Plain "exit" without slash MUST also be treated as an AI query
            mock_chat.reset_mock()
            cli_studio.handle_chat_submode_interaction("exit")
            assert mock_chat.called
            mock_chat.assert_called_with("exit")
            assert cli_studio.sub_mode == "chat"

            # Plain "status" without slash is treated as an AI prompt
            mock_chat.reset_mock()
            cli_studio.handle_chat_submode_interaction("status")
            assert mock_chat.called
            mock_chat.assert_called_with("status")

        # Now with leading slash, /status calls project status
        with patch.object(cli_studio, "cmd_project_status") as mock_status:
            cli_studio.handle_chat_submode_interaction("/status")
            assert mock_status.called

        # /back cleanly exits chat session back to chat hub
        cli_studio.handle_chat_submode_interaction("/back")
        assert cli_studio.sub_mode == "chat_hub"

        # back from chat hub exits to project root
        cli_studio.handle_chat_hub_interaction("back")
        assert cli_studio.sub_mode is None

    def test_dashboard_status_without_args_prompts_open_project(
        self,
        test_console: Console,
        project_manager: ProjectManager,
    ) -> None:
        studio = CLIStudio(console=test_console, project_manager=project_manager)
        studio.handle_dashboard_command("status")
        output = test_console.file.getvalue()
        assert "Perintah 'status' digunakan di dalam project" in output

    def test_enter_chat_mode_via_slash_chat(
        self,
        cli_studio: CLIStudio,
    ) -> None:
        cli_studio.sub_mode = None
        cli_studio.handle_project_interaction("/chat")
        assert cli_studio.sub_mode == "chat_hub"
        output = cli_studio.console.file.getvalue()
        assert "Saved Chat Sessions" in output

        # In chat hub, create new chat
        cli_studio.handle_project_interaction("new my_session")
        assert cli_studio.sub_mode == "chat"
        assert cli_studio.active_chat_session is not None
        assert cli_studio.active_chat_session.name == "my_session"

        # Exit back to chat hub via /back
        cli_studio.handle_project_interaction("/back")
        assert cli_studio.sub_mode == "chat_hub"

        # Exit back to root via back
        cli_studio.handle_project_interaction("back")
        assert cli_studio.sub_mode is None

    def test_cmd_findpaper_dispatches_background_scout_task(
        self,
        cli_studio: CLIStudio,
    ) -> None:
        cli_studio.cmd_findpaper("quantum machine learning 2024")
        output = cli_studio.console.file.getvalue()
        assert "Background Literature Scout Dispatched" in output
        assert "quantum machine learning 2024" in output
        tasks = cli_studio.task_manager.list_tasks()
        assert len(tasks) > 0
        assert tasks[0].task_type == "scout"


class TestAntigravityDesignSystem:
    """Tests for borderless layout, outlier input framing, slash autocomplete, and interactive help."""

    def test_print_banner_shown_at_most_once(
        self,
        test_console: Console,
        project_manager: ProjectManager,
    ) -> None:
        studio = CLIStudio(console=test_console, project_manager=project_manager)
        assert studio._has_printed_banner is False

        # First call prints logo
        studio.print_banner()
        assert studio._has_printed_banner is True
        output1 = test_console.file.getvalue()
        assert "Local-First Academic Research Studio" in output1

        # Second call does not re-print logo
        studio.print_banner()
        output2 = test_console.file.getvalue()
        assert output2.count("Local-First Academic Research Studio") == 1

    def test_clean_dashboard_banner_no_command_clutter(
        self,
        test_console: Console,
        project_manager: ProjectManager,
    ) -> None:
        studio = CLIStudio(console=test_console, project_manager=project_manager)
        studio.print_dashboard_banner()
        output = test_console.file.getvalue()
        assert "scripcy" in output
        assert "dashboard" in output
        # Ensure no giant table of command suggestions is dumped
        assert "Dashboard Commands" not in output
        assert "/new" not in output
        assert "/delete" not in output

    def test_clean_project_banner_no_tips_clutter(
        self,
        cli_studio: CLIStudio,
    ) -> None:
        cli_studio.console.file.truncate(0)
        cli_studio.console.file.seek(0)
        cli_studio.print_project_banner()
        output = cli_studio.console.file.getvalue()
        assert "scripcy" in output
        assert cli_studio.current_project.name in output
        # Ensure no command tip footer is printed
        assert "Type /help for commands" not in output

    def test_interactive_input_prompt_non_interactive_outliers(
        self,
        test_console: Console,
    ) -> None:
        from scripcy.ui.interactive import interactive_input_prompt

        with patch.object(test_console, "input", return_value="/vault list"):
            res = interactive_input_prompt(
                console=test_console,
                prompt_label="> ",
                is_project_mode=True,
            )

        assert res == "/vault list"
        output = test_console.file.getvalue()
        # No divider line '─' rendered on prompt input
        assert "─" not in output

    def test_interactive_help_view_structure(
        self,
        test_console: Console,
    ) -> None:
        from scripcy.ui.interactive import interactive_help_view

        with patch("sys.stdin.isatty", return_value=True):
            with patch("scripcy.ui.interactive._read_key", side_effect=["ENTER"]):
                interactive_help_view(
                    console=test_console,
                    current_model="gemini/gemini-2.5-pro",
                    current_effort="high",
                    is_project_mode=True,
                )

        output = test_console.file.getvalue()
        assert "Available Commands [Project]" in output
        assert "/search" in output
        assert "/vault" in output
        assert "Scripcy CLI" not in output
        assert "General Workspace" not in output
        assert "Keyboard Shortcuts" not in output

    def test_interactive_help_view_navigation(
        self,
        test_console: Console,
    ) -> None:
        from scripcy.ui.interactive import interactive_help_view

        # Press DOWN then ENTER (exit)
        with patch("sys.stdin.isatty", return_value=True):
            with patch("scripcy.ui.interactive._read_key", side_effect=["DOWN", "ENTER"]):
                interactive_help_view(
                    console=test_console,
                    current_model="gemini/gemini-2.5-pro",
                    current_effort="high",
                    is_project_mode=True,
                )

        output = test_console.file.getvalue()
        assert "Available Commands [Project]" in output
        assert "/search" in output
        assert "/vault" in output

    def test_interactive_slash_autocomplete_and_subcommand_tab(
        self,
        test_console: Console,
    ) -> None:
        from scripcy.ui.interactive import interactive_input_prompt

        # Keystrokes:
        # 1. '/' -> triggers suggestions dropdown
        # 2. 'v' -> filters to /vault
        # 3. 'TAB' -> auto-completes to '/vault ' and drills down to subcommands (/vault list, add, view, archive)
        # 4. 'ENTER' -> confirms /vault list
        keystrokes = ["/", "v", "TAB", "ENTER"]

        with patch("sys.stdin.isatty", return_value=True):
            with patch("scripcy.ui.interactive._read_key", side_effect=keystrokes):
                result = interactive_input_prompt(
                    console=test_console,
                    prompt_label="> ",
                    is_project_mode=True,
                )

        assert result.startswith("/vault")


class TestAgenticUXAndFormatting:
    """Tests for math formatting, action telemetry, input prompt scrollback UX, and skills protocol."""

    def test_beautify_academic_math(self) -> None:
        from scripcy.ui.formatters import beautify_academic_math

        latex_sample = (
            r"Fungsi dot-product: $$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$ "
            r"dengan parameter $\lambda \approx 0.05$ dan loss $\sum_{i=1}^N (y_i - \hat{y}_i)^2$."
        )
        beautified = beautify_academic_math(latex_sample)
        assert "softmax((Q Kᵀ / √(dₖ))) V" in beautified
        assert "λ ≈ 0.05" in beautified
        assert "∑(i=1..N) (yᵢ - ŷᵢ)²" in beautified
        # Ensure raw backslashes and LaTeX tags are not present
        assert r"\mathrm" not in beautified
        assert r"\frac" not in beautified
        assert r"\sqrt" not in beautified

    def test_render_academic_markdown(self, test_console: Console) -> None:
        from scripcy.ui.formatters import render_academic_markdown

        md_sample = (
            "## 1. Analisis Metodologi\n\n"
            "Penelitian ini membuktikan bahwa **transformer** mengungguli *recurrent networks*.\n\n"
            "- Poin pertama: $d_k = 64$\n"
            "- Poin kedua: $\\alpha \\le 0.01$"
        )
        render_academic_markdown(md_sample, console=test_console)
        out = test_console.file.getvalue()
        assert "1. Analisis Metodologi" in out
        assert "transformer" in out
        assert "**transformer**" not in out
        assert "α ≤ 0.01" in out

    def test_skills_agentic_protocol_completeness(self) -> None:
        from scripcy.core.skills import SkillsLoader

        loader = SkillsLoader()
        skills = loader.list_skills()
        skill_names = [s.name for s in skills]
        assert "scripcy_manual" in skill_names

        manual = loader.get_skill("scripcy_manual")
        assert "/search" in manual.system_prompt
        assert "/vault" in manual.system_prompt
        assert "/docx" in manual.system_prompt
        assert "```scripcy" in manual.system_prompt

    def test_interactive_input_prompt_clean_history(self, test_console: Console) -> None:
        from scripcy.ui.interactive import interactive_input_prompt

        with patch("sys.stdin.isatty", return_value=True):
            with patch("scripcy.ui.interactive._read_key", side_effect=["h", "i", "ENTER"]):
                res = interactive_input_prompt(
                    console=test_console,
                    prompt_label="> ",
                    is_project_mode=True,
                )

        assert res == "hi"
        out = test_console.file.getvalue()
        assert "─" in out
        assert "? for shortcuts" in out
        assert "> hi" in out

    def test_handle_chat_telemetry_and_actions(self, cli_studio: CLIStudio) -> None:
        from scripcy.core.engine import EvidenceChunk

        cli_studio.config.api_keys["gemini"] = "dummy-key"
        mock_evidence = [
            EvidenceChunk(
                chunk_id="chk1",
                source="paper1.pdf.md",
                text="Transformer models rely entirely on self-attention mechanisms.",
                score=0.92,
                token_estimate=12,
            )
        ]
        with patch.object(cli_studio.engine.evidence_synthesizer, "retrieve_evidence", return_value=mock_evidence):
            async def mock_stream(*args: Any, **kwargs: Any):
                yield "Berdasarkan literatur, model ini menggunakan rumus $$L = \\frac{1}{2} x^2$$.\n"
                yield "```scripcy\n/docx bab1.docx\n```"

            with patch.object(cli_studio.engine, "stream_query", side_effect=mock_stream):
                with patch("scripcy.cli_studio.interactive_action_confirm", return_value=False):
                    cli_studio.handle_chat("Bagaimana cara kerja self-attention?")

        out = cli_studio.console.file.getvalue()
        assert "Membaca" in out
        assert "paper1.pdf.md" in out
        assert "Berpikir" in out
        assert "Menulis" in out or "drafts/bab1.docx" in out
        assert "Tokens:" in out
        assert "in" in out
        assert "out" in out
        assert "total" in out
        assert "L = (1 / 2) x²" in out

    def test_interactive_action_confirm_in_place_no_clear(self, test_console: Console) -> None:
        from scripcy.ui.interactive import interactive_action_confirm

        with patch("sys.stdin.isatty", return_value=True):
            with patch("scripcy.ui.interactive._read_key", side_effect=["ENTER"]):
                with patch.object(test_console, "clear") as mock_clear:
                    confirmed = interactive_action_confirm("/search test", console=test_console)

        assert confirmed[0] == "execute" or confirmed is True
        # Ensure c.clear() was NEVER called (prevents screen jumps and lost history)
        mock_clear.assert_not_called()

    def test_interactive_confirm_yes_and_no(self, test_console: Console) -> None:
        from scripcy.ui.interactive import interactive_confirm

        with patch("sys.stdin.isatty", return_value=True):
            # Test default Enter on Yes
            with patch("scripcy.ui.interactive._read_key", side_effect=["ENTER"]):
                assert interactive_confirm("Hapus project?", default=True, console=test_console) is True

            # Test direct 'y' key
            with patch("scripcy.ui.interactive._read_key", side_effect=["y"]):
                assert interactive_confirm("Hapus project?", default=False, console=test_console) is True

            # Test direct 'n' key
            with patch("scripcy.ui.interactive._read_key", side_effect=["n"]):
                assert interactive_confirm("Hapus project?", default=True, console=test_console) is False

            # Test ESC key
            with patch("scripcy.ui.interactive._read_key", side_effect=["ESC"]):
                assert interactive_confirm("Hapus project?", default=True, console=test_console) is False

    def test_interactive_project_browser_delete_deregisters_project(self, cli_studio: CLIStudio) -> None:
        proj_a = cli_studio.project_manager.create_project(name="to_delete_proj")
        assert cli_studio.project_manager.get_project("to_delete_proj") is not None

        # Simulate browser selecting 'delete' on proj_a, then 'back'
        browser_calls = [
            ("delete", 0, proj_a),
            ("back", 0, None),
        ]
        with patch("scripcy.cli_studio.interactive_item_browser", side_effect=browser_calls):
            with patch("scripcy.cli_studio.interactive_confirm", return_value=True):
                cli_studio.run_interactive_dashboard_projects()

        # Project should now be completely removed from registry
        assert cli_studio.project_manager.get_project("to_delete_proj") is None

    def test_cmd_search_zero_results_triggers_ai_recovery(self, cli_studio: CLIStudio) -> None:
        async def mock_empty_search(*args: Any, **kwargs: Any):
            return []

        with patch.object(cli_studio, "_execute_search", mock_empty_search):
            with patch.object(cli_studio, "handle_chat") as mock_chat:
                cli_studio.cmd_search('"nonexistent quantum algorithm 1900"')

        output = cli_studio.console.file.getvalue()
        assert "No academic results found" in output
        assert "Menganalisis kendala" in output
        mock_chat.assert_called_once()
        call_arg = mock_chat.call_args[0][0]
        assert "nonexistent quantum algorithm 1900" in call_arg
        assert "kata kunci pencarian alternatif" in call_arg

    def test_chat_submode_prompt_label_and_slash_catalog(
        self,
        cli_studio: CLIStudio,
    ) -> None:
        from scripcy.ui.interactive import SLASH_COMMANDS_CHAT

        cli_studio.sub_mode = "chat"
        session = cli_studio._ensure_active_chat_session()
        assert session is not None
        session.name = "chat4"

        sess_name = session.name if session else "chat1"
        prompt_lbl = f"{cli_studio.current_project.name}/{sess_name}> " if cli_studio.current_project else f"{sess_name}> "
        assert prompt_lbl == f"{cli_studio.current_project.name}/chat4> "
        assert prompt_lbl == "test_proj/chat4> "

        # Verify SLASH_COMMANDS_CHAT contains /back, /new, /resume, /history, /clear, /vault, etc.
        cmd_names = [cmd for cmd, _ in SLASH_COMMANDS_CHAT]
        assert "/back" in cmd_names
        assert "/new" in cmd_names
        assert "/resume" in cmd_names
        assert "/history" in cmd_names
        assert "/clear" in cmd_names
        assert "/export" in cmd_names

    def test_interactive_help_view_in_place_no_clear(self, test_console: Console) -> None:
        """Verify interactive_help_view redraws in place without c.clear() to prevent duplicate blocks."""
        from scripcy.ui.interactive import interactive_help_view

        with patch("sys.stdin.isatty", return_value=True):
            with patch("scripcy.ui.interactive._read_key", side_effect=["DOWN", "ESC"]):
                with patch.object(test_console, "clear") as mock_clear:
                    interactive_help_view(console=test_console)

        # c.clear() must NEVER be called, preventing line duplication
        mock_clear.assert_not_called()

    def test_interactive_token_warning_confirm_options(self, test_console: Console) -> None:
        """Verify token warning confirmation returns proceed, limit, and cancel."""
        from scripcy.ui.interactive import interactive_token_warning_confirm

        with patch("sys.stdin.isatty", return_value=True):
            # 1. ENTER on first option -> 'proceed'
            with patch("scripcy.ui.interactive._read_key", side_effect=["ENTER"]):
                res1 = interactive_token_warning_confirm(8, 4, 6000, console=test_console)
                assert res1 == "proceed"

            # 2. DOWN then ENTER -> 'limit'
            with patch("scripcy.ui.interactive._read_key", side_effect=["DOWN", "ENTER"]):
                res2 = interactive_token_warning_confirm(8, 4, 6000, console=test_console)
                assert res2 == "limit"

            # 3. ESC -> 'cancel'
            with patch("scripcy.ui.interactive._read_key", side_effect=["ESC"]):
                res3 = interactive_token_warning_confirm(8, 4, 6000, console=test_console)
                assert res3 == "cancel"

    def test_handle_chat_token_warning_cancel_aborts_llm(self, cli_studio: CLIStudio) -> None:
        """Verify if token warning is cancelled, LLM query is aborted and no generation occurs."""
        from scripcy.core.engine import EvidenceChunk

        cli_studio.config.api_keys["gemini"] = "dummy-key"
        # 8 chunks (> 5 chunk threshold)
        mock_evidence = [
            EvidenceChunk(chunk_id=f"c{i}", source=f"paper{i}", text=f"Evidence text {i}", score=1.0)
            for i in range(8)
        ]
        with patch.object(cli_studio.engine.evidence_synthesizer, "retrieve_evidence", return_value=mock_evidence):
            with patch("scripcy.cli_studio.interactive_token_warning_confirm", return_value="cancel"):
                with patch.object(cli_studio.engine, "stream_query") as mock_stream:
                    cli_studio.handle_chat("rangkum semua paper di vault")
                    mock_stream.assert_not_called()

        out = cli_studio.console.file.getvalue()
        assert "Pertanyaan dibatalkan oleh pengguna (hemat token)" in out

    def test_handle_chat_keyboard_interrupt_cancels_cleanly(self, cli_studio: CLIStudio) -> None:
        """Verify user can cancel a question with Ctrl+C without crashing or exiting chat mode."""
        from scripcy.core.engine import EvidenceChunk

        cli_studio.config.api_keys["gemini"] = "dummy-key"
        cli_studio.sub_mode = "chat"
        mock_evidence = [EvidenceChunk(chunk_id="c0", source="paper1", text="Sample", score=1.0)]

        with patch.object(cli_studio.engine.evidence_synthesizer, "retrieve_evidence", return_value=mock_evidence):
            async def mock_interrupt(*args: Any, **kwargs: Any):
                raise KeyboardInterrupt()
                yield "never"

            with patch.object(cli_studio.engine, "stream_query", side_effect=mock_interrupt):
                # Should NOT raise KeyboardInterrupt to caller
                cli_studio.handle_chat("jelaskan teori ini")

        assert cli_studio.sub_mode == "chat"
        out = cli_studio.console.file.getvalue()
        assert "Pertanyaan dibatalkan oleh pengguna (Ctrl+C)" in out

    def test_handle_chat_detach_signal_hands_off_to_background(self, cli_studio: CLIStudio) -> None:
        """Verify typing /detach while agent is streaming hands off task to background TaskManager."""
        from scripcy.core.engine import EvidenceChunk
        from scripcy.cli_studio import SessionDetachSignal

        cli_studio.config.api_keys["gemini"] = "dummy-key"
        cli_studio.sub_mode = "chat"
        session = cli_studio._ensure_active_chat_session()
        assert session is not None

        mock_evidence = [EvidenceChunk(chunk_id="c0", source="paper1", text="Sample", score=1.0)]

        with patch.object(cli_studio.engine.evidence_synthesizer, "retrieve_evidence", return_value=mock_evidence):
            with patch.object(cli_studio, "_run_agent_turn_interactive", side_effect=SessionDetachSignal("detach")):
                import pytest
                with pytest.raises(SessionDetachSignal):
                    cli_studio.handle_chat("analisa setiap journal yang ada di vault")

        # Session must be marked detached
        updated_s = cli_studio.chat_manager.get_session(session.id)
        assert updated_s is not None
        assert updated_s.status == "detached"

        # Background task must be submitted to TaskManager
        tasks = cli_studio.task_manager.list_tasks_for_session(session.id)
        assert len(tasks) > 0
        assert tasks[0].task_type == "agent"

        out = cli_studio.console.file.getvalue()
        assert "di-detach ke latar belakang" in out

    def test_handle_chat_close_signal_terminates_session(self, cli_studio: CLIStudio) -> None:
        """Verify typing /close terminates session and cancels background task."""
        from scripcy.core.engine import EvidenceChunk
        from scripcy.cli_studio import SessionCloseSignal

        cli_studio.config.api_keys["gemini"] = "dummy-key"
        cli_studio.sub_mode = "chat"
        session = cli_studio._ensure_active_chat_session()
        assert session is not None

        mock_evidence = [EvidenceChunk(chunk_id="c0", source="paper1", text="Sample", score=1.0)]

        with patch.object(cli_studio.engine.evidence_synthesizer, "retrieve_evidence", return_value=mock_evidence):
            with patch.object(cli_studio, "_run_agent_turn_interactive", side_effect=SessionCloseSignal("close")):
                import pytest
                with pytest.raises(SessionCloseSignal):
                    cli_studio.handle_chat("analisa setiap journal yang ada di vault")

        updated_s = cli_studio.chat_manager.get_session(session.id)
        assert updated_s is not None
        assert updated_s.status == "closed"

        out = cli_studio.console.file.getvalue()
        assert "ditutup (terminated)" in out

    def test_interactive_input_prompt_esc_returns_detach_in_chat(self) -> None:
        """Verify ESC on empty input returns /detach in chat mode."""
        from scripcy.ui.interactive import interactive_input_prompt

        with patch("sys.stdin.isatty", return_value=True):
            with patch("scripcy.ui.interactive._read_key", side_effect=["ESC"]):
                res = interactive_input_prompt(
                    mode="chat",
                    is_chat_mode=True,
                )
                assert res == "/detach"

    def test_interactive_input_prompt_ctrl_c_raises_keyboard_interrupt(self) -> None:
        """Verify Ctrl+C raises KeyboardInterrupt."""
        from scripcy.ui.interactive import interactive_input_prompt
        import pytest

        with patch("sys.stdin.isatty", return_value=True):
            with patch("scripcy.ui.interactive._read_key", side_effect=["\x03"]):
                with pytest.raises(KeyboardInterrupt):
                    interactive_input_prompt(
                        mode="chat",
                        is_chat_mode=True,
                    )

    def test_interactive_detail_viewer_live_streaming(self) -> None:
        """Verify interactive_detail_viewer pulls updated lines from lines_provider and responds to keys."""
        from scripcy.ui.interactive import interactive_detail_viewer

        log_state = ["Line 1", "Line 2"]
        def provider():
            return list(log_state)

        with patch("sys.stdin.isatty", return_value=True):
            # First poll returns None (streaming tick), second poll returns ESC
            with patch("scripcy.ui.interactive._poll_key", side_effect=[None, "ESC"]):
                # Update log_state to simulate background task producing new lines
                log_state.append("Line 3 (newly streamed)")
                choice = interactive_detail_viewer(
                    title="Test Task Log",
                    lines_provider=provider,
                    is_live=lambda: True,
                )
                assert choice == "back"


    def test_background_task_completion_records_note_without_redundant_ai_eval(self, cli_studio: CLIStudio) -> None:
        from scripcy.core.tasks import TaskInfo
        sess = cli_studio.chat_manager.create_session("Riset Karhutla")
        t1 = TaskInfo(id="task-901", name="Scout Karhutla Ekonomi", task_type="scout", session_id=sess.id, status="completed", result_summary="Added 3 papers")
        cli_studio.task_manager._tasks[t1.id] = t1

        eval_calls: list[str] = []
        with patch.object(cli_studio, "_run_task_ai_evaluation", side_effect=lambda task, s_id: eval_calls.append(task.id)):
            cli_studio._on_background_task_completed(t1)
            # Must NOT trigger redundant AI evaluation
            assert len(eval_calls) == 0

        # System note must be recorded in session history
        msgs = cli_studio.chat_manager.get_messages(sess.id)
        assert any("PROSES LATAR BELAKANG SELESAI" in m.content for m in msgs)

    def test_interactive_input_prompt_on_idle_clears_and_rerenders(self, test_console: Console) -> None:
        from scripcy.ui.interactive import interactive_input_prompt

        idle_called = [0]
        def idle_handler(clear_prompt_fn):
            idle_called[0] += 1
            if idle_called[0] == 1:
                clear_prompt_fn()
                test_console.print("Async Event Received")
                return True
            return False

        keys = [None, "h", "i", "ENTER"]
        key_idx = [0]
        def fake_read_key(timeout=None):
            k = keys[key_idx[0]]
            key_idx[0] += 1
            return k

        with patch("sys.stdin.isatty", return_value=True):
            with patch("scripcy.ui.interactive._read_key", side_effect=fake_read_key):
                res = interactive_input_prompt(
                    console=test_console,
                    prompt_label="> ",
                    on_idle=idle_handler,
                )

        assert res == "hi"
        assert idle_called[0] >= 1

    def test_background_ui_events_enqueued_and_flushed_on_main_thread(self, cli_studio: CLIStudio) -> None:
        from scripcy.core.tasks import TaskInfo
        sess = cli_studio.chat_manager.create_session("Riset ML")
        cli_studio.sub_mode = "chat"
        cli_studio.active_chat_session = sess

        # Add assistant response to session history as background worker does
        cli_studio.chat_manager.add_message(sess.id, "assistant", "Bumi adalah planet ketiga dari Matahari.")

        chat_task = TaskInfo(
            id="task-999",
            name="Chat: jelaskan konsep dasar ML",
            task_type="chat",
            session_id=sess.id,
            status="completed",
            result_summary="Penjelasan selesai",
        )

        # Completing the task must enqueue a UI event rather than printing directly to stdout
        cli_studio._on_background_task_completed(chat_task)
        assert len(cli_studio._pending_ui_events) >= 1

        # Flushing executes the queued function cleanly on the main thread and renders AI answer
        cleared = [False]
        with cli_studio.console.capture() as capture:
            did_flush = cli_studio._flush_pending_ui_events(clear_prompt_fn=lambda: cleared.__setitem__(0, True))
        assert did_flush is True
        assert cleared[0] is True
        assert len(cli_studio._pending_ui_events) == 0
        captured_text = capture.get()
        assert "Bumi adalah planet ketiga dari Matahari." in captured_text
        assert "Respons AI" in captured_text

    def test_interactive_input_prompt_renders_status_line(self, test_console: Console) -> None:
        from scripcy.ui.interactive import interactive_input_prompt

        keys = ["/", "q", "u", "i", "t", "ENTER"]
        with patch("sys.stdin.isatty", return_value=True):
            with patch("scripcy.ui.interactive._read_key", side_effect=keys):
                res = interactive_input_prompt(
                    console=test_console,
                    prompt_label="> ",
                    status_line_callback=lambda: "[bold cyan]⠋[/bold cyan] [bold bright_white]Sedang download paper...[/bold bright_white]",
                )
        assert res == "/quit"

    def test_chat_session_restores_running_task_indicator_on_reentry(self, cli_studio: CLIStudio) -> None:
        from scripcy.core.tasks import TaskInfo
        sess = cli_studio.chat_manager.create_session("Riset Karhutla")
        cli_studio.sub_mode = "chat"
        cli_studio.active_chat_session = sess

        # Simulate background task still running for this session
        running_task = TaskInfo(
            id="task-777",
            name="Chat (Riset Karhutla): jelaskan hotspot satelit",
            task_type="agent",
            session_id=sess.id,
            status="running",
            progress="Menghubungi AI...",
        )
        cli_studio.task_manager._tasks[running_task.id] = running_task

        # Verify active tasks are detected
        active = [t for t in cli_studio.task_manager.list_tasks_for_session(sess.id) if t.status == "running"]
        assert len(active) == 1
        assert active[0].id == "task-777"

    def test_render_processing_view_displays_clean_session_and_tokens(self, cli_studio: CLIStudio) -> None:
        from scripcy.core.tasks import TaskInfo
        sess = cli_studio.chat_manager.create_session("cari_paper")
        task = TaskInfo(
            id="task-99",
            name="Chat: sintesis",
            task_type="chat",
            session_id=sess.id,
            status="completed",
            tokens_burned=2450,
            progress="Selesai",
        )
        cli_studio.task_manager._tasks[task.id] = task

        cli_studio.render_processing_view()
        rendered = cli_studio.console.file.getvalue()
        # Origin session must show direct session name, not "session: cari_paper"
        assert "cari_paper" in rendered
        assert "session: cari_paper" not in rendered
        assert "session : cari_paper" not in rendered
        # Tokens must be displayed in the list
        assert "2,450 tok" in rendered

    def test_get_sanitized_task_logs_filters_raw_chat_dumps(self, cli_studio: CLIStudio) -> None:
        from scripcy.core.tasks import TaskInfo
        raw_logs = [
            "[21:58:12] Started task: 'Chat: uji mutasi'",
            "[21:58:12] Pertanyaan: uji mutasi",
            "[21:58:13] Model: gemini-2.5-flash",
            "[21:58:24] Token: 382 in · 1,009 out · 1,391 total (gemini-2.5-flash) • 12.1s",
            "[21:58:24] Selesai! 4323 karakter disimpan ke riwayat sesi.",
            "Respons AI:\n### Analisis Mutasi EGFR\nBerdasarkan data vault...",
        ]
        task = TaskInfo(
            id="task-101",
            name="Chat: uji mutasi",
            task_type="chat",
            status="completed",
            logs=raw_logs,
            result_summary="Sintesis selesai (4323 karakter, 1,391 total token)",
        )
        sanitized = cli_studio.get_sanitized_task_logs(task)
        # Should keep system execution telemetry
        assert any("Token: 382 in" in line for line in sanitized)
        assert any("Selesai! 4323 karakter" in line for line in sanitized)
        assert any("Summary: Sintesis selesai" in line for line in sanitized)
        # Should NOT dump the raw chat dialogue or markdown header
        assert not any("### Analisis Mutasi EGFR" in line for line in sanitized)
        assert not any("Respons AI:" in line for line in sanitized) 
