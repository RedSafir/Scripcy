"""Comprehensive unit tests for the Dashboard and Project CLI research studio."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from scrpcy.cli_studio import CLIStudio, SCRPCY_LOGO, SCRPCY_VERSION
from scrpcy.core.config import ConfigManager, ScrpcyConfig
from scrpcy.core.models import DocumentMetadata
from scrpcy.core.project import ProjectInfo, ProjectManager
from scrpcy.core.search import SearchResult
from scrpcy.core.workspace import WorkspaceManager


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
    cfg = ScrpcyConfig(
        default_model="gemini/gemini-2.5-pro",
        workspace_path=str(temp_workspace.root_dir),
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
        assert "_____" in SCRPCY_LOGO
        assert SCRPCY_VERSION.startswith("v")

    def test_banner_rendering(self, test_console: Console, project_manager: ProjectManager) -> None:
        studio = CLIStudio(console=test_console, project_manager=project_manager)
        studio.print_banner()

        output = test_console.file.getvalue()
        assert SCRPCY_VERSION in output
        assert "Academic Research Studio" in output


class TestDashboardFlow:
    """Tests for initial Dashboard mode, project creation, and system status."""

    def test_dashboard_help(self, test_console: Console, project_manager: ProjectManager) -> None:
        studio = CLIStudio(console=test_console, project_manager=project_manager)
        studio.print_dashboard_help()
        output = test_console.file.getvalue()
        assert "Dashboard Commands" in output
        assert "projects" in output
        assert "new" in output
        assert "open" in output

    def test_dashboard_status(self, test_console: Console, project_manager: ProjectManager, tmp_path: Path) -> None:
        cfg = ScrpcyConfig(default_model="gemini/gemini-2.5-pro")
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

    def test_cmd_skills_and_switch(self, cli_studio: CLIStudio) -> None:
        cli_studio.cmd_skills()
        output = cli_studio.console.file.getvalue()
        assert "Available Research Personas / Skills" in output
        assert "academic_writer" in output
        assert "literature_synthesizer" in output
        assert "[Active]" in output

        # Switch by name
        cli_studio.cmd_skill("literature_synthesizer")
        assert cli_studio.active_skill_name == "literature_synthesizer"

    def test_cmd_skill_without_args_renders_catalog(self, cli_studio: CLIStudio) -> None:
        cli_studio.cmd_skill("")
        output = cli_studio.console.file.getvalue()
        assert "Available Research Personas / Skills" in output
        assert "Add custom persona:" in output

    def test_cmd_skill_switch_by_numeric_index(self, cli_studio: CLIStudio) -> None:
        skills = cli_studio.skills_loader.list_skills()
        assert len(skills) >= 2

        # Switch to skill 2 by number
        target_skill_2 = skills[1].name
        cli_studio.cmd_skill("2")
        assert cli_studio.active_skill_name == target_skill_2

        # Switch back to skill 1 by number
        target_skill_1 = skills[0].name
        cli_studio.cmd_skill("1")
        assert cli_studio.active_skill_name == target_skill_1

    def test_cmd_skill_invalid_index(self, cli_studio: CLIStudio) -> None:
        cli_studio.cmd_skill("99")
        output = cli_studio.console.file.getvalue()
        assert "Invalid skill number: 99" in output

    def test_dashboard_skills_catalog(self, test_console: Console, project_manager: ProjectManager) -> None:
        studio = CLIStudio(console=test_console, project_manager=project_manager)
        studio.handle_dashboard_command("skills")
        output = test_console.file.getvalue()
        assert "Installed AI Research Personas / Skills" in output
        assert "[Installed]" in output
        assert "academic_writer" in output

    def test_project_slash_skill_routing(self, cli_studio: CLIStudio) -> None:
        # Route /skills
        cli_studio.handle_project_command("/skills")
        output = cli_studio.console.file.getvalue()
        assert "Available Research Personas / Skills" in output

        # Route /skill 2
        skills = cli_studio.skills_loader.list_skills()
        cli_studio.handle_project_command("/skill 2")
        assert cli_studio.active_skill_name == skills[1].name

    def test_cmd_model_and_style(self, cli_studio: CLIStudio) -> None:
        cli_studio.cmd_model("openai/gpt-4o")
        assert cli_studio.config.default_model == "openai/gpt-4o"

        cli_studio.cmd_style("ieee")
        assert cli_studio.config.citation.style == "ieee"

    def test_cmd_config(self, cli_studio: CLIStudio) -> None:
        cli_studio.cmd_config("show")
        output = cli_studio.console.file.getvalue()
        assert "scrpcy Configuration" in output

        cli_studio.cmd_config("set-key gemini test-key-secret-999")
        assert cli_studio.config.api_keys["gemini"] == "test-key-secret-999"

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
        assert (tmp_path / "paper1" / ".scrpcy" / "project.json").is_file()

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
        assert "removed from Scrpcy" in output
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
        assert "removed from Scrpcy" in output
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
        assert "removed from Scrpcy" in output
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
