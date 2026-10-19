"""Comprehensive unit tests for the Arch Linux interactive CLI research studio."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from scrpcy.cli_studio import ARCH_LOGO, CLIStudio
from scrpcy.core.config import ConfigManager, ScrpcyConfig
from scrpcy.core.models import DocumentMetadata
from scrpcy.core.search import SearchResult
from scrpcy.core.workspace import WorkspaceManager


@pytest.fixture
def test_console() -> Console:
    """Fixture providing an in-memory capturing console."""
    return Console(file=io.StringIO(), force_terminal=True, color_system=None)


@pytest.fixture
def cli_studio(temp_workspace: WorkspaceManager, test_console: Console) -> CLIStudio:
    """Fixture providing an isolated CLIStudio instance."""
    cfg = ScrpcyConfig(
        default_model="gemini/gemini-2.5-pro",
        workspace_path=str(temp_workspace.root_dir),
    )
    studio = CLIStudio(
        workspace=temp_workspace,
        config=cfg,
        console=test_console,
    )
    return studio


class TestArchLinuxBanner:
    """Tests for Arch Linux ASCII logo and fastfetch banner formatting."""

    def test_arch_logo_structure(self) -> None:
        assert "S C R P C Y" in ARCH_LOGO
        assert "/\\" in ARCH_LOGO

    def test_banner_rendering(self, cli_studio: CLIStudio) -> None:
        banner = cli_studio.get_banner()
        assert banner is not None
        cli_studio.print_banner()

        output = cli_studio.console.file.getvalue()
        assert "scrpcy@studio" in output
        assert "Arch Linux CLI Studio" in output
        assert "Vault" in output
        assert "Engine Model" in output


class TestCLICommands:
    """Tests for interactive CLI slash commands."""

    def test_cmd_vault_empty(self, cli_studio: CLIStudio) -> None:
        cli_studio.cmd_vault()
        output = cli_studio.console.file.getvalue()
        assert "Vault is currently empty" in output

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

        # Toggle clear
        cli_studio.cmd_active("clear")
        assert len(cli_studio.active_documents) == 0

        # Toggle by stem
        cli_studio.cmd_active("doc1")
        assert "doc1" in cli_studio.active_documents

        # Toggle off
        cli_studio.cmd_active("doc1")
        assert "doc1" not in cli_studio.active_documents

        # Toggle by index
        cli_studio.cmd_active("1")
        assert len(cli_studio.active_documents) == 1

        # Toggle all
        cli_studio.cmd_active("all")
        assert len(cli_studio.active_documents) == 2

    def test_cmd_skills_and_switch(self, cli_studio: CLIStudio) -> None:
        cli_studio.cmd_skills()
        output = cli_studio.console.file.getvalue()
        assert "academic_writer" in output

        # Switch skill
        cli_studio.cmd_skill("literature_synthesizer")
        assert cli_studio.active_skill_name == "literature_synthesizer"

        # Switch to invalid skill
        cli_studio.cmd_skill("non_existent_skill_xyz")
        assert cli_studio.active_skill_name == "literature_synthesizer"

    def test_cmd_model_and_style(self, cli_studio: CLIStudio) -> None:
        cli_studio.cmd_model("openai/gpt-4o")
        assert cli_studio.config.default_model == "openai/gpt-4o"

        cli_studio.cmd_style("ieee")
        assert cli_studio.config.citation.style == "ieee"

        cli_studio.cmd_style("invalid_style")
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

        docx_path = temp_workspace.vault_dir / out_docx
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

    def test_handle_command_routing(self, cli_studio: CLIStudio) -> None:
        cli_studio.handle_command("/help")
        cli_studio.handle_command("/status")
        cli_studio.handle_command("/unknown_command_abc")
        output = cli_studio.console.file.getvalue()
        assert "scrpcy CLI Studio Commands" in output
        assert "Unknown command" in output
