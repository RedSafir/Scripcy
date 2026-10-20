"""Unit and integration tests for TokenTracker, 9Router integration, and models CLI."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from rich.console import Console
from typer.testing import CliRunner

from scrpcy.cli import app as cli_app
from scrpcy.cli_studio import CLIStudio
from scrpcy.core.config import ScrpcyConfig
from scrpcy.core.engine import LLMEngine
from scrpcy.core.project import ProjectManager
from scrpcy.core.tracker import TokenRecord, TokenTracker
from scrpcy.core.workspace import WorkspaceManager


class TestTokenTracker:
    """Tests for token usage logging to global and per-project logs with timestamps."""

    def test_record_usage_global_and_project(self, tmp_path: Path) -> None:
        global_log = tmp_path / "global_tokens.jsonl"
        proj_dir = tmp_path / "my_research_project"
        proj_dir.mkdir(parents=True, exist_ok=True)

        tracker = TokenTracker(global_log_path=global_log)
        rec = tracker.record_usage(
            model="9router/gpt-4o",
            input_tokens=150,
            output_tokens=300,
            project_name="my_research_project",
            project_path=proj_dir,
            skill="academic_writer",
        )

        assert rec.input_tokens == 150
        assert rec.output_tokens == 300
        assert rec.total_tokens == 450
        assert rec.model == "9router/gpt-4o"
        assert "T" in rec.timestamp  # ISO 8601 format

        # Check global log file
        assert global_log.is_file()
        global_lines = global_log.read_text(encoding="utf-8").strip().splitlines()
        assert len(global_lines) == 1
        data = json.loads(global_lines[0])
        assert data["total_tokens"] == 450
        assert data["project_name"] == "my_research_project"

        # Check per-project log file
        proj_log = proj_dir / ".scrpcy" / "token_usage.jsonl"
        assert proj_log.is_file()
        proj_lines = proj_log.read_text(encoding="utf-8").strip().splitlines()
        assert len(proj_lines) == 1
        proj_data = json.loads(proj_lines[0])
        assert proj_data["total_tokens"] == 450

    def test_get_summary_aggregates_properly(self, tmp_path: Path) -> None:
        global_log = tmp_path / "global_tokens.jsonl"
        tracker = TokenTracker(global_log_path=global_log)

        tracker.record_usage("gemini/gemini-2.5-pro", 100, 200, project_name="p1")
        tracker.record_usage("openai/gpt-4o", 50, 50, project_name="p1")
        tracker.record_usage("9router/claude-3-5-sonnet", 200, 400, project_name="p2")

        summary = tracker.get_summary()
        assert summary.total_input_tokens == 350
        assert summary.total_output_tokens == 650
        assert summary.total_tokens == 1000
        assert summary.total_requests == 3

        assert "p1" in summary.projects
        assert summary.projects["p1"].total_tokens == 400
        assert summary.projects["p1"].request_count == 2

        assert "p2" in summary.projects
        assert summary.projects["p2"].total_tokens == 600
        assert summary.projects["p2"].request_count == 1

        # Check model aggregation
        assert summary.models["gemini/gemini-2.5-pro"] == 300
        assert summary.models["openai/gpt-4o"] == 100
        assert summary.models["9router/claude-3-5-sonnet"] == 600

    def test_get_project_tokens(self, tmp_path: Path) -> None:
        proj_dir = tmp_path / "thesis"
        proj_dir.mkdir(parents=True, exist_ok=True)
        tracker = TokenTracker(global_log_path=tmp_path / "global.jsonl")

        tracker.record_usage("openai/gpt-4o", 120, 80, project_name="thesis", project_path=proj_dir)
        tracker.record_usage("openai/gpt-4o", 200, 100, project_name="thesis", project_path=proj_dir)

        p_summary = TokenTracker.get_project_tokens(proj_dir)
        assert p_summary.input_tokens == 320
        assert p_summary.output_tokens == 180
        assert p_summary.total_tokens == 500
        assert p_summary.request_count == 2


class Test9RouterConfiguration:
    """Tests for 9Router configuration, base URL resolution, and setup helper."""

    def test_setup_9router_defaults(self) -> None:
        cfg = ScrpcyConfig()
        cfg.setup_9router(port=20128, host="localhost", api_key="sk-9r-test", default_model="9router/gpt-4o")

        assert cfg.default_model == "9router/gpt-4o"
        assert cfg.api_bases["9router"] == "http://localhost:20128/v1"
        assert cfg.get_api_key("9router") == "sk-9r-test"
        assert cfg.get_api_key_for_model("9router/gpt-4o") == "sk-9r-test"
        assert cfg.get_api_base_for_model("9router/gpt-4o") == "http://localhost:20128/v1"

    def test_get_api_base_for_openai_and_custom(self) -> None:
        cfg = ScrpcyConfig()
        cfg.set_api_base("openai", "http://my-proxy:8080/v1")
        assert cfg.get_api_base_for_model("openai/gpt-4o") == "http://my-proxy:8080/v1"

        cfg.set_api_base("groq", "https://api.groq.com/openai/v1")
        assert cfg.get_api_base_for_model("groq/llama-3") == "https://api.groq.com/openai/v1"


class TestEngine9RouterAndTokens:
    """Tests LLMEngine token logging and 9Router request translation."""

    @pytest.mark.asyncio
    async def test_llm_engine_translates_9router_and_tracks_tokens(self, tmp_path: Path) -> None:
        global_log = tmp_path / "engine_tokens.jsonl"
        proj_dir = tmp_path / "proj_alpha"
        proj_dir.mkdir(parents=True, exist_ok=True)
        tracker = TokenTracker(global_log_path=global_log)

        cfg = ScrpcyConfig()
        cfg.setup_9router(port=20128, api_key="sk-test-key", default_model="9router/gpt-4o")

        invoked_kwargs: dict = {}

        class DummyStream:
            def __init__(self, text: str):
                self.text = text

            def __aiter__(self):
                self._done = False
                return self

            async def __anext__(self):
                if self._done:
                    raise StopAsyncIteration
                self._done = True

                class ChunkChoice:
                    class Delta:
                        content = self.text
                    delta = Delta()

                class Chunk:
                    choices = [ChunkChoice()]

                return Chunk()

        async def mock_acompletion(**kwargs):
            invoked_kwargs.update(kwargs)
            return DummyStream("9Router synthesized literature review.")

        engine = LLMEngine(
            config=cfg,
            acompletion_fn=mock_acompletion,
            token_tracker=tracker,
            project_name="proj_alpha",
            project_path=proj_dir,
        )

        res = await engine.query("Synthesize findings on self-attention mechanisms.")
        assert "9Router synthesized" in res.answer

        # Verify LiteLLM model translation and base URL
        assert invoked_kwargs["model"] == "openai/gpt-4o"
        assert invoked_kwargs["api_base"] == "http://localhost:20128/v1"
        assert invoked_kwargs["api_key"] == "sk-test-key"

        # Verify token consumption was logged
        summary = tracker.get_summary()
        assert summary.total_requests == 1
        assert summary.total_input_tokens > 0
        assert summary.total_output_tokens > 0
        assert summary.total_tokens == summary.total_input_tokens + summary.total_output_tokens
        assert "proj_alpha" in summary.projects


class TestCLIModelsAndStatus:
    """CLI testing for models management, 9Router, and per-project status."""

    def test_cli_models_tokens_display(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        global_log = tmp_path / "global_tokens.jsonl"
        tracker = TokenTracker(global_log_path=global_log)
        tracker.record_usage("openai/gpt-4o", 100, 200, project_name="Dissertation")

        monkeypatch.setattr("scrpcy.cli.TokenTracker", lambda: tracker)

        runner = CliRunner()
        res = runner.invoke(cli_app, ["models", "tokens"])
        assert res.exit_code == 0
        assert "Global Token Consumption Summary" in res.output
        assert "Total Input Tokens" in res.output
        assert "Total Output Tokens" in res.output
        assert "Dissertation" in res.output

    def test_cli_status_per_project(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pm = ProjectManager(registry_file=tmp_path / "projects.json")
        proj_dir = tmp_path / "SkripsiAI"
        proj_dir.mkdir(parents=True, exist_ok=True)
        pm.create_project("SkripsiAI", proj_dir)

        monkeypatch.setattr("scrpcy.cli.ProjectManager", lambda: pm)

        runner = CliRunner()
        res = runner.invoke(cli_app, ["status", "SkripsiAI"])
        assert res.exit_code == 0
        assert "Research Project Status: skripsiai" in res.output
        assert "Project Token Usage" in res.output

    def test_cli_studio_models_and_status(self, tmp_path: Path) -> None:
        import io
        test_console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
        project_manager = ProjectManager(registry_file=tmp_path / "studio_projects.json")

        proj_dir = tmp_path / "nlp_study"
        proj_dir.mkdir(parents=True, exist_ok=True)
        project_manager.create_project("nlp_study", proj_dir)

        studio = CLIStudio(console=test_console, project_manager=project_manager)

        # Test status at dashboard prompts to open project
        studio.handle_dashboard_command("status")
        output = test_console.file.getvalue()
        assert "Perintah 'status' digunakan di dalam project" in output

        # Test project status inside active project
        proj = project_manager.get_project("nlp_study")
        studio.open_project(proj)
        studio.cmd_project_status()
        output2 = test_console.file.getvalue()
        assert "Status Project: nlp_study" in output2
        assert "Project Model" in output2
        assert "Project Token Usage" in output2

        # Test models command at dashboard
        studio.handle_dashboard_command("models use 9router/claude-3-5-sonnet")
        assert studio.project_manager.get_project_model("nlp_study") == "9router/claude-3-5-sonnet"

        # Test models 9router setup
        studio.handle_dashboard_command("models 9router 20128")
        assert studio.config.default_model.startswith("9router/")
        assert studio.config.api_bases.get("9router") == "http://localhost:20128/v1"

    def test_interactive_model_selection_updates_project_and_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import io
        test_console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
        project_manager = ProjectManager(registry_file=tmp_path / "studio_projects.json")

        proj_dir = tmp_path / "thesis"
        proj_dir.mkdir(parents=True, exist_ok=True)
        project_manager.create_project("thesis", proj_dir)

        studio = CLIStudio(console=test_console, project_manager=project_manager)
        proj = project_manager.get_project("thesis")
        studio.open_project(proj)

        # Initial model should be default or claude
        studio.project_manager.set_project_model("thesis", "9router/claude-3-5-sonnet")
        studio.current_project.model = "9router/claude-3-5-sonnet"

        # Mock isatty and interactive_model_selector
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr(
            "scrpcy.cli_studio.interactive_model_selector",
            lambda **kwargs: ("9router/gemini-2.5-flash", "high"),
        )

        studio.cmd_models("")

        # Now check status output
        test_console.file.seek(0)
        test_console.file.truncate(0)
        studio.cmd_project_status()
        status_output = test_console.file.getvalue()

        assert "Gemini" in status_output or "gemini-2.5-flash" in status_output
        assert studio.get_active_project_model() == "9router/gemini-2.5-flash"
        assert studio.current_project.model == "9router/gemini-2.5-flash"
        assert studio.engine.default_model == "9router/gemini-2.5-flash"

