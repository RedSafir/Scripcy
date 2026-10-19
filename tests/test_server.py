"""Comprehensive unit test suite for FastAPI backend endpoints and Typer CLI commands."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
import pytest
from typer.testing import CliRunner

from scrpcy.cli import app as cli_app
from scrpcy.core.config import ScrpcyConfig
from scrpcy.core.engine import EvidenceChunk, LLMEngine
from scrpcy.core.extractor import MockMetadataExtractor
from scrpcy.core.mcp_client import MockMCPTransport, WordCraftClient
from scrpcy.core.models import DocumentMetadata
from scrpcy.core.parser import DocumentParser
from scrpcy.core.search import PaperSearcher, SearchResult
from scrpcy.core.skills import SkillsLoader
from scrpcy.core.workspace import WorkspaceManager
from scrpcy.server import create_app


@pytest.fixture
def test_app_and_deps(tmp_path: Path):
    workspace = WorkspaceManager(root_dir=tmp_path / "server_ws")
    config = ScrpcyConfig(
        default_model="gemini/gemini-2.5-pro",
        skills_dir=str(tmp_path / "skills"),
        workspace_path=str(workspace.root_dir),
    )
    skills = SkillsLoader(skills_dir=config.skills_dir, auto_init=True)

    mock_extractor = MockMetadataExtractor(
        fixed_metadata=DocumentMetadata(
            title="Synthetic Attention Paper",
            authors=["Alice Researcher", "Bob Scientist"],
            year=2024,
            doi="10.1000/182",
            cite_key="alice2024synthetic",
        )
    )
    parser = DocumentParser(workspace=workspace, extractor=mock_extractor)

    async def mock_stream_fn(**kwargs: Any):
        class MockChoiceDelta:
            def __init__(self, content: str) -> None:
                self.content = content

        class MockChoice:
            def __init__(self, content: str) -> None:
                self.delta = MockChoiceDelta(content)

        class MockChunk:
            def __init__(self, content: str) -> None:
                self.choices = [MockChoice(content)]

        for token in ["The ", "Transformer ", "architecture ", "is ", "effective."]:
            yield MockChunk(token)

    engine = LLMEngine(
        config=config,
        skills_loader=skills,
        acompletion_fn=mock_stream_fn,
    )

    searcher = PaperSearcher(workspace=workspace)
    wordcraft = WordCraftClient(
        workspace=workspace,
        transport=MockMCPTransport(),
        fallback_to_docx=True,
    )

    app = create_app(
        workspace=workspace,
        config=config,
        engine=engine,
        searcher=searcher,
        parser=parser,
        wordcraft=wordcraft,
    )

    client = TestClient(app)
    return client, workspace, app


class TestServerEndpoints:
    """Tests for core REST and SSE endpoints."""

    def test_health_endpoint(self, test_app_and_deps) -> None:
        client, _, _ = test_app_and_deps
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["app"] == "scrpcy"

    def test_config_endpoint(self, test_app_and_deps) -> None:
        client, _, _ = test_app_and_deps
        response = client.get("/api/config")
        assert response.status_code == 200
        data = response.json()
        assert data["default_model"] == "gemini/gemini-2.5-pro"
        assert "citation" in data

    def test_skills_endpoint(self, test_app_and_deps) -> None:
        client, _, _ = test_app_and_deps
        response = client.get("/api/skills")
        assert response.status_code == 200
        skills = response.json()
        names = [s["name"] for s in skills]
        assert "academic_writer" in names

    def test_list_and_toggle_documents(self, test_app_and_deps) -> None:
        client, workspace, _ = test_app_and_deps

        # Initially empty
        res_empty = client.get("/api/documents")
        assert res_empty.status_code == 200
        assert res_empty.json() == []

        # Add sample document to workspace cache
        meta = DocumentMetadata(
            title="Residual Networks",
            authors=["Kaiming He"],
            year=2016,
            cite_key="he2016resnet",
        )
        content = f"{meta.to_frontmatter()}\n# Deep Residual Learning\nContent here."
        workspace.save_to_cache("resnet.md", content)

        res_populated = client.get("/api/documents")
        assert res_populated.status_code == 200
        docs = res_populated.json()
        assert len(docs) == 1
        assert docs[0]["title"] == "Residual Networks"
        assert docs[0]["stem"] == "resnet"
        assert docs[0]["active"] is True

        # Toggle document inactive
        res_toggle = client.post("/api/documents/toggle", json={"stem": "resnet", "active": False})
        assert res_toggle.status_code == 200
        assert res_toggle.json()["active"] is False

    def test_get_document_content(self, test_app_and_deps) -> None:
        client, workspace, _ = test_app_and_deps

        workspace.save_to_cache(
            "doc1.md",
            "---\ntitle: Sample Doc\nauthors: [Jane]\nyear: 2024\n---\nBody text.",
        )

        res_found = client.get("/api/documents/doc1")
        assert res_found.status_code == 200
        data = res_found.json()
        assert data["stem"] == "doc1"
        assert data["metadata"]["title"] == "Sample Doc"
        assert "Body text." in data["body"]

        res_not_found = client.get("/api/documents/missing_stem")
        assert res_not_found.status_code == 404

    def test_upload_document(self, test_app_and_deps, synthetic_pdf_factory) -> None:
        client, _, _ = test_app_and_deps
        pdf_path = synthetic_pdf_factory(filename="upload_test.pdf")

        with open(pdf_path, "rb") as f:
            response = client.post(
                "/api/documents/upload",
                files={"file": ("upload_test.pdf", f, "application/pdf")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["metadata"]["title"] == "Synthetic Attention Paper"
        assert data["cache_file"] == "upload_test.md"

    def test_upload_non_pdf_fails(self, test_app_and_deps) -> None:
        client, _, _ = test_app_and_deps
        response = client.post(
            "/api/documents/upload",
            files={"file": ("notes.txt", io.BytesIO(b"plain text content"), "text/plain")},
        )
        assert response.status_code == 400
        assert "Only PDF files" in response.json()["detail"]

    def test_search_papers_endpoint(self, test_app_and_deps, monkeypatch: pytest.MonkeyPatch) -> None:
        client, _, _ = test_app_and_deps

        sample_results = [
            SearchResult(
                title="Attention Is All You Need",
                authors=["Ashish Vaswani"],
                year=2017,
                pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
            )
        ]

        async def mock_search(self, query: str, **kwargs: Any):
            return sample_results

        monkeypatch.setattr(PaperSearcher, "search", mock_search)

        response = client.get("/api/search?q=attention&limit=3")
        assert response.status_code == 200
        items = response.json()
        assert len(items) == 1
        assert items[0]["title"] == "Attention Is All You Need"
        assert items[0]["has_pdf"] is True

    def test_chat_stream_endpoint(self, test_app_and_deps) -> None:
        client, workspace, _ = test_app_and_deps

        meta = DocumentMetadata(title="Paper", cite_key="paper2024")
        workspace.save_to_cache("paper.md", f"{meta.to_frontmatter()}\nTransformer content.")

        response = client.post(
            "/api/chat/stream",
            json={
                "query": "How do Transformers work?",
                "skill_name": "academic_writer",
                "active_documents": ["paper.md"],
            },
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        body_text = response.text
        assert "event: evidence" in body_text
        assert "event: token" in body_text
        assert "event: done" in body_text
        assert "Transformer" in body_text

    def test_document_generate_and_download(self, test_app_and_deps) -> None:
        client, _, _ = test_app_and_deps

        gen_payload = {
            "title": "Generated Review",
            "sections": [
                {
                    "heading": "1. Introduction",
                    "level": 1,
                    "content": "Self-attention models have proven highly effective.",
                    "citations": [{"authors": ["Ashish Vaswani"], "year": 2017}],
                }
            ],
            "references": [
                {
                    "title": "Attention Is All You Need",
                    "authors": ["Ashish Vaswani"],
                    "year": 2017,
                }
            ],
            "style": "apa",
        }

        res_gen = client.post("/api/document/generate", json=gen_payload)
        assert res_gen.status_code == 200
        gen_data = res_gen.json()
        assert gen_data["status"] == "success"
        filename = gen_data["filename"]
        assert filename.endswith(".docx")

        # Download manuscript
        res_dl = client.get(f"/api/document/download/{filename}")
        assert res_dl.status_code == 200
        assert len(res_dl.content) > 0
        assert res_dl.content.startswith(b"PK")  # Standard ZIP/DOCX header


class TestCLIOperations:
    """Tests for Typer command-line interface commands."""

    def test_cli_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli_app, ["--help"])
        assert result.exit_code == 0
        assert "scrpcy" in result.output
        assert "start" in result.output
        assert "config" in result.output

    def test_cli_config_show(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli_app, ["config", "show"])
        assert result.exit_code == 0
        assert "Default Model" in result.output

    def test_cli_config_set_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg_file = tmp_path / "cli_cfg.json"
        monkeypatch.setenv("SCRPCY_CONFIG", str(cfg_file))

        runner = CliRunner()
        result = runner.invoke(cli_app, ["config", "set-key", "gemini", "ai-key-secret-123"])
        assert result.exit_code == 0
        assert "Saved API key for 'gemini'" in result.output

        cfg = ScrpcyConfig.load(cfg_file)
        assert cfg.api_keys["gemini"] == "ai-key-secret-123"

    def test_cli_config_set_model(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg_file = tmp_path / "cli_cfg.json"
        monkeypatch.setenv("SCRPCY_CONFIG", str(cfg_file))

        runner = CliRunner()
        result = runner.invoke(cli_app, ["config", "set-model", "openai/gpt-4o"])
        assert result.exit_code == 0
        assert "Updated default model to 'openai/gpt-4o'" in result.output

        cfg = ScrpcyConfig.load(cfg_file)
        assert cfg.default_model == "openai/gpt-4o"
