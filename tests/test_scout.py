"""Unit tests for the Intelligent Literature Scout pipeline and CLI integration."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from rich.console import Console

from scrpcy.cli_studio import CLIStudio
from scrpcy.core.config import ScrpcyConfig
from scrpcy.core.engine import LLMEngine
from scrpcy.core.models import DocumentMetadata
from scrpcy.core.project import ProjectManager
from scrpcy.core.scout import (
    LiteratureScout,
    ScoutQuery,
    ScoutResult,
    reconstruct_inverted_abstract,
)
from scrpcy.core.search import PaperSearcher, SearchResult
from scrpcy.core.workspace import WorkspaceManager
from scrpcy.exceptions import SearchError


class TestScoutModels:
    """Tests for ScoutQuery and ScoutResult schemas and methods."""

    def test_scout_query_defaults(self) -> None:
        query = ScoutQuery(keywords="deep learning")
        assert query.keywords == "deep learning"
        assert query.year_start is None
        assert query.year_end is None
        assert query.is_oa_only is False
        assert query.authors == []
        assert query.concepts == []

    def test_scout_result_ensure_cite_key(self) -> None:
        paper = ScoutResult(
            title="Generative Adversarial Nets",
            authors=["Ian J. Goodfellow", "Jean Pouget-Abadie"],
            year=2014,
            source="openalex",
        )
        cite_key = paper.ensure_cite_key()
        assert cite_key == "goodfellow2014generative"
        assert paper.cite_key == "goodfellow2014generative"

    def test_scout_result_ensure_cite_key_fallback(self) -> None:
        paper = ScoutResult(
            title="!!!",
            authors=[],
            year=None,
        )
        cite_key = paper.ensure_cite_key()
        assert cite_key == "unknownnodate"


class TestReconstructInvertedAbstract:
    """Tests for reconstructing full-text abstracts from OpenAlex inverted indexes."""

    def test_empty_or_none(self) -> None:
        assert reconstruct_inverted_abstract(None) == ""
        assert reconstruct_inverted_abstract({}) == ""

    def test_reconstruction_order(self) -> None:
        index = {
            "is": [1],
            "This": [0],
            "a": [2],
            "test": [3],
            "abstract.": [4],
        }
        reconstructed = reconstruct_inverted_abstract(index)
        assert reconstructed == "This is a test abstract."


class TestLiteratureScoutCriteriaExtraction:
    """Tests for extracting search criteria via heuristic and LLM."""

    @pytest.fixture
    def scout(self, temp_workspace: WorkspaceManager) -> LiteratureScout:
        return LiteratureScout(workspace=temp_workspace)

    def test_heuristic_indonesian_input(self, scout: LiteratureScout) -> None:
        text = "saya butuh paper tentang kesehatan mental remaja 2018-2019 open access"
        q = scout.extract_search_criteria(text)
        assert q.year_start == 2018
        assert q.year_end == 2019
        assert q.is_oa_only is True
        assert "kesehatan mental remaja" in q.keywords
        assert "saya butuh" not in q.keywords

    def test_heuristic_english_input(self, scout: LiteratureScout) -> None:
        text = "i need papers about quantum computing year 2022 pdf"
        q = scout.extract_search_criteria(text)
        assert q.year_start == 2022
        assert q.year_end == 2022
        assert q.is_oa_only is True
        assert "quantum computing" in q.keywords

    def test_heuristic_empty_input(self, scout: LiteratureScout) -> None:
        q = scout.extract_search_criteria("")
        assert q.keywords == "academic research"
        assert q.summary != ""

    @patch("litellm.completion")
    def test_llm_extraction_success(
        self,
        mock_completion: MagicMock,
        scout: LiteratureScout,
    ) -> None:
        cfg = ScrpcyConfig(
            default_model="gemini/gemini-2.5-pro",
            api_keys={"gemini": "test-gemini-key"},
        )
        engine = LLMEngine(config=cfg)

        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps({
                        "keywords": "adolescent mental health intervention",
                        "year_start": 2018,
                        "year_end": 2019,
                        "concepts": ["Psychology", "Adolescence"],
                        "is_oa_only": True,
                        "summary": "Interventions for youth mental health (2018-2019)",
                    })
                )
            )
        ]
        mock_completion.return_value = mock_resp

        q = scout.extract_search_criteria("paper kesehatan 2018-2019", engine=engine)
        assert q.keywords == "adolescent mental health intervention"
        assert q.year_start == 2018
        assert q.year_end == 2019
        assert q.is_oa_only is True
        assert "Psychology" in q.concepts

    @patch("litellm.completion", side_effect=Exception("API Error"))
    def test_llm_extraction_failure_falls_back(
        self,
        mock_completion: MagicMock,
        scout: LiteratureScout,
    ) -> None:
        cfg = ScrpcyConfig(
            default_model="gemini/gemini-2.5-pro",
            gemini_api_key="test-gemini-key",
        )
        engine = LLMEngine(config=cfg)

        q = scout.extract_search_criteria("transformer models 2021", engine=engine)
        assert q.year_start == 2021
        assert "transformer models" in q.keywords


class TestLiteratureScoutSearch:
    """Tests for multi-source search and OpenAlex integration."""

    @pytest.fixture
    def scout(self, temp_workspace: WorkspaceManager) -> LiteratureScout:
        return LiteratureScout(workspace=temp_workspace)

    @patch("scrpcy.core.scout.PYALEX_AVAILABLE", True)
    @patch("scrpcy.core.scout.Works")
    def test_search_openalex_success(
        self,
        mock_works_cls: MagicMock,
        scout: LiteratureScout,
    ) -> None:
        mock_works = MagicMock()
        mock_works_cls.return_value = mock_works
        mock_works.search.return_value = mock_works
        mock_works.filter.return_value = mock_works
        mock_works.get.return_value = [
            {
                "title": "Deep Residual Learning for Image Recognition",
                "authorships": [
                    {"author": {"display_name": "Kaiming He"}},
                    {"author": {"display_name": "Xiangyu Zhang"}},
                ],
                "publication_year": 2016,
                "doi": "https://doi.org/10.1109/cvpr.2016.90",
                "open_access": {
                    "is_oa": True,
                    "oa_url": "https://arxiv.org/pdf/1512.03385.pdf",
                },
                "cited_by_count": 150000,
                "abstract_inverted_index": {"Deeper": [0], "neural": [1], "networks": [2]},
            }
        ]

        query = ScoutQuery(keywords="resnet", year_start=2016, year_end=2016, is_oa_only=True)
        results = scout.search_openalex(query)

        assert len(results) == 1
        res = results[0]
        assert res.title == "Deep Residual Learning for Image Recognition"
        assert res.authors == ["Kaiming He", "Xiangyu Zhang"]
        assert res.year == 2016
        assert res.doi == "10.1109/cvpr.2016.90"
        assert res.has_pdf is True
        assert res.citation_count == 150000
        assert res.source == "openalex"

    @pytest.mark.asyncio
    async def test_search_multi_source_deduplication(
        self,
        scout: LiteratureScout,
    ) -> None:
        s1 = ScoutResult(
            title="Attention Is All You Need",
            authors=["Ashish Vaswani"],
            year=2017,
            source="openalex",
        )
        s2 = SearchResult(
            paper_id="s2-123",
            title="Attention Is All You Need",
            authors=["Ashish Vaswani"],
            year=2017,
            has_pdf=True,
            pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
        )
        s3 = SearchResult(
            paper_id="s2-456",
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            authors=["Jacob Devlin"],
            year=2019,
            has_pdf=True,
            pdf_url="https://arxiv.org/pdf/1810.04805.pdf",
        )

        with patch.object(scout, "search_openalex", return_value=[s1]):
            with patch.object(scout.searcher, "search", new_callable=AsyncMock) as mock_s2:
                mock_s2.return_value = [s2, s3]
                query = ScoutQuery(keywords="transformers")
                combined = await scout.search_multi_source(query)

                assert len(combined) == 2
                titles = [c.title for c in combined]
                assert "Attention Is All You Need" in titles
                assert "BERT: Pre-training of Deep Bidirectional Transformers" in titles


class TestLiteratureScoutCurateAndRank:
    """Tests for offline and LLM-assisted paper curation."""

    @pytest.fixture
    def scout(self, temp_workspace: WorkspaceManager) -> LiteratureScout:
        return LiteratureScout(workspace=temp_workspace)

    def test_offline_heuristic_ranking(self, scout: LiteratureScout) -> None:
        p1 = ScoutResult(title="Paper A", citation_count=50, has_pdf=False)
        p2 = ScoutResult(title="Paper B", citation_count=10, has_pdf=True)
        p3 = ScoutResult(title="Paper C", citation_count=500, has_pdf=True)

        ranked = scout.curate_and_rank("ai", [p1, p2, p3], engine=None)
        assert ranked[0].title == "Paper C"
        assert ranked[1].title == "Paper B"
        assert ranked[2].title == "Paper A"

    @patch("litellm.completion")
    def test_llm_curation_with_seminal_paper(
        self,
        mock_completion: MagicMock,
        scout: LiteratureScout,
    ) -> None:
        cfg = ScrpcyConfig(
            default_model="gemini/gemini-2.5-pro",
            api_keys={"gemini": "test-gemini-key"},
        )
        engine = LLMEngine(config=cfg)

        c1 = ScoutResult(title="Recent Paper On Attention", year=2023, source="openalex")
        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps({
                        "papers": [
                            {
                                "candidate_index": 1,
                                "relevance_reason": "Directly evaluates contemporary attention mechanisms.",
                            },
                            {
                                "is_ai_seminal": True,
                                "title": "Attention Is All You Need",
                                "authors": ["Ashish Vaswani", "Noam Shazeer"],
                                "year": 2017,
                                "relevance_reason": "Foundational seminal paper introducing the Transformer architecture.",
                            },
                        ]
                    })
                )
            )
        ]
        mock_completion.return_value = mock_resp

        ranked = scout.curate_and_rank("transformers", [c1], engine=engine)
        assert len(ranked) == 2
        assert ranked[0].title == "Recent Paper On Attention"
        assert ranked[0].relevance_reason == "Directly evaluates contemporary attention mechanisms."
        assert ranked[1].title == "Attention Is All You Need"
        assert ranked[1].source == "ai_recommended"


class TestLiteratureScoutDownload:
    """Tests for downloading open-access papers into the project vault."""

    @pytest.fixture
    def scout(self, temp_workspace: WorkspaceManager) -> LiteratureScout:
        return LiteratureScout(workspace=temp_workspace)

    @pytest.mark.asyncio
    async def test_download_no_pdf_raises(self, scout: LiteratureScout) -> None:
        paper = ScoutResult(title="No PDF Paper", has_pdf=False, pdf_url=None)
        with pytest.raises(SearchError, match="No open-access PDF"):
            await scout.download_scout_paper(paper)

    @pytest.mark.asyncio
    async def test_download_http_error_raises(self, scout: LiteratureScout) -> None:
        paper = ScoutResult(
            title="Broken PDF Paper",
            has_pdf=True,
            pdf_url="https://example.com/broken.pdf",
        )

        mock_resp = MagicMock(status_code=404)
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            with pytest.raises(SearchError, match="Failed to download PDF"):
                await scout.download_scout_paper(paper)

    @pytest.mark.asyncio
    async def test_download_success(
        self,
        scout: LiteratureScout,
        temp_workspace: WorkspaceManager,
    ) -> None:
        paper = ScoutResult(
            title="Sample Research Paper",
            authors=["Jane Doe"],
            year=2021,
            has_pdf=True,
            pdf_url="https://example.com/paper.pdf",
            abstract="Sample abstract content.",
        )

        fake_pdf_bytes = b"%PDF-1.4\nfake pdf content for testing"
        mock_resp = MagicMock(status_code=200, content=fake_pdf_bytes)

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            pdf_path = await scout.download_scout_paper(paper)

            assert pdf_path.exists()
            assert pdf_path.parent == temp_workspace.vault_dir
            assert pdf_path.read_bytes() == fake_pdf_bytes

            cache_files = list(temp_workspace.cache_dir.glob("*.md"))
            assert len(cache_files) >= 1


class TestCLIStudioScoutCommand:
    """Tests for /scout command execution in CLIStudio."""

    @pytest.fixture
    def test_console(self) -> Console:
        return Console(file=io.StringIO(), force_terminal=True, color_system=None)

    @pytest.fixture
    def studio(
        self,
        temp_workspace: WorkspaceManager,
        test_console: Console,
        tmp_path: Path,
    ) -> CLIStudio:
        pm = ProjectManager(registry_file=tmp_path / "projects.json")
        cfg = ScrpcyConfig(workspace_path=str(temp_workspace.root_dir))
        st = CLIStudio(
            workspace=temp_workspace,
            config=cfg,
            console=test_console,
            project_manager=pm,
        )
        info = pm.create_project(name="my_research", path=temp_workspace.root_dir)
        st.open_project(info)
        return st

    def test_scout_requires_project_mode(self, test_console: Console, tmp_path: Path) -> None:
        pm = ProjectManager(registry_file=tmp_path / "projects.json")
        st = CLIStudio(console=test_console, project_manager=pm)
        st.cmd_scout()
        output = test_console.file.getvalue()
        assert "Please open a project first" in output

    def test_scout_interactive_cancel(
        self,
        studio: CLIStudio,
        test_console: Console,
    ) -> None:
        with patch.object(studio.console, "input", side_effect=["q"]):
            studio.cmd_scout()
            output = test_console.file.getvalue()
            assert "Scout canceled." in output

    def test_scout_interactive_search_and_download(
        self,
        studio: CLIStudio,
        test_console: Console,
    ) -> None:
        candidate = ScoutResult(
            title="Transformer Attention Models",
            authors=["A. Author"],
            year=2022,
            has_pdf=True,
            pdf_url="https://example.com/test.pdf",
            source="openalex",
            relevance_reason="Highly relevant attention paper.",
        )

        mock_query = ScoutQuery(
            keywords="transformers",
            year_start=2022,
            year_end=2022,
            is_oa_only=True,
            summary="Transformer models inquiry",
        )

        with patch.object(
            studio.scout,
            "extract_search_criteria",
            return_value=mock_query,
        ), patch.object(
            studio.scout,
            "search_multi_source",
            new_callable=AsyncMock,
            return_value=[candidate],
        ), patch.object(
            studio.scout,
            "curate_and_rank",
            return_value=[candidate],
        ), patch.object(
            studio.scout,
            "download_scout_paper",
            new_callable=AsyncMock,
            return_value=Path("vault/test.pdf"),
        ), patch.object(
            studio.console,
            "input",
            side_effect=["paper transformers 2022", "1"],
        ):
            studio.cmd_scout()
            for future in list(studio.task_manager._futures.values()):
                future.result(timeout=5)
            output = test_console.file.getvalue()
            assert "Literature Scout" in output
            assert "Transformer Attention Models" in output
            assert "Background Download Agent Dispatched" in output
            assert "test" in studio.active_documents
            tasks = studio.task_manager.list_tasks()
            assert len(tasks) > 0
            assert tasks[0].status == "completed"
            assert "Added #1 to vault & indexed" in "\n".join(tasks[0].logs)
