"""Unit tests for interactive journal and official web search & ingestion in Vault."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from scripcy.cli_studio import CLIStudio
from scripcy.core.config import ScripcyConfig
from scripcy.core.project import ProjectManager
from scripcy.core.scout import ScoutQuery, ScoutResult
from scripcy.core.web_scout import WebSearchResult
from scripcy.core.workspace import WorkspaceManager
from scripcy.ui.interactive import interactive_carousel_table


@pytest.fixture
def test_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=True, color_system=None)


@pytest.fixture
def studio_fixture(temp_workspace: WorkspaceManager, test_console: Console, tmp_path: Path) -> CLIStudio:
    cfg = ScripcyConfig(
        default_model="gemini/gemini-2.5-flash",
        workspace_path=str(temp_workspace.root_dir),
    )
    pm = ProjectManager(registry_file=tmp_path / "projects.json")
    studio = CLIStudio(
        workspace=temp_workspace,
        config=cfg,
        console=test_console,
        project_manager=pm,
    )
    proj = pm.create_project(name="search_test_proj", path=temp_workspace.root_dir)
    studio.open_project(proj)
    return studio


def test_interactive_carousel_table_non_interactive(test_console: Console):
    columns = [
        {"name": "#", "justify": "right", "width": 3},
        {"name": "Title", "style": "bold white"},
    ]
    rows = [
        ["1", "Paper One"],
        ["2", "Paper Two"],
        ["3", "Paper Three"],
    ]

    with patch.object(test_console, "input", return_value="1, 3"):
        result = interactive_carousel_table(
            title="Test Papers",
            columns=columns,
            rows=rows,
            console=test_console,
            prompt_label="download > ",
        )
    assert result == "1, 3"


def test_vault_find_paper_interactive_flow(studio_fixture: CLIStudio, monkeypatch: pytest.MonkeyPatch):
    mock_candidates = [
        ScoutResult(
            title="Deep Learning in Peatland Wildfire",
            authors=["A. Sutanto", "B. Wijaya"],
            year=2023,
            has_pdf=True,
            pdf_url="https://example.com/paper1.pdf",
            source="openalex",
            doi="10.1234/test1",
            relevance_reason="Relevant",
        ),
        ScoutResult(
            title="Hydrological Modeling of Peatlands",
            authors=["C. Gunawan"],
            year=2022,
            has_pdf=False,
            abstract="Study of peat hydrology.",
            source="openalex",
            doi="10.1234/test2",
            relevance_reason="Relevant",
        ),
    ]

    # Mock scout methods
    monkeypatch.setattr(
        studio_fixture.scout,
        "extract_search_criteria",
        lambda query, engine=None: ScoutQuery(keywords=query),
    )
    monkeypatch.setattr(
        studio_fixture.scout,
        "search_multi_source",
        AsyncMock(return_value=mock_candidates),
    )

    downloaded = []
    fake_pdf = studio_fixture.workspace.vault_dir / "Deep_Learning_in_Peatland_Wildfire.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 test")

    monkeypatch.setattr(
        studio_fixture.scout,
        "download_scout_paper",
        AsyncMock(return_value=fake_pdf),
    )
    monkeypatch.setattr(
        studio_fixture.scout,
        "ingest_abstract_paper",
        lambda paper: downloaded.append(paper.title),
    )

    # Mock user choosing paper 1 in carousel
    monkeypatch.setattr(
        "scripcy.studio.vault.interactive_carousel_table",
        lambda **kwargs: "1",
    )

    studio_fixture.run_interactive_vault_find_paper(initial_query="kebakaran lahan gambut")

    # Verify history recorded
    hist = studio_fixture.vault_manager.list_history()
    assert any(h.get("action") == "tambah_journal" for h in hist)

    output = studio_fixture.console.file.getvalue()
    assert "Berhasil menambahkan 1 journal ke Vault" in output


def test_vault_find_paper_cancel_flow(studio_fixture: CLIStudio, monkeypatch: pytest.MonkeyPatch):
    mock_candidates = [
        ScoutResult(
            title="Paper One",
            authors=["Author A"],
            year=2023,
            has_pdf=True,
            pdf_url="https://example.com/p1.pdf",
        ),
    ]

    monkeypatch.setattr(
        studio_fixture.scout,
        "extract_search_criteria",
        lambda query, engine=None: ScoutQuery(keywords=query),
    )
    monkeypatch.setattr(
        studio_fixture.scout,
        "search_multi_source",
        AsyncMock(return_value=mock_candidates),
    )

    # User cancels carousel
    monkeypatch.setattr(
        "scripcy.studio.vault.interactive_carousel_table",
        lambda **kwargs: "q",
    )

    studio_fixture.run_interactive_vault_find_paper(initial_query="test query")

    output = studio_fixture.console.file.getvalue()
    assert "Pencarian selesai tanpa mengunduh berkas" in output


def test_vault_find_web_interactive_flow(studio_fixture: CLIStudio, monkeypatch: pytest.MonkeyPatch):
    mock_web_results = [
        WebSearchResult(
            title="Laporan Data Kebakaran Hutan KLHK",
            url="https://sipongi.menlhk.go.id/publikasi/laporan2023",
            snippet="Data resmi karhutla tahun 2023 di Indonesia",
            domain="menlhk.go.id",
            authority_score=0.98,
            is_official=True,
            institution="KLHK RI",
            authority_tier=1,
        )
    ]

    mock_parsed_doc = MagicMock()
    mock_parsed_doc.file_path = Path("laporan2023.html")
    mock_parsed_doc.cache_path = Path("laporan2023.md")
    mock_parsed_doc.metadata.title = "Laporan Data Kebakaran Hutan KLHK"
    mock_parsed_doc.metadata.institution = "KLHK RI"
    mock_parsed_doc.metadata.cite_key = "klhk2023karhutla"
    mock_parsed_doc.metadata.authority_score = 0.98
    mock_parsed_doc.metadata.authority_tier = 1

    fake_web_scout = MagicMock()
    fake_web_scout.search_official_web = AsyncMock(return_value=mock_web_results)
    fake_web_scout.ingest_url = AsyncMock(return_value=mock_parsed_doc)
    studio_fixture.web_scout = fake_web_scout

    # User selects item 1
    monkeypatch.setattr(
        "scripcy.studio.vault.interactive_carousel_table",
        lambda **kwargs: "1",
    )

    studio_fixture.run_interactive_vault_find_web(initial_target="data karhutla")

    # Verify history recorded
    hist = studio_fixture.vault_manager.list_history()
    assert any(h.get("action") == "tambah_web" for h in hist)

    output = studio_fixture.console.file.getvalue()
    assert "Berhasil menambahkan 1 dokumen web resmi ke Vault" in output


def test_vault_find_web_direct_url_flow(studio_fixture: CLIStudio):
    mock_parsed_doc = MagicMock()
    mock_parsed_doc.file_path = Path("pedoman_bmkg.html")
    mock_parsed_doc.cache_path = Path("pedoman_bmkg.md")
    mock_parsed_doc.metadata.title = "Pedoman Cuaca Ekstrem BMKG"
    mock_parsed_doc.metadata.institution = "BMKG"
    mock_parsed_doc.metadata.cite_key = "bmkg2024cuaca"
    mock_parsed_doc.metadata.authority_score = 0.95
    mock_parsed_doc.metadata.authority_tier = 1

    fake_web_scout = MagicMock()
    fake_web_scout.ingest_url = AsyncMock(return_value=mock_parsed_doc)
    studio_fixture.web_scout = fake_web_scout

    studio_fixture.run_interactive_vault_find_web(initial_target="https://bmkg.go.id/pedoman_cuaca")

    hist = studio_fixture.vault_manager.list_history()
    assert any(h.get("action") == "tambah_web" for h in hist)

    output = studio_fixture.console.file.getvalue()
    assert "Dokumen web resmi berhasil dimasukkan ke Vault" in output
