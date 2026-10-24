"""Unit tests for FactChecker, Devil's Advocate, and Reference Mapping."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import AsyncMock, patch
import pytest
from rich.console import Console

from scrpcy.core.config import ScrpcyConfig
from scrpcy.core.critic import DevilsAdvocate, CritiqueReport
from scrpcy.core.engine import EvidenceChunk, LLMEngine
from scrpcy.core.project import ProjectManager
from scrpcy.core.scout import LiteratureScout, ScoutResult
from scrpcy.core.search import PaperSearcher, SearchResult
from scrpcy.core.verifier import FactChecker, VerificationReport
from scrpcy.core.workspace import WorkspaceManager
from scrpcy.cli_studio import CLIStudio


class TestFactChecker:
    """Tests for claim extraction and grounding verification against vault evidence."""

    def test_extract_atomic_claims(self) -> None:
        checker = FactChecker()
        text = (
            "# Introduction\n\n"
            "Kebakaran hutan gambut tropis melepaskan emisi karbon dalam jumlah masif ke atmosfer. "
            "Partikel PM2.5 yang dihasilkan dapat menempuh jarak ratusan kilometer lintas batas negara. "
            "Pendekatan restorasi hidrologis melalui rewetting terbukti menurunkan risiko kebakaran secara signifikan."
        )
        claims = checker.extract_atomic_claims(text)
        assert len(claims) == 3
        assert "Kebakaran hutan gambut tropis" in claims[0]

    def test_verify_claims_grounded(self) -> None:
        checker = FactChecker()
        evidence = [
            EvidenceChunk(
                chunk_id="doc1_c1",
                source="Page2002",
                text="Tropical peatland fires release massive carbon emissions into the atmosphere and cause hazardous PM2.5 haze.",
            ),
        ]
        text = "Kebakaran lahan gambut melepaskan carbon emissions massive ke atmosphere dan menyebabkan PM2.5 haze."
        report = checker.verify_synthesis(text, evidence)
        assert report.total_claims >= 1
        assert report.grounding_score >= 0.5
        assert report.supported_claims >= 1
        assert report.hallucination_warning is None

    def test_verify_claims_ungrounded_triggers_warning(self) -> None:
        checker = FactChecker()
        evidence = [
            EvidenceChunk(
                chunk_id="doc1_c1",
                source="OceanPaper",
                text="Deep ocean iron fertilization impacts phytoplankton bloom dynamics.",
            ),
        ]
        text = "Kebakaran hutan gambut di Kalimantan memicu penurunan populasi orangutan hingga 80 persen pada tahun 2015."
        report = checker.verify_synthesis(text, evidence)
        assert report.grounding_score < 0.5
        assert report.unsupported_claims >= 1
        assert report.hallucination_warning is not None


class TestDevilsAdvocate:
    """Tests for adversarial critique, recency detection, and counter-argument generation."""

    def test_extract_citation_years(self) -> None:
        critic = DevilsAdvocate()
        text = "Menurut studi terdahulu [Page2002] dan laporan lanjutan (Marlier, 2019), emisi gambut sangat fluktuatif."
        years = critic.extract_citation_years(text)
        assert 2002 in years
        assert 2019 in years

    def test_check_recency_outdated(self) -> None:
        critic = DevilsAdvocate(recency_threshold_years=5)
        # Citations only up to 2015
        is_outdated, msg = critic.check_recency([2002, 2010, 2014])
        assert is_outdated is True
        assert msg is not None
        assert "lebih dari 5 tahun" in msg

    def test_check_recency_fresh(self) -> None:
        critic = DevilsAdvocate(recency_threshold_years=5)
        is_outdated, msg = critic.check_recency([2023, 2024])
        assert is_outdated is False
        assert msg is None

    def test_critique_heuristic_scores_and_counter_arguments(self) -> None:
        critic = DevilsAdvocate()
        query = "Mengapa restorasi gambut selalu efektif?"
        one_sided_answer = (
            "Restorasi gambut melalui sekat kanal selalu berhasil 100% tanpa adanya kendala sosial maupun ekonomi. "
            "Tidak ada kegagalan yang pernah tercatat dalam sejarah implementasi."
        )
        report = critic.critique_heuristic(query, one_sided_answer)
        assert report.score < 8
        assert len(report.weaknesses) > 0
        assert len(report.counter_arguments) > 0
        assert report.needs_citation_mapping is True


class TestReferenceMapping:
    """Tests for citation network traversal and vault reference mapping."""

    @pytest.mark.asyncio
    async def test_traverse_citation_network(self, tmp_path: Path) -> None:
        searcher = PaperSearcher()
        mock_cites = [
            SearchResult(
                title="Recent Advance on Peat Emissions",
                authors=["Budi", "Siti"],
                year=2024,
                doi="10.1234/recent2024",
                source_api="semantic_scholar_citations",
            )
        ]
        mock_refs = [
            SearchResult(
                title="Foundational Peatland Carbon Study",
                authors=["Page", "Siegert"],
                year=2002,
                doi="10.1234/page2002",
                source_api="semantic_scholar_references",
            )
        ]

        searcher.get_paper_citations = AsyncMock(return_value=mock_cites)
        searcher.get_paper_references = AsyncMock(return_value=mock_refs)

        scout = LiteratureScout(searcher=searcher)
        network = await scout.traverse_citation_network("10.1234/seed", direction="both", limit=5)

        assert len(network) == 2
        titles = [p.title for p in network]
        assert "Recent Advance on Peat Emissions" in titles
        assert "Foundational Peatland Carbon Study" in titles

    @pytest.mark.asyncio
    async def test_map_vault_citation_network(self) -> None:
        searcher = PaperSearcher()
        searcher.get_paper_citations = AsyncMock(return_value=[
            SearchResult(title="Connected Paper Alpha", authors=["Alice"], year=2023, source_api="semantic_scholar_citations")
        ])
        searcher.get_paper_references = AsyncMock(return_value=[])

        scout = LiteratureScout(searcher=searcher)
        vault_stems = ["peatland_wildfire_2015"]
        results = await scout.map_vault_citation_network(vault_stems, limit=2)
        assert len(results) == 1
        assert results[0].title == "Connected Paper Alpha"


class TestAgentLoopVerificationIntegration:
    """Verify FactChecker and Devil's Advocate execution inside CLIStudio handle_chat."""

    def test_handle_chat_telemetry_includes_grounding_and_critic_badges(
        self,
        temp_workspace: WorkspaceManager,
        tmp_path: Path,
    ) -> None:
        temp_workspace.save_to_cache(
            "peat_study.md",
            "# Peat Study\nTropical peatland fires emit carbon and PM2.5 haze particles.",
        )

        console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
        pm = ProjectManager(registry_file=tmp_path / "reg.json")
        proj = pm.create_project("thesis", temp_workspace.root_dir)
        cfg = ScrpcyConfig(workspace_path=str(temp_workspace.root_dir))

        class MockChunk:
            def __init__(self, text: str) -> None:
                self.text = text

        async def mock_acompletion(**kwargs: Any):
            yield MockChunk("Berdasarkan [Page2002], kebakaran lahan gambut tropis memicu pelepasan karbon dan partikel PM2.5.")

        engine = LLMEngine(config=cfg, acompletion_fn=mock_acompletion)
        studio = CLIStudio(
            workspace=temp_workspace,
            config=cfg,
            console=console,
            project_manager=pm,
            engine=engine,
        )
        studio.open_project(proj)
        studio.vault_manager.reload_vault(force=False)
        studio._sync_vault_cache()
        studio.config.api_keys["gemini"] = "mock_key"

        studio.handle_chat("Jelaskan emisi kebakaran lahan gambut?")

        output = studio.console.file.getvalue()
        assert "Grounding:" in output
        assert "Critic:" in output
        assert "Tokens:" in output


class TestDualLayerCriticAudit:
    """Tests separating Layer 1 (Contract Conformance) and Layer 2 (Epistemic Rigor)."""

    def test_dod_checked_but_hallucinated_fails_epistemic_rigor(self) -> None:
        from scrpcy.core.contracts import TaskContract
        critic = DevilsAdvocate()
        contract = TaskContract(
            primary_goal="Analisis komparatif restorasi",
            min_words=50,
            max_words=200,
            definition_of_done=["Menyajikan parameter", "Menyajikan data komparasi"],
        )

        answer = (
            "Parameter komparasi mencakup laju penurunan emisi dan biaya implementasi teknologi pembasahan kembali [Page2023]. "
            "Karakteristik hidrologis gambut tropis menunjukkan bahwa elevasi muka air tanah perlu dipertahankan pada ambang batas -40 cm. "
            "Namun demikian, terdapat tantangan teknis struktural pada sekat kanal sekunder yang belum terselesaikan di lapangan riset [Smith2022]."
        )
        assert len(answer.split()) >= 45

        # Fact check reports low grounding & hallucination warning
        failing_fact_check = VerificationReport(
            grounding_score=0.45,
            total_claims=4,
            supported_claims=1,
            unsupported_claims=3,
            hallucination_warning="Klaim angka penurunan emisi tidak ditemukan pada dokumen vault.",
        )

        report = critic.critique_heuristic(
            query="Bandingkan metode restorasi",
            answer=answer,
            contract=contract,
            fact_check_verdict=failing_fact_check,
        )

        # Layer 1 succeeds, but Layer 2 fails
        assert report.contract_conformance is True
        assert report.epistemic_rigor is False
        assert report.passed is False
        assert report.score < 7
        assert any("grounding" in w.lower() or "halusinasi" in w.lower() for w in report.weaknesses)

    def test_word_count_tolerance_285_words_passes_conformance(self) -> None:
        from scrpcy.core.contracts import TaskContract
        critic = DevilsAdvocate()
        contract = TaskContract(
            primary_goal="Analisis restorasi",
            min_words=300,
            max_words=500,
            tolerance_pct=0.15,
        )

        # Build an answer of exactly 285 words (which is within [255, 575])
        body_words = ["emisi", "lahan", "gambut", "mengalami", "dinamika", "signifikan", "[Page2023]", "namun", "terdapat", "limitasi"]
        repeated = (body_words * 29)[:285]
        answer_285 = " ".join(repeated)
        assert len(answer_285.split()) == 285

        passing_fact_check = VerificationReport(
            grounding_score=0.90,
            total_claims=5,
            supported_claims=5,
            unsupported_claims=0,
        )

        report = critic.critique_heuristic(
            query="Analisis restorasi",
            answer=answer_285,
            contract=contract,
            fact_check_verdict=passing_fact_check,
        )

        assert report.contract_conformance is True
        assert report.epistemic_rigor is True
        assert report.passed is True
        assert report.score >= 7

    def test_word_count_below_tolerance_fails_conformance(self) -> None:
        from scrpcy.core.contracts import TaskContract
        critic = DevilsAdvocate()
        contract = TaskContract(
            primary_goal="Analisis mendalam",
            min_words=300,
            max_words=600,
            tolerance_pct=0.15,  # min effective: 255 words
        )

        # 150 words is far below 255 words
        short_answer = " ".join(["gambut", "terbakar", "emisi", "[Smith2023]", "namun", "ada", "kendala"] * 25)[:150]

        passing_fact_check = VerificationReport(
            grounding_score=0.95,
            total_claims=3,
            supported_claims=3,
            unsupported_claims=0,
        )

        report = critic.critique_heuristic(
            query="Analisis mendalam",
            answer=short_answer,
            contract=contract,
            fact_check_verdict=passing_fact_check,
        )

        assert report.contract_conformance is False
        assert report.passed is False
        assert any("batas minimum toleransi" in d for d in report.conformance_details)
