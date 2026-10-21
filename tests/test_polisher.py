"""Unit tests for AcademicPolisher: context-preserving humanizer and anti-AI slop polisher."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
import pytest
from rich.console import Console

from scrpcy.core.config import ScrpcyConfig
from scrpcy.core.engine import LLMEngine
from scrpcy.core.polisher import AcademicPolisher, PolishResult
from scrpcy.core.project import ProjectManager
from scrpcy.core.workspace import WorkspaceManager
from scrpcy.cli_studio import CLIStudio


class TestAcademicPolisher:
    """Tests for citation extraction, burstiness calculation, heuristic de-smoothing, and assertion checks."""

    def test_extract_citations(self) -> None:
        polisher = AcademicPolisher()
        text = (
            "Model Transformer [Vaswani2017] memecahkan masalah dependensi sekuensial. "
            "Studi gambut [Page2002] menemukan pelepasan karbon masif, selaras dengan [1, 2]. "
            "Kunjungi [dokumentasi](https://example.com) untuk rincian."
        )
        cites = polisher.extract_citations(text)
        assert "[Vaswani2017]" in cites
        assert "[Page2002]" in cites
        assert "[1, 2]" in cites
        # Markdown link should not be classified as a citation
        assert "[dokumentasi]" not in cites

    def test_calculate_burstiness(self) -> None:
        polisher = AcademicPolisher()
        # Monotonous sentences (similar length)
        monotone = (
            "Kebakaran hutan gambut tropis menghasilkan emisi karbon masif ke atmosfer global. "
            "Pelepasan gas rumah kaca tersebut mempercepat laju perubahan iklim secara drastis. "
            "Partikel kabut asap membahayakan kesehatan populasi di berbagai kawasan terdekat."
        )
        # Bursty sentences (mix of short punchy and long compound)
        bursty = (
            "Emisi ini destruktif. "
            "Berdasarkan analisis biogeokimia lintas-dekade, kebakaran gambut tropis tidak hanya melepaskan gigaton karbon ke atmosfer, melainkan juga merusak struktur hidrologis bawah permukaan yang membutuhkan waktu pemulihan ratusan tahun. "
            "Risikonya nyata."
        )
        score_mono = polisher.calculate_burstiness(monotone)
        score_burst = polisher.calculate_burstiness(bursty)
        assert score_burst > score_mono

    def test_polish_heuristic_removes_slop_preserves_citations(self) -> None:
        polisher = AcademicPolisher()
        text = (
            "Penting untuk dicatat bahwa kebakaran lahan gambut memicu emisi karbon masif [Page2002]. "
            "Perlu dicatat bahwa partikel PM2.5 menembus batas negara dengan konsentrasi 450 ug/m3. "
            "Sebagai kesimpulan, restorasi hidrologis melalui rewetting terbukti sangat efektif [Smith2020]."
        )
        cleaned = polisher.polish_heuristic(text)
        # Slop phrases must be stripped
        assert "Penting untuk dicatat bahwa" not in cleaned
        assert "Perlu dicatat bahwa" not in cleaned
        assert "Sebagai kesimpulan," not in cleaned
        # Facts, numbers, and citations must remain intact
        assert "[Page2002]" in cleaned
        assert "[Smith2020]" in cleaned
        assert "450 ug/m3" in cleaned
        assert "restorasi hidrologis melalui rewetting" in cleaned

    def test_verify_integrity_passes_when_intact(self) -> None:
        polisher = AcademicPolisher()
        orig = "Kebakaran gambut tropis memicu pelepasan 2.5 gigaton karbon [Page2002] pada tingkat 95%."
        pol = "Kebakaran gambut tropis melepaskan 2.5 gigaton karbon [Page2002] dengan tingkat 95%."
        valid, msg = polisher.verify_integrity(orig, pol)
        assert valid is True
        assert "Integrity verified" in msg

    def test_verify_integrity_fails_when_citation_dropped(self) -> None:
        polisher = AcademicPolisher()
        orig = "Studi awal [Page2002] mendokumentasikan emisi karbon masif."
        # Drop [Page2002]
        corrupted = "Studi awal mendokumentasikan emisi karbon masif tanpa rujukan."
        valid, msg = polisher.verify_integrity(orig, corrupted)
        assert valid is False
        assert "Missing citation keys" in msg
        assert "[Page2002]" in msg

    def test_fallback_to_original_when_integrity_fails(self) -> None:
        polisher = AcademicPolisher()
        orig = "Analisis mendalam [Page2002] membuktikan bahwa 85% area gambut terdampak."
        # Simulate corrupt candidate dropping citation
        res = polisher.polish(orig)
        # Result must guarantee that citations are preserved
        assert "[Page2002]" in res.polished_text

    def test_polisher_integration_in_chat(
        self,
        temp_workspace: WorkspaceManager,
        tmp_path: Path,
    ) -> None:
        temp_workspace.save_to_cache(
            "peat_doc.md",
            "# Peatland Wildfire\nTropical peatland fires emit carbon and PM2.5 haze particles [Page2002].",
        )

        console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
        pm = ProjectManager(registry_file=tmp_path / "reg.json")
        proj = pm.create_project("thesis", temp_workspace.root_dir)
        cfg = ScrpcyConfig(workspace_path=str(temp_workspace.root_dir))

        from unittest.mock import MagicMock

        class MockChunk:
            def __init__(self, content: str) -> None:
                self.choices = [MagicMock(delta=MagicMock(content=content))]
                self.text = content

        async def mock_acompletion(**kwargs: Any):
            yield MockChunk(
                "Penting untuk dicatat bahwa kebakaran lahan gambut tropis melepaskan emisi karbon masif [Page2002]. "
                "Perlu dicatat bahwa fenomena ini berdampak signifikan terhadap iklim global."
            )

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

        studio.handle_chat("Jelaskan emisi kebakaran gambut?")

        output = studio.console.file.getvalue()
        # Verify that robotic AI phrases were de-smoothed
        assert "Penting untuk dicatat bahwa" not in output
        assert "Perlu dicatat bahwa" not in output
        # Verify that citation [Page2002] is safely intact
        assert "[Page2002]" in output
        # Telemetry remains intact
        assert "Grounding:" in output
        assert "Critic:" in output
