"""Unit tests for AcademicPolisher: context-preserving humanizer and anti-AI slop polisher."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
import pytest
from rich.console import Console

from scripcy.core.config import ScripcyConfig
from scripcy.core.engine import LLMEngine
from scripcy.core.polisher import AcademicPolisher, PolishResult
from scripcy.core.project import ProjectManager
from scripcy.core.workspace import WorkspaceManager
from scripcy.cli_studio import CLIStudio


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
        cfg = ScripcyConfig(workspace_path=str(temp_workspace.root_dir))

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

    def test_adversarial_masking_and_unmasking(self) -> None:
        polisher = AcademicPolisher()
        sample = (
            "Model Transformer [Vaswani2017] mencapai latensi 120 ms dengan akurasi 95.4% "
            "dan signifikansi statistik p < 0.05 menurut (Devlin et al., 2019) serta catatan [^1]."
        )
        masked, lookup = polisher.mask_sensitive_entities(sample)

        # Ensure placeholders exist
        assert "__CITE_" in masked
        assert "__STAT_" in masked
        assert "[Vaswani2017]" not in masked
        assert "120 ms" not in masked
        assert "95.4%" not in masked
        assert "p < 0.05" not in masked
        assert "(Devlin et al., 2019)" not in masked
        assert "[^1]" not in masked

        # Unmask and verify 100% verbatim restoration
        unmasked = polisher.unmask_sensitive_entities(masked, lookup)
        assert unmasked == sample

    def test_polish_heuristic_indonesian_cliches(self) -> None:
        polisher = AcademicPolisher()
        text = (
            "Secara keseluruhan, temuan ini sangat relevan. "
            "Hal ini membuktikan bahwa arsitektur neural meningkatkan throughput sebesar 25% [Devlin2019]. "
            "Menyelami lebih dalam terkait efisiensi, pendekatan ini memainkan peran krusial dalam komputasi."
        )
        cleaned = polisher.polish_heuristic(text)
        assert "Secara keseluruhan," not in cleaned
        assert "Hal ini membuktikan bahwa" not in cleaned
        assert "temuan ini mengindikasikan bahwa" in cleaned
        assert "Menyelami lebih dalam terkait" not in cleaned
        assert "menganalisis" in cleaned
        assert "memainkan peran krusial dalam" not in cleaned
        assert "menentukan" in cleaned
        assert "[Devlin2019]" in cleaned
        assert "25%" in cleaned

    def test_verify_integrity_numerical_metrics(self) -> None:
        polisher = AcademicPolisher()
        orig = "Eksperimen mencatat latensi 45 ms, akurasi 98.2%, dan pelepasan 3.4 gigaton emisi [Page2002]."
        # Missing multiple critical metrics
        corrupted = "Eksperimen mencatat latensi cepat, akurasi tinggi, dan pelepasan emisi [Page2002]."
        valid, msg = polisher.verify_integrity(orig, corrupted)
        assert valid is False
        assert "Numerical data points missing" in msg
