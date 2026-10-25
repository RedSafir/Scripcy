"""Tests for Turn 0 Official Website Prioritization and Unlimited Token Settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pytest

from scrpcy.core.config import ConfigManager, ScrpcyConfig
from scrpcy.core.contractor import ContractDecomposer, SubContract
from scrpcy.core.contracts import RubricCriterion, TaskContract, TaskContractExtractor
from scrpcy.core.critic import CritiqueReport
from scrpcy.core.engine import EvidenceChunk, LLMEngine
from scrpcy.core.vector_vault import HybridSearchResult


def test_unlimited_token_contract_properties() -> None:
    """TaskContract respects unlimited_tokens flag, removing upper ceiling constraints."""
    contract = TaskContract(
        archetype="DATA_FACTUAL",
        primary_goal="Statistik kebakaran hutan",
        in_scope=["Angka hektare"],
        out_of_scope=["Bab 2"],
        format_requirements="Tabel data",
        target_output_tokens=300,
        token_margin_pct=0.30,
        unlimited_tokens=True,
    )
    assert contract.unlimited_tokens is True
    assert contract.effective_min_tokens == 210
    assert contract.effective_max_tokens == 999_999
    assert contract.is_token_count_acceptable(250) is True
    assert contract.is_token_count_acceptable(5000) is True
    assert contract.is_token_count_acceptable(100) is False  # below min


def test_contractor_gatekeeper_allows_expansion_when_unlimited() -> None:
    """ContractDecomposer never rejects elaboration due to token limits in unlimited mode."""
    gatekeeper = ContractDecomposer()
    contract_limited = TaskContract(
        archetype="DATA_FACTUAL",
        primary_goal="Uji kuota",
        in_scope=["Data"],
        out_of_scope=["Bab 2"],
        format_requirements="Tabel",
        target_output_tokens=200,
        token_margin_pct=0.10,
        unlimited_tokens=False,
    )
    # 200 * 1.10 = 220
    is_vetoed, reason = gatekeeper._is_critique_out_of_contract(
        "uraikan lebih lanjut dan tambahkan pembahasan lengkap",
        contract_limited,
        draft_tokens=250,
    )
    assert is_vetoed is True
    assert "telah mencapai batas atas kuota kontrak" in reason

    # With unlimited_tokens=True
    contract_unlimited = TaskContract(
        archetype="DATA_FACTUAL",
        primary_goal="Uji kuota",
        in_scope=["Data"],
        out_of_scope=["Bab 2"],
        format_requirements="Tabel",
        target_output_tokens=200,
        token_margin_pct=0.10,
        unlimited_tokens=True,
    )
    is_vetoed_unl, _ = gatekeeper._is_critique_out_of_contract(
        "uraikan lebih lanjut dan tambahkan pembahasan lengkap",
        contract_unlimited,
        draft_tokens=250,
    )
    assert is_vetoed_unl is False


def test_detect_evidence_gap_prioritizes_official_web_for_factual_data() -> None:
    """Detect evidence gap flags missing official web primary sources for DATA_FACTUAL queries."""
    decomposer = ContractDecomposer()
    contract = TaskContract(
        archetype="DATA_FACTUAL",
        primary_goal="Data kebakaran hutan Riau 2023",
        in_scope=["Luas kebakaran", "Sumber KLHK"],
        out_of_scope=["Bab 2"],
        format_requirements="Tabel data",
        target_output_tokens=300,
    )

    # Scenario A: Vault only has academic peer-reviewed paper chunks (no official web)
    academic_chunk = EvidenceChunk(
        chunk_id="paper_1_c1",
        source="jurnal_kehutanan_2020",
        text="Studi ini meneliti karakteristik tanah gambut terhadap kebakaran hutan di Riau.",
        token_estimate=120,
    )
    has_gap, gap_reason, web_q, lit_q = decomposer.detect_evidence_gap(
        query="berapa luas kebakaran hutan riau tahun 2023",
        contract=contract,
        evidence=[academic_chunk],
    )
    assert has_gap is True
    assert "rujukan primer dari portal resmi" in gap_reason
    assert any("data resmi" in q or "portal resmi" in q for q in web_q)

    # Scenario B: Vault already has official web chunk from BPS / KLHK
    official_chunk = EvidenceChunk(
        chunk_id="web_1_c1",
        source="web_menlhk_go_id_si_sipongi",
        text="Rekapitulasi resmi KLHK mencatat luas kebakaran hutan di Riau tahun 2023 mencapai 1200 ha.",
        token_estimate=100,
    )
    has_gap_b, _, _, _ = decomposer.detect_evidence_gap(
        query="berapa luas kebakaran hutan riau tahun 2023",
        contract=contract,
        evidence=[official_chunk],
    )
    assert has_gap_b is False


def test_official_web_scoring_boost_in_evidence_synthesizer() -> None:
    """EvidenceSynthesizer boosts chunks from official web domains on factual questions."""
    from scrpcy.core.engine import EvidenceSynthesizer

    synth = EvidenceSynthesizer()
    query = "data resmi luas kebakaran hutan riau tahun 2023"

    doc_academic = (
        "Penelitian tentang kebakaran hutan di Riau menunjukkan faktor cuaca dan degradasi lahan "
        "mempengaruhi tingkat kebakaran. Berdasarkan analisis satelit, area terbakar bervariasi."
    )
    doc_official = (
        "Kementerian Lingkungan Hidup dan Kehutanan (KLHK) melalui SiPongi merilis data resmi "
        "luas kebakaran hutan di Riau tahun 2023 adalah 1.250 hektare."
    )

    chunks = synth.retrieve_evidence(
        query=query,
        documents=[doc_academic, doc_official],
        max_chunks=2,
    )
    assert len(chunks) >= 1
    # Check that chunks from official source get scored higher or are ranked at the top
    top_chunk = chunks[0]
    assert "klhk" in top_chunk.text.lower() or "1.250" in top_chunk.text.lower()


def test_settings_commands_for_unlimited_tokens_and_margin(tmp_path: Path) -> None:
    """Studio settings command toggles and configures unlimited tokens and margin of error."""
    from scrpcy.studio.commands import StudioCommandsMixin
    from rich.console import Console

    class DummyStudio(StudioCommandsMixin):
        def __init__(self, workspace_path: Path) -> None:
            self.config_manager = ConfigManager(config_path=workspace_path / "config.json")
            self.config = self.config_manager.config
            self.console = Console(force_terminal=False)
            self.scout = None
            self.engine = None

    studio = DummyStudio(tmp_path)
    assert studio.config.contract_unlimited_tokens is False

    # 1. Toggle unlimited tokens
    studio.cmd_settings("toggle unlimited-tokens")
    assert studio.config.contract_unlimited_tokens is True

    # 2. Toggle back off
    studio.cmd_settings("toggle unlimited-tokens")
    assert studio.config.contract_unlimited_tokens is False

    # 3. Explicit set-unlimited-tokens on
    studio.cmd_settings("set-unlimited-tokens on")
    assert studio.config.contract_unlimited_tokens is True

    # 4. set-token-margin with numeric percentage restores limited mode
    studio.cmd_settings("set-token-margin 25")
    assert studio.config.contract_token_margin_pct == 0.25
    assert studio.config.contract_unlimited_tokens is False

    # 5. set-token-margin unlimited enables unlimited mode
    studio.cmd_settings("set-token-margin unlimited")
    assert studio.config.contract_unlimited_tokens is True
