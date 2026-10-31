"""Unit tests for context-driven min_sources, effort tiers, and user confirmation protocol."""

import pytest
from scripcy.core.config import ScripcyConfig, EffortTierConfig
from scripcy.core.contracts import TaskContract, TaskContractExtractor
from scripcy.core.contractor import ContractDecomposer
from scripcy.core.critic import DevilsAdvocate
from scripcy.core.orchestrator import SubagentOrchestrator
from scripcy.core.engine import EvidenceChunk


def test_effort_tier_config() -> None:
    """Verify ScripcyConfig defines low, medium, and extra effort tiers."""
    cfg = ScripcyConfig()
    assert cfg.default_effort_tier == "medium"
    assert cfg.effort_tier_low.min_sources == 0
    assert cfg.effort_tier_low.max_sources == 1
    assert cfg.effort_tier_medium.min_sources == 1
    assert cfg.effort_tier_medium.max_sources == 3
    assert cfg.effort_tier_extra.min_sources == 3
    assert cfg.effort_tier_extra.max_sources == 7

    # Token limits removed
    assert not hasattr(cfg, "contract_token_margin_pct")
    assert not hasattr(cfg, "contract_unlimited_tokens")


def test_context_driven_source_extraction() -> None:
    """Verify that different query archetypes dynamically resolve appropriate min_sources and effort."""
    extractor = TaskContractExtractor()

    # 1. Purely deductive / formatting / stylistic -> effort: low, min_sources: 0
    c_deductive = extractor.extract_heuristic("tolong perbaiki tata bahasa paragraf ini tanpa jurnal")
    assert c_deductive.effort_level == "low"
    assert c_deductive.min_sources == 0
    assert c_deductive.max_sources == 1

    # 2. Focused QA / Factual -> effort: medium, min_sources: 1
    c_qa = extractor.extract_heuristic("berapa luas tutupan hutan gambut di Riau tahun 2023?")
    assert c_qa.effort_level == "medium"
    assert c_qa.min_sources == 1

    # 3. Comparison -> effort: medium, min_sources: 2
    c_comp = extractor.extract_heuristic("bandingkan efektivitas sekat kanal kayu vs sekat kanal beton")
    assert c_comp.effort_level == "medium"
    assert c_comp.min_sources == 2

    # 4. Comprehensive Literature Review -> effort: extra, min_sources: 3
    c_lit = extractor.extract_heuristic("buatkan tinjauan pustaka bab 2 komprehensif restorasi hidrologi gambut")
    assert c_lit.effort_level == "extra"
    assert c_lit.min_sources >= 3


def test_explicit_min_source_extraction() -> None:
    """Verify regex extraction when user explicitly requests a specific minimum number of sources."""
    extractor = TaskContractExtractor()
    c = extractor.extract_heuristic("Bandingkan minimal 4 paper tentang arsitektur Mixture of Experts")
    assert c.min_sources == 4
    assert c.effort_level == "extra"
    assert c.max_sources >= 6


def test_contractor_gap_detection_on_sources() -> None:
    """Verify Contractor detects gap when distinct local sources < contract.min_sources."""
    contractor = ContractDecomposer()

    # Evidence with only 1 distinct source stem
    single_source_ev = [
        EvidenceChunk(
            chunk_id="c1",
            text="DeepSeek MoE menggunakan fine-grained expert segmentation 2024.",
            source="deepseek_2024",
        ),
        EvidenceChunk(
            chunk_id="c2",
            text="Efisiensi komputasi meningkat drastis pada arsitektur MoE 2024.",
            source="deepseek_2024",
        ),
    ]

    contract_need_3 = TaskContract(
        archetype="COMPARISON",
        primary_goal="Bandingkan 3 model MoE",
        effort_level="extra",
        min_sources=3,
        max_search_queries=4,
    )

    has_gap, reason, web_q, lit_q = contractor.detect_evidence_gap(
        query="Bandingkan 3 model MoE",
        contract=contract_need_3,
        evidence=single_source_ev,
    )

    assert has_gap is True
    assert "minimal 3 sumber" in reason
    assert len(lit_q) > 0
    assert len(lit_q) <= contract_need_3.max_search_queries


def test_contractor_no_gap_for_zero_source_deductive() -> None:
    """Verify Contractor does not trigger external gap for zero-source low-effort queries."""
    contractor = ContractDecomposer()

    contract_zero = TaskContract(
        archetype="FOCUSED_QA",
        primary_goal="Perbaiki kalimat deduktif",
        effort_level="low",
        min_sources=0,
    )

    has_gap, reason, web_q, lit_q = contractor.detect_evidence_gap(
        query="Perbaiki tata bahasa kalimat deduktif",
        contract=contract_zero,
        evidence=[],
    )

    assert has_gap is False
    assert web_q == []
    assert lit_q == []


def test_critic_source_evaluation() -> None:
    """Verify Devil's Advocate / Critic checks distinct sources cited in answer."""
    critic = DevilsAdvocate()
    contract_2_sources = TaskContract(
        archetype="COMPARISON",
        primary_goal="Bandingkan 2 metode",
        min_sources=2,
    )

    # Answer citing only 1 source
    answer_1_source = (
        "Analisis komparatif menunjukkan bahwa metode A lebih cepat 30% [PenulisA, 2023]. "
        "Selain itu, metode A memiliki konsumsi memori yang lebih efisien."
    )
    report_1 = critic.critique_heuristic(
        query="Bandingkan 2 metode",
        answer=answer_1_source,
        contract=contract_2_sources,
    )
    assert any("belum memenuhi target kontrak (minimal 2 sumber)" in d for d in report_1.conformance_details)

    # Answer citing 2 distinct sources
    answer_2_sources = (
        "Analisis komparatif menunjukkan bahwa metode A lebih cepat 30% [PenulisA, 2023]. "
        "Di sisi lain, metode B unggul dalam hal akurasi pada dataset besar [PenulisB, 2024]."
    )
    report_2 = critic.critique_heuristic(
        query="Bandingkan 2 metode",
        answer=answer_2_sources,
        contract=contract_2_sources,
    )
    assert not any("belum memenuhi target kontrak (minimal 2 sumber)" in d for d in report_2.conformance_details)


@pytest.mark.asyncio
async def test_orchestrator_source_exhaustion_confirmation() -> None:
    """Verify SubagentOrchestrator invokes confirmation callback when sources < min_sources."""
    orchestrator = SubagentOrchestrator()
    contract = TaskContract(
        archetype="COMPARISON",
        primary_goal="Uji konfirmasi sumber",
        min_sources=3,
        effort_level="medium",
    )

    confirmed_payloads = []

    def fake_confirm(info: dict) -> str:
        confirmed_payloads.append(info)
        return "continue"

    # Evidence with only 1 source and autonomous scout disabled
    ev = [
        EvidenceChunk(
            chunk_id="chunk1",
            text="Data empiris tentang sekat kanal gambut 2023.",
            source="sumber_tunggal_2023",
        )
    ]

    orch_res = await orchestrator.orchestrate(
        query="Uji konfirmasi sumber",
        contract=contract,
        evidence=ev,
        enable_autonomous_scout=False,
        on_source_exhausted_confirmation=fake_confirm,
    )

    assert len(confirmed_payloads) == 1
    assert confirmed_payloads[0]["found_count"] == 1
    assert confirmed_payloads[0]["min_sources"] == 3
    assert contract.allow_honest_gap is True
    assert any("Hanya 1 dari target minimal 3 sumber" in m or "hanya 1 dari target minimal 3 sumber" in m for m in orch_res.missing_notes)


def test_user_configurable_effort_tiers(tmp_path) -> None:
    """Verify user can configure custom min/max journals for each effort tier via ConfigManager."""
    from scripcy.core.config import ConfigManager

    cfg_file = tmp_path / "config.json"
    mgr = ConfigManager(config_path=cfg_file)

    # Initial defaults
    assert mgr.config.effort_tier_low.min_sources == 0
    assert mgr.config.effort_tier_low.max_sources == 1
    assert mgr.config.effort_tier_extra.min_sources == 3
    assert mgr.config.effort_tier_extra.max_sources == 7

    # User modifies low effort tier: min 0, max 2
    mgr.update_effort_tier("low", min_sources=0, max_sources=2)
    assert mgr.config.effort_tier_low.min_sources == 0
    assert mgr.config.effort_tier_low.max_sources == 2

    # User modifies extra effort tier: min 5, max 10
    mgr.update_effort_tier("extra", min_sources=5, max_sources=10)
    assert mgr.config.effort_tier_extra.min_sources == 5
    assert mgr.config.effort_tier_extra.max_sources == 10

    # Verify persistence by reloading from disk
    mgr_reloaded = ConfigManager(config_path=cfg_file)
    assert mgr_reloaded.config.effort_tier_low.max_sources == 2
    assert mgr_reloaded.config.effort_tier_extra.min_sources == 5
    assert mgr_reloaded.config.effort_tier_extra.max_sources == 10


def test_session_runtime_pending_contract_preservation() -> None:
    """Verify SessionRuntime preserves pending contract payload on failure and clears on success."""
    from scripcy.core.chats import SessionRuntime

    runtime = SessionRuntime(session_id="test_session")
    assert runtime.get_pending_contract() is None

    contract = TaskContract(
        archetype="FOCUSED_QA",
        primary_goal="Riset deep learning 5-10 tahun",
        effort_level="medium",
        min_sources=1,
    )

    payload = {
        "query": "carikan aku penelitian terbaru deep learning",
        "current_query": "sintesis query",
        "contract": contract,
        "current_docs": ["cache/doc1.md", "cache/doc2.md"],
        "verified_facts": [{"claim": "DeepSeek MoE efisien", "source": "doc1"}],
        "model_name": "9router/ag/gemini-3.8-flash-medium",
        "turn": 0,
    }

    # Store on streaming failure
    runtime.set_pending_contract(payload)
    cached = runtime.get_pending_contract()
    assert cached is not None
    assert cached["contract"].primary_goal == "Riset deep learning 5-10 tahun"
    assert len(cached["current_docs"]) == 2
    assert len(cached["verified_facts"]) == 1

    # Clear after successful run
    runtime.clear_pending_contract()
    assert runtime.get_pending_contract() is None


def test_dashboard_tier_settings_menu_structure() -> None:
    """Verify Dashboard settings hub uses Effort Tiers and does not expose agent turns/refinement/tokens."""
    from scripcy.studio.dashboard import StudioDashboardMixin

    # Verify method presence
    assert hasattr(StudioDashboardMixin, "_run_interactive_tier_settings")
    assert hasattr(StudioDashboardMixin, "_configure_single_tier")
    assert hasattr(StudioDashboardMixin, "_run_interactive_agent_limits_settings")


def test_cmd_settings_shows_tiers_and_no_turns_or_tokens(capsys) -> None:
    """Verify /settings output displays effort tier journal quotas without agent turns or token ceilings."""
    from scripcy.core.config import ConfigManager
    from scripcy.studio.commands import StudioCommandsMixin
    from rich.console import Console
    import io

    class DummyStudio(StudioCommandsMixin):
        def __init__(self):
            self.config_manager = ConfigManager()
            self.config = self.config_manager.config
            self.console = Console(file=io.StringIO(), color_system=None)

    studio = DummyStudio()
    studio.cmd_settings("show")
    output = studio.console.file.getvalue()

    # Must contain Effort Tiers and journal bounds
    assert "Low Effort Tier" in output
    assert "Medium Effort Tier" in output
    assert "Extra Effort Tier" in output

    # Must not contain legacy agent turn ceiling or refinement ceiling
    assert "Max Agent Turns" not in output
    assert "Refinement Rounds" not in output
    assert "Plafon Token" not in output


def test_user_extra_tier_customization_reflects_in_contract_min_sources(tmp_path) -> None:
    """Verify that when user changes EXTRA tier to min 4 sources, contract extraction uses 4 (not 3)."""
    from scripcy.core.config import ConfigManager
    from scripcy.core.engine import LLMEngine

    cfg_file = tmp_path / "config.json"
    mgr = ConfigManager(config_path=cfg_file)

    # User modifies extra tier: min 4, max 10 (as seen in user screenshot)
    mgr.update_effort_tier("extra", min_sources=4, max_sources=10)

    engine = LLMEngine(config=mgr.config)
    extractor = TaskContractExtractor(engine=engine)

    # 1. Heuristic extraction of latest research query
    c_heuristic = extractor.extract_heuristic("carikan aku penelitian terbaru YOLOS")
    assert c_heuristic.effort_level == "extra"
    assert c_heuristic.min_sources == 4
    assert c_heuristic.max_sources >= 10

    # 2. LLM completion fallback / extraction simulation where model suggested 3 or extra
    def mock_llm_completion(sys_p: str, q: str) -> dict:
        return {
            "archetype": "LITERATURE_SYNTHESIS",
            "effort_level": "extra",
            "primary_goal": "Menyajikan sintesis penelitian terbaru seputar YOLOS",
            "in_scope": ["Arsitektur YOLOS", "Hasil benchmark"],
            "out_of_scope": ["YOLOv5"],
            "format_requirements": "Format berdensitas tinggi, sitasi APA.",
            "min_words": 350,
            "max_words": 800,
        }

    extractor_llm = TaskContractExtractor(engine=engine, llm_completion_fn=mock_llm_completion)
    c_llm = extractor_llm.extract("carikan aku penelitian terbaru YOLOS")
    assert c_llm.effort_level == "extra"
    # Even if LLM omitted or guessed, it MUST respect the user's setting of 4
    assert c_llm.min_sources == 4
    assert c_llm.max_sources == 10



