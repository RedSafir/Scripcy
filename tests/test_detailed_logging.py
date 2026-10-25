"""Unit tests for detailed logging across all Scrpcy core and studio components."""

from __future__ import annotations

import json
from pathlib import Path

from scrpcy.core.logger import StudioLogger


def test_studio_logger_records_all_extended_component_events(tmp_path: Path) -> None:
    logger = StudioLogger(project_path=tmp_path)
    session_id = "test_audit_session"

    # 1. Task Contract
    logger.log_task_contract(
        session_id=session_id,
        archetype="DATA_FACTUAL",
        primary_goal="Menyajikan data karhutla 5 tahun terakhir",
        target_tokens=350,
        token_margin_pct=0.30,
        effective_min_tokens=245,
        effective_max_tokens=455,
        in_scope=["Luas terbakar tahunan", "Provinsi utama"],
        out_of_scope=["Teori umum", "Debat birokrasi"],
        search_channels=["vault_lokal", "web_resmi"],
        duration_s=1.25,
    )

    # 2. Subagent Orchestration
    logger.log_subagent_orchestration(
        session_id=session_id,
        roles_executed=["VaultExtractor", "WebScout", "GapExtractor", "ScopeAuditor"],
        facts_extracted_count=7,
        new_stems_count=2,
        input_tokens=1949,
        output_tokens=320,
        duration_s=3.45,
        channels_attempted=["vault_lokal", "web_resmi"],
    )

    # 3. Contractor Adjudication
    logger.log_contractor_adjudication(
        session_id=session_id,
        override_passed=True,
        override_score=9,
        accepted_revisions=[],
        rejected_critiques=["Menuntut pembahasan Bab 2", "Menuntut pencarian lanjutan"],
        search_exhausted=True,
        channels_summary=["Vault Lokal", "Portal Web Resmi"],
    )

    # 4. SmartCrusher Compression
    logger.log_crusher_compression(
        session_id=session_id,
        stage="evidence_chunks",
        original_chars=4500,
        crushed_chars=2850,
        savings_pct=36.7,
    )

    # 5. Internal Tool
    logger.log_internal_tool(
        session_id=session_id,
        action="Membaca",
        target="sipongi2023",
        turn=1,
        output_length=1200,
    )

    # 6. Literature Scout
    logger.log_literature_scout(
        session_id=session_id,
        query="karhutla indonesia",
        candidates_count=35,
        ingested_count=5,
        ingested_stems=["stem1", "stem2"],
        duration_s=8.5,
    )

    # 7. Web Scout
    logger.log_web_scout(
        session_id=session_id,
        query="data karhutla sipongi",
        hits_count=4,
        official_count=4,
        ingested_stem="sipongi_2023",
        duration_s=2.1,
    )

    # 8. Vector Vault
    logger.log_vector_vault_search(
        session_id=session_id,
        query="luas karhutla",
        hits_count=5,
        top_score=0.032,
        duration_s=0.45,
    )

    # Read events.jsonl
    events_file = tmp_path / "logs" / "events.jsonl"
    assert events_file.exists()
    lines = events_file.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines]

    event_types = [r["event"] for r in records]
    assert "TASK_CONTRACT" in event_types
    assert "SUBAGENTS" in event_types
    assert "CONTRACTOR_GATEKEEPER" in event_types
    assert "SMARTCRUSHER" in event_types
    assert "INTERNAL_TOOL" in event_types
    assert "LITERATURE_SCOUT" in event_types
    assert "WEB_SCOUT" in event_types
    assert "VECTOR_VAULT" in event_types

    # Read studio.log
    text_log_file = tmp_path / "logs" / "studio.log"
    assert text_log_file.exists()
    log_content = text_log_file.read_text(encoding="utf-8")
    assert "[TASK_CONTRACT] Archetype=DATA_FACTUAL" in log_content
    assert "[SUBAGENTS] Roles=['VaultExtractor', 'WebScout', 'GapExtractor', 'ScopeAuditor']" in log_content
    assert "[CONTRACTOR_GATEKEEPER] OverridePassed=True" in log_content
    assert "[SMARTCRUSHER] Stage='evidence_chunks'" in log_content
    assert "[INTERNAL_TOOL] Action='Membaca'" in log_content
    assert "[LITERATURE_SCOUT] Query='karhutla indonesia'" in log_content
    assert "[WEB_SCOUT] Query='data karhutla sipongi'" in log_content
    assert "[VECTOR_VAULT] Query='luas karhutla'" in log_content
