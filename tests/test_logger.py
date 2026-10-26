"""Unit tests for StudioLogger runtime audit and event tracking."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripcy.core.engine import EvidenceChunk
from scripcy.core.logger import StudioLogger, get_studio_logger


class TestStudioLogger:
    """Tests for dual-format (human-readable text + structured JSONL) audit logging."""

    def test_logger_creates_logs_directory_and_files(self, tmp_path: Path) -> None:
        proj_dir = tmp_path / "research_project"
        proj_dir.mkdir()
        logger = StudioLogger(proj_dir)

        assert logger.logs_dir == proj_dir / "logs"
        assert logger.logs_dir.exists()
        assert logger.text_log_file == proj_dir / "logs" / "studio.log"
        assert logger.jsonl_log_file == proj_dir / "logs" / "events.jsonl"

    def test_logger_log_session_lifecycle(self, tmp_path: Path) -> None:
        logger = StudioLogger(tmp_path)
        logger.log_session_start(
            session_id="sess_123",
            session_name="Peatland Investigation",
            model="gemini/gemini-2.5-pro",
            skill="academic_writer",
        )
        logger.log_session_detach(session_id="sess_123", task_id="task_bg_99")
        logger.log_session_close(session_id="sess_123")

        # Verify studio.log
        text_content = logger.text_log_file.read_text(encoding="utf-8")
        assert "[SESSION_START] ID='sess_123'" in text_content
        assert "Peatland Investigation" in text_content
        assert "[SESSION_DETACH] Session 'sess_123' detached to background Task 'task_bg_99'" in text_content
        assert "[SESSION_CLOSE] Session 'sess_123' closed." in text_content

        # Verify events.jsonl
        lines = [json.loads(line) for line in logger.jsonl_log_file.read_text(encoding="utf-8").strip().splitlines()]
        assert len(lines) == 3

        start_event = lines[0]
        assert start_event["event"] == "SESSION_START"
        assert start_event["session_id"] == "sess_123"
        assert start_event["model"] == "gemini/gemini-2.5-pro"
        assert start_event["skill"] == "academic_writer"

        detach_event = lines[1]
        assert detach_event["event"] == "SESSION_DETACH"
        assert detach_event["task_id"] == "task_bg_99"

        close_event = lines[2]
        assert close_event["event"] == "SESSION_CLOSE"

    def test_logger_log_query_and_evidence(self, tmp_path: Path) -> None:
        logger = StudioLogger(tmp_path)
        logger.log_user_query(
            session_id="s1",
            query="Berapa emisi karbon kebakaran gambut 1997?",
            active_docs=["page2002.pdf", "siegert2001.pdf"],
        )

        chunks = [
            EvidenceChunk(chunk_id="c1", source="page2002.pdf", text="Carbon release", token_estimate=120),
            EvidenceChunk(chunk_id="c2", source="siegert2001.pdf", text="Satellite fire", token_estimate=150),
        ]
        logger.log_evidence_retrieved(
            session_id="s1",
            query="Berapa emisi karbon kebakaran gambut 1997?",
            chunks=chunks,
        )

        text_content = logger.text_log_file.read_text(encoding="utf-8")
        assert "[USER_PROMPT] Query='Berapa emisi karbon kebakaran gambut 1997?'" in text_content
        assert "page2002.pdf" in text_content
        assert "[EVIDENCE] Retrieved 2 chunk(s) from 2 source(s)" in text_content
        assert "~270 tokens" in text_content

        lines = [json.loads(line) for line in logger.jsonl_log_file.read_text(encoding="utf-8").strip().splitlines()]
        assert len(lines) == 2
        assert lines[0]["event"] == "USER_PROMPT"
        assert lines[0]["active_documents"] == ["page2002.pdf", "siegert2001.pdf"]
        assert lines[1]["event"] == "EVIDENCE_RETRIEVED"
        assert lines[1]["chunk_count"] == 2
        assert lines[1]["token_estimate"] == 270

    def test_logger_log_streaming_and_reviewers(self, tmp_path: Path) -> None:
        logger = StudioLogger(tmp_path)
        logger.log_llm_stream_start(session_id="s2", turn=1, model="gpt-4o", step_title="Menghubungi AI...")
        logger.log_llm_stream_finish(
            session_id="s2",
            turn=1,
            model="gpt-4o",
            output_tokens=350,
            elapsed_s=3.5,
            char_count=1400,
        )
        logger.log_polisher_result(
            session_id="s2",
            applied=True,
            duration_s=1.2,
            reason="Variasi kalimat diperbaiki",
            burst_before=2.1,
            burst_after=4.8,
            citations_preserved=True,
        )
        logger.log_fact_check_result(
            session_id="s2",
            grounding_score=0.92,
            total_claims=5,
            supported_claims=5,
            unsupported_claims=0,
            duration_s=1.5,
        )
        logger.log_critic_result(
            session_id="s2",
            score=9,
            passed=True,
            duration_s=2.0,
            recency_alert=None,
            needs_citation_mapping=False,
            weaknesses=[],
            summary="Sintesis sangat solid",
        )
        logger.log_citation_mapping(
            session_id="s2",
            stems=["peat fire", "carbon release"],
            found_papers=["page2002.pdf"],
            duration_s=0.4,
        )
        logger.log_bibliography_appended(
            session_id="s2",
            style="apa",
            entries_count=1,
            filenames=["page2002.pdf"],
        )
        logger.log_response_rendered(
            session_id="s2",
            total_input_tokens=1500,
            total_output_tokens=350,
            elapsed_total_s=8.6,
            model_label="gpt-4o",
            grounding_score=0.92,
            critic_score=9,
        )

        lines = [json.loads(line) for line in logger.jsonl_log_file.read_text(encoding="utf-8").strip().splitlines()]
        events = [rec["event"] for rec in lines]
        assert events == [
            "LLM_STREAM_START",
            "LLM_STREAM_FINISH",
            "POLISHER",
            "FACTCHECKER",
            "CRITIC_DEVILS_ADVOCATE",
            "CITATION_MAPPING",
            "BIBLIOGRAPHY",
            "RENDER_COMPLETE",
        ]

        # Verify stream finish metrics
        finish_rec = lines[1]
        assert finish_rec["tokens_per_second"] == 100.0
        assert finish_rec["output_tokens"] == 350

        # Verify polisher metrics
        polisher_rec = lines[2]
        assert polisher_rec["applied"] is True
        assert polisher_rec["burstiness_after"] == 4.8
        assert polisher_rec["citations_preserved"] is True

        # Verify fact checker metrics
        fc_rec = lines[3]
        assert fc_rec["grounding_score"] == 0.92
        assert fc_rec["supported_claims"] == 5

        # Verify critic metrics
        critic_rec = lines[4]
        assert critic_rec["score"] == 9
        assert critic_rec["passed"] is True

        # Verify render complete
        render_rec = lines[7]
        assert render_rec["total_tokens"] == 1850
        assert render_rec["critic_score"] == 9

    def test_logger_log_error_records_traceback(self, tmp_path: Path) -> None:
        logger = StudioLogger(tmp_path)
        try:
            raise ValueError("Koneksi API timeout 9Router")
        except ValueError as exc:
            logger.log_error(session_id="s_err", stage="LLM_INFERENCE", error=exc)

        text_content = logger.text_log_file.read_text(encoding="utf-8")
        assert "[ERROR] Stage='LLM_INFERENCE' Error='Koneksi API timeout 9Router'" in text_content
        assert "Traceback:" in text_content
        assert "ValueError: Koneksi API timeout 9Router" in text_content

        lines = [json.loads(line) for line in logger.jsonl_log_file.read_text(encoding="utf-8").strip().splitlines()]
        assert len(lines) == 1
        err_rec = lines[0]
        assert err_rec["event"] == "ERROR"
        assert err_rec["stage"] == "LLM_INFERENCE"
        assert err_rec["error"] == "Koneksi API timeout 9Router"
        assert "ValueError" in err_rec["traceback"]

    def test_get_studio_logger_singleton_and_project_switch(self, tmp_path: Path) -> None:
        proj_a = tmp_path / "proj_a"
        proj_b = tmp_path / "proj_b"
        proj_a.mkdir()
        proj_b.mkdir()

        logger_a1 = get_studio_logger(proj_a)
        logger_a2 = get_studio_logger(proj_a)
        logger_b = get_studio_logger(proj_b)

        assert logger_a1 is logger_a2
        assert logger_a1 is not logger_b

        # Switch project path on existing instance
        logger_a1.set_project_path(proj_b)
        assert logger_a1.logs_dir == proj_b / "logs"
        assert logger_a1.text_log_file == proj_b / "logs" / "studio.log"
