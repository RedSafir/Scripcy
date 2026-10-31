"""Comprehensive Audit & Event Logging for Scripcy Research Studio.

Maintains human-readable text logs (studio.log) and structured JSONL logs (events.jsonl)
in the active project's logs/ directory. Captures complete execution traces:
- User prompts and active vault documents
- Evidence chunks retrieved and token estimates
- LLM generation turns, token streaming, and latency
- AcademicPolisher execution, citation preservation, and burstiness
- FactChecker claim audit results and grounding scores
- Devil's Advocate adversarial scores, recency warnings, and weaknesses
- Automatic bibliography generation and file tags
- Background task transitions and detailed error tracebacks
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
import threading
import traceback
from typing import Any


class StudioLogger:
    """Thread-safe, dual-format (human text + JSONL) runtime auditor for Scripcy."""

    _instances: dict[str, StudioLogger] = {}
    _lock = threading.Lock()

    def __init__(self, project_path: str | Path | None = None) -> None:
        self.project_path: Path | None = Path(project_path).expanduser().resolve() if project_path else None
        self.logs_dir: Path | None = None
        self.text_log_file: Path | None = None
        self.jsonl_log_file: Path | None = None

        self._logger = logging.getLogger(f"scripcy.studio.{self.project_path.name if self.project_path else 'global'}")
        self._logger.setLevel(logging.DEBUG)
        # Prevent propagation to root logger to avoid polluting Rich console output
        self._logger.propagate = False

        self._init_handlers()

    def set_project_path(self, project_path: str | Path) -> None:
        """Switch active project path and reinitialize log destinations."""
        self.project_path = Path(project_path).expanduser().resolve()
        self._init_handlers()

    def _init_handlers(self) -> None:
        """Configure rotating text file and JSONL loggers."""
        # Clear existing handlers
        for h in list(self._logger.handlers):
            self._logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

        if not self.project_path:
            # Fallback to current working directory or user home
            base = Path.cwd() / ".scripcy"
            self.logs_dir = base / "logs"
        else:
            self.logs_dir = self.project_path / "logs"

        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            self.text_log_file = self.logs_dir / "studio.log"
            self.jsonl_log_file = self.logs_dir / "events.jsonl"

            # 1. Human-readable rotating text handler (5 MB max, 3 backups)
            text_handler = RotatingFileHandler(
                str(self.text_log_file),
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            text_formatter = logging.Formatter(
                fmt="[%(asctime)s.%(msecs)03d] [%(levelname)-7s] [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            text_handler.setFormatter(text_formatter)
            text_handler.setLevel(logging.DEBUG)
            self._logger.addHandler(text_handler)

        except Exception as exc:
            # If filesystem error, attach null handler so operations never throw
            null_handler = logging.NullHandler()
            self._logger.addHandler(null_handler)
            sys.stderr.write(f"[WARN] Failed to initialize file logger at {self.logs_dir}: {exc}\n")

    def _write_jsonl_event(self, event_type: str, session_id: str | None, payload: dict[str, Any]) -> None:
        """Append structured event JSON object to events.jsonl."""
        if not self.jsonl_log_file:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "session_id": session_id or "global",
            "project": self.project_path.name if self.project_path else None,
            **payload,
        }
        try:
            with open(self.jsonl_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # =========================================================================
    # HIGH-LEVEL AUDIT HOOKS
    # =========================================================================

    def log_session_start(self, session_id: str, session_name: str, model: str, skill: str) -> None:
        msg = f"[SESSION_START] ID='{session_id}' Name='{session_name}' Model='{model}' Skill='{skill}'"
        self._logger.info(msg)
        self._write_jsonl_event("SESSION_START", session_id, {
            "session_name": session_name,
            "model": model,
            "skill": skill,
        })

    def log_user_query(
        self,
        session_id: str,
        query: str,
        active_docs: list[str] | set[str] | None = None,
    ) -> None:
        docs_list = sorted(list(active_docs)) if active_docs else []
        msg = f"[USER_PROMPT] Query='{query}' ActiveDocs={docs_list}"
        self._logger.info(msg)
        self._write_jsonl_event("USER_PROMPT", session_id, {
            "query": query,
            "active_documents": docs_list,
        })

    def log_evidence_retrieved(
        self,
        session_id: str,
        query: str,
        chunks: list[Any],
    ) -> None:
        sources = sorted(list({getattr(c, "source", str(c)) for c in chunks}))
        token_sum = sum(getattr(c, "token_estimate", 0) for c in chunks)
        msg = f"[EVIDENCE] Retrieved {len(chunks)} chunk(s) from {len(sources)} source(s): {sources} (~{token_sum} tokens)"
        self._logger.info(msg)
        self._write_jsonl_event("EVIDENCE_RETRIEVED", session_id, {
            "chunk_count": len(chunks),
            "sources": sources,
            "token_estimate": token_sum,
        })

    def log_llm_stream_start(
        self,
        session_id: str,
        turn: int,
        model: str,
        step_title: str,
    ) -> None:
        msg = f"[LLM_STREAM_START] Turn={turn} Model='{model}' Step='{step_title}'"
        self._logger.info(msg)
        self._write_jsonl_event("LLM_STREAM_START", session_id, {
            "turn": turn,
            "model": model,
            "step_title": step_title,
        })

    def log_llm_stream_finish(
        self,
        session_id: str,
        turn: int,
        model: str,
        output_tokens: int,
        elapsed_s: float,
        char_count: int,
    ) -> None:
        tps = (output_tokens / elapsed_s) if elapsed_s > 0 else 0.0
        msg = f"[LLM_STREAM_FINISH] Turn={turn} Model='{model}' OutputTokens={output_tokens:,} Chars={char_count:,} Elapsed={elapsed_s:.2f}s ({tps:.1f} tok/s)"
        self._logger.info(msg)
        self._write_jsonl_event("LLM_STREAM_FINISH", session_id, {
            "turn": turn,
            "model": model,
            "output_tokens": output_tokens,
            "char_count": char_count,
            "elapsed_seconds": round(elapsed_s, 2),
            "tokens_per_second": round(tps, 1),
        })

    def log_polisher_result(
        self,
        session_id: str,
        applied: bool,
        duration_s: float,
        reason: str,
        burst_before: float,
        burst_after: float,
        citations_preserved: bool,
    ) -> None:
        status_str = "APPLIED" if applied else "SKIPPED/FALLBACK"
        msg = (
            f"[POLISHER] Status={status_str} Elapsed={duration_s:.2f}s "
            f"Burstiness={burst_before:.2f}->{burst_after:.2f} CitationsPreserved={citations_preserved} Reason='{reason}'"
        )
        self._logger.info(msg)
        self._write_jsonl_event("POLISHER", session_id, {
            "applied": applied,
            "duration_seconds": round(duration_s, 2),
            "burstiness_before": round(burst_before, 2),
            "burstiness_after": round(burst_after, 2),
            "citations_preserved": citations_preserved,
            "reason": reason,
        })

    def log_fact_check_result(
        self,
        session_id: str,
        grounding_score: float,
        total_claims: int,
        supported_claims: int,
        unsupported_claims: int,
        duration_s: float,
        hallucination_warning: str | None = None,
    ) -> None:
        pct = int(grounding_score * 100)
        msg = (
            f"[FACTCHECKER] Grounding={pct}% ({supported_claims}/{total_claims} supported, "
            f"{unsupported_claims} unsupported) Elapsed={duration_s:.2f}s"
        )
        if hallucination_warning:
            msg += f" Warning='{hallucination_warning}'"
        self._logger.info(msg)
        self._write_jsonl_event("FACTCHECKER", session_id, {
            "grounding_score": grounding_score,
            "total_claims": total_claims,
            "supported_claims": supported_claims,
            "unsupported_claims": unsupported_claims,
            "hallucination_warning": hallucination_warning,
            "duration_seconds": round(duration_s, 2),
        })

    def log_critic_result(
        self,
        session_id: str,
        score: int,
        passed: bool,
        duration_s: float,
        recency_alert: str | None = None,
        needs_citation_mapping: bool = False,
        weaknesses: list[str] | None = None,
        summary: str = "",
    ) -> None:
        msg = (
            f"[CRITIC_DEVILS_ADVOCATE] Score={score}/10 Passed={passed} NeedsCitationMapping={needs_citation_mapping} "
            f"Elapsed={duration_s:.2f}s Summary='{summary}'"
        )
        if recency_alert:
            msg += f" RecencyAlert='{recency_alert}'"
        if weaknesses:
            msg += f" Weaknesses={weaknesses}"
        self._logger.info(msg)
        self._write_jsonl_event("CRITIC_DEVILS_ADVOCATE", session_id, {
            "score": score,
            "passed": passed,
            "duration_seconds": round(duration_s, 2),
            "recency_alert": recency_alert,
            "needs_citation_mapping": needs_citation_mapping,
            "weaknesses": weaknesses or [],
            "summary": summary,
        })

    def log_citation_mapping(
        self,
        session_id: str,
        stems: list[str],
        found_papers: list[str],
        duration_s: float,
    ) -> None:
        msg = f"[CITATION_MAPPING] Stems={stems} Found={len(found_papers)} papers ({found_papers}) Elapsed={duration_s:.2f}s"
        self._logger.info(msg)
        self._write_jsonl_event("CITATION_MAPPING", session_id, {
            "stems": stems,
            "found_papers": found_papers,
            "duration_seconds": round(duration_s, 2),
        })

    def log_bibliography_appended(
        self,
        session_id: str,
        style: str,
        entries_count: int,
        filenames: list[str],
    ) -> None:
        msg = f"[BIBLIOGRAPHY] Appended {entries_count} {style.upper()} reference(s) from files: {filenames}"
        self._logger.info(msg)
        self._write_jsonl_event("BIBLIOGRAPHY", session_id, {
            "style": style,
            "entries_count": entries_count,
            "filenames": filenames,
        })

    def log_response_rendered(
        self,
        session_id: str,
        total_input_tokens: int,
        total_output_tokens: int,
        elapsed_total_s: float,
        model_label: str,
        grounding_score: float,
        critic_score: int,
    ) -> None:
        msg = (
            f"[RENDER_COMPLETE] Model='{model_label}' Tokens={total_input_tokens:,} in · "
            f"{total_output_tokens:,} out · {total_input_tokens + total_output_tokens:,} total "
            f"Grounding={int(grounding_score * 100)}% Critic={critic_score}/10 TotalElapsed={elapsed_total_s:.2f}s"
        )
        self._logger.info(msg)
        self._write_jsonl_event("RENDER_COMPLETE", session_id, {
            "model": model_label,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "grounding_score": grounding_score,
            "critic_score": critic_score,
            "total_elapsed_seconds": round(elapsed_total_s, 2),
        })

    def log_task_contract(
        self,
        session_id: str,
        archetype: str,
        primary_goal: str,
        in_scope: list[str],
        out_of_scope: list[str],
        search_channels: list[str],
        duration_s: float,
        effort_level: str = "medium",
        min_sources: int = 1,
        max_sources: int = 3,
        **kwargs: Any,
    ) -> None:
        msg = (
            f"[TASK_CONTRACT] Archetype={archetype} Effort={effort_level} MinSources={min_sources} Goal='{primary_goal}' "
            f"InScope={in_scope[:3]} OutOfScope={out_of_scope[:2]} Elapsed={duration_s:.2f}s"
        )
        self._logger.info(msg)
        payload = {
            "archetype": archetype,
            "primary_goal": primary_goal,
            "effort_level": effort_level,
            "min_sources": min_sources,
            "max_sources": max_sources,
            "in_scope": in_scope,
            "out_of_scope": out_of_scope,
            "search_channels": search_channels,
            "duration_seconds": round(duration_s, 2),
        }
        self._write_jsonl_event("TASK_CONTRACT", session_id, payload)

    def log_subagent_orchestration(
        self,
        session_id: str,
        roles_executed: list[str],
        facts_extracted_count: int,
        new_stems_count: int,
        input_tokens: int,
        output_tokens: int,
        duration_s: float,
        channels_attempted: list[str] | None = None,
    ) -> None:
        msg = (
            f"[SUBAGENTS] Roles={roles_executed} Facts={facts_extracted_count} "
            f"NewDocs={new_stems_count} Tokens={input_tokens}in/{output_tokens}out "
            f"Channels={channels_attempted or []} Elapsed={duration_s:.2f}s"
        )
        self._logger.info(msg)
        self._write_jsonl_event("SUBAGENTS", session_id, {
            "roles_executed": roles_executed,
            "facts_extracted_count": facts_extracted_count,
            "new_stems_count": new_stems_count,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "channels_attempted": channels_attempted or [],
            "duration_seconds": round(duration_s, 2),
        })

    def log_contractor_adjudication(
        self,
        session_id: str,
        override_passed: bool,
        override_score: int,
        accepted_revisions: list[str],
        rejected_critiques: list[str],
        search_exhausted: bool,
        channels_summary: list[str],
    ) -> None:
        msg = (
            f"[CONTRACTOR_GATEKEEPER] OverridePassed={override_passed} Score={override_score} "
            f"Accepted={len(accepted_revisions)} Vetoed={len(rejected_critiques)} "
            f"SearchExhausted={search_exhausted} Channels={channels_summary}"
        )
        self._logger.info(msg)
        self._write_jsonl_event("CONTRACTOR_GATEKEEPER", session_id, {
            "override_passed": override_passed,
            "override_score": override_score,
            "accepted_revisions": accepted_revisions,
            "rejected_critiques": rejected_critiques,
            "search_exhausted": search_exhausted,
            "channels_summary": channels_summary,
        })

    def log_crusher_compression(
        self,
        session_id: str,
        stage: str,
        original_chars: int,
        crushed_chars: int,
        savings_pct: float,
    ) -> None:
        msg = (
            f"[SMARTCRUSHER] Stage='{stage}' Chars={original_chars}->{crushed_chars} "
            f"Saved={savings_pct:.1f}%"
        )
        self._logger.info(msg)
        self._write_jsonl_event("SMARTCRUSHER", session_id, {
            "stage": stage,
            "original_chars": original_chars,
            "crushed_chars": crushed_chars,
            "savings_pct": round(savings_pct, 1),
        })

    def log_internal_tool(
        self,
        session_id: str,
        action: str,
        target: str,
        turn: int,
        output_length: int,
    ) -> None:
        msg = f"[INTERNAL_TOOL] Action='{action}' Target='{target}' Turn={turn} OutputChars={output_length}"
        self._logger.info(msg)
        self._write_jsonl_event("INTERNAL_TOOL", session_id, {
            "action": action,
            "target": target,
            "turn": turn,
            "output_length": output_length,
        })

    def log_literature_scout(
        self,
        session_id: str,
        query: str,
        candidates_count: int,
        ingested_count: int,
        ingested_stems: list[str],
        duration_s: float,
    ) -> None:
        msg = (
            f"[LITERATURE_SCOUT] Query='{query}' Candidates={candidates_count} "
            f"Ingested={ingested_count} Stems={ingested_stems} Elapsed={duration_s:.2f}s"
        )
        self._logger.info(msg)
        self._write_jsonl_event("LITERATURE_SCOUT", session_id, {
            "query": query,
            "candidates_count": candidates_count,
            "ingested_count": ingested_count,
            "ingested_stems": ingested_stems,
            "duration_seconds": round(duration_s, 2),
        })

    def log_web_scout(
        self,
        session_id: str,
        query: str,
        hits_count: int,
        official_count: int,
        ingested_stem: str | None,
        duration_s: float,
    ) -> None:
        msg = (
            f"[WEB_SCOUT] Query='{query}' Hits={hits_count} OfficialHits={official_count} "
            f"Ingested='{ingested_stem or ''}' Elapsed={duration_s:.2f}s"
        )
        self._logger.info(msg)
        self._write_jsonl_event("WEB_SCOUT", session_id, {
            "query": query,
            "hits_count": hits_count,
            "official_count": official_count,
            "ingested_stem": ingested_stem,
            "duration_seconds": round(duration_s, 2),
        })

    def log_vector_vault_search(
        self,
        session_id: str,
        query: str,
        hits_count: int,
        top_score: float,
        duration_s: float,
    ) -> None:
        msg = f"[VECTOR_VAULT] Query='{query}' Hits={hits_count} TopRRF={top_score:.3f} Elapsed={duration_s:.2f}s"
        self._logger.info(msg)
        self._write_jsonl_event("VECTOR_VAULT", session_id, {
            "query": query,
            "hits_count": hits_count,
            "top_score": round(top_score, 3),
            "duration_seconds": round(duration_s, 2),
        })

    def log_session_detach(self, session_id: str, task_id: str) -> None:
        msg = f"[SESSION_DETACH] Session '{session_id}' detached to background Task '{task_id}'"
        self._logger.info(msg)
        self._write_jsonl_event("SESSION_DETACH", session_id, {"task_id": task_id})

    def log_session_close(self, session_id: str) -> None:
        msg = f"[SESSION_CLOSE] Session '{session_id}' closed."
        self._logger.info(msg)
        self._write_jsonl_event("SESSION_CLOSE", session_id, {})

    def log_error(
        self,
        session_id: str | None,
        stage: str,
        error: BaseException | str,
        traceback_str: str | None = None,
    ) -> None:
        tb = traceback_str or (traceback.format_exc() if isinstance(error, BaseException) else None)
        err_msg = str(error)
        msg = f"[ERROR] Stage='{stage}' Error='{err_msg}'"
        if tb and "NoneType: None" not in tb:
            msg += f"\nTraceback:\n{tb}"
        self._logger.error(msg)
        self._write_jsonl_event("ERROR", session_id, {
            "stage": stage,
            "error": err_msg,
            "traceback": tb,
        })


def get_studio_logger(project_path: str | Path | None = None) -> StudioLogger:
    """Retrieve or create singleton StudioLogger instance for the given project path."""
    key = str(Path(project_path).expanduser().resolve()) if project_path else "__global__"
    with StudioLogger._lock:
        if key not in StudioLogger._instances:
            StudioLogger._instances[key] = StudioLogger(project_path)
        return StudioLogger._instances[key]
