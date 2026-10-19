"""Token usage tracker for logging and reporting LLM token consumption.

Maintains timestamped JSON Lines logs globally (~/.scrpcy/token_usage.jsonl)
and per-project (<project_path>/.scrpcy/token_usage.jsonl).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

DEFAULT_GLOBAL_USAGE_PATH = Path.home() / ".scrpcy" / "token_usage.jsonl"


class TokenRecord(BaseModel):
    """An individual LLM inference token consumption record with timestamp."""

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of LLM execution.",
    )
    project_name: str | None = Field(default=None, description="Associated project name if inside workspace.")
    project_path: str | None = Field(default=None, description="Absolute filesystem path to project root.")
    model: str = Field(description="Model identifier invoked (e.g. openai/gpt-4o, 9router/claude-3-5-sonnet).")
    input_tokens: int = Field(ge=0, description="Number of prompt/input tokens consumed.")
    output_tokens: int = Field(ge=0, description="Number of completion/output tokens generated.")
    total_tokens: int = Field(ge=0, description="Total tokens (input + output).")
    skill: str | None = Field(default=None, description="Active research skill persona used.")


class ProjectTokenSummary(BaseModel):
    """Aggregated token statistics for a single project."""

    project_name: str
    project_path: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0
    last_active: str | None = None


class GlobalTokenSummary(BaseModel):
    """Global aggregated token consumption across all projects and models."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_requests: int = 0
    projects: dict[str, ProjectTokenSummary] = Field(default_factory=dict)
    models: dict[str, int] = Field(default_factory=dict)
    recent_logs: list[TokenRecord] = Field(default_factory=list)


class TokenTracker:
    """Manages appending and aggregating token consumption logs."""

    def __init__(self, global_log_path: Path | str | None = None) -> None:
        self.global_log_path = Path(global_log_path).expanduser().resolve() if global_log_path else DEFAULT_GLOBAL_USAGE_PATH

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        project_name: str | None = None,
        project_path: str | Path | None = None,
        skill: str | None = None,
        timestamp: str | None = None,
    ) -> TokenRecord:
        """Record token consumption event to global log and per-project log."""
        proj_path_str = str(Path(project_path).expanduser().resolve()) if project_path else None
        tot = max(0, input_tokens) + max(0, output_tokens)

        record = TokenRecord(
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            project_name=project_name,
            project_path=proj_path_str,
            model=model,
            input_tokens=max(0, input_tokens),
            output_tokens=max(0, output_tokens),
            total_tokens=tot,
            skill=skill,
        )

        record_line = record.model_dump_json() + "\n"

        # 1. Master global log in ~/.scrpcy/token_usage.jsonl
        try:
            self.global_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.global_log_path, "a", encoding="utf-8") as f:
                f.write(record_line)
        except OSError:
            pass

        # 2. Per-project log in <project_path>/.scrpcy/token_usage.jsonl
        if proj_path_str:
            try:
                proj_scrpcy_dir = Path(proj_path_str) / ".scrpcy"
                proj_scrpcy_dir.mkdir(parents=True, exist_ok=True)
                proj_log_file = proj_scrpcy_dir / "token_usage.jsonl"
                with open(proj_log_file, "a", encoding="utf-8") as f:
                    f.write(record_line)
            except OSError:
                pass

        return record

    def get_summary(self, max_recent: int = 10) -> GlobalTokenSummary:
        """Aggregate total token usage and breakdown per project from the global log."""
        summary = GlobalTokenSummary()
        if not self.global_log_path.is_file():
            return summary

        records: list[TokenRecord] = []
        try:
            with open(self.global_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        rec = TokenRecord.model_validate(data)
                        records.append(rec)
                    except (json.JSONDecodeError, Exception):
                        continue
        except OSError:
            return summary

        for rec in records:
            summary.total_input_tokens += rec.input_tokens
            summary.total_output_tokens += rec.output_tokens
            summary.total_tokens += rec.total_tokens
            summary.total_requests += 1

            # Model breakdown
            summary.models[rec.model] = summary.models.get(rec.model, 0) + rec.total_tokens

            # Project breakdown
            proj_key = rec.project_name or "(Standalone / No Project)"
            if proj_key not in summary.projects:
                summary.projects[proj_key] = ProjectTokenSummary(
                    project_name=proj_key,
                    project_path=rec.project_path,
                )
            p_sum = summary.projects[proj_key]
            p_sum.input_tokens += rec.input_tokens
            p_sum.output_tokens += rec.output_tokens
            p_sum.total_tokens += rec.total_tokens
            p_sum.request_count += 1
            p_sum.last_active = rec.timestamp
            if rec.project_path and not p_sum.project_path:
                p_sum.project_path = rec.project_path

        summary.recent_logs = records[-max_recent:] if records else []
        return summary

    @staticmethod
    def get_project_tokens(project_path: str | Path) -> ProjectTokenSummary:
        """Retrieve aggregated token usage directly from a project's local log."""
        path = Path(project_path).expanduser().resolve()
        log_file = path / ".scrpcy" / "token_usage.jsonl"
        summary = ProjectTokenSummary(
            project_name=path.name,
            project_path=str(path),
        )
        if not log_file.is_file():
            return summary

        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        rec = TokenRecord.model_validate(data)
                        summary.input_tokens += rec.input_tokens
                        summary.output_tokens += rec.output_tokens
                        summary.total_tokens += rec.total_tokens
                        summary.request_count += 1
                        summary.last_active = rec.timestamp
                        if rec.project_name:
                            summary.project_name = rec.project_name
                    except (json.JSONDecodeError, Exception):
                        continue
        except OSError:
            pass

        return summary
