"""
Session Context Manager for Scripcy.
Lightweight working memory that stores verified facts, small contract summaries,
and baseline clean drafts instead of raw, token-heavy conversation history.
"""

from __future__ import annotations

import re
from typing import Any
from pydantic import BaseModel, Field


class SessionTurnRecord(BaseModel):
    """Encapsulates a compact, token-efficient memory record for one conversational turn."""
    turn_index: int
    user_query: str
    small_contracts_summary: list[str] = Field(default_factory=list)
    verified_facts: list[str] = Field(default_factory=list)
    clean_draft: str = ""  # Draft text without bibliography block
    citations_used: list[str] = Field(default_factory=list)


class SessionContextManager:
    """Manages session-level working memory and constructs targeted context prompts."""

    REVISION_PATTERN = re.compile(
        r"\b(perbaiki|revisi|koreksi|ubah|tambahkan|update|tulis\s+ulang|edit|kurang\s+detail|ganti\s+bagian|"
        r"refine|revise|rewrite|modify|correct|expand|shorten|elaborate|add\s+details|replace|change)\b",
        re.IGNORECASE,
    )

    BIBLIOGRAPHY_HEADER_PATTERN = re.compile(
        r"(?:^|\n)(?:📚\s*)?(?:Daftar\s+Pustaka|Referensi|References|Daftar\s+Referensi)\s*[:：]?",
        re.IGNORECASE,
    )

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.turns: list[SessionTurnRecord] = []

    def is_revision_query(self, query: str) -> bool:
        """Determines if the user's input requests a revision/refinement of the previous response."""
        return bool(self.REVISION_PATTERN.search(query.strip()))

    def strip_bibliography(self, response_text: str) -> str:
        """Strips trailing bibliography sections and noisy formatting to save tokens and isolate text."""
        match = self.BIBLIOGRAPHY_HEADER_PATTERN.search(response_text)
        if match:
            clean = response_text[:match.start()].strip()
            # Strip trailing decorative horizontal lines
            clean = re.sub(r"\n[-=_]{3,}\s*$", "", clean).strip()
            return clean
        return response_text.strip()

    def record_turn(
        self,
        query: str,
        full_response: str,
        small_contracts: list[str] | None = None,
        verified_facts: list[str] | None = None,
        citations_used: list[str] | None = None,
    ) -> None:
        """Records a completed turn into working memory, cleaning the response text."""
        clean_draft = self.strip_bibliography(full_response)
        turn_rec = SessionTurnRecord(
            turn_index=len(self.turns) + 1,
            user_query=query.strip(),
            small_contracts_summary=small_contracts or [],
            verified_facts=verified_facts or [],
            clean_draft=clean_draft,
            citations_used=citations_used or [],
        )
        self.turns.append(turn_rec)

    def get_last_clean_draft(self) -> str:
        """Retrieves the clean draft text from the most recent turn, if any."""
        if not self.turns:
            return ""
        return self.turns[-1].clean_draft

    def get_last_verified_facts(self) -> list[str]:
        """Retrieves the list of verified facts from the most recent turn."""
        if not self.turns:
            return []
        return self.turns[-1].verified_facts

    def build_context_directive(self, current_query: str) -> dict[str, Any]:
        """
        Builds a compact context payload for the Manager and Worker.
        - If revision: supplies the last clean draft + verified facts baseline.
        - If new topic: supplies only the past small contract objectives (high density, ~100 tokens).
        """
        is_rev = self.is_revision_query(current_query)

        if not self.turns:
            return {
                "is_revision": is_rev,
                "has_previous_context": False,
                "directive_text": "",
                "baseline_draft": "",
                "verified_facts": [],
            }

        last_turn = self.turns[-1]

        if is_rev:
            # Mode Revisi: Suplai Draf Sebelumnya + Fakta Terverifikasi
            facts_block = "\n".join(f"- {f}" for f in last_turn.verified_facts) if last_turn.verified_facts else "- (Gunakan fakta dari draf sebelumnya)"
            directive = (
                "[INSTRUKSI REVISI DRAF SEBELUMNYA]:\n"
                "Pengguna meminta penyempurnaan/perbaikan pada respon sebelumnya.\n"
                "Fokuskan perubahan HANYA pada bagian yang diminta pengguna dengan tetap mempertahankan fakta ilmiah yang valid.\n"
            )
            return {
                "is_revision": True,
                "has_previous_context": True,
                "directive_text": directive,
                "baseline_draft": last_turn.clean_draft,
                "verified_facts": last_turn.verified_facts,
            }
        else:
            # Mode Topik Lanjutan / Normal: Suplai ringkasan Small Contract lampau (sangat hemat token)
            past_summary_lines = []
            for t in self.turns[-3:]:  # Ambil maksimal 3 turn terakhir
                sc_str = ", ".join(t.small_contracts_summary) if t.small_contracts_summary else t.user_query
                past_summary_lines.append(f"Turn {t.turn_index}: {t.user_query} -> [Poin dibahas: {sc_str}]")

            summary_block = "\n".join(past_summary_lines)
            directive = (
                "[KONTEKS DISKUSI SEBELUMNYA]:\n"
                f"{summary_block}\n"
                "(Jangan ulangi penjelasan dasar topik di atas; langsung fokus ke pertanyaan baru).\n"
            )
            return {
                "is_revision": False,
                "has_previous_context": True,
                "directive_text": directive,
                "baseline_draft": "",
                "verified_facts": [],
            }
