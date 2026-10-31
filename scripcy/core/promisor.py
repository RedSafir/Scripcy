"""Promisor: AI Intent & Spam Gatekeeper for Scripcy.

Promisor intercepts user input before invoking LLM pipelines.
- Deterministic slash commands (/back, /cancel, /vault, etc.) are routed to CLI commands.
- Non-slash natural language inputs are screened for empty/spam/gibberish or fast polite greetings,
  and then routed directly to AI agent pipelines (Contractor, Manager, Examiner).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


@dataclass
class PromisorDecision:
    """Decision output produced by the Promisor heuristic & AI entry-point dispatcher."""

    is_system_tool: bool
    action: str  # e.g. "ai_chat", "casual.reply", "spam.reject", "slash.command", "vault.images", etc.
    display_label: str  # Human-readable action label
    command_equivalent: str = ""  # CLI / Slash command equivalent
    args: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    reason: str = ""
    # Hierarchical Entry-Point Dispatch Governance:
    # "contractor" | "examiner" | "manager" | "worker" | "direct"
    entry_point: str = "contractor"
    direct_reply: str | None = None


class Promisor:
    """0-token intent & anti-spam gatekeeper routing natural language directly to AI."""

    def __init__(self) -> None:
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile essential classification grammars."""
        # 0. Casual Chit-chat & Greetings (Direct Reply - 0 Token)
        self.re_casual_chitchat = re.compile(
            r"^(?:(?:halo|hai|hi|hey|hei|pagi|selamat\s+(?:pagi|siang|sore|malam)|assalamu['a]?laikum|"
            r"terima\s+kasih|makasih|thanks?|tq|matur\s+nuwun|oke|siap|noted|baiklah|mantap|keren|good\s+job|nice|"
            r"wkwk+|bye|sampai\s+jumpa|tes(?:\s+123)?|testing(?:\s+123)?|kamu\s+siapa|siapa\s+kamu|"
            r"banyak|ya|mas|bro|gan|bang|scripcy|ai|paham)[\s,!.]*)+$",
            re.IGNORECASE,
        )

        # 1. Revision / Patching indicators (Manager Entry Point)
        self.re_revision_intent = re.compile(
            r"\b(perbaiki|revisi|koreksi|ubah|tambahkan|update|tulis\s+ulang|edit|kurang\s+detail|ganti\s+bagian|"
            r"ringkas\s+lagi|perpendek|perpanjang|terjemahkan\s+jawaban\s+tadi)\b",
            re.IGNORECASE,
        )

        # 2. Hypothesis Testing / Claim Verification (Examiner Entry Point)
        self.re_hypothesis_audit = re.compile(
            r"\b(apakah\s+benar|uji\s+hipotesis|verifikasi\s+klaim|validkah|apakah\s+klaim|apakah\s+betul|"
            r"benarkah\s+bahwa|is\s+it\s+true\s+that|verify\s+claim|audit\s+klaim|uji\s+apakah)\b",
            re.IGNORECASE,
        )

    def evaluate(
        self,
        user_input: str,
        active_session_id: str | None = None,
        session_id: str | None = None,
        legacy_system_intent: bool = False,
    ) -> PromisorDecision:
        """Evaluate raw user input.
        
        - Slash commands (starting with '/') are marked as system tools.
        - Non-slash inputs are checked for spam or polite greetings, then forwarded directly to AI.
        """
        clean = user_input.strip()
        sess_id = session_id or active_session_id
        if not clean:
            return PromisorDecision(
                is_system_tool=False,
                action="none",
                display_label="Pesan Kosong",
                entry_point="direct",
                direct_reply="",
            )

        # 1. Slash Command routing (deterministic CLI commands like /back, /cancel, /vault, etc.)
        if clean.startswith("/"):
            parts = clean.split(maxsplit=1)
            cmd_name = parts[0].lstrip("/")
            arg = parts[1] if len(parts) > 1 else ""
            return PromisorDecision(
                is_system_tool=True,
                action=f"slash.{cmd_name}",
                display_label=f"Perintah Slash: {parts[0]}",
                command_equivalent=clean,
                args={"cmd": cmd_name, "arg": arg},
                reason="Perintah slash CLI deterministik.",
                entry_point="direct",
            )

        # 2. Spam / Pure gibberish check (no valid alphanumeric tokens)
        alphanumeric = re.sub(r"[^a-zA-Z0-9\u00C0-\u024F\u1E00-\u1EFF]", "", clean)
        if not alphanumeric:
            return PromisorDecision(
                is_system_tool=False,
                action="spam.reject",
                display_label="Spam / Simbol Tanpa Substansi",
                reason="Input hanya terdiri dari simbol atau karakter tanpa kata substantif.",
                entry_point="direct",
                direct_reply="Input Anda tidak memuat teks atau pertanyaan riset yang dapat diproses. Silakan ajukan pertanyaan atau instruksi riset.",
            )

        # 3. Casual Chit-chat & Greetings (Direct Reply - 0 Token)
        if self.re_casual_chitchat.match(clean):
            lower_c = clean.lower()
            if any(w in lower_c for w in ("terima kasih", "makasih", "thanks", "matur nuwun")):
                reply = "Sama-sama! Senang bisa membantu. Ada topik riset atau dokumen lain di vault yang ingin dianalisis?"
            elif any(w in lower_c for w in ("oke", "siap", "noted", "baiklah", "mantap", "keren")):
                reply = "Siap. Silakan sampaikan pertanyaan atau instruksi riset berikutnya kapan saja."
            elif any(w in lower_c for w in ("kamu siapa", "siapa kamu")):
                reply = "Saya Scripcy, asisten riset akademis berbasis bukti dokumen vault dan literatur ilmiah."
            elif any(w in lower_c for w in ("tes", "testing")):
                reply = "Scripcy aktif dan siap memproses naskah atau pertanyaan riset Anda."
            else:
                reply = "Halo! Ada naskah di vault atau topik penelitian yang ingin kita telaah bersama?"
            return PromisorDecision(
                is_system_tool=False,
                action="casual.reply",
                display_label="Respon Ramah Langsung (0 Token LLM)",
                reason="Terdeteksi sapaan / ucapan terima kasih / konfirmasi santai.",
                entry_point="direct",
                direct_reply=reply,
            )

        # Optional legacy tool evaluation for explicit compatibility
        if legacy_system_intent:
            legacy_dec = self._evaluate_legacy_system_intent(clean, sess_id)
            if legacy_dec:
                return legacy_dec

        # 4. Revision / Patching Intent -> Dispatched to Manager (Working Memory)
        if self.re_revision_intent.search(clean):
            return PromisorDecision(
                is_system_tool=False,
                action="ai_chat",
                display_label="Revisi & Perbaikan Draf Sesi (Manager)",
                reason="Permintaan revisi/perbaikan dialihkan langsung ke Manager menggunakan draf baseline sebelumnya.",
                entry_point="manager",
            )

        # 5. Hypothesis Testing / Claim Verification -> Dispatched to Examiner
        if self.re_hypothesis_audit.search(clean):
            return PromisorDecision(
                is_system_tool=False,
                action="ai_chat",
                display_label="Audit & Pengujian Hipotesis (Examiner)",
                reason="Permintaan verifikasi kebenaran/klaim dialihkan langsung ke Examiner untuk audit empiris.",
                entry_point="examiner",
            )

        # 6. Default: Direct AI Research Pipeline (Contractor)
        return PromisorDecision(
            is_system_tool=False,
            action="ai_chat",
            display_label="Riset & Pertanyaan AI (Contractor)",
            reason="Pertanyaan riset akademis diteruskan langsung ke Contractor untuk formulasi Small Contracts.",
            entry_point="contractor",
        )

    def _evaluate_legacy_system_intent(self, clean: str, sess_id: str | None) -> PromisorDecision | None:
        """Legacy system tool matcher for backward compatibility."""
        lower_c = clean.lower()
        if any(w in lower_c for w in ("berkas error", "file error", "file rusak", "berkas rusak", "broken files")):
            return PromisorDecision(
                is_system_tool=True,
                action="vault.errors",
                display_label="Tampilkan Berkas Error di Vault",
                command_equivalent="/vault errors",
                entry_point="worker",
            )
        if any(w in lower_c for w in ("gambar", "figure", "visual")) and "vault" in lower_c:
            return PromisorDecision(
                is_system_tool=True,
                action="vault.images",
                display_label="Tampilkan Visual & Gambar di Vault",
                command_equivalent="/vault images",
                entry_point="worker",
            )
        if any(w in lower_c for w in ("tabel", "table")) and "vault" in lower_c:
            return PromisorDecision(
                is_system_tool=True,
                action="vault.tables",
                display_label="Tampilkan Seluruh Tabel di Vault",
                command_equivalent="/vault tables",
                entry_point="worker",
            )
        if "token" in lower_c:
            return PromisorDecision(
                is_system_tool=True,
                action="models.tokens",
                display_label="Cek Penggunaan Token",
                command_equivalent="/tokens",
                args={"scope": "session" if "sesi" in lower_c else "project"},
                entry_point="worker",
            )
        if "update vault" in lower_c or "sinkron" in lower_c:
            return PromisorDecision(
                is_system_tool=True,
                action="vault.update",
                display_label="Sinkronisasi Vault",
                command_equivalent="/vault update",
                entry_point="worker",
            )
        return None
