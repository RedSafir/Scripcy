"""Promisor: Heuristic Rule-Based Intent & Action Dispatcher for Scripcy.

Promisor is a 0-token, zero-latency gatekeeper that intercepts user input before
invoking LLM pipelines. It classifies whether an input is a direct system tool request
(to execute local Scripcy functions directly) or an academic research chat question
(to proceed into the TaskContract -> Worker -> Critic -> Verifier pipeline).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


@dataclass
class PromisorDecision:
    """Decision output produced by the Promisor heuristic & AI entry-point dispatcher."""

    is_system_tool: bool
    action: str  # e.g. "vault.images", "vault.tables", "vault.errors", "models.tokens", "vault.list", "casual.reply", "ai_chat"
    display_label: str  # Human-readable action label shown in confirmation dialog
    command_equivalent: str = ""  # CLI / Slash command equivalent
    args: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    reason: str = ""
    # Hierarchical Entry-Point Dispatch Governance:
    # "contractor" | "examiner" | "manager" | "worker" | "direct"
    entry_point: str = "contractor"
    direct_reply: str | None = None


class Promisor:
    """Heuristic rule-based intent router with extensive multilingual regex patterns."""

    def __init__(self) -> None:
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex grammars for high-throughput zero-latency matching."""
        # 0. Casual Chit-chat & Greetings (Direct Reply - 0 Token)
        self.re_casual_chitchat = re.compile(
            r"^(halo|hai|hi|hey|hei|pagi|selamat\s+(pagi|siang|sore|malam)|assalamu['a]?laikum|"
            r"(terima\s+kasih|makasih|thanks?|tq|matur\s+nuwun)(\s+(banyak|ya|mas|bro|gan|bang|scripcy|ai))*|"
            r"(oke|siap|noted|baiklah|mantap|keren|good\s+job|nice|wkwk+|bye|sampai\s+jumpa)(\s+(ya|mas|bro|gan|bang|scripcy|ai|paham|noted|oke|siap))*|"
            r"tes(\s+123)?|testing(\s+123)?|kamu\s+siapa|siapa\s+kamu)[!.\s]*$",
            re.IGNORECASE,
        )

        # 0b. Revision / Patching indicators (Manager Entry Point)
        self.re_revision_intent = re.compile(
            r"\b(perbaiki|revisi|koreksi|ubah|tambahkan|update|tulis\s+ulang|edit|kurang\s+detail|ganti\s+bagian|"
            r"ringkas\s+lagi|perpendek|perpanjang|terjemahkan\s+jawaban\s+tadi)\b",
            re.IGNORECASE,
        )

        # 0c. Hypothesis Testing / Claim Verification (Examiner Entry Point)
        self.re_hypothesis_audit = re.compile(
            r"\b(apakah\s+benar|uji\s+hipotesis|verifikasi\s+klaim|validkah|apakah\s+klaim|apakah\s+betul|"
            r"benarkah\s+bahwa|is\s+it\s+true\s+that|verify\s+claim|audit\s+klaim|uji\s+apakah)\b",
            re.IGNORECASE,
        )

        # 1. Synthesis & deep analysis guards (if present, route to AI chat)
        self.re_ai_indicators = re.compile(
            r"\b(lalu|kemudian|dan)\s+(jelaskan|simpulkan|analisis|uraikan|interpretasikan|bandingkan|kritisi)\b|"
            r"\b(then|and)\s+(explain|summarize|analyze|interpret|compare|critique)\b|"
            r"\b(mengapa|kenapa|bagaimana|apa\s+hubungan|apa\s+dampak|apa\s+pengaruh|apakah|sejauh\s+mana)\b|"
            r"\b(why|how|what\s+is\s+the\s+relationship|what\s+is\s+the\s+impact|to\s+what\s+extent)\b|"
            r"\b(buatkan|susunkan|tuliskan|draft)\s+(latar\s+belakang|tinjauan\s+pustaka|bab|paragraf|abstrak|metodologi)\b|"
            r"\b(write|draft|generate)\s+(background|literature\s+review|chapter|paragraph|abstract|methodology)\b",
            re.IGNORECASE,
        )

        # 2. Vault broken / error files
        self.re_vault_errors = re.compile(
            r"\b(apa\s+saja|tampilkan|lihat|daftar|cek|mana)\s+(berkas|file|dokumen)\s+(yang\s+)?(error|rusak|bermasalah|gagal|unsupported)\b|"
            r"\b(berkas|file|dokumen)\s+(yang\s+)?(error|rusak|bermasalah|gagal)\b|"
            r"\b(broken\s+files|failed\s+files|list\s+errors|show\s+errors)\b",
            re.IGNORECASE,
        )

        # 3. Vault images per paper or all
        self.re_vault_paper_images = re.compile(
            r"\b(tampilkan|lihat|daftar|semua|setiap|extract)\s+(gambar|visual|grafik|figure|image)s?\s+(dari|pada|untuk|in|from|of)\s+(paper|berkas|file|jurnal|dokumen)?\s*[:\"']?([a-zA-Z0-9_\-.\s]+)[\"']?",
            re.IGNORECASE,
        )
        self.re_vault_all_images = re.compile(
            r"\b(tampilkan|lihat|daftar|semua|seluruh|list)\s+(setiap\s+|seluruh\s+)?(gambar|visual|grafik|figures?|images?)\s*(di|dalam|pada)?\s*(vault)?\b|"
            r"\b(list\s+images|show\s+images|daftar\s+gambar)\b",
            re.IGNORECASE,
        )

        # 4. Vault tables per paper or all
        self.re_vault_paper_tables = re.compile(
            r"\b(tampilkan|lihat|daftar|semua|setiap|extract)\s+(tabel|table)s?\s+(dari|pada|untuk|in|from|of)\s+(paper|berkas|file|jurnal|dokumen)?\s*[:\"']?([a-zA-Z0-9_\-.\s]+)[\"']?",
            re.IGNORECASE,
        )
        self.re_vault_all_tables = re.compile(
            r"\b(tampilkan|lihat|daftar|semua|seluruh|list)\s+(setiap\s+|seluruh\s+)?(tabel|tables?)\s*(di|dalam|pada)?\s*(vault)?\b|"
            r"\b(list\s+tables|show\s+tables|daftar\s+tabel)\b",
            re.IGNORECASE,
        )

        # 5. Token consumption (session vs project)
        self.re_tokens_session = re.compile(
            r"\b(token|penggunaan\s+token|konsumsi\s+token)\s+(yang\s+sudah\s+digunakan\s+)?(pada|di|dalam|untuk)?\s*(sesi|sessi|session|chat)\s*(ini|aktif)?\b|"
            r"\b(berapa\s+token\s+(dipakai|digunakan)\s+(di|pada|dalam)\s+(sesi|sessi|chat|percakapan)\s*(ini)?)\b|"
            r"\b(session\s+tokens?|tokens?\s+used\s+(in|this)\s+session)\b",
            re.IGNORECASE,
        )
        self.re_tokens_global = re.compile(
            r"\b(berapa\s+token|cek\s+token|lihat\s+token|cek\s+penggunaan\s+token|konsumsi\s+token|total\s+token|biaya\s+token|sisa\s+kuota|token\s+usage|token\s+count)\b",
            re.IGNORECASE,
        )

        # 6. Vault update / sync
        self.re_vault_update = re.compile(
            r"(?<!riwayat\s)(?<!history\s)(?<!log\s)\b(update\s+vault|sinkronkan\s+vault|sinkronisasi\s+vault|refresh\s+vault|muat\s+ulang\s+vault)\b|"
            r"\b(scan\s+(berkas|file)\s+local|perbarui\s+vault|sinkronkan\s+berkas(\s+baru)?)\b|"
            r"\b(sync\s+vault|reload\s+vault)\b",
            re.IGNORECASE,
        )

        # 7. Vault health / status
        self.re_vault_health = re.compile(
            r"\b(cek\s+kesehatan\s+vault|status\s+vault|kondisi\s+vault|audit\s+vault|vault\s+health|vault\s+status)\b",
            re.IGNORECASE,
        )

        # 8. Vault history / audit log
        self.re_vault_history = re.compile(
            r"\b(riwayat\s+vault|log\s+aktivitas\s+vault|log\s+sinkronisasi|history\s+vault|audit\s+log\s+vault|lihat\s+riwayat\s+sinkronisasi)\b|"
            r"\b(vault\s+history|show\s+vault\s+history)\b",
            re.IGNORECASE,
        )

        # 9. Vault list files / papers / web
        self.re_vault_list = re.compile(
            r"\b(tampilkan|lihat|daftar|list|semua)\s+(berkas|file|dokumen|papers?|web)\s*(di|dalam|pada)?\s*(vault)?\b|"
            r"\b(list\s+vault|semua\s+berkas|daftar\s+paper\s+di\s+vault|tampilkan\s+paper\s+di\s+vault)\b",
            re.IGNORECASE,
        )

        # 10. Archive / restore files
        self.re_vault_archive = re.compile(
            r"\b(arsipkan|pindahkan\s+ke\s+arsip|archive)\s+(berkas|file|dokumen|paper)?\s*[:\"']?([a-zA-Z0-9_\-.]+)[\"']?\b",
            re.IGNORECASE,
        )
        self.re_vault_restore = re.compile(
            r"\b(pulihkan|kembalikan\s+dari\s+arsip|restore)\s+(berkas|file|dokumen|paper)?\s*[:\"']?([a-zA-Z0-9_\-.]+)[\"']?\b",
            re.IGNORECASE,
        )

        # 11. Model list & switch
        self.re_models_switch = re.compile(
            r"\b(ganti|ubah|switch|use|set)\s+(model|llm)\s+(ke|menjadi|to)?\s*[:\"']?([a-zA-Z0-9_\-./]+)[\"']?\b|"
            r"\b(gunakan\s+model)\s*[:\"']?([a-zA-Z0-9_\-./]+)[\"']?\b",
            re.IGNORECASE,
        )
        self.re_models_list = re.compile(
            r"\b(model\s+aktif|lihat\s+model|daftar\s+model|katalog\s+model|model\s+apa\s+yang\s+(aktif|dipakai)|list\s+models|show\s+models)\b",
            re.IGNORECASE,
        )

        # 12. 9Router status
        self.re_router9_status = re.compile(
            r"\b(cek\s+9router|status\s+9router|gateway\s+9router|9router\s+status)\b",
            re.IGNORECASE,
        )

        # 13. Citation style
        self.re_citation_settings = re.compile(
            r"\b(format\s+sitasi|style\s+sitasi|ganti\s+style\s+sitasi|citation\s+style|ubah\s+format\s+sitasi)\b",
            re.IGNORECASE,
        )

        # 14. Find and save paper/web directly
        self.re_scout_paper_save = re.compile(
            r"\b(cari|find|search|unduh|download)\s+(paper|jurnal|artikel)\s+(tentang|about|mengenai)?\s*[\"']?([^\"']+)[\"']?\s*(dan|lalu)?\s*(simpan|masukkan|tambahkan|save|add)\s*(ke|di|into|to)?\s*(vault)?\b",
            re.IGNORECASE,
        )
        self.re_scout_web_save = re.compile(
            r"\b(cari|find|search)\s+(website|web|situs)\s+(tentang|about|mengenai)?\s*[\"']?([^\"']+)[\"']?\s*(dan|lalu)?\s*(simpan|masukkan|tambahkan|save|add)\s*(ke|di|into|to)?\s*(vault)?\b",
            re.IGNORECASE,
        )

        # 15. Chat clear / new
        self.re_chat_clear = re.compile(
            r"\b(bersihkan\s+riwayat\s+chat|hapus\s+pesan\s+sesi\s+ini|clear\s+chat|clear\s+messages)\b",
            re.IGNORECASE,
        )

        # 16. Graph RAG / Knowledge Network
        self.re_vault_graph = re.compile(
            r"\b(tampilkan|lihat|buka|visualisasikan|plot|show|render)\s*(graph|grafik|network|jaringan|peta|hubungan|relasi)?\s*(rag|knowledge\s+graph|pengetahuan|dokumen|paper)?\s*(di|pada)?\s*(browser|vault)?\b|"
            r"\b(graph\s+rag|rag\s+graph|knowledge\s+graph|peta\s+relasi)\b",
            re.IGNORECASE,
        )
        self.re_chat_new = re.compile(
            r"\b(buka\s+chat\s+baru|buat\s+chat\s+baru|sesi\s+chat\s+baru|new\s+chat\s+session|create\s+new\s+chat)\b",
            re.IGNORECASE,
        )

        # 16. Processing queue / background tasks
        self.re_processing_tasks = re.compile(
            r"\b(cek\s+antrean(\s+proses)?|status\s+task(\s+background)?|lihat\s+proses\s+latar\s+belakang|processing\s+tasks|task\s+queue)\b",
            re.IGNORECASE,
        )

    def evaluate(
        self,
        user_input: str,
        active_session_id: str | None = None,
        session_id: str | None = None,
    ) -> PromisorDecision:
        """Evaluate raw user input with zero LLM tokens and determine system action vs AI chat."""
        clean = user_input.strip()
        sess_id = session_id or active_session_id
        if not clean:
            return PromisorDecision(is_system_tool=False, action="none", display_label="Pesan Kosong", entry_point="direct")

        # 0. Casual Chit-chat & Greetings (Direct Reply - 0 Token)
        if self.re_casual_chitchat.match(clean):
            # Fast polite direct response without heavy pipeline overhead
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

        # 0b. Revision / Patching Intent -> Dispatched to Manager (Working Memory)
        if self.re_revision_intent.search(clean):
            return PromisorDecision(
                is_system_tool=False,
                action="ai_chat",
                display_label="Revisi & Perbaikan Draf Sesi (Manager)",
                reason="Permintaan revisi/perbaikan dialihkan langsung ke Manager menggunakan draf baseline sebelumnya.",
                entry_point="manager",
            )

        # 0c. Hypothesis Testing / Claim Verification -> Dispatched to Examiner
        if self.re_hypothesis_audit.search(clean):
            return PromisorDecision(
                is_system_tool=False,
                action="ai_chat",
                display_label="Audit & Pengujian Hipotesis (Examiner)",
                reason="Permintaan verifikasi kebenaran/klaim dialihkan langsung ke Examiner untuk audit empiris.",
                entry_point="examiner",
            )

        # 1. Vault Errors & Broken Files -> Dispatched to Worker
        if self.re_vault_errors.search(clean):
            return PromisorDecision(
                is_system_tool=True,
                action="vault.errors",
                display_label="Tampilkan Daftar Berkas Rusak / Error di Vault",
                command_equivalent="/vault errors",
                reason="Pemeriksaan berkas yang mengalami kegagalan konversi atau tidak didukung.",
                entry_point="worker",
            )

        # 2. Vault Paper Images
        m_pimg = self.re_vault_paper_images.search(clean)
        if m_pimg:
            target_paper = m_pimg.group(5).strip().strip("'\"")
            # If target paper is not a generic word
            if target_paper and target_paper.lower() not in ("vault", "semua", "all", "ini"):
                return PromisorDecision(
                    is_system_tool=True,
                    action="vault.images",
                    display_label=f"Tampilkan Visual & Gambar dari Paper '{target_paper}'",
                    command_equivalent=f"/vault images -p {target_paper}",
                    args={"paper": target_paper},
                    reason=f"Filter visual khusus untuk paper {target_paper}.",
                )

        # 3. Vault All Images
        if self.re_vault_all_images.search(clean):
            return PromisorDecision(
                is_system_tool=True,
                action="vault.images",
                display_label="Tampilkan Seluruh Visual & Gambar di Vault",
                command_equivalent="/vault images",
                reason="Daftar inventaris visual dan gambar terindeks.",
            )

        # 4. Vault Paper Tables
        m_ptbl = self.re_vault_paper_tables.search(clean)
        if m_ptbl:
            target_paper = m_ptbl.group(5).strip().strip("'\"")
            if target_paper and target_paper.lower() not in ("vault", "semua", "all", "ini"):
                return PromisorDecision(
                    is_system_tool=True,
                    action="vault.tables",
                    display_label=f"Tampilkan Tabel dari Paper '{target_paper}'",
                    command_equivalent=f"/vault tables -p {target_paper}",
                    args={"paper": target_paper},
                    reason=f"Filter tabel khusus untuk paper {target_paper}.",
                )

        # 5. Vault All Tables
        if self.re_vault_all_tables.search(clean):
            return PromisorDecision(
                is_system_tool=True,
                action="vault.tables",
                display_label="Tampilkan Seluruh Tabel di Vault",
                command_equivalent="/vault tables",
                reason="Daftar inventaris tabel terindeks.",
            )

        # 6. Session Tokens
        if self.re_tokens_session.search(clean):
            return PromisorDecision(
                is_system_tool=True,
                action="models.tokens",
                display_label="Cek Konsumsi Token Sesi Chat Ini",
                command_equivalent="/tokens session",
                args={"scope": "session", "session_id": active_session_id},
                reason="Pemeriksaan penggunaan token pada sesi percakapan aktif.",
            )

        # 7. Global / Project Tokens
        if self.re_tokens_global.search(clean):
            return PromisorDecision(
                is_system_tool=True,
                action="models.tokens",
                display_label="Cek Statistik Penggunaan Token Proyek & Global",
                command_equivalent="/tokens",
                args={"scope": "project"},
                reason="Pemeriksaan riwayat dan kuota token keseluruhan.",
            )

        # 8. Vault History
        if self.re_vault_history.search(clean):
            return PromisorDecision(
                is_system_tool=True,
                action="vault.history",
                display_label="Tampilkan Riwayat Log Aktivitas Vault",
                command_equivalent="/vault history",
                reason="Inspeksi audit log sinkronisasi dan ingestion berkas.",
            )

        # 9. Vault Update / Sync
        if self.re_vault_update.search(clean):
            return PromisorDecision(
                is_system_tool=True,
                action="vault.update",
                display_label="Sinkronisasi & Pembaruan Berkas Vault",
                command_equivalent="/vault update",
                reason="Pemindaian staging local/ dan sinkronisasi cache markdown.",
            )

        # 10. Vault Health
        if self.re_vault_health.search(clean):
            return PromisorDecision(
                is_system_tool=True,
                action="vault.health",
                display_label="Pemeriksaan Status Kesehatan Vault",
                command_equivalent="/vault status",
                reason="Audit kesehatan berkas, tabel, dan metadata vault.",
            )

        # 11. Vault Graph RAG Network
        if self.re_vault_graph.search(clean):
            return PromisorDecision(
                is_system_tool=True,
                action="vault.graph",
                display_label="Buka Visualisasi Interaktif GraphRAG Knowledge Network",
                command_equivalent="/vault graph",
                reason="Render force-directed graph hubungan dokumen, entitas, dan visual.",
            )

        # 11. Vault Archive
        m_arch = self.re_vault_archive.search(clean)
        if m_arch:
            target_f = m_arch.group(3).strip()
            return PromisorDecision(
                is_system_tool=True,
                action="vault.archive",
                display_label=f"Arsipkan Berkas '{target_f}'",
                command_equivalent=f"/vault remove {target_f}",
                args={"target": target_f},
                reason=f"Pemindahan berkas {target_f} ke folder arsip.",
            )

        # 12. Vault Restore
        m_rest = self.re_vault_restore.search(clean)
        if m_rest:
            target_f = m_rest.group(3).strip()
            return PromisorDecision(
                is_system_tool=True,
                action="vault.restore",
                display_label=f"Pulihkan Berkas '{target_f}' dari Arsip",
                command_equivalent=f"/vault restore {target_f}",
                args={"target": target_f},
                reason=f"Pemulihan berkas {target_f} ke vault aktif.",
            )

        # 13. Model Switch
        m_use = self.re_models_switch.search(clean)
        if m_use:
            target_model = m_use.group(4) or m_use.group(6)
            if target_model:
                target_model = target_model.strip()
                return PromisorDecision(
                    is_system_tool=True,
                    action="models.use",
                    display_label=f"Ganti Model Aktif ke '{target_model}'",
                    command_equivalent=f"/models use {target_model}",
                    args={"model": target_model},
                    reason=f"Penggantian model LLM aktif ke {target_model}.",
                )

        # 14. Model Catalog
        if self.re_models_list.search(clean):
            return PromisorDecision(
                is_system_tool=True,
                action="models.list",
                display_label="Lihat Model Aktif & Katalog Provider",
                command_equivalent="/models list",
                reason="Daftar model yang didukung dan status provider aktif.",
            )

        # 15. 9Router Gateway Status
        if self.re_router9_status.search(clean):
            return PromisorDecision(
                is_system_tool=True,
                action="models.9router",
                display_label="Periksa Status Gateway 9Router Lokal",
                command_equivalent="/models 9router",
                reason="Inspeksi konektivitas 9Router proxy lokal.",
            )

        # 16. Citation Style
        if self.re_citation_settings.search(clean):
            return PromisorDecision(
                is_system_tool=True,
                action="settings.citation",
                display_label="Buka Pengaturan Format & Gaya Sitasi",
                command_equivalent="/citation",
                reason="Pengaturan style sitasi (APA, IEEE, BibTeX).",
            )

        # 17. Scout Paper & Save
        m_scout_p = self.re_scout_paper_save.search(clean)
        if m_scout_p:
            scout_q = m_scout_p.group(4).strip()
            return PromisorDecision(
                is_system_tool=True,
                action="scout.paper",
                display_label=f"Cari Paper tentang '{scout_q}' & Simpan ke Vault",
                command_equivalent=f"/scout {scout_q}",
                args={"query": scout_q},
                reason="Pencarian literatur eksternal dan ingestion otomatis.",
            )

        # 18. Scout Web & Save
        m_scout_w = self.re_scout_web_save.search(clean)
        if m_scout_w:
            scout_q = m_scout_w.group(4).strip()
            return PromisorDecision(
                is_system_tool=True,
                action="scout.web",
                display_label=f"Cari Website tentang '{scout_q}' & Simpan ke Vault",
                command_equivalent=f"/web-scout {scout_q}",
                args={"query": scout_q},
                reason="Pencarian sumber web resmi dan ingestion otomatis.",
            )

        # 19. Vault List
        m_list = self.re_vault_list.search(clean)
        if m_list:
            cat = "all"
            if "paper" in clean.lower():
                cat = "papers"
            elif "web" in clean.lower():
                cat = "web"
            return PromisorDecision(
                is_system_tool=True,
                action="vault.list",
                display_label=f"Tampilkan Daftar Berkas di Vault (Kategori: {cat.title()})",
                command_equivalent=f"/vault list --category {cat}",
                args={"category": cat},
                reason="Daftar berkas aktif di dalam Vault riset.",
            )

        # 20. Chat Clear & New
        if self.re_chat_clear.search(clean):
            return PromisorDecision(
                is_system_tool=True,
                action="chat.clear",
                display_label="Bersihkan Riwayat Percakapan Sesi Ini",
                command_equivalent="/clear",
                reason="Pembersihan pesan dalam sesi chat aktif.",
            )
        if self.re_chat_new.search(clean):
            return PromisorDecision(
                is_system_tool=True,
                action="chat.new",
                display_label="Buat Sesi Percakapan Baru",
                command_equivalent="/new",
                reason="Pembuatan sesi chat riset baru.",
            )

        # 21. Processing Tasks
        if self.re_processing_tasks.search(clean):
            return PromisorDecision(
                is_system_tool=True,
                action="processing.tasks",
                display_label="Lihat Status Antrean Proses Latar Belakang",
                command_equivalent="/proc",
                reason="Monitoring background workers dan antrean proses.",
            )

        # Default fallback: Route to AI Chat (Contractor Entry Point)
        return PromisorDecision(
            is_system_tool=False,
            action="ai_chat",
            display_label="Riset & Pertanyaan AI (Contractor)",
            reason="Pertanyaan riset akademis baru dialihkan ke Contractor untuk formulasi Small Contracts.",
            entry_point="contractor",
        )
