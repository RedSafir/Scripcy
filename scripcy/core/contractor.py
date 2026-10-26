"""Contract decomposer and context slicer for isolated subagent execution."""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from scripcy.core.contracts import TaskContract
from scripcy.core.engine import EvidenceChunk, EvidenceSynthesizer, estimate_tokens


CHANNEL_LABELS: dict[str, str] = {
    "vault_lokal": "Vault Lokal",
    "web_resmi": "Portal Web Resmi (WebScout)",
    "repositori_jurnal_multi_source": "Repositori Jurnal Multi-Source (OpenAlex/Semantic Scholar/CrossRef)",
    "reference_mapping_snowballing_backward_forward": "Reference Mapping & Snowballing (Backward/Forward Citations)",
}


class CritiqueAdjudication(BaseModel):
    """Gatekeeper decision by Contractor over Devil's Advocate critique using the Absolute Contract."""

    should_revise: bool = Field(
        default=False,
        description="True only when a revision is strictly necessary under the Absolute Contract.",
    )
    allow_citation_mapping: bool = Field(
        default=False,
        description="True only when Reference Mapping + Snowballing has not yet been exhausted.",
    )
    accepted_revisions: list[str] = Field(
        default_factory=list,
        description="Filtered, in-scope revision items approved by Contractor.",
    )
    rejected_critiques: list[str] = Field(
        default_factory=list,
        description="Critiques/questions from Devil's Advocate vetoed by Contractor as out-of-scope or non-constructive.",
    )
    override_passed: bool = Field(
        default=True,
        description="True when Contractor vetoes Devil's Advocate failure and marks the response as passing.",
    )
    override_score: int = Field(
        default=8,
        ge=1,
        le=10,
        description="Effective critic score after Contractor gatekeeper adjudication.",
    )
    search_exhausted: bool = Field(
        default=False,
        description="True when required search channels were already executed and further searching is halted.",
    )
    attempted_channels_summary: list[str] = Field(
        default_factory=list,
        description="Human-readable list of search channels already executed.",
    )
    exhausted_search_disclosure: str | None = Field(
        default=None,
        description="Transparent note detailing exhausted search channels when specific data remains unavailable.",
    )
    adjudication_summary: str = Field(
        default="",
        description="Concise rationale of Contractor gatekeeper decision.",
    )


class EvidenceAdjudication(BaseModel):
    """Adjudication decision by Contractor over Devil's Advocate post-scout evidence audit."""

    decision: Literal["expand", "honest_gap", "approved"] = "approved"
    follow_up_subcontracts: list[SubContract] = Field(
        default_factory=list,
        description="Targeted follow-up SubContracts to be executed if decision is expand.",
    )
    gap_notes: list[str] = Field(
        default_factory=list,
        description="Declared data gaps if decision is honest_gap.",
    )
    rejected_critiques: list[str] = Field(
        default_factory=list,
        description="Devil's Advocate complaints vetoed as out-of-scope or beyond contract limits.",
    )
    adjudication_summary: str = Field(default="", description="Summary of Contractor adjudication.")


SubagentRole = Literal[
    "VaultExtractor",
    "DialecticalExtractor",
    "WebScout",
    "LitScout",
    "GapExtractor",
    "ScopeAuditor",
]


class SubContract(BaseModel):
    """Scoped work unit and isolated context slice assigned to a single subagent."""

    id: str = Field(description="Unique subcontract identifier (e.g. sub-vault-1).")
    role: SubagentRole = Field(description="Specialized subagent role.")
    objective: str = Field(description="Focused extraction or scouting instruction.")
    allowed_scope: list[str] = Field(default_factory=list, description="Topics permitted in this subcontract.")
    forbidden_scope: list[str] = Field(default_factory=list, description="Topics that must be ignored.")
    evidence_slice: list[EvidenceChunk] = Field(
        default_factory=list,
        description="Strict subset of vault chunks allocated to this subagent.",
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description="Targeted external queries when role is WebScout or LitScout.",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Upstream subcontract IDs that must finish before this subcontract starts.",
    )
    dos: list[str] = Field(default_factory=list, description="Explicit substantive requirements and instructions for this subagent.")
    donts: list[str] = Field(default_factory=list, description="Explicit out-of-scope restrictions and non-goals for this subagent.")
    target_channel: str = Field(
        default="vault_lokal",
        description="Target search/retrieval channel (vault_lokal, web_resmi, repositori_jurnal, reference_mapping).",
    )
    max_output_tokens: int = Field(
        default=220,
        ge=64,
        le=600,
        description="Hard token ceiling for compact structured output.",
    )


class ExecutionPlan(BaseModel):
    """DAG execution plan produced by the Contractor from a TaskContract."""

    query: str
    contract: TaskContract
    is_fast_path: bool = Field(
        default=True,
        description="True when a single local extraction pass suffices without external gap scouting.",
    )
    has_evidence_gap: bool = Field(
        default=False,
        description="True when local vault chunks miss requested temporal, statistical, or topical coverage.",
    )
    gap_reason: str | None = Field(default=None, description="Explanation of detected coverage gap.")
    subcontracts: list[SubContract] = Field(default_factory=list)


class ContractDecomposer:
    """Decomposes a TaskContract into isolated SubContracts and slices context per subagent."""

    def __init__(self, max_chunks_per_subagent: int = 4) -> None:
        self.max_chunks_per_subagent = max_chunks_per_subagent
        self._synthesizer = EvidenceSynthesizer()
        self._current_year = datetime.now().year

    def detect_evidence_gap(
        self,
        query: str,
        contract: TaskContract,
        evidence: list[EvidenceChunk],
    ) -> tuple[bool, str | None, list[str], list[str]]:
        """Detect whether local evidence has temporal, statistical, or topical gaps.

        Returns (has_gap, gap_reason, web_queries, lit_queries).
        """
        lower_q = query.lower()
        clean_words = [
            w for w in re.findall(r"\b\w{4,}\b", lower_q)
            if w not in {
                "yang", "untuk", "dengan", "dalam", "pada", "oleh", "dari", "tentang",
                "bagaimana", "apakah", "berapa", "carikan", "tolong", "buatkan", "data",
                "tahun", "terakhir",
            }
        ]
        base_topic = " ".join(clean_words[:5]) if clean_words else query[:50]

        if not evidence:
            web_q = [f"{base_topic} data resmi statistik"] if contract.archetype == "DATA_FACTUAL" else []
            lit_q = [f"{base_topic} empirical study review"]
            return True, "Vault lokal belum memiliki potongan bukti yang cocok.", web_q, lit_q

        combined_text = " ".join(c.text.lower() for c in evidence)
        coverage = self._synthesizer.compute_coverage_score(query, evidence)

        # Check official primary source gap for factual data / official web channel
        has_official_source = any(
            (
                hasattr(c, "source")
                and (
                    str(c.source).startswith("web_")
                    or ".go.id" in str(c.source).lower()
                    or ".gov" in str(c.source).lower()
                )
            )
            or (
                hasattr(c, "text")
                and any(
                    domain_marker in c.text.lower()
                    for domain_marker in (".go.id", ".gov")
                )
            )
            for c in evidence
        )
        if (contract.archetype == "DATA_FACTUAL" or "web_resmi" in getattr(contract, "required_search_channels", [])) and not has_official_source:
            reason = "Kueri data faktual memerlukan rujukan primer dari portal resmi pemerintah/institusi terverifikasi."
            web_q = [f"{base_topic} data resmi statistik", f"{base_topic} portal resmi"]
            lit_q = [f"{base_topic} data empiris"] if coverage < 0.50 else []
            return True, reason, web_q, lit_q

        # Check temporal gap when user asks for recent N years or explicit modern years
        asks_recent = any(
            p in lower_q
            for p in ("5 tahun terakhir", "lima tahun terakhir", "terbaru", "mutakhir", "2022", "2023", "2024", "2025")
        )
        years_in_ev = [
            int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", combined_text)
            if 1990 <= int(y) <= self._current_year
        ]
        max_ev_year = max(years_in_ev) if years_in_ev else 0

        if asks_recent and (not years_in_ev or max_ev_year < self._current_year - 5):
            reason = (
                f"Pertanyaan meminta data periode terbaru, namun bukti vault lokal "
                f"hanya mencakup hingga tahun {max_ev_year or 'lama'}."
            )
            web_q = [f"{base_topic} data statistik resmi {self._current_year - 4} {self._current_year}"]
            lit_q = [f"{base_topic} {self._current_year - 3} {self._current_year}"]
            return True, reason, web_q, lit_q

        # Check national/aggregate statistical gap when local vault only has localized case studies
        if contract.archetype == "DATA_FACTUAL":
            asks_national = any(w in lower_q for w in ("indonesia", "nasional", "rekap", "rekapitulasi", "tabel"))
            has_national_table = any(
                w in combined_text for w in ("rekapitulasi nasional", "sipongi", "bps", "seluruh indonesia", "hektare per tahun")
            )
            if asks_national and not has_national_table and coverage < 0.75:
                reason = "Dokumen vault lokal didominasi studi kasus spesifik dan belum memuat rekapitulasi agregat nasional."
                web_q = [f"{base_topic} data resmi pemerintah indonesia statistik tahunan"]
                lit_q = [f"{base_topic} indonesia national trend data"]
                return True, reason, web_q, lit_q

        if coverage < 0.40:
            reason = f"Cakupan leksikal bukti lokal rendah ({coverage:.2f} < 0.40)."
            web_q = [f"{base_topic} publikasi resmi"] if contract.archetype == "DATA_FACTUAL" else []
            lit_q = [f"{base_topic} research evidence"]
            return True, reason, web_q, lit_q

        return False, None, [], []

    def slice_evidence(
        self,
        focus_query: str,
        evidence: list[EvidenceChunk],
        limit: int | None = None,
        exclude_chunk_ids: set[str] | None = None,
    ) -> list[EvidenceChunk]:
        """Select a compact, non-overlapping slice of evidence chunks for a specific subcontract."""
        if not evidence:
            return []
        cap = limit or self.max_chunks_per_subagent
        excluded = exclude_chunk_ids or set()
        pool = [c for c in evidence if c.chunk_id not in excluded]
        if not pool:
            pool = list(evidence)

        focus_terms = set(re.findall(r"\b\w{4,}\b", focus_query.lower()))
        if not focus_terms:
            return pool[:cap]

        def _score_chunk(chunk: EvidenceChunk) -> tuple[int, float]:
            text_lower = chunk.text.lower()
            hits = sum(1 for t in focus_terms if t in text_lower)
            has_numbers = 1 if re.search(r"\b\d+(?:[.,]\d+)?\s*(?:%|ha|hektar|hektare|ribu|juta|ton|orang|kasus)\b", text_lower) else 0
            return (hits * 2 + has_numbers, -len(chunk.text))

        ranked = sorted(pool, key=_score_chunk, reverse=True)
        return ranked[:cap]

    def decompose(
        self,
        query: str,
        contract: TaskContract,
        evidence: list[EvidenceChunk],
        enable_external_scout: bool = True,
    ) -> ExecutionPlan:
        """Build a DAG ExecutionPlan of SubContracts with strict context slicing."""
        has_gap, gap_reason, web_queries, lit_queries = self.detect_evidence_gap(query, contract, evidence)
        subcontracts: list[SubContract] = []
        used_chunk_ids: set[str] = set()

        forbidden = list(contract.out_of_scope) + list(contract.forbidden_drift_topics)

        if contract.archetype == "DATA_FACTUAL":
            slice_1 = self.slice_evidence(query, evidence, limit=self.max_chunks_per_subagent)
            used_chunk_ids.update(c.chunk_id for c in slice_1)
            subcontracts.append(
                SubContract(
                    id="sub-vault-data",
                    role="VaultExtractor",
                    objective=(
                        f"Ekstrak HANYA angka, metrik, tahun, lokasi, dan fakta terukur dari bukti lokal "
                        f"yang menjawab: '{query}'."
                    ),
                    allowed_scope=contract.in_scope,
                    forbidden_scope=forbidden,
                    evidence_slice=slice_1,
                    target_channel="vault_lokal",
                    dos=[
                        "Ekstrak metrik numerik spesifik, tahun pengamatan, dan unit kuantitatif.",
                        "Cantumkan stem sitasi resmi untuk setiap klaim fakta.",
                    ],
                    donts=[
                        "Dilarang menulis pengantar teori panjang atau narasi opini.",
                        "Dilarang mengarang angka di luar teks bukti dokumen.",
                    ],
                    max_output_tokens=220,
                )
            )
        elif contract.archetype in ("DIALECTICAL_CRITIQUE", "COMPARISON"):
            slice_pro = self.slice_evidence(f"{query} keunggulan efektivitas bukti positif", evidence, limit=3)
            used_chunk_ids.update(c.chunk_id for c in slice_pro)
            subcontracts.append(
                SubContract(
                    id="sub-vault-pro",
                    role="VaultExtractor",
                    objective=f"Ekstrak klaim pendukung utama, parameter unggulan, dan bukti empiris untuk: '{query}'.",
                    allowed_scope=contract.in_scope[:2],
                    forbidden_scope=forbidden,
                    evidence_slice=slice_pro,
                    target_channel="vault_lokal",
                    dos=[
                        "Identifikasi bukti positif dan parameter keunggulan terukur.",
                        "Kaitkan klaim dengan kutipan dokumen sumber.",
                    ],
                    donts=[
                        "Jangan menyertakan kritik atau kontra-argumen pada subkontrak ini.",
                    ],
                    max_output_tokens=220,
                )
            )
            slice_contra = self.slice_evidence(
                f"{query} limitasi kendala tantangan kontra trade-off",
                evidence,
                limit=3,
                exclude_chunk_ids=used_chunk_ids,
            )
            subcontracts.append(
                SubContract(
                    id="sub-vault-contra",
                    role="DialecticalExtractor",
                    objective=f"Ekstrak kondisi batas, limitasi metodologis, atau trade-off empiris untuk: '{query}'.",
                    allowed_scope=contract.in_scope,
                    forbidden_scope=forbidden,
                    evidence_slice=slice_contra,
                    target_channel="vault_lokal",
                    dos=[
                        "Ekstrak trade-off, batasan penerapan, dan kendala empiris yang terbukti.",
                    ],
                    donts=[
                        "Jangan mengulang data keunggulan yang sudah menjadi fokus subkontrak pro.",
                    ],
                    max_output_tokens=220,
                )
            )
        else:
            slice_main = self.slice_evidence(query, evidence, limit=self.max_chunks_per_subagent)
            subcontracts.append(
                SubContract(
                    id="sub-vault-main",
                    role="VaultExtractor",
                    objective=f"Ekstrak temuan inti dan fakta terverifikasi untuk menjawab: '{query}'.",
                    allowed_scope=contract.in_scope,
                    forbidden_scope=forbidden,
                    evidence_slice=slice_main,
                    target_channel="vault_lokal",
                    dos=["Ekstrak klaim substansial inti beserta sitasi dokumen."],
                    donts=["Dilarang membahas topik out-of-scope."],
                    max_output_tokens=240,
                )
            )

        upstream_scout_ids: list[str] = []
        if has_gap and enable_external_scout:
            if web_queries:
                subcontracts.append(
                    SubContract(
                        id="sub-scout-web",
                        role="WebScout",
                        objective=f"Cari data resmi portal pemerintah/institusi untuk menutup gap: {gap_reason}",
                        allowed_scope=contract.in_scope,
                        forbidden_scope=forbidden,
                        search_queries=web_queries[:2],
                        target_channel="web_resmi",
                        dos=[
                            "Fokus pada portal resmi .go.id, lembaga pemerintah, atau badan internasional.",
                            "Ambil tabel data dan siaran pers statistik resmi.",
                        ],
                        donts=[
                            "Dilarang mengambil blog opini pribadi atau forum komunitas.",
                        ],
                        max_output_tokens=180,
                    )
                )
                upstream_scout_ids.append("sub-scout-web")

            if lit_queries:
                subcontracts.append(
                    SubContract(
                        id="sub-scout-lit",
                        role="LitScout",
                        objective=f"Cari literatur akademik terbaru untuk menutup gap: {gap_reason}",
                        allowed_scope=contract.in_scope,
                        forbidden_scope=forbidden,
                        search_queries=lit_queries[:2],
                        target_channel="repositori_jurnal",
                        dos=[
                            "Prioritaskan jurnal bereputasi dengan peer-review dan DOI resmi.",
                        ],
                        donts=[
                            "Hindari naskah tanpa kejelasan metodologi atau data empiris.",
                        ],
                        max_output_tokens=180,
                    )
                )
                upstream_scout_ids.append("sub-scout-lit")

            if upstream_scout_ids:
                subcontracts.append(
                    SubContract(
                        id="sub-gap-extract",
                        role="GapExtractor",
                        objective=(
                            f"Ekstrak fakta dan angka baru dari dokumen hasil pencarian eksternal "
                            f"untuk melengkapi: '{query}'."
                        ),
                        allowed_scope=contract.in_scope,
                        forbidden_scope=forbidden,
                        depends_on=upstream_scout_ids,
                        target_channel="vault_lokal",
                        dos=["Ekstrak temuan baru yang belum tercakup pada ekstraksi awal."],
                        donts=["Jangan mengulang fakta yang sudah ada."],
                        max_output_tokens=220,
                    )
                )

        is_fast = len(subcontracts) == 1 and not upstream_scout_ids
        return ExecutionPlan(
            query=query,
            contract=contract,
            is_fast_path=is_fast,
            has_evidence_gap=has_gap,
            gap_reason=gap_reason,
            subcontracts=subcontracts,
        )

    def adjudicate_plan_audit(
        self,
        plan: ExecutionPlan,
        audit_review: Any,
    ) -> ExecutionPlan:
        """Adjudicate Devil's Advocate pre-flight plan audit: prune unsubstantial/redundant subcontracts and add valid ones."""
        if not audit_review:
            return plan

        prune_ids = set(getattr(audit_review, "prune_ids", []) or [])
        additions = getattr(audit_review, "additions", []) or []

        # 1. Prune unsubstantial / redundant subcontracts identified by Devil's Advocate
        remaining_subcontracts = [sc for sc in plan.subcontracts if sc.id not in prune_ids]
        if not remaining_subcontracts:
            # Always ensure at least 1 core subcontract survives
            remaining_subcontracts = plan.subcontracts[:1]

        # 2. Add substantive missing subcontracts approved by Contractor
        forbidden = list(plan.contract.out_of_scope) + list(plan.contract.forbidden_drift_topics)
        for idx, add_item in enumerate(additions, 1):
            if not isinstance(add_item, dict):
                continue
            obj = str(add_item.get("objective") or "").strip()
            if not obj:
                continue
            # Scope filter: reject if addition is in forbidden scope
            if any(f.lower() in obj.lower() for f in forbidden if len(f) > 3):
                continue
            new_id = f"sub-audit-add-{idx}"
            remaining_subcontracts.append(
                SubContract(
                    id=new_id,
                    role=add_item.get("role", "VaultExtractor"),
                    objective=obj,
                    allowed_scope=plan.contract.in_scope,
                    forbidden_scope=forbidden,
                    target_channel=add_item.get("target_channel", "vault_lokal"),
                    dos=add_item.get("dos", ["Ekstrak bukti spesifik untuk melengkapi gap yang ditemukan Devil's Advocate."]),
                    donts=add_item.get("donts", ["Dilarang melebar ke topik out-of-scope."]),
                    search_queries=add_item.get("search_queries", []),
                    max_output_tokens=220,
                )
            )

        return ExecutionPlan(
            query=plan.query,
            contract=plan.contract,
            is_fast_path=len(remaining_subcontracts) <= 1,
            has_evidence_gap=plan.has_evidence_gap,
            gap_reason=plan.gap_reason,
            subcontracts=remaining_subcontracts,
        )

    def adjudicate_evidence_audit(
        self,
        query: str,
        contract: TaskContract,
        audit_review: Any,
        expansion_count: int = 0,
        max_expansions: int = 1,
        previous_worker_notes: list[str] | None = None,
    ) -> EvidenceAdjudication:
        """Adjudicate Devil's Advocate post-scout audit: decide whether to expand search, declare honest gap, or approve."""
        if not audit_review or getattr(audit_review, "all_satisfied", True):
            return EvidenceAdjudication(
                decision="approved",
                adjudication_summary="Temuan worker telah memenuhi kontrak; tidak diperlukan penelusuran tambahan.",
            )

        detected_gaps = list(getattr(audit_review, "detected_gaps", []) or [])
        suggested_queries = list(getattr(audit_review, "suggested_expansion_queries", []) or [])

        # Filter gaps against contract out_of_scope and forbidden_drift_topics
        valid_gaps: list[str] = []
        rejected: list[str] = []
        forbidden = [p.lower() for p in (list(contract.out_of_scope) + list(contract.forbidden_drift_topics)) if p]

        for gap in detected_gaps:
            lower_g = gap.lower()
            if any(f in lower_g for f in forbidden if len(f) > 3):
                rejected.append(f"[Ditolak - Out of Scope] {gap}")
                continue
            if contract.archetype == "DATA_FACTUAL" and any(m in lower_g for m in ("bab 2", "teori", "birokrasi", "landasan")):
                rejected.append(f"[Ditolak - Tuntutan teori pada DATA_FACTUAL] {gap}")
                continue
            valid_gaps.append(gap)

        if not valid_gaps:
            return EvidenceAdjudication(
                decision="approved",
                rejected_critiques=rejected,
                adjudication_summary=(
                    f"Contractor memveto {len(rejected)} komplain Devil's Advocate karena melanggar batas kontrak; "
                    "fakta yang ada disahkan."
                ),
            )

        # Check if expansion budget is exhausted
        try:
            effective_max = int(max_expansions)
        except (TypeError, ValueError):
            effective_max = 1

        if expansion_count >= effective_max:
            return EvidenceAdjudication(
                decision="honest_gap",
                gap_notes=valid_gaps,
                rejected_critiques=rejected,
                adjudication_summary=(
                    f"Batas ekspansi penelusuran ({expansion_count}/{effective_max}) telah tercapai. "
                    f"Contractor mencatat {len(valid_gaps)} gap sebagai Honest Data Gap."
                ),
            )

        # Create targeted follow-up subcontract
        target_role: SubagentRole = "WebScout" if (contract.archetype == "DATA_FACTUAL" or "web_resmi" in getattr(contract, "required_search_channels", [])) else "LitScout"
        channel = "web_resmi" if target_role == "WebScout" else "repositori_jurnal"
        scout_id = f"sub-followup-scout-{expansion_count + 1}"
        extract_id = f"sub-followup-extract-{expansion_count + 1}"

        clean_gap_desc = "; ".join(valid_gaps[:2])
        notes_str = "; ".join(previous_worker_notes[:2]) if previous_worker_notes else ""
        handoff_ctx = f" (Catatan worker sebelumnya: {notes_str})" if notes_str else ""

        follow_up_scout = SubContract(
            id=scout_id,
            role=target_role,
            objective=f"Pencarian lanjutan terarah untuk menutup gap: {clean_gap_desc}{handoff_ctx}",
            allowed_scope=contract.in_scope,
            forbidden_scope=list(contract.out_of_scope) + list(contract.forbidden_drift_topics),
            target_channel=channel,
            dos=[
                "Prioritaskan portal resmi/institusi atau repositori ilmiah untuk parameter/tahun yang masih bolong.",
                "Gunakan variasi kata kunci alternatif jika pencarian sebelumnya nihil.",
            ],
            donts=[
                "Dilarang mengambil blog opini pribadi atau forum komunitas.",
                "Jangan mengulang kueri persis yang sudah terbukti nihil.",
            ],
            search_queries=suggested_queries[:2] or [f"{query} data resmi"],
            max_output_tokens=180,
        )

        follow_up_extractor = SubContract(
            id=extract_id,
            role="GapExtractor",
            objective=f"Ekstrak fakta baru dari dokumen hasil pencarian lanjutan untuk melengkapi: '{query}'.",
            allowed_scope=contract.in_scope,
            forbidden_scope=list(contract.out_of_scope) + list(contract.forbidden_drift_topics),
            depends_on=[scout_id],
            target_channel="vault_lokal",
            dos=["Ekstrak data dan angka spesifik yang menjawab gap yang ditemukan."],
            donts=["Jangan menduplikasi fakta yang sudah terekstrak sebelumnya."],
            max_output_tokens=220,
        )

        return EvidenceAdjudication(
            decision="expand",
            follow_up_subcontracts=[follow_up_scout, follow_up_extractor],
            gap_notes=valid_gaps,
            rejected_critiques=rejected,
            adjudication_summary=(
                f"Contractor menugaskan 1 putaran follow-up search ({scout_id}) untuk melengkapi {len(valid_gaps)} gap."
            ),
        )

    def _is_critique_out_of_contract(self, item: str, contract: TaskContract, draft_tokens: int) -> tuple[bool, str]:
        """Check whether a single Devil's Advocate complaint violates the Absolute Contract."""
        lower_item = item.lower()

        # 1. Explicit out-of-scope or forbidden drift topic match
        forbidden_phrases = [p.lower() for p in (list(contract.out_of_scope) + list(contract.forbidden_drift_topics)) if p]
        for fp in forbidden_phrases:
            fp_words = [w for w in re.findall(r"\b\w{4,}\b", fp) if w not in {"yang", "tidak", "tanpa", "dalam", "untuk", "pada"}]
            if fp in lower_item or (len(fp_words) >= 2 and sum(1 for w in fp_words if w in lower_item) >= 2):
                return True, f"Melanggar batas Out-of-Scope/Forbidden Drift kontrak ('{fp}')"

        # 2. Archetype DATA_FACTUAL or FOCUSED_QA drift patterns (pushing Bab 2 / bureaucracy / methodology essays)
        if contract.archetype == "DATA_FACTUAL":
            drift_markers = (
                "bab 2", "tinjauan pustaka", "landasan teori", "kerangka teoritis",
                "birokrasi", "kelembagaan", "tata kelola", "metodologi", "epistemologi",
                "sosiologis", "implikasi kebijakan", "perdebatan teoretis", "diskursus",
            )
            if any(m in lower_item for m in drift_markers):
                return True, "Menuntut pembahasan teori/birokrasi/Bab 2 pada kontrak DATA_FACTUAL"

        if not contract.allow_counter_arguments:
            dialectic_markers = (
                "argumen tandingan", "counter-argument", "sanggahan", "pro dan kontra",
                "pro-kontra", "debat dialektis", "sudut pandang berlawanan",
            )
            if any(m in lower_item for m in dialectic_markers):
                return True, "Menuntut argumen tandingan padahal allow_counter_arguments=False"

        # 3. Token budget ceiling check: reject demands to expand/elaborate when draft already hits effective_max_tokens
        if not getattr(contract, "unlimited_tokens", False) and draft_tokens >= contract.effective_max_tokens:
            expansion_markers = (
                "perluas", "lebih mendalam", "lebih komprehensif", "tambahkan pembahasan",
                "uraikan lebih lanjut", "elaborasi", "kurang panjang", "terlalu singkat",
                "tambahkan sub-bab",
            )
            if any(m in lower_item for m in expansion_markers):
                return True, (
                    f"Menuntut perluasan teks padahal draf ({draft_tokens} tok) telah mencapai "
                    f"batas atas kuota kontrak ({contract.effective_max_tokens} tok)"
                )

        return False, ""

    def adjudicate_critique(
        self,
        query: str,
        draft: str,
        contract: TaskContract,
        critique_report: Any,
        fact_verdict: Any | None = None,
        attempted_search_channels: set[str] | list[str] | None = None,
    ) -> CritiqueAdjudication:
        """Act as Gatekeeper over Devil's Advocate using the Absolute TaskContract.

        Filters out non-constructive or out-of-scope complaints, enforces the token budget
        margin of error, and halts repeated search loops once required search channels
        (Vault, Official Web, Multi-Source Journals, Reference Mapping + Snowballing) are exhausted.
        """
        attempted_set = set(attempted_search_channels or ["vault_lokal"])
        ordered_channels = [
            ch for ch in (
                "vault_lokal",
                "web_resmi",
                "repositori_jurnal_multi_source",
                "reference_mapping_snowballing_backward_forward",
            )
            if ch in attempted_set
        ]
        for extra_ch in sorted(attempted_set):
            if extra_ch not in ordered_channels:
                ordered_channels.append(extra_ch)

        attempted_labels = [CHANNEL_LABELS.get(ch, ch) for ch in ordered_channels]
        draft_tokens = estimate_tokens(draft or "")

        required_set = set(contract.required_search_channels or [])
        snowball_attempted = "reference_mapping_snowballing_backward_forward" in attempted_set
        all_required_attempted = bool(required_set and required_set.issubset(attempted_set))
        search_exhausted = snowball_attempted or all_required_attempted

        has_hallucination = bool(fact_verdict and getattr(fact_verdict, "hallucination_warning", None))
        recency_alert = getattr(critique_report, "recency_alert", None)
        needs_cit_map = bool(getattr(critique_report, "needs_citation_mapping", False))

        accepted: list[str] = []
        rejected: list[str] = []

        # 1. Counter-arguments check
        raw_counters = list(getattr(critique_report, "counter_arguments", []) or [])
        if not contract.allow_counter_arguments:
            for ca in raw_counters:
                rejected.append(f"[Ditolak - Kontrak melarang counter-argument] {ca}")
        else:
            for ca in raw_counters:
                is_out, reason = self._is_critique_out_of_contract(ca, contract, draft_tokens)
                if is_out:
                    rejected.append(f"[Ditolak - {reason}] {ca}")
                elif not search_exhausted:
                    accepted.append(ca)
                else:
                    rejected.append(f"[Ditolak - Pencarian telah mencapai batas maksimum] {ca}")

        # 2. Weaknesses check
        raw_weaknesses = list(getattr(critique_report, "weaknesses", []) or [])
        missing_data_markers = (
            "belum lengkap", "tidak tersedia", "belum tersedia", "data tahun",
            "lima tahun", "5 tahun", "rekapitulasi", "cakupan", "referensi tambahan",
            "literatur tambahan", "data empiris", "studi kasus", "kurang mutakhir",
        )
        for wk in raw_weaknesses:
            is_out, reason = self._is_critique_out_of_contract(wk, contract, draft_tokens)
            if is_out:
                rejected.append(f"[Ditolak - {reason}] {wk}")
                continue

            lower_wk = wk.lower()
            is_missing_data_complaint = any(m in lower_wk for m in missing_data_markers)
            if search_exhausted and is_missing_data_complaint and not has_hallucination:
                rejected.append(
                    f"[Ditolak - Seluruh jalur pencarian ({', '.join(attempted_labels)}) telah ditempuh] {wk}"
                )
                continue

            if not getattr(contract, "unlimited_tokens", False) and draft_tokens >= contract.effective_max_tokens and not has_hallucination:
                rejected.append(
                    f"[Ditolak - Kuota token kontrak ({draft_tokens}/{contract.effective_max_tokens} tok) sudah penuh] {wk}"
                )
                continue

            accepted.append(wk)

        # 3. Always accept factual hallucination removal if FactChecker flagged unsupported claims
        if has_hallucination:
            hallucination_note = str(getattr(fact_verdict, "hallucination_warning", "Klaim tanpa bukti vault terdeteksi."))
            if hallucination_note not in accepted:
                accepted.insert(0, f"Hapus/koreksi klaim tanpa rujukan dokumen vault: {hallucination_note}")

        # 4. Decide whether Reference Mapping + Snowballing is permitted on this turn
        allow_snowball = False
        if needs_cit_map and not search_exhausted:
            if recency_alert or accepted or has_hallucination:
                allow_snowball = True

        # 5. Build transparent disclosure if search is exhausted and Devil's Advocate complained about missing data
        disclosure: str | None = None
        if search_exhausted and (needs_cit_map or recency_alert or any("pencarian" in r.lower() for r in rejected)):
            channels_str = ", ".join(attempted_labels)
            disclosure = (
                f"\n\n> [!NOTE]\n"
                f"> **Batas Penelusuran Literatur & Data (Kontrak Absolut)**: Sistem telah menelusuri "
                f"seluruh jalur pencarian sesuai kontrak ({channels_str}). Data atau rincian periode yang "
                f"tidak tercantum di atas memang belum tersedia secara terbuka pada dokumen sumber maupun "
                f"jaringan sitasi (backward/forward snowballing) yang dapat diakses saat ini."
            )

        should_revise = bool(has_hallucination or allow_snowball or accepted)
        orig_score = int(getattr(critique_report, "score", 8) or 8)
        orig_passed = bool(getattr(critique_report, "passed", True))

        if not should_revise:
            override_passed = True
            override_score = max(orig_score, 8)
            if rejected:
                summary = (
                    f"Contractor memveto {len(rejected)} komplain Devil's Advocate karena melanggar "
                    f"Kontrak Absolut ({contract.archetype}, budget {contract.effective_min_tokens}-"
                    f"{contract.effective_max_tokens} tok) atau pencarian telah tuntas."
                )
            else:
                summary = "Draf telah memenuhi Kontrak Absolut; tidak diperlukan revisi."
        else:
            override_passed = orig_passed and not has_hallucination
            override_score = orig_score
            summary = (
                f"Contractor menyetujui {len(accepted)} poin perbaikan terfokus dalam batas kontrak "
                f"(menolak {len(rejected)} poin di luar kontrak)."
            )

        return CritiqueAdjudication(
            should_revise=should_revise,
            allow_citation_mapping=allow_snowball,
            accepted_revisions=accepted,
            rejected_critiques=rejected,
            override_passed=override_passed,
            override_score=override_score,
            search_exhausted=search_exhausted,
            attempted_channels_summary=attempted_labels,
            exhausted_search_disclosure=disclosure,
            adjudication_summary=summary,
        )
