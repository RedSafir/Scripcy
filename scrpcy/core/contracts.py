"""Dynamic Task Contract & Adaptive Evaluation Rubric for Scrpcy.

Implements query-conditioned goal extraction, flexible Definition of Done (DoD),
and dynamic rubric generation inspired by Prometheus 2, G-Eval, and Auto-J.
Eliminates rigid intent pigeonholing and prevents false-positive critic penalties.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from pydantic import BaseModel, Field

from scrpcy.core.engine import LLMEngine, estimate_tokens


DEFAULT_SEARCH_CHANNELS: list[str] = [
    "vault_lokal",
    "web_resmi",
    "repositori_jurnal_multi_source",
    "reference_mapping_snowballing_backward_forward",
]

DEFAULT_SEARCH_EXHAUSTION_POLICY: str = (
    "Telusuri bertahap: (1) Vault Lokal, (2) Portal Web Resmi (WebScout), "
    "(3) Repositori Jurnal Multi-Source (OpenAlex/Semantic Scholar/CrossRef), dan "
    "(4) Reference Mapping + Snowballing (Backward & Forward Citations). "
    "Jika setelah seluruh jalur tersebut ditempuh data spesifik tetap tidak ditemukan, "
    "HENTIKAN pencarian berulang, tolak penalti Devil's Advocate, dan laporkan secara jujur batas ketersediaan data."
)


def _clean_json_payload(raw: str) -> str:
    clean = raw.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```[a-zA-Z]*\r?\n", "", clean)
        clean = re.sub(r"\r?\n```$", "", clean)
    return clean.strip()


def _ensure_str(val: Any, default: str = "") -> str:
    """Coerce LLM JSON field (str or list of str) into a clean string."""
    if isinstance(val, list):
        joined = "; ".join(str(x).strip() for x in val if str(x).strip())
        return joined or default
    if isinstance(val, str):
        return val.strip() or default
    if val is not None:
        return str(val).strip() or default
    return default


def _ensure_str_list(val: Any, default: list[str] | None = None) -> list[str]:
    """Coerce LLM JSON field (list of str or single str) into a list of strings."""
    fallback = list(default) if default else []
    if isinstance(val, list):
        items = [str(x).strip() for x in val if str(x).strip()]
        return items or fallback
    if isinstance(val, str):
        stripped = val.strip()
        if not stripped:
            return fallback
        parts = [p.strip(" -•\t") for p in re.split(r"(?:\r?\n|;)+", stripped) if p.strip(" -•\t")]
        return parts or [stripped]
    return fallback


def _normalize_archetype(raw_arch: Any, query: str) -> str:
    """Normalize LLM archetype string into one of the five canonical archetypes."""
    valid = {
        "DATA_FACTUAL",
        "FOCUSED_QA",
        "COMPARISON",
        "DIALECTICAL_CRITIQUE",
        "LITERATURE_SYNTHESIS",
    }
    text = str(raw_arch or "").strip().upper()
    if text in valid:
        return text
    if any(k in text for k in ("DATA", "FACT", "STAT", "EMPIR", "METRIC", "TABEL")):
        return "DATA_FACTUAL"
    if any(k in text for k in ("COMPAR", "VS", "VERSUS", "KOMPAR")):
        return "COMPARISON"
    if any(k in text for k in ("DIALECT", "CRITIQUE", "ARGUMENT", "DEBAT", "PRO")):
        return "DIALECTICAL_CRITIQUE"
    if any(k in text for k in ("LITERATURE", "SYNTHESIS", "REVIEW", "PUSTAKA", "BAB")):
        return "LITERATURE_SYNTHESIS"
    if _has_factual_data_signal(query.lower()):
        return "DATA_FACTUAL"
    return "FOCUSED_QA"


def _default_rubric_for_archetype(archetype: str) -> list[RubricCriterion]:
    """Return proportional evaluation rubric criteria for the given archetype."""
    if archetype == "DATA_FACTUAL":
        return [
            RubricCriterion(
                name="Ketepatan Data & Sitasi",
                weight=0.60,
                description="Kesesuaian angka/fakta dengan sumber vault/web resmi tanpa halusinasi.",
                score_5_benchmark="Data spesifik disajikan akurat dengan sitasi atau jujur menyatakan batas ketersediaan data.",
                score_3_benchmark="Data disajikan namun kurang rinci.",
                score_1_benchmark="Mengarang angka statistik yang tidak ada di sumber.",
            ),
            RubricCriterion(
                name="Keringkasan & Fokus",
                weight=0.40,
                description="Langsung menjawab poin data tanpa narasi bertele-tele.",
                score_5_benchmark="Padat, terstruktur (tabel/poin), dan tidak melebar ke topik lain.",
                score_3_benchmark="Terdapat pengantar yang agak panjang.",
                score_1_benchmark="Bertele-tele dan keluar dari topik data yang diminta.",
            ),
        ]
    return [
        RubricCriterion(
            name="Pemenuhan Tujuan & Fokus Cakupan",
            weight=0.50,
            description="Ketepatan menjawab primary_goal dan kepatuhan pada batas in_scope/out_of_scope.",
            score_5_benchmark="Menjawab langsung inti pertanyaan secara tajam tanpa melebar ke luar cakupan.",
            score_3_benchmark="Menjawab pertanyaan namun masih ada bagian yang kurang fokus.",
            score_1_benchmark="Melenceng dari tujuan utama atau melanggar batas out_of_scope.",
        ),
        RubricCriterion(
            name="Ketajaman Bukti & Kejujuran Akademis",
            weight=0.50,
            description="Klaim didukung sitasi sumber yang valid dan jujur melaporkan batas data.",
            score_5_benchmark="Seluruh klaim kunci bersitasi akurat dan proporsional terhadap bukti yang tersedia.",
            score_3_benchmark="Sebagian klaim kurang rujukan spesifik.",
            score_1_benchmark="Terdapat klaim spekulatif tanpa rujukan atau halusinasi sitasi.",
        ),
    ]


class RubricCriterion(BaseModel):
    """Dynamic evaluation criterion conditioned on user request."""

    name: str = Field(description="Name of criterion (e.g. Evidentiary Grounding, Structural Clarity).")
    weight: float = Field(default=0.25, ge=0.0, le=1.0, description="Relative importance weight.")
    description: str = Field(description="What this criterion assesses in the context of the user query.")
    score_5_benchmark: str = Field(description="Condition for full marks (Score 5).")
    score_3_benchmark: str = Field(description="Condition for partial/adequate marks (Score 3).")
    score_1_benchmark: str = Field(description="Condition for failing marks (Score 1).")


class DoDCheckItem(BaseModel):
    """Individual boolean-verifiable checklist item from Definition of Done."""

    criterion: str = Field(description="Observable condition to verify.")
    satisfied: bool = Field(default=True, description="True if condition is fulfilled in synthesis.")
    evidence_or_rationale: str = Field(default="", description="Brief justification for check result.")


class CriterionScore(BaseModel):
    """Score on a specific dynamic rubric criterion."""

    criterion_name: str
    score: int = Field(ge=1, le=5)
    justification: str = Field(default="")


class TaskContract(BaseModel):
    """Semantic execution contract derived dynamically from user query."""

    archetype: str = Field(
        default="FOCUSED_QA",
        description="Execution archetype: DATA_FACTUAL, FOCUSED_QA, COMPARISON, DIALECTICAL_CRITIQUE, LITERATURE_SYNTHESIS.",
    )
    primary_goal: str = Field(description="Core cognitive task requested by user without rigid enum reduction.")
    in_scope: list[str] = Field(default_factory=list, description="Key topics, entities, or concepts to analyze.")
    out_of_scope: list[str] = Field(default_factory=list, description="Aspects excluded, irrelevant, or explicitly forbidden.")
    forbidden_drift_topics: list[str] = Field(
        default_factory=list,
        description="Tangential sub-topics that must be stripped if generated.",
    )
    format_requirements: str = Field(default="Analisis akademis terstruktur", description="Structure, layout, and style requested.")
    min_words: int = Field(default=300, description="Minimum recommended word count.")
    max_words: int = Field(default=750, description="Maximum word ceiling to prevent token truncation.")
    tolerance_pct: float = Field(
        default=0.15,
        ge=0.0,
        le=0.5,
        description="Tolerance margin percentage for word limits (default 15%).",
    )
    allow_counter_arguments: bool = Field(
        default=False,
        description="True only when dialectical debate, critique, or trade-offs are requested.",
    )
    allow_honest_gap: bool = Field(
        default=True,
        description="True to permit concise acknowledgment of missing empirical data instead of theoretical padding.",
    )
    definition_of_done: list[str] = Field(
        default_factory=list,
        description="Atomic boolean checklist items representing concrete completion.",
    )
    is_modular_section_needed: bool = Field(
        default=False,
        description="True if requested output is genuinely extensive (e.g. whole multi-section thesis chapter).",
    )
    is_absolute: bool = Field(
        default=True,
        description="True: Contract serves as the absolute supreme boundary over Devil's Advocate and synthesis.",
    )
    target_output_tokens: int = Field(
        default=450,
        ge=60,
        le=8192,
        description="Estimated target output tokens determined by AI contract.",
    )
    token_margin_pct: float = Field(
        default=0.30,
        ge=0.05,
        le=0.50,
        description="Configurable margin of error percentage for output token budget (default 0.30 = +-30%).",
    )
    unlimited_tokens: bool = Field(
        default=False,
        description="Whether token budget limits are disabled (unlimited output tokens).",
    )
    required_search_channels: list[str] = Field(
        default_factory=lambda: list(DEFAULT_SEARCH_CHANNELS),
        description="Ordered search channels to exhaust before stopping further searches.",
    )
    search_exhaustion_policy: str = Field(
        default=DEFAULT_SEARCH_EXHAUSTION_POLICY,
        description="Explicit search depth and stopping condition when data remains unavailable.",
    )
    rubric: list[RubricCriterion] = Field(
        default_factory=list,
        description="Dynamic rubric criteria for evaluation.",
    )

    @property
    def effective_min_words(self) -> int:
        """Lower word limit with tolerance margin (e.g. 300 * 0.85 = 255)."""
        return int(self.min_words * (1.0 - self.tolerance_pct))

    @property
    def effective_max_words(self) -> int:
        """Upper word limit with tolerance margin (e.g. 500 * 1.15 = 575)."""
        return int(self.max_words * (1.0 + self.tolerance_pct))

    @property
    def effective_min_tokens(self) -> int:
        """Lower token limit with configured margin of error (default -30%)."""
        return max(40, int(round(self.target_output_tokens * (1.0 - self.token_margin_pct))))

    @property
    def effective_max_tokens(self) -> int:
        """Upper token limit with configured margin of error (default +30%), or 999999 if unlimited."""
        if self.unlimited_tokens:
            return 999_999
        return int(round(self.target_output_tokens * (1.0 + self.token_margin_pct)))

    def is_word_count_acceptable(self, actual_words: int) -> bool:
        """Verify whether actual word count is within tolerance margin."""
        return self.effective_min_words <= actual_words <= self.effective_max_words

    def is_token_count_acceptable(self, actual_tokens: int) -> bool:
        """Verify whether actual output token count falls within target +- margin of error."""
        if self.unlimited_tokens:
            return actual_tokens >= self.effective_min_tokens
        return self.effective_min_tokens <= actual_tokens <= self.effective_max_tokens


class AdaptiveCriticVerdict(BaseModel):
    """Proportional evaluation report from Devil's Advocate using Task Contract."""

    overall_pass: bool = Field(default=True, description="True if response adequately fulfills contract.")
    score: int = Field(default=8, ge=1, le=10, description="Score on standard 1-10 scale for engine telemetry.")
    passed: bool = Field(default=True, description="True if score >= 7 and no critical violations.")
    contract_adherence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Adherence to constraints & format.")
    dod_evaluation: list[DoDCheckItem] = Field(default_factory=list, description="Verification of Definition of Done.")
    in_scope_critiques: list[str] = Field(default_factory=list, description="Dialectical critique of claims inside in-scope.")
    proportional_refinements: list[str] = Field(default_factory=list, description="Actionable refinements without expanding scope.")
    weaknesses: list[str] = Field(default_factory=list, description="Identified analytical vulnerabilities.")
    counter_arguments: list[str] = Field(default_factory=list, description="Relevant counter-perspectives within scope.")
    recency_alert: str | None = Field(default=None, description="Alert if citations are outdated.")
    needs_citation_mapping: bool = Field(default=False, description="True if additional literature is strictly needed.")
    critique_summary: str = Field(default="", description="Reviewer summary verdict.")
    input_tokens: int = Field(default=0)
    output_tokens: int = Field(default=0)


INTENT_CATEGORIES: dict[str, list[str]] = {
    "comparison": ["bandingkan", "komparasi", "perbandingan", "vs", "versus", "perbedaan"],
    "argument": ["argumen", "argument", "pro kontra", "sisi positif", "sisi negatif", "sudut pandang", "tesis", "dialektika", "debat"],
    "fact": [
        "berapa", "apakah benar", "kapan", "validkah", "siapa",
        "data statistik", "carikan data", "cari data", "data ", "statistik",
        "angka", "rekap", "rekapitulasi", "tabel", "jumlah", "tren",
    ],
    "literature_review": ["tinjauan pustaka", "bab 2", "bab 1", "bab 3", "draf lengkap", "sintesis komprehensif"],
}


def _has_factual_data_signal(lower_q: str) -> bool:
    """Return True when the query primarily requests empirical data, metrics, or factual verification."""
    word_tokens = set(re.findall(r"\b\w+\b", lower_q))
    if word_tokens & {"berapa", "kapan", "siapa", "validkah", "data", "statistik", "angka", "rekap", "rekapitulasi", "tabel"}:
        return True
    return any(p in lower_q for p in ("apakah benar", "data statistik", "lima tahun terakhir", "tahun terakhir"))


class TaskContractExtractor:
    """Extracts an AI-First Absolute TaskContract and evaluation rubric from user query."""

    def __init__(
        self,
        engine: LLMEngine | None = None,
        llm_completion_fn: Callable[[str, str], dict[str, Any]] | None = None,
    ) -> None:
        self.engine = engine
        self.llm_completion_fn = llm_completion_fn

    def _configured_margin(self) -> float:
        """Return configured token margin of error percentage (default 0.30 = +-30%)."""
        cfg = getattr(self.engine, "config", None) if self.engine else None
        raw = getattr(cfg, "contract_token_margin_pct", 0.30) if cfg else 0.30
        try:
            val = float(raw)
            return max(0.05, min(0.50, val))
        except Exception:
            return 0.30

    def _configured_unlimited(self) -> bool:
        """Return configured unlimited token setting."""
        cfg = getattr(self.engine, "config", None) if self.engine else None
        return bool(getattr(cfg, "contract_unlimited_tokens", False))

    def is_ambiguous_intent(self, query: str) -> bool:
        """Determine if query contains competing signals across multiple intent categories."""
        lower_q = query.lower()
        matched = [
            cat for cat, keywords in INTENT_CATEGORIES.items()
            if any(k in lower_q for k in keywords)
        ]
        return len(matched) >= 2

    def _extract_via_llm(self, query: str) -> TaskContract | None:
        """Formulate an AI-First Absolute TaskContract with token budget and search-depth rules."""
        margin_pct = self._configured_margin()
        unlimited = self._configured_unlimited()
        system_prompt = (
            "Anda adalah AI Contract Architect untuk Scrpcy.\n"
            "Rumuskan KONTRAK TUGAS ABSOLUT berdasarkan kebutuhan pengguna. "
            "Kontrak ini bersifat ABSOLUT: Devil's Advocate maupun penulis draf dilarang melanggar batas cakupan.\n"
            "Balas HANYA dalam JSON valid (tanpa teks lain) dengan field berikut:\n"
            "- archetype: string, WAJIB salah satu dari ['DATA_FACTUAL', 'FOCUSED_QA', 'COMPARISON', 'DIALECTICAL_CRITIQUE', 'LITERATURE_SYNTHESIS']\n"
            "- primary_goal: string (tujuan inti yang menjawab kebutuhan pengguna)\n"
            "- in_scope: array of strings (2-4 poin fokus yang wajib dijawab)\n"
            "- out_of_scope: array of strings (2-3 pembahasan di luar konteks)\n"
            "- forbidden_drift_topics: array of strings (2-3 topik melebar yang wajib ditolak dari Devil's Advocate)\n"
            "- format_requirements: string tunggal (format penyajian terbaik, misal tabel + ringkasan sumber)\n"
            "- target_output_tokens: integer (estimasi token output ideal: 250-350 untuk data singkat, 450 untuk QA fokus, 600 untuk komparasi, 1200 untuk bab pustaka)\n"
            "- min_words: integer (50 s/d 900)\n"
            "- max_words: integer (250 s/d 1500)\n"
            "- allow_counter_arguments: boolean (true HANYA jika kueri meminta debat/pro-kontra/komparasi)\n"
            "- allow_honest_gap: boolean (true)\n"
            "- definition_of_done: array of strings (3-4 kriteria selesai yang terukur)\n"
            "- required_search_channels: array of strings dari ['vault_lokal', 'web_resmi', 'repositori_jurnal_multi_source', 'reference_mapping_snowballing_backward_forward']\n"
            "- search_exhaustion_policy: string tunggal (aturan batas pencarian via vault, web resmi, jurnal, dan reference mapping + snowballing forward/backward, serta berhenti mencari jika data tetap tidak ditemukan)\n"
            "- is_modular_section_needed: boolean"
        )

        try:
            data: dict[str, Any] | None = None
            if self.llm_completion_fn is not None:
                data = self.llm_completion_fn(system_prompt, query)
            else:
                if not self.engine or not getattr(self.engine, "config", None):
                    return None
                cfg = getattr(self.engine, "config", None)
                model_name = getattr(cfg, "default_model", None) if cfg else getattr(self.engine, "default_model", "gemini/gemini-2.5-flash")
                if not isinstance(model_name, str):
                    model_name = str(model_name)
                api_key = self.engine.config.get_api_key_for_model(model_name)
                if (
                    not api_key
                    or str(api_key).startswith("mock")
                    or getattr(self.engine, "_acompletion_fn", None) is not None
                ):
                    return None

                import litellm
                litellm.suppress_debug_info = True
                litellm.set_verbose = False
                litellm.drop_params = True

                effective_model = model_name
                fast_contract_model = model_name
                api_base = None
                if model_name.startswith("9router/"):
                    submodel = model_name.split("/", 1)[1]
                    effective_model = f"openai/{submodel}"
                    # Use the low-thinking variant on 9router for fast control-plane JSON contract extraction
                    if submodel.endswith("-medium"):
                        fast_contract_model = f"openai/{submodel[:-7]}-low"
                    elif submodel.endswith("-high"):
                        fast_contract_model = f"openai/{submodel[:-5]}-low"
                    else:
                        fast_contract_model = effective_model
                    api_bases = getattr(self.engine.config, "api_bases", {}) or {}
                    api_base = api_bases.get("9router", "http://localhost:20128/v1")

                base_kwargs: dict[str, Any] = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"USER QUERY: {query}"},
                    ],
                    "api_key": api_key,
                    "temperature": 0.1,
                    "max_tokens": 450,
                    "response_format": {"type": "json_object"},
                    "timeout": 15,
                }
                if api_base:
                    base_kwargs["api_base"] = api_base

                try:
                    resp = litellm.completion(model=fast_contract_model, **base_kwargs)
                except Exception:
                    if fast_contract_model != effective_model:
                        resp = litellm.completion(model=effective_model, **base_kwargs)
                    else:
                        raise

                content = resp.choices[0].message.content or "{}"
                data = json.loads(_clean_json_payload(content))

            if not isinstance(data, dict) or not data:
                return None

            archetype = _normalize_archetype(data.get("archetype"), query)

            default_tok_map = {
                "DATA_FACTUAL": 280,
                "FOCUSED_QA": 420,
                "DIALECTICAL_CRITIQUE": 520,
                "COMPARISON": 600,
                "LITERATURE_SYNTHESIS": 1200,
            }
            default_words_map = {
                "DATA_FACTUAL": (60, 350),
                "FOCUSED_QA": (180, 550),
                "DIALECTICAL_CRITIQUE": (300, 650),
                "COMPARISON": (350, 800),
                "LITERATURE_SYNTHESIS": (700, 1400),
            }
            def_min_w, def_max_w = default_words_map.get(archetype, (180, 550))

            try:
                target_tokens = int(data.get("target_output_tokens") or default_tok_map.get(archetype, 420))
            except Exception:
                target_tokens = default_tok_map.get(archetype, 420)
            target_tokens = max(60, min(8192, target_tokens))

            try:
                min_words = int(data.get("min_words") or def_min_w)
            except Exception:
                min_words = def_min_w
            try:
                max_words = int(data.get("max_words") or def_max_w)
            except Exception:
                max_words = def_max_w
            if max_words < min_words:
                max_words = max(min_words + 100, def_max_w)

            raw_channels = _ensure_str_list(data.get("required_search_channels"), DEFAULT_SEARCH_CHANNELS)
            canonical_set = set(DEFAULT_SEARCH_CHANNELS)
            req_channels = [ch for ch in raw_channels if ch in canonical_set]
            for ch in DEFAULT_SEARCH_CHANNELS:
                if ch not in req_channels:
                    req_channels.append(ch)

            exhaustion_policy = _ensure_str(
                data.get("search_exhaustion_policy"),
                DEFAULT_SEARCH_EXHAUSTION_POLICY,
            )

            return TaskContract(
                archetype=archetype,
                primary_goal=_ensure_str(data.get("primary_goal"), f"Menjawab secara akurat: {query.strip()}"),
                in_scope=_ensure_str_list(data.get("in_scope"), [query.strip()]),
                out_of_scope=_ensure_str_list(
                    data.get("out_of_scope"),
                    ["Pembahasan teoretis umum di luar fokus pertanyaan"],
                ),
                forbidden_drift_topics=_ensure_str_list(
                    data.get("forbidden_drift_topics"),
                    ["Perluasan topik yang tidak diminta pengguna"],
                ),
                format_requirements=_ensure_str(
                    data.get("format_requirements"),
                    "Penyajian terstruktur, langsung pada inti data/analisis, dengan sitasi sumber.",
                ),
                min_words=min_words,
                max_words=max_words,
                allow_counter_arguments=bool(
                    data.get("allow_counter_arguments", archetype in ("DIALECTICAL_CRITIQUE", "COMPARISON"))
                ),
                allow_honest_gap=bool(data.get("allow_honest_gap", True)),
                definition_of_done=_ensure_str_list(
                    data.get("definition_of_done"),
                    [
                        "Menjawab langsung fokus utama pertanyaan pengguna.",
                        "Menyertakan rujukan sumber/sitasi yang valid.",
                        "Jujur melaporkan batas ketersediaan data jika sumber tidak memuat rincian spesifik.",
                    ],
                ),
                is_modular_section_needed=bool(data.get("is_modular_section_needed", False)),
                is_absolute=True,
                target_output_tokens=target_tokens,
                token_margin_pct=margin_pct,
                unlimited_tokens=unlimited,
                required_search_channels=req_channels,
                search_exhaustion_policy=exhaustion_policy,
                rubric=_default_rubric_for_archetype(archetype),
            )
        except Exception:
            return None

    def _extract_topics(self, text: str) -> list[str]:
        """Extract clean core topics from text."""
        cleaned = re.sub(r"[^\w\s-]", " ", text)
        words = [w for w in cleaned.split() if len(w) > 3]
        stopwords = {
            "yang", "untuk", "dengan", "dalam", "pada", "oleh", "dari", "tentang",
            "adalah", "sebagai", "dapat", "akan", "telah", "bisa", "aku", "saya",
            "kamu", "anda", "tolong", "buatkan", "carikan", "analisis", "penelitianku",
            "penelitian", "tentang", "mengenai", "indonesia", "riset", "skripsi",
            "bagaimana", "apakah", "mengapa", "seperti", "sudah", "lebih",
        }
        filtered = [w for w in words if w.lower() not in stopwords]
        seen: set[str] = set()
        topics: list[str] = []
        for t in filtered:
            tl = t.lower()
            if tl not in seen:
                seen.add(tl)
                topics.append(t)
            if len(topics) >= 5:
                break
        return topics

    def extract_heuristic(self, query: str) -> TaskContract:
        """Rule-based zero-latency fallback contract extraction."""
        q_clean = query.strip()
        lower_q = q_clean.lower()
        word_count = len(q_clean.split())
        topics = self._extract_topics(q_clean)
        topic_str = ", ".join(topics[:3]) if topics else q_clean[:40]
        margin_pct = self._configured_margin()
        unlimited = self._configured_unlimited()

        if any(k in lower_q for k in INTENT_CATEGORIES["comparison"]) and any(k in lower_q for k in INTENT_CATEGORIES["literature_review"]):
            return TaskContract(
                archetype="COMPARISON",
                primary_goal=f"Analisis komparatif metodologis {topic_str} untuk integrasi tinjauan pustaka (Bab 2)",
                in_scope=[
                    f"Parameter komparasi antar metode/teori dalam {topic_str}",
                    "Kelebihan, limitasi, dan trade-off untuk landasan teoritis",
                    "Sintesis kritis posisi penelitian terhadap metode yang dibandingkan",
                ],
                out_of_scope=["Penulisan bab komprehensif monolitik tanpa komparasi", "Ulasan sejarah panjang yang tidak relevan"],
                forbidden_drift_topics=["Ulasan latar belakang umum tanpa parameter perbandingan"],
                format_requirements="Tinjauan komparatif akademis terstruktur (matriks komparasi + narasi sintesis kritis).",
                min_words=450,
                max_words=1000,
                target_output_tokens=720,
                token_margin_pct=margin_pct,
                unlimited_tokens=unlimited,
                is_absolute=True,
                allow_counter_arguments=True,
                allow_honest_gap=True,
                definition_of_done=[
                    "Menetapkan parameter komparasi yang jelas dan relevan.",
                    "Menyajikan perbandingan berimbang dengan sitasi bukti empiris vault.",
                    "Menyajikan sintesis kritis atau arahan pemilihan metode untuk bab skripsi.",
                ],
                is_modular_section_needed=False,
                rubric=[
                    RubricCriterion(
                        name="Ketajaman Komparasi Teoretis",
                        weight=0.45,
                        description="Kedalaman komparasi parameter metode untuk tinjauan pustaka.",
                        score_5_benchmark="Parameter tajam, evaluatif, dan relevan dengan konteks skripsi.",
                        score_3_benchmark="Hanya komparasi deskriptif superfisial.",
                        score_1_benchmark="Gagal membandingkan metode secara terarah.",
                    ),
                    RubricCriterion(
                        name="Rigor Bukti Vault",
                        weight=0.40,
                        description="Klaim perbandingan ditopang sitasi vault akurat.",
                        score_5_benchmark="Setiap metrik perbandingan merujuk sitasi vault.",
                        score_3_benchmark="Sebagian klaim komparasi tidak bersitasi.",
                        score_1_benchmark="Tanpa sitasi atau terdapat halusinasi.",
                    ),
                ],
            )

        if any(k in lower_q for k in INTENT_CATEGORIES["argument"]):
            is_brief = any(k in lower_q for k in ["singkat", "ringkas", "pendek", "brief"])
            min_w = 60 if is_brief else 350
            max_w = 400 if is_brief else 650
            target_tok = 280 if is_brief else 520
            return TaskContract(
                archetype="DIALECTICAL_CRITIQUE",
                primary_goal=f"Eksplorasi dialektika argumen seputar {topic_str}",
                in_scope=[f"Argumen pro dan landasan bukti empiris {topic_str}", f"Sanggahan kontra dan limitasi {topic_str}", "Tesis sentral dan celah riset"],
                out_of_scope=["Penulisan bab komprehensif monolitik", "Penurunan rumus matematis yang tidak diminta", "Pengantar buku teks umum"],
                forbidden_drift_topics=["Esai naratif satu arah tanpa matriks pro-kontra"],
                format_requirements="Matriks Dialektika: Tesis Sentral, Poin Pendukung (Pro + sitasi), Poin Sanggahan (Kontra + sitasi), dan Celah Riset (Gap).",
                min_words=min_w,
                max_words=max_w,
                target_output_tokens=target_tok,
                token_margin_pct=margin_pct,
                unlimited_tokens=unlimited,
                is_absolute=True,
                allow_counter_arguments=True,
                allow_honest_gap=True,
                definition_of_done=[
                    "Menyajikan tesis sentral yang jelas (1-2 kalimat).",
                    "Menyajikan minimal 2 argumen pendukung dengan rujukan sitasi vault.",
                    "Menyajikan minimal 2 sanggahan/kondisi pembatas dengan rujukan sitasi vault.",
                    "Menyajikan minimal 1 pertanyaan celah riset terbuka.",
                    "Panjang respons berada dalam batas aman (tidak terpotong).",
                ],
                is_modular_section_needed=False,
                rubric=[
                    RubricCriterion(
                        name="Keseimbangan Dialektika",
                        weight=0.35,
                        description="Keseimbangan antara poin pendukung dan sanggahan kritis.",
                        score_5_benchmark="Menyajikan kedua sisi (pro dan kontra) secara berbobot dan adil.",
                        score_3_benchmark="Hanya fokus pada satu sisi dengan sanggahan minimal.",
                        score_1_benchmark="Hanya menyajikan opini sepihak tanpa sanggahan.",
                    ),
                    RubricCriterion(
                        name="Kepatuhan Batas Format",
                        weight=0.30,
                        description="Kepatuhan terhadap format matriks dan keringkasan padat.",
                        score_5_benchmark="Tersusun rapi dalam format matriks tanpa esai monolitik.",
                        score_3_benchmark="Format agak melebar namun masih terstruktur.",
                        score_1_benchmark="Esai monolitik panjang yang melanggar batas kata.",
                    ),
                    RubricCriterion(
                        name="Validitas Bukti Vault",
                        weight=0.35,
                        description="Klaim didukung oleh sitasi dokumen vault yang valid.",
                        score_5_benchmark="Semua klaim substantif memiliki sitasi vault yang akurat.",
                        score_3_benchmark="Sebagian klaim tidak memiliki sitasi langsung.",
                        score_1_benchmark="Klaim tanpa sitasi atau halusinasi.",
                    ),
                ],
            )

        if any(k in lower_q for k in ["bandingkan", "komparasi", "perbandingan", "vs", "versus", "perbedaan"]):
            return TaskContract(
                archetype="COMPARISON",
                primary_goal=f"Analisis komparasi dan perbandingan analitis {topic_str}",
                in_scope=[f"Karakteristik dan keunggulan masing-masing entitas dalam {topic_str}", "Parameter perbandingan", "Trade-off praktis/metodologis"],
                out_of_scope=["Ulasan sejarah panjang yang tidak relevan", "Opini subjektif tanpa data"],
                forbidden_drift_topics=["Tinjauan pustaka umum di luar entitas yang dibandingkan"],
                format_requirements="Analisis komparatif terstruktur (disarankan tabel perbandingan atau poin-poin parameter komparasi).",
                min_words=300,
                max_words=700,
                target_output_tokens=560,
                token_margin_pct=margin_pct,
                unlimited_tokens=unlimited,
                is_absolute=True,
                allow_counter_arguments=True,
                allow_honest_gap=True,
                definition_of_done=[
                    "Menetapkan parameter komparasi yang jelas.",
                    "Membandingkan entitas secara seimbang pada setiap parameter.",
                    "Menyertakan sitasi pendukung untuk data komparasi.",
                    "Menyajikan ringkasan sintesis trade-off.",
                ],
                is_modular_section_needed=False,
                rubric=[
                    RubricCriterion(
                        name="Ketajaman Komparasi",
                        weight=0.40,
                        description="Perbandingan berbasis parameter objektif.",
                        score_5_benchmark="Parameter komparasi tajam dan kontras terpetakan jelas.",
                        score_3_benchmark="Perbandingan bersifat deskriptif umum.",
                        score_1_benchmark="Tidak membandingkan parameter kunci.",
                    ),
                    RubricCriterion(
                        name="Bukti Sitasi",
                        weight=0.30,
                        description="Rujukan empiris dari vault.",
                        score_5_benchmark="Sitasi relevan mendukung setiap data perbandingan.",
                        score_3_benchmark="Sebagian perbandingan tidak bersitasi.",
                        score_1_benchmark="Tanpa sitasi pendukung.",
                    ),
                ],
            )

        if any(k in lower_q for k in ["tinjauan pustaka", "bab 2", "bab 1", "bab 3", "draf lengkap", "sintesis komprehensif"]):
            return TaskContract(
                archetype="LITERATURE_SYNTHESIS",
                primary_goal=f"Penyusunan tinjauan pustaka dan sintesis komprehensif {topic_str}",
                in_scope=[f"Kerangka teoritis {topic_str}", f"Temuan empiris mutakhir {topic_str}", "Sintesis konseptual dan integrasi temuan"],
                out_of_scope=["Klaim tanpa landasan literatur", "Bahasa informal"],
                format_requirements="Tinjauan akademis terstruktur dengan sub-heading bertahap.",
                min_words=800,
                max_words=1600,
                target_output_tokens=1200,
                token_margin_pct=margin_pct,
                unlimited_tokens=unlimited,
                is_absolute=True,
                allow_counter_arguments=True,
                allow_honest_gap=True,
                definition_of_done=[
                    "Menyusun kerangka sub-topik yang koheren.",
                    "Menyintesis minimal 3-5 temuan paper aktif di vault.",
                    "Setiap klaim metodologis dan konseptual memiliki sitasi APA.",
                    "Menyajikan sintesis kritis, bukan sekadar ringkasan terpisah per paper.",
                ],
                is_modular_section_needed=word_count > 60 or "lengkap" in lower_q or "komprehensif" in lower_q,
                rubric=[
                    RubricCriterion(
                        name="Kedalaman Sintesis",
                        weight=0.40,
                        description="Kepaduan sintesis lintas literatur.",
                        score_5_benchmark="Sintesis integratif mendalam antar paper vault.",
                        score_3_benchmark="Hanya merangkum paper satu per satu.",
                        score_1_benchmark="Ringkasan dangkal tanpa integrasi.",
                    ),
                    RubricCriterion(
                        name="Rigour & Sitasi",
                        weight=0.35,
                        description="Ketelitian sitasi dan klaim ilmiah.",
                        score_5_benchmark="Seluruh paragraf memiliki rujukan sitasi akurat.",
                        score_3_benchmark="Sebagian paragraf substantif tidak bersitasi.",
                        score_1_benchmark="Klaim spekulatif tanpa sitasi.",
                    ),
                ],
            )

        if _has_factual_data_signal(lower_q):
            return TaskContract(
                archetype="DATA_FACTUAL",
                primary_goal=f"Verifikasi data dan fakta spesifik mengenai {topic_str}",
                in_scope=[f"Angka, data, atau fakta terverifikasi terkait {topic_str}", "Sumber dan cakupan tahun/wilayah data"],
                out_of_scope=["Narasi esai panjang", "Spekulasi tanpa data vault", "Tinjauan pustaka melebar"],
                forbidden_drift_topics=[
                    "Sub-bab tinjauan pustaka Bab 2 yang tidak diminta",
                    "Perdebatan teoretis atau birokrasi di luar angka/data yang ditanyakan",
                ],
                format_requirements="Jawaban langsung, ringkas, dan presisi (poin/tabel data disertai sitasi, maksimal 1-3 paragraf pendukung).",
                min_words=50,
                max_words=350,
                target_output_tokens=280,
                token_margin_pct=margin_pct,
                unlimited_tokens=unlimited,
                is_absolute=True,
                allow_counter_arguments=False,
                allow_honest_gap=True,
                definition_of_done=[
                    "Menjawab pertanyaan data/fakta secara langsung di awal.",
                    "Menyertakan angka/metrik spesifik jika tersedia dalam sumber.",
                    "Menyertakan sitasi sumber yang memuat data tersebut.",
                    "Menyebutkan secara jujur dan singkat jika cakupan tahun/wilayah di sumber belum lengkap tanpa mengarang teori.",
                ],
                is_modular_section_needed=False,
                rubric=[
                    RubricCriterion(
                        name="Akurasi Data Faktual",
                        weight=0.50,
                        description="Kesesuaian data dengan isi dokumen sumber.",
                        score_5_benchmark="Data 100% akurat dan terverifikasi di dokumen sumber.",
                        score_3_benchmark="Data ada namun cakupan tahun/wilayah sebagian.",
                        score_1_benchmark="Data keliru atau halusinasi angka.",
                    ),
                    RubricCriterion(
                        name="Keringkasan Presisi",
                        weight=0.50,
                        description="Menjawab langsung tanpa esai Bab 2 yang melebar.",
                        score_5_benchmark="Padat, langsung ke inti jawaban dan data.",
                        score_3_benchmark="Terdapat penjelasan konteks singkat.",
                        score_1_benchmark="Bertele-tele menjadi bab tinjauan pustaka tanpa fokus data.",
                    ),
                ],
            )

        return TaskContract(
            archetype="FOCUSED_QA",
            primary_goal=f"Analisis akademis kontekstual mengenai: {q_clean[:100]}",
            in_scope=[f"Konsep dan temuan relevan terkait {topic_str}", "Bukti ilmiah langsung dari literatur"],
            out_of_scope=["Spekulasi di luar literatur", "Penyimpangan topik"],
            forbidden_drift_topics=["Pembahasan melebar di luar fokus pertanyaan"],
            format_requirements="Respons akademis terfokus, jelas, dan langsung menjawab inti pertanyaan.",
            min_words=50,
            max_words=600,
            target_output_tokens=420,
            token_margin_pct=margin_pct,
            unlimited_tokens=unlimited,
            is_absolute=True,
            allow_counter_arguments=False,
            allow_honest_gap=True,
            definition_of_done=[
                "Menjawab inti pertanyaan secara substantif.",
                "Mengaitkan jawaban dengan literatur vault jika tersedia.",
                "Menjaga keterbacaan dan struktur penjelasan yang logis.",
            ],
            is_modular_section_needed=False,
            rubric=[
                RubricCriterion(
                    name="Relevansi Substantif",
                    weight=0.50,
                    description="Kesesuaian substansi dengan pertanyaan.",
                    score_5_benchmark="Menjawab tuntas dan relevan dengan pertanyaan.",
                    score_3_benchmark="Menjawab sebagian aspek namun ada yang terlewat.",
                    score_1_benchmark="Melenceng dari pertanyaan pengguna.",
                ),
                RubricCriterion(
                    name="Dasar Bukti Ilmiah",
                    weight=0.50,
                    description="Keabsahan dasar ilmiah dan sitasi.",
                    score_5_benchmark="Argumen logis didukung bukti literatur.",
                    score_3_benchmark="Argumen logis namun minim rujukan spesifik.",
                    score_1_benchmark="Argumen spekulatif.",
                ),
            ],
        )

    def extract(self, query: str) -> TaskContract:
        """Extract an AI-First Absolute TaskContract, falling back to heuristics when offline/mock."""
        if self.llm_completion_fn is not None or self.engine is not None:
            ai_contract = self._extract_via_llm(query)
            if ai_contract:
                return ai_contract

        return self.extract_heuristic(query)
