"""Dynamic Task Contract & Adaptive Evaluation Rubric for Scrpcy.

Implements query-conditioned goal extraction, flexible Definition of Done (DoD),
and dynamic rubric generation inspired by Prometheus 2, G-Eval, and Auto-J.
Eliminates rigid intent pigeonholing and prevents false-positive critic penalties.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from scrpcy.core.engine import LLMEngine, estimate_tokens


def _clean_json_payload(raw: str) -> str:
    clean = raw.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```[a-zA-Z]*\r?\n", "", clean)
        clean = re.sub(r"\r?\n```$", "", clean)
    return clean.strip()


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

    primary_goal: str = Field(description="Core cognitive task requested by user without rigid enum reduction.")
    in_scope: list[str] = Field(default_factory=list, description="Key topics, entities, or concepts to analyze.")
    out_of_scope: list[str] = Field(default_factory=list, description="Aspects excluded, irrelevant, or explicitly forbidden.")
    format_requirements: str = Field(default="Analisis akademis terstruktur", description="Structure, layout, and style requested.")
    min_words: int = Field(default=300, description="Minimum recommended word count.")
    max_words: int = Field(default=750, description="Maximum word ceiling to prevent token truncation.")
    tolerance_pct: float = Field(
        default=0.15,
        ge=0.0,
        le=0.5,
        description="Tolerance margin percentage for word limits (default 15%).",
    )
    definition_of_done: list[str] = Field(
        default_factory=list,
        description="Atomic boolean checklist items representing concrete completion.",
    )
    is_modular_section_needed: bool = Field(
        default=False,
        description="True if requested output is genuinely extensive (e.g. whole multi-section thesis chapter).",
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

    def is_word_count_acceptable(self, actual_words: int) -> bool:
        """Verify whether actual word count is within tolerance margin."""
        return self.effective_min_words <= actual_words <= self.effective_max_words


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
    "fact": ["berapa", "apakah benar", "kapan", "validkah", "siapa", "data statistik"],
    "literature_review": ["tinjauan pustaka", "bab 2", "bab 1", "bab 3", "draf lengkap", "sintesis komprehensif"],
}


class TaskContractExtractor:
    """Extracts dynamic execution contract and evaluation rubric from user query."""

    def __init__(self, engine: LLMEngine | None = None) -> None:
        self.engine = engine

    def is_ambiguous_intent(self, query: str) -> bool:
        """Determine if query contains competing signals across multiple intent categories."""
        lower_q = query.lower()
        matched = [
            cat for cat, keywords in INTENT_CATEGORIES.items()
            if any(k in lower_q for k in keywords)
        ]
        return len(matched) >= 2

    def _extract_via_llm(self, query: str) -> TaskContract | None:
        """Disambiguate overlapping user intents and formulate a harmonized contract via LLMEngine."""
        if not self.engine or not getattr(self.engine, "config", None):
            return None
        try:
            model_name = self.engine.config.default_model
            api_key = self.engine.config.get_api_key_for_model(model_name)
            if not api_key or api_key.startswith("mock") or getattr(self.engine, "_acompletion_fn", None) is not None:
                return None

            import litellm
            litellm.suppress_debug_info = True
            litellm.set_verbose = False
            litellm.drop_params = True

            effective_model = model_name
            api_base = None
            if model_name.startswith("9router/"):
                submodel = model_name.split("/", 1)[1]
                effective_model = f"openai/{submodel}"
                api_base = "http://localhost:20128/v1"

            system_prompt = (
                "You are an elite academic research supervisor and contract designer.\n"
                "The user query contains multiple or competing cognitive intents (e.g. comparing methods while writing a chapter draft).\n"
                "Disambiguate the core objective, balance the priorities, and formulate an explicit Task Contract.\n"
                "Respond in strict JSON with:\n"
                "- primary_goal: string\n"
                "- in_scope: list of 2-4 strings\n"
                "- out_of_scope: list of 2-3 strings\n"
                "- format_requirements: string\n"
                "- min_words: integer (e.g. 350 to 900)\n"
                "- max_words: integer (e.g. 700 to 1400)\n"
                "- definition_of_done: list of 3-5 concrete boolean criteria strings\n"
                "- is_modular_section_needed: boolean"
            )

            kwargs: dict[str, Any] = {
                "model": effective_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"USER QUERY: {query}"},
                ],
                "api_key": api_key,
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
                "timeout": 15,
            }
            if api_base:
                kwargs["api_base"] = api_base

            resp = litellm.completion(**kwargs)
            content = resp.choices[0].message.content or "{}"
            data = json.loads(_clean_json_payload(content))

            return TaskContract(
                primary_goal=data.get("primary_goal", f"Analisis terpadu: {query[:80]}"),
                in_scope=data.get("in_scope", []),
                out_of_scope=data.get("out_of_scope", []),
                format_requirements=data.get("format_requirements", "Analisis akademis terstruktur"),
                min_words=int(data.get("min_words", 350)),
                max_words=int(data.get("max_words", 800)),
                definition_of_done=data.get("definition_of_done", []),
                is_modular_section_needed=bool(data.get("is_modular_section_needed", False)),
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
        """Rule-based zero-latency dynamic contract extraction."""
        q_clean = query.strip()
        lower_q = q_clean.lower()
        word_count = len(q_clean.split())
        topics = self._extract_topics(q_clean)
        topic_str = ", ".join(topics[:3]) if topics else q_clean[:40]

        # Case 0: Ambiguous Multi-Intent Heuristic Synthesis (e.g., Comparison within Literature Review / Bab 2)
        if any(k in lower_q for k in INTENT_CATEGORIES["comparison"]) and any(k in lower_q for k in INTENT_CATEGORIES["literature_review"]):
            return TaskContract(
                primary_goal=f"Analisis komparatif metodologis {topic_str} untuk integrasi tinjauan pustaka (Bab 2)",
                in_scope=[
                    f"Parameter komparasi antar metode/teori dalam {topic_str}",
                    "Kelebihan, limitasi, dan trade-off untuk landasan teoritis",
                    "Sintesis kritis posisi penelitian terhadap metode yang dibandingkan",
                ],
                out_of_scope=["Penulisan bab komprehensif monolitik tanpa komparasi", "Ulasan sejarah panjang yang tidak relevan"],
                format_requirements="Tinjauan komparatif akademis terstruktur (matriks komparasi + narasi sintesis kritis).",
                min_words=450,
                max_words=1000,
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

        # Case 1: Argument / Debate / Dialectic exploration
        if any(k in lower_q for k in INTENT_CATEGORIES["argument"]):
            is_brief = any(k in lower_q for k in ["singkat", "ringkas", "pendek", "brief"])
            min_w = 60 if is_brief else 350
            max_w = 400 if is_brief else 650
            return TaskContract(
                primary_goal=f"Eksplorasi dialektika argumen seputar {topic_str}",
                in_scope=[f"Argumen pro dan landasan bukti empiris {topic_str}", f"Sanggahan kontra dan limitasi {topic_str}", "Tesis sentral dan celah riset"],
                out_of_scope=["Penulisan bab komprehensif monolitik", "Penurunan rumus matematis yang tidak diminta", "Pengantar buku teks umum"],
                format_requirements="Matriks Dialektika: Tesis Sentral, Poin Pendukung (Pro + sitasi), Poin Sanggahan (Kontra + sitasi), dan Celah Riset (Gap).",
                min_words=min_w,
                max_words=max_w,
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

        # Case 2: Comparison / Trade-off analysis
        if any(k in lower_q for k in ["bandingkan", "komparasi", "perbandingan", "vs", "versus", "perbedaan"]):
            return TaskContract(
                primary_goal=f"Analisis komparasi dan perbandingan analitis {topic_str}",
                in_scope=[f"Karakteristik dan keunggulan masing-masing entitas dalam {topic_str}", "Parameter perbandingan", "Trade-off praktis/metodologis"],
                out_of_scope=["Ulasan sejarah panjang yang tidak relevan", "Opini subjektif tanpa data"],
                format_requirements="Analisis komparatif terstruktur (disarankan tabel perbandingan atau poin-poin parameter komparasi).",
                min_words=300,
                max_words=700,
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

        # Case 3: Fact / Data / Metric verification
        if any(k in lower_q for k in ["berapa", "apakah benar", "kapan", "validkah", "siapa", "data statistik"]):
            return TaskContract(
                primary_goal=f"Verifikasi data dan fakta spesifik mengenai {topic_str}",
                in_scope=[f"Angka, data, atau fakta terverifikasi terkait {topic_str}", "Konteks metodologi pengambilan data"],
                out_of_scope=["Narasi esai panjang", "Spekulasi tanpa data vault", "Tinjauan pustaka melebar"],
                format_requirements="Jawaban langsung, ringkas, dan presisi (1-3 paragraf padat).",
                min_words=100,
                max_words=400,
                definition_of_done=[
                    "Menjawab pertanyaan data/fakta secara langsung di awal.",
                    "Menyertakan angka/metrik spesifik jika tersedia dalam vault.",
                    "Menyertakan sitasi vault yang memuat data tersebut.",
                ],
                is_modular_section_needed=False,
                rubric=[
                    RubricCriterion(
                        name="Akurasi Data Faktual",
                        weight=0.50,
                        description="Kesesuaian data dengan isi dokumen vault.",
                        score_5_benchmark="Data 100% akurat dan terverifikasi di dokumen vault.",
                        score_3_benchmark="Data ada namun konteksnya kurang lengkap.",
                        score_1_benchmark="Data keliru atau halusinasi angka.",
                    ),
                    RubricCriterion(
                        name="Keringkasan Presisi",
                        weight=0.50,
                        description="Menjawab langsung tanpa basa-basi berlebih.",
                        score_5_benchmark="Padat, langsung ke inti jawaban dan data.",
                        score_3_benchmark="Terdapat penjelasan tambahan yang agak panjang.",
                        score_1_benchmark="Bertele-tele tanpa menjawab inti data.",
                    ),
                ],
            )

        # Case 4: Long Manuscript / Comprehensive Review / Chapter drafting
        if any(k in lower_q for k in ["tinjauan pustaka", "bab 2", "bab 1", "bab 3", "draf lengkap", "sintesis komprehensif"]):
            return TaskContract(
                primary_goal=f"Penyusunan tinjauan pustaka dan sintesis komprehensif {topic_str}",
                in_scope=[f"Kerangka teoritis {topic_str}", f"Temuan empiris mutakhir {topic_str}", "Sintesis konseptual dan integrasi temuan"],
                out_of_scope=["Klaim tanpa landasan literatur", "Bahasa informal"],
                format_requirements="Tinjauan akademis terstruktur dengan sub-heading bertahap.",
                min_words=800,
                max_words=1600,
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

        # Case 5: Default General Academic Query
        return TaskContract(
            primary_goal=f"Analisis akademis kontekstual mengenai: {q_clean[:100]}",
            in_scope=[f"Konsep dan temuan relevan terkait {topic_str}", "Aplikasi dan implikasi riset"],
            out_of_scope=["Spekulasi di luar literatur", "Penyimpangan topik"],
            format_requirements="Respons akademis terstruktur dan berbobot ilmiah formal.",
            min_words=50,
            max_words=600,
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
        """Extract contract using rule heuristics or LLM disambiguation for ambiguous intent."""
        # 1. When ambiguous intent signals are detected and AI engine is available, disambiguate via LLM
        if self.is_ambiguous_intent(query) and self.engine:
            ai_contract = self._extract_via_llm(query)
            if ai_contract:
                return ai_contract

        # 2. For queries with long or complex user instructions, optionally refine with LLM
        if len(query.split()) > 40 and self.engine:
            ai_contract = self._extract_via_llm(query)
            if ai_contract:
                return ai_contract

        # 3. Fallback to rule-based heuristic extraction
        return self.extract_heuristic(query)
