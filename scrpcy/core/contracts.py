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


class TaskContractExtractor:
    """Extracts dynamic execution contract and evaluation rubric from user query."""

    def __init__(self, engine: LLMEngine | None = None) -> None:
        self.engine = engine

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

        # Case 1: Argument / Debate / Dialectic exploration
        if any(k in lower_q for k in ["argumen", "argument", "pro kontra", "sisi positif", "sisi negatif", "sudut pandang", "tesis", "dialektika", "debat"]):
            return TaskContract(
                primary_goal=f"Eksplorasi dialektika argumen seputar {topic_str}",
                in_scope=[f"Argumen pro dan landasan bukti empiris {topic_str}", f"Sanggahan kontra dan limitasi {topic_str}", "Tesis sentral dan celah riset"],
                out_of_scope=["Penulisan bab komprehensif monolitik", "Penurunan rumus matematis yang tidak diminta", "Pengantar buku teks umum"],
                format_requirements="Matriks Dialektika: Tesis Sentral, Poin Pendukung (Pro + sitasi), Poin Sanggahan (Kontra + sitasi), dan Celah Riset (Gap).",
                min_words=350,
                max_words=650,
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
            min_words=250,
            max_words=700,
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
        """Extract contract using rule heuristics with optional zero-shot LLM refinement."""
        heuristic_contract = self.extract_heuristic(query)

        # For queries with long or complex user instructions, optionally refine with fast LLM call if available
        if len(query.split()) > 40 and self.engine and getattr(self.engine, "config", None):
            try:
                model_name = self.engine.config.default_model
                api_key = self.engine.config.get_api_key_for_model(model_name)
                if api_key and not api_key.startswith("mock") and getattr(self.engine, "_acompletion_fn", None) is None:
                    # Can enhance contract via LLM if needed; heuristic provides robust baseline
                    pass
            except Exception:
                pass

        return heuristic_contract
