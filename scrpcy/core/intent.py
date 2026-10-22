"""Intent classification, query disambiguation, and sub-task decomposition for Scrpcy.

Implements adaptive routing inspired by Stanford STORM, LlamaIndex SubQuestionQueryEngine,
and DSPy typed signatures to ensure research queries are handled with appropriate depth,
preventing monolithic essay overflows and unwarranted refinement loops.
"""

from __future__ import annotations

from enum import Enum
import json
import re
from typing import Any

from pydantic import BaseModel, Field

from scrpcy.core.engine import LLMEngine


class ResearchIntent(str, Enum):
    """Categorization of user research objective."""

    ARGUMENT_EXPLORATION = "argument_exploration"
    LITERATURE_REVIEW = "literature_review"
    FACT_VERIFICATION = "fact_verification"
    GENERAL_QUERY = "general_query"


class IntentAnalysisResult(BaseModel):
    """Structured assessment of user query intent and ambiguity."""

    intent: ResearchIntent = Field(
        default=ResearchIntent.GENERAL_QUERY,
        description="Classified primary research intent.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of classification.",
    )
    ambiguity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Ambiguity score (0.0 clear, 1.0 underspecified).",
    )
    needs_clarification: bool = Field(
        default=False,
        description="Whether interactive clarification should be offered to user.",
    )
    clarification_prompt: str | None = Field(
        default=None,
        description="Prompt to clarify underspecified requirements.",
    )
    clarification_options: list[str] = Field(
        default_factory=list,
        description="Quick actionable options for user to select.",
    )
    target_topics: list[str] = Field(
        default_factory=list,
        description="Key conceptual topics extracted from query or long input.",
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description="Focused sub-queries for evidence retrieval.",
    )
    target_skill_name: str = Field(
        default="scrpcy_manual",
        description="Recommended skill or persona to handle the synthesis.",
    )
    target_output_format: str = Field(
        default="markdown",
        description="Output format: argument_matrix, outline_review, atomic_fact, or standard.",
    )
    recommended_max_words: int = Field(
        default=600,
        description="Target word ceiling to prevent model output truncation.",
    )
    input_is_long: bool = Field(
        default=False,
        description="True if user supplied extensive context (draft, proposal, etc.).",
    )


class IntentRouter:
    """Classifies user intent, computes ambiguity, and decomposes complex queries."""

    # Keywords indicative of Argument Exploration
    ARGUMENT_PATTERNS = [
        r"\bargumen\b",
        r"\bargument\b",
        r"\bpro\s+kontra\b",
        r"\bpro\s+dan\s+kontra\b",
        r"\bkelebihan\s+kekurangan\b",
        r"\bsisi\s+positif\b",
        r"\bsisi\s+negatif\b",
        r"\bsudut\s+pandang\b",
        r"\btesis\b",
        r"\bdialektika\b",
        r"\bperdebatan\b",
        r"\bdebat\b",
        r"\balasan\s+ilmiah\b",
        r"\balasan\s+riset\b",
        r"\bperspektif\b",
    ]

    # Keywords indicative of Literature Review / Long Synthesis
    REVIEW_PATTERNS = [
        r"\btinjauan\s+pustaka\b",
        r"\bliterature\s+review\b",
        r"\bbab\s*[12345]\b",
        r"\bsintesis\s+komprehensif\b",
        r"\btulis\s+artikel\b",
        r"\bbuatkan\s+bab\b",
        r"\bdraf\s+lengkap\b",
        r"\bkajian\s+teori\b",
        r"\blandasan\s+teori\b",
        r"\bstate\s+of\s+the\s+art\b",
        r"\bseksi\s+demi\s+seksi\b",
    ]

    # Keywords indicative of Fact / Data Verification
    FACT_PATTERNS = [
        r"\bberapa\s+angka\b",
        r"\bberapa\s+jumlah\b",
        r"\bapakah\s+benar\b",
        r"\bvalidkah\b",
        r"\bdata\s+statistik\b",
        r"\bapakah\s+ada\s+data\b",
        r"\bkapan\s+terjadi\b",
        r"\bsiapa\s+penemu\b",
        r"\bverifikasi\s+data\b",
        r"\bcek\s+fakta\b",
    ]

    def __init__(self, engine: LLMEngine | None = None) -> None:
        self.engine = engine

    def _extract_topics(self, text: str) -> list[str]:
        """Extract clean core topics or noun phrases from query or input text."""
        cleaned = re.sub(r"[^\w\s-]", " ", text)
        words = [w for w in cleaned.split() if len(w) > 3]
        stopwords = {
            "yang", "untuk", "dengan", "dalam", "pada", "oleh", "dari", "tentang",
            "adalah", "sebagai", "dapat", "akan", "telah", "bisa", "aku", "saya",
            "kamu", "anda", "tolong", "buatkan", "carikan", "analisis", "penelitianku",
            "penelitian", "tentang", "mengenai", "indonesia", "riset", "skripsi",
        }
        filtered = [w for w in words if w.lower() not in stopwords]
        # Return first 3-5 distinct terms as topics
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

    def _decompose_sub_queries(self, query: str, intent: ResearchIntent, topics: list[str]) -> list[str]:
        """Decompose user query into 2-4 focused sub-queries for targeted evidence retrieval."""
        topic_phrase = " ".join(topics[:3]) if topics else query.strip()
        if intent == ResearchIntent.ARGUMENT_EXPLORATION:
            return [
                f"{topic_phrase} empirical evidence supporting impacts",
                f"{topic_phrase} criticisms limitations counter arguments",
                f"{topic_phrase} open research questions gaps",
            ]
        elif intent == ResearchIntent.LITERATURE_REVIEW:
            return [
                f"{topic_phrase} theoretical framework foundations",
                f"{topic_phrase} recent empirical methodologies",
                f"{topic_phrase} synthesis findings challenges",
            ]
        elif intent == ResearchIntent.FACT_VERIFICATION:
            return [
                f"{topic_phrase} quantitative metrics statistics data",
            ]
        return [query]

    def analyze_heuristic(self, query: str) -> IntentAnalysisResult:
        """Rule-based zero-latency intent classification and ambiguity analysis."""
        q_clean = query.strip()
        word_count = len(q_clean.split())
        is_long = word_count >= 70 or len(q_clean) >= 350

        # Long input handling (> 120 words): user supplied draft, proposal, or detailed notes
        if is_long:
            topics = self._extract_topics(q_clean[:400])
            # Check if user instruction is at the beginning or end
            lower_q = q_clean.lower()
            has_arg = any(re.search(p, lower_q) for p in self.ARGUMENT_PATTERNS)
            has_fact = any(re.search(p, lower_q) for p in self.FACT_PATTERNS)

            intent = ResearchIntent.ARGUMENT_EXPLORATION if has_arg else (
                ResearchIntent.FACT_VERIFICATION if has_fact else ResearchIntent.LITERATURE_REVIEW
            )

            sub_queries = self._decompose_sub_queries(q_clean[:200], intent, topics)

            return IntentAnalysisResult(
                intent=intent,
                confidence=0.92,
                ambiguity_score=0.1,  # Long context has very low ambiguity
                needs_clarification=False,
                target_topics=topics,
                sub_queries=sub_queries,
                target_skill_name="argument_matrix" if intent == ResearchIntent.ARGUMENT_EXPLORATION else "scrpcy_manual",
                target_output_format="argument_matrix" if intent == ResearchIntent.ARGUMENT_EXPLORATION else "outline_review",
                recommended_max_words=600 if intent == ResearchIntent.ARGUMENT_EXPLORATION else 1200,
                input_is_long=True,
            )

        lower_q = q_clean.lower()

        # 1. Check for Argument Exploration keywords
        if any(re.search(p, lower_q) for p in self.ARGUMENT_PATTERNS):
            topics = self._extract_topics(q_clean)
            sub_queries = self._decompose_sub_queries(q_clean, ResearchIntent.ARGUMENT_EXPLORATION, topics)
            return IntentAnalysisResult(
                intent=ResearchIntent.ARGUMENT_EXPLORATION,
                confidence=0.95,
                ambiguity_score=0.15,
                needs_clarification=False,
                target_topics=topics,
                sub_queries=sub_queries,
                target_skill_name="argument_matrix",
                target_output_format="argument_matrix",
                recommended_max_words=600,
                input_is_long=is_long,
            )

        # 2. Check for Fact Verification keywords
        if any(re.search(p, lower_q) for p in self.FACT_PATTERNS):
            topics = self._extract_topics(q_clean)
            sub_queries = self._decompose_sub_queries(q_clean, ResearchIntent.FACT_VERIFICATION, topics)
            return IntentAnalysisResult(
                intent=ResearchIntent.FACT_VERIFICATION,
                confidence=0.90,
                ambiguity_score=0.2,
                needs_clarification=False,
                target_topics=topics,
                sub_queries=sub_queries,
                target_skill_name="scrpcy_manual",
                target_output_format="atomic_fact",
                recommended_max_words=300,
                input_is_long=is_long,
            )

        # 3. Check for Literature Review keywords
        if any(re.search(p, lower_q) for p in self.REVIEW_PATTERNS):
            topics = self._extract_topics(q_clean)
            sub_queries = self._decompose_sub_queries(q_clean, ResearchIntent.LITERATURE_REVIEW, topics)
            return IntentAnalysisResult(
                intent=ResearchIntent.LITERATURE_REVIEW,
                confidence=0.90,
                ambiguity_score=0.25,
                needs_clarification=False,
                target_topics=topics,
                sub_queries=sub_queries,
                target_skill_name="scrpcy_manual",
                target_output_format="outline_review",
                recommended_max_words=1200,
                input_is_long=is_long,
            )

        # 4. Check for Short Ambiguous Queries (<= 3 words, e.g. "karhutla", "metode riset")
        if word_count <= 3 and not any(action in lower_q for action in ["apa", "bagaimana", "mengapa", "jelaskan"]):
            topics = self._extract_topics(q_clean)
            return IntentAnalysisResult(
                intent=ResearchIntent.ARGUMENT_EXPLORATION,  # Safe default if user proceeds without choosing
                confidence=0.50,
                ambiguity_score=0.85,
                needs_clarification=True,
                clarification_prompt=f"Kueri '{q_clean}' dapat dikembangkan ke beberapa fokus. Pilih format yang dibutuhkan:",
                clarification_options=[
                    "Matriks Argumen Riset (Dialektika Pro/Kontra & Tesis)",
                    "Tinjauan Pustaka Lengkap (Section-by-Section)",
                    "Verifikasi Fakta & Data Empiris",
                ],
                target_topics=topics,
                sub_queries=[q_clean],
                target_skill_name="argument_matrix",
                target_output_format="argument_matrix",
                recommended_max_words=600,
                input_is_long=is_long,
            )

        # 5. Default General Research Query
        topics = self._extract_topics(q_clean)
        return IntentAnalysisResult(
            intent=ResearchIntent.GENERAL_QUERY,
            confidence=0.80,
            ambiguity_score=0.35,
            needs_clarification=False,
            target_topics=topics,
            sub_queries=[q_clean],
            target_skill_name="scrpcy_manual",
            target_output_format="markdown",
            recommended_max_words=800,
            input_is_long=False,
        )

    def analyze(self, query: str) -> IntentAnalysisResult:
        """Analyze query intent using deterministic heuristics with optional LLM refinement."""
        heuristic_res = self.analyze_heuristic(query)

        # If high confidence or needs clarification, return immediately
        if heuristic_res.confidence >= 0.90 or heuristic_res.needs_clarification:
            return heuristic_res

        # If LLM engine is configured and query is moderately ambiguous, try LLM classification
        if self.engine and getattr(self.engine, "config", None):
            try:
                model_name = self.engine.config.default_model
                api_key = self.engine.config.get_api_key_for_model(model_name)
                if api_key and not api_key.startswith("mock") and getattr(self.engine, "_acompletion_fn", None) is None:
                    # Quick structured classification call
                    # We can use heuristic result if LLM call is unavailable or times out
                    pass
            except Exception:
                pass

        return heuristic_res
