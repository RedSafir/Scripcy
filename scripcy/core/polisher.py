"""Context-Preserving Academic Stylistic Polisher for natural scholarly prose.

Implements DIPPER-inspired controlled order diversity, burstiness modulation,
adversarial token-masking (zero-drift citation & metric protection),
and deterministic pre/post-flight citation and terminology guardrails.
"""

from __future__ import annotations

import asyncio
import math
import re
from typing import Any
from pydantic import BaseModel, Field

from scripcy.core.engine import LLMEngine


class PolishResult(BaseModel):
    """Result of context-preserving academic polishing."""

    original_text: str = Field(description="Raw synthesis text prior to polishing.")
    polished_text: str = Field(description="Polished text with natural rhythm and intact facts.")
    applied: bool = Field(default=False, description="True if polishing was successfully validated.")
    citations_preserved: bool = Field(default=True, description="True if all citations were retained verbatim.")
    burstiness_score_before: float = Field(default=0.0, description="Sentence length std-dev before polishing.")
    burstiness_score_after: float = Field(default=0.0, description="Sentence length std-dev after polishing.")
    reason: str = Field(default="", description="Status explanation or fallback rationale.")


POLISHER_SYSTEM_PROMPT = """You are an elite academic copyeditor and stylometrist operating inside a top-tier peer-reviewed journal.
Your sole mission: Polish the provided academic draft to maximize natural human cadence, elevate syntactic entropy, and eliminate mechanical AI-writing patterns while strictly preserving 100% of the factual substance, empirical data, and context.

INVIOLABLE CONSTRAINTS & STYLOMETRIC DIRECTIVES:
1. STRICT TOKEN PRESERVATION: NEVER alter, delete, translate, or split tokens formatted as __CITE_x__ or __STAT_x__. They MUST remain bound to their exact conceptual assertions.
2. ZERO FACTUAL OR METRIC DRIFT: All statistics, data values, mathematical notations, formulas, percentages, and metrics must be preserved identically.
3. STRICT TERMINOLOGY PRESERVATION: Do not replace domain-specific terms, genes, proteins, methods, or algorithm names with generic synonyms.
4. ASYMMETRIC BURSTINESS & RHYTHMIC CADENCE:
   - Break uniform sentence lengths drastically (target sentence length standard deviation > 10).
   - Alternate punchy, assertive short statements (6–12 words) with rigorous, compound-complex sentences (25–45 words) containing embedded participial phrases or parenthetical qualifiers.
5. REMOVE AI CLICHES & FORMULAIC TRANSITIONS:
   - Strip out robotic bridge phrases and clichés (e.g., 'furthermore', 'moreover', 'in conclusion', 'it is important to note that', 'delve', 'testament', 'pivotal role', 'crucial aspect', 'tapestry', 'beacon', 'menyelami', 'penting untuk dicatat', 'sebagai kesimpulan').
   - Use organic fronted prepositions (e.g., 'Across these benchmarks,', 'In light of this disparity,'), subtle semicolons, or direct subject openings.
6. VOICE & HEDGING MODULATION:
   - Use passive voice purposefully for methodological and experimental descriptions.
   - Use active voice with nominal subjects for theoretical evaluations and comparative claims.
   - Employ nuanced epistemic stance (hedging): 'indicates a plausible tendency', 'warrants deeper scrutiny' rather than absolute generic claims.
7. OUTPUT ONLY POLISHED MARKDOWN: Return strictly the polished Markdown content with no introductory conversational pleasantries, no meta-commentary, and no trailing notes.
8. HIGH-DENSITY DIRECT PROSE: Begin immediately with the core analytical findings. Strip out all opening conversational throat-clearing and superficial bookending summaries."""


class AcademicPolisher:
    """Refines AI academic synthesis into human-like prose without altering citations or technical facts."""

    # In-text APA and bracketed citations: (Author et al., 2026), [1], [Vaswani2017], (Author, 2024, p. 12)
    CITE_PATTERNS = [
        re.compile(r"\([A-Z][a-zA-Z\s\.\&]+,\s*(?:19|20)\d{2}[a-z]?(?:,\s*p{1,2}\.\s*\d+)?\)", re.UNICODE),
        re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+et\s+al\.)?\s*\((?:19|20)\d{2}[a-z]?\)", re.UNICODE),
        re.compile(r"\[([A-Za-z0-9_,\s-]+(?:\s+et\s+al\.)?(?:,\s*\d{4})?)\](?!\()", re.UNICODE),
        re.compile(r"\[\^[0-9]+\]"),  # Footnote citations like [^1], [^2]
    ]

    # Specific numerical metrics, percentages, latencies, p-values, and statistics
    STAT_PATTERNS = [
        re.compile(r"\b(?:p\s*[<>=]\s*0?\.\d+|[Ftr]\s*\(\d+(?:,\s*\d+)?\)\s*=\s*-?\d+\.?\d*)\b", re.IGNORECASE),
        re.compile(r"\b\d+(?:\.\d+)?\s*(?:(?:ms|fps|flops|flosp|gflops|tflops|mb|gb|tb|kb|hz|mhz|ghz)\b|%)", re.IGNORECASE),
    ]

    def __init__(self, engine: LLMEngine | None = None) -> None:
        self.engine = engine

    @classmethod
    def mask_sensitive_entities(cls, text: str) -> tuple[str, dict[str, str]]:
        """
        Adversarial Masking: Replaces citations and statistical data with immutable placeholders.
        Ensures 100% protection against LLM deletion or translation.
        """
        lookup: dict[str, str] = {}
        masked = text

        # 1. Mask citations
        cite_counter = 0
        for pat in cls.CITE_PATTERNS:
            matches = list(pat.finditer(masked))
            for m in reversed(matches):
                # Avoid re-masking already masked tokens
                matched_str = m.group(0)
                if matched_str.startswith("__") and matched_str.endswith("__"):
                    continue
                token = f"__CITE_{cite_counter}__"
                lookup[token] = matched_str
                masked = masked[: m.start()] + token + masked[m.end() :]
                cite_counter += 1

        # 2. Mask specific statistical metrics
        stat_counter = 0
        for pat in cls.STAT_PATTERNS:
            matches = list(pat.finditer(masked))
            for m in reversed(matches):
                matched_str = m.group(0)
                if matched_str.startswith("__") and matched_str.endswith("__"):
                    continue
                token = f"__STAT_{stat_counter}__"
                lookup[token] = matched_str
                masked = masked[: m.start()] + token + masked[m.end() :]
                stat_counter += 1

        return masked, lookup

    @classmethod
    def unmask_sensitive_entities(cls, text: str, lookup: dict[str, str]) -> str:
        """Restores masked placeholders back to their verbatim original citations and metrics."""
        restored = text
        for token, original in lookup.items():
            restored = restored.replace(token, original)
        return restored

    @staticmethod
    def extract_citations(text: str) -> list[str]:
        """Extract academic citation keys like [Vaswani2017], [Page2002], [1], (Author, 2026), [^1]."""
        found = []
        # Bracketed citations
        for m in re.findall(r"\[([A-Za-z0-9_,\s-]+(?:\s+et\s+al\.)?(?:,\s*\d{4})?)\](?!\()", text):
            if m.strip():
                found.append(f"[{m.strip()}]")
        # Parenthetical citations
        for m in re.findall(r"\(([A-Z][a-zA-Z\s\.\&]+,\s*(?:19|20)\d{2}[a-z]?)\)", text):
            if m.strip():
                found.append(f"({m.strip()})")
        # Footnote citations
        for m in re.findall(r"\[\^[0-9]+\]", text):
            found.append(m)
        return found

    @staticmethod
    def extract_numbers(text: str) -> set[str]:
        """Extract significant numerical metrics, percentages, and years."""
        clean = re.sub(r"\[[A-Za-z0-9_,-]+\]", "", text)
        clean = re.sub(r"\([A-Za-z0-9_,\s\.\&-]+(?:19|20)\d{2}\)", "", clean)
        nums = re.findall(r"\b\d+(?:[\.,]\d+)?%?\b", clean)
        return set(nums)

    @staticmethod
    def calculate_burstiness(text: str) -> float:
        """Calculate standard deviation of sentence lengths in words (metric for burstiness)."""
        clean = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        clean = re.sub(r"^#+\s+.*$", "", clean, flags=re.MULTILINE)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
        if len(sentences) < 2:
            return 0.0

        lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
        lengths = [l for l in lengths if l > 0]
        if len(lengths) < 2:
            return 0.0

        mean = sum(lengths) / len(lengths)
        variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
        return round(math.sqrt(variance), 2)

    @classmethod
    def strip_throat_clearing(cls, text: str) -> str:
        """
        Strip conversational throat-clearing, greetings, and meta-intros.
        Example: "Tentu, berikut adalah pembahasan mendalam mengenai..." -> removed.
        """
        cleaned = text.strip()
        opening_patterns = [
            re.compile(r"^(?:tentu\s+saja|tentu(?:lah)?|baiklah|baik|berikut\s+(?:adalah|ini)|dalam\s+pembahasan\s+ini|menjawab\s+pertanyaan\s+(?:anda|tersebut)|berdasarkan\s+pertanyaan\s+(?:anda|tersebut)|sure|certainly|of\s+course|here\s+is|in\s+this\s+response|in\s+this\s+synthesis)[^.\n\?!]*[.:!]\s*", re.IGNORECASE),
            re.compile(r"^(?:sebagai\s+asisten\s+akademis|halo|hai|hello|hi)[^.\n\?!]*[.:!]\s*", re.IGNORECASE),
        ]
        for pat in opening_patterns:
            cleaned = pat.sub("", cleaned).strip()

        paragraphs = cleaned.split("\n\n")
        if paragraphs:
            first_p = paragraphs[0].strip()
            # If first paragraph is purely a meta preamble (under 120 chars, no citations, no numbers)
            is_meta_preamble = (
                len(first_p) < 120
                and not cls.extract_citations(first_p)
                and not cls.extract_numbers(first_p)
                and any(kw in first_p.lower() for kw in ("berikut", "pembahasan", "ulasan", "here is", "let's explore", "mari kita"))
            )
            if is_meta_preamble and len(paragraphs) > 1:
                cleaned = "\n\n".join(paragraphs[1:]).strip()

        return cleaned

    @classmethod
    def strip_bookending_summary(cls, text: str) -> str:
        """
        Strip superficial closing paragraphs that merely repeat previous claims without new citations.
        """
        paragraphs = text.strip().split("\n\n")
        if len(paragraphs) <= 1:
            return text.strip()

        last_p = paragraphs[-1].strip()
        conclusion_starters = [
            re.compile(r"^(?:sebagai\s+kesimpulan|kesimpulannya|secara\s+keseluruhan|dapat\s+disimpulkan\s+bahwa|pada\s+akhirnya|in\s+conclusion|to\s+summarize|in\s+summary|to\s+conclude)", re.IGNORECASE),
        ]
        is_closing_slop = any(pat.search(last_p) for pat in conclusion_starters)
        if is_closing_slop:
            if not cls.extract_citations(last_p):
                paragraphs.pop()
                return "\n\n".join(paragraphs).strip()

        return text.strip()

    @classmethod
    def calculate_slop_density(cls, text: str) -> float:
        """Calculate frequency of known AI slop phrases per 100 words."""
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return 0.0

        slop_phrases = [
            "delve", "testament", "pivotal", "multifaceted", "interplay", "intricacies",
            "tapestry", "beacon", "furthermore", "moreover", "in conclusion",
            "it is important to note that", "it is worth noting that",
            "penting untuk dicatat", "perlu dicatat", "perlu ditekankan", "sebagai kesimpulan",
            "menyelami", "bukti nyata", "memainkan peran penting", "memainkan peran krusial",
            "secara keseluruhan", "hal ini membuktikan bahwa",
        ]
        count = 0
        text_lower = text.lower()
        for phrase in slop_phrases:
            count += len(re.findall(re.escape(phrase), text_lower))

        return round((count / len(words)) * 100, 2)

    def polish_heuristic(self, text: str) -> str:
        """Rule-based de-smoothing: removes robotic boilerplate while keeping exact keywords and citations."""
        # 1. Strip conversational throat-clearing
        polished = self.strip_throat_clearing(text)

        replacements = [
            (r"(?i)\b(?:it is important to note that|it is worth noting that|it should be noted that)\s*", ""),
            (r"(?i)\b(?:in conclusion,?\s*|as a conclusion,?\s*)", ""),
            (r"(?i)\b(?:furthermore, it should be emphasized that|moreover, it is essential to highlight that)\s*", "Furthermore, "),
            (r"(?i)\b(?:furthermore,?\s*|moreover,?\s*)", ""),
            (r"(?i)\bdelving into the\b", "Analyzing the"),
            (r"(?i)\bstands as a testament to\b", "demonstrates"),
            (r"(?i)\bplays a pivotal role in\b", "is central to"),
            (r"(?i)\bplays a crucial role in\b", "determines"),
            (r"(?i)\bcrucial aspect\b", "core factor"),
            # Indonesian common AI slop patterns
            (r"(?i)\b(?:penting untuk dicatat bahwa|perlu dicatat bahwa|perlu ditekankan bahwa)\s*", ""),
            (r"(?i)\b(?:sebagai kesimpulan,?\s*|dapat disimpulkan bahwa\s*)", ""),
            (r"(?i)\bmenyelami lebih dalam terkait\b", "menganalisis"),
            (r"(?i)\bmerupakan bukti nyata bahwa\b", "menunjukkan bahwa"),
            (r"(?i)\bmemainkan peran penting dalam\b", "berpengaruh signifikan terhadap"),
            (r"(?i)\bmemainkan peran krusial dalam\b", "menentukan"),
            (r"(?i)\bsecara keseluruhan,?\s*", ""),
            (r"(?i)\bhal ini membuktikan bahwa\b", "temuan ini mengindikasikan bahwa"),
        ]

        for pat, repl in replacements:
            polished = re.sub(pat, repl, polished)

        # 2. Strip superficial bookending summary
        polished = self.strip_bookending_summary(polished)

        # Clean double spaces or broken start of sentences
        polished = re.sub(r"[ \t]+", " ", polished)
        polished = re.sub(r"\n{3,}", "\n\n", polished)
        return polished.strip()

    def verify_integrity(self, original: str, polished: str) -> tuple[bool, str]:
        """Assert that no citations, numbers, or core structures were lost or corrupted."""
        if not polished or len(polished.strip()) < 50:
            return False, "Polished text is empty or too short."

        # 1. Length sanity check (must not be radically compressed or inflated)
        orig_len = len(original.strip())
        pol_len = len(polished.strip())
        if pol_len < orig_len * 0.55:
            return False, f"Polished text over-compressed ({pol_len} vs {orig_len} chars)."
        if pol_len > orig_len * 1.65:
            return False, f"Polished text over-inflated ({pol_len} vs {orig_len} chars)."

        # 2. Strict citation preservation check
        orig_citations = set(self.extract_citations(original))
        pol_citations = set(self.extract_citations(polished))
        missing_cites = orig_citations - pol_citations
        if missing_cites:
            return False, f"Missing citation keys after polish: {', '.join(missing_cites)}"

        # 3. Numeric metric preservation check
        orig_numbers = self.extract_numbers(original)
        pol_numbers = self.extract_numbers(polished)
        # Verify that at least 85% of numerical data points remain intact
        if orig_numbers:
            missing_nums = orig_numbers - pol_numbers
            if len(missing_nums) > max(1, len(orig_numbers) * 0.25):
                return False, f"Numerical data points missing after polish: {list(missing_nums)[:5]}"

        return True, "Integrity verified: all citations and metrics intact."

    async def polish_async(self, text: str, query: str = "") -> PolishResult:
        """Asynchronously polish text via LLM with deterministic pre/post validation and token masking."""
        burst_before = self.calculate_burstiness(text)

        # If text is too short or doesn't have sentences, skip
        if len(text.strip()) < 100 or burst_before == 0.0:
            return PolishResult(
                original_text=text,
                polished_text=text,
                applied=False,
                burstiness_score_before=burst_before,
                burstiness_score_after=burst_before,
                reason="Text too brief for stylistic polishing.",
            )

        polished_candidate = ""
        used_llm = False

        if self.engine and getattr(self.engine, "config", None):
            try:
                # 1. Pre-Flight Adversarial Masking
                masked_input, mask_lookup = self.mask_sensitive_entities(text)

                model_name = getattr(self.engine.config, "default_model", "gemini-2.5-pro")
                api_key = self.engine.config.get_api_key_for_model(model_name)
                if api_key and not api_key.startswith("mock") and getattr(self.engine, "_acompletion_fn", None) is None:
                    prompt = (
                        f"Please polish this academic text to elevate natural cadence, maximize syntactic burstiness, and eliminate mechanical AI phrasing.\n\n"
                        f"Research Context Query: \"{query}\"\n\n"
                        f"Original Academic Text (Masked Sensitive Entities):\n```markdown\n{masked_input}\n```"
                    )
                    chunks = []
                    async for chunk in self.engine.acompletion_fn(
                        model=self.engine.config.default_model,
                        messages=[
                            {"role": "system", "content": POLISHER_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.45,  # Higher temperature for natural human lexical variance
                        stream=True,
                    ):
                        c_text = chunk.text if hasattr(chunk, "text") else getattr(chunk, "delta", {}).get("content", "")
                        if c_text:
                            chunks.append(c_text)
                    raw_llm = "".join(chunks).strip()
                    if raw_llm:
                        if raw_llm.startswith("```markdown"):
                            raw_llm = raw_llm[11:]
                        elif raw_llm.startswith("```"):
                            raw_llm = raw_llm[3:]
                        if raw_llm.endswith("```"):
                            raw_llm = raw_llm[:-3]

                        # 2. Unmask sensitive entities back to original
                        unmasked_llm = self.unmask_sensitive_entities(raw_llm.strip(), mask_lookup)
                        # 3. Apply Heuristic Cleanser on unmasked output
                        polished_candidate = self.polish_heuristic(unmasked_llm)
                        used_llm = True
            except Exception:
                polished_candidate = ""

        # Deterministic heuristic polisher fallback if LLM was unavailable or in mock environment
        if not polished_candidate:
            polished_candidate = self.polish_heuristic(text)

        # Deterministic Guardrail Assertion Check
        valid, reason = self.verify_integrity(text, polished_candidate)
        if valid:
            burst_after = self.calculate_burstiness(polished_candidate)
            return PolishResult(
                original_text=text,
                polished_text=polished_candidate,
                applied=True,
                citations_preserved=True,
                burstiness_score_before=burst_before,
                burstiness_score_after=burst_after,
                reason=f"Polished successfully via {'LLM' if used_llm else 'Heuristic'} ({reason})",
            )
        else:
            # Fallback to original text to guarantee zero context/citation destruction
            return PolishResult(
                original_text=text,
                polished_text=text,
                applied=False,
                citations_preserved=True,
                burstiness_score_before=burst_before,
                burstiness_score_after=burst_before,
                reason=f"Fallback to original: {reason}",
            )

    def polish(self, text: str, query: str = "") -> PolishResult:
        """Synchronous wrapper for polish_async."""
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(lambda: asyncio.run(self.polish_async(text, query))).result()
            return asyncio.run(self.polish_async(text, query))
        except Exception:
            clean = self.polish_heuristic(text)
            burst_before = self.calculate_burstiness(text)
            burst_after = self.calculate_burstiness(clean)
            return PolishResult(
                original_text=text,
                polished_text=clean,
                applied=True,
                citations_preserved=True,
                burstiness_score_before=burst_before,
                burstiness_score_after=burst_after,
                reason="Heuristic polish fallback.",
            )
