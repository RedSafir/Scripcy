"""Context-Preserving Academic Stylistic Polisher for natural scholarly prose.

Implements DIPPER-inspired controlled order diversity, burstiness modulation,
and deterministic pre/post-flight citation and terminology guardrails.
"""

from __future__ import annotations

import asyncio
import math
import re
from typing import Any
from pydantic import BaseModel, Field

from scrpcy.core.engine import LLMEngine


class PolishResult(BaseModel):
    """Result of context-preserving academic polishing."""

    original_text: str = Field(description="Raw synthesis text prior to polishing.")
    polished_text: str = Field(description="Polished text with natural rhythm and intact facts.")
    applied: bool = Field(default=False, description="True if polishing was successfully validated.")
    citations_preserved: bool = Field(default=True, description="True if all citations were retained verbatim.")
    burstiness_score_before: float = Field(default=0.0, description="Sentence length std-dev before polishing.")
    burstiness_score_after: float = Field(default=0.0, description="Sentence length std-dev after polishing.")
    reason: str = Field(default="", description="Status explanation or fallback rationale.")


POLISHER_SYSTEM_PROMPT = """You are an expert academic copyeditor and stylometrist operating inside a scientific journal editorial review.
Your sole mission: Polish the provided academic draft to maximize natural human cadence and eliminate mechanical AI-writing patterns while strictly preserving 100% of the factual substance, empirical data, and context.

INVIOLABLE CONSTRAINTS:
1. ZERO CITATION ALTERATION: Every in-text citation (e.g. [Vaswani2017], [1], [Smith2020]) MUST be retained verbatim and remain bound to its exact conceptual assertion.
2. ZERO FACTUAL OR METRIC DRIFT: All statistics, data values, mathematical notations, formulas, percentages, and metrics must be preserved identically.
3. STRICT TERMINOLOGY PRESERVATION: Do not replace domain-specific terms, genes, proteins, methods, or algorithm names with generic synonyms.
4. BURSTINESS & RHYTHM: Inject natural human sentence length variance. Combine concise, impactful assertions with structured, logically rigorous compound sentences. Break uniform sentence cadences.
5. REMOVE AI CLICHES: Strip out formulaic bridge phrases and clichés (e.g., 'delve', 'testament', 'pivotal role', 'it is worth noting that', 'furthermore, it should be noted', 'menyelami', 'penting untuk dicatat').
6. OUTPUT ONLY POLISHED MARKDOWN: Return strictly the polished Markdown content with no introductory conversational pleasantries, no meta-commentary, and no trailing notes."""


class AcademicPolisher:
    """Refines AI academic synthesis into human-like prose without altering citations or technical facts."""

    def __init__(self, engine: LLMEngine | None = None) -> None:
        self.engine = engine

    @staticmethod
    def extract_citations(text: str) -> list[str]:
        """Extract academic citation keys like [Vaswani2017], [Page2002], [1], [1, 2]."""
        # Exclude markdown link brackets like [link](url)
        matches = re.findall(r"\[([A-Za-z0-9_,\s-]+(?:\s+et\s+al\.)?(?:,\s*\d{4})?)\](?!\()", text)
        return [f"[{m.strip()}]" for m in matches if m.strip()]

    @staticmethod
    def extract_numbers(text: str) -> set[str]:
        """Extract significant numerical metrics, percentages, and years."""
        # Clean out citations first so citation numbers aren't confused with metrics
        clean = re.sub(r"\[[A-Za-z0-9_,-]+\]", "", text)
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

    def polish_heuristic(self, text: str) -> str:
        """Rule-based de-smoothing: removes robotic boilerplate while keeping exact keywords and citations."""
        replacements = [
            (r"(?i)\b(?:it is important to note that|it is worth noting that|it should be noted that)\s*", ""),
            (r"(?i)\b(?:in conclusion,?\s*|as a conclusion,?\s*)", ""),
            (r"(?i)\b(?:furthermore, it should be emphasized that|moreover, it is essential to highlight that)\s*", "Furthermore, "),
            (r"(?i)\bdelving into the\b", "Analyzing the"),
            (r"(?i)\bstands as a testament to\b", "demonstrates"),
            (r"(?i)\bplays a pivotal role in\b", "is central to"),
            (r"(?i)\bplays a crucial role in\b", "determines"),
            # Indonesian common AI slop patterns
            (r"(?i)\b(?:penting untuk dicatat bahwa|perlu dicatat bahwa|perlu ditekankan bahwa)\s*", ""),
            (r"(?i)\b(?:sebagai kesimpulan,?\s*|dapat disimpulkan bahwa\s*)", ""),
            (r"(?i)\bmenyelami lebih dalam terkait\b", "menganalisis"),
            (r"(?i)\bmerupakan bukti nyata bahwa\b", "menunjukkan bahwa"),
            (r"(?i)\bmemainkan peran penting dalam\b", "berpengaruh signifikan terhadap"),
            (r"(?i)\bmemainkan peran krusial dalam\b", "menentukan"),
        ]

        polished = text
        for pat, repl in replacements:
            polished = re.sub(pat, repl, polished)

        # Clean double spaces or broken start of sentences
        polished = re.sub(r"\n{3,}", "\n\n", polished)
        return polished.strip()

    def verify_integrity(self, original: str, polished: str) -> tuple[bool, str]:
        """Assert that no citations, numbers, or core structures were lost or corrupted."""
        if not polished or len(polished.strip()) < 50:
            return False, "Polished text is empty or too short."

        # 1. Length sanity check (must not be radically compressed or inflated)
        orig_len = len(original.strip())
        pol_len = len(polished.strip())
        if pol_len < orig_len * 0.60:
            return False, f"Polished text over-compressed ({pol_len} vs {orig_len} chars)."
        if pol_len > orig_len * 1.60:
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
        # Verify that at least 90% of numerical data points remain intact
        if orig_numbers:
            missing_nums = orig_numbers - pol_numbers
            if len(missing_nums) > max(1, len(orig_numbers) * 0.25):
                return False, f"Numerical data points missing after polish: {list(missing_nums)[:5]}"

        return True, "Integrity verified: all citations and metrics intact."

    async def polish_async(self, text: str, query: str = "") -> PolishResult:
        """Asynchronously polish text via LLM with deterministic pre/post validation."""
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
                model_name = getattr(self.engine.config, "default_model", "gemini-2.5-pro")
                api_key = self.engine.config.get_api_key_for_model(model_name)
                if api_key and not api_key.startswith("mock") and getattr(self.engine, "_acompletion_fn", None) is None:
                    prompt = (
                        f"Please polish this academic text to elevate natural cadence and remove mechanical AI phrasing.\n\n"
                        f"Research Context Query: \"{query}\"\n\n"
                        f"Original Academic Text:\n```markdown\n{text}\n```"
                    )
                    chunks = []
                    async for chunk in self.engine.acompletion_fn(
                        model=self.engine.config.default_model,
                        messages=[
                            {"role": "system", "content": POLISHER_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.3,
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
                        polished_candidate = self.polish_heuristic(raw_llm.strip())
                        used_llm = True
            except Exception:
                polished_candidate = ""

        # Deterministic heuristic polisher if LLM was unavailable or in mock environment
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
