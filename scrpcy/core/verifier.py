"""FactChecker module for claim extraction and evidence grounding verification."""

from __future__ import annotations

import re
from typing import Any
from pydantic import BaseModel, Field

from scrpcy.core.engine import EvidenceChunk, LLMEngine


class ClaimVerification(BaseModel):
    """An individual atomic claim and its verification status."""

    claim: str = Field(description="Extracted atomic proposition or assertion.")
    status: str = Field(description="'supported', 'weak', or 'unsupported'.")
    supporting_source: str | None = Field(default=None, description="Identifier of the supporting evidence source.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score of grounding.")
    reason: str = Field(default="", description="Verification rationale.")


class VerificationReport(BaseModel):
    """Comprehensive fact-checking and grounding report."""

    grounding_score: float = Field(ge=0.0, le=1.0, description="Percentage of claims verified by evidence (0.0 to 1.0).")
    total_claims: int = Field(default=0, description="Total number of evaluated claims.")
    supported_claims: int = Field(default=0, description="Number of claims verified by vault evidence.")
    unsupported_claims: int = Field(default=0, description="Number of claims lacking direct evidence.")
    claims: list[ClaimVerification] = Field(default_factory=list, description="Detailed list of claim verifications.")
    hallucination_warning: str | None = Field(default=None, description="Warning message if hallucinations detected.")
    audited_text: str = Field(default="", description="Clean response text with ungrounded claims addressed.")


class FactChecker:
    """Verifies that generated claims and summaries are grounded in vault evidence."""

    def __init__(self, engine: LLMEngine | None = None) -> None:
        self.engine = engine

    def extract_atomic_claims(self, text: str) -> list[str]:
        """Split synthesis text into salient atomic factual sentences."""
        clean = text.strip()
        # Remove markdown headers and code blocks
        clean = re.sub(r"```.*?```", "", clean, flags=re.DOTALL)
        clean = re.sub(r"^#+\s+.*$", "", clean, flags=re.MULTILINE)

        sentences = re.split(r"(?<=[.!?])\s+", clean)
        salient_claims: list[str] = []
        for s in sentences:
            s_clean = s.strip()
            # Filter out non-factual rhetorical phrases or very short fragments
            if len(s_clean.split()) < 5:
                continue
            if any(s_clean.startswith(prefix) for prefix in ("Berdasarkan", "Secara umum", "Sebagai contoh", "Dalam hal ini")):
                # Keep substantive claims
                pass
            salient_claims.append(s_clean)

        return salient_claims[:15]  # Cap at 15 claims for performance

    def verify_claims_lexical(self, claims: list[str], evidence_chunks: list[EvidenceChunk]) -> list[ClaimVerification]:
        """Fast rule-based / lexical overlap check of claims against evidence chunks."""
        results: list[ClaimVerification] = []
        combined_evidence = " ".join(c.text.lower() for c in evidence_chunks)

        for claim in claims:
            words = [w.lower() for w in re.findall(r"\b\w{4,}\b", claim)]
            if not words:
                results.append(ClaimVerification(claim=claim, status="supported", confidence=0.8, reason="Informational assertion."))
                continue

            matches = sum(1 for w in words if w in combined_evidence)
            ratio = matches / len(words)

            best_source = None
            if ratio >= 0.4:
                # Find matching chunk source
                for chunk in evidence_chunks:
                    c_text = chunk.text.lower()
                    c_matches = sum(1 for w in words if w in c_text)
                    if c_matches >= max(2, len(words) * 0.3):
                        best_source = chunk.source
                        break

            if ratio >= 0.5:
                results.append(ClaimVerification(
                    claim=claim,
                    status="supported",
                    supporting_source=best_source or (evidence_chunks[0].source if evidence_chunks else "vault"),
                    confidence=min(1.0, 0.6 + ratio * 0.4),
                    reason="Matched relevant keywords in vault evidence.",
                ))
            elif ratio >= 0.25:
                results.append(ClaimVerification(
                    claim=claim,
                    status="weak",
                    supporting_source=best_source,
                    confidence=0.5,
                    reason="Partial evidence match in vault.",
                ))
            else:
                results.append(ClaimVerification(
                    claim=claim,
                    status="unsupported",
                    supporting_source=None,
                    confidence=0.2,
                    reason="No direct grounding found in active vault evidence passages.",
                ))

        return results

    def verify_synthesis(
        self,
        text: str,
        evidence_chunks: list[EvidenceChunk],
    ) -> VerificationReport:
        """Run complete factual grounding audit on synthesized response."""
        if not evidence_chunks:
            # When no evidence chunks provided, all factual claims are ungrounded
            claims = self.extract_atomic_claims(text)
            verifications = [
                ClaimVerification(claim=c, status="unsupported", confidence=0.0, reason="No vault evidence chunks active.")
                for c in claims
            ]
            total = len(verifications)
            return VerificationReport(
                grounding_score=0.0 if total > 0 else 1.0,
                total_claims=total,
                supported_claims=0,
                unsupported_claims=total,
                claims=verifications,
                hallucination_warning="Jawaban tidak didukung oleh dokumen vault yang terindeks." if total > 0 else None,
                audited_text=text,
            )

        claims = self.extract_atomic_claims(text)
        if not claims:
            return VerificationReport(
                grounding_score=1.0,
                total_claims=0,
                supported_claims=0,
                unsupported_claims=0,
                claims=[],
                audited_text=text,
            )

        verifications = self.verify_claims_lexical(claims, evidence_chunks)
        supported_count = sum(1 for v in verifications if v.status == "supported")
        unsupported_count = sum(1 for v in verifications if v.status == "unsupported")
        total = len(verifications)
        score = round(supported_count / total, 2) if total > 0 else 1.0

        warning = None
        if unsupported_count > 0 and score < 0.6:
            warning = f"Terdeteksi {unsupported_count} pernyataan yang tidak didukung langsung oleh literatur di vault."

        return VerificationReport(
            grounding_score=score,
            total_claims=total,
            supported_claims=supported_count,
            unsupported_claims=unsupported_count,
            claims=verifications,
            hallucination_warning=warning,
            audited_text=text,
        )
