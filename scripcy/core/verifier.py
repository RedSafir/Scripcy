"""FactChecker module for claim extraction and evidence grounding verification."""

from __future__ import annotations

import json
import re
from typing import Any
from pydantic import BaseModel, Field

from scripcy.core.engine import EvidenceChunk, LLMEngine, estimate_tokens


def _clean_json_payload(raw: str) -> str:
    clean = raw.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```[a-zA-Z]*\r?\n", "", clean)
        clean = re.sub(r"\r?\n```$", "", clean)
    return clean.strip()


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
    input_tokens: int = Field(default=0, description="Estimated input tokens consumed during verification.")
    output_tokens: int = Field(default=0, description="Estimated output tokens consumed during verification.")


class FactChecker:
    """Verifies that generated claims and summaries are grounded in vault evidence."""

    def __init__(self, engine: LLMEngine | None = None) -> None:
        self.engine = engine
        self._claim_cache: dict[str, ClaimVerification] = {}

    def clear_cache(self) -> None:
        """Clear cached claim verifications."""
        self._claim_cache.clear()

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
            salient_claims.append(s_clean)

        total_words = len(clean.split())
        claim_budget = max(15, min(60, total_words // 35))
        return salient_claims[:claim_budget]

    def verify_claims_llm(
        self,
        claims: list[str],
        evidence_chunks: list[EvidenceChunk],
    ) -> tuple[list[ClaimVerification] | None, int, int]:
        """Verify claims against evidence using LLM semantic entailment check (multilingual)."""
        if not claims or not evidence_chunks or not self.engine:
            return None, 0, 0

        try:
            model_name = self.engine.config.default_model
            api_key = self.engine.config.get_api_key_for_model(model_name)
            if not api_key or api_key.startswith("mock") or getattr(self.engine, "_acompletion_fn", None) is not None:
                return None, 0, 0

            evidence_text = "\n\n".join(f"[{c.source}]: {c.text}" for c in evidence_chunks)
            claims_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))

            system_prompt = (
                "You are a strict academic fact-checker. Verify each numbered claim against the provided evidence excerpts.\n"
                "The evidence may be in English while claims may be in Indonesian or English.\n"
                "Determine whether the semantics of each claim are directly supported by the evidence.\n"
                "Respond in valid JSON object matching:\n"
                "{\"verifications\": [{\"claim_idx\": 1, \"status\": \"supported\", \"source\": \"cite_key\", \"confidence\": 0.95, \"reason\": \"...\"}]}\n"
                "status must be one of: 'supported', 'weak', or 'unsupported'."
            )

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

            kwargs: dict[str, Any] = {
                "model": effective_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"EVIDENCE EXCERPTS:\n{evidence_text}\n\nCLAIMS TO VERIFY:\n{claims_text}"},
                ],
                "api_key": api_key,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
                "timeout": 15,
            }
            if api_base:
                kwargs["api_base"] = api_base

            resp = litellm.completion(**kwargs)
            content = resp.choices[0].message.content or "{}"
            data = json.loads(_clean_json_payload(content))
            raw_verifs = data.get("verifications", [])

            results: list[ClaimVerification] = []
            for i, claim in enumerate(claims):
                v_entry = next((v for v in raw_verifs if v.get("claim_idx") == i + 1), None)
                if v_entry:
                    results.append(ClaimVerification(
                        claim=claim,
                        status=v_entry.get("status", "supported"),
                        supporting_source=v_entry.get("source") or (evidence_chunks[0].source if evidence_chunks else None),
                        confidence=float(v_entry.get("confidence", 0.9)),
                        reason=v_entry.get("reason", "Verified via LLM entailment."),
                    ))
                else:
                    results.append(ClaimVerification(
                        claim=claim,
                        status="supported",
                        supporting_source=evidence_chunks[0].source if evidence_chunks else None,
                        confidence=0.8,
                        reason="Default grounded assertion.",
                    ))
            in_tok = estimate_tokens(evidence_text) + estimate_tokens(claims_text) + 150
            out_tok = estimate_tokens(content)
            return results, in_tok, out_tok
        except Exception:
            return None, 0, 0

    def verify_claims_lexical(self, claims: list[str], evidence_chunks: list[EvidenceChunk]) -> list[ClaimVerification]:
        """Fast rule-based and multilingual semantic embedding check of claims against evidence chunks."""
        results: list[ClaimVerification] = []
        combined_evidence = " ".join(c.text.lower() for c in evidence_chunks)

        # Pre-extract embedder for semantic cross-lingual matching if lexical overlap fails
        embedder = None
        chunk_embeddings = None
        try:
            from scripcy.core.vector_vault import get_shared_embedder
            embedder = get_shared_embedder()
            if embedder and evidence_chunks:
                chunk_texts = [c.text for c in evidence_chunks]
                chunk_embeddings = list(embedder.embed(chunk_texts))
        except Exception:
            embedder = None

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

            if ratio >= 0.4:
                results.append(ClaimVerification(
                    claim=claim,
                    status="supported",
                    supporting_source=best_source or (evidence_chunks[0].source if evidence_chunks else "vault"),
                    confidence=min(1.0, 0.6 + ratio * 0.4),
                    reason="Matched relevant keywords in vault evidence.",
                ))
            elif ratio >= 0.2:
                results.append(ClaimVerification(
                    claim=claim,
                    status="weak",
                    supporting_source=best_source or (evidence_chunks[0].source if evidence_chunks else "vault"),
                    confidence=0.5,
                    reason="Partial evidence match in vault.",
                ))
            else:
                # Cross-Lingual Semantic Fallback via FastEmbed ONNX
                semantic_matched = False
                if embedder is not None and chunk_embeddings:
                    try:
                        import numpy as np
                        claim_vec = list(embedder.embed([claim]))[0]
                        claim_norm = np.linalg.norm(claim_vec)
                        if claim_norm > 1e-6:
                            claim_unit = claim_vec / claim_norm
                            best_sim = -1.0
                            best_c_idx = 0
                            for idx, c_vec in enumerate(chunk_embeddings):
                                c_norm = np.linalg.norm(c_vec)
                                if c_norm > 1e-6:
                                    sim = float(np.dot(claim_unit, c_vec / c_norm))
                                    if sim > best_sim:
                                        best_sim = sim
                                        best_c_idx = idx

                            if best_sim >= 0.58:
                                semantic_matched = True
                                results.append(ClaimVerification(
                                    claim=claim,
                                    status="supported",
                                    supporting_source=evidence_chunks[best_c_idx].source,
                                    confidence=min(1.0, 0.60 + (best_sim - 0.58) * 0.9),
                                    reason=f"Cross-lingual semantic verification via FastEmbed (cos_sim={best_sim:.2f}).",
                                ))
                            elif best_sim >= 0.42:
                                semantic_matched = True
                                results.append(ClaimVerification(
                                    claim=claim,
                                    status="weak",
                                    supporting_source=evidence_chunks[best_c_idx].source,
                                    confidence=0.50,
                                    reason=f"Weak cross-lingual semantic alignment via FastEmbed (cos_sim={best_sim:.2f}).",
                                ))
                    except Exception:
                        semantic_matched = False

                if not semantic_matched:
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

        verifications_map: dict[str, ClaimVerification] = {}
        unresolved_claims: list[str] = []

        # 1. Claim Cache Check: Reuse already verified claims across turns (0 tokens)
        for claim in claims:
            norm_c = re.sub(r"\s+", " ", claim.strip().lower())
            if norm_c in self._claim_cache:
                cached_v = self._claim_cache[norm_c]
                if cached_v.status in ("supported", "weak"):
                    verifications_map[claim] = cached_v
                    continue
            unresolved_claims.append(claim)

        # 2. Tiered Lexical Pre-Filter: Fast keyword/n-gram match against vault (0 tokens)
        lexical_results = self.verify_claims_lexical(unresolved_claims, evidence_chunks)
        llm_candidate_claims: list[str] = []

        for v in lexical_results:
            # If strong direct lexical match found (confidence >= 0.70), accept immediately without LLM!
            if v.status == "supported" and v.confidence >= 0.70:
                verifications_map[v.claim] = v
                norm_c = re.sub(r"\s+", " ", v.claim.strip().lower())
                self._claim_cache[norm_c] = v
            else:
                # Ambiguous or unsupported: candidate for deep multilingual LLM entailment
                llm_candidate_claims.append(v.claim)

        # 3. LLM Semantic Entailment: ONLY call 9Router for unresolved/ambiguous claims
        fc_in_tokens = 0
        fc_out_tokens = 0
        if llm_candidate_claims:
            llm_results, fc_in_tokens, fc_out_tokens = self.verify_claims_llm(llm_candidate_claims, evidence_chunks)
            if llm_results:
                for v in llm_results:
                    verifications_map[v.claim] = v
                    norm_c = re.sub(r"\s+", " ", v.claim.strip().lower())
                    if v.status in ("supported", "weak"):
                        self._claim_cache[norm_c] = v
            else:
                # Fallback to lexical results if LLM is unavailable
                for v in lexical_results:
                    if v.claim in llm_candidate_claims:
                        verifications_map[v.claim] = v

        verifications: list[ClaimVerification] = [
            verifications_map.get(c, ClaimVerification(claim=c, status="unsupported", confidence=0.0, reason="Unverified claim."))
            for c in claims
        ]

        supported_count = sum(1 for v in verifications if v.status == "supported")
        weak_count = sum(1 for v in verifications if v.status == "weak")
        unsupported_count = sum(1 for v in verifications if v.status == "unsupported")
        total = len(verifications)
        score = round((supported_count + 0.8 * weak_count) / total, 2) if total > 0 else 1.0

        warning = None
        if unsupported_count > 0 and score < 0.6:
            warning = f"Terdeteksi {unsupported_count} pernyataan yang tidak didukung langsung oleh literatur di vault."

        return VerificationReport(
            grounding_score=score,
            total_claims=total,
            supported_claims=supported_count + weak_count,
            unsupported_claims=unsupported_count,
            claims=verifications,
            hallucination_warning=warning,
            audited_text=text,
            input_tokens=fc_in_tokens,
            output_tokens=fc_out_tokens,
        )
