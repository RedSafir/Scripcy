"""Isolated subagent worker and concurrency pool for token-efficient fact extraction."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import threading
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from scripcy.core.contractor import SubContract
from scripcy.core.engine import LLMEngine, estimate_tokens


def _clean_json_payload(raw: str) -> str:
    clean = raw.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```[a-zA-Z]*\r?\n", "", clean)
        clean = re.sub(r"\r?\n```$", "", clean)
    return clean.strip()


class ExtractedFact(BaseModel):
    """Atomic verified claim or empirical metric extracted by an isolated subagent."""

    claim_or_metric: str = Field(description="Concise factual statement or quantitative figure.")
    citation_stem: str = Field(description="Document stem or source key backing this fact.")
    year: str | None = Field(default=None, description="Publication or observation year if present.")
    is_quantitative: bool = Field(default=False, description="True if fact contains explicit numerical/statistical data.")
    visual_ref: str | None = Field(default=None, description="Optional [Visual #N] tag.")


class SubagentArtifact(BaseModel):
    """Compact structured output returned by a worker after its isolated context is discarded."""

    subcontract_id: str
    role: str
    status: Literal["complete", "partial", "no_data"] = "complete"
    extracted_facts: list[ExtractedFact] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    worker_notes: str = Field(default="", description="Field notes and search limitations encountered by the worker.")
    ingested_stems: list[str] = Field(default_factory=list)
    coverage_score: float = Field(default=1.0, description="Local information sufficiency ratio (0.0 to 1.0).")
    missing_slots: list[str] = Field(default_factory=list, description="Specific missing entities or metric slots.")
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def tokens_used(self) -> int:
        return self.input_tokens + self.output_tokens


class InformationSlot(BaseModel):
    """Target entities and required empirical dimensions decomposed from a SubContract."""

    target_entities: list[str] = Field(default_factory=list, description="Specific models, methods, or entities that must be covered.")
    requires_quantitative: bool = Field(default=False, description="True if numerical metrics, percentages, or statistics are required.")
    found_entities: set[str] = Field(default_factory=set, description="Entities found in extracted facts.")
    found_metrics: list[str] = Field(default_factory=list, description="Specific numerical metrics located.")


WorkerArtifact = SubagentArtifact


class SubagentWorker:
    """Executes a single SubContract in an isolated context window without chat history or full drafts."""

    def __init__(
        self,
        engine: LLMEngine | None = None,
        llm_completion_fn: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.engine = engine
        self.llm_completion_fn = llm_completion_fn

    def _can_use_live_llm(self) -> bool:
        if self.llm_completion_fn is not None:
            return True
        if not self.engine or not getattr(self.engine, "config", None):
            return False
        if getattr(self.engine, "_acompletion_fn", None) is not None or getattr(self.engine, "acompletion_fn", None) is not None:
            return False
        model_name = self.engine.config.default_model
        api_key = self.engine.config.get_api_key_for_model(model_name)
        if not api_key or str(api_key).startswith("mock") or str(model_name).startswith("mock"):
            return False
        return True

    @classmethod
    def decompose_slots(cls, subcontract: SubContract) -> InformationSlot:
        """Decompose a SubContract's objective, dos, and allowed scope into target slots."""
        text = f"{subcontract.objective} {' '.join(subcontract.dos or [])} {' '.join(subcontract.allowed_scope or [])}"
        text_lower = text.lower()

        # 1. Detect quantitative requirement
        quant_triggers = [
            "angka", "metrik", "kuantitatif", "data", "statistik", "persen", "%", "luas", "hektar", "ha",
            "latensi", "latency", "fps", "throughput", "akurasi", "map", "loss", "skor", "nilai", "jumlah",
            "gigaton", "ton", "ug/m3", "ms", "mb", "gb",
        ]
        requires_quant = any(re.search(r"\b" + re.escape(w) + r"\b", text_lower) for w in quant_triggers)

        # 2. Extract target entities (e.g. models, methods, or comparative subjects)
        entities: list[str] = []
        vs_match = re.search(r"([A-Za-z0-9_-]+)\s+(?:vs\.?|versus)\s+([A-Za-z0-9_-]+)", text, re.IGNORECASE)
        if vs_match:
            entities.extend([vs_match.group(1).strip(), vs_match.group(2).strip()])
        else:
            antara_match = re.search(r"(?:antara|perbandingan|komparasi)\s+([A-Za-z0-9_-]+)\s+(?:dan|dengan)\s+([A-Za-z0-9_-]+)", text, re.IGNORECASE)
            if antara_match:
                entities.extend([antara_match.group(1).strip(), antara_match.group(2).strip()])

        for term in (subcontract.allowed_scope or []):
            term_clean = term.strip()
            if (
                len(term_clean) >= 3
                and term_clean not in entities
                and term_clean.lower() not in quant_triggers
                and not any(w in term_clean.lower() for w in ("analisis", "bukti", "fakta", "data", "umum", "metrik"))
            ):
                entities.append(term_clean)

        unique_entities: list[str] = []
        for e in entities:
            if e and e.lower() not in [u.lower() for u in unique_entities]:
                unique_entities.append(e)

        return InformationSlot(
            target_entities=unique_entities,
            requires_quantitative=requires_quant,
        )

    @classmethod
    def evaluate_sufficiency(
        cls,
        facts: list[ExtractedFact],
        slots: InformationSlot,
    ) -> tuple[str, list[str], float, set[str]]:
        """
        Evaluate if extracted facts satisfy all required slots.
        Returns: (status: 'complete'|'partial'|'no_data', missing_requirements: list[str], coverage_score: float, found_entities: set[str])
        """
        if not facts:
            missing = ["Tidak ada fakta tervalidasi yang berhasil diekstrak."]
            if slots.target_entities:
                missing.append(f"Entitas target belum ditemukan: {', '.join(slots.target_entities)}.")
            return "no_data", missing, 0.0, set()

        all_facts_text = " ".join(f.claim_or_metric.lower() for f in facts)
        found_entities: set[str] = set()
        missing_slots: list[str] = []

        # Check entity representation
        for ent in slots.target_entities:
            ent_clean = ent.lower().strip()
            if re.search(r"\b" + re.escape(ent_clean) + r"\b", all_facts_text):
                found_entities.add(ent)
            else:
                missing_slots.append(f"Data empiris untuk entitas '{ent}' belum ditemukan.")

        # Check quantitative metric requirement
        has_quant = any(f.is_quantitative for f in facts)
        if slots.requires_quantitative and not has_quant:
            missing_slots.append("Belum memuat metrik angka/kuantitatif eksplisit yang dipersyaratkan.")

        # Compute coverage score
        if slots.target_entities:
            entity_coverage = len(found_entities) / len(slots.target_entities)
        else:
            entity_coverage = 1.0

        quant_factor = 1.0 if (not slots.requires_quantitative or has_quant) else 0.5
        coverage_score = round(entity_coverage * quant_factor, 2)

        if not missing_slots and coverage_score >= 0.99:
            status = "complete"
        elif facts and coverage_score > 0.0:
            status = "partial"
        else:
            status = "no_data"

        return status, missing_slots, coverage_score, found_entities

    def extract_heuristic(self, subcontract: SubContract) -> SubagentArtifact:
        """Zero-token deterministic fact extraction with self-sufficiency audit & local self-healing."""
        slots = self.decompose_slots(subcontract)
        if not subcontract.evidence_slice:
            status, missing, cov, _ = self.evaluate_sufficiency([], slots)
            return SubagentArtifact(
                subcontract_id=subcontract.id,
                role=subcontract.role,
                status="no_data",
                extracted_facts=[],
                missing_requirements=[f"Tidak ada potongan dokumen lokal untuk {subcontract.id}."] + missing,
                coverage_score=0.0,
                missing_slots=missing,
            )

        facts: list[ExtractedFact] = []
        seen_claims: set[str] = set()
        obj_terms = set(re.findall(r"\b\w{4,}\b", subcontract.objective.lower()))

        # Pass 1: Parse candidate sentences from all evidence chunks
        all_candidate_sentences: list[tuple[str, str, bool, str | None]] = []
        for chunk in subcontract.evidence_slice:
            stem = chunk.source
            raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", chunk.text.strip())
            for sent in raw_sentences:
                clean_s = re.sub(r"\s+", " ", sent).strip(" -*•#")
                if len(clean_s) < 25 or len(clean_s) > 350:
                    continue
                s_lower = clean_s.lower()
                if any(f.lower() in s_lower for f in subcontract.forbidden_scope if len(f) > 4):
                    continue
                has_num = bool(
                    re.search(
                        r"\b\d+(?:[.,]\d+)?\s*(?:%|ha|hektar|hektare|km|ton|ribu|juta|miliar|orang|kasus|titik|hotspot|tahun|ms|fps|gb|mb|kb|flops|flosp|p\s*[<>=])\b",
                        s_lower,
                    )
                )
                year_match = re.search(r"\b(19\d{2}|20\d{2})\b", clean_s)
                year_val = year_match.group(1) if year_match else None
                all_candidate_sentences.append((clean_s, stem, has_num, year_val))

        # Score candidates against general objective & target entities
        scored_sentences: list[tuple[int, str, str, bool, str | None]] = []
        for clean_s, stem, has_num, year_val in all_candidate_sentences:
            s_lower = clean_s.lower()
            overlap = sum(1 for t in obj_terms if t in s_lower)
            entity_overlap = sum(2 for ent in slots.target_entities if ent.lower() in s_lower)
            score = overlap * 2 + entity_overlap * 3 + (3 if has_num else 0) + (1 if year_val else 0)
            scored_sentences.append((score, clean_s, stem, has_num, year_val))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        for _, sent_text, stem, has_num, year_val in scored_sentences[:3]:
            norm_key = sent_text[:60].lower()
            if norm_key in seen_claims:
                continue
            seen_claims.add(norm_key)
            facts.append(
                ExtractedFact(
                    claim_or_metric=sent_text,
                    citation_stem=stem,
                    year=year_val,
                    is_quantitative=has_num,
                )
            )

        # Pass 2: Local Self-Healing Loop for Missing Slots
        status, missing, coverage, found_ents = self.evaluate_sufficiency(facts, slots)
        if status != "complete":
            missing_entities = [ent for ent in slots.target_entities if ent not in found_ents]
            for missing_ent in missing_entities:
                m_low = missing_ent.lower()
                for clean_s, stem, has_num, year_val in all_candidate_sentences:
                    if m_low in clean_s.lower():
                        norm_key = clean_s[:60].lower()
                        if norm_key not in seen_claims:
                            seen_claims.add(norm_key)
                            facts.append(
                                ExtractedFact(
                                    claim_or_metric=clean_s,
                                    citation_stem=stem,
                                    year=year_val,
                                    is_quantitative=has_num,
                                )
                            )
                            found_ents.add(missing_ent)
                            break

            # If quantitative metrics were required and still missing, hunt for any quantitative candidate
            if slots.requires_quantitative and not any(f.is_quantitative for f in facts):
                for clean_s, stem, has_num, year_val in all_candidate_sentences:
                    if has_num:
                        norm_key = clean_s[:60].lower()
                        if norm_key not in seen_claims:
                            seen_claims.add(norm_key)
                            facts.append(
                                ExtractedFact(
                                    claim_or_metric=clean_s,
                                    citation_stem=stem,
                                    year=year_val,
                                    is_quantitative=True,
                                )
                            )
                            break

            # Re-evaluate sufficiency after self-healing pass
            status, missing, coverage, found_ents = self.evaluate_sufficiency(facts, slots)

        w_notes = "; ".join(missing) if missing else ("Fakta terekstrak lengkap memenuhi seluruh slot." if facts else "Tidak ada fakta yang cocok pada bukti lokal.")
        return SubagentArtifact(
            subcontract_id=subcontract.id,
            role=subcontract.role,
            status=status,
            extracted_facts=facts,
            missing_requirements=missing,
            worker_notes=w_notes,
            coverage_score=coverage,
            missing_slots=missing,
            input_tokens=0,
            output_tokens=0,
        )

    def execute(self, subcontract: SubContract) -> SubagentArtifact:
        """Run subcontract in an isolated context window, discarding internal prompts upon completion."""
        slots = self.decompose_slots(subcontract)
        if not subcontract.evidence_slice:
            return self.extract_heuristic(subcontract)

        if not self._can_use_live_llm():
            return self.extract_heuristic(subcontract)

        try:
            slice_blocks = []
            for idx, ch in enumerate(subcontract.evidence_slice, 1):
                snippet = ch.text[:1200].strip()
                slice_blocks.append(f"[Chunk #{idx} | stem={ch.source}]\n{snippet}")
            context_str = "\n\n".join(slice_blocks)

            target_ents_str = ", ".join(slots.target_entities) if slots.target_entities else "seluruh entitas terkait"
            system_prompt = (
                f"You are an isolated Scripcy Subagent ({subcontract.role}).\n"
                f"Required Target Entities: {target_ents_str}\n"
                f"Quantitative Metrics Required: {'YES (numbers, %, metrics)' if slots.requires_quantitative else 'No'}\n"
                "Extract ONLY verified facts, empirical metrics, and years from the provided chunks.\n"
                "DO NOT write introductory prose, theoretical filler, or out-of-scope commentary.\n"
                "Return strict JSON:\n"
                "{\n"
                '  "status": "complete" | "partial" | "no_data",\n'
                '  "extracted_facts": [\n'
                '    {"claim_or_metric": "...", "citation_stem": "...", "year": "2023", "is_quantitative": true}\n'
                "  ],\n"
                '  "missing_requirements": ["..."]\n'
                "}"
            )
            dos_lines = "\n".join(f"  - {d}" for d in (subcontract.dos or ["Ekstrak data dan fakta spesifik."]))
            donts_lines = "\n".join(f"  - {d}" for d in (subcontract.donts or ["Dilarang melebar ke topik out-of-scope."]))

            user_prompt = (
                f"OBJECTIVE: {subcontract.objective}\n\n"
                f"DO (WAJIB DILAKUKAN):\n{dos_lines}\n\n"
                f"DONT (DILARANG):\n{donts_lines}\n\n"
                f"ALLOWED SCOPE: {', '.join(subcontract.allowed_scope)}\n"
                f"FORBIDDEN DRIFT: {', '.join(subcontract.forbidden_scope)}\n\n"
                f"EVIDENCE SLICE:\n{context_str}"
            )

            in_tok = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)

            if self.llm_completion_fn is not None:
                raw_data = self.llm_completion_fn(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=subcontract.max_output_tokens,
                )
                content = json.dumps(raw_data) if isinstance(raw_data, dict) else str(raw_data)
            else:
                import litellm
                litellm.suppress_debug_info = True
                litellm.set_verbose = False
                litellm.drop_params = True

                model_name = self.engine.config.default_model
                api_key = self.engine.config.get_api_key_for_model(model_name)
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
                        {"role": "user", "content": user_prompt},
                    ],
                    "api_key": api_key,
                    "temperature": 0.1,
                    "max_tokens": subcontract.max_output_tokens,
                    "response_format": {"type": "json_object"},
                    "timeout": 18,
                }
                if api_base:
                    kwargs["api_base"] = api_base

                resp = litellm.completion(**kwargs)
                content = resp.choices[0].message.content or "{}"

            out_tok = estimate_tokens(content)
            parsed = json.loads(_clean_json_payload(content))
            raw_facts = parsed.get("extracted_facts", [])
            valid_stems = {c.source for c in subcontract.evidence_slice}
            default_stem = subcontract.evidence_slice[0].source

            facts: list[ExtractedFact] = []
            for item in raw_facts[:6]:
                if not isinstance(item, dict) or not item.get("claim_or_metric"):
                    continue
                stem = str(item.get("citation_stem") or default_stem)
                if stem not in valid_stems:
                    stem = default_stem
                facts.append(
                    ExtractedFact(
                        claim_or_metric=str(item["claim_or_metric"]).strip(),
                        citation_stem=stem,
                        year=str(item["year"]) if item.get("year") else None,
                        is_quantitative=bool(item.get("is_quantitative", False)),
                    )
                )

            # Evaluate Sufficiency on LLM-extracted facts
            status_val, missing_list, coverage_val, found_ents = self.evaluate_sufficiency(facts, slots)

            # Local Self-Healing Loop if LLM missed some slots that exist in raw evidence slice
            if status_val != "complete" and subcontract.evidence_slice:
                missing_entities = [ent for ent in slots.target_entities if ent not in found_ents]
                if missing_entities:
                    heur = self.extract_heuristic(subcontract)
                    for hf in heur.extracted_facts:
                        if any(me.lower() in hf.claim_or_metric.lower() for me in missing_entities):
                            facts.append(hf)
                    status_val, missing_list, coverage_val, found_ents = self.evaluate_sufficiency(facts, slots)

            if not facts:
                heur = self.extract_heuristic(subcontract)
                heur.input_tokens = in_tok
                heur.output_tokens = out_tok
                return heur

            w_note = str(parsed.get("worker_notes") or "; ".join(missing_list) or ("Fakta terekstrak sesuai kontrak." if facts else "Data tidak ditemukan."))
            return SubagentArtifact(
                subcontract_id=subcontract.id,
                role=subcontract.role,
                status=status_val,
                extracted_facts=facts,
                missing_requirements=missing_list,
                worker_notes=w_note,
                coverage_score=coverage_val,
                missing_slots=missing_list,
                input_tokens=in_tok,
                output_tokens=out_tok,
            )
        except Exception:
            return self.extract_heuristic(subcontract)


class SubagentPool:
    """Semaphore-bounded pool executing read-only extraction subagents in parallel (max_concurrent=2)."""

    def __init__(
        self,
        engine: LLMEngine | None = None,
        max_concurrent_llm: int = 2,
        llm_completion_fn: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.engine = engine
        self.max_concurrent_llm = max(1, max_concurrent_llm)
        self._semaphore = threading.Semaphore(self.max_concurrent_llm)
        self.llm_completion_fn = llm_completion_fn

    def run_subcontract(
        self,
        subcontract: SubContract,
        on_start: Callable[[SubContract], None] | None = None,
        on_done: Callable[[SubContract, SubagentArtifact], None] | None = None,
    ) -> SubagentArtifact:
        """Acquire semaphore slot, run isolated subagent worker, and release slot."""
        with self._semaphore:
            if on_start:
                on_start(subcontract)
            worker = SubagentWorker(engine=self.engine, llm_completion_fn=self.llm_completion_fn)
            artifact = worker.execute(subcontract)
            if on_done:
                on_done(subcontract, artifact)
            return artifact

    def run_parallel_extractions(
        self,
        subcontracts: list[SubContract],
        on_start: Callable[[SubContract], None] | None = None,
        on_done: Callable[[SubContract, SubagentArtifact], None] | None = None,
    ) -> list[SubagentArtifact]:
        """Fan-out independent read-only subcontracts concurrently up to max_concurrent_llm."""
        if not subcontracts:
            return []
        if len(subcontracts) == 1:
            return [self.run_subcontract(subcontracts[0], on_start=on_start, on_done=on_done)]

        results_by_id: dict[str, SubagentArtifact] = {}
        with ThreadPoolExecutor(max_workers=self.max_concurrent_llm) as pool:
            future_map = {
                pool.submit(self.run_subcontract, sc, on_start, on_done): sc.id
                for sc in subcontracts
            }
            for fut in as_completed(future_map):
                sc_id = future_map[fut]
                try:
                    results_by_id[sc_id] = fut.result()
                except Exception as exc:
                    results_by_id[sc_id] = SubagentArtifact(
                        subcontract_id=sc_id,
                        role="VaultExtractor",
                        status="no_data",
                        missing_requirements=[f"Subagent error: {exc}"],
                    )

        return [results_by_id[sc.id] for sc in subcontracts if sc.id in results_by_id]
