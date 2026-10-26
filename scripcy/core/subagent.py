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
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def tokens_used(self) -> int:
        return self.input_tokens + self.output_tokens


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

    def extract_heuristic(self, subcontract: SubContract) -> SubagentArtifact:
        """Zero-token deterministic fact extraction from the subcontract's isolated evidence slice."""
        if not subcontract.evidence_slice:
            return SubagentArtifact(
                subcontract_id=subcontract.id,
                role=subcontract.role,
                status="no_data",
                extracted_facts=[],
                missing_requirements=[f"Tidak ada potongan dokumen lokal untuk {subcontract.id}."],
            )

        facts: list[ExtractedFact] = []
        seen_claims: set[str] = set()
        obj_terms = set(re.findall(r"\b\w{4,}\b", subcontract.objective.lower()))

        for chunk in subcontract.evidence_slice:
            stem = chunk.source
            raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", chunk.text.strip())
            scored_sentences: list[tuple[int, str, bool, str | None]] = []

            for sent in raw_sentences:
                clean_s = re.sub(r"\s+", " ", sent).strip(" -*•#")
                if len(clean_s) < 25 or len(clean_s) > 320:
                    continue
                s_lower = clean_s.lower()
                if any(f.lower() in s_lower for f in subcontract.forbidden_scope if len(f) > 4):
                    continue
                overlap = sum(1 for t in obj_terms if t in s_lower)
                has_num = bool(
                    re.search(
                        r"\b\d+(?:[.,]\d+)?\s*(?:%|ha|hektar|hektare|km|ton|ribu|juta|miliar|orang|kasus|titik|hotspot|tahun)\b",
                        s_lower,
                    )
                )
                year_match = re.search(r"\b(19\d{2}|20\d{2})\b", clean_s)
                year_val = year_match.group(1) if year_match else None
                score = overlap * 2 + (3 if has_num else 0) + (1 if year_val else 0)
                scored_sentences.append((score, clean_s, has_num, year_val))

            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            for _, sent_text, has_num, year_val in scored_sentences[:2]:
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
                if len(facts) >= 5:
                    break
            if len(facts) >= 5:
                break

        status: Literal["complete", "partial", "no_data"] = "complete" if facts else "no_data"
        missing: list[str] = []
        if subcontract.role == "VaultExtractor" and not any(f.is_quantitative for f in facts):
            if any(w in subcontract.objective.lower() for w in ("angka", "metrik", "data", "statistik")):
                status = "partial" if facts else "no_data"
                missing.append("Potongan dokumen lokal belum memuat metrik angka eksplisit yang diminta.")

        w_notes = "; ".join(missing) if missing else ("Fakta terekstrak dari potongan bukti lokal." if facts else "Tidak ada fakta yang cocok pada bukti lokal.")
        return SubagentArtifact(
            subcontract_id=subcontract.id,
            role=subcontract.role,
            status=status,
            extracted_facts=facts,
            missing_requirements=missing,
            worker_notes=w_notes,
            input_tokens=0,
            output_tokens=0,
        )

    def execute(self, subcontract: SubContract) -> SubagentArtifact:
        """Run subcontract in an isolated context window, discarding internal prompts upon completion."""
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

            system_prompt = (
                f"You are an isolated Scripcy Subagent ({subcontract.role}).\n"
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

            if not facts:
                heur = self.extract_heuristic(subcontract)
                heur.input_tokens = in_tok
                heur.output_tokens = out_tok
                return heur

            status_val = parsed.get("status", "complete")
            if status_val not in ("complete", "partial", "no_data"):
                status_val = "complete"

            missing_list = [str(m) for m in parsed.get("missing_requirements", [])[:3]]
            w_note = str(parsed.get("worker_notes") or "; ".join(missing_list) or ("Fakta terekstrak sesuai kontrak." if facts else "Data tidak ditemukan."))
            return SubagentArtifact(
                subcontract_id=subcontract.id,
                role=subcontract.role,
                status=status_val,
                extracted_facts=facts,
                missing_requirements=missing_list,
                worker_notes=w_note,
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
