"""Hybrid Fan-Out / Batch-Barrier / Fan-In Subagent Orchestrator for Scripcy."""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Any, Callable

from pydantic import BaseModel, Field

from scripcy.core.contractor import ContractDecomposer, ExecutionPlan, SubContract
from scripcy.core.contracts import TaskContract
from scripcy.core.critic import DevilsAdvocate
from scripcy.core.crusher import SmartCrusher
from scripcy.core.engine import EvidenceChunk, LLMEngine
from scripcy.core.search import sanitize_filename
from scripcy.core.subagent import ExtractedFact, SubagentArtifact, SubagentPool
from scripcy.core.tasks import TaskManager


class OrchestrationResult(BaseModel):
    """Aggregated output from the SubagentOrchestrator ready for single-pass final assembly."""

    plan: ExecutionPlan
    artifacts: list[SubagentArtifact] = Field(default_factory=list)
    verified_facts: list[ExtractedFact] = Field(default_factory=list)
    missing_notes: list[str] = Field(default_factory=list)
    newly_ingested_stems: list[str] = Field(default_factory=list)
    attempted_search_channels: list[str] = Field(default_factory=lambda: ["vault_lokal"])
    subagent_input_tokens: int = 0
    subagent_output_tokens: int = 0
    tree_lines: list[str] = Field(default_factory=list)
    assembly_directive: str = ""
    grounding_score: float = 1.0
    plan_audit: Any = None
    evidence_audit: Any = None
    evidence_adjudication: Any = None


class SubagentOrchestrator:
    """Coordinates isolated subagents using Fan-Out parallel reads, a single Batch Write Barrier, and Fan-In audit."""

    def __init__(
        self,
        engine: LLMEngine | None = None,
        task_manager: TaskManager | None = None,
        scout: Any | None = None,
        web_scout: Any | None = None,
        vault_manager: Any | None = None,
        workspace: Any | None = None,
        max_concurrent_llm: int = 2,
        llm_completion_fn: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.engine = engine
        self.task_manager = task_manager
        self.scout = scout
        self.web_scout = web_scout
        self.vault_manager = vault_manager
        self.workspace = workspace
        self.decomposer = ContractDecomposer()
        self.pool = SubagentPool(
            engine=engine,
            max_concurrent_llm=max_concurrent_llm,
            llm_completion_fn=llm_completion_fn,
        )
        self.auditor = DevilsAdvocate(engine=engine)

    def _is_mock_test_env(self) -> bool:
        if "PYTEST_CURRENT_TEST" in os.environ:
            return True
        if self.engine and (
            getattr(self.engine, "_acompletion_fn", None) is not None
            or getattr(self.engine, "acompletion_fn", None) is not None
        ):
            return True
        cfg = getattr(self.engine, "config", None) if self.engine else None
        if cfg:
            if str(getattr(cfg, "default_model", "")).startswith("mock"):
                return True
            if any(str(v).startswith("mock") for v in getattr(cfg, "api_keys", {}).values()):
                return True
        return False

    def format_tree_status(
        self,
        contract: TaskContract,
        subcontract_states: dict[str, tuple[str, str]],
    ) -> str:
        """Render the CLI subagent execution tree with status icons."""
        mode_tag = f"Kontrak: {contract.archetype}"
        lines = [
            f"[bold cyan]●[/bold cyan] [bold bright_white][Orchestrator][/bold bright_white] "
            f"[dim]Mengelola worker terisolasi ({mode_tag})...[/dim]"
        ]
        items = list(subcontract_states.items())
        for idx, (label, (state, detail)) in enumerate(items):
            branch = "└──" if idx == len(items) - 1 else "├──"
            if state == "done":
                icon = "[bold green]✓[/bold green]"
            elif state == "running":
                icon = "[bold cyan]⠋[/bold cyan]"
            elif state == "waiting":
                icon = "[yellow]⏳[/yellow]"
            else:
                icon = "[dim]○[/dim]"
            lines.append(f"  {branch} {icon} [bold bright_white]{label}[/bold bright_white]: [dim]{detail}[/dim]")
        return "\n".join(lines)

    async def _execute_autonomous_scout_subcontracts(
        self,
        scout_subcontracts: list[SubContract],
        session_id: str | None = None,
        cancel_event: threading.Event | None = None,
    ) -> list[str]:
        """Run WebScout and LitScout subcontracts in parallel and coalesce document ingestion into a single batch."""
        if not scout_subcontracts:
            return []

        is_mock = self._is_mock_test_env()
        ingested_stems: list[str] = []

        async def _run_web(sc: SubContract) -> list[str]:
            stems: list[str] = []
            if not self.web_scout:
                return stems
            is_web_mocked = (
                hasattr(self.web_scout, "_is_mock")
                or "search_official_web" in getattr(self.web_scout, "__dict__", {})
                or "Mock" in type(getattr(self.web_scout, "search_official_web", None)).__name__
                or "Mock" in type(self.web_scout).__name__
            )
            if is_mock and not is_web_mocked:
                return stems

            vec_vault = getattr(self.vault_manager, "vector_vault", None) if self.vault_manager else None
            for q in sc.search_queries[:2]:
                if cancel_event and cancel_event.is_set():
                    break
                try:
                    hits = await self.web_scout.search_official_web(q, limit=2, require_official=True)
                    for hit in (hits or [])[:2]:
                        parsed = await self.web_scout.ingest_url(hit.url, vector_vault=vec_vault)
                        if parsed and getattr(parsed, "metadata", None) and parsed.metadata.cite_key:
                            stems.append(parsed.metadata.cite_key)
                except Exception:
                    continue
            return stems

        async def _run_lit(sc: SubContract) -> list[str]:
            stems: list[str] = []
            if not self.scout:
                return stems
            is_lit_mocked = (
                hasattr(self.scout, "_is_mock")
                or "search_multi_source" in getattr(self.scout, "__dict__", {})
                or "Mock" in type(getattr(self.scout, "search_multi_source", None)).__name__
                or "Mock" in type(self.scout).__name__
            )
            if is_mock and not is_lit_mocked:
                return stems

            from scripcy.core.scout import ScoutQuery
            for q in sc.search_queries[:2]:
                if cancel_event and cancel_event.is_set():
                    break
                try:
                    sq = ScoutQuery(keywords=q, limit=3)
                    papers = await self.scout.search_multi_source(sq, limit=3)
                    for p in (papers or [])[:2]:
                        stem = sanitize_filename(p.title)
                        if p.has_pdf and p.pdf_url:
                            try:
                                await self.scout.download_scout_paper(p)
                                stems.append(stem)
                                continue
                            except Exception:
                                pass
                        await asyncio.to_thread(self.scout.ingest_abstract_paper, p)
                        stems.append(stem)
                except Exception:
                    continue
            return stems

        coros = []
        for sc in scout_subcontracts:
            if sc.role == "WebScout":
                coros.append(_run_web(sc))
            elif sc.role == "LitScout":
                coros.append(_run_lit(sc))

        if coros:
            results = await asyncio.gather(*coros, return_exceptions=True)
            for res in results:
                if isinstance(res, list):
                    for s in res:
                        if s not in ingested_stems:
                            ingested_stems.append(s)

        # Single Batch Indexing Barrier in CPU lane when new documents are downloaded.
        if ingested_stems and self.vault_manager:
            def _single_batch_index() -> str:
                try:
                    self.vault_manager.reload_vault(force=False)
                except Exception:
                    pass
                return f"Indexed {len(ingested_stems)} documents"

            if self.task_manager:
                _, barrier_task, _ = self.task_manager.submit_coalesced_batch(
                    io_specs=[],
                    batch_barrier_name=f"Batch Vault Index ({len(ingested_stems)} docs)",
                    batch_barrier_fn=_single_batch_index,
                    session_id=session_id,
                )
                await asyncio.to_thread(
                    self.task_manager.wait_for_tasks,
                    [barrier_task.id],
                    15.0,
                    cancel_event,
                )
            else:
                await asyncio.to_thread(_single_batch_index)

        return ingested_stems

    def build_assembly_directive(
        self,
        contract: TaskContract,
        verified_facts: list[ExtractedFact],
        missing_notes: list[str],
    ) -> str:
        """Build a compact, token-efficient fact ledger directive for the single-pass final synthesis."""
        parts: list[str] = []

        if verified_facts:
            tsv_ledger = SmartCrusher.crush_verified_facts_ledger(verified_facts[:10])
            parts.append(
                "[LEDGER FAKTA TERVERIFIKASI SUBAGENT (TSV SMARTCRUSHER - INTI JAWABAN)]:\n"
                f"```tsv\n{tsv_ledger}\n```"
            )

        margin_disp = int(round(contract.token_margin_pct * 100))
        parts.append(
            f"[ALOKASI KUOTA TOKEN KONTRAK ABSOLUT]:\n"
            f"- Target Token Output: ~{contract.target_output_tokens} token "
            f"(Margin +-{margin_disp}% -> {contract.effective_min_tokens} s/d {contract.effective_max_tokens} token)."
        )

        if contract.archetype == "DATA_FACTUAL":
            parts.append(
                "[ATURAN KETAT MODE DATA_FACTUAL]:\n"
                "- Sajikan data/angka secara langsung (gunakan tabel atau daftar poin terukur beserta sitasi).\n"
                "- DILARANG menulis sub-bab tinjauan pustaka (Bab 2), latar belakang teori panjang, atau pembahasan birokrasi yang tidak ditanyakan.\n"
                "- Jika data yang tersedia di sumber hanya mencakup tahun atau wilayah tertentu, tuliskan data yang ada apa adanya dan tambahkan 1 kalimat catatan cakupan data secara jujur."
            )
        elif not contract.allow_counter_arguments:
            parts.append(
                "[ATURAN SCOPE GUARD]:\n"
                "- Jawab langsung ke substansi pertanyaan pengguna tanpa melebar ke sub-topik di luar cakupan."
            )

        if missing_notes and contract.allow_honest_gap:
            parts.append(
                "[CATATAN CAKUPAN BUKTI (HONEST DATA GAP)]:\n"
                + "\n".join(f"  - {m}" for m in missing_notes[:2])
            )

        parts.append(
            "[INSTRUKSI DRAF AKHIR GOLDEN PASS]:\n"
            "- Seluruh investigasi Worker telah selesai tuntas. DILARANG memanggil perintah alat atau sintaks tool internal (seperti /web, /scout, /read).\n"
            "- TULISKAN JAWABAN AKHIR LENGKAP secara langsung berdasarkan fakta dari ledger di atas."
        )

        return "\n\n".join(parts)

    async def orchestrate(
        self,
        query: str,
        contract: TaskContract,
        evidence: list[EvidenceChunk],
        session_id: str | None = None,
        enable_autonomous_scout: bool = True,
        cancel_event: threading.Event | None = None,
        on_tree_update: Callable[[str], None] | None = None,
    ) -> OrchestrationResult:
        """Execute Fan-Out subagents, Single Batch Write Barrier, and Fan-In Scope Audit."""
        # 1. Dynamic decomposition into granular small contracts
        raw_plan = self.decomposer.decompose(
            query=query,
            contract=contract,
            evidence=evidence,
            enable_external_scout=enable_autonomous_scout,
        )

        # 2. Pre-flight Devil's Advocate Audit (challenges unsubstantial/redundant tasks & proposes additions)
        plan_audit = self.auditor.audit_execution_plan(
            query=query,
            contract=contract,
            subcontracts=raw_plan.subcontracts,
        )

        # 3. Contractor Adjudication of Plan Audit (prunes challenged subcontracts and adds accepted ones)
        plan = self.decomposer.adjudicate_plan_audit(raw_plan, plan_audit)

        # Ensure newly added local subcontracts receive an evidence slice
        for sc in plan.subcontracts:
            if sc.target_channel == "vault_lokal" and not sc.evidence_slice and evidence:
                sc.evidence_slice = self.decomposer.slice_evidence(query, evidence, limit=3)

        local_extractors = [
            sc for sc in plan.subcontracts
            if sc.target_channel == "vault_lokal" and sc.role != "GapExtractor"
        ]
        scout_contracts = [
            sc for sc in plan.subcontracts
            if sc.target_channel in ("web_resmi", "repositori_jurnal") or sc.role in ("WebScout", "LitScout")
        ]
        gap_extractors = [sc for sc in plan.subcontracts if sc.role == "GapExtractor"]

        states: dict[str, tuple[str, str]] = {}
        if plan_audit and (plan_audit.prune_ids or plan_audit.additions):
            p_cnt = len(plan_audit.prune_ids)
            a_cnt = len(plan_audit.additions)
            states["Audit Rencana [Devil's Advocate]"] = (
                "done",
                f"Diaudit: {p_cnt} dipangkas (tidak substansial), {a_cnt} usulan penambahan",
            )
        for sc in local_extractors:
            states[f"Worker [{sc.role}]"] = ("running", f"Mengekstrak {len(sc.evidence_slice)} chunk lokal (Paralel)")
        for sc in scout_contracts:
            states[f"Worker [{sc.role}]"] = ("running", f"Menelusuri sumber eksternal ({sc.target_channel})")
        for sc in gap_extractors:
            states[f"Worker [{sc.role}]"] = ("waiting", "Menunggu hasil pencarian & batch index (Antre)")
        states["Worker [ScopeAuditor]"] = ("waiting", "Menunggu seluruh worker selesai (Antre)")

        if on_tree_update:
            on_tree_update(self.format_tree_status(contract, states))

        local_future = asyncio.to_thread(self.pool.run_parallel_extractions, local_extractors)
        scout_future = self._execute_autonomous_scout_subcontracts(
            scout_contracts,
            session_id=session_id,
            cancel_event=cancel_event,
        )

        local_artifacts, newly_ingested_stems = await asyncio.gather(local_future, scout_future)

        # 1x automated retry fallback for any local extractor that returned no facts
        for idx, art in enumerate(local_artifacts):
            if (art.status == "no_data" or not art.extracted_facts) and idx < len(local_extractors):
                sc = local_extractors[idx]
                if sc.evidence_slice:
                    from scripcy.core.subagent import SubagentWorker
                    worker = SubagentWorker(engine=self.engine)
                    fallback_art = worker.extract_heuristic(sc)
                    if fallback_art.extracted_facts:
                        local_artifacts[idx] = fallback_art
                        art = fallback_art

            states[f"Worker [{art.role}]"] = (
                "done",
                f"{len(art.extracted_facts)} fakta diekstrak ({art.tokens_used} tok)",
            )
        for sc in scout_contracts:
            states[f"Worker [{sc.role}]"] = (
                "done",
                f"{len(newly_ingested_stems)} sumber baru diindeks" if newly_ingested_stems else "Selesai",
            )

        all_artifacts = list(local_artifacts)

        if gap_extractors and newly_ingested_stems and self.workspace and self.engine:
            new_paths = [
                self.workspace.get_cache_path(f"{s}.md")
                for s in newly_ingested_stems
                if self.workspace.get_cache_path(f"{s}.md").exists()
            ]
            if new_paths:
                vec_vault = getattr(self.vault_manager, "vector_vault", None) if self.vault_manager else None
                new_ev = self.engine.evidence_synthesizer.retrieve_evidence(
                    query=query,
                    documents=new_paths,
                    max_chunks=4,
                    vector_vault=vec_vault,
                )
                for ge in gap_extractors:
                    ge.evidence_slice = self.decomposer.slice_evidence(query, new_ev, limit=3)
                    states[f"Worker [{ge.role}]"] = ("running", f"Mengekstrak {len(ge.evidence_slice)} chunk baru")
                gap_artifacts = await asyncio.to_thread(self.pool.run_parallel_extractions, gap_extractors)
                for g_art in gap_artifacts:
                    g_art.ingested_stems = list(newly_ingested_stems)
                    states[f"Worker [{g_art.role}]"] = (
                        "done",
                        f"{len(g_art.extracted_facts)} fakta baru diekstrak ({g_art.tokens_used} tok)",
                    )
                all_artifacts.extend(gap_artifacts)
            else:
                for ge in gap_extractors:
                    states[f"Worker [{ge.role}]"] = ("done", "Tidak ada dokumen baru")
        else:
            for ge in gap_extractors:
                states[f"Worker [{ge.role}]"] = ("done", "Dilewati (cukup dengan bukti lokal)")

        # 4. Post-Scout Evidence Audit by Devil's Advocate & Contractor Adjudication
        raw_max = getattr(getattr(self.engine, "config", None), "max_worker_expansions", 1)
        try:
            max_expansions = int(raw_max)
        except (TypeError, ValueError):
            max_expansions = 1
        prev_worker_notes = [art.worker_notes for art in all_artifacts if getattr(art, "worker_notes", None)]
        ev_audit = self.auditor.audit_worker_evidence(
            query=query,
            contract=contract,
            subcontracts=plan.subcontracts,
            artifacts=all_artifacts,
        )
        ev_adj = self.decomposer.adjudicate_evidence_audit(
            query=query,
            contract=contract,
            audit_review=ev_audit,
            expansion_count=0,
            max_expansions=max_expansions,
            previous_worker_notes=prev_worker_notes,
        )

        missing_notes_from_adj: list[str] = []
        if ev_adj.decision == "expand" and ev_adj.follow_up_subcontracts:
            states["Evaluasi Temuan [Devil's Advocate]"] = (
                "done",
                f"{len(ev_adj.gap_notes)} gap ditemukan. Contractor menugaskan pencarian lanjutan.",
            )
            follow_scouts = [sc for sc in ev_adj.follow_up_subcontracts if sc.role in ("WebScout", "LitScout")]
            follow_extractors = [sc for sc in ev_adj.follow_up_subcontracts if sc.role == "GapExtractor"]

            for sc in follow_scouts:
                states[f"Worker [{sc.role}]"] = ("running", f"Follow-up: {sc.objective[:45]}...")
            if on_tree_update:
                on_tree_update(self.format_tree_status(contract, states))

            follow_stems = await self._execute_autonomous_scout_subcontracts(
                follow_scouts,
                session_id=session_id,
                cancel_event=cancel_event,
            )
            for sc in follow_scouts:
                states[f"Worker [{sc.role}]"] = ("done", f"{len(follow_stems)} sumber baru diindeks" if follow_stems else "Selesai")

            if follow_extractors and follow_stems and self.workspace and self.engine:
                f_paths = [
                    self.workspace.get_cache_path(f"{s}.md")
                    for s in follow_stems
                    if self.workspace.get_cache_path(f"{s}.md").exists()
                ]
                if f_paths:
                    vec_vault = getattr(self.vault_manager, "vector_vault", None) if self.vault_manager else None
                    f_ev = self.engine.evidence_synthesizer.retrieve_evidence(
                        query=query,
                        documents=f_paths,
                        max_chunks=4,
                        vector_vault=vec_vault,
                    )
                    for fe in follow_extractors:
                        fe.evidence_slice = self.decomposer.slice_evidence(query, f_ev, limit=3)
                        states[f"Worker [{fe.role}]"] = ("running", "Mengekstrak temuan follow-up...")
                    f_artifacts = await asyncio.to_thread(self.pool.run_parallel_extractions, follow_extractors)
                    for f_art in f_artifacts:
                        f_art.ingested_stems = list(follow_stems)
                        states[f"Worker [{f_art.role}]"] = ("done", f"{len(f_art.extracted_facts)} fakta baru diekstrak")
                    all_artifacts.extend(f_artifacts)

            for s in follow_stems:
                if s not in newly_ingested_stems:
                    newly_ingested_stems.append(s)

            plan.subcontracts.extend(ev_adj.follow_up_subcontracts)
        elif ev_adj.decision == "honest_gap":
            states["Evaluasi Temuan [Contractor]"] = (
                "done",
                f"Mencatat {len(ev_adj.gap_notes)} Honest Data Gap (pencarian dihentikan).",
            )
            missing_notes_from_adj.extend(ev_adj.gap_notes)
        else:
            states["Evaluasi Temuan [Contractor]"] = ("done", "Seluruh Small Contract terbukti terpenuhi.")

        states["Worker [ScopeAuditor]"] = ("running", "Memvalidasi sitasi, memfilter out-of-scope & menyusun fact sheet...")
        audit_res = self.auditor.audit_fact_ledger(
            query=query,
            contract=contract,
            artifacts=all_artifacts,
        )
        verified_facts: list[ExtractedFact] = audit_res["verified_facts"]
        missing_notes: list[str] = audit_res["missing_notes"]
        for g_note in missing_notes_from_adj:
            if g_note not in missing_notes:
                missing_notes.append(g_note)
        if plan.has_evidence_gap and plan.gap_reason and not newly_ingested_stems:
            if plan.gap_reason not in missing_notes:
                missing_notes.insert(0, plan.gap_reason)

        total_raw_facts = len(verified_facts) + audit_res.get("dropped_count", 0)
        deterministic_grounding = round(len(verified_facts) / total_raw_facts, 2) if total_raw_facts > 0 else 1.0

        states["Worker [ScopeAuditor]"] = (
            "done",
            f"{len(verified_facts)} fakta terverifikasi (Fact Sheet Grounding: {int(deterministic_grounding * 100)}%)",
        )
        final_tree = self.format_tree_status(contract, states)
        if on_tree_update:
            on_tree_update(final_tree)

        directive = self.build_assembly_directive(
            contract=contract,
            verified_facts=verified_facts,
            missing_notes=missing_notes,
        )

        attempted_channels = ["vault_lokal"]
        if any(sc.target_channel == "web_resmi" or sc.role == "WebScout" for sc in scout_contracts):
            attempted_channels.append("web_resmi")
        if any(sc.target_channel == "repositori_jurnal" or sc.role == "LitScout" for sc in scout_contracts):
            attempted_channels.append("repositori_jurnal_multi_source")

        return OrchestrationResult(
            plan=plan,
            plan_audit=plan_audit,
            evidence_audit=ev_audit,
            evidence_adjudication=ev_adj,
            artifacts=all_artifacts,
            verified_facts=verified_facts,
            missing_notes=missing_notes,
            newly_ingested_stems=newly_ingested_stems,
            attempted_search_channels=attempted_channels,
            subagent_input_tokens=audit_res["subagent_input_tokens"],
            subagent_output_tokens=audit_res["subagent_output_tokens"],
            tree_lines=final_tree.splitlines(),
            assembly_directive=directive,
            grounding_score=deterministic_grounding,
        )


WorkerOrchestrator = SubagentOrchestrator
