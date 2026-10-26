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
            f"[dim]Mengelola subagent terisolasi ({mode_tag})...[/dim]"
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
        plan = self.decomposer.decompose(
            query=query,
            contract=contract,
            evidence=evidence,
            enable_external_scout=enable_autonomous_scout,
        )

        local_extractors = [sc for sc in plan.subcontracts if sc.role in ("VaultExtractor", "DialecticalExtractor")]
        scout_contracts = [sc for sc in plan.subcontracts if sc.role in ("WebScout", "LitScout")]
        gap_extractors = [sc for sc in plan.subcontracts if sc.role == "GapExtractor"]

        states: dict[str, tuple[str, str]] = {}
        for sc in local_extractors:
            states[f"Subagent [{sc.role}]"] = ("running", f"Mengekstrak {len(sc.evidence_slice)} chunk lokal (Paralel)")
        for sc in scout_contracts:
            states[f"Subagent [{sc.role}]"] = ("running", "Menelusuri sumber eksternal (Paralel)")
        for sc in gap_extractors:
            states[f"Subagent [{sc.role}]"] = ("waiting", "Menunggu hasil pencarian & batch index (Antre)")
        states["Subagent [ScopeAuditor]"] = ("waiting", "Menunggu seluruh ekstraktor selesai (Antre)")

        if on_tree_update:
            on_tree_update(self.format_tree_status(contract, states))

        local_future = asyncio.to_thread(self.pool.run_parallel_extractions, local_extractors)
        scout_future = self._execute_autonomous_scout_subcontracts(
            scout_contracts,
            session_id=session_id,
            cancel_event=cancel_event,
        )

        local_artifacts, newly_ingested_stems = await asyncio.gather(local_future, scout_future)

        for art in local_artifacts:
            states[f"Subagent [{art.role}]"] = (
                "done",
                f"{len(art.extracted_facts)} fakta diekstrak ({art.tokens_used} tok)",
            )
        for sc in scout_contracts:
            states[f"Subagent [{sc.role}]"] = (
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
                    states[f"Subagent [{ge.role}]"] = ("running", f"Mengekstrak {len(ge.evidence_slice)} chunk baru")
                gap_artifacts = await asyncio.to_thread(self.pool.run_parallel_extractions, gap_extractors)
                for g_art in gap_artifacts:
                    g_art.ingested_stems = list(newly_ingested_stems)
                    states[f"Subagent [{g_art.role}]"] = (
                        "done",
                        f"{len(g_art.extracted_facts)} fakta baru diekstrak ({g_art.tokens_used} tok)",
                    )
                all_artifacts.extend(gap_artifacts)
            else:
                for ge in gap_extractors:
                    states[f"Subagent [{ge.role}]"] = ("done", "Tidak ada dokumen baru")
        else:
            for ge in gap_extractors:
                states[f"Subagent [{ge.role}]"] = ("done", "Dilewati (cukup dengan bukti lokal)")

        states["Subagent [ScopeAuditor]"] = ("running", "Memvalidasi sitasi & memfilter out-of-scope...")
        audit_res = self.auditor.audit_fact_ledger(
            query=query,
            contract=contract,
            artifacts=all_artifacts,
        )
        verified_facts: list[ExtractedFact] = audit_res["verified_facts"]
        missing_notes: list[str] = audit_res["missing_notes"]
        if plan.has_evidence_gap and plan.gap_reason and not newly_ingested_stems:
            if plan.gap_reason not in missing_notes:
                missing_notes.insert(0, plan.gap_reason)

        states["Subagent [ScopeAuditor]"] = (
            "done",
            f"{len(verified_facts)} fakta terverifikasi siap dirakit",
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
        if any(sc.role == "WebScout" for sc in scout_contracts):
            attempted_channels.append("web_resmi")
        if any(sc.role == "LitScout" for sc in scout_contracts):
            attempted_channels.append("repositori_jurnal_multi_source")

        return OrchestrationResult(
            plan=plan,
            artifacts=all_artifacts,
            verified_facts=verified_facts,
            missing_notes=missing_notes,
            newly_ingested_stems=newly_ingested_stems,
            attempted_search_channels=attempted_channels,
            subagent_input_tokens=audit_res["subagent_input_tokens"],
            subagent_output_tokens=audit_res["subagent_output_tokens"],
            tree_lines=final_tree.splitlines(),
            assembly_directive=directive,
        )
