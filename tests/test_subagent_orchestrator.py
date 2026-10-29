"""Tests for isolated Subagents, Contractor context slicing, Heavy Task Batching, and SubagentOrchestrator."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from scripcy.core.contractor import ContractDecomposer, SubContract
from scripcy.core.contracts import TaskContractExtractor
from scripcy.core.critic import DevilsAdvocate
from scripcy.core.engine import EvidenceChunk
from scripcy.core.orchestrator import SubagentOrchestrator
from scripcy.core.subagent import SubagentPool, SubagentWorker
from scripcy.core.tasks import TaskManager


def test_task_contract_data_factual_classification() -> None:
    """Query asking for 'data karhutla 5 tahun terakhir' must map to DATA_FACTUAL with counter-arguments disabled."""
    extractor = TaskContractExtractor()
    contract = extractor.extract("carikan aku data karhutla di indonesia 5 tahun terakhir")

    assert contract.archetype == "DATA_FACTUAL"
    assert contract.allow_counter_arguments is False
    assert contract.allow_honest_gap is True
    assert contract.max_words <= 450
    assert any("Bab 2" in t or "birokrasi" in t for t in contract.forbidden_drift_topics)


def test_contractor_context_slicing_and_gap_detection() -> None:
    """Contractor slices chunks per subagent and schedules WebScout/LitScout + waiting GapExtractor on gap."""
    extractor = TaskContractExtractor()
    query = "carikan aku data karhutla di indonesia 5 tahun terakhir"
    contract = extractor.extract(query)

    local_chunks = [
        EvidenceChunk(
            chunk_id=f"c{i}",
            source=f"local_study_{i}",
            text=f"Studi kasus kebakaran gambut Kabupaten Bengkalis tahun 2018 mencatat {100 + i} titik panas.",
            token_estimate=25,
        )
        for i in range(6)
    ]

    decomposer = ContractDecomposer(max_chunks_per_subagent=3)
    plan = decomposer.decompose(query=query, contract=contract, evidence=local_chunks)

    assert plan.has_evidence_gap is True
    assert plan.is_fast_path is False

    vault_sc = next(sc for sc in plan.subcontracts if sc.role == "VaultExtractor")
    assert len(vault_sc.evidence_slice) == 3

    roles = [sc.role for sc in plan.subcontracts]
    assert "WebScout" in roles
    assert "LitScout" in roles
    assert "GapExtractor" in roles

    gap_sc = next(sc for sc in plan.subcontracts if sc.role == "GapExtractor")
    assert "sub-scout-web" in gap_sc.depends_on
    assert "sub-scout-lit" in gap_sc.depends_on


def test_subagent_worker_and_pool_parallel_execution() -> None:
    """SubagentPool executes isolated subcontracts concurrently within semaphore limits and returns compact artifacts."""
    chunks_a = [
        EvidenceChunk(
            chunk_id="ca",
            source="sipongi2023",
            text="Luas kebakaran hutan dan lahan Indonesia tahun 2023 tercatat sebesar 1.161.193 hektare menurut KLHK.",
            token_estimate=20,
        )
    ]
    chunks_b = [
        EvidenceChunk(
            chunk_id="cb",
            source="bengkalis2022",
            text="Pada tahun 2022 terjadi penurunan titik panas sebesar 45% akibat anomali La Nina basah.",
            token_estimate=20,
        )
    ]

    sc1 = SubContract(
        id="sub-1",
        role="VaultExtractor",
        objective="Ekstrak angka luas karhutla 2023",
        evidence_slice=chunks_a,
    )
    sc2 = SubContract(
        id="sub-2",
        role="VaultExtractor",
        objective="Ekstrak persentase penurunan titik panas 2022",
        evidence_slice=chunks_b,
    )

    active_count = 0
    max_observed = 0
    lock = threading.Lock()

    def mock_llm_fn(system_prompt: str, user_prompt: str, max_tokens: int) -> dict[str, Any]:
        nonlocal active_count, max_observed
        with lock:
            active_count += 1
            max_observed = max(max_observed, active_count)
        time.sleep(0.05)
        with lock:
            active_count -= 1
        stem = "sipongi2023" if "sipongi2023" in user_prompt else "bengkalis2022"
        return {
            "status": "complete",
            "extracted_facts": [
                {
                    "claim_or_metric": f"Fakta terukur dari {stem}: 1.161.193 hektare.",
                    "citation_stem": stem,
                    "year": "2023",
                    "is_quantitative": True,
                }
            ],
            "missing_requirements": [],
        }

    pool = SubagentPool(max_concurrent_llm=2, llm_completion_fn=mock_llm_fn)
    artifacts = pool.run_parallel_extractions([sc1, sc2])

    assert len(artifacts) == 2
    assert max_observed <= 2
    assert artifacts[0].extracted_facts[0].citation_stem == "sipongi2023"
    assert artifacts[1].extracted_facts[0].citation_stem == "bengkalis2022"
    assert all(a.tokens_used > 0 for a in artifacts)


def test_coalesced_batch_barrier_and_waiting_subagents(tmp_path) -> None:
    """submit_coalesced_batch deduplicates I/O, runs 1 CPU batch barrier, and wakes waiting_deps subagents."""
    tm = TaskManager(max_workers=4, db_path=tmp_path / "tasks.db")
    execution_order: list[str] = []
    gate = threading.Event()

    def io_worker() -> str:
        gate.wait(timeout=2.0)
        execution_order.append("io_done")
        return "downloaded"

    def barrier_worker() -> str:
        execution_order.append("batch_index_done")
        return "indexed"

    def subagent_worker() -> str:
        execution_order.append("subagent_done")
        return "extracted"

    io_specs = [
        {"name": "Web Scout 1", "task_type": "web", "target_fn": io_worker, "dedup_key": "web:karhutla"},
        {"name": "Web Scout Dup", "task_type": "web", "target_fn": io_worker, "dedup_key": "web:karhutla"},
    ]
    downstream_specs = [
        {"name": "GapExtractor Subagent", "task_type": "agent", "target_fn": subagent_worker},
    ]

    io_tasks, barrier_task, downstream_tasks = tm.submit_coalesced_batch(
        io_specs=io_specs,
        batch_barrier_name="Single Batch Vault Index",
        batch_barrier_fn=barrier_worker,
        session_id="sess-1",
        downstream_specs=downstream_specs,
    )

    assert len(io_tasks) == 1
    assert barrier_task.status == "waiting_deps"
    assert downstream_tasks[0].status == "waiting_deps"

    gate.set()
    tm.wait_for_tasks([downstream_tasks[0].id], timeout=5.0)

    assert execution_order == ["io_done", "batch_index_done", "subagent_done"]
    tm.shutdown()


def test_critic_scope_guard_does_not_force_counter_arguments_on_data_factual() -> None:
    """DevilsAdvocate does not penalize a direct data answer for lacking counter-arguments on DATA_FACTUAL contract."""
    extractor = TaskContractExtractor()
    contract = extractor.extract("carikan aku data karhutla di indonesia 5 tahun terakhir")
    critic = DevilsAdvocate()

    concise_data_answer = (
        "Berdasarkan laporan resmi KLHK [SiPongi, 2024], luas kebakaran hutan dan lahan di Indonesia "
        "pada tahun 2020 tercatat 296.942 ha, tahun 2021 sebesar 358.867 ha, tahun 2022 sebesar 204.894 ha, "
        "dan tahun 2023 mencapai 1.161.193 ha [KLHK, 2024]. Data tahun 2024 masih dalam tahap rekapitulasi berjalan."
    )

    rep = critic.critique_heuristic(
        query="carikan aku data karhutla di indonesia 5 tahun terakhir",
        answer=concise_data_answer,
        contract=contract,
    )

    assert rep.passed is True
    assert rep.score >= 8
    assert rep.counter_arguments == []


def test_orchestrator_autonomous_fan_out_and_fan_in(tmp_path) -> None:
    """SubagentOrchestrator runs local extraction + autonomous web scout, batch indexes once, and builds assembly directive."""
    extractor = TaskContractExtractor()
    query = "carikan aku data karhutla di indonesia 5 tahun terakhir"
    contract = extractor.extract(query)

    local_ev = [
        EvidenceChunk(
            chunk_id="c1",
            source="oki2018",
            text="Pada tahun 2018 kebakaran lahan gambut di Kabupaten OKI mencapai 12.400 hektare.",
            token_estimate=20,
        )
    ]

    mock_workspace = MagicMock()
    new_md_path = tmp_path / "sipongi2024.md"
    new_md_path.write_text(
        "Rekapitulasi nasional KLHK mencatat luas karhutla 2023 sebesar 1.161.193 hektare di seluruh Indonesia.",
        encoding="utf-8",
    )
    mock_workspace.get_cache_path.side_effect = lambda name: new_md_path if "sipongi2024" in name else (tmp_path / name)

    mock_web_scout = MagicMock()
    mock_web_scout._is_mock = True
    mock_hit = MagicMock()
    mock_hit.url = "https://sipongi.menlhk.go.id/data-2024"
    mock_web_scout.search_official_web = AsyncMock(return_value=[mock_hit])

    mock_parsed = MagicMock()
    mock_parsed.metadata.cite_key = "sipongi2024"
    mock_web_scout.ingest_url = AsyncMock(return_value=mock_parsed)

    mock_vault_manager = MagicMock()
    mock_vault_manager.vector_vault = None

    mock_engine = MagicMock()
    mock_engine.evidence_synthesizer.retrieve_evidence.return_value = [
        EvidenceChunk(
            chunk_id="c_new",
            source="sipongi2024",
            text="Rekapitulasi nasional KLHK mencatat luas karhutla tahun 2023 sebesar 1.161.193 hektare.",
            token_estimate=20,
        )
    ]

    tm = TaskManager(max_workers=2, db_path=tmp_path / "tasks.db")
    orchestrator = SubagentOrchestrator(
        engine=mock_engine,
        task_manager=tm,
        web_scout=mock_web_scout,
        vault_manager=mock_vault_manager,
        workspace=mock_workspace,
    )

    tree_snapshots: list[str] = []
    res = asyncio.run(
        orchestrator.orchestrate(
            query=query,
            contract=contract,
            evidence=local_ev,
            session_id="karhutla",
            enable_autonomous_scout=True,
            on_tree_update=lambda t: tree_snapshots.append(t),
        )
    )

    assert "sipongi2024" in res.newly_ingested_stems
    assert len(res.verified_facts) >= 2
    assert any(f.citation_stem == "sipongi2024" for f in res.verified_facts)
    assert "LEDGER FAKTA TERVERIFIKASI" in res.assembly_directive
    assert "ATURAN KETAT MODE DATA_FACTUAL" in res.assembly_directive
    assert len(tree_snapshots) >= 2
    assert "Worker [VaultExtractor]" in tree_snapshots[-1]
    assert "Worker [ScopeAuditor]" in tree_snapshots[-1]
    tm.shutdown()


def test_contract_token_margin_pct_and_ai_first_extraction() -> None:
    """AI-First contract extraction estimates target_output_tokens with configurable +-30% margin and search policy."""
    from scripcy.core.config import ScripcyConfig
    from scripcy.core.engine import LLMEngine

    cfg = ScripcyConfig()
    cfg.contract_unlimited_tokens = False
    assert cfg.contract_token_margin_pct == 0.30

    def fake_ai_contract_fn(sys_prompt: str, user_query: str) -> dict[str, Any]:
        return {
            "archetype": "DATA_FACTUAL",
            "primary_goal": "Menyajikan tabel luas karhutla Indonesia 5 tahun terakhir",
            "in_scope": ["Data luas karhutla per tahun (ha)", "Sumber resmi KLHK/BPS"],
            "out_of_scope": ["Tinjauan pustaka Bab 2", "Analisis sosiologis birokrasi"],
            "forbidden_drift_topics": ["Pembahasan konflik kelembagaan", "Sejarah regulasi kehutanan"],
            "format_requirements": "Tabel ringkas + 1 paragraf catatan cakupan data",
            "target_output_tokens": 300,
            "min_words": 80,
            "max_words": 350,
            "allow_counter_arguments": False,
            "allow_honest_gap": True,
            "definition_of_done": [
                "Tabel angka karhutla tersaji",
                "Sitasi sumber tercantum",
            ],
            "required_search_channels": [
                "vault_lokal",
                "web_resmi",
                "repositori_jurnal_multi_source",
                "reference_mapping_snowballing_backward_forward",
            ],
            "search_exhaustion_policy": (
                "Telusuri Vault Lokal, Web Resmi, Jurnal Multi-Source, dan Reference Mapping + Snowballing "
                "(Forward/Backward). Jika data tetap tidak ditemukan, hentikan pencarian dan laporkan jujur."
            ),
        }

    engine = LLMEngine(config=cfg)
    extractor = TaskContractExtractor(engine=engine, llm_completion_fn=fake_ai_contract_fn)
    contract = extractor.extract("carikan data karhutla 5 tahun terakhir")

    assert contract.is_absolute is True
    assert contract.archetype == "DATA_FACTUAL"
    assert contract.target_output_tokens == 300
    assert contract.token_margin_pct == 0.30
    assert contract.effective_min_tokens == 210
    assert contract.effective_max_tokens == 390
    assert contract.is_token_count_acceptable(280) is True
    assert contract.is_token_count_acceptable(450) is False
    assert "reference_mapping_snowballing_backward_forward" in contract.required_search_channels
    assert "Snowballing" in contract.search_exhaustion_policy

    # Verify custom margin from config (e.g. +-20%)
    cfg.contract_token_margin_pct = 0.20
    contract_20 = extractor.extract("carikan data karhutla 5 tahun terakhir")
    assert contract_20.token_margin_pct == 0.20
    assert contract_20.effective_min_tokens == 240
    assert contract_20.effective_max_tokens == 360


def test_contractor_gatekeeper_vetoes_devil_advocate_out_of_scope_and_exhausted_search() -> None:
    """Contractor vetoes Devil's Advocate complaints that push Bab 2 drift, counter-arguments, or post-exhaustion searches."""
    from scripcy.core.critic import CritiqueReport
    from scripcy.core.verifier import VerificationReport

    extractor = TaskContractExtractor()
    query = "carikan aku data karhutla di indonesia 5 tahun terakhir"
    contract = extractor.extract(query)

    # Devil's Advocate emits a failing report demanding Bab 2 theory, bureaucracy analysis, and more searches
    bad_critique = CritiqueReport(
        score=5,
        passed=False,
        weaknesses=[
            "Perlu membahas kerangka teoritis Bab 2 dan tata kelola birokrasi kehutanan secara mendalam.",
            "Data rekapitulasi nasional tahun 2024 belum lengkap, cari referensi tambahan.",
        ],
        counter_arguments=[
            "Terdapat perdebatan sosiologis mengenai efektivitas sanksi administratif korporasi.",
        ],
        recency_alert="Sebagian literatur lokal berasal dari tahun 2019.",
        needs_citation_mapping=True,
        critique_summary="Draf terlalu singkat dan belum membahas dinamika kelembagaan.",
    )
    clean_fact_verdict = VerificationReport(
        grounding_score=1.0,
        total_claims=3,
        supported_claims=3,
        unsupported_claims=0,
        hallucination_warning=None,
    )

    decomposer = ContractDecomposer()
    # All channels including reference mapping + backward/forward snowballing were already attempted
    exhausted_channels = {
        "vault_lokal",
        "web_resmi",
        "repositori_jurnal_multi_source",
        "reference_mapping_snowballing_backward_forward",
    }

    adjudication = decomposer.adjudicate_critique(
        query=query,
        draft="Berdasarkan data resmi [SiPongi, 2023], luas karhutla 2023 adalah 1.161.193 ha.",
        contract=contract,
        critique_report=bad_critique,
        fact_verdict=clean_fact_verdict,
        attempted_search_channels=exhausted_channels,
    )

    assert adjudication.should_revise is False
    assert adjudication.override_passed is True
    assert adjudication.override_score >= 8
    assert adjudication.search_exhausted is True
    assert len(adjudication.rejected_critiques) == 3
    assert adjudication.accepted_revisions == []
    assert adjudication.exhausted_search_disclosure is not None
    assert "Reference Mapping & Snowballing" in adjudication.exhausted_search_disclosure


def test_contractor_gatekeeper_allows_hallucination_fix_and_first_snowball() -> None:
    """Contractor allows revision when FactChecker flags hallucination or when snowballing has not yet run."""
    from scripcy.core.critic import CritiqueReport
    from scripcy.core.verifier import VerificationReport

    extractor = TaskContractExtractor()
    query = "carikan aku data karhutla di indonesia 5 tahun terakhir"
    contract = extractor.extract(query)

    critique_rep = CritiqueReport(
        score=5,
        passed=False,
        weaknesses=["Data tahun 2023 belum tercantum dari sumber terbaru."],
        counter_arguments=[],
        recency_alert="Semua sitasi di bawah tahun 2019.",
        needs_citation_mapping=True,
        critique_summary="Perlu pemetaan sitasi terbaru.",
    )
    fact_rep = VerificationReport(
        grounding_score=0.5,
        total_claims=2,
        supported_claims=1,
        unsupported_claims=1,
        hallucination_warning="1 klaim tidak memiliki bukti di vault.",
    )

    decomposer = ContractDecomposer()
    # Only vault_lokal attempted so far -> snowballing should be allowed once, and hallucination must be fixed
    adj = decomposer.adjudicate_critique(
        query=query,
        draft="Klaim angka tanpa sitasi.",
        contract=contract,
        critique_report=critique_rep,
        fact_verdict=fact_rep,
        attempted_search_channels={"vault_lokal"},
    )

    assert adj.should_revise is True
    assert adj.allow_citation_mapping is True
    assert adj.search_exhausted is False
    assert any("klaim tanpa rujukan" in r.lower() for r in adj.accepted_revisions)


def test_ai_contract_extraction_coerces_nonstandard_json_types() -> None:
    """AI contract extraction safely coerces list format_requirements, string definition_of_done, and descriptive archetype without falling back to heuristics."""
    def nonstandard_ai_fn(sys_prompt: str, user_query: str) -> dict[str, Any]:
        return {
            "archetype": "Informational Retrieval & Empirical Data Synthesis",
            "primary_goal": "Menyajikan data luas karhutla di Indonesia 2020-2024 dari sumber resmi",
            "in_scope": "Data luas terbakar tahunan; Sebaran provinsi utama",
            "out_of_scope": "Opini politik",
            "forbidden_drift_topics": ["Konflik agraria", "Kerangka teori Bab 2"],
            "format_requirements": ["Tabel komparatif tahunan", "Poin ringkasan tren"],
            "target_output_tokens": "320",
            "min_words": "100",
            "max_words": "400",
            "allow_counter_arguments": False,
            "allow_honest_gap": True,
            "definition_of_done": "1. Tabel data tersedia; 2. Sumber resmi dicantumkan",
            "required_search_channels": ["web_resmi"],
            "search_exhaustion_policy": ["Cari di SiPongi KLHK dan BPS", "Hentikan jika tidak ditemukan"],
        }

    extractor = TaskContractExtractor(llm_completion_fn=nonstandard_ai_fn)
    contract = extractor.extract("berikan aku data karhutla di indo 5 tahun terakhir")

    assert contract.archetype == "DATA_FACTUAL"
    assert contract.primary_goal == "Menyajikan data luas karhutla di Indonesia 2020-2024 dari sumber resmi"
    assert "Tabel komparatif tahunan" in contract.format_requirements
    assert len(contract.definition_of_done) == 2
    assert contract.target_output_tokens == 320
    assert "reference_mapping_snowballing_backward_forward" in contract.required_search_channels


def test_devils_advocate_preflight_plan_audit_prunes_unsubstantial_subcontracts() -> None:
    """Devil's Advocate audits proposed execution plan pre-flight, pruning unsubstantial/redundant subcontracts and challenging out-of-scope tasks."""
    from scripcy.core.critic import DevilsAdvocate
    from scripcy.core.contractor import SubContract
    from scripcy.core.contracts import TaskContract

    contract = TaskContract(
        archetype="DATA_FACTUAL",
        primary_goal="Menyajikan data kebakaran hutan di Riau",
        in_scope=["Luas kebakaran tahunan di Riau", "Titik hotspot"],
        out_of_scope=["Analisis hukum korupsi sawit", "Opini LSM internasional"],
        forbidden_drift_topics=["Bab 2 Tinjauan Pustaka", "Konflik agraria"],
    )

    subcontracts = [
        SubContract(
            id="sub-1",
            role="VaultExtractor",
            objective="Ekstrak data angka luas kebakaran hutan di Riau dari dokumen lokal.",
            dos=["Ambil angka hektare."],
            donts=["Dilarang beropini."],
        ),
        SubContract(
            id="sub-2",
            role="VaultExtractor",
            objective="Ekstrak data angka luas kebakaran hutan di Riau dari dokumen lokal.",  # Redundant!
            dos=["Ambil angka hektare."],
            donts=["Dilarang beropini."],
        ),
        SubContract(
            id="sub-3",
            role="VaultExtractor",
            objective="Analisis hukum korupsi sawit dan sengketa lahan.",  # Out-of-scope!
            dos=["Analisis perizinan."],
            donts=[],
        ),
    ]

    da = DevilsAdvocate()
    review = da.audit_execution_plan(
        query="data kebakaran hutan di Riau",
        contract=contract,
        subcontracts=subcontracts,
    )

    assert "sub-2" in review.prune_ids  # Redundant pruned
    assert "sub-3" in review.prune_ids  # Out-of-scope pruned
    assert "sub-1" not in review.prune_ids  # Substantive retained
    assert review.prune_reasons["sub-2"] == "Subkontrak redundant/tumpang tindih dengan subkontrak lain."
    assert review.prune_reasons["sub-3"] == "Subkontrak melanggar batas out-of-scope kontrak."


def test_contractor_decomposer_adjudicate_plan_audit_adds_substantive_dimensions() -> None:
    """Contractor Decomposer applies Devil's Advocate audit by pruning rejected items and injecting missing substantive dimensions."""
    from scripcy.core.contractor import ContractDecomposer, ExecutionPlan, SubContract
    from scripcy.core.critic import PlanAuditReview
    from scripcy.core.contracts import TaskContract

    contract = TaskContract(
        archetype="DATA_FACTUAL",
        primary_goal="Data restorasi gambut",
        in_scope=["Restorasi gambut"],
        out_of_scope=["Politik"],
    )

    initial_plan = ExecutionPlan(
        query="data restorasi gambut",
        contract=contract,
        subcontracts=[
            SubContract(id="sub-1", role="VaultExtractor", objective="Ekstrak data restorasi."),
            SubContract(id="sub-bad", role="VaultExtractor", objective="Ekstrak opini politik."),
        ],
    )

    audit_review = PlanAuditReview(
        prune_ids=["sub-bad"],
        prune_reasons={"sub-bad": "Melanggar batas kontrak"},
        additions=[
            {
                "role": "WebScout",
                "objective": "Ekstrak data resmi target restorasi BRGM dari portal web pemerintah.",
                "target_channel": "web_resmi",
                "dos": ["Fokus pada angka hektare kubah gambut yang telah dibasahi kembali."],
                "donts": ["Jangan mengutip blog non-resmi."],
                "search_queries": ["restorasi gambut BRGM data capaian"],
            }
        ],
        summary="Audit selesai.",
    )

    decomposer = ContractDecomposer()
    adjudicated_plan = decomposer.adjudicate_plan_audit(initial_plan, audit_review)

    sub_ids = [sc.id for sc in adjudicated_plan.subcontracts]
    assert "sub-1" in sub_ids
    assert "sub-bad" not in sub_ids
    assert any("sub-audit-add" in sid for sid in sub_ids)

    added_sc = next(sc for sc in adjudicated_plan.subcontracts if "sub-audit-add" in sc.id)
    assert added_sc.target_channel == "web_resmi"
    assert "angka hektare" in added_sc.dos[0]
    assert "blog" in added_sc.donts[0]


def test_subagent_worker_isolation_with_dos_and_donts() -> None:
    """SubagentWorker executes within an isolated context with explicit DOs and DONTs instructions."""
    from scripcy.core.subagent import SubagentWorker
    from scripcy.core.contractor import SubContract
    from scripcy.core.engine import EvidenceChunk

    recorded_prompts: dict[str, str] = {}

    def mock_llm(system_prompt: str, user_prompt: str, max_tokens: int) -> dict[str, Any]:
        recorded_prompts["system"] = system_prompt
        recorded_prompts["user"] = user_prompt
        return {
            "status": "complete",
            "extracted_facts": [
                {
                    "claim_or_metric": "Restorasi gambut mencapai 1,4 juta hektare pada 2022.",
                    "citation_stem": "brgm2022",
                    "year": "2022",
                    "is_quantitative": True,
                }
            ],
            "missing_requirements": [],
        }

    sc = SubContract(
        id="sub-isolated",
        role="VaultExtractor",
        objective="Ekstrak capaian restorasi gambut.",
        evidence_slice=[
            EvidenceChunk(
                chunk_id="ch-1",
                source="brgm2022",
                text="Pada laporan tahunan 2022, BRGM mencatat total area restorasi gambut mencapai 1,4 juta hektare.",
            )
        ],
        dos=["Ekstrak capaian luas hektare terverifikasi.", "Cantumkan tahun 2022."],
        donts=["Dilarang menulis latar belakang filosofis.", "Dilarang berasumsi."],
    )

    worker = SubagentWorker(llm_completion_fn=mock_llm)
    artifact = worker.execute(sc)

    assert artifact.status == "complete"
    assert len(artifact.extracted_facts) == 1
    assert artifact.extracted_facts[0].citation_stem == "brgm2022"

    user_p = recorded_prompts.get("user", "")
    assert "DO (WAJIB DILAKUKAN):" in user_p
    assert "Ekstrak capaian luas hektare terverifikasi." in user_p
    assert "DONT (DILARANG):" in user_p
    assert "Dilarang menulis latar belakang filosofis." in user_p


def test_devils_advocate_post_scout_evidence_audit_identifies_gaps() -> None:
    """Devil's Advocate audits worker artifacts and flags temporal or quantitative gaps."""
    from scripcy.core.subagent import ExtractedFact, SubagentArtifact

    critic = DevilsAdvocate()
    query = "carikan data karhutla di indonesia 5 tahun terakhir"
    extractor = TaskContractExtractor()
    contract = extractor.extract(query)

    # Worker only found old 2018 data
    old_art = SubagentArtifact(
        subcontract_id="sub-1",
        role="VaultExtractor",
        status="complete",
        extracted_facts=[
            ExtractedFact(
                claim_or_metric="Luas kebakaran hutan tahun 2018 adalah 510.000 ha.",
                citation_stem="klhk2018",
                year="2018",
                is_quantitative=True,
            )
        ],
        worker_notes="Hanya menemukan data arsip 2018.",
    )

    review = critic.audit_worker_evidence(
        query=query,
        contract=contract,
        subcontracts=[],
        artifacts=[old_art],
    )

    assert review.all_satisfied is False
    assert any("5 tahun terakhir" in g or "terbaru" in g for g in review.detected_gaps)
    assert len(review.suggested_expansion_queries) >= 1


def test_contractor_adjudicates_evidence_audit_expansion_vs_honest_gap() -> None:
    """Contractor permits 1 follow-up search expansion, then caps at honest gap upon exhaustion."""
    from scripcy.core.critic import EvidenceAuditReview

    decomposer = ContractDecomposer()
    query = "carikan data karhutla di indonesia 5 tahun terakhir"
    extractor = TaskContractExtractor()
    contract = extractor.extract(query)

    review = EvidenceAuditReview(
        all_satisfied=False,
        detected_gaps=["Data statistik 2022-2023 belum ditemukan."],
        suggested_expansion_queries=["karhutla indonesia 2022 2023 data resmi statistik"],
    )

    # Pass 0: Within max_expansions=1 -> Decision is expand
    adj_pass0 = decomposer.adjudicate_evidence_audit(
        query=query,
        contract=contract,
        audit_review=review,
        expansion_count=0,
        max_expansions=1,
        previous_worker_notes=["Portal A nihil."],
    )
    assert adj_pass0.decision == "expand"
    assert len(adj_pass0.follow_up_subcontracts) == 2
    assert adj_pass0.follow_up_subcontracts[0].role in ("WebScout", "LitScout")
    assert "Catatan worker sebelumnya: Portal A nihil." in adj_pass0.follow_up_subcontracts[0].objective

    # Pass 1: Limit reached -> Decision is honest_gap
    adj_pass1 = decomposer.adjudicate_evidence_audit(
        query=query,
        contract=contract,
        audit_review=review,
        expansion_count=1,
        max_expansions=1,
    )
    assert adj_pass1.decision == "honest_gap"
    assert len(adj_pass1.gap_notes) >= 1
    assert "Honest Data Gap" in adj_pass1.adjudication_summary


def test_configurable_max_worker_expansions_setting() -> None:
    """ScripcyConfig supports configurable max_worker_expansions with a default of 1."""
    from scripcy.core.config import ScripcyConfig

    cfg_default = ScripcyConfig()
    assert cfg_default.max_worker_expansions == 1

    cfg_custom = ScripcyConfig(max_worker_expansions=3)
    assert cfg_custom.max_worker_expansions == 3




