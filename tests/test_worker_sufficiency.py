"""Unit tests for Worker Self-Sufficiency Audit & Context Completeness Verification."""

from __future__ import annotations

import pytest

from scripcy.core.contractor import SubContract
from scripcy.core.engine import EvidenceChunk
from scripcy.core.subagent import ExtractedFact, InformationSlot, SubagentWorker


class TestWorkerSufficiency:
    """Test suite verifying slot decomposition, sufficiency evaluation, and local self-healing."""

    def test_decompose_slots_comparative_and_quantitative(self) -> None:
        sc = SubContract(
            id="SC-01",
            role="VaultExtractor",
            objective="Bandingkan throughput dan latensi antara YOLOv8 dan YOLOS pada perangkat edge",
            dos=["Ambil data angka latensi ms dan FPS."],
            allowed_scope=["YOLOv8", "YOLOS", "latensi", "throughput"],
        )
        slots = SubagentWorker.decompose_slots(sc)

        assert "YOLOv8" in slots.target_entities
        assert "YOLOS" in slots.target_entities
        assert slots.requires_quantitative is True

    def test_evaluate_sufficiency_complete_when_all_slots_met(self) -> None:
        slots = InformationSlot(
            target_entities=["YOLOv8", "YOLOS"],
            requires_quantitative=True,
        )
        facts = [
            ExtractedFact(
                claim_or_metric="Model YOLOv8 mencapai latensi 12 ms dengan 120 fps.",
                citation_stem="yolo8_paper",
                is_quantitative=True,
            ),
            ExtractedFact(
                claim_or_metric="Arsitektur YOLOS mencatat latensi 45 ms pada resolusi 640x640.",
                citation_stem="yolos_paper",
                is_quantitative=True,
            ),
        ]
        status, missing, coverage, found_ents = SubagentWorker.evaluate_sufficiency(facts, slots)

        assert status == "complete"
        assert len(missing) == 0
        assert coverage >= 0.99
        assert "YOLOv8" in found_ents
        assert "YOLOS" in found_ents

    def test_evaluate_sufficiency_partial_when_entity_missing(self) -> None:
        slots = InformationSlot(
            target_entities=["YOLOv8", "YOLOS"],
            requires_quantitative=True,
        )
        # Only YOLOv8 is present
        facts = [
            ExtractedFact(
                claim_or_metric="Model YOLOv8 mencatat throughput 120 fps dan latensi 12 ms.",
                citation_stem="yolo8_paper",
                is_quantitative=True,
            ),
        ]
        status, missing, coverage, found_ents = SubagentWorker.evaluate_sufficiency(facts, slots)

        assert status == "partial"
        assert any("YOLOS" in m for m in missing)
        assert coverage == 0.50
        assert "YOLOv8" in found_ents
        assert "YOLOS" not in found_ents

    def test_evaluate_sufficiency_partial_when_quantitative_missing(self) -> None:
        slots = InformationSlot(
            target_entities=["BERT"],
            requires_quantitative=True,
        )
        # Only qualitative narrative without numbers
        facts = [
            ExtractedFact(
                claim_or_metric="Arsitektur BERT menggunakan mekanisme self-attention bidirectional.",
                citation_stem="bert_paper",
                is_quantitative=False,
            ),
        ]
        status, missing, coverage, found_ents = SubagentWorker.evaluate_sufficiency(facts, slots)

        assert status == "partial"
        assert any("angka/kuantitatif" in m for m in missing)
        assert coverage == 0.50

    def test_worker_local_self_healing_recovers_missing_entity(self) -> None:
        """Worker initially extracts top sentence about YOLOv8, but Pass 2 self-healing finds YOLOS chunk."""
        sc = SubContract(
            id="SC-02",
            role="VaultExtractor",
            objective="Ekstrak perbandingan latensi YOLOv8 vs YOLOS",
            allowed_scope=["YOLOv8", "YOLOS", "latensi"],
            evidence_slice=[
                EvidenceChunk(
                    chunk_id="chk_1",
                    text="YOLOv8 adalah detektor objek real-time mutakhir dengan latensi 12 ms pada GPU modern. "
                         "Struktur backbone YOLOv8 dioptimalkan untuk efisiensi inferensi tinggi.",
                    source="doc_yolo",
                ),
                EvidenceChunk(
                    chunk_id="chk_2",
                    text="Sementara itu, model YOLOS berbasis Vision Transformer mencatat latensi 48 ms pada benchmark yang sama.",
                    source="doc_yolos",
                ),
            ],
        )
        worker = SubagentWorker()
        artifact = worker.extract_heuristic(sc)

        assert artifact.status == "complete"
        assert artifact.coverage_score >= 0.99
        # Both entities must be present in extracted facts due to self-healing pass
        all_claims = " ".join(f.claim_or_metric for f in artifact.extracted_facts)
        assert "YOLOv8" in all_claims
        assert "YOLOS" in all_claims
        assert len(artifact.missing_requirements) == 0

    def test_worker_reports_transparent_partial_when_entity_absent_from_slice(self) -> None:
        """When evidence slice truly lacks YOLOS, worker truthfully reports status='partial'."""
        sc = SubContract(
            id="SC-03",
            role="VaultExtractor",
            objective="Ekstrak perbandingan latensi YOLOv8 vs YOLOS",
            allowed_scope=["YOLOv8", "YOLOS", "latensi"],
            evidence_slice=[
                EvidenceChunk(
                    chunk_id="chk_yolo_only",
                    text="YOLOv8 adalah detektor objek real-time mutakhir dengan latensi 12 ms pada GPU modern.",
                    source="doc_yolo",
                ),
            ],
        )
        worker = SubagentWorker()
        artifact = worker.extract_heuristic(sc)

        assert artifact.status == "partial"
        assert artifact.coverage_score == 0.50
        assert any("YOLOS" in m for m in artifact.missing_requirements)
        assert len(artifact.missing_slots) > 0
