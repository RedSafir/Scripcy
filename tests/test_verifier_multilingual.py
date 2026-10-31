"""Unit tests for Verifier FastEmbed cross-lingual semantic grounding and dynamic claim budget."""

import pytest
from scripcy.core.engine import EvidenceChunk
from scripcy.core.verifier import FactChecker


def test_verifier_dynamic_claim_budget():
    checker = FactChecker()
    # 700 words synthesis text
    long_text = " ".join(["Sintesis akademis tentang restorasi hidrologis lahan gambut tropis terdegradasi."] * 70)
    claims = checker.extract_atomic_claims(long_text)
    # Long text should exceed the old 15 cap:
    assert len(claims) >= 15


def test_verifier_cross_lingual_grounding_indonesian_to_english():
    checker = FactChecker()

    # English evidence in vault (e.g. from an international PDF paper)
    evidence = [
        EvidenceChunk(
            chunk_id="c1",
            text="Peatland rewetting and hydrological canal blocking substantially decrease greenhouse gas emissions and CO2 flux from degraded peat soils.",
            source="Page_2020",
            score=0.95,
        ),
        EvidenceChunk(
            chunk_id="c2",
            text="Object detection architectures using deep convolutional neural networks demonstrate high mean average precision on standard benchmark datasets.",
            source="Redmon_2016",
            score=0.90,
        ),
    ]

    # Indonesian claims with ZERO lexical word overlap against the English text
    claims = [
        "Restorasi hidrologis melalui penutupan kanal mampu menurunkan emisi gas rumah kaca pada lahan gambut terdegradasi.",
        "Arsitektur deep learning konvolusional memiliki akurasi deteksi objek yang tinggi pada dataset tolok ukur standar.",
    ]

    verifications = checker.verify_claims_lexical(claims, evidence)
    assert len(verifications) == 2

    # Both claims should be recognized as supported or weak via semantic FastEmbed cross-lingual embeddings, NOT unsupported!
    for v in verifications:
        assert v.status in ("supported", "weak"), f"Claim '{v.claim}' was marked as {v.status} with reason: {v.reason}"
        assert v.supporting_source in ("Page_2020", "Redmon_2016")


def test_verifier_rejects_unrelated_hallucination():
    checker = FactChecker()
    evidence = [
        EvidenceChunk(
            chunk_id="c3",
            text="Photosynthesis in C3 plants involves the enzyme RuBisCO fixing carbon dioxide during the Calvin cycle.",
            source="Biology_2021",
            score=0.90,
        )
    ]
    # Unrelated claim
    claims = ["Pesawat supersonik modern menggunakan mesin scramjet untuk mencapai kecepatan hipersonik di stratosfer."]
    verifications = checker.verify_claims_lexical(claims, evidence)
    assert len(verifications) == 1
    assert verifications[0].status == "unsupported"
