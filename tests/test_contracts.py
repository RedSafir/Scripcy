"""Unit tests for Dynamic Task Contract, DoD Extraction, and Adaptive Critic."""

from scrpcy.core.contracts import TaskContract, TaskContractExtractor, RubricCriterion
from scrpcy.core.critic import DevilsAdvocate


def test_task_contract_argument_extraction() -> None:
    extractor = TaskContractExtractor()
    contract = extractor.extract("carikan aku argumen untuk penelitianku tentang karhutla di indonesia")
    assert "karhutla" in contract.primary_goal.lower() or "argumen" in contract.primary_goal.lower()
    assert len(contract.in_scope) >= 2
    assert "Penulisan bab komprehensif monolitik" in contract.out_of_scope
    assert contract.max_words <= 650
    assert any("tesis" in d.lower() for d in contract.definition_of_done)
    assert any("pendukung" in d.lower() or "pro" in d.lower() for d in contract.definition_of_done)


def test_task_contract_comparison_extraction() -> None:
    extractor = TaskContractExtractor()
    contract = extractor.extract("bandingkan metodologi remote sensing Sentinel-2 vs MODIS untuk deteksi hotspot")
    assert "komparasi" in contract.primary_goal.lower() or "perbandingan" in contract.primary_goal.lower()
    assert "tabel" in contract.format_requirements.lower() or "komparatif" in contract.format_requirements.lower()
    assert any("parameter" in d.lower() for d in contract.definition_of_done)


def test_task_contract_fact_verification_extraction() -> None:
    extractor = TaskContractExtractor()
    contract = extractor.extract("berapa angka total luas lahan terbakar di Kalimantan Tengah tahun 2023?")
    assert "verifikasi" in contract.primary_goal.lower() or "fakta" in contract.primary_goal.lower()
    assert contract.max_words <= 400
    assert any("langsung" in d.lower() for d in contract.definition_of_done)


def test_task_contract_literature_review_extraction() -> None:
    extractor = TaskContractExtractor()
    contract = extractor.extract("buatkan draf lengkap tinjauan pustaka Bab 2 tentang restorasi hidrologi gambut")
    assert "tinjauan pustaka" in contract.primary_goal.lower() or "sintesis" in contract.primary_goal.lower()
    assert contract.is_modular_section_needed is True
    assert contract.max_words >= 1000


def test_adaptive_critic_honors_concise_contract() -> None:
    critic = DevilsAdvocate()
    extractor = TaskContractExtractor()
    contract = extractor.extract("carikan argumen singkat pro kontra tentang sekat kanal gambut")

    # A concise 2-paragraph dialectical answer
    concise_answer = (
        "Secara garis besar, implementasi sekat kanal (canal blocking) memicu perdebatan dialektika dalam pemulihan hidrologi gambut.\n\n"
        "Argumen afirmatif (pro) menegaskan bahwa sekat kanal berhasil menaikkan muka air tanah hingga batas aman -40 cm, "
        "yang terbukti secara signifikan menurunkan frekuensi titik api dan laju oksidasi karbon [Suryadiputra2018]. "
        "Namun demikian, kritik tandingan (kontra) menggarisbawahi bahwa tanpa pemeliharaan berkala, bendungan kayu rentan jebol "
        "saat debit puncak dan dapat mengganggu akses transportasi logistik perahu masyarakat lokal [Page2020]."
    )

    # Heuristic critique with contract
    report = critic.critique_heuristic(
        query="carikan argumen singkat pro kontra tentang sekat kanal gambut",
        answer=concise_answer,
        contract=contract,
    )

    # The critic must recognize the nuance, not penalize for conciseness, and pass
    assert report.score >= 7
    assert report.passed is True
    assert report.needs_citation_mapping is False
