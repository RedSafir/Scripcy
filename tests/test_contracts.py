"""Unit tests for Dynamic Task Contract, DoD Extraction, and Adaptive Critic."""

from scripcy.core.contracts import TaskContract, TaskContractExtractor, RubricCriterion
from scripcy.core.critic import DevilsAdvocate


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


def test_task_contract_tolerance_margin_properties() -> None:
    contract = TaskContract(
        primary_goal="Analisis komparatif",
        min_words=300,
        max_words=500,
        tolerance_pct=0.15,
    )
    # 300 * 0.85 = 255
    assert contract.effective_min_words == 255
    # 500 * 1.15 = 575
    assert contract.effective_max_words == 575

    # 285 words is below 300 but within 15% tolerance
    assert contract.is_word_count_acceptable(285) is True
    # 520 words is above 500 but within 15% tolerance
    assert contract.is_word_count_acceptable(520) is True

    # 240 words is strictly below tolerance
    assert contract.is_word_count_acceptable(240) is False
    # 600 words is strictly above tolerance
    assert contract.is_word_count_acceptable(600) is False


def test_ambiguous_intent_detection_and_heuristic_synthesis() -> None:
    extractor = TaskContractExtractor()
    query = "Tolong bandingkan metode X dan Y untuk bab 2 skripsi saya"
    assert extractor.is_ambiguous_intent(query) is True

    # Heuristic synthesis should harmonize comparison within literature review
    contract = extractor.extract(query)
    assert "komparatif" in contract.primary_goal.lower() or "perbandingan" in contract.primary_goal.lower()
    assert "bab 2" in contract.primary_goal.lower() or "tinjauan" in contract.primary_goal.lower()
    assert any("parameter" in d.lower() for d in contract.definition_of_done)
    assert contract.min_words >= 400


def test_ambiguous_intent_llm_disambiguation() -> None:
    from unittest.mock import MagicMock, patch

    mock_llm_json = """{
        "primary_goal": "Harmonisasi komparasi metode X dan Y pada tinjauan pustaka Bab 2",
        "in_scope": ["Parameter evaluasi metode X vs Y", "Landasan teoritis untuk skripsi"],
        "out_of_scope": ["Sejarah panjang", "Opini spekulatif"],
        "format_requirements": "Matriks perbandingan akademis dan sintesis",
        "min_words": 380,
        "max_words": 750,
        "definition_of_done": ["Parameter komparasi jelas", "Sitasi empiris valid"],
        "is_modular_section_needed": false
    }"""

    mock_engine = MagicMock()
    mock_engine.config.default_model = "test-model"
    mock_engine.config.get_api_key_for_model.return_value = "real-api-key"
    mock_engine._acompletion_fn = None

    extractor = TaskContractExtractor(engine=mock_engine)
    query = "Tolong bandingkan metode X dan Y untuk bab 2 skripsi saya"

    with patch("litellm.completion") as mock_completion:
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content=mock_llm_json))]
        mock_completion.return_value = mock_resp

        contract = extractor.extract(query)
        assert "Harmonisasi komparasi" in contract.primary_goal
        assert contract.min_words == 380
        assert contract.max_words == 750
        assert len(contract.definition_of_done) == 2
