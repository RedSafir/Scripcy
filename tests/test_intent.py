"""Unit tests for IntentRouter, ResearchIntent, and query disambiguation."""

from scrpcy.core.intent import IntentRouter, ResearchIntent


def test_intent_router_argument_exploration() -> None:
    router = IntentRouter()
    res = router.analyze("carikan aku argumen untuk penelitianku tentang karhutla di indonesia")
    assert res.intent == ResearchIntent.ARGUMENT_EXPLORATION
    assert res.target_skill_name == "argument_matrix"
    assert res.needs_clarification is False
    assert len(res.sub_queries) >= 2


def test_intent_router_pro_contra() -> None:
    router = IntentRouter()
    res = router.analyze("apa pro kontra implementasi canal blocking di lahan gambut?")
    assert res.intent == ResearchIntent.ARGUMENT_EXPLORATION
    assert res.target_skill_name == "argument_matrix"
    assert res.needs_clarification is False


def test_intent_router_literature_review() -> None:
    router = IntentRouter()
    res = router.analyze("buatkan draf tinjauan pustaka Bab 2 tentang restorasi gambut")
    assert res.intent == ResearchIntent.LITERATURE_REVIEW
    assert res.needs_clarification is False
    assert res.target_output_format == "outline_review"


def test_intent_router_fact_verification() -> None:
    router = IntentRouter()
    res = router.analyze("berapa angka luas karhutla di Riau tahun 2019?")
    assert res.intent == ResearchIntent.FACT_VERIFICATION
    assert res.needs_clarification is False
    assert res.target_output_format == "atomic_fact"


def test_intent_router_ambiguous_short_query() -> None:
    router = IntentRouter()
    res = router.analyze("karhutla")
    assert res.ambiguity_score >= 0.70
    assert res.needs_clarification is True
    assert len(res.clarification_options) == 3


def test_intent_router_long_input_draft() -> None:
    router = IntentRouter()
    long_proposal = (
        "Latar belakang penelitian ini berfokus pada efektivitas kebijakan restorasi gambut 3R "
        "(Rewetting, Revegetation, Revitalization) di Provinsi Riau. Kebakaran hutan dan lahan "
        "berulang kali terjadi akibat pengeringan kanal oleh perkebunan kelapa sawit monokultur. "
        "Data satelit MODIS dan VIIRS menunjukkan lonjakan titik api pada musim kemarau El Nino. "
        "Penelitian ini bertujuan untuk menguji dampak sosial-ekonomi terhadap masyarakat lokal "
        "serta mengukur penurunan emisi gas rumah kaca dari sekat kanal yang dibangun BRGM. "
        "Metode yang digunakan adalah mixed-methods dengan wawancara mendalam kepada 50 petani "
        "dan pengukuran langsung water table level menggunakan piezometer selama 6 bulan. "
        "Tolong carikan argumen dan kritik metodologis terhadap desain penelitian ini."
    )
    res = router.analyze(long_proposal)
    assert res.input_is_long is True
    assert res.ambiguity_score <= 0.20
    assert res.needs_clarification is False
    assert res.intent == ResearchIntent.ARGUMENT_EXPLORATION
