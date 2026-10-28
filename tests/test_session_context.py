"""
Unit tests for SessionContextManager and the new Examiner & Manager roles.
"""

from scripcy.core.session_context import SessionContextManager
from scripcy.core.examiner import Examiner
from scripcy.core.manager import SubagentManager


def test_session_context_manager_strip_bibliography():
    scm = SessionContextManager(session_id="test_sess")
    raw = (
        "Analisis menunjukkan Vision Transformer unggul dalam ekstraksi dokumen.\n\n"
        "📚 Daftar Pustaka:\n"
        "1. Smith (2025). Paper Title. DOI:123\n"
        "2. Doe (2026). Another Paper. DOI:456\n"
    )
    clean = scm.strip_bibliography(raw)
    assert "Daftar Pustaka" not in clean
    assert "Vision Transformer unggul" in clean


def test_session_context_manager_turn_and_revision():
    scm = SessionContextManager(session_id="test_sess")
    assert not scm.is_revision_query("apa kelemahan YOLOS?")
    assert scm.is_revision_query("perbaiki respon sebelumnya dong")
    assert scm.is_revision_query("revisi tabel tadi, tambahkan throughput")

    # Record Turn 1
    resp_1 = "Model YOLOS memiliki arsitektur self-attention murni.\n\n📚 Daftar Pustaka:\n1. Wang et al."
    scm.record_turn(
        query="apa itu YOLOS?",
        full_response=resp_1,
        small_contracts=["Karakteristik arsitektur YOLOS"],
        verified_facts=["YOLOS mengeliminasi inductive bias CNN"],
    )

    assert len(scm.turns) == 1
    assert scm.get_last_clean_draft() == "Model YOLOS memiliki arsitektur self-attention murni."
    assert scm.get_last_verified_facts() == ["YOLOS mengeliminasi inductive bias CNN"]

    # Normal Follow-up Context
    ctx_normal = scm.build_context_directive("apakah YOLOS boros memori?")
    assert not ctx_normal["is_revision"]
    assert ctx_normal["has_previous_context"]
    assert "Karakteristik arsitektur YOLOS" in ctx_normal["directive_text"]

    # Revision Context
    ctx_rev = scm.build_context_directive("perbaiki respon sebelumnya, tambahkan perbandingan memory")
    assert ctx_rev["is_revision"]
    assert ctx_rev["baseline_draft"] == "Model YOLOS memiliki arsitektur self-attention murni."
    assert "YOLOS mengeliminasi inductive bias CNN" in ctx_rev["verified_facts"]


def test_role_imports():
    assert Examiner is not None
    assert SubagentManager is not None
