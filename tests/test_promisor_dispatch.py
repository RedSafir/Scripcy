"""
Unit test to verify that Promisor accurately dispatches query categories:
- Slash commands are handled deterministically as system tools.
- Non-slash natural language queries are screened for spam/greetings and forwarded to AI (Contractor/Manager/Examiner).
"""

from scripcy.core.promisor import Promisor


def test_promisor_5_entry_point_dispatch():
    promisor = Promisor()

    # 1. Direct Fast Reply (Casual / Chit-chat / Greetings - 0 Token)
    d_casual_1 = promisor.evaluate("halo")
    assert d_casual_1.entry_point == "direct"
    assert d_casual_1.direct_reply is not None
    assert not d_casual_1.is_system_tool

    d_casual_2 = promisor.evaluate("terima kasih banyak bro")
    assert d_casual_2.entry_point == "direct"
    assert "Sama-sama" in str(d_casual_2.direct_reply)

    d_casual_3 = promisor.evaluate("oke siap noted")
    assert d_casual_3.entry_point == "direct"

    # 2. Spam / Pure gibberish rejection
    d_spam = promisor.evaluate("??????  .... !!!")
    assert d_spam.action == "spam.reject"
    assert d_spam.entry_point == "direct"
    assert not d_spam.is_system_tool

    # 3. Deterministic Slash Commands (/vault, /cancel, /back)
    d_slash = promisor.evaluate("/vault errors")
    assert d_slash.is_system_tool is True
    assert d_slash.action == "slash.vault"

    # 4. Non-slash requests forward to AI (Contractor)
    d_worker = promisor.evaluate("tampilkan file yang error di vault")
    assert d_worker.entry_point == "contractor"
    assert not d_worker.is_system_tool
    assert d_worker.action == "ai_chat"

    # 5. Manager (Revision / Patching Draft / Session Memory)
    d_manager = promisor.evaluate("perbaiki respon sebelumnya, tambahkan perbandingan memory")
    assert d_manager.entry_point == "manager"
    assert not d_manager.is_system_tool
    assert d_manager.action == "ai_chat"

    # 6. Examiner (Hypothesis Testing / Claim Verification)
    d_examiner = promisor.evaluate("apakah benar yolov8 selalu lebih lambat dari cnn konvensional?")
    assert d_examiner.entry_point == "examiner"
    assert not d_examiner.is_system_tool
    assert d_examiner.action == "ai_chat"

    # 7. Contractor (Deep Research / Gap / Broad Synthesis)
    d_contractor = promisor.evaluate("carikan aku gap penelitian deep learning terbaru")
    assert d_contractor.entry_point == "contractor"
    assert not d_contractor.is_system_tool
    assert d_contractor.action == "ai_chat"
