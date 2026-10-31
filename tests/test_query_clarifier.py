"""Unit tests for Pre-Promisor Query Clarification Gate (Hybrid Cascading)."""

from __future__ import annotations

import pytest

from scripcy.core.clarifier import ClarificationVerdict, QueryClarifier
from scripcy.core.config import ConfigManager, ScripcyConfig


def test_clarifier_named_entity_bypasses_micro_probe() -> None:
    clarifier = QueryClarifier()
    # Contains named model 'YOLOS' and action 'carikan' + topics -> should bypass immediately
    verdict = clarifier.clarify("carikan aku penelitian tentang perkembangan model YOLOS")
    assert not verdict.is_ambiguous


def test_clarifier_ignores_casual_and_commands() -> None:
    clarifier = QueryClarifier()
    assert not clarifier.clarify("halo").is_ambiguous
    assert not clarifier.clarify("terima kasih").is_ambiguous
    assert not clarifier.clarify("/vault").is_ambiguous
    assert not clarifier.clarify("/back").is_ambiguous


def test_clarifier_heuristic_raw_broad_topic() -> None:
    clarifier = QueryClarifier()
    v1 = clarifier.evaluate_heuristic("yolo")
    assert v1.is_ambiguous
    assert v1.reason == "raw_broad_topic"
    assert v1.suggested_action == "menu_select"
    assert v1.options is not None
    assert len(v1.options) >= 3

    v2 = clarifier.evaluate_heuristic("deep learning")
    assert v2.is_ambiguous
    assert v2.reason == "raw_broad_topic"


def test_clarifier_heuristic_missing_topic() -> None:
    clarifier = QueryClarifier()
    v1 = clarifier.evaluate_heuristic("carikan penelitian terbaru")
    assert v1.is_ambiguous
    assert v1.reason == "missing_topic"
    assert v1.suggested_action == "input_prompt"

    v2 = clarifier.evaluate_heuristic("buatkan analisis komprehensif")
    assert v2.is_ambiguous
    assert v2.reason == "missing_topic"


def test_clarifier_heuristic_dangling_reference() -> None:
    clarifier = QueryClarifier()
    # No vault docs, no history -> ambiguous
    v_empty = clarifier.evaluate_heuristic(
        "bagaimana dengan metodenya?",
        active_vault_count=0,
        history_len=0,
    )
    assert v_empty.is_ambiguous
    assert v_empty.reason == "dangling_reference"
    assert v_empty.suggested_action == "menu_select"

    # With vault docs present -> not ambiguous (has context to resolve)
    v_with_docs = clarifier.evaluate_heuristic(
        "bagaimana dengan metodenya?",
        active_vault_count=2,
        history_len=0,
    )
    assert not v_with_docs.is_ambiguous


def test_clarifier_heuristic_detailed_query_passes() -> None:
    clarifier = QueryClarifier()
    q = "analisis perbandingan performa mAP arsitektur YOLOS terhadap YOLOv8 pada dataset COCO 2017"
    v = clarifier.evaluate_heuristic(q)
    assert not v.is_ambiguous


def test_clarifier_micro_probe_ai_ambiguous() -> None:
    def mock_probe(q: str) -> dict:
        return {
            "needs_clarification": True,
            "question": "Fokus aplikasi medis mana yang ingin Anda teliti?",
            "options": [
                "Analisis Citra Medis (Radiologi/MRI)",
                "Penemuan Obat (Drug Discovery)",
                "Sistem Prediksi Diagnosis Klinis",
            ],
        }

    clarifier = QueryClarifier(llm_probe_fn=mock_probe)
    # Query with enough words to bypass raw 1-word heuristic, but semantically broad
    verdict = clarifier.clarify("pengaruh artificial intelligence di sektor pelayanan medis rumah sakit")
    assert verdict.is_ambiguous
    assert verdict.reason == "micro_probe_broad"
    assert verdict.suggested_action == "menu_select"
    assert "Fokus aplikasi medis" in str(verdict.question)
    assert len(verdict.options) == 4  # 3 AI options + 1 reinput option


def test_clarifier_micro_probe_ai_specific() -> None:
    def mock_probe(q: str) -> dict:
        return {"needs_clarification": False}

    clarifier = QueryClarifier(llm_probe_fn=mock_probe)
    verdict = clarifier.clarify("pengaruh artificial intelligence di sektor pelayanan medis rumah sakit")
    assert not verdict.is_ambiguous


def test_clarifier_micro_probe_resilience_fallback() -> None:
    def failing_probe(q: str) -> dict:
        raise TimeoutError("Micro probe timeout after 2.5s")

    clarifier = QueryClarifier(llm_probe_fn=failing_probe)
    # Query that fails probe should gracefully fall back to is_ambiguous = False
    verdict = clarifier.clarify("analisis penerapan machine learning pada credit scoring")
    assert not verdict.is_ambiguous


def test_config_enable_query_clarification_toggle(tmp_path) -> None:
    cfg_file = tmp_path / "config.json"
    mgr = ConfigManager(config_path=cfg_file)
    assert mgr.config.enable_query_clarification is True

    mgr.update(enable_query_clarification=False)
    assert mgr.config.enable_query_clarification is False


def test_agent_loop_clarify_non_interactive_fallback(monkeypatch) -> None:
    """Verify that in non-interactive environment (CI/tests), queries are passed through without blocking."""
    from unittest.mock import MagicMock
    from scripcy.studio.agent_loop import StudioAgentLoopMixin

    loop = StudioAgentLoopMixin()
    loop.config = ScripcyConfig()
    loop.engine = None
    loop.clarifier = QueryClarifier()
    loop.active_documents = set()
    loop.conversation_history = []

    # sys.stdin.isatty is False in tests
    import sys
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    # Even for broad query 'yolo', it should return original string without blocking
    result = loop._clarify_user_query_if_needed("yolo")
    assert result == "yolo"


def test_agent_loop_clarify_interactive_menu_selection(monkeypatch) -> None:
    """Verify that when user picks an option in the interactive menu, query is enriched."""
    from unittest.mock import MagicMock
    from scripcy.studio.agent_loop import StudioAgentLoopMixin

    loop = StudioAgentLoopMixin()
    loop.config = ScripcyConfig()
    loop.engine = None
    loop.console = MagicMock()
    loop.clarifier = QueryClarifier()
    loop.active_documents = set()
    loop.conversation_history = []

    import sys
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    # Mock user selecting "synth"
    monkeypatch.setattr("scripcy.studio.agent_loop.interactive_menu", lambda opts, **kw: "synth")

    result = loop._clarify_user_query_if_needed("yolo")
    assert result is not None
    assert "yolo" in result
    assert "Fokus spesifik:" in result
    assert "Tinjauan Pustaka" in result


def test_agent_loop_clarify_interactive_cancel(monkeypatch) -> None:
    """Verify that when user cancels menu, None is returned to cleanly abort."""
    from unittest.mock import MagicMock
    from scripcy.studio.agent_loop import StudioAgentLoopMixin

    loop = StudioAgentLoopMixin()
    loop.config = ScripcyConfig()
    loop.engine = None
    loop.console = MagicMock()
    loop.clarifier = QueryClarifier()
    loop.active_documents = set()
    loop.conversation_history = []

    import sys
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    # Mock user pressing esc / back
    monkeypatch.setattr("scripcy.studio.agent_loop.interactive_menu", lambda opts, **kw: "back")

    result = loop._clarify_user_query_if_needed("yolo")
    assert result is None

