"""Unit tests for SmartCrusher (JSON compression) and MarkdownMinifier."""

from __future__ import annotations

import json

from scripcy.core.crusher import MarkdownMinifier, SmartCrusher
from scripcy.core.engine import EvidenceChunk, EvidenceSynthesizer
from scripcy.core.orchestrator import SubagentOrchestrator
from scripcy.core.subagent import ExtractedFact
from scripcy.core.contracts import TaskContract


def test_markdown_minifier_strips_html_images_and_redundant_newlines() -> None:
    raw = (
        "# Title\n\n\n\n"
        "<!-- Comment to strip -->\n"
        "<div class=\"banner\"><p>Some <b>bold</b> text in html</p></div>\n\n"
        "![Alt Graph](https://example.com/image.png)\n\n"
        "[Official Portal](https://sipongi.menlhk.go.id/data?utm_source=tracker&session=123)\n\n\n"
        "------------------\n\n"
        "Paragraph with trailing spaces.   \n"
    )
    minified = MarkdownMinifier.minify(raw)

    assert "<!--" not in minified
    assert "<div" not in minified
    assert "<p>" not in minified
    assert "<b>" not in minified
    assert "Some bold text in html" in minified
    assert "[Gambar: Alt Graph]" in minified
    assert "utm_source" not in minified
    assert "[Official Portal](https://sipongi.menlhk.go.id/data)" in minified
    assert "------------------" not in minified
    assert "\n\n\n" not in minified


def test_markdown_minifier_preserves_tables_and_code_blocks() -> None:
    raw = (
        "Here is a table:\n\n"
        "| Tahun | Luas Karhutla (ha) | Sumber |\n"
        "| :--- | :--- | :--- |\n"
        "| 2020  | 296.942            | KLHK   |\n"
        "| 2023  | 1.161.193          | SiPongi|\n\n"
        "And a code snippet:\n"
        "```python\n"
        "def hello():\n"
        "    print('world')\n"
        "```\n"
    )
    minified = MarkdownMinifier.minify(raw)

    assert "| Tahun | Luas Karhutla (ha) | Sumber |" in minified
    assert "| 2023  | 1.161.193          | SiPongi|" in minified
    assert "```python\ndef hello():\n    print('world')\n```" in minified


def test_smart_crusher_compact_tsv_and_null_pruning() -> None:
    data = [
        {"id": 1, "name": "Forest Fire 2023", "metric": "1.1M ha", "extra": None},
        {"id": 2, "name": "Riau Peatland", "metric": "450k ha", "extra": ""},
    ]
    tsv = SmartCrusher.to_compact_tsv(data, columns=["id", "name", "metric"])
    lines = tsv.splitlines()

    assert lines[0] == "id\tname\tmetric"
    assert lines[1] == "1\tForest Fire 2023\t1.1M ha"
    assert lines[2] == "2\tRiau Peatland\t450k ha"

    nested = {
        "title": "Karhutla",
        "empty_str": "",
        "none_val": None,
        "empty_list": [],
        "nested_dict": {"valid": 123, "empty": None},
    }
    pruned = SmartCrusher.prune_nulls(nested)
    assert pruned == {"title": "Karhutla", "nested_dict": {"valid": 123}}

    crushed = SmartCrusher.crush_json(nested)
    assert "\"title\": \"Karhutla\"" in crushed
    assert "empty_str" not in crushed
    assert "none_val" not in crushed


def test_smart_crusher_verified_facts_ledger() -> None:
    facts = [
        ExtractedFact(
            claim_or_metric="Luas karhutla nasional tahun 2023 adalah 1.161.193 ha.",
            citation_stem="sipongi2023",
            year="2023",
            is_quantitative=True,
        ),
        ExtractedFact(
            claim_or_metric="Sebagian besar kebakaran terjadi di lahan mineral (82%).",
            citation_stem="gaveau2021",
            year="2021",
            is_quantitative=True,
        ),
    ]
    tsv_ledger = SmartCrusher.crush_verified_facts_ledger(facts)

    assert "Fakta_Klaim\tSumber\tTahun" in tsv_ledger
    assert "1.161.193 ha.\tsipongi2023\t2023" in tsv_ledger
    assert "lahan mineral (82%).\tgaveau2021\t2021" in tsv_ledger


def test_evidence_synthesizer_uses_markdown_minifier() -> None:
    synth = EvidenceSynthesizer()
    chunk = EvidenceChunk(
        chunk_id="test_c1",
        source="paper2024",
        text="Paragraph 1 with redundant spaces   \n\n\n\n<!-- hidden comment -->\n\nText continuing.",
        token_estimate=15,
        score=2.5,
    )
    prompt = synth.format_context_prompt([chunk])

    assert "<!-- hidden comment -->" not in prompt
    assert "\n\n\n\n" not in prompt
    assert "Paragraph 1 with redundant spaces\n\nText continuing." in prompt


def test_orchestrator_builds_tsv_assembly_directive() -> None:
    from scripcy.core.orchestrator import SubagentOrchestrator
    from unittest.mock import MagicMock

    contract = TaskContract(
        archetype="DATA_FACTUAL",
        primary_goal="Menyajikan data karhutla",
        target_output_tokens=300,
    )
    facts = [
        ExtractedFact(
            claim_or_metric="Data tahun 2023 tercatat 1.16M ha",
            citation_stem="klhk2023",
            year="2023",
            is_quantitative=True,
        )
    ]
    orch = SubagentOrchestrator(
        engine=MagicMock(),
        task_manager=MagicMock(),
        workspace=MagicMock(),
    )
    directive = orch.build_assembly_directive(
        contract=contract,
        verified_facts=facts,
        missing_notes=[],
    )

    assert "[LEDGER FAKTA TERVERIFIKASI SUBAGENT (TSV SMARTCRUSHER - INTI JAWABAN)]:" in directive
    assert "```tsv" in directive
    assert "Data tahun 2023 tercatat 1.16M ha\tklhk2023\t2023" in directive
