"""Verification suite for the permanent multilingual scout, universal domain filter,
and citation reconciliation architecture.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripcy.core.contracts import AcademicSearchBlueprint, TaskContract, TaskContractExtractor
from scripcy.core.contractor import ContractDecomposer
from scripcy.core.engine import EvidenceChunk, EvidenceSynthesizer
from scripcy.core.references import append_vault_bibliography, resolve_vault_reference_entries
from scripcy.core.search import PaperSearcher, is_cs_ai_domain
from scripcy.core.vector_vault import ProjectVectorVault, get_shared_embedder
from scripcy.core.web_scout import OfficialDomainScorer, QueryDisambiguator


def test_academic_search_blueprint_extraction() -> None:
    """Verify TaskContract builds semantic search blueprint without hardcoded static rules."""
    extractor = TaskContractExtractor()
    contract = extractor.extract("perkembangan arsitektur vision transformer dan yolos untuk deteksi objek")

    assert isinstance(contract, TaskContract)
    assert hasattr(contract, "search_blueprint")
    assert isinstance(contract.search_blueprint, AcademicSearchBlueprint)
    assert len(contract.search_blueprint.queries_local) >= 1
    assert "perkembangan arsitektur vision transformer dan yolos untuk deteksi objek" in contract.search_blueprint.queries_local[0]
    assert len(contract.search_blueprint.canonical_entities) >= 1

    # Verify contractor decomposes search queries from blueprint without hardcoded 'YOLOS' block
    decomposer = ContractDecomposer()
    has_gap, reason, web_q, lit_q = decomposer.detect_evidence_gap(
        query="perkembangan arsitektur vision transformer dan yolos untuk deteksi objek",
        contract=contract,
        evidence=[],
    )
    assert has_gap is True
    assert len(lit_q) >= 1
    # Check that literature queries contain technical terminology
    assert any("vision" in q.lower() or "transformer" in q.lower() or "yolos" in q.lower() for q in lit_q)


def test_universal_official_domain_filter() -> None:
    """Verify universal sovereign and academic TLD parsing across Germany, Switzerland, Netherlands, and IGOs."""
    # Sovereign domains
    germany = OfficialDomainScorer.evaluate("https://www.bund.de/presse/statistik.html")
    assert germany.is_authoritative is True
    assert germany.authority_tier == 1

    swiss = OfficialDomainScorer.evaluate("https://www.admin.ch/gov/de/start.html")
    assert swiss.is_authoritative is True
    assert swiss.authority_tier == 1

    netherlands = OfficialDomainScorer.evaluate("https://www.overheid.nl/open-data")
    assert netherlands.is_authoritative is True
    assert netherlands.authority_tier == 1

    eu = OfficialDomainScorer.evaluate("https://ec.europa.eu/eurostat/data/database")
    assert eu.is_authoritative is True
    assert eu.authority_tier == 1

    igo = OfficialDomainScorer.evaluate("https://www.who.int/data/gho")
    assert igo.is_authoritative is True
    assert igo.authority_tier == 1

    indonesia = OfficialDomainScorer.evaluate("https://www.bps.go.id/id/statistics")
    assert indonesia.is_authoritative is True
    assert indonesia.authority_tier == 1

    # Spam / UGC rejection
    spam = OfficialDomainScorer.evaluate("https://mypersonalopinion.blogspot.com/2024/data-palsu.html")
    assert spam.is_authoritative is False

    # Disambiguator universal generation (no forced site:go.id)
    dq = QueryDisambiguator.disambiguate("French public health statistics")
    for q in dq.targeted_queries:
        assert "site:go.id" not in q
        assert "luas OR angka" not in q


def test_openalex_abstract_reconstruction_and_cs_domain() -> None:
    """Verify OpenAlex inverted index abstract reconstruction and bilingual CS/AI domain matching."""
    searcher = PaperSearcher()

    # Bilingual AI domain matching
    assert is_cs_ai_domain("deteksi objek dengan vision transformer")
    assert is_cs_ai_domain("perbandingan model yolos dan cnn")
    assert is_cs_ai_domain("pembelajaran mendalam arsitektur transformer")
    assert not is_cs_ai_domain("analisis hukum perdata internasional")

    # Inverted index mock test
    mock_work = {
        "title": "You Only Look at One Sequence: Rethinking Transformer in Vision",
        "authorships": [{"author": {"display_name": "Yuxin Fang"}}],
        "publication_year": 2021,
        "doi": "https://doi.org/10.1145/12345",
        "open_access": {"is_oa": True, "oa_url": "https://arxiv.org/pdf/2106.00666.pdf"},
        "abstract_inverted_index": {
            "Transformers": [0],
            "have": [1],
            "revolutionized": [2],
            "computer": [3],
            "vision.": [4],
        },
    }

    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"results": [mock_work]}
    mock_client.get.return_value = mock_resp

    with patch.object(searcher, "_get_client", return_value=mock_client):
        results = pytest.run_async = asyncio_run(searcher.search_openalex("YOLOS", limit=5))

    assert len(results) == 1
    assert results[0].title == "You Only Look at One Sequence: Rethinking Transformer in Vision"
    assert results[0].abstract == "Transformers have revolutionized computer vision."
    assert results[0].pdf_url == "https://arxiv.org/pdf/2106.00666.pdf"


def asyncio_run(coro):
    import asyncio
    return asyncio.run(coro)


def test_multilingual_embedder_dimension() -> None:
    """Verify FastEmbed uses multilingual paraphrase model with 384 dimensions."""
    embedder = get_shared_embedder()
    assert embedder is not None
    # Generate test embedding for multilingual text
    vecs = list(embedder.embed(["Vision Transformer untuk deteksi objek"]))
    assert len(vecs) == 1
    assert len(vecs[0]) == 384


def test_markdown_structural_chunking() -> None:
    """Verify Markdown tables and code blocks remain coherent without mid-table severing."""
    synthesizer = EvidenceSynthesizer(chunk_size_words=20, chunk_overlap_words=5)

    sample_md = (
        "# Evaluasi Model\n\n"
        "Berikut ringkasan performa model deteksi objek pada dataset COCO:\n\n"
        "| Model | mAP | Params | Latency |\n"
        "|---|---|---|---|\n"
        "| YOLOv8 | 53.9 | 43M | 12ms |\n"
        "| YOLOS | 52.3 | 30M | 45ms |\n"
        "| DETR | 42.0 | 41M | 80ms |\n\n"
        "Analisis menunjukkan bahwa YOLOv8 mempertahankan latensi terendah."
    )

    chunks = synthesizer.chunk_document(sample_md, source_name="eval_report")
    assert len(chunks) >= 2

    # Check that any chunk containing table rows includes table headers
    for c in chunks:
        if "| YOLOv8" in c.text or "| YOLOS" in c.text:
            assert "| Model | mAP |" in c.text
            assert "|---|" in c.text


def test_apa7_secondary_source_annotation(tmp_path: Path) -> None:
    """Verify APA 7th secondary source note appears when cited paper is embedded in a review document."""
    # Create fake vault manager workspace with review document
    workspace = MagicMock()
    cache_file = tmp_path / "survey2023.md"
    cache_file.write_text(
        "---\ntitle: Comprehensive Survey on Object Detection\ncite_key: Li2023\n---\n"
        "# Tinjauan Pustaka\n"
        "Sebagaimana dipelopori oleh Redmon et al. (2016) dalam arsitektur YOLO...",
        encoding="utf-8",
    )
    workspace.get_cache_path.side_effect = lambda name: tmp_path / name if name == "survey2023.md" else tmp_path / "other.md"

    vault_item = MagicMock()
    vault_item.stem = "survey2023"
    vault_item.filename = "survey2023.pdf"
    vault_item.title = "Comprehensive Survey on Object Detection"
    vault_item.cite_key = "Li2023"
    vault_item.category = "papers"
    vault_item.path = tmp_path / "survey2023.pdf"

    vault_mgr = MagicMock()
    vault_mgr.workspace = workspace
    vault_mgr.list_active_files.return_value = [vault_item]
    vault_mgr._read_frontmatter_metadata.return_value = None

    ai_text = (
        "Arsitektur single-stage dirintis oleh Redmon et al. (2016) yang merevolusi deteksi objek. "
        "Selain itu [Li2023] merangkum kemajuan transformer."
    )

    result = append_vault_bibliography(ai_text, vault_manager=vault_mgr, style="apa")
    assert "### 📚 Daftar Pustaka:" in result
    # Secondary citation note must be present explaining Redmon et al. (2016)
    assert "Catatan Rujukan Sekunder (APA 7th):" in result
    assert "Redmon et al., 2016" in result or "Redmon" in result
    assert "survey2023" in result or "Li2023" in result


def test_action_suggestion_strict_validation() -> None:
    """Verify ASCII table and flowchart lines can NEVER be proposed or confirmed as Scripcy commands."""
    from scripcy.studio.agent_loop import StudioAgentLoopMixin

    class MockStudio(StudioAgentLoopMixin):
        def __init__(self):
            self.console = MagicMock()
            self.actions_executed = []

        def handle_project_command(self, cmd: str):
            self.actions_executed.append(cmd)

    studio = MockStudio()

    # AI output with ASCII diagram in ```scripcy fence
    dirty_response = (
        "Analisis perbandingan:\n"
        "```scripcy\n"
        "+---------------------------------------------------------------------------------------------------+\n"
        "| ALUR EVOLUSI ARSITEKTUR DETEKSI OBJEK                                                             |\n"
        "| Two-Stage (R-CNN) -> Deteksi terpisah, akurasi tinggi, latensi komputasi berat                    |\n"
        "| | |                                                                                               |\n"
        "| ▼ |                                                                                               |\n"
        "| Single-Stage CNN (YOLO) -> Lokalisasi & klasifikasi simultan, cepat                               |\n"
        "+---------------------------------------------------------------------------------------------------+\n"
        "/findpaper Vision Transformer 2024\n"
        "```"
    )

    confirmed_actions = []
    with patch("scripcy.cli_studio.interactive_action_confirm", side_effect=lambda a, **kw: confirmed_actions.append(a) or ("execute", None)):
        studio._handle_suggested_actions(dirty_response)

    # Diagram lines must be strictly filtered out; only valid Scripcy commands allowed
    assert len(confirmed_actions) == 1
    assert confirmed_actions[0] == "/findpaper Vision Transformer 2024"
    for forbidden in ("+---", "| ALUR", "Two-Stage", "| ▼ |"):
        assert not any(forbidden in a for a in confirmed_actions)
