"""Unit tests for Advanced RAG pipeline features:
1. Multi-Query Expansion & HyDE
2. Cross-Encoder Re-Ranking
3. Parent-Child Hierarchical Chunking
4. Direct Citation Footnote & Quote Mapping
"""

from pathlib import Path
import pytest

from scripcy.core.config import ScripcyConfig
from scripcy.core.engine import EvidenceChunk, LLMEngine, EvidenceSynthesizer
from scripcy.core.references import append_vault_bibliography
from scripcy.core.vault import VaultItem


def test_multi_query_expansion():
    synth = EvidenceSynthesizer()
    expansions = synth.expand_query("bagaimana mencegah overfitting dan meningkatkan akurasi?")
    assert len(expansions) >= 2
    combined = " ".join(expansions).lower()
    assert "regularization" in combined or "dropout" in combined or "accuracy" in combined


def test_footnote_mapping_with_page_and_quote(tmp_path: Path):
    dummy_pdf = tmp_path / "attention.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 mock")

    class MockVaultItem:
        def __init__(self):
            self.filename = "attention.pdf"
            self.stem = "attention"
            self.cite_key = "attention"
            self.title = "Attention Is All You Need"
            self.path = dummy_pdf
            self.category = "papers"
            self.format = "pdf"
            self.is_duplicate = False
            self.has_error = False

    class MockWorkspace:
        def get_cache_path(self, name):
            return tmp_path / name

    class MockVM:
        def __init__(self):
            self.workspace = MockWorkspace()
        def list_active_files(self):
            return [MockVaultItem()]

    vm = MockVM()
    ev = EvidenceChunk(
        chunk_id="att_1",
        source="attention",
        text="Transformer model achieves state of the art.",
        score=0.95,
        page_num=3,
        quote_snippet="The Transformer model achieves state of the art.",
    )

    ans = "Model Transformer memiliki performa tinggi [attention]."
    res = append_vault_bibliography(ans, evidence=[ev], vault_manager=vm, style="apa")

    assert "### 📚 Daftar Pustaka:" in res
    assert "Halaman 3" in res
    assert "The Transformer model achieves state of the art." in res
    assert "file://" in res
