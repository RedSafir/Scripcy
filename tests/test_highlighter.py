"""Unit tests for EvidenceHighlighter and Deep-Linking Bibliography."""

from pathlib import Path
import pytest
import pymupdf as fitz

from scripcy.core.engine import EvidenceChunk
from scripcy.core.highlighter import EvidenceHighlighter
from scripcy.core.references import append_vault_bibliography
from scripcy.core.vault import VaultItem


def test_highlight_pdf_passage_creates_yellow_annotation(tmp_path: Path):
    # 1. Create a dummy PDF with wrapped text box
    pdf_path = tmp_path / "sample_paper.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    sample_text = (
        "Penelitian ini menganalisis kinerja model deep learning pada dataset citra medis. "
        "Metode random forest menghasilkan akurasi 95% pada klasifikasi tabular. "
        "Kesimpulan menunjukkan efisiensi komputasi yang optimal."
    )
    rect = fitz.Rect(50, 50, 500, 300)
    page.insert_textbox(rect, sample_text, fontsize=11)
    doc.save(str(pdf_path))
    doc.close()

    # 2. Highlight passage
    annot_dir = tmp_path / "annotated"
    result_pdf = EvidenceHighlighter.highlight_pdf_passage(
        pdf_path=pdf_path,
        page_num=1,
        quote_snippet="Metode random forest menghasilkan akurasi 95%",
        output_dir=annot_dir,
        prefix_id="ev1",
    )

    assert result_pdf is not None
    assert result_pdf.exists()
    assert result_pdf.parent == annot_dir
    assert "sample_paper_ev1_p1.pdf" in result_pdf.name

    # 3. Verify annotation exists and is a yellow highlight
    annot_doc = fitz.open(str(result_pdf))
    ann_page = annot_doc[0]
    annots = [a for a in ann_page.annots()]
    assert len(annots) >= 1
    annot = annots[0]
    assert annot.type[0] == 8  # Highlight
    colors = annot.colors
    assert "stroke" in colors
    assert pytest.approx(colors["stroke"][0], 0.05) == 1.0
    assert pytest.approx(colors["stroke"][1], 0.05) == 0.9
    annot_doc.close()


def test_generate_web_highlight_url():
    base_url = "https://kemkes.go.id/id/profil-kesehatan-indonesia-2023"
    snippet = "Jumlah puskesmas di Indonesia mencapai 10.416 unit pada akhir tahun 2023"
    result_url = EvidenceHighlighter.generate_web_highlight_url(base_url, snippet)

    assert result_url.startswith(base_url + "#:~:text=")
    assert "Jumlah" in result_url
    assert "puskesmas" in result_url


class MockWorkspace:
    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path

    def get_cache_path(self, filename: str) -> Path:
        return self.tmp_path / filename


class MockVaultManager:
    def __init__(self, tmp_path: Path, items: list[VaultItem]):
        self.vault_dir = tmp_path
        self.workspace = MockWorkspace(tmp_path)
        self._items = items

    def list_active_files(self) -> list[VaultItem]:
        return self._items

    def _read_frontmatter_metadata(self, path: Path):
        return None


def test_append_vault_bibliography_with_pdf_stabilo(tmp_path: Path):
    # Create sample PDF
    pdf_path = tmp_path / "ai_ethics_2024.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(
        fitz.Rect(50, 50, 500, 300),
        "Etika kecerdasan buatan menjadi pilar penting regulasi.",
        fontsize=11,
    )
    doc.save(str(pdf_path))
    doc.close()

    v_item = VaultItem(
        filename="ai_ethics_2024.pdf",
        stem="ai_ethics_2024",
        path=str(pdf_path),
        extension=".pdf",
        title="Etika Kecerdasan Buatan",
        category="papers",
        size_bytes=1000,
        cite_key="Ethics2024",
    )

    vm = MockVaultManager(tmp_path, [v_item])

    ai_text = "Penerapan AI harus memperhatikan regulasi ketat [Ethics2024]."
    evidence = [
        EvidenceChunk(
            chunk_id="chk_ethics_1",
            text="Etika kecerdasan buatan menjadi pilar penting regulasi.",
            source="ai_ethics_2024",
            score=0.92,
            page_num=1,
            quote_snippet="Etika kecerdasan buatan menjadi pilar penting",
        )
    ]

    result = append_vault_bibliography(
        text=ai_text,
        evidence=evidence,
        vault_manager=vm,  # type: ignore
        style="apa",
    )

    assert "### 📚 Daftar Pustaka:" in result
    assert "(Tersorot Stabilo)" in result
    assert "file://" in result
    assert "#page=1" in result
    annotated_files = list((tmp_path / "annotated").glob("*.pdf"))
    assert len(annotated_files) == 1
