"""Pytest fixtures for scrpcy parser and workspace testing."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pydantic
import pymupdf
import pytest

from scrpcy.core.extractor import MockMetadataExtractor
from scrpcy.core.models import DocumentMetadata
from scrpcy.core.workspace import WorkspaceManager


@pytest.fixture
def temp_workspace(tmp_path: Path) -> WorkspaceManager:
    """Provide an isolated WorkspaceManager backed by a temporary directory."""
    workspace_root = tmp_path / "scrpcy_test_workspace"
    return WorkspaceManager(root_dir=workspace_root)


@pytest.fixture
def sample_metadata() -> DocumentMetadata:
    """Provide standard academic publication metadata for testing."""
    return DocumentMetadata(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        year=2017,
        doi="10.48550/arXiv.1706.03762",
        cite_key="vaswani2017attention",
    )


@pytest.fixture
def mock_extractor(sample_metadata: DocumentMetadata) -> MockMetadataExtractor:
    """Provide a deterministic MockMetadataExtractor with sample metadata."""
    return MockMetadataExtractor(fixed_metadata=sample_metadata)


@pytest.fixture
def synthetic_pdf_factory(tmp_path: Path) -> Callable[..., Path]:
    """Factory fixture to create valid synthetic academic PDF files."""

    def _create_pdf(
        filename: str = "paper.pdf",
        title: str = "Attention Is All You Need",
        authors: list[str] | None = None,
        year: int = 2017,
        doi: str = "10.48550/arXiv.1706.03762",
        body_sections: list[str] | None = None,
        pages_count: int = 1,
    ) -> Path:
        author_list = authors or ["Ashish Vaswani", "Noam Shazeer"]
        sections = body_sections or [
            "Abstract\nThe dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
            "1. Introduction\nRecurrent neural networks, long short-term memory and gated recurrent neural networks have been firmly established.",
        ]

        pdf_path = tmp_path / filename
        doc = pymupdf.open()

        # First page with academic title, metadata and opening text
        first_page = doc.new_page()
        page_content = (
            f"{title}\n\n"
            f"Authors: {', '.join(author_list)}\n"
            f"Year: {year}\n"
            f"DOI: {doi}\n\n"
            f"{sections[0]}\n"
        )
        first_page.insert_text((50, 72), page_content)

        # Additional pages if requested
        for i in range(1, pages_count):
            extra_page = doc.new_page()
            section_text = sections[i] if i < len(sections) else f"Page {i + 1} continued research content."
            extra_page.insert_text((50, 72), section_text)

        doc.save(pdf_path)
        doc.close()
        return pdf_path

    return _create_pdf


@pytest.fixture
def corrupted_pdf_factory(tmp_path: Path) -> Callable[[str, bytes], Path]:
    """Factory fixture to generate corrupted, empty, or invalid PDF files."""

    def _create_corrupted(filename: str, payload: bytes) -> Path:
        target = tmp_path / filename
        target.write_bytes(payload)
        return target

    return _create_corrupted
