"""Unit tests for automatic vault bibliography generation and chat integration."""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
import pytest
from rich.console import Console

from scrpcy.core.config import ScrpcyConfig
from scrpcy.core.engine import EvidenceChunk, LLMEngine
from scrpcy.core.project import ProjectManager
from scrpcy.core.references import (
    append_vault_bibliography,
    resolve_vault_reference_entries,
)
from scrpcy.core.vault import VaultManager
from scrpcy.core.workspace import WorkspaceManager
from scrpcy.cli_studio import CLIStudio


class TestVaultBibliographyAppender:
    """Tests for resolving citations to vault metadata and formatting bibliographies."""

    def test_resolve_vault_reference_entries_apa(
        self,
        temp_workspace: WorkspaceManager,
    ) -> None:
        # Create a sample vault document with metadata in cache
        doc_content = (
            "---\n"
            "title: The amount of carbon released from peatland fires in Indonesia during 1997\n"
            "authors:\n"
            "  - Susan E. Page\n"
            "  - Florian Siegert\n"
            "  - John O. Rieley\n"
            "year: 2002\n"
            "doi: 10.1038/nature01131\n"
            "cite_key: Page2002\n"
            "---\n"
            "# Content\nPeatland fire emissions.\n"
        )
        pdf = temp_workspace.vault_dir / "peatland_fire_2002.pdf"
        pdf.write_bytes(b"%PDF-mock")
        os.utime(pdf, (time.time() - 100, time.time() - 100))
        temp_workspace.save_to_cache("peatland_fire_2002.md", doc_content)

        vm = VaultManager(workspace=temp_workspace)
        text = "Kebakaran lahan gambut melepaskan karbon dalam jumlah masif [Page2002]."
        evidence = [EvidenceChunk(chunk_id="c1", source="peatland_fire_2002", text="carbon release")]

        entries = resolve_vault_reference_entries(text, evidence, vm, style="apa")
        assert len(entries) == 1
        entry = entries[0]
        assert entry["filename"] == "peatland_fire_2002.pdf"
        assert "Page, S. E." in entry["formatted_entry"]
        assert "(2002)." in entry["formatted_entry"]
        assert "https://doi.org/10.1038/nature01131" in entry["formatted_entry"]

    def test_resolve_vault_reference_entries_ieee(
        self,
        temp_workspace: WorkspaceManager,
    ) -> None:
        doc_content = (
            "---\n"
            "title: Attention Is All You Need\n"
            "authors:\n"
            "  - Ashish Vaswani\n"
            "  - Noam Shazeer\n"
            "year: 2017\n"
            "cite_key: Vaswani2017\n"
            "---\n"
            "# Content\nTransformer architecture.\n"
        )
        pdf = temp_workspace.vault_dir / "vaswani2017.pdf"
        pdf.write_bytes(b"%PDF-mock")
        os.utime(pdf, (time.time() - 100, time.time() - 100))
        temp_workspace.save_to_cache("vaswani2017.md", doc_content)

        vm = VaultManager(workspace=temp_workspace)
        text = "Arsitektur Transformer berbasis self-attention [1]."
        evidence = [EvidenceChunk(chunk_id="c1", source="vaswani2017", text="self-attention")]

        entries = resolve_vault_reference_entries(text, evidence, vm, style="ieee")
        assert len(entries) == 1
        entry = entries[0]
        assert entry["filename"] == "vaswani2017.pdf"
        assert "A. Vaswani and N. Shazeer" in entry["formatted_entry"]
        assert "2017." in entry["formatted_entry"]

    def test_append_vault_bibliography_apa_formatting(
        self,
        temp_workspace: WorkspaceManager,
    ) -> None:
        doc_content = (
            "---\n"
            "title: Wildfire Dynamics\n"
            "authors:\n"
            "  - Jane Smith\n"
            "year: 2021\n"
            "cite_key: Smith2021\n"
            "---\n"
            "# Content\nWildfire.\n"
        )
        pdf = temp_workspace.vault_dir / "smith2021.pdf"
        pdf.write_bytes(b"%PDF-mock")
        os.utime(pdf, (time.time() - 100, time.time() - 100))
        temp_workspace.save_to_cache("smith2021.md", doc_content)

        vm = VaultManager(workspace=temp_workspace)
        original_text = "Dinamika kebakaran sangat dipengaruhi kondisi hidrologis [Smith2021]."
        enriched = append_vault_bibliography(original_text, vault_manager=vm, style="apa")

        assert "### 📚 Referensi Vault (APA):" in enriched
        assert "Smith, J. (2021)." in enriched
        assert "[File: smith2021.pdf]" in enriched

    def test_append_vault_bibliography_avoids_duplicate_section(
        self,
        temp_workspace: WorkspaceManager,
    ) -> None:
        vm = VaultManager(workspace=temp_workspace)
        text_with_ref = (
            "Sintesis ilmiah.\n\n"
            "### Referensi:\n1. Referensi sudah ada sebelumnya."
        )
        result = append_vault_bibliography(text_with_ref, vault_manager=vm, style="apa")
        # Should not append a second reference header
        assert result == text_with_ref

    def test_chat_integration_appends_bibliography_automatically(
        self,
        temp_workspace: WorkspaceManager,
        tmp_path: Path,
    ) -> None:
        doc_content = (
            "---\n"
            "title: Peat Fire Carbon Study\n"
            "authors:\n"
            "  - Susan Page\n"
            "year: 2023\n"
            "cite_key: Page2023\n"
            "---\n"
            "# Content\nKebakaran lahan gambut tropis memicu pelepasan karbon masif dan partikel PM2.5 [Page2023].\n"
        )
        pdf = temp_workspace.vault_dir / "page2023.pdf"
        pdf.write_bytes(b"%PDF-mock")
        os.utime(pdf, (time.time() - 100, time.time() - 100))
        temp_workspace.save_to_cache("page2023.md", doc_content)

        console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
        pm = ProjectManager(registry_file=tmp_path / "reg.json")
        proj = pm.create_project("thesis", temp_workspace.root_dir)
        cfg = ScrpcyConfig(workspace_path=str(temp_workspace.root_dir))
        cfg.citation.style = "apa"

        class MockChunk:
            def __init__(self, content: str) -> None:
                self.choices = [MagicMock(delta=MagicMock(content=content))]
                self.text = content

        async def mock_acompletion(**kwargs: Any):
            yield MockChunk(
                "Kebakaran lahan gambut tropis memicu pelepasan karbon masif ke atmosfer global [Page2023]."
            )

        engine = LLMEngine(config=cfg, acompletion_fn=mock_acompletion)
        studio = CLIStudio(
            workspace=temp_workspace,
            config=cfg,
            console=console,
            project_manager=pm,
            engine=engine,
        )
        studio.open_project(proj)
        studio.vault_manager.reload_vault(force=False)
        studio._sync_vault_cache()
        studio.config.api_keys["gemini"] = "mock_key"
        studio.scout = None  # Prevent interactive deep-research prompt during automated test

        studio.handle_chat("Jelaskan kebakaran gambut?")

        output = studio.console.file.getvalue()
        # Verify that transparent bibliography was automatically appended to the chat
        assert "📚 Referensi Vault (APA):" in output
        assert "Page, S. (2023)." in output
        assert "[File: page2023.pdf]" in output

        # Verify that chat session history also stores the bibliography
        sess = studio.active_chat_session
        msgs = studio.chat_manager.get_messages(sess.id)
        assistant_msgs = [m for m in msgs if m.role == "assistant"]
        assert len(assistant_msgs) > 0
        assert "### 📚 Referensi Vault (APA):" in assistant_msgs[-1].content
        assert "[File: page2023.pdf]" in assistant_msgs[-1].content
