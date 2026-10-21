"""Unit tests for the autonomous agentic tool execution loop and scrpcy_manual skill."""

import asyncio
import io
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from scrpcy.cli_studio import CLIStudio
from scrpcy.core.config import ScrpcyConfig
from scrpcy.core.engine import LLMEngine
from scrpcy.core.project import ProjectManager
from scrpcy.core.scout import LiteratureScout, ScoutResult
from scrpcy.core.skills import DEFAULT_SCRPCY_MANUAL_PROMPT, SkillsLoader
from scrpcy.core.workspace import WorkspaceManager


class MockStreamChunk:
    def __init__(self, content: str | None) -> None:
        self.choices = [MagicMock(delta=MagicMock(content=content))]


def test_scrpcy_manual_is_default_skill(tmp_path: Path) -> None:
    """Verify scrpcy_manual is auto-seeded as default skill and obsolete personas cleaned up."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    (skills_dir / "academic_writer.md").write_text("old", encoding="utf-8")

    loader = SkillsLoader(skills_dir=skills_dir, auto_init=True)
    skills = loader.list_skills()
    names = [s.name for s in skills]

    assert "scrpcy_manual" in names
    assert "academic_writer" not in names
    assert not (skills_dir / "academic_writer.md").exists()

    manual = loader.get_skill("scrpcy_manual")
    assert "/vault read" in manual.system_prompt
    assert "/vault list" in manual.system_prompt
    assert "/scout" in manual.system_prompt


def test_agentic_loop_internal_vault_read(
    temp_workspace: WorkspaceManager,
    tmp_path: Path,
) -> None:
    """When AI requests an internal read tool (/vault read 1), Scrpcy reads it, shows 'Membaca' & 'Menganalisis', and completes."""
    temp_workspace.save_to_cache(
        "attention_paper.md",
        "# Attention Is All You Need\nVaswani et al.\n\nMulti-Head Attention projects queries, keys, and values.",
    )

    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "reg.json")
    proj = pm.create_project("thesis", temp_workspace.root_dir)
    cfg = ScrpcyConfig(workspace_path=str(temp_workspace.root_dir))

    call_count = 0

    async def mock_acompletion(**kwargs: Any):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield MockStreamChunk("```scrpcy\n/vault read 1\n```")
        else:
            yield MockStreamChunk("Berdasarkan Paper #1 Vaswani et al., Multi-Head Attention menggunakan proyeksi matriks linier.")

    engine = LLMEngine(
        config=cfg,
        acompletion_fn=mock_acompletion,
    )

    studio = CLIStudio(
        workspace=temp_workspace,
        config=cfg,
        console=console,
        project_manager=pm,
        engine=engine,
    )
    studio.open_project(proj)
    studio.config.api_keys["gemini"] = "mock_key"

    studio.handle_chat("Bagaimana cara kerja attention di paper Vaswani?")

    output = studio.console.file.getvalue()
    assert "Berpikir" in output
    assert "Membaca" in output
    assert "Menganalisis" in output
    assert "Multi-Head Attention" in output
    assert "matriks linier" in output
    assert call_count == 2


def test_agentic_loop_direct_final_response(
    temp_workspace: WorkspaceManager,
    tmp_path: Path,
) -> None:
    """When evidence exists and AI does not request internal tool calls, it streams the final response directly."""
    temp_workspace.save_to_cache(
        "methodology.md",
        "# Metodologi Penelitian\nPerbedaan kualitatif dan kuantitatif terletak pada paradigma dan teknik analisis.",
    )

    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "reg.json")
    proj = pm.create_project("thesis", temp_workspace.root_dir)
    cfg = ScrpcyConfig(workspace_path=str(temp_workspace.root_dir))

    async def mock_acompletion(**kwargs: Any):
        yield MockStreamChunk("Ini adalah jawaban konseptual langsung tanpa perlu membaca file lain.")

    engine = LLMEngine(
        config=cfg,
        acompletion_fn=mock_acompletion,
    )

    studio = CLIStudio(
        workspace=temp_workspace,
        config=cfg,
        console=console,
        project_manager=pm,
        engine=engine,
    )
    studio.open_project(proj)
    studio.config.api_keys["gemini"] = "mock_key"

    studio.handle_chat("Apa perbedaan kualitatif dan kuantitatif?")

    output = studio.console.file.getvalue()
    assert "Berpikir" in output
    assert "Ini adalah jawaban konseptual langsung" in output


def test_auto_scout_when_evidence_is_empty(
    temp_workspace: WorkspaceManager,
    tmp_path: Path,
) -> None:
    """When vault has no evidence, Scrpcy proposes a scout plan, user confirms (Y), auto-ingests, and completes."""
    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "reg.json")
    proj = pm.create_project("thesis", temp_workspace.root_dir)
    cfg = ScrpcyConfig(workspace_path=str(temp_workspace.root_dir))

    async def mock_acompletion(**kwargs: Any):
        yield MockStreamChunk("LoRA mengoptimalkan bobot matriks rank rendah secara efisien.")

    engine = LLMEngine(
        config=cfg,
        acompletion_fn=mock_acompletion,
    )

    studio = CLIStudio(
        workspace=temp_workspace,
        config=cfg,
        console=console,
        project_manager=pm,
        engine=engine,
    )
    studio.open_project(proj)
    studio.config.api_keys["gemini"] = "mock_key"

    async def mock_auto_scout(query: str, max_papers: int = 2, on_status=None):
        if on_status:
            on_status("Mengunduh", "Hu et al. LoRA (PDF)")
        temp_workspace.save_to_cache(
            "hu2021_lora.md",
            "# LoRA: Low-Rank Adaptation\nHu et al.\n\nFreezes pre-trained weights and injects trainable rank decomposition matrices.",
        )
        return ["hu2021_lora"]

    studio.scout.auto_scout_and_ingest = mock_auto_scout

    # User presses Enter to approve automatic scouting
    with patch.object(studio.console, "input", return_value=""):
        studio.handle_chat("Bagaimana arsitektur LoRA parameter-efficient fine-tuning?")

    output = studio.console.file.getvalue()
    assert "Mengevaluasi kecukupan literatur di vault" in output
    assert "Bukti di vault belum mencukupi" in output
    assert "Rencana: Mencari & mengunduh literatur" in output
    assert "Mencari" in output
    assert "Mengunduh" in output
    assert "Berhasil mengindeks 1 dokumen baru ke vault" in output
    assert "Membaca" in output
    assert "Menganalisis" in output
    assert "LoRA mengoptimalkan bobot matriks" in output


def test_auto_scout_user_declines(
    temp_workspace: WorkspaceManager,
    tmp_path: Path,
) -> None:
    """When vault has no evidence and user declines (n), system skips scouting and continues to general answer."""
    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "reg.json")
    proj = pm.create_project("thesis", temp_workspace.root_dir)
    cfg = ScrpcyConfig(workspace_path=str(temp_workspace.root_dir))

    async def mock_acompletion(**kwargs: Any):
        yield MockStreamChunk("Jawaban umum tanpa data vault.")

    engine = LLMEngine(config=cfg, acompletion_fn=mock_acompletion)
    studio = CLIStudio(
        workspace=temp_workspace,
        config=cfg,
        console=console,
        project_manager=pm,
        engine=engine,
    )
    studio.open_project(proj)
    studio.config.api_keys["gemini"] = "mock_key"

    # User types 'n'
    with patch.object(studio.console, "input", return_value="n"):
        studio.handle_chat("Apa itu quantum computing?")

    output = studio.console.file.getvalue()
    assert "Pencarian otomatis dilewati" in output
    assert "Jawaban umum tanpa data vault" in output


def test_auto_scout_timeout_falls_back_to_background(
    temp_workspace: WorkspaceManager,
    tmp_path: Path,
) -> None:
    """When scouting takes > 20s, it times out and dispatches to background TaskManager."""
    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "reg.json")
    proj = pm.create_project("thesis", temp_workspace.root_dir)
    cfg = ScrpcyConfig(workspace_path=str(temp_workspace.root_dir))

    async def mock_acompletion(**kwargs: Any):
        yield MockStreamChunk("Jawaban sementara saat unduhan berjalan di background.")

    engine = LLMEngine(config=cfg, acompletion_fn=mock_acompletion)
    studio = CLIStudio(
        workspace=temp_workspace,
        config=cfg,
        console=console,
        project_manager=pm,
        engine=engine,
    )
    studio.open_project(proj)
    studio.config.api_keys["gemini"] = "mock_key"

    async def mock_hanging_scout(query: str, max_papers: int = 2, on_status=None):
        raise asyncio.TimeoutError()

    studio.scout.auto_scout_and_ingest = mock_hanging_scout

    with patch.object(studio.console, "input", return_value="y"):
        studio.handle_chat("Jelaskan transformer deep architectures")

    output = studio.console.file.getvalue()
    assert "Unduhan repositori membutuhkan waktu > 20 detik" in output
    assert "Proses dialihkan ke background:" in output


def test_ingest_abstract_paper_creates_markdown(
    temp_workspace: WorkspaceManager,
) -> None:
    """Test fallback abstract ingestion for closed-access/paywalled papers."""
    scout = LiteratureScout(workspace=temp_workspace)

    paper = ScoutResult(
        title="Commercial Closed Access Study",
        authors=["John Smith", "Jane Doe"],
        year=2023,
        doi="10.1000/182",
        abstract="This is the official published abstract detailing novel algorithms.",
        has_pdf=False,
        pdf_url=None,
    )

    created_path = scout.ingest_abstract_paper(paper)
    assert created_path.exists()
    content = created_path.read_text(encoding="utf-8")
    assert "Commercial Closed Access Study" in content
    assert "John Smith" in content
    assert "paywall komersial" in content
    assert "This is the official published abstract" in content
    assert temp_workspace.get_cache_path(f"{created_path.stem}.md").exists()


def test_vault_remove_internal_tool(
    temp_workspace: WorkspaceManager,
    tmp_path: Path,
) -> None:
    """Verify /vault remove archives the file to .archive/ and excludes it from active documents."""
    (temp_workspace.vault_dir / "old_paper.pdf").write_bytes(b"%PDF-1.4 mock content")
    temp_workspace.save_to_cache("old_paper.md", "# Old Paper Content")

    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "reg.json")
    proj = pm.create_project("thesis", temp_workspace.root_dir)
    cfg = ScrpcyConfig(workspace_path=str(temp_workspace.root_dir))

    engine = LLMEngine(config=cfg)
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
    assert "old_paper" in studio._get_cached_stems()

    action_label, target_label, tool_output = studio._execute_internal_vault_tool("/vault remove 1")
    assert action_label == "Mengarsipkan"
    assert "old_paper" in target_label
    assert ".archive/" in tool_output
    assert (temp_workspace.vault_dir / ".archive" / "old_paper.pdf").exists()


def test_auto_scout_error_recovery_picks_oa_alternative(
    temp_workspace: WorkspaceManager,
) -> None:
    """When primary candidate fails with paywall/network error, scout tries alternative OA candidate."""
    scout = LiteratureScout(workspace=temp_workspace)

    p1 = ScoutResult(
        title="Paywalled Candidate Paper",
        authors=["Alice"],
        year=2023,
        has_pdf=True,
        pdf_url="https://example.com/paywalled.pdf",
        abstract="Paywalled abstract",
    )
    p2 = ScoutResult(
        title="Open Access Alternative Paper",
        authors=["Bob"],
        year=2023,
        has_pdf=True,
        pdf_url="https://example.com/openaccess.pdf",
        abstract="Open access abstract",
    )

    from unittest.mock import AsyncMock
    scout.search_all_repositories = AsyncMock(return_value=[p1, p2])
    scout.curate_and_rank = MagicMock(return_value=[p1, p2])

    download_calls = []

    async def mock_download(paper: ScoutResult):
        download_calls.append(paper.title)
        if "Paywalled" in paper.title:
            raise RuntimeError("403 Forbidden: Institutional Subscription Required")
        # Alternative succeeds
        path = temp_workspace.vault_dir / "open_access_alternative_paper.pdf"
        path.write_bytes(b"%PDF-1.4 open access")
        temp_workspace.save_to_cache("open_access_alternative_paper.md", "# OA Paper")
        return path

    scout.download_scout_paper = mock_download

    status_updates = []

    def on_status(action: str, label: str):
        status_updates.append((action, label))

    stems = asyncio.run(scout.auto_scout_and_ingest("machine learning", max_papers=1, on_status=on_status))

    assert "Paywalled Candidate Paper" in download_calls
    assert "Open Access Alternative Paper" in download_calls
    assert any("Mencoba alternatif Open-Access" in lbl for _, lbl in status_updates)
    assert any("open_access_alternative_paper" in s.lower() for s in stems)


def test_auto_scout_timeout_consults_ai_with_background_notice(
    temp_workspace: WorkspaceManager,
    tmp_path: Path,
) -> None:
    """When auto-scouting times out, AI query receives background task notification so it can consult user."""
    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "reg.json")
    proj = pm.create_project("thesis", temp_workspace.root_dir)
    cfg = ScrpcyConfig(workspace_path=str(temp_workspace.root_dir))

    received_prompts = []

    async def mock_acompletion(**kwargs: Any):
        # Inspect system/user prompt passed to LLM
        messages = kwargs.get("messages", [])
        for m in messages:
            received_prompts.append(m.get("content", ""))
        yield MockStreamChunk("Sintesis awal sambil menunggu unduhan di background...")

    engine = LLMEngine(config=cfg, acompletion_fn=mock_acompletion)
    studio = CLIStudio(
        workspace=temp_workspace,
        config=cfg,
        console=console,
        project_manager=pm,
        engine=engine,
    )
    studio.open_project(proj)
    studio.config.api_keys["gemini"] = "mock_key"

    async def mock_timeout_scout(*args, **kwargs):
        raise asyncio.TimeoutError()

    studio.scout.auto_scout_and_ingest = mock_timeout_scout

    with patch.object(studio.console, "input", return_value="y"):
        studio.handle_chat("Jelaskan implementasi distributed reinforcement learning")

    output = studio.console.file.getvalue()
    assert "Unduhan repositori membutuhkan waktu > 20 detik" in output
    assert "Proses dialihkan ke background:" in output

    all_received_text = " ".join(received_prompts)
    assert "CATATAN SISTEM SCRPCY: Proses penelusuran & pengunduhan literatur" in all_received_text
    assert "Task ID:" in all_received_text


def test_agentic_loop_internal_vault_remove(
    temp_workspace: WorkspaceManager,
    tmp_path: Path,
) -> None:
    """When AI requests an internal remove tool (/vault remove 1), Scrpcy archives it, shows 'Mengarsipkan' & 'Menganalisis', and completes."""
    (temp_workspace.vault_dir / "ocean_paper.pdf").write_bytes(b"%PDF-1.4 dummy")
    temp_workspace.save_to_cache(
        "ocean_paper.md",
        "# Ocean Iron Cycle\nIrrelevant paper to wildfire research.",
    )

    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "reg.json")
    proj = pm.create_project("thesis", temp_workspace.root_dir)
    cfg = ScrpcyConfig(workspace_path=str(temp_workspace.root_dir))

    call_count = 0

    async def mock_acompletion(**kwargs: Any):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield MockStreamChunk("```scrpcy\n/vault remove 1\n```")
        else:
            yield MockStreamChunk("Dokumen ocean iron cycle telah disingkirkan dari analisis utama.")

    engine = LLMEngine(
        config=cfg,
        acompletion_fn=mock_acompletion,
    )

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

    studio.handle_chat("Tolong bersihkan paper ocean dari vault.")

    output = studio.console.file.getvalue()
    assert "Mengarsipkan" in output
    assert "Menganalisis" in output
    assert "telah disingkirkan" in output
    assert call_count == 2
    assert (temp_workspace.vault_dir / ".archive" / "ocean_paper.pdf").exists()


def test_detailed_background_chat_task_logs(
    temp_workspace: WorkspaceManager,
    tmp_path: Path,
) -> None:
    """Background chat worker logs user query, model, tokens, and full AI response."""
    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "reg.json")
    proj = pm.create_project("thesis", temp_workspace.root_dir)
    cfg = ScrpcyConfig(workspace_path=str(temp_workspace.root_dir))

    async def mock_acompletion(**kwargs: Any):
        yield MockStreamChunk("Ini adalah sintesis lengkap dari respons AI latar belakang.")

    engine = LLMEngine(config=cfg, acompletion_fn=mock_acompletion)
    studio = CLIStudio(
        workspace=temp_workspace,
        config=cfg,
        console=console,
        project_manager=pm,
        engine=engine,
    )
    studio.open_project(proj)
    studio.config.api_keys["gemini"] = "mock_key"

    query = "Apa implikasi kebakaran hutan gambut tropis?"
    studio.submit_chat_background(query)

    import time
    time.sleep(0.4)

    tasks = studio.task_manager.list_tasks()
    assert len(tasks) > 0
    t = tasks[0]
    log_dump = "\n".join(t.logs)
    assert "Pertanyaan: Apa implikasi kebakaran hutan gambut tropis?" in log_dump
    assert "Model:" in log_dump
    assert "Token:" in log_dump
    assert "Selesai!" in log_dump
    assert "Respons AI:" not in log_dump
    assert t.tokens_burned > 0
