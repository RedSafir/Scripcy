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


def test_bare_line_internal_tool_call_is_hidden_and_executed(
    temp_workspace: WorkspaceManager,
    tmp_path: Path,
) -> None:
    """Bare line tool commands (e.g. /vault read 1) without fences are intercepted and hidden from the user."""
    temp_workspace.save_to_cache(
        "vaswani_2024.md",
        "# Modern Attention Networks\nVaswani et al. 2024\n\nTransformer architecture uses multi-head attention.",
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
            yield MockStreamChunk("/vault read 1\n")
        else:
            yield MockStreamChunk("Berdasarkan (Vaswani et al., 2024), transformer mengandalkan mekanisme self-attention namun memiliki limitasi memori.")

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

    studio.handle_chat("Jelaskan self-attention di paper Vaswani.")

    output = studio.console.file.getvalue()
    assert "/vault read 1" not in output
    assert "Membaca" in output
    assert "Menganalisis" in output
    assert "Vaswani et al., 2024" in output
    assert call_count == 2


def test_sanitize_final_response_strips_slash_commands_and_prompts(
    temp_workspace: WorkspaceManager,
    tmp_path: Path,
) -> None:
    """Ensure _sanitize_final_response removes ambiguous slash commands, prompts, and code blocks from output."""
    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    cfg = ScrpcyConfig(workspace_path=str(temp_workspace.root_dir))
    studio = CLIStudio(workspace=temp_workspace, config=cfg, console=console)

    raw_response = (
        "[STATUS: OK]\n"
        "Model transformer menunjukkan performa superior (Vaswani et al., 2017).\n\n"
        "Ketik /scout transformer efficiency untuk mencari literatur pendukung.\n\n"
        "```scrpcy\n"
        "/scout deep learning\n"
        "/vault read 2\n"
        "```\n\n"
        "Rekomendasi riset mendatang adalah memperluas arsitektur ke domain multimodal."
    )

    cleaned = studio._sanitize_final_response(raw_response)
    assert "[STATUS: OK]" in cleaned
    assert "Vaswani et al., 2017" in cleaned
    assert "multimodal" in cleaned
    assert "/scout" not in cleaned
    assert "/vault read" not in cleaned
    assert "Ketik /scout" not in cleaned
    assert "```scrpcy" not in cleaned


def test_format_vault_citation_guide_injects_metadata(
    temp_workspace: WorkspaceManager,
    tmp_path: Path,
) -> None:
    """Ensure _format_vault_citation_guide produces structured metadata and APA in-text instructions."""
    (temp_workspace.vault_dir / "brown_2020.pdf").write_bytes(b"%PDF-1.4 mock")
    temp_workspace.save_to_cache(
        "brown_2020.md",
        "---\ntitle: Language Models are Few-Shot Learners\nauthors:\n  - Tom Brown\n  - Benjamin Mann\n  - Nick Ryder\nyear: 2020\ncite_key: Brown2020\n---\n# GPT-3\nContent.",
    )
    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "reg.json")
    proj = pm.create_project("thesis", temp_workspace.root_dir)
    cfg = ScrpcyConfig(workspace_path=str(temp_workspace.root_dir))
    studio = CLIStudio(workspace=temp_workspace, config=cfg, console=console, project_manager=pm)
    studio.open_project(proj)

    guide = studio._format_vault_citation_guide()
    assert "PANDUAN SITASI DAN REFERENSI VAULT (GAYA APA):" in guide
    assert "brown_2020" in guide or "Brown" in guide
    assert "Sitasi in-text APA:" in guide


def test_permanent_grounding_and_critic_badges_in_chat_history(
    temp_workspace: WorkspaceManager,
    tmp_path: Path,
) -> None:
    """Ensure grounding and critic badges are permanently saved in chat history and re-displayed upon session re-entry."""
    (temp_workspace.vault_dir / "vaswani_2024.pdf").write_bytes(b"%PDF-1.4 mock")
    temp_workspace.save_to_cache(
        "vaswani_2024.md",
        "---\ntitle: Modern Attention Networks\nauthors:\n  - Vaswani\nyear: 2024\ncite_key: Vaswani2024\n---\n# Content\nSelf-attention is key.",
    )

    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    pm = ProjectManager(registry_file=tmp_path / "reg.json")
    proj = pm.create_project("thesis", temp_workspace.root_dir)
    cfg = ScrpcyConfig(workspace_path=str(temp_workspace.root_dir))

    async def mock_acompletion(**kwargs: Any):
        yield MockStreamChunk("Berdasarkan (Vaswani, 2024), transformer mengandalkan mekanisme self-attention namun memiliki keterbatasan memori.")

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

    sess = studio.chat_manager.create_session("research_eval")
    studio.active_chat_session = sess

    studio.handle_chat("Jelaskan cara kerja self-attention.")

    # 1. Check messages in SQLite chats.db
    messages = studio.chat_manager.get_messages(sess.id)
    assert len(messages) >= 2
    assistant_msg = messages[-1]
    assert assistant_msg.role == "assistant"
    assert "Grounding:" in assistant_msg.content
    assert "Critic:" in assistant_msg.content
    assert "Tokens:" in assistant_msg.content

    # 2. Simulate exiting and re-entering session
    reloaded_studio = CLIStudio(
        workspace=temp_workspace,
        config=cfg,
        console=Console(file=io.StringIO(), force_terminal=True, color_system=None),
        project_manager=pm,
        engine=engine,
    )
    reloaded_studio.open_project(proj)
    reloaded_msgs = reloaded_studio.chat_manager.get_messages(sess.id)
    assert len(reloaded_msgs) >= 2
    reloaded_assist = reloaded_msgs[-1]
    assert "Grounding:" in reloaded_assist.content
    assert "Critic:" in reloaded_assist.content


def test_ask_ai_paywall_action_abstract_vs_alternative(
    temp_workspace: WorkspaceManager,
    tmp_path: Path,
) -> None:
    """Verify _ask_ai_paywall_action queries LLM and parses JSON decision."""
    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    cfg = ScrpcyConfig(workspace_path=str(temp_workspace.root_dir))

    call_count = 0
    async def mock_acompletion(**kwargs: Any):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield MockStreamChunk('```json\n{"decision": "ABSTRACT_ENOUGH", "reason": "Abstrak memuat data penting"}\n```')
        else:
            yield MockStreamChunk('```json\n{"decision": "FIND_ALTERNATIVE", "reason": "Perlu data empiris lengkap"}\n```')

    engine = LLMEngine(config=cfg, acompletion_fn=mock_acompletion)
    studio = CLIStudio(workspace=temp_workspace, config=cfg, console=console, engine=engine)
    studio.config.api_keys["gemini"] = "mock_key"

    paper = ScoutResult(
        title="Impacts of Peatland Wildfire",
        authors=["John Doe"],
        year=2023,
        abstract="This paper studies soil subsidence after fires.",
        source="semantic_scholar",
    )

    res1 = studio._ask_ai_paywall_action(paper, "Bagaimana dampak kebakaran gambut?")
    assert res1["decision"] == "ABSTRACT_ENOUGH"
    assert "Abstrak memuat data" in res1["reason"]

    res2 = studio._ask_ai_paywall_action(paper, "Bagaimana dampak kebakaran gambut?")
    assert res2["decision"] == "FIND_ALTERNATIVE"
    assert "data empiris lengkap" in res2["reason"]


def test_citation_mapping_autodownload_and_paywall_ingest(
    temp_workspace: WorkspaceManager,
    tmp_path: Path,
) -> None:
    """When Devil's Advocate triggers needs_citation_mapping, mapped papers are auto-downloaded, paywalls evaluated, and evidence refreshed."""
    from scrpcy.core.search import sanitize_filename
    from scrpcy.core.critic import CritiqueReport

    (temp_workspace.vault_dir / "seed_paper.pdf").write_bytes(b"%PDF-1.4 mock")
    temp_workspace.save_to_cache(
        "seed_paper.md",
        "---\ntitle: Canal Blocking Seed Study\nauthors:\n  - Seed Author\nyear: 2018\ncite_key: Seed2018\n---\n# Seed\nCanal blocking is implemented in Central Kalimantan to restore peatland hydrology.",
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
            # Grounded in seed paper (year 2018 is outdated > 6 years so critic flags needs_citation_mapping)
            yield MockStreamChunk("Berdasarkan (Seed Author, 2018), sekat kanal memulihkan hidrologi gambut di Kalimantan Tengah.")
        elif call_count == 2:
            # Paywall AI decision: abstract is enough!
            yield MockStreamChunk('```json\n{"decision": "ABSTRACT_ENOUGH", "reason": "Abstrak memuat data regulasi"}\n```')
        else:
            # Synthesizes with newly downloaded papers
            yield MockStreamChunk("Berdasarkan (Seed Author, 2018) dan (Policy Author, 2024), regulasi sekat kanal efektif namun menghadapi limitasi pemeliharaan.")

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

    paywalled_paper = ScoutResult(
        title="Policy on Canal Blocking in Peatland",
        authors=["Policy Author"],
        year=2024,
        abstract="Regulation and canal blocking policy in Central Kalimantan.",
        source="semantic_scholar",
        has_pdf=False,
        pdf_url=None,
    )

    async def mock_map(stems: list[str], limit: int = 10):
        return [paywalled_paper]

    studio.scout.map_vault_citation_network = mock_map

    studio.handle_chat("Bagaimana efektivitas regulasi sekat kanal di Kalimantan?")

    output = studio.console.file.getvalue()
    assert "Reference Mapping" in output
    assert "Paywall Terdeteksi" in output
    assert "Abstrak Memadai" in output

    ingested_file = temp_workspace.vault_dir / f"{sanitize_filename(paywalled_paper.title)}.md"
    assert ingested_file.exists()



