"""Unit tests for Promisor heuristic rule-based intent router."""

import io
from pathlib import Path
import pytest
from rich.console import Console

from scripcy.cli_studio import CLIStudio
from scripcy.core.config import ScripcyConfig
from scripcy.core.project import ProjectManager
from scripcy.core.promisor import Promisor, PromisorDecision
from scripcy.core.workspace import WorkspaceManager


@pytest.fixture
def promisor() -> Promisor:
    return Promisor()


def test_promisor_user_requested_phrases(promisor: Promisor):
    # 1. "tampilkan setiap gambar dari paper ..."
    d1 = promisor.evaluate("tampilkan setiap gambar dari paper wildfire_study")
    assert d1.is_system_tool is True
    assert d1.action == "vault.images"
    assert d1.args.get("paper") == "wildfire_study"
    assert "wildfire_study" in d1.display_label

    # 2. "token yang sudah digunakan pada sessi ini" / "token yang sudah digunakan pada sesi ini"
    d2 = promisor.evaluate("token yang sudah digunakan pada sesi ini")
    assert d2.is_system_tool is True
    assert d2.action == "models.tokens"
    assert d2.args.get("scope") == "session"

    d2_typo = promisor.evaluate("token yang sudah digunakan pada sessi ini")
    assert d2_typo.is_system_tool is True
    assert d2_typo.action == "models.tokens"

    # 3. "apa saja yang bekas yang error" / "apa saja berkas yang error"
    d3 = promisor.evaluate("apa saja berkas yang error")
    assert d3.is_system_tool is True
    assert d3.action == "vault.errors"

    d3_alt = promisor.evaluate("berkas yang bermasalah di vault")
    assert d3_alt.is_system_tool is True
    assert d3_alt.action == "vault.errors"


def test_promisor_tables_per_paper_and_all(promisor: Promisor):
    # Paper tables
    d1 = promisor.evaluate("tampilkan setiap tabel dari paper 'climate_change'")
    assert d1.is_system_tool is True
    assert d1.action == "vault.tables"
    assert d1.args.get("paper") == "climate_change"

    # All tables
    d2 = promisor.evaluate("tampilkan seluruh tabel di vault")
    assert d2.is_system_tool is True
    assert d2.action == "vault.tables"
    assert "paper" not in d2.args


def test_promisor_vault_operations(promisor: Promisor):
    # Update
    d_up = promisor.evaluate("tolong update vault sekarang")
    assert d_up.is_system_tool is True
    assert d_up.action == "vault.update"

    d_sync = promisor.evaluate("sinkronkan berkas local")
    assert d_sync.is_system_tool is True
    assert d_sync.action == "vault.update"

    # Health
    d_health = promisor.evaluate("cek kesehatan vault")
    assert d_health.is_system_tool is True
    assert d_health.action == "vault.health"

    # History
    d_hist = promisor.evaluate("lihat riwayat sinkronisasi vault")
    assert d_hist.is_system_tool is True
    assert d_hist.action == "vault.history"

    # Archive & restore
    d_arch = promisor.evaluate("arsipkan berkas draft_paper.pdf")
    assert d_arch.is_system_tool is True
    assert d_arch.action == "vault.archive"
    assert d_arch.args.get("target") == "draft_paper.pdf"

    d_rest = promisor.evaluate("pulihkan berkas draft_paper.pdf")
    assert d_rest.is_system_tool is True
    assert d_rest.action == "vault.restore"
    assert d_rest.args.get("target") == "draft_paper.pdf"


def test_promisor_models_and_settings(promisor: Promisor):
    # Model switch
    d_switch = promisor.evaluate("ganti model ke gemini/gemini-2.5-pro")
    assert d_switch.is_system_tool is True
    assert d_switch.action == "models.use"
    assert d_switch.args.get("model") == "gemini/gemini-2.5-pro"

    # Model catalog
    d_list = promisor.evaluate("model apa yang aktif saat ini?")
    assert d_list.is_system_tool is True
    assert d_list.action == "models.list"

    # 9Router
    d_r9 = promisor.evaluate("cek status 9router")
    assert d_r9.is_system_tool is True
    assert d_r9.action == "models.9router"

    # Citation
    d_cit = promisor.evaluate("ganti style sitasi")
    assert d_cit.is_system_tool is True
    assert d_cit.action == "settings.citation"


def test_promisor_scout_and_save(promisor: Promisor):
    d_paper = promisor.evaluate("cari paper tentang machine learning lalu simpan ke vault")
    assert d_paper.is_system_tool is True
    assert d_paper.action == "scout.paper"
    assert "machine learning" in d_paper.args.get("query", "").lower()

    d_web = promisor.evaluate("cari website tentang bmkg cuaca simpan ke vault")
    assert d_web.is_system_tool is True
    assert d_web.action == "scout.web"
    assert "bmkg cuaca" in d_web.args.get("query", "").lower()


def test_promisor_ai_chat_routing(promisor: Promisor):
    # Pure research questions
    d1 = promisor.evaluate("mengapa restorasi gambut penting dalam mitigasi kebakaran hutan?")
    assert d1.is_system_tool is False
    assert d1.action == "ai_chat"

    d2 = promisor.evaluate("bagaimana korelasi antara tinggi muka air tanah dengan laju emisi co2?")
    assert d2.is_system_tool is False
    assert d2.action == "ai_chat"

    d3 = promisor.evaluate("analisis perbandingan model transformer dengan rnn untuk runtun waktu")
    assert d3.is_system_tool is False
    assert d3.action == "ai_chat"

    # System phrase combined with analytical request -> Must route to AI chat!
    d4 = promisor.evaluate("tampilkan setiap gambar dari paper wildfire lalu jelaskan maknanya")
    assert d4.is_system_tool is False
    assert d4.action == "ai_chat"

    d5 = promisor.evaluate("cari paper tentang AI di vault lalu simpulkan intisarinya")
    assert d5.is_system_tool is False
    assert d5.action == "ai_chat"


@pytest.fixture
def test_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=True, color_system=None)


@pytest.fixture
def studio_fixture(temp_workspace: WorkspaceManager, test_console: Console, tmp_path: Path) -> CLIStudio:
    cfg = ScripcyConfig(
        default_model="gemini/gemini-2.5-flash",
        workspace_path=str(temp_workspace.root_dir),
    )
    pm = ProjectManager(registry_file=tmp_path / "projects.json")
    studio = CLIStudio(
        workspace=temp_workspace,
        config=cfg,
        console=test_console,
        project_manager=pm,
    )
    proj = pm.create_project(name="promisor_test", path=temp_workspace.root_dir)
    studio.open_project(proj)
    # Ensure active chat session
    if studio.chat_manager:
        sess = studio.chat_manager.create_session(name="promisor_chat")
        studio.active_chat_session = sess
    return studio


def test_promisor_dispatch_confirmed(studio_fixture: CLIStudio, monkeypatch: pytest.MonkeyPatch):
    called = []

    def mock_confirm(*args, **kwargs):
        return True

    def mock_errors():
        called.append("errors")

    monkeypatch.setattr("scripcy.studio.agent_loop.interactive_confirm", mock_confirm)
    monkeypatch.setattr(studio_fixture, "_handle_promisor_vault_errors", mock_errors)

    decision = PromisorDecision(
        is_system_tool=True,
        action="vault.errors",
        display_label="Periksa berkas bermasalah di Vault",
        command_equivalent="/vault health",
        args={},
        confidence=1.0,
        reason="Test",
    )

    handled = studio_fixture._dispatch_promisor_action(decision, original_prompt="apa saja berkas yang error")
    assert handled is True
    assert "errors" in called


def test_promisor_dispatch_cancelled(studio_fixture: CLIStudio, monkeypatch: pytest.MonkeyPatch):
    called = []

    def mock_confirm(*args, **kwargs):
        return False

    def mock_errors():
        called.append("errors")

    monkeypatch.setattr("scripcy.studio.agent_loop.interactive_confirm", mock_confirm)
    monkeypatch.setattr(studio_fixture, "_handle_promisor_vault_errors", mock_errors)

    decision = PromisorDecision(
        is_system_tool=True,
        action="vault.errors",
        display_label="Periksa berkas bermasalah di Vault",
        command_equivalent="/vault health",
        args={},
        confidence=1.0,
        reason="Test",
    )

    handled = studio_fixture._dispatch_promisor_action(decision, original_prompt="apa saja berkas yang error")
    assert handled is False
    assert len(called) == 0


def test_promisor_handle_chat_submode_system_action(studio_fixture: CLIStudio, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("scripcy.studio.agent_loop.interactive_confirm", lambda *args, **kwargs: True)
    chat_called = []
    monkeypatch.setattr(studio_fixture, "handle_chat", lambda p: chat_called.append(p))

    # User inputs a system phrase
    studio_fixture.handle_chat_submode_interaction("apa saja berkas yang error")

    # Confirm dialog ran, tool ran, handle_chat was NOT called (0 AI tokens)
    assert len(chat_called) == 0


def test_promisor_handle_chat_submode_cancelled_falls_through(studio_fixture: CLIStudio, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("scripcy.studio.agent_loop.interactive_confirm", lambda *args, **kwargs: False)
    chat_called = []
    monkeypatch.setattr(studio_fixture, "handle_chat", lambda p: chat_called.append(p))

    # User inputs a system phrase but cancels confirmation
    studio_fixture.handle_chat_submode_interaction("apa saja berkas yang error")

    # Cancelled -> routed forward to AI chat
    assert len(chat_called) == 1
    assert chat_called[0] == "apa saja berkas yang error"


def test_promisor_handle_chat_submode_direct_ai_question(studio_fixture: CLIStudio, monkeypatch: pytest.MonkeyPatch):
    confirm_called = []
    monkeypatch.setattr("scripcy.studio.agent_loop.interactive_confirm", lambda *args, **kwargs: confirm_called.append(1))
    chat_called = []
    monkeypatch.setattr(studio_fixture, "handle_chat", lambda p: chat_called.append(p))

    # Pure AI conversational research question
    studio_fixture.handle_chat_submode_interaction("mengapa restorasi gambut penting dalam mitigasi kebakaran hutan?")

    # Confirmation was NOT prompted, went straight to handle_chat
    assert len(confirm_called) == 0
    assert len(chat_called) == 1


def test_promisor_handlers_execution(studio_fixture: CLIStudio):
    # Vault errors handler
    studio_fixture._handle_promisor_vault_errors()

    # Vault images handler
    studio_fixture._handle_promisor_vault_images(paper_filter="test")

    # Vault tables handler
    studio_fixture._handle_promisor_vault_tables(paper_filter="test")

    # Tokens handler
    studio_fixture._handle_promisor_tokens(scope="session")

