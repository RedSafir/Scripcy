"""Unit tests for Promisor AI intent & anti-spam gatekeeper."""

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


def test_promisor_natural_language_routes_to_ai(promisor: Promisor):
    # Natural language queries are forwarded to AI (Contractor/Examiner/Manager)
    d1 = promisor.evaluate("tampilkan setiap gambar dari paper wildfire_study")
    assert d1.is_system_tool is False
    assert d1.action == "ai_chat"

    d2 = promisor.evaluate("token yang sudah digunakan pada sesi ini")
    assert d2.is_system_tool is False
    assert d2.action == "ai_chat"

    d3 = promisor.evaluate("apa saja berkas yang error")
    assert d3.is_system_tool is False
    assert d3.action == "ai_chat"


def test_promisor_slash_commands_are_deterministic(promisor: Promisor):
    d_vault = promisor.evaluate("/vault errors")
    assert d_vault.is_system_tool is True
    assert d_vault.action == "slash.vault"

    d_tokens = promisor.evaluate("/tokens session")
    assert d_tokens.is_system_tool is True
    assert d_tokens.action == "slash.tokens"

    d_help = promisor.evaluate("/help")
    assert d_help.is_system_tool is True
    assert d_help.action == "slash.help"


def test_promisor_legacy_system_intent_flag(promisor: Promisor):
    # When explicitly opted into legacy intent matching:
    d_up = promisor.evaluate("tolong update vault sekarang", legacy_system_intent=True)
    assert d_up.is_system_tool is True
    assert d_up.action == "vault.update"

    d_img = promisor.evaluate("tampilkan gambar di vault", legacy_system_intent=True)
    assert d_img.is_system_tool is True
    assert d_img.action == "vault.images"


def test_promisor_spam_and_greetings(promisor: Promisor):
    # Spam / pure symbols
    d_spam = promisor.evaluate("...... ??!?!@#$")
    assert d_spam.action == "spam.reject"
    assert d_spam.entry_point == "direct"

    # Casual greetings
    d_greet = promisor.evaluate("halo scripcy")
    assert d_greet.action == "casual.reply"
    assert d_greet.entry_point == "direct"
    assert d_greet.direct_reply is not None


def test_promisor_ai_chat_routing(promisor: Promisor):
    # Pure research questions
    d1 = promisor.evaluate("mengapa restorasi gambut penting dalam mitigasi kebakaran hutan?")
    assert d1.is_system_tool is False
    assert d1.action == "ai_chat"
    assert d1.entry_point == "contractor"

    d2 = promisor.evaluate("bagaimana korelasi antara tinggi muka air tanah dengan laju emisi co2?")
    assert d2.is_system_tool is False
    assert d2.action == "ai_chat"

    d3 = promisor.evaluate("analisis perbandingan model transformer dengan rnn untuk runtun waktu")
    assert d3.is_system_tool is False
    assert d3.action == "ai_chat"

    # Revision
    d4 = promisor.evaluate("revisi paragraf kedua dan buat lebih ringkas")
    assert d4.is_system_tool is False
    assert d4.entry_point == "manager"

    # Hypothesis
    d5 = promisor.evaluate("apakah benar restnet50 lebih akurat dari vgg16?")
    assert d5.is_system_tool is False
    assert d5.entry_point == "examiner"


@pytest.fixture
def studio_fixture(temp_workspace: WorkspaceManager) -> CLIStudio:
    config = ScripcyConfig()
    studio = CLIStudio(workspace=temp_workspace, config=config)
    pm = ProjectManager(registry_file=temp_workspace.root_dir / ".scripcy" / "projects.json")
    proj = pm.create_project(name="promisor_test", path=temp_workspace.root_dir)
    studio.current_project = proj
    studio.project_manager = pm
    if studio.chat_manager:
        sess = studio.chat_manager.create_session(name="promisor_chat")
        studio.active_chat_session = sess
    return studio


def test_promisor_dispatch_confirmed(studio_fixture: CLIStudio, monkeypatch: pytest.MonkeyPatch):
    called = []

    def mock_errors():
        called.append(True)

    monkeypatch.setattr(studio_fixture, "_handle_promisor_vault_errors", mock_errors)

    decision = PromisorDecision(
        is_system_tool=True,
        action="vault.errors",
        display_label="Tampilkan Daftar Berkas Rusak",
        command_equivalent="/vault errors",
        reason="Test",
    )
    monkeypatch.setattr("scripcy.studio.agent_loop.interactive_confirm", lambda *args, **kwargs: True)

    handled = studio_fixture._dispatch_promisor_action(decision, original_prompt="apa saja berkas yang error")
    assert handled is True
    assert len(called) == 1


def test_promisor_dispatch_cancelled(studio_fixture: CLIStudio, monkeypatch: pytest.MonkeyPatch):
    called = []

    def mock_errors():
        called.append(True)

    monkeypatch.setattr(studio_fixture, "_handle_promisor_vault_errors", mock_errors)

    decision = PromisorDecision(
        is_system_tool=True,
        action="vault.errors",
        display_label="Tampilkan Daftar Berkas Rusak",
        command_equivalent="/vault errors",
        reason="Test",
    )
    monkeypatch.setattr("scripcy.studio.agent_loop.interactive_confirm", lambda *args, **kwargs: False)

    handled = studio_fixture._dispatch_promisor_action(decision, original_prompt="apa saja berkas yang error")
    assert handled is False
    assert len(called) == 0


def test_promisor_handlers_execution(studio_fixture: CLIStudio):
    # Ensure handlers run cleanly without raising exceptions
    studio_fixture._handle_promisor_vault_errors()
    studio_fixture._handle_promisor_vault_images(paper_filter="test")
    studio_fixture._handle_promisor_vault_tables(paper_filter="test")
    studio_fixture._handle_promisor_tokens(scope="session")
