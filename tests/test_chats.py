"""Unit tests for SQLite-backed ChatManager."""

from __future__ import annotations

from pathlib import Path

import pytest

from scrpcy.core.chats import ChatManager


@pytest.fixture
def chat_manager(tmp_path: Path) -> ChatManager:
    """Fixture providing an isolated ChatManager backed by SQLite."""
    return ChatManager(db_path=tmp_path / "chats.db")


class TestChatManager:
    """Tests for multi-session chat storage, history querying, and exports."""

    def test_create_and_list_sessions(self, chat_manager: ChatManager) -> None:
        # Auto-naming default
        s1 = chat_manager.create_session()
        assert s1.name == "session1"
        assert s1.id == "session1"

        # Sequential auto-naming
        s2 = chat_manager.create_session()
        assert s2.name == "session2"

        # Custom naming
        s3 = chat_manager.create_session("Bab 2 Teori", active_skill="literature_synthesizer")
        assert s3.name == "Bab 2 Teori"
        assert s3.active_skill == "literature_synthesizer"

        sessions = chat_manager.list_sessions()
        assert len(sessions) == 3

    def test_add_and_retrieve_messages(self, chat_manager: ChatManager) -> None:
        session = chat_manager.create_session("my_dialogue")

        m1 = chat_manager.add_message(session.id, "user", "What is transformer attention?")
        assert m1.role == "user"
        assert m1.content == "What is transformer attention?"

        m2 = chat_manager.add_message(
            session.id,
            "assistant",
            "Attention connects all tokens directly [Vaswani2017].",
            citations=["Vaswani2017"],
        )
        assert m2.role == "assistant"
        assert m2.citations == ["Vaswani2017"]

        messages = chat_manager.get_messages(session.id)
        assert len(messages) == 2
        assert messages[0].content == "What is transformer attention?"
        assert messages[1].content.startswith("Attention connects")

        # Check session message count updated
        updated_session = chat_manager.get_session(session.id)
        assert updated_session is not None
        assert updated_session.message_count == 2

    def test_export_to_markdown(self, chat_manager: ChatManager, tmp_path: Path) -> None:
        session = chat_manager.create_session("thesis_discussion")
        chat_manager.add_message(session.id, "user", "Explain my gap analysis.")
        chat_manager.add_message(session.id, "assistant", "The gap is in low-resource datasets.")

        out_file = tmp_path / "exported_chat.md"
        result_path = chat_manager.export_to_markdown(session.id, out_file)
        assert result_path.exists()

        content = result_path.read_text(encoding="utf-8")
        assert "# Chat Session: thesis_discussion" in content
        assert "Explain my gap analysis." in content
        assert "The gap is in low-resource datasets." in content

    def test_clear_messages(self, chat_manager: ChatManager) -> None:
        session = chat_manager.create_session("clear_test")
        chat_manager.add_message(session.id, "user", "Hello")
        assert len(chat_manager.get_messages(session.id)) == 1

        chat_manager.clear_messages(session.id)
        assert len(chat_manager.get_messages(session.id)) == 0
        # Session still exists
        assert chat_manager.get_session(session.id) is not None
