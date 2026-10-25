"""Unit tests for SQLite-backed ChatManager."""

from __future__ import annotations

from pathlib import Path
import threading
import time

import pytest

from scrpcy.core.chats import ChatManager, SessionEvent
from scrpcy.core.tasks import TaskManager


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


class TestTaskDAGAndSessionActorRuntime:
    """Tests for DAG task coordination, zero-starvation readiness queue, and SessionRuntime actor isolation."""

    def test_dag_depends_on_without_thread_starvation(self, tmp_path: Path) -> None:
        # Even with max_workers=1, a dependent task submitted before/after upstream completes cleanly
        # because waiting_deps tasks do not occupy a ThreadPoolExecutor slot.
        tm = TaskManager(max_workers=1, db_path=tmp_path / "tasks.db")
        execution_order: list[str] = []
        gate = threading.Event()

        def worker_a(task_id: str) -> str:
            gate.wait(timeout=2.0)
            execution_order.append("A")
            tm.record_ingested_stems(task_id, ["paper_alpha"])
            return "A done"

        def worker_b(task_id: str) -> str:
            execution_order.append("B")
            return "B done"

        task_a = tm.submit(name="Scout A", task_type="scout", target_fn=worker_a, session_id="s1", lane="io")
        task_b = tm.submit(
            name="Synthesize B",
            task_type="chat",
            target_fn=worker_b,
            session_id="s1",
            lane="turn",
            depends_on=[task_a.id],
        )

        # Before gate is released, task_b must be in waiting_deps
        assert tm.get_task(task_b.id).status == "waiting_deps"
        gate.set()

        finished = tm.wait_for_tasks([task_a.id, task_b.id], timeout=5.0)
        assert finished is True
        assert execution_order == ["A", "B"]
        assert tm.get_task(task_a.id).ingested_stems == ["paper_alpha"]
        assert tm.get_task(task_b.id).status == "completed"

    def test_auto_depend_session_io_and_serialized_turns(self, tmp_path: Path) -> None:
        tm = TaskManager(max_workers=4, db_path=tmp_path / "tasks.db")
        events: list[str] = []

        io_gate = threading.Event()

        def io_worker(task_id: str, label: str) -> str:
            io_gate.wait(timeout=3.0)
            events.append(label)
            return label

        def turn_worker(task_id: str, label: str) -> str:
            events.append(label)
            return label

        io1 = tm.submit("IO 1", "scout", io_worker, session_id="sess1", lane="io", label="io1")
        io2 = tm.submit("IO 2", "download", io_worker, session_id="sess1", lane="io", label="io2")

        turn1 = tm.submit(
            "Turn 1",
            "chat",
            turn_worker,
            session_id="sess1",
            lane="turn",
            auto_depend_session_io=True,
            auto_serialize_turn=True,
            label="turn1",
        )
        turn2 = tm.submit(
            "Turn 2",
            "chat",
            turn_worker,
            session_id="sess1",
            lane="turn",
            auto_depend_session_io=True,
            auto_serialize_turn=True,
            label="turn2",
        )

        assert set(turn1.depends_on) == {io1.id, io2.id}
        assert turn1.id in turn2.depends_on

        # Release IO workers now that dependencies are verified
        io_gate.set()

        tm.wait_for_tasks([io1.id, io2.id, turn1.id, turn2.id], timeout=5.0)
        # Both io1 and io2 must complete before turn1, and turn1 before turn2
        assert events.index("io1") < events.index("turn1")
        assert events.index("io2") < events.index("turn1")
        assert events.index("turn1") < events.index("turn2")

    def test_promise_coalescing_dedup_key(self, tmp_path: Path) -> None:
        tm = TaskManager(max_workers=2, db_path=tmp_path / "tasks.db")
        gate = threading.Event()
        call_count = 0

        def slow_scout(task_id: str) -> str:
            nonlocal call_count
            call_count += 1
            gate.wait(timeout=2.0)
            return "ok"

        t1 = tm.submit("Scout transformer", "scout", slow_scout, dedup_key="scout:transformer")
        t2 = tm.submit("Scout transformer duplicate", "scout", slow_scout, dedup_key="scout:transformer")

        assert t1.id == t2.id
        gate.set()
        tm.wait_for_tasks([t1.id], timeout=3.0)
        assert call_count == 1

    def test_wait_or_promote_two_phase_execution(self, tmp_path: Path) -> None:
        tm = TaskManager(max_workers=2, db_path=tmp_path / "tasks.db")
        gate = threading.Event()

        def fast_fn(task_id: str) -> str:
            return "fast"

        def slow_fn(task_id: str) -> str:
            gate.wait(timeout=2.0)
            return "slow"

        fast_task = tm.submit("Fast", "scout", fast_fn)
        done_sync, info_fast = tm.wait_or_promote(fast_task.id, wait_ms=1000)
        assert done_sync is True
        assert info_fast.status == "completed"

        slow_task = tm.submit("Slow", "scout", slow_fn)
        done_sync_slow, info_slow = tm.wait_or_promote(slow_task.id, wait_ms=50)
        assert done_sync_slow is False
        assert info_slow.status in ("running", "pending")
        gate.set()
        tm.wait_for_tasks([slow_task.id], timeout=3.0)

    def test_session_runtime_isolation_and_inbox(self, chat_manager: ChatManager) -> None:
        s1 = chat_manager.create_session("riset_a")
        s2 = chat_manager.create_session("riset_b")

        rt1 = chat_manager.get_runtime(s1.id, initial_documents={"doc_shared"})
        rt2 = chat_manager.get_runtime(s2.id, initial_documents={"doc_shared"})

        # Mutating active documents in s1 does not bleed into s2
        rt1.add_documents(["doc_only_s1"])
        assert "doc_only_s1" in rt1.snapshot_documents()
        assert "doc_only_s1" not in rt2.snapshot_documents()

        # Turn-boundary inbox & output ring buffer isolation
        rt1.push_event(
            SessionEvent(
                event_type="task_completed",
                task_id="task-01",
                task_type="scout",
                task_name="Scout S1",
                status="completed",
                summary="2 papers",
                ingested_stems=["paper_s1_new"],
            )
        )
        rt1.append_output("Background notification for S1")

        assert len(rt2.drain_inbox()) == 0
        assert len(rt2.drain_output()) == 0

        drained_events = rt1.drain_inbox()
        assert len(drained_events) == 1
        assert drained_events[0].ingested_stems == ["paper_s1_new"]
        assert "paper_s1_new" in rt1.snapshot_documents()

        drained_out = rt1.drain_output()
        assert drained_out == ["Background notification for S1"]
        assert rt1.drain_output() == []

    def test_studio_detached_output_arbitration_and_reattach_flush(self, tmp_path: Path) -> None:
        import io
        from rich.console import Console
        from scrpcy.cli_studio import CLIStudio
        from scrpcy.core.config import ScrpcyConfig
        from scrpcy.core.project import ProjectManager
        from scrpcy.core.workspace import WorkspaceManager

        ws = WorkspaceManager(tmp_path / "ws")
        pm = ProjectManager(registry_file=tmp_path / "reg.json")
        proj = pm.create_project("riset_ui", ws.root_dir)
        cfg = ScrpcyConfig(workspace_path=str(ws.root_dir))
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=True, color_system=None)

        studio = CLIStudio(workspace=ws, config=cfg, console=console, project_manager=pm)
        studio.open_project(proj)

        s1 = studio.chat_manager.create_session("sesi_alpha")
        s2 = studio.chat_manager.create_session("sesi_beta")

        # User is currently attached to s2 in chat mode; s1 is detached
        studio.chat_manager.update_session_status(s1.id, "detached")
        s1.status = "detached"
        studio.active_chat_session = s2
        studio.sub_mode = "chat"

        before_len = len(buf.getvalue())
        studio.emit_session_output(s1.id, "Notifikasi background untuk sesi_alpha", session_obj=s1)
        after_len = len(buf.getvalue())

        # Must NOT bleed into s2's terminal view
        assert after_len == before_len
        rt1 = studio.get_session_runtime(s1.id)
        assert rt1 is not None

        # Now user switches/attaches to s1 and drains buffer
        studio.active_chat_session = s1
        studio.chat_manager.update_session_status(s1.id, "running")
        s1.status = "running"
        studio._drain_and_render_session_buffer(s1.id)

        assert "Notifikasi background untuk sesi_alpha" in buf.getvalue()


