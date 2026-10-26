"""Unit tests for background TaskManager."""

from __future__ import annotations

import time

import pytest

from scripcy.core.tasks import TaskManager


@pytest.fixture
def task_manager() -> TaskManager:
    """Fixture providing a fresh TaskManager."""
    tm = TaskManager(max_workers=2)
    yield tm
    tm.shutdown(wait=False)


class TestTaskManager:
    """Tests for background task submission, logging, status, and cancellation."""

    def test_submit_and_complete_task(self, task_manager: TaskManager) -> None:
        def quick_job(x: int, y: int) -> int:
            return x + y

        task = task_manager.submit(
            name="Addition Job",
            task_type="calc",
            target_fn=quick_job,
            x=10,
            y=20,
        )
        assert task.id.startswith("task-")
        assert task.name == "Addition Job"

        # Wait for completion
        time.sleep(0.15)
        updated = task_manager.get_task(task.id)
        assert updated is not None
        assert updated.status == "completed"
        assert updated.result_summary == "30"

    def test_task_logging(self, task_manager: TaskManager) -> None:
        def logged_job() -> str:
            return "done"

        task = task_manager.submit("Logged Task", "log_test", logged_job)
        task_manager.log(task.id, "Step 1 started")
        task_manager.log(task.id, "Step 2 completed")

        time.sleep(0.1)
        updated = task_manager.get_task(task.id)
        assert updated is not None
        logs = "\n".join(updated.logs)
        assert "Step 1 started" in logs
        assert "Step 2 completed" in logs

    def test_cancel_task(self, task_manager: TaskManager) -> None:
        def long_job(cancel_event=None) -> None:
            for _ in range(20):
                if cancel_event and cancel_event.is_set():
                    break
                time.sleep(0.05)

        task = task_manager.submit("Long Job", "long", long_job)
        time.sleep(0.05)
        cancelled = task_manager.cancel_task(task.id)
        assert cancelled is True

        time.sleep(0.1)
        updated = task_manager.get_task(task.id)
        assert updated is not None
        assert updated.status == "cancelled"

    def test_clear_completed_tasks(self, task_manager: TaskManager) -> None:
        task_manager.submit("Fast 1", "test", lambda: 1)
        task_manager.submit("Fast 2", "test", lambda: 2)
        time.sleep(0.15)

        cleared = task_manager.clear_completed()
        assert cleared == 2
        assert len(task_manager.list_tasks()) == 0

    def test_persistence_sqlite(self, tmp_path: Path) -> None:
        db_path = tmp_path / "tasks.db"
        tm1 = TaskManager(db_path=db_path)
        t = tm1.submit("Persistent Task", "persist", lambda: "success result")
        time.sleep(0.15)
        tm1.shutdown(wait=False)

        # Create new manager pointing to the same db
        tm2 = TaskManager(db_path=db_path)
        loaded = tm2.get_task(t.id)
        assert loaded is not None
        assert loaded.name == "Persistent Task"
        assert loaded.status == "completed"
        assert loaded.result_summary == "success result"
        tm2.shutdown(wait=False)

    def test_update_tokens_and_display(self, task_manager: TaskManager) -> None:
        task = task_manager.submit("Token Task", "chat", lambda: "ok")
        assert task.tokens_burned == 0
        assert task.display_tokens == 0

        task_manager.update_tokens(task.id, 1450)
        updated = task_manager.get_task(task.id)
        assert updated is not None
        assert updated.tokens_burned == 1450
        assert updated.display_tokens == 1450

    def test_display_tokens_from_legacy_log(self, task_manager: TaskManager) -> None:
        task = task_manager.submit("Legacy Log Task", "chat", lambda: "ok")
        task_manager.log(task.id, "Token: 382 in · 1,009 out · 1,391 total (gemini-2.5-flash) • 12.1s")
        updated = task_manager.get_task(task.id)
        assert updated is not None
        assert updated.display_tokens == 1391
