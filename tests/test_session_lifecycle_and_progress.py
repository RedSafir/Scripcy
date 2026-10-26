"""Unit tests verifying session lifecycle binding and timeout progress checks."""

import asyncio
import time
from unittest.mock import MagicMock
import pytest

from scripcy.core.tasks import TaskManager, TaskInfo


def test_cancel_session_tasks():
    tm = TaskManager()

    def dummy_worker(cancel_event):
        while not cancel_event.is_set():
            time.sleep(0.01)

    t1 = tm.submit(name="Task Session A 1", task_type="agent", target_fn=dummy_worker, session_id="sess_a")
    t2 = tm.submit(name="Task Session A 2", task_type="scout", target_fn=dummy_worker, session_id="sess_a")
    t3 = tm.submit(name="Task Session B", task_type="agent", target_fn=dummy_worker, session_id="sess_b")

    time.sleep(0.05)
    assert t1.status == "running"
    assert t2.status == "running"
    assert t3.status == "running"

    cancelled = tm.cancel_session_tasks("sess_a")
    time.sleep(0.05)

    assert set(cancelled) == {t1.id, t2.id}
    assert tm.get_task(t1.id).status == "cancelled"
    assert tm.get_task(t2.id).status == "cancelled"
    # Task in sess_b must remain running!
    assert tm.get_task(t3.id).status == "running"

    tm.cancel_task(t3.id)
    tm.shutdown()


def test_cancel_session_tasks_ignores_completed():
    tm = TaskManager()

    def quick_worker():
        return "Done"

    t1 = tm.submit(name="Quick Task", task_type="agent", target_fn=quick_worker, session_id="sess_x")
    time.sleep(0.05)
    assert t1.status == "completed"

    cancelled = tm.cancel_session_tasks("sess_x")
    assert cancelled == []
    assert tm.get_task(t1.id).status == "completed"

    tm.shutdown()


@pytest.mark.asyncio
async def test_progress_checkpoint_no_timeout_kill():
    tm = TaskManager()

    def slow_worker(cancel_event, task_id):
        tm.log(task_id, "Step 1: Inisialisasi analisis")
        time.sleep(0.1)
        tm.log(task_id, "Step 2: Menghitung skor dialektis")
        time.sleep(0.1)
        return "Kalkulasi selesai"

    task = tm.submit(name="Analysis Task", task_type="agent", target_fn=slow_worker, session_id="sess_test")

    start_t = time.monotonic()
    last_checkpoint_t = start_t
    checkpoint_interval = 0.05  # Short interval for test

    completed_task = None
    while True:
        t_info = tm.get_task(task.id)
        if not t_info:
            break
        if t_info.status in ("completed", "failed", "cancelled"):
            completed_task = t_info
            break
        fut = tm._futures.get(task.id)
        if fut and fut.done():
            completed_task = tm.get_task(task.id)
            break

        now = time.monotonic()
        if now - last_checkpoint_t >= checkpoint_interval:
            last_checkpoint_t = now
            elapsed = int((now - start_t) * 1000)
            tm.log(task.id, f"[Pemeriksaan Progres {elapsed}ms] Status: {t_info.progress}")

        await asyncio.sleep(0.02)

    assert completed_task is not None
    assert completed_task.status == "completed"
    assert any("[Pemeriksaan Progres" in l for l in completed_task.logs)
    assert completed_task.result_summary == "Kalkulasi selesai"

    tm.shutdown()
