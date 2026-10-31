"""Unit tests for TaskManager zero-latency headless pass-through."""

import time
from scripcy.core.tasks import TaskManager
from scripcy.core.workspace import WorkspaceManager


def test_request_checkpoint_headless_pass_through(temp_workspace: WorkspaceManager):
    tm = TaskManager(db_path=temp_workspace.root_dir / ".scripcy" / "tasks.db")
    # Even with a 300s timeout, under pytest environment it must return immediately with default without blocking
    start = time.time()
    choice = tm.request_checkpoint(
        task_id="t1",
        prompt_data={"prompt": "Should we continue?", "default_choice": "proceed"},
        timeout=300.0,
    )
    elapsed = time.time() - start
    assert choice == "proceed"
    assert elapsed < 0.5  # Zero-latency
