"""Background Task Manager for non-blocking asynchronous operations in Scrpcy."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
import threading
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, Field


class TaskInfo(BaseModel):
    """Metadata and runtime status for a background job."""

    id: str = Field(..., description="Unique task identifier, e.g. task-1.")
    name: str = Field(..., description="Human-readable description of the operation.")
    task_type: str = Field(..., description="Category: scout, download, convert, gdrive, agent.")
    session_id: str | None = Field(default=None, description="Optional chat session ID bound to this task.")
    status: str = Field(default="running", description="Status: running, completed, failed, cancelled.")
    progress: str = Field(default="In progress...", description="Latest progress message.")
    started_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 start timestamp.",
    )
    finished_at: str | None = Field(default=None, description="ISO 8601 completion timestamp.")
    logs: list[str] = Field(default_factory=list, description="Logged messages during execution.")
    error: str | None = Field(default=None, description="Error message if failed.")
    result_summary: str | None = Field(default=None, description="Short summary of task result.")

    @property
    def duration_seconds(self) -> float:
        """Calculate elapsed or total runtime in seconds."""
        try:
            start_dt = datetime.fromisoformat(self.started_at)
            end_dt = (
                datetime.fromisoformat(self.finished_at)
                if self.finished_at
                else datetime.now(timezone.utc)
            )
            return max(0.0, (end_dt - start_dt).total_seconds())
        except Exception:
            return 0.0


class TaskManager:
    """Manages background threads, logging, cancellation, and task status."""

    def __init__(self, max_workers: int = 4) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="scrpcy-worker")
        self._tasks: dict[str, TaskInfo] = {}
        self._futures: dict[str, Future[Any]] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._completion_callbacks: list[Callable[[TaskInfo], None]] = []
        self._lock = threading.RLock()
        self._counter = 1

    def submit(
        self,
        name: str,
        task_type: str,
        target_fn: Callable[..., Any],
        *args: Any,
        task_id: str | None = None,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> TaskInfo:
        """Submit a background job to run asynchronously without freezing Scrpcy."""
        with self._lock:
            if not task_id:
                task_id = f"task-{self._counter}"
                self._counter += 1

            cancel_event = threading.Event()
            self._cancel_events[task_id] = cancel_event

            info = TaskInfo(
                id=task_id,
                name=name,
                task_type=task_type,
                session_id=session_id,
                status="running",
                progress="Starting task...",
            )
            self._tasks[task_id] = info

        def _worker_wrapper() -> None:
            self.log(task_id, f"Started task: '{name}'")
            try:
                # Pass cancel_event and task_id to target_fn if it accepts them
                sig_params = __import__("inspect").signature(target_fn).parameters.values()
                if "cancel_event" in kwargs or any(param.name == "cancel_event" for param in sig_params):
                    kwargs["cancel_event"] = cancel_event
                if "task_id" not in kwargs and any(param.name == "task_id" for param in sig_params):
                    kwargs["task_id"] = task_id

                result = target_fn(*args, **kwargs)

                with self._lock:
                    if cancel_event.is_set():
                        info.status = "cancelled"
                        info.progress = "Cancelled by user."
                        self.log(task_id, "Task was cancelled.")
                    else:
                        info.status = "completed"
                        info.progress = "Finished successfully."
                        info.result_summary = str(result)[:300] if result is not None else "Done"
                        self.log(task_id, "Task completed successfully.")
                    info.finished_at = datetime.now(timezone.utc).isoformat()
            except Exception as exc:
                with self._lock:
                    info.status = "failed"
                    info.error = str(exc)
                    info.progress = f"Failed: {exc}"
                    info.finished_at = datetime.now(timezone.utc).isoformat()
                self.log(task_id, f"Task encountered error: {exc}")
            finally:
                with self._lock:
                    callbacks = list(self._completion_callbacks)
                for cb in callbacks:
                    try:
                        cb(info)
                    except Exception:
                        pass

        future = self._executor.submit(_worker_wrapper)
        with self._lock:
            self._futures[task_id] = future

        return info


    def add_completion_callback(self, callback: Callable[[TaskInfo], None]) -> None:
        """Register a callback invoked whenever any background task completes, fails, or is cancelled."""
        with self._lock:
            if callback not in self._completion_callbacks:
                self._completion_callbacks.append(callback)

    def list_tasks_for_session(self, session_id: str) -> list[TaskInfo]:
        """Return all tasks ordered newest first that belong to a specific chat session."""
        with self._lock:
            tasks = [t for t in self._tasks.values() if t.session_id == session_id]
        tasks.sort(key=lambda t: t.started_at, reverse=True)
        return tasks

    def log(self, task_id: str, message: str) -> None:
        """Append a log entry with timestamp to task logs."""
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.logs.append(f"[{timestamp}] {message}")
                task.progress = message

    def list_tasks(self, limit: int = 50) -> list[TaskInfo]:
        """Return all tasks ordered newest first."""
        with self._lock:
            tasks = list(self._tasks.values())
        tasks.sort(key=lambda t: t.started_at, reverse=True)
        return tasks[:limit]

    def get_task(self, identifier: str | int) -> TaskInfo | None:
        """Retrieve task by ID or 1-based index from list_tasks()."""
        all_tasks = self.list_tasks()
        if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
            idx = int(identifier) - 1
            if 0 <= idx < len(all_tasks):
                return all_tasks[idx]
            return None

        clean_id = str(identifier).strip().lower()
        with self._lock:
            for tid, t in self._tasks.items():
                if tid.lower() == clean_id:
                    return t
        return None

    def cancel_task(self, identifier: str | int) -> bool:
        """Signal cancellation for a running task."""
        target = self.get_task(identifier)
        if not target:
            return False

        with self._lock:
            event = self._cancel_events.get(target.id)
            if event:
                event.set()

            future = self._futures.get(target.id)
            if future and not future.done():
                future.cancel()

            if target.status == "running":
                target.status = "cancelled"
                target.progress = "Cancelled by user."
                target.finished_at = datetime.now(timezone.utc).isoformat()
                self.log(target.id, "Cancellation requested by user.")
                return True
        return False

    def clear_completed(self) -> int:
        """Remove finished or cancelled tasks from memory."""
        with self._lock:
            to_remove = [
                tid for tid, t in self._tasks.items()
                if t.status in ("completed", "failed", "cancelled")
            ]
            for tid in to_remove:
                del self._tasks[tid]
                self._futures.pop(tid, None)
                self._cancel_events.pop(tid, None)
            return len(to_remove)

    def shutdown(self, wait: bool = False) -> None:
        """Shut down background executor."""
        self._executor.shutdown(wait=wait, cancel_futures=True)
