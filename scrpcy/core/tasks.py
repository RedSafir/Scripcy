"""Background Task Manager for non-blocking asynchronous operations in Scrpcy."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
import threading
import json
from pathlib import Path
import sqlite3
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
    tokens_burned: int = Field(default=0, description="Total tokens consumed by the task.")

    @property
    def result(self) -> str | None:
        """Alias for result_summary for backward and API compatibility."""
        return self.result_summary

    @result.setter
    def result(self, value: str | None) -> None:
        self.result_summary = value

    @property
    def display_tokens(self) -> int:
        """Return explicit burned tokens or parse total tokens from execution logs."""
        if self.tokens_burned:
            return self.tokens_burned
        import re
        for entry in self.logs:
            entry_s = str(entry).lower()
            if "total" in entry_s and ("token" in entry_s or "in ·" in entry_s or "in •" in entry_s or "in *" in entry_s):
                m = re.search(r"(\d[\d,]*)\s*total", entry_s)
                if m:
                    try:
                        return int(m.group(1).replace(",", ""))
                    except Exception:
                        pass
        return 0

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
    """Manages background threads, logging, cancellation, and task status with SQLite persistence."""

    def __init__(self, db_path: Path | str | None = None, max_workers: int = 4) -> None:
        self.db_path = Path(db_path).expanduser().resolve() if db_path else None
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="scrpcy-worker")
        self._tasks: dict[str, TaskInfo] = {}
        self._futures: dict[str, Future[Any]] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._completion_callbacks: list[Callable[[TaskInfo], None]] = []
        self._lock = threading.RLock()
        self._counter = 1
        if self.db_path:
            self.set_db_path(self.db_path)

    def set_db_path(self, db_path: Path | str) -> None:
        """Set SQLite database path for task persistence and load existing tasks."""
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._load_from_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        if not self.db_path:
            return
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    session_id TEXT,
                    status TEXT NOT NULL,
                    progress TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    logs TEXT,
                    error TEXT,
                    result_summary TEXT,
                    tokens_burned INTEGER DEFAULT 0
                )
            """)
            try:
                conn.execute("ALTER TABLE tasks ADD COLUMN tokens_burned INTEGER DEFAULT 0")
            except Exception:
                pass
            conn.commit()

    def _load_from_db(self) -> None:
        if not self.db_path:
            return
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM tasks ORDER BY started_at ASC")
            rows = cur.fetchall()
            max_num = 0
            with self._lock:
                for r in rows:
                    t_id = r["id"]
                    if t_id.startswith("task-"):
                        try:
                            num = int(t_id[5:])
                            if num > max_num:
                                max_num = num
                        except ValueError:
                            pass

                    status = r["status"]
                    err = r["error"]
                    if status == "running":
                        status = "failed"
                        err = "Studio exited while task was running."

                    logs = []
                    if r["logs"]:
                        try:
                            logs = json.loads(r["logs"])
                        except Exception:
                            logs = [r["logs"]]

                    tokens_burned = 0
                    if "tokens_burned" in r.keys() and r["tokens_burned"] is not None:
                        try:
                            tokens_burned = int(r["tokens_burned"])
                        except Exception:
                            tokens_burned = 0

                    info = TaskInfo(
                        id=t_id,
                        name=r["name"],
                        task_type=r["task_type"],
                        session_id=r["session_id"],
                        status=status,
                        progress=r["progress"] or "",
                        started_at=r["started_at"],
                        finished_at=r["finished_at"],
                        logs=logs,
                        error=err,
                        result_summary=r["result_summary"],
                        tokens_burned=tokens_burned,
                    )
                    self._tasks[t_id] = info

                self._counter = max_num + 1

    def _persist_task(self, info: TaskInfo) -> None:
        if not self.db_path:
            return
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO tasks (
                        id, name, task_type, session_id, status, progress,
                        started_at, finished_at, logs, error, result_summary, tokens_burned
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        status=excluded.status,
                        progress=excluded.progress,
                        finished_at=excluded.finished_at,
                        logs=excluded.logs,
                        error=excluded.error,
                        result_summary=excluded.result_summary,
                        tokens_burned=excluded.tokens_burned
                """, (
                    info.id,
                    info.name,
                    info.task_type,
                    info.session_id,
                    info.status,
                    info.progress,
                    info.started_at,
                    info.finished_at,
                    json.dumps(info.logs),
                    info.error,
                    info.result_summary,
                    getattr(info, "tokens_burned", 0),
                ))
                conn.commit()
        except Exception:
            pass

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
            self._persist_task(info)

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
                    self._persist_task(info)
            except Exception as exc:
                with self._lock:
                    info.status = "failed"
                    info.error = str(exc)
                    info.progress = f"Failed: {exc}"
                    info.finished_at = datetime.now(timezone.utc).isoformat()
                    self._persist_task(info)
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
                self._persist_task(task)

    def update_tokens(self, task_id: str, tokens: int) -> None:
        """Update total tokens consumed by a task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.tokens_burned = max(0, int(tokens))
                self._persist_task(task)

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
                self._persist_task(target)
                return True
        return False

    cancel = cancel_task

    def clear_completed(self) -> int:
        """Remove finished or cancelled tasks from memory and database."""
        with self._lock:
            to_remove = [
                tid for tid, t in self._tasks.items()
                if t.status in ("completed", "failed", "cancelled", "interrupted")
            ]
            for tid in to_remove:
                del self._tasks[tid]
                self._futures.pop(tid, None)
                self._cancel_events.pop(tid, None)
            if self.db_path and to_remove:
                try:
                    with self._get_connection() as conn:
                        q = f"DELETE FROM tasks WHERE id IN ({','.join(['?']*len(to_remove))})"
                        conn.execute(q, to_remove)
                        conn.commit()
                except Exception:
                    pass
            return len(to_remove)

    def shutdown(self, wait: bool = False) -> None:
        """Shut down background executor."""
        self._executor.shutdown(wait=wait, cancel_futures=True)
