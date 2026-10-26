"""Background Task Manager with DAG dependency barriers, two-phase execution, and SQLite WAL persistence."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
import asyncio
import inspect
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Callable

from pydantic import BaseModel, Field


TERMINAL_TASK_STATUSES = ("completed", "failed", "cancelled", "interrupted")
ACTIVE_TASK_STATUSES = ("waiting_deps", "queued", "running")


class TaskInfo(BaseModel):
    """Metadata, dependency graph links, and runtime status for a background job."""

    id: str = Field(..., description="Unique task identifier, e.g. task-1.")
    name: str = Field(..., description="Human-readable description of the operation.")
    task_type: str = Field(..., description="Category: scout, download, convert, gdrive, web, docx, chat, agent.")
    session_id: str | None = Field(default=None, description="Optional chat session ID bound to this task.")
    lane: str = Field(
        default="io",
        description="Execution lane: 'turn' (serialized per session) or 'io' (parallel background I/O).",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="List of upstream task IDs that must reach terminal state before this task starts.",
    )
    dedup_key: str | None = Field(
        default=None,
        description="Optional key for promise coalescing to avoid duplicate concurrent I/O tasks.",
    )
    ingested_stems: list[str] = Field(
        default_factory=list,
        description="Vault document stems produced or ingested by this task.",
    )
    status: str = Field(
        default="running",
        description="Status: waiting_deps, queued, running, completed, failed, cancelled, interrupted.",
    )
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
    def is_active(self) -> bool:
        """Return True if task is waiting on dependencies, queued, or actively running."""
        return self.status in ACTIVE_TASK_STATUSES

    @property
    def is_terminal(self) -> bool:
        """Return True if task has finished execution."""
        return self.status in TERMINAL_TASK_STATUSES

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


class CheckpointRequest:
    """Thread-safe request from background agent loop to UI main thread."""

    def __init__(self, task_id: str, prompt_data: dict[str, Any]) -> None:
        self.task_id = task_id
        self.prompt_data = prompt_data
        self.response: Any = None
        self.done_event = threading.Event()


class TaskManager:
    """Manages DAG-coordinated background tasks, promise barriers, and SQLite WAL persistence."""

    def __init__(self, db_path: Path | str | None = None, max_workers: int = 8) -> None:
        self.db_path = Path(db_path).expanduser().resolve() if db_path else None
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="scripcy-worker")
        self._tasks: dict[str, TaskInfo] = {}
        self._futures: dict[str, Future[Any]] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._done_events: dict[str, threading.Event] = {}
        self._pending_specs: dict[str, Callable[[], None]] = {}
        self._completion_callbacks: list[Callable[[TaskInfo], None]] = []
        self._lock = threading.RLock()
        self._db_write_lock = threading.Lock()
        self._counter = 1
        self._checkpoints: dict[str, CheckpointRequest] = {}
        if self.db_path:
            self.set_db_path(self.db_path)

    def set_db_path(self, db_path: Path | str) -> None:
        """Set SQLite database path for task persistence and load existing tasks."""
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._load_from_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA busy_timeout = 5000")
        except Exception:
            pass
        return conn

    def _init_db(self) -> None:
        if not self.db_path:
            return
        with self._db_write_lock:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS tasks (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        task_type TEXT NOT NULL,
                        session_id TEXT,
                        lane TEXT DEFAULT 'io',
                        depends_on TEXT DEFAULT '[]',
                        ingested_stems TEXT DEFAULT '[]',
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
                cur = conn.execute("PRAGMA table_info(tasks)")
                cols = {row["name"] for row in cur.fetchall()}
                if "tokens_burned" not in cols:
                    conn.execute("ALTER TABLE tasks ADD COLUMN tokens_burned INTEGER DEFAULT 0")
                if "lane" not in cols:
                    conn.execute("ALTER TABLE tasks ADD COLUMN lane TEXT DEFAULT 'io'")
                if "depends_on" not in cols:
                    conn.execute("ALTER TABLE tasks ADD COLUMN depends_on TEXT DEFAULT '[]'")
                if "ingested_stems" not in cols:
                    conn.execute("ALTER TABLE tasks ADD COLUMN ingested_stems TEXT DEFAULT '[]'")
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
                    if status in ACTIVE_TASK_STATUSES:
                        status = "failed"
                        err = "Studio exited while task was running."

                    logs = []
                    if r["logs"]:
                        try:
                            logs = json.loads(r["logs"])
                        except Exception:
                            logs = [r["logs"]]

                    depends_on = []
                    if "depends_on" in r.keys() and r["depends_on"]:
                        try:
                            depends_on = json.loads(r["depends_on"])
                        except Exception:
                            depends_on = []

                    ingested_stems = []
                    if "ingested_stems" in r.keys() and r["ingested_stems"]:
                        try:
                            ingested_stems = json.loads(r["ingested_stems"])
                        except Exception:
                            ingested_stems = []

                    lane = r["lane"] if ("lane" in r.keys() and r["lane"]) else ("turn" if r["task_type"] in ("chat", "agent") else "io")

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
                        lane=lane,
                        depends_on=depends_on,
                        ingested_stems=ingested_stems,
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
                    done_ev = threading.Event()
                    if info.is_terminal:
                        done_ev.set()
                    self._done_events[t_id] = done_ev

                self._counter = max_num + 1

    def _persist_task(self, info: TaskInfo) -> None:
        if not self.db_path:
            return
        try:
            with self._db_write_lock:
                with self._get_connection() as conn:
                    conn.execute("""
                        INSERT INTO tasks (
                            id, name, task_type, session_id, lane, depends_on, ingested_stems,
                            status, progress, started_at, finished_at, logs, error, result_summary, tokens_burned
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            status=excluded.status,
                            progress=excluded.progress,
                            finished_at=excluded.finished_at,
                            logs=excluded.logs,
                            error=excluded.error,
                            result_summary=excluded.result_summary,
                            tokens_burned=excluded.tokens_burned,
                            depends_on=excluded.depends_on,
                            ingested_stems=excluded.ingested_stems
                    """, (
                        info.id,
                        info.name,
                        info.task_type,
                        info.session_id,
                        info.lane,
                        json.dumps(info.depends_on),
                        json.dumps(info.ingested_stems),
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

    def _get_unmet_dependencies(self, dep_ids: list[str]) -> list[str]:
        """Return list of task IDs from dep_ids that have not yet reached a terminal state."""
        unmet: list[str] = []
        with self._lock:
            for d_id in dep_ids:
                dep_task = self._tasks.get(d_id)
                if dep_task and not dep_task.is_terminal:
                    unmet.append(d_id)
        return unmet

    def _promote_ready_dependents(self) -> None:
        """Check all waiting_deps tasks and dispatch those whose dependencies have all completed."""
        to_launch: list[tuple[str, Callable[[], None]]] = []
        with self._lock:
            for tid, launch_fn in list(self._pending_specs.items()):
                task = self._tasks.get(tid)
                if not task or task.status != "waiting_deps":
                    self._pending_specs.pop(tid, None)
                    continue
                unmet = self._get_unmet_dependencies(task.depends_on)
                if not unmet:
                    self._pending_specs.pop(tid, None)
                    task.status = "running"
                    task.progress = "Starting task..."
                    task.started_at = datetime.now(timezone.utc).isoformat()
                    self._persist_task(task)
                    to_launch.append((tid, launch_fn))
                else:
                    task.progress = f"Menunggu selesai: {', '.join(unmet)}"

        for tid, launch_fn in to_launch:
            fut = self._executor.submit(launch_fn)
            with self._lock:
                self._futures[tid] = fut

    def submit(
        self,
        name: str,
        task_type: str,
        target_fn: Callable[..., Any],
        *args: Any,
        task_id: str | None = None,
        session_id: str | None = None,
        depends_on: list[str] | None = None,
        lane: str | None = None,
        dedup_key: str | None = None,
        auto_depend_session_io: bool = False,
        auto_serialize_turn: bool = False,
        **kwargs: Any,
    ) -> TaskInfo:
        """Submit a background job with optional DAG dependencies and promise coalescing.

        Tasks with unmet `depends_on` enter `waiting_deps` state without consuming a worker thread
        slot until all upstream dependencies finish.
        """
        resolved_lane = lane or ("turn" if task_type in ("chat", "agent") else "io")

        with self._lock:
            # Reuse an active task when an identical dedup_key is already running in this session.
            if dedup_key:
                for existing in self._tasks.values():
                    if (
                        existing.dedup_key == dedup_key
                        and existing.session_id == session_id
                        and existing.is_active
                    ):
                        return existing

            if not task_id:
                task_id = f"task-{self._counter}"
                self._counter += 1

            dep_list: list[str] = list(depends_on or [])
            if session_id and (auto_depend_session_io or auto_serialize_turn):
                for existing in self._tasks.values():
                    if existing.id == task_id or existing.session_id != session_id or not existing.is_active:
                        continue
                    if auto_depend_session_io and existing.lane == "io" and existing.id not in dep_list:
                        dep_list.append(existing.id)
                    if auto_serialize_turn and existing.lane == "turn" and existing.id not in dep_list:
                        dep_list.append(existing.id)

            unmet_deps = self._get_unmet_dependencies(dep_list)

            cancel_event = threading.Event()
            done_event = threading.Event()
            self._cancel_events[task_id] = cancel_event
            self._done_events[task_id] = done_event

            initial_status = "waiting_deps" if unmet_deps else "running"
            initial_progress = (
                f"Menunggu selesai: {', '.join(unmet_deps)}"
                if unmet_deps
                else "Starting task..."
            )

            info = TaskInfo(
                id=task_id,
                name=name,
                task_type=task_type,
                session_id=session_id,
                lane=resolved_lane,
                depends_on=dep_list,
                dedup_key=dedup_key,
                status=initial_status,
                progress=initial_progress,
            )
            self._tasks[task_id] = info
            self._persist_task(info)

        def _worker_wrapper() -> None:
            if cancel_event.is_set():
                with self._lock:
                    info.status = "cancelled"
                    info.progress = "Cancelled by user."
                    info.finished_at = datetime.now(timezone.utc).isoformat()
                    self._persist_task(info)
                done_event.set()
                self._promote_ready_dependents()
                return

            self.log(task_id, f"Started task: '{name}'")
            try:
                sig_params = inspect.signature(target_fn).parameters.values()
                call_kwargs = dict(kwargs)
                if "cancel_event" in call_kwargs or any(param.name == "cancel_event" for param in sig_params):
                    call_kwargs["cancel_event"] = cancel_event
                if "task_id" not in call_kwargs and any(param.name == "task_id" for param in sig_params):
                    call_kwargs["task_id"] = task_id

                result = target_fn(*args, **call_kwargs)

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
            except (KeyboardInterrupt, asyncio.CancelledError):
                with self._lock:
                    info.status = "cancelled"
                    info.progress = "Cancelled by user."
                    info.finished_at = datetime.now(timezone.utc).isoformat()
                    self._persist_task(info)
                self.log(task_id, "Task was cancelled.")
            except Exception as exc:
                with self._lock:
                    info.status = "failed"
                    info.error = str(exc) or type(exc).__name__
                    info.progress = f"Failed: {exc or type(exc).__name__}"
                    info.finished_at = datetime.now(timezone.utc).isoformat()
                    self._persist_task(info)
                self.log(task_id, f"Task encountered error: {exc}")
            finally:
                done_event.set()
                self._promote_ready_dependents()
                with self._lock:
                    callbacks = list(self._completion_callbacks)
                for cb in callbacks:
                    try:
                        cb(info)
                    except Exception:
                        pass

        with self._lock:
            # Re-verify unmet_deps under lock in case a dependency finished while creating wrapper
            still_unmet = self._get_unmet_dependencies(info.depends_on)
            if still_unmet:
                self._pending_specs[task_id] = _worker_wrapper
                return info
            else:
                info.status = "running"
                info.progress = "Starting task..."
                self._persist_task(info)

        future = self._executor.submit(_worker_wrapper)
        with self._lock:
            self._futures[task_id] = future

        return info

    def wait_or_promote(self, task_id: str, wait_ms: int = 2000) -> tuple[bool, TaskInfo | None]:
        """Two-phase execution (`WaitMsBeforeAsync`): wait briefly for fast completion, else promote to background.

        Returns `(completed_synchronously, task_info)`.
        """
        with self._lock:
            done_ev = self._done_events.get(task_id)
        if not done_ev:
            return True, self.get_task(task_id)

        finished = done_ev.wait(timeout=max(0.0, wait_ms / 1000.0))
        return finished, self.get_task(task_id)

    def wait_for_tasks(
        self,
        task_ids: list[str],
        timeout: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> bool:
        """Deterministically wait for all specified `task_ids` to reach a terminal state without busy-spinning."""
        if not task_ids:
            return True

        deadline = (time.monotonic() + timeout) if timeout is not None else None
        for tid in task_ids:
            with self._lock:
                done_ev = self._done_events.get(tid)
                t_info = self._tasks.get(tid)
            if not done_ev or (t_info and t_info.is_terminal):
                continue

            while not done_ev.is_set():
                if cancel_event and cancel_event.is_set():
                    return False
                rem = (deadline - time.monotonic()) if deadline is not None else 0.1
                if deadline is not None and rem <= 0:
                    return False
                done_ev.wait(timeout=min(0.1, max(0.01, rem)))
        return True

    def submit_coalesced_batch(
        self,
        io_specs: list[dict[str, Any]],
        batch_barrier_name: str,
        batch_barrier_fn: Callable[..., Any],
        session_id: str | None = None,
        downstream_specs: list[dict[str, Any]] | None = None,
    ) -> tuple[list[TaskInfo], TaskInfo, list[TaskInfo]]:
        """Coalesce parallel I/O tasks into a single CPU batch barrier, queuing downstream subagents in waiting_deps.

        Returns `(io_tasks, barrier_task, downstream_tasks)`.
        """
        io_tasks: list[TaskInfo] = []
        for spec in io_specs:
            t_info = self.submit(
                name=spec.get("name", "Background I/O"),
                task_type=spec.get("task_type", "scout"),
                target_fn=spec["target_fn"],
                session_id=session_id,
                lane="io",
                dedup_key=spec.get("dedup_key"),
            )
            if t_info.id not in [t.id for t in io_tasks]:
                io_tasks.append(t_info)

        io_ids = [t.id for t in io_tasks]
        barrier_task = self.submit(
            name=batch_barrier_name,
            task_type="convert",
            target_fn=batch_barrier_fn,
            session_id=session_id,
            lane="cpu",
            depends_on=io_ids,
            dedup_key=f"batch_barrier:{session_id}:{','.join(sorted(io_ids))}" if io_ids else None,
        )

        downstream_tasks: list[TaskInfo] = []
        for d_spec in downstream_specs or []:
            d_task = self.submit(
                name=d_spec.get("name", "Subagent Worker"),
                task_type=d_spec.get("task_type", "agent"),
                target_fn=d_spec["target_fn"],
                session_id=session_id,
                lane=d_spec.get("lane", "io"),
                depends_on=[barrier_task.id],
                dedup_key=d_spec.get("dedup_key"),
            )
            downstream_tasks.append(d_task)

        return io_tasks, barrier_task, downstream_tasks

    def wait_for_session_io(
        self,
        session_id: str,
        exclude_task_id: str | None = None,
        timeout: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> list[TaskInfo]:
        """Wait for all active I/O tasks (`scout`, `download`, `web`, `gdrive`) in `session_id` to finish."""
        with self._lock:
            io_ids = [
                t.id for t in self._tasks.values()
                if t.session_id == session_id
                and t.id != exclude_task_id
                and t.lane == "io"
                and t.is_active
            ]
        if io_ids:
            self.wait_for_tasks(io_ids, timeout=timeout, cancel_event=cancel_event)
        with self._lock:
            return [self._tasks[tid] for tid in io_ids if tid in self._tasks]

    def record_ingested_stems(self, task_id: str, stems: list[str]) -> None:
        """Record document stems ingested by a task for downstream consumers."""
        if not stems:
            return
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                for s in stems:
                    if s and s not in task.ingested_stems:
                        task.ingested_stems.append(s)
                self._persist_task(task)

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

    def list_active_tasks_for_session(self, session_id: str, lane: str | None = None) -> list[TaskInfo]:
        """Return active (`waiting_deps`, `queued`, `running`) tasks for a session, optionally filtered by lane."""
        with self._lock:
            tasks = [
                t for t in self._tasks.values()
                if t.session_id == session_id
                and t.is_active
                and (lane is None or t.lane == lane)
            ]
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

    def update_progress(self, task_id: str, message: str) -> None:
        """Update progress string for an active task without adding a log entry."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
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
        """Signal cancellation for a running or waiting task."""
        target = self.get_task(identifier)
        if not target:
            return False

        cancelled_now = False
        with self._lock:
            self._pending_specs.pop(target.id, None)
            req = self._checkpoints.get(target.id)
            if req:
                req.response = "cancelled"
                req.done_event.set()
            event = self._cancel_events.get(target.id)
            if event:
                event.set()

            future = self._futures.get(target.id)
            if future and not future.done():
                future.cancel()

            if target.is_active:
                target.status = "cancelled"
                target.progress = "Cancelled by user."
                target.finished_at = datetime.now(timezone.utc).isoformat()
                self.log(target.id, "Cancellation requested by user.")
                self._persist_task(target)
                cancelled_now = True

            done_ev = self._done_events.get(target.id)
            if done_ev:
                done_ev.set()

        if cancelled_now:
            self._promote_ready_dependents()
        return cancelled_now

    cancel = cancel_task

    def cancel_session_tasks(self, session_id: str) -> list[str]:
        """Cancel all active tasks associated with a specific session."""
        with self._lock:
            active_ids = [
                t.id for t in self._tasks.values()
                if t.session_id == session_id and t.is_active
            ]
        cancelled = []
        for tid in active_ids:
            if self.cancel_task(tid):
                cancelled.append(tid)
        return cancelled

    def clear_completed(self) -> int:
        """Remove finished or cancelled tasks from memory and database."""
        with self._lock:
            to_remove = [
                tid for tid, t in self._tasks.items()
                if t.is_terminal
            ]
            for tid in to_remove:
                del self._tasks[tid]
                self._futures.pop(tid, None)
                self._cancel_events.pop(tid, None)
                self._done_events.pop(tid, None)
                self._pending_specs.pop(tid, None)
            if self.db_path and to_remove:
                try:
                    with self._db_write_lock:
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

    def request_checkpoint(self, task_id: str, prompt_data: dict[str, Any], timeout: float = 300.0) -> Any:
        """Called by background worker thread to pause and ask user for decision."""
        req = CheckpointRequest(task_id, prompt_data)
        with self._lock:
            self._checkpoints[task_id] = req
        signaled = req.done_event.wait(timeout=timeout)
        with self._lock:
            self._checkpoints.pop(task_id, None)
        if not signaled:
            return None
        return req.response

    def get_pending_checkpoint(self, task_id: str) -> CheckpointRequest | None:
        """Called by main UI thread to inspect if worker is waiting for decision."""
        with self._lock:
            return self._checkpoints.get(task_id)

    def resolve_checkpoint(self, task_id: str, response: Any) -> bool:
        """Called by main UI thread to provide user decision and unblock worker."""
        with self._lock:
            req = self._checkpoints.get(task_id)
            if req:
                req.response = response
                req.done_event.set()
                return True
        return False
