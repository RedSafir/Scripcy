"""SQLite-backed multi-session chat manager for Scrpcy academic research studio."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Individual message in a conversation session."""

    id: int = Field(default=0)
    session_id: str
    role: str = Field(description="Role: user, assistant, or system.")
    content: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    citations: list[str] = Field(default_factory=list)


class ChatSession(BaseModel):
    """Metadata describing a chat session."""

    id: str
    name: str
    active_skill: str = Field(default="scrpcy_manual")
    status: str = Field(default="running", description="Status: running, detached, closed.")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    message_count: int = Field(default=0)


class ChatManager:
    """Manages chat sessions and messages stored inside a project's chats.db SQLite database."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection with foreign keys enabled."""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        """Create sessions and messages tables with indexes."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    active_skill TEXT NOT NULL DEFAULT 'scrpcy_manual',
                    status TEXT NOT NULL DEFAULT 'running',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    citations TEXT NOT NULL DEFAULT '[]',
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
            # Migration check for existing databases
            cur = conn.execute("PRAGMA table_info(sessions)")
            existing_cols = [row["name"] for row in cur.fetchall()]
            if "status" not in existing_cols:
                conn.execute("ALTER TABLE sessions ADD COLUMN status TEXT NOT NULL DEFAULT 'running'")

    def create_session(
        self,
        name: str | None = None,
        active_skill: str = "scrpcy_manual",
        status: str = "running",
    ) -> ChatSession:
        """Create a new chat session. Auto-generates 'session1', 'session2', etc. if name is omitted."""
        with self._get_connection() as conn:
            if not name or not name.strip():
                # Find next sequential session number
                cur = conn.execute("SELECT name FROM sessions WHERE name LIKE 'session%' OR name LIKE 'chat%'")
                existing_names = [row["name"] for row in cur.fetchall()]
                idx = 1
                while f"session{idx}" in existing_names or f"chat{idx}" in existing_names:
                    idx += 1
                clean_name = f"session{idx}"
            else:
                clean_name = name.strip()

            session_id = clean_name.lower().replace(" ", "_")
            now_str = datetime.now(timezone.utc).isoformat()

            # Ensure unique session_id if collision occurs
            orig_id = session_id
            counter = 1
            while True:
                cur = conn.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
                if not cur.fetchone():
                    break
                session_id = f"{orig_id}_{counter}"
                counter += 1

            conn.execute(
                """
                INSERT INTO sessions (id, name, active_skill, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, clean_name, active_skill, status, now_str, now_str),
            )
            conn.commit()

        return ChatSession(
            id=session_id,
            name=clean_name,
            active_skill=active_skill,
            status=status,
            created_at=now_str,
            updated_at=now_str,
            message_count=0,
        )

    def list_sessions(self) -> list[ChatSession]:
        """Return all chat sessions sorted by updated_at descending."""
        with self._get_connection() as conn:
            cur = conn.execute("""
                SELECT s.id, s.name, s.active_skill, s.status, s.created_at, s.updated_at,
                       COUNT(m.id) AS msg_count
                FROM sessions s
                LEFT JOIN messages m ON s.id = m.session_id
                GROUP BY s.id
                ORDER BY s.updated_at DESC
            """)
            rows = cur.fetchall()

        result: list[ChatSession] = []
        for r in rows:
            st = r["status"] if "status" in r.keys() and r["status"] else "running"
            result.append(ChatSession(
                id=r["id"],
                name=r["name"],
                active_skill=r["active_skill"],
                status=st,
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                message_count=r["msg_count"],
            ))
        return result

    def get_session(self, identifier: str | int) -> ChatSession | None:
        """Retrieve a session by ID, name, or 1-based index from list_sessions()."""
        sessions = self.list_sessions()
        if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
            idx = int(identifier) - 1
            if 0 <= idx < len(sessions):
                return sessions[idx]
            return None

        clean_id = str(identifier).strip().lower().replace(" ", "_")
        for s in sessions:
            if s.id.lower() == clean_id or s.name.lower() == str(identifier).strip().lower():
                return s
        return None

    def delete_session(self, identifier: str | int) -> bool:
        """Delete a chat session and all associated messages."""
        session = self.get_session(identifier)
        if not session:
            return False

        with self._get_connection() as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session.id,))
            conn.commit()
        return True

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: list[str] | None = None,
    ) -> ChatMessage:
        """Append a message to a session and touch the session updated_at timestamp."""
        now_str = datetime.now(timezone.utc).isoformat()
        citations_json = json.dumps(citations or [])

        with self._get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO messages (session_id, role, content, created_at, citations)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, role, content, now_str, citations_json),
            )
            msg_id = cur.lastrowid
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now_str, session_id),
            )
            conn.commit()

        return ChatMessage(
            id=msg_id or 0,
            session_id=session_id,
            role=role,
            content=content,
            created_at=now_str,
            citations=citations or [],
        )

    def get_messages(self, session_id: str, limit: int | None = None) -> list[ChatMessage]:
        """Retrieve chronological messages for a session, optionally limited to last N."""
        query = "SELECT id, session_id, role, content, created_at, citations FROM messages WHERE session_id = ? ORDER BY id ASC"
        params: list[Any] = [session_id]

        if limit is not None and limit > 0:
            query = f"""
                SELECT * FROM (
                    SELECT id, session_id, role, content, created_at, citations
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                ) ORDER BY id ASC
            """
            params = [session_id, limit]

        with self._get_connection() as conn:
            cur = conn.execute(query, params)
            rows = cur.fetchall()

        result: list[ChatMessage] = []
        for r in rows:
            try:
                cites = json.loads(r["citations"])
            except Exception:
                cites = []
            result.append(ChatMessage(
                id=r["id"],
                session_id=r["session_id"],
                role=r["role"],
                content=r["content"],
                created_at=r["created_at"],
                citations=cites,
            ))
        return result

    def clear_messages(self, session_id: str) -> None:
        """Delete all messages inside a session without deleting the session itself."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.commit()


    def update_session_status(self, session_id: str, status: str) -> None:
        """Update lifecycle status for a session (running, detached, closed)."""
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE sessions SET status = ?, updated_at = ? WHERE id = ?",
                (status, now_str, session_id),
            )
            conn.commit()

    def get_active_sessions_count(self) -> tuple[int, int]:
        """Return count of (active_or_detached, closed) sessions."""
        sessions = self.list_sessions()
        active_cnt = sum(1 for s in sessions if s.status in ("running", "detached"))
        closed_cnt = sum(1 for s in sessions if s.status == "closed")
        return active_cnt, closed_cnt

    def update_session_skill(self, session_id: str, skill_name: str) -> None:
        """Update active persona skill bound to a session."""
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE sessions SET active_skill = ?, updated_at = ? WHERE id = ?",
                (skill_name, now_str, session_id),
            )
            conn.commit()

    def export_to_markdown(self, session_id: str, target_file: Path | str) -> Path:
        """Export session dialogue to clean, readable Markdown."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Chat session '{session_id}' not found.")

        messages = self.get_messages(session.id)
        target_path = Path(target_file).expanduser().resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = [
            f"# Chat Session: {session.name}",
            f"- **Date**: {session.created_at}",
            f"- **Persona**: `{session.active_skill}`",
            f"- **Total Exchanges**: {len(messages)}",
            "",
            "---",
            "",
        ]

        for msg in messages:
            role_header = "**User**" if msg.role == "user" else f"**AI ({session.active_skill})**"
            lines.append(f"### {role_header}")
            lines.append(msg.content.strip())
            if msg.citations:
                lines.append("")
                lines.append(f"*Citations*: {', '.join(msg.citations)}")
            lines.append("")
            lines.append("---")
            lines.append("")

        target_path.write_text("\n".join(lines), encoding="utf-8")
        return target_path
