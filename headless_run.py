"""Headless Flow Runner for Scrpcy End-to-End Automated Testing."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console

from scrpcy.cli_studio import CLIStudio
from scrpcy.core.chats import ChatManager
from scrpcy.core.config import ConfigManager
from scrpcy.core.project import ProjectManager
from scrpcy.core.workspace import WorkspaceManager


def run_headless_flow(
    project_name: str,
    session_name: str,
    query: str,
    auto_approve_scout: bool = True,
) -> dict:
    console = Console(file=sys.stdout, force_terminal=True, legacy_windows=False)
    base_dir = Path.cwd() / project_name
    base_dir.mkdir(parents=True, exist_ok=True)

    wm = WorkspaceManager(base_dir)
    cm = ConfigManager()
    pm = ProjectManager()

    # Create or get project
    proj = pm.get_project(project_name)
    if not proj:
        proj = pm.create_project(project_name, base_dir)

    studio = CLIStudio(workspace=wm, config=cm.config, console=console, project_manager=pm)
    studio.open_project(proj)

    # Open session in chat_manager
    session = studio.chat_manager.get_session(session_name)
    if not session:
        session = studio.chat_manager.create_session(session_name)
    studio.cmd_chat(session.name)

    # Patch interactive confirms so headless run executes completely autonomously
    with patch("scrpcy.studio.agent_loop.interactive_scout_confirm", return_value=("yes", query)):
        t0 = time.time()
        studio.handle_chat(query)
        elapsed = time.time() - t0

    # Read output results from chats.db
    chat_mgr = ChatManager(base_dir / "chats.db")
    messages = chat_mgr.get_messages(session.id)
    last_assistant_msg = next((m for m in reversed(messages) if m.role == "assistant"), None)

    # Read events.jsonl
    events_file = base_dir / "logs" / "events.jsonl"
    events = []
    if events_file.exists():
        for line in events_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass

    return {
        "project": project_name,
        "session": session_name,
        "query": query,
        "elapsed_seconds": round(elapsed, 2),
        "total_messages": len(messages),
        "last_response": last_assistant_msg.content if last_assistant_msg else "(Tidak ada respon)",
        "citations": last_assistant_msg.citations if last_assistant_msg else [],
        "events_count": len(events),
        "events": events,
    }


if __name__ == "__main__":
    p_name = sys.argv[1] if len(sys.argv) > 1 else "riset_stunting_indonesia"
    s_name = sys.argv[2] if len(sys.argv) > 2 else "sesi_data_faktual"
    test_query = (
        sys.argv[3]
        if len(sys.argv) > 3
        else "Berapa angka prevalensi stunting di Indonesia tahun 2023 berdasarkan data resmi Kemenkes dan BPS?"
    )

    print(f"=== Menjalankan Headless Flow Scrpcy ===")
    print(f"Project : {p_name}")
    print(f"Session : {s_name}")
    print(f"Query   : {test_query}\n")

    result = run_headless_flow(p_name, s_name, test_query)

    print("\n=== HASIL EKSEKUSI HEADLESS ===")
    print(f"Durasi Total : {result['elapsed_seconds']} detik")
    print(f"Pesan Sesi   : {result['total_messages']} pesan")
    print(f"Total Event  : {result['events_count']} event tercatat di events.jsonl")
    print(f"Event Types  : {[e.get('event') for e in result['events']]}")
    print("\n--- RESPON AKHIR ASISTEN ---")
    print(result["last_response"])
    print("\n--- SITASI TERVERIFIKASI ---")
    print(result["citations"])
