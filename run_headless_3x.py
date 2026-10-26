"""Headless Test Flow for Scripcy with 3 Sequential Questions."""

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

from scripcy.cli_studio import CLIStudio
from scripcy.core.chats import ChatManager
from scripcy.core.config import ConfigManager
from scripcy.core.project import ProjectManager
from scripcy.core.workspace import WorkspaceManager


def run_test_flow(
    project_name: str = "project_eval_3x",
    session_name: str = "eval_flow_session",
    queries: list[str] | None = None,
) -> dict:
    if queries is None:
        queries = [
            "Carikan aku data terbaru luas kebakaran hutan dan lahan (karhutla) di Indonesia serta provinsi yang paling terdampak menurut data resmi pemerintah.",
            "Berdasarkan data yang sudah ada di vault, bandingkan tingkat kerentanan karhutla di lahan gambut vs mineral dan faktor pemicu utamanya.",
            "Bagaimana evaluasi efektivitas kebijakan restorasi gambut dan sistem deteksi dini (early warning) BMKG/KLHK, serta apa saran riset lanjutannya?",
        ]

    console = Console(file=sys.stdout, force_terminal=True, legacy_windows=False)
    pm = ProjectManager()

    # Get existing project or create new one
    proj = pm.get_project(project_name)
    if not proj:
        proj = pm.create_project(project_name)
    base_dir = proj.resolved_path

    wm = WorkspaceManager(base_dir)
    cm = ConfigManager()

    studio = CLIStudio(workspace=wm, config=cm.config, console=console, project_manager=pm)
    studio.open_project(proj)

    # Open session in chat_manager
    session = studio.chat_manager.get_session(session_name)
    if not session:
        session = studio.chat_manager.create_session(session_name)
    studio.cmd_chat(session.name)

    # Check existing conversation history in this session
    existing_msgs = studio.chat_manager.get_messages(session.id)
    existing_user_queries = [m.content.strip() for m in existing_msgs if m.role == "user"]

    query_results = []
    total_t0 = time.time()

    # Patch interactive confirms so headless run executes completely autonomously
    with patch(
        "scripcy.studio.agent_loop.interactive_scout_confirm",
        side_effect=lambda suggested_topic, **kw: ("yes", suggested_topic),
    ):
        for idx, q in enumerate(queries, 1):
            console.print(f"\n[bold bright_cyan]════════════════════════════════════════════════════════════════[/bold bright_cyan]")
            console.print(f"[bold bright_yellow]▶ PERTANYAAN #{idx}:[/bold bright_yellow] {q}")
            console.print(f"[bold bright_cyan]════════════════════════════════════════════════════════════════[/bold bright_cyan]\n")

            # Check if this exact query was already answered in the session
            matched_msg_idx = None
            for m_idx, m in enumerate(existing_msgs):
                if m.role == "user" and m.content.strip() == q.strip():
                    matched_msg_idx = m_idx
                    break

            if matched_msg_idx is not None and matched_msg_idx + 1 < len(existing_msgs):
                ans_msg = existing_msgs[matched_msg_idx + 1]
                if ans_msg.role == "assistant":
                    console.print(f"[bold green][✓] Pertanyaan #{idx} telah selesai diproses di sesi ini. Memuat hasil respons...[/bold green]\n")
                    vault_docs = [f for f in wm.vault_dir.iterdir() if f.is_file()] if wm.vault_dir.is_dir() else []
                    query_results.append({
                        "query_index": idx,
                        "query": q,
                        "elapsed_seconds": 0.0,
                        "response_length": len(ans_msg.content),
                        "response_preview": ans_msg.content[:600] + ("..." if len(ans_msg.content) > 600 else ""),
                        "full_response": ans_msg.content,
                        "vault_count": len(vault_docs),
                    })
                    continue

            t_start = time.time()
            studio.handle_chat(q)
            t_elapsed = round(time.time() - t_start, 2)

            # Wait for any active background tasks initiated by this turn
            if hasattr(studio, "task_manager"):
                active_tasks = [t for t in studio.task_manager.list_tasks() if t.status == "running"]
                if active_tasks:
                    console.print(f"[dim]Menunggu {len(active_tasks)} tugas latar belakang selesai...[/dim]")
                    studio.task_manager.wait_all(timeout=60.0)

            # Fetch the latest assistant message
            chat_mgr = ChatManager(base_dir / "chats.db")
            msgs = chat_mgr.get_messages(session.id)
            assistant_msgs = [m for m in msgs if m.role == "assistant"]
            latest_assistant = assistant_msgs[-1].content if assistant_msgs else ""

            # Check vault status
            vault_docs = [f for f in wm.vault_dir.iterdir() if f.is_file()] if wm.vault_dir.is_dir() else []

            # Match events for this query turn
            events_file = base_dir / "logs" / "events.jsonl"
            render_events = []
            subagent_events = []
            contract_events = []
            if events_file.exists():
                for line in events_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            ev = json.loads(line)
                            if ev.get("event") == "RENDER_COMPLETE":
                                render_events.append(ev)
                            elif ev.get("event") == "SUBAGENTS":
                                subagent_events.append(ev)
                            elif ev.get("event") == "TASK_CONTRACT":
                                contract_events.append(ev)
                        except Exception:
                            pass
            latest_render = render_events[-1] if render_events and len(render_events) >= idx else (render_events[-1] if render_events else {})
            latest_subagent = subagent_events[-1] if subagent_events and len(subagent_events) >= idx else (subagent_events[-1] if subagent_events else {})
            latest_contract = contract_events[-1] if contract_events and len(contract_events) >= idx else (contract_events[-1] if contract_events else {})

            sub_contracts = latest_contract.get("sub_contracts", [])
            roles_executed = latest_subagent.get("roles_executed", [])
            contracts_count = len(sub_contracts) if sub_contracts else len(roles_executed)

            query_results.append({
                "query_index": idx,
                "query": q,
                "elapsed_seconds": t_elapsed,
                "response_length": len(latest_assistant),
                "response_preview": latest_assistant[:600] + ("..." if len(latest_assistant) > 600 else ""),
                "full_response": latest_assistant,
                "vault_count": len(vault_docs),
                "contracts_count": contracts_count,
                "roles_executed": roles_executed,
                "facts_extracted_count": latest_subagent.get("facts_extracted_count", 0),
                "new_stems_count": latest_subagent.get("new_stems_count", 0),
                "input_tokens": latest_render.get("input_tokens", 0),
                "output_tokens": latest_render.get("output_tokens", 0),
                "total_tokens": latest_render.get("total_tokens", 0),
                "grounding_score": latest_render.get("grounding_score", 0.0),
                "critic_score": latest_render.get("critic_score", 0),
            })

    total_elapsed = round(time.time() - total_t0, 2)

    # Final summary of vault and events
    vault_docs = [f for f in wm.vault_dir.iterdir() if f.is_file()] if wm.vault_dir.is_dir() else []
    events_file = base_dir / "logs" / "events.jsonl"
    events = []
    if events_file.exists():
        for line in events_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass

    summary = {
        "project": project_name,
        "session": session_name,
        "total_elapsed_seconds": total_elapsed,
        "total_queries": len(queries),
        "query_results": query_results,
        "final_vault_documents": [d.name for d in vault_docs],
        "total_events_logged": len(events),
        "events_summary": [e.get("event") for e in events],
    }

    report_path = base_dir / "headless_eval_report.json"
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[bold green][✓] Laporan evaluasi headless disimpan di: {report_path}[/bold green]\n")

    return summary


if __name__ == "__main__":
    p_name = sys.argv[1] if len(sys.argv) > 1 else "project_eval_3x"
    res = run_test_flow(project_name=p_name)
    print("\n" + "=" * 60)
    print("HASIL AKHIR TEST FLOW HEADLESS (3X PERTANYAAN)")
    print("=" * 60)
    print(f"Project         : {res['project']}")
    print(f"Durasi Total    : {res['total_elapsed_seconds']} detik")
    print(f"Dokumen di Vault: {len(res['final_vault_documents'])} file ({res['final_vault_documents']})")
    print(f"Total Event     : {res['total_events_logged']}")
    for qr in res["query_results"]:
        print(f"\n[Pertanyaan #{qr['query_index']}] Durasi: {qr['elapsed_seconds']}s | Small Contracts: {qr.get('contracts_count', 0)} | Roles: {qr.get('roles_executed', [])} | Vault: {qr['vault_count']} docs")
        critic_display = f"{qr.get('critic_score')}/10" if qr.get('critic_score') else "Dilewati (One-Pass Golden Pass)"
        print(f"Token: {qr.get('input_tokens', 0)} in · {qr.get('output_tokens', 0)} out · {qr.get('total_tokens', 0)} total | Grounding: {int(qr.get('grounding_score', 0.0) * 100)}% | Critic: {critic_display}")
        print(f"Preview Respons : {qr['response_preview'][:250]}...")
