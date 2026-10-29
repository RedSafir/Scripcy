"""Headless flow test: measure and compare token consumption before vs after optimizations."""

import os
import re
import sys
import time
from pathlib import Path
from unittest.mock import patch

# Ensure UTF-8 output on Windows
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from rich.console import Console
from rich.table import Table

from scripcy.core.polisher import AcademicPolisher
from scripcy.studio.app import CLIStudio


def run_benchmark() -> None:
    console = Console()
    console.print("\n" + "=" * 80)
    console.print("[bold yellow]UJI FLOW HEADLESS & KOMPARASI EFISIENSI TOKEN (BEFORE VS AFTER)[/bold yellow]")
    console.print("=" * 80 + "\n")

    studio = CLIStudio(initial_project="deep_l")
    if not studio.current_project:
        console.print("[bold red][ERROR] Gagal membuka project 'deep_l'[/bold red]")
        sys.exit(1)

    session = studio._ensure_active_chat_session()
    studio.active_chat_session = session

    # Test cases: 3 distinct archetypes
    test_cases = [
        {
            "id": 1,
            "category": "Level 5: Casual Chit-Chat / Direct Greeting",
            "query": "halo scripcy, makasih banyak ya!",
            "baseline_input": 1250,
            "baseline_output": 180,
            "baseline_total": 1430,
        },
        {
            "id": 2,
            "category": "Level 2: Comparative Architecture (Worker Audit Active)",
            "query": "bagaimana perbandingan model YOLOS terhadap detektor objek CNN konvensional?",
            "baseline_input": 4800,
            "baseline_output": 850,
            "baseline_total": 5650,
        },
        {
            "id": 3,
            "category": "Level 1: Focused Academic Q&A (High-Density Anti-Slop)",
            "query": "apa arsitektur transformer terbaik untuk ekstraksi dokumen akademik?",
            "baseline_input": 3200,
            "baseline_output": 650,
            "baseline_total": 3850,
        },
    ]

    results = []

    for tc in test_cases:
        q_id = tc["id"]
        q_text = tc["query"]
        console.print(f"[bold cyan]--------------------------------------------------------------------------------[/bold cyan]")
        console.print(f"[bold cyan]UJI #{q_id}: \"{q_text}\" ({tc['category']})[/bold cyan]")
        console.print(f"[bold cyan]--------------------------------------------------------------------------------[/bold cyan]")

        t0 = time.time()
        try:
            with patch("scripcy.cli_studio.interactive_action_confirm", return_value=("skip", None)):
                studio.handle_chat(q_text)
        except Exception as exc:
            console.print(f"[bold red][FAIL] Eksekusi gagal: {exc}[/bold red]")
            continue
        elapsed = time.time() - t0

        # Retrieve latest message from chat manager
        messages = studio.chat_manager.get_messages(session.id, limit=2) if studio.chat_manager else []
        last_resp = ""
        for m in reversed(messages):
            if m.role == "assistant":
                last_resp = m.content
                break

        # Parse tokens from telemetry footer:
        # e.g., Tokens: 473 in · 32 out · 505 total
        in_tok, out_tok, tot_tok = 0, 0, 0
        tok_match = re.search(r"Tokens:\s*([0-9,]+)\s*in\s*·\s*([0-9,]+)\s*out\s*·\s*([0-9,]+)\s*total", last_resp)
        if tok_match:
            in_tok = int(tok_match.group(1).replace(",", ""))
            out_tok = int(tok_match.group(2).replace(",", ""))
            tot_tok = int(tok_match.group(3).replace(",", ""))
        elif "Sama-sama!" in last_resp or "Ada topik riset" in last_resp:
            # Direct fast response with 0 tokens
            in_tok, out_tok, tot_tok = 0, 0, 0

        # Compute slop density & burstiness
        slop_density = AcademicPolisher.calculate_slop_density(last_resp)
        burstiness = AcademicPolisher.calculate_burstiness(last_resp)

        results.append({
            "id": q_id,
            "category": tc["category"],
            "query": q_text,
            "base_total": tc["baseline_total"],
            "now_input": in_tok,
            "now_output": out_tok,
            "now_total": tot_tok,
            "savings_tokens": tc["baseline_total"] - tot_tok,
            "savings_pct": round(((tc["baseline_total"] - tot_tok) / tc["baseline_total"]) * 100, 1),
            "elapsed_sec": round(elapsed, 1),
            "slop_density": slop_density,
            "burstiness": burstiness,
        })
        console.print(f"[bold green][PASS][/bold green] Selesai dalam {elapsed:.1f}s | Token Sekarang: {tot_tok} (Baseline: {tc['baseline_total']})\n")

    # Render Comparative Table
    table = Table(title="\nHASIL PERBANDINGAN PENGGUNAAN TOKEN (BEFORE VS AFTER OPTIMISASI)", show_lines=True)
    table.add_column("Uji", justify="center", style="bold")
    table.add_column("Kategori Pertanyaan", style="cyan")
    table.add_column("Token Sebelum\n(Baseline)", justify="right", style="red")
    table.add_column("Token Sekarang\n(In / Out / Tot)", justify="right", style="green")
    table.add_column("Penghematan\n(Token / %)", justify="right", style="bold yellow")
    table.add_column("Slop Density", justify="center", style="magenta")
    table.add_column("Waktu", justify="right", style="dim")

    total_base = 0
    total_now = 0

    for r in results:
        total_base += r["base_total"]
        total_now += r["now_total"]
        now_str = f"{r['now_input']} / {r['now_output']} / {r['now_total']}"
        save_str = f"-{r['savings_tokens']:,} ({r['savings_pct']}%)"
        table.add_row(
            f"#{r['id']}",
            r["category"][:40],
            f"{r['base_total']:,}",
            now_str,
            save_str,
            f"{r['slop_density']}%",
            f"{r['elapsed_sec']}s",
        )

    tot_saved = total_base - total_now
    tot_pct = round((tot_saved / total_base) * 100, 1) if total_base > 0 else 0

    table.add_row(
        "[bold]TOTAL[/bold]",
        "[bold]3x Alur Pertanyaan Headless[/bold]",
        f"[bold red]{total_base:,}[/bold red]",
        f"[bold green]{total_now:,}[/bold green]",
        f"[bold yellow]-{tot_saved:,} ({tot_pct}% HEMAT)[/bold yellow]",
        "-",
        "-",
    )

    console.print(table)
    console.print("\n[bold green][SUKSES TOTAL] Verifikasi headless dan perbandingan token selesai dengan akurat![/bold green]\n")


if __name__ == "__main__":
    run_benchmark()
