"""
Test Flow Headless:
1. Uji Flow 5 Jenis Pertanyaan (Verifikasi Akurasi Dispatch Promisor ke Masing-Masing Komponen).
2. Uji Flow 3x Pertanyaan Riset Headless di Project 'deep_l'.
"""

import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from rich.console import Console

from scripcy.core.promisor import Promisor
from scripcy.studio.app import CLIStudio


def run_test_5_dispatch_types() -> bool:
    console = Console()
    console.print("\n" + "=" * 80)
    console.print("[bold yellow]BAGIAN 1: UJI FLOW 5 JENIS PERTANYAAN (ENTRY-POINT DISPATCHER)[/bold yellow]")
    console.print("=" * 80 + "\n")

    promisor = Promisor()

    test_cases = [
        {
            "category": "Level 5: Direct Fast Reply (Casual Chit-chat / Greetings)",
            "query": "halo scripcy, makasih banyak ya!",
            "expected_entry": "direct",
            "expected_system_tool": False,
        },
        {
            "category": "Level 4: Worker (System Tool / File Error / Local Inspection)",
            "query": "apakah ada file yang error atau rusak di vault?",
            "expected_entry": "worker",
            "expected_system_tool": True,
        },
        {
            "category": "Level 3: Manager (Revision / Patching Draft / Session Memory)",
            "query": "perbaiki respon sebelumnya, tambahkan tabel komparasi latency",
            "expected_entry": "manager",
            "expected_system_tool": False,
        },
        {
            "category": "Level 2: Examiner (Hypothesis Testing / Claim Verification)",
            "query": "apakah benar bahwa model YOLOS selalu kalah cepat dari YOLOv8 pada raspberry pi?",
            "expected_entry": "examiner",
            "expected_system_tool": False,
        },
        {
            "category": "Level 1: Contractor (Deep Literature Research / Research Gap)",
            "query": "carikan aku gap penelitian deep learning terbaru",
            "expected_entry": "contractor",
            "expected_system_tool": False,
        },
    ]

    all_passed = True
    for idx, tc in enumerate(test_cases, 1):
        decision = promisor.evaluate(tc["query"])
        passed = (decision.entry_point == tc["expected_entry"]) and (decision.is_system_tool == tc["expected_system_tool"])

        if passed:
            console.print(f"[bold green][PASS][/bold green] - [cyan]{tc['category']}[/cyan]")
            console.print(f"    Query       : \"{tc['query']}\"")
            console.print(f"    Entry Point : [bold cyan]{decision.entry_point.upper()}[/bold cyan] (Alasan: {decision.reason})")
            if decision.direct_reply:
                console.print(f"    Direct Reply: [dim]\"{decision.direct_reply}\" (0 token LLM)[/dim]")
        else:
            console.print(f"[bold red][FAIL][/bold red] - {tc['category']}")
            console.print(f"    Expected: {tc['expected_entry']} | Actual: {decision.entry_point}")
            all_passed = False
        console.print("-" * 60)

    return all_passed


def run_test_3_headless_flow() -> bool:
    console = Console()
    console.print("\n" + "=" * 80)
    console.print("[bold yellow]BAGIAN 2: UJI FLOW 3X PERTANYAAN HEADLESS DI PROJECT 'deep_l'[/bold yellow]")
    console.print("=" * 80 + "\n")

    studio = CLIStudio(initial_project="deep_l")
    if not studio.current_project:
        console.print("[red][FAIL] Gagal membuka project 'deep_l'[/red]")
        return False

    session = studio._ensure_active_chat_session()
    studio.active_chat_session = session

    queries = [
        "carikan aku gap penelitian deep l terbaru",
        "apa arsitektur transformer terbaik untuk ekstraksi dokumen akademik?",
        "bagaimana perbandingan model YOLOS terhadap detektor objek CNN konvensional?",
    ]

    for idx, q in enumerate(queries, 1):
        console.print(f"\n[bold cyan]--------------------------------------------------------------------------------[/bold cyan]")
        console.print(f"[bold cyan]PERTANYAAN #{idx}: \"{q}\"[/bold cyan]")
        console.print(f"[bold cyan]--------------------------------------------------------------------------------[/bold cyan]")

        t0 = time.time()
        studio.handle_chat(q)
        elapsed = time.time() - t0

        console.print(f"\n[bold green][✓] Pertanyaan #{idx} SELESAI dalam {elapsed:.1f} detik![/bold green]")

    return True


if __name__ == "__main__":
    p_ok = run_test_5_dispatch_types()
    if not p_ok:
        print("[FAIL] Uji 5 jenis dispatch gagal.")
        sys.exit(1)

    print("\n[OK] Bagian 1 selesai sukses. Melanjutkan ke Bagian 2 (Flow 3x Pertanyaan Headless)...\n")
    h_ok = run_test_3_headless_flow()
    if not h_ok:
        print("[FAIL] Uji headless flow gagal.")
        sys.exit(1)

    print("\n[SUKSES TOTAL] Seluruh pengujian flow 5 dispatch dan 3x pertanyaan headless berhasil!")
