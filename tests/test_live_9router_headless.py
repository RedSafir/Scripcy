"""Live end-to-end headless synthesis test using running 9Router.

Tests:
1. Real LLM inference via 9Router (ag/gemini-3.8-flash-medium).
2. Dual-tier vault: literature in vault/Downloads vs local researcher notes in USER INPUT HERE.
3. Cohesive narrative synthesis (Anti fact-stacking).
4. In-text APA citation linking & bibliography exclusion for researcher's data.
"""

import sys
import shutil
import asyncio
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console

from scripcy.core.config import ConfigManager
from scripcy.core.workspace import WorkspaceManager
from scripcy.core.vault import VaultManager
from scripcy.core.engine import LLMEngine
from scripcy.cli_studio import CLIStudio
from scripcy.core.project import ProjectInfo, ProjectManager


def run_live_test():
    console = Console(force_terminal=True, legacy_windows=False)
    console.print("\n[bold cyan]================================================================[/bold cyan]")
    console.print("[bold cyan]       UJI LIVE HEADLESS 9ROUTER DENGAN DUAL-TIER VAULT         [/bold cyan]")
    console.print("[bold cyan]================================================================[/bold cyan]\n")

    # 1. Workspace Setup
    test_dir = Path("projects") / "_test_live_9router"
    if test_dir.exists():
        shutil.rmtree(test_dir)

    ws = WorkspaceManager(test_dir)
    ws.ensure_directories()
    vm = VaultManager(workspace=ws)

    # 2. Input Paper to Downloads (External Literature)
    paper_path = ws.downloads_dir / "deepseek_v3_report.txt"
    paper_path.write_text(
        "Title: DeepSeek-V3 Technical Report\n"
        "Authors: DeepSeek-AI, Haowei Zhang, Damai Dai, Dejian Yang, Zhenda Xie\n"
        "Year: 2024\n"
        "Abstract:\n"
        "We present DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters "
        "with 37B activated for each token. DeepSeek-V3 pioneers an auxiliary-loss-free strategy for load balancing "
        "and sets a multi-token prediction training objective for stronger performance.\n\n"
        "Key Findings:\n"
        "1. DeepSeekMoE architecture employs 256 routed experts and 1 shared expert.\n"
        "2. Traditional auxiliary loss degrades model performance by penalizing optimal routing representations; "
        "DeepSeek-V3's auxiliary-loss-free load balancing dynamically adjusts expert bias terms to ensure balanced routing "
        "without compromising semantic representation.\n"
        "3. FP8 mixed precision training prevents numerical instability, eliminating irrecoverable loss spikes across 14.8T tokens.\n",
        encoding="utf-8"
    )

    # 3. Input Researcher's Local Data to USER INPUT HERE
    user_file = ws.user_input_dir / "my_custom_moe_training_log.txt"
    user_file.write_text(
        "Catatan Eksperimen Riset Lokal Peneliti:\n"
        "Model: Mini-MoE Prototyping (Total 8 expert, 2 aktif per token)\n"
        "Hasil Observasi Training 60 Epoch:\n"
        "- Epoch 1-25: Menggunakan penalti auxiliary loss standar. Terjadi routing collapse di mana Expert #2 dan #5 "
        "menerima 78% beban token, menyebabkan loss tersangkut di 2.45 dan degradasi kapasitas representasi.\n"
        "- Epoch 26-60: Beralih mengadopsi mekanisme bias balancing tanpa auxiliary loss (mengikuti prinsip DeepSeekMoE). "
        "Beban antar 8 expert merata dengan deviasi di bawah 6%. Loss turun drastis dan stabil konvergen di 1.08 tanpa spike.\n"
        "- Kesimpulan Peneliti: Mekanisme bias adjustment dinamis terbukti mengatasi bottle-neck saturasi expert lokal.\n",
        encoding="utf-8"
    )

    console.print("[bold cyan]●[/bold cyan] [bold bright_white]Sinkronisasi Vault & Konversi Markdown...[/bold bright_white]")
    vm.scan_local_staging()
    vm.reload_vault()

    # 4. Configure Studio & 9Router Engine
    cm = ConfigManager()
    cfg = cm.config
    cfg.default_model = "9router/ag/gemini-3.8-flash-medium"
    cfg.contract_unlimited_tokens = True

    engine = LLMEngine(config=cfg)

    proj_mgr = ProjectManager(registry_file=test_dir / "reg.json")
    proj = proj_mgr.create_project("test_live_9r", test_dir)
    proj.model = "9router/ag/gemini-3.8-flash-medium"

    studio = CLIStudio(
        workspace=ws,
        config=cfg,
        console=console,
        project_manager=proj_mgr,
        engine=engine,
    )
    studio.current_project = proj
    studio.sub_mode = "chat"
    studio.vault_manager = vm

    query = (
        "Bandingkan mekanisme load balancing DeepSeek-V3 pada literatur dengan hasil eksperimen "
        "training Mini-MoE saya di catatan riset. Bagaimana mekanisme bias adjustment tersebut "
        "menyelesaikan masalah routing collapse yang saya alami?"
    )

    console.print(f"\n[bold cyan]●[/bold cyan] [bold bright_white]Menjalankan Sintesis Live ke 9Router ({cfg.default_model})...[/bold bright_white]")
    console.print(f"[dim]Kueri: {query}[/dim]\n")

    studio.handle_chat(query)

    console.print("\n[bold green]✔ Pengujian Selesai![/bold green]\n")

    # Clean up
    try:
        shutil.rmtree(test_dir)
    except Exception:
        pass


if __name__ == "__main__":
    run_live_test()
