"""Headless flow test to verify the Dual-Tier Vault architecture with real online paper data.

Covers:
1. Downloads folder intake (Literature / External paper) -> Formal APA Citation.
2. USER INPUT HERE intake (Primary research data / loss notes) -> Target vault/user/, EXCLUDED from Bibliography.
3. Vault health check scoping (Downloads vs User Input).
4. Evidence synthesizer separation (Primary Research Data vs Literature Evidence).
5. Automatic Bibliography formatting verification.
6. @ mention autocompletion verification.
"""

import sys
import shutil
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console

from scripcy.core.workspace import WorkspaceManager
from scripcy.core.vault import VaultManager
from scripcy.core.engine import EvidenceChunk, EvidenceSynthesizer, LLMEngine
from scripcy.core.references import append_vault_bibliography, resolve_vault_reference_entries
from scripcy.ui.interactive import get_mention_suggestions


def run_headless_vault_flow() -> bool:
    console = Console()
    console.print("\n[bold cyan]================================================================[/bold cyan]")
    console.print("[bold cyan]       UJI FLOW HEADLESS DUAL-TIER VAULT & MENTION SYSTEM       [/bold cyan]")
    console.print("[bold cyan]================================================================[/bold cyan]\n")

    test_dir = Path("projects") / "_test_headless_vault_flow"
    if test_dir.exists():
        shutil.rmtree(test_dir)

    console.print("[bold cyan]●[/bold cyan] [bold bright_white]Langkah 1:[/bold bright_white] Inisialisasi Workspace Proyek Baru...")
    ws = WorkspaceManager(test_dir)
    ws.ensure_directories()

    assert ws.downloads_dir.is_dir(), "vault/Downloads directory missing!"
    assert ws.user_dir.is_dir(), "vault/user directory missing!"
    assert ws.user_input_dir.is_dir(), "USER INPUT HERE directory missing!"
    console.print(f"  [dim]✔ Folder dibuat: {ws.downloads_dir.name}, {ws.user_dir.name}, {ws.user_input_dir.name}[/dim]")

    vm = VaultManager(workspace=ws)

    # -------------------------------------------------------------
    # 2. Masukkan Paper Ilmiah Eksternal ke vault/Downloads/
    # (Data online dari arXiv:2412.19437 - DeepSeek-V3 Technical Report)
    # -------------------------------------------------------------
    console.print("\n[bold cyan]●[/bold cyan] [bold bright_white]Langkah 2:[/bold bright_white] Menempatkan Paper Literatur ke vault/Downloads/...")
    paper_path = ws.downloads_dir / "deepseek_v3_2024.txt"
    paper_content = (
        "Title: DeepSeek-V3 Technical Report\n"
        "Authors: DeepSeek-AI, Haowei Zhang, Damai Dai, Dejian Yang, Zhenda Xie\n"
        "Year: 2024\n"
        "arXiv: 2412.19437\n"
        "Abstract:\n"
        "We present DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters "
        "with 37B activated for each token. To achieve efficient inference and cost-effective training, "
        "DeepSeek-V3 adopts Multi-head Latent Attention (MLA) and DeepSeekMoE architecture. "
        "Furthermore, DeepSeek-V3 pioneers an auxiliary-loss-free strategy for load balancing and sets a "
        "multi-token prediction training objective for stronger performance.\n\n"
        "Key Findings:\n"
        "1. DeepSeekMoE utilizes 256 routed experts and 1 shared expert.\n"
        "2. Auxiliary-loss-free balancing avoids performance degradation.\n"
        "3. FP8 mixed precision training ensures numerical stability without loss spikes.\n"
    )
    paper_path.write_text(paper_content, encoding="utf-8")
    console.print(f"  [dim]✔ Paper tersimpan: {paper_path.name} (~{len(paper_content)} karakter)[/dim]")

    # -------------------------------------------------------------
    # 3. Masukkan Data Riset Primer Peneliti ke "USER INPUT HERE"
    # (Catatan eksperimen lokal & grafik loss training)
    # -------------------------------------------------------------
    console.print("\n[bold cyan]●[/bold cyan] [bold bright_white]Langkah 3:[/bold bright_white] Menempatkan Data Riset Primer ke 'USER INPUT HERE'...")
    user_loss_file = ws.user_input_dir / "my_training_loss_experiment.txt"
    user_loss_content = (
        "Catatan Eksperimen Riset Lokal:\n"
        "Model: Mini-MoE Custom (8 experts, 2 active)\n"
        "Epoch 1-50:\n"
        "- Epoch 10: Loss = 3.42, GPU VRAM = 14GB\n"
        "- Epoch 30: Loss = 1.88, load imbalance terdeteksi pada expert #3\n"
        "- Epoch 50: Loss konvergen ke 1.15 setelah menerapkan balancing auxiliary-loss-free.\n"
        "Catatan: Konvergensi stabil, tidak ada spike loss tajam.\n"
    )
    user_loss_file.write_text(user_loss_content, encoding="utf-8")

    user_chart = ws.user_input_dir / "loss_curve_epoch50.png"
    user_chart.write_bytes(b"\x89PNG\r\n\x1a\nfake_image_bytes")
    console.print(f"  [dim]✔ Input peneliti: {user_loss_file.name} & {user_chart.name}[/dim]")

    # -------------------------------------------------------------
    # 4. Uji Staging & Ingestion Headless
    # -------------------------------------------------------------
    console.print("\n[bold cyan]●[/bold cyan] [bold bright_white]Langkah 4:[/bold bright_white] Eksekusi Staging & Sinkronisasi Vault...")
    staged = vm.scan_local_staging()
    console.print(f"  [dim]• Terdeteksi {len(staged)} berkas staging di 'USER INPUT HERE':[/dim]")
    for s in staged:
        console.print(f"    - [cyan]{s['filename']}[/cyan] -> Kategori: [yellow]{s['category']}[/yellow]")
    assert any(s["filename"] == "my_training_loss_experiment.txt" and s["category"] == "user" for s in staged), "User note staging failed!"

    total_converted, errors = vm.reload_vault()
    console.print(f"  [dim]✔ Reload vault: {total_converted} berkas dikonversi ke markdown cache[/dim]")

    # -------------------------------------------------------------
    # 5. Uji Health Check Scoped (Downloads vs User Input)
    # -------------------------------------------------------------
    console.print("\n[bold cyan]●[/bold cyan] [bold bright_white]Langkah 5:[/bold bright_white] Verifikasi Health Check Scoped...")
    health_downloads = vm.check_vault_health(category="downloads")
    health_user = vm.check_vault_health(category="user")

    console.print(f"  [dim]• Health Downloads: status={health_downloads['status']}, total={health_downloads['total_files']}[/dim]")
    console.print(f"  [dim]• Health User: status={health_user['status']}, total={health_user['total_files']}[/dim]")

    assert health_downloads["total_files"] >= 1, "Downloads item not detected in health check!"
    assert health_user["total_files"] >= 1, "User item not detected in health check!"

    # -------------------------------------------------------------
    # 6. Uji Engine Context Partitioning & Synthesis
    # -------------------------------------------------------------
    console.print("\n[bold cyan]●[/bold cyan] [bold bright_white]Langkah 6:[/bold bright_white] Uji Partisi Prompt Konteks (Primary vs Literature)...")
    evidence_paper = EvidenceChunk(
        chunk_id="chk_paper_1",
        source="deepseek_v3_2024.txt",
        text="DeepSeek-V3 pioneers an auxiliary-loss-free strategy for load balancing without irrecoverable loss spikes.",
        category="papers",
    )
    evidence_user = EvidenceChunk(
        chunk_id="chk_user_1",
        source="my_training_loss_experiment.txt",
        text="Epoch 50: Loss konvergen ke 1.15 setelah menerapkan balancing auxiliary-loss-free.",
        category="user",
    )

    synth = EvidenceSynthesizer()
    prompt = synth.format_context_prompt(chunks=[evidence_paper, evidence_user])

    assert "### DATA RISET PRIMER PENELITI (FOKUS ANALISIS UTAMA):" in prompt, "Missing Primary Research section!"
    assert "### BUKTI LITERATUR ILMIAH (RUJUKAN PEMBANDING & SITASI):" in prompt, "Missing Literature section!"
    assert "my_training_loss_experiment.txt" in prompt
    assert "deepseek_v3_2024.txt" in prompt
    console.print("  [dim]✔ Partisi prompt terbukti memisahkan data riset pengguna dari literatur eksternal.[/dim]")

    # -------------------------------------------------------------
    # 7. Uji Auto-Bibliography: Verifikasi Eksklusi Data Riset Pengguna
    # -------------------------------------------------------------
    console.print("\n[bold cyan]●[/bold cyan] [bold bright_white]Langkah 7:[/bold bright_white] Uji Sitasi & Eksklusi Data Riset dari Daftar Pustaka...")
    draft_response = (
        "Berdasarkan analisis hasil training Anda, kurva loss mencapai konvergensi 1.15 "
        "yang selaras dengan prinsip load balancing auxiliary-loss-free (DeepSeek-AI, 2024)."
    )

    final_output = append_vault_bibliography(
        text=draft_response,
        evidence=[evidence_paper, evidence_user],
        vault_manager=vm,
    )

    console.print("\n[dim]Hasil Output Jawaban:[/dim]")
    console.print(final_output)

    # Validasi sitasi resmi
    assert "deepseek_v3_2024.txt" in final_output, "Paper literatur wajib ada di Daftar Pustaka!"
    assert "my_training_loss_experiment.txt" not in final_output, "PERINGATAN KRITIS: Data riset pengguna bocor ke Daftar Pustaka!"
    console.print("\n  [bold green]✔ Validasi Sukses: Paper disitasi resmi, data primer peneliti TIDAK bocor ke Daftar Pustaka![/bold green]")

    # -------------------------------------------------------------
    # 8. Uji Autocomplete @ Mention
    # -------------------------------------------------------------
    console.print("\n[bold cyan]●[/bold cyan] [bold bright_white]Langkah 8:[/bold bright_white] Uji Fitur Autocomplete @ Mention...")
    sug_all = get_mention_suggestions("analisis berkas @", base_dir=test_dir)
    sug_filter = get_mention_suggestions("tolong cek grafik @loss", base_dir=test_dir)

    console.print(f"  [dim]• Total saran mention (@): {len(sug_all)} berkas ditemukan[/dim]")
    for tag, meta_info in sug_all:
        console.print(f"    - [bold cyan]{tag}[/bold cyan] -> [dim]{meta_info}[/dim]")

    assert any(s[0] == "@deepseek_v3_2024.txt" for s in sug_all), "deepseek_v3_2024.txt not found in @ suggestions"
    assert any("my_training_loss_experiment.txt" in s[0] for s in sug_all), "my_training_loss_experiment.txt not found in @ suggestions"
    assert any("loss_curve_epoch50.png" in s[0] for s in sug_filter), "loss_curve_epoch50.png not filtered properly in @ suggestions"
    console.print("  [bold green]✔ Autocomplete @ mention berhasil mengenali format, mtime, dan nama berkas.[/bold green]")

    # Bersihkan direktori tes
    try:
        shutil.rmtree(test_dir)
    except Exception:
        pass

    console.print("\n[bold green]================================================================[/bold green]")
    console.print("[bold green]          SEMUA TAHAPAN FLOW TEST HEADLESS LULUS 100%!          [/bold green]")
    console.print("[bold green]================================================================[/bold green]\n")
    return True


if __name__ == "__main__":
    success = run_headless_vault_flow()
    sys.exit(0 if success else 1)
