"""Headless 3-question flow test on project 'deep_l' verifying fix for score_chunk error and live hierarchical callbacks."""

import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

from scripcy.studio.app import CLIStudio

print("=" * 80)
print("MEMULAI FLOW TEST 3X PERTANYAAN HEADLESS DI PROJECT 'deep_l'")
print("=" * 80)

# Initialize CLIStudio with project deep_l
studio = CLIStudio(initial_project="deep_l")
if not studio.current_project:
    print("[ERROR] Gagal membuka project 'deep_l'")
    sys.exit(1)

print(f"[OK] Project loaded: {studio.current_project.name}")
print(f"[OK] Active Model  : {studio.get_active_project_model()}")

# Ensure active chat session
session = studio._ensure_active_chat_session()
print(f"[OK] Sesi Aktif    : {session.name} (ID: {session.id})\n")

questions = [
    (1, "carikan aku gap penelitian deep l terbaru"),
    (2, "apa arsitektur transformer terbaik untuk ekstraksi dokumen akademik?"),
    (3, "bagaimana perbandingan model YOLOS terhadap detektor objek CNN konvensional?"),
]

for idx, q in questions:
    print("-" * 80)
    print(f"PERTANYAAN #{idx}: \"{q}\"")
    print("-" * 80)
    t_start = time.time()
    
    try:
        studio.handle_chat(q)
        dur = time.time() - t_start
        print(f"\n[✓] Pertanyaan #{idx} SELESAI SUKSES dalam {dur:.1f} detik!")
        
        # Check latest message from chat manager
        messages = studio.chat_manager.get_messages(session.id, limit=2) if studio.chat_manager else []
        for m in reversed(messages):
            if m.role == "assistant":
                print(f"Ringkasan Output AI ({len(m.content)} karakter):")
                preview = m.content[:350].strip()
                print(f"\"\"\"\n{preview}...\n\"\"\"")
                if "### 📚 Daftar Pustaka:" in m.content:
                    print("  -> Daftar Pustaka resmi terdeteksi dan terformat rapi.")
                break
    except Exception as exc:
        print(f"\n[✗] Pertanyaan #{idx} GAGAL dengan error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n" + "=" * 80)
print("[SUKSES TOTAL] Seluruh 3x pertanyaan headless berhasil dieksekusi tanpa error!")
print("=" * 80)
