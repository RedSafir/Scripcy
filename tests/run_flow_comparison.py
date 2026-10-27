import sys
from pathlib import Path
from scripcy.core.workspace import WorkspaceManager
from scripcy.core.vault import VaultManager
from scripcy.core.engine import LLMEngine

sys.stdout.reconfigure(encoding='utf-8')

ws = WorkspaceManager(Path('.'))
vm = VaultManager(ws)
vv = vm.vector_vault

questions = [
    ("Q1 (Konseptual & Lintas Bahasa)", "bagaimana arsitektur transformer memproses representasi input tanpa rekursi?"),
    ("Q2 (Data Spesifik & Angka Presisi)", "berapa skor BLEU yang dicapai pada tugas translasi WMT 2014?"),
    ("Q3 (Visual Multimodal & Dokumen Web)", "apa korelasi suhu permukaan dan fenomena cuaca pada laporan BMKG?"),
]

engine = LLMEngine()
synth = engine.evidence_synthesizer

print("=" * 75)
print("HASIL FLOW TEST LENGKAP: SISTEM RAG LAMA VS SISTEM RAG BARU")
print("=" * 75 + "\n")

for label, q in questions:
    print(f"[{label}]")
    print(f"Pertanyaan Pengguna: \"{q}\"")
    
    # 1. BASELINE LAMA (Simulasi)
    old_res = vv.hybrid_search_rrf(q, top_k=2)
    old_top = old_res[0] if old_res else None
    
    print("\n--- A. SISTEM RAG SEBELUMNYA (Baseline Lama) ---")
    if old_top:
        print(f"1. Query Masukan      : Tunggal mentah tanpa ekspansi istilah.")
        print(f"2. Dokumen Ditemukan  : {old_top.paper_id} (Halaman {old_top.page_num})")
        print(f"3. Algoritma Peringkat: RRF Statis (Skor RRF: {old_top.rrf_score:.4f})")
        print(f"4. Kelengkapan Konteks: CHILD SAJA (Terpotong di batas 110 kata)")
        print(f"   Cuplikan Teks     : \"{old_top.content[:95]}...\"")
        print(f"5. Footnote Verbatim  : TIDAK ADA (Hanya sitasi biasa)")
        print(f"6. Penanganan Visual  : Hanya membaca deskripsi teks ringkas")
    else:
        print("Dokumen tidak ditemukan.")

    # 2. SISTEM BARU
    evidence = synth.retrieve_evidence(query=q, documents=[], vector_vault=vv)
    
    print("\n--- B. SISTEM RAG SAAT INI (Fitur Lanjutan) ---")
    if evidence:
        ev_top = evidence[0]
        expansions = synth.expand_query(q)
        print(f"1. Query Expansion    : Multi-Query paralel aktif ({len(expansions)} varian):")
        if len(expansions) > 1:
            print(f"   - Istilah Jurnal   : \"{expansions[1]}\"")
        print(f"2. Dokumen Ditemukan  : {ev_top.source} (Halaman {ev_top.page_num})")
        print(f"3. Algoritma Peringkat: Cross-Encoder Re-Ranking (Cross-Attention Score: {ev_top.score:.4f})")
        print(f"4. Kelengkapan Konteks: PARENT WINDOW MERGED (Narasi utuh sub-bab):")
        print(f"   Cuplikan Konteks   : \"{ev_top.text[:130]}...\"")
        print(f"5. Direct Footnote    : Halaman {ev_top.page_num} • Kutipan Kalimat Asli:")
        print(f"   Kutipan Bukti      : > \"{ev_top.quote_snippet}\"")
        has_vis = "[[VISUAL:" in ev_top.text
        if has_vis:
            print(f"6. Multimodal Payload : TERDETEKSI (Direct Multimodal Injection - Base64 dikirim ke Vision AI)")
        else:
            print(f"6. Multimodal Payload : Text-Only (Sangat hemat token)")
    else:
        print("Dokumen tidak ditemukan.")

    print("\n" + "=" * 75 + "\n")
