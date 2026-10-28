"""
Benchmark Memory Baseline Before Optimization.
Measures process RSS memory before, during, and after embedding & PDF processing.
"""

import gc
import os
import psutil
from pathlib import Path

# Measure current process
process = psutil.Process()

def get_mem_mb() -> float:
    return round(process.memory_info().rss / (1024 * 1024), 2)

print("=" * 60)
print("BENCHMARK BASELINE MEMORI (SEBELUM OPTIMASI)")
print("=" * 60)

initial_mem = get_mem_mb()
print(f"1. Memori Awal (Initial RSS)            : {initial_mem} MB")

# Load embedder with default batch_size=256
from fastembed import TextEmbedding
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
loaded_mem = get_mem_mb()
print(f"2. Setelah Load Model Embedder         : {loaded_mem} MB (+{round(loaded_mem - initial_mem, 2)} MB)")

# Simulate heavy chunk embedding with default batching (256 chunks)
dummy_chunks = [
    f"Ini adalah sampel paragraf ilmiah nomor {i} yang berisi pembahasan arsitektur vision transformer dan perbandingan kinerja deep learning."
    for i in range(400)
]

peak_before = get_mem_mb()
# Default FastEmbed batch_size = 256
embeddings = list(embedder.embed(dummy_chunks, batch_size=256))
peak_after = get_mem_mb()

print(f"3. Puncak Memori Saat Batch Embedding  : {peak_after} MB (+{round(peak_after - loaded_mem, 2)} MB)")
print("=" * 60)
