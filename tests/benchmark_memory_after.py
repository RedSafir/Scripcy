"""
Benchmark Memory After Optimization.
Measures process RSS memory with batch_size=32, threads=2, and gc.collect().
"""

import gc
import psutil
from fastembed import TextEmbedding

process = psutil.Process()

def get_mem_mb() -> float:
    return round(process.memory_info().rss / (1024 * 1024), 2)

print("=" * 60)
print("BENCHMARK MEMORI (SESUDAH OPTIMASI)")
print("=" * 60)

initial_mem = get_mem_mb()
print(f"1. Memori Awal (Initial RSS)            : {initial_mem} MB")

# Load embedder with threads=2 limit
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", threads=2)
loaded_mem = get_mem_mb()
print(f"2. Setelah Load Model Embedder (threads=2): {loaded_mem} MB (+{round(loaded_mem - initial_mem, 2)} MB)")

# Same 400 chunks
dummy_chunks = [
    f"Ini adalah sampel paragraf ilmiah nomor {i} yang berisi pembahasan arsitektur vision transformer dan perbandingan kinerja deep learning."
    for i in range(400)
]

# Optimized streaming batch_size = 32
embeddings = list(embedder.embed(dummy_chunks, batch_size=32))
peak_after = get_mem_mb()
print(f"3. Puncak Memori Saat Batch Embedding  : {peak_after} MB (+{round(peak_after - loaded_mem, 2)} MB)")

# Explicit garbage collection
del embeddings
del dummy_chunks
gc.collect()
final_mem = get_mem_mb()
print(f"4. Memori Setelah Garbage Collection    : {final_mem} MB (Turun {round(peak_after - final_mem, 2)} MB)")
print("=" * 60)
