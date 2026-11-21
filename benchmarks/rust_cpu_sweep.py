"""The Rust CPU exact L1 search (rust_cpu): index build time, and search time by the number of queries each
thread takes at a time (`chunk`), against FAISS's flat L1. Also counts the vector instructions in the
built extension, to confirm that the inner loop uses NEON (`fabd.4s`, `fadd.4s`) and has no scalar
absolute differences. BENCHMARKS.md, "Exact search on the CPU".

    uv run --group bench python rust_cpu_sweep.py
"""
import argparse
import re
import subprocess
import time
from pathlib import Path

import faiss
import numpy as np

import siann._cpu as siann_cpu
from siann._cpu import FlatIndex


def instruction_counts() -> dict[str, int]:
    so = next(Path(siann_cpu.__file__).parent.glob("*.so"))
    asm = subprocess.run(["objdump", "-d", "--no-show-raw-insn", str(so)], check=True, capture_output=True,
                         text=True, encoding="utf-8").stdout
    patterns = {"fabd.4s (vector)": r"\tfabd\.4s", "fadd.4s (vector)": r"\tfadd\.4s",
                "fabd (scalar)": r"\tfabd\s+s\d"}
    return {name: len(re.findall(pat, asm)) for name, pat in patterns.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--q", type=int, default=1000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--chunks", default="8,16,32,64")
    a = p.parse_args()
    print("instructions in the extension: " + ", ".join(f"{k} {v}" for k, v in instruction_counts().items()))
    rng = np.random.default_rng(0)
    xb = rng.standard_normal((a.n, a.d), dtype=np.float32)
    xq = rng.standard_normal((a.q, a.d), dtype=np.float32)
    start = time.perf_counter()
    index = FlatIndex(xb, a.d, "l1")
    print(f"index build: {time.perf_counter() - start:.2f} s for {a.n:,} x {a.d}")
    flat = faiss.IndexFlat(a.d, faiss.METRIC_L1)
    flat.add(xb)
    flat.search(xq[:50], a.k)
    start = time.perf_counter()
    ref = flat.search(xq, a.k)[1]
    t_faiss = time.perf_counter() - start
    print(f"FAISS flat L1: {t_faiss:.3f} s ({faiss.omp_get_max_threads()} threads)")
    for chunk in map(int, a.chunks.split(",")):
        index.search(xq, a.k, chunk)   # warm up
        times = []
        for _ in range(3):
            start = time.perf_counter()
            got = np.array(index.search(xq, a.k, chunk)).reshape(-1, a.k)
            times.append(time.perf_counter() - start)
        t = sorted(times)[1]
        same = np.mean([len(set(g) & set(r)) / a.k for g, r in zip(got, ref)])
        print(f"  chunk {chunk:3d}: {t:.3f} s, {t_faiss / t:.2f}x FAISS, same neighbors {same:.4f}")


if __name__ == "__main__":
    main()
