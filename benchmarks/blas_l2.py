"""Exact L2 on the CPU without FAISS: a BLAS matrix multiply (NumPy's `@`, which on a Mac is Apple's
Accelerate, the same BLAS FAISS links) for the products q.x, and siann's Rust top-k (`L2TopK`), which
ranks by |x|^2 - 2 q.x and keeps each query's best k across blocks of the base. Against FAISS's flat
index (IndexFlatL2, BLAS threshold 20; see faiss_blas.py), timed the same way: median of three after a
warm-up, all queries at once. Agreement with FAISS counts ties as correct (backends.agreement).

Sweeps the query tile (queries per multiply) and the base block (base vectors per multiply), and
reports how the time splits between the multiplies and the selection. The first row is the default
(default_tiles), then the sweep.

    uv run --group bench python blas_l2.py --dataset sift
    uv run --group bench python blas_l2.py --dataset glove-raw --metric ip   # inner product

With --metric ip, the same path ranks by -q.x (zero norms: |x|^2 - 2 q.x becomes -2 q.x), against
FAISS's IndexFlatIP.
"""
import argparse
import time

import faiss
import numpy as np
from siann.blas_l2 import default_tiles, siann_l2

from backends import agreement
from datasets import load

K = 10


def timed(fn, repeats=3):
    fn()
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t)
    return sorted(times)[len(times) // 2], result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sift", choices=["sift", "glove", "glove-raw", "music100"])
    p.add_argument("--metric", default="l2", choices=["l2", "ip"])
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--qtiles", default="256,1000")
    p.add_argument("--blocks", default="16384,65536,262144")
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args()
    faiss.omp_set_num_threads(a.threads)
    faiss.cvar.distance_compute_blas_threshold = 20
    xb, xq = load(a.dataset, a.queries)
    norms = (np.einsum("ij,ij->i", xb, xb).astype(np.float32) if a.metric == "l2"
             else np.zeros(len(xb), dtype=np.float32))
    index = faiss.IndexFlatL2(xb.shape[1]) if a.metric == "l2" else faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    t_faiss, ref = timed(lambda: index.search(xq, K)[1])
    print(f"{a.dataset}, {a.metric}: {len(xb):,} vectors, {xb.shape[1]} dimensions, {len(xq):,} queries, k={K}; "
          f"NumPy's BLAS: {np.show_config(mode='dicts')['Build Dependencies']['blas']['name']}")
    print(f"  FAISS {type(index).__name__}: {t_faiss:.3f} s\n")
    print(f"  {'query tile':>10s} {'base block':>10s} {'time':>8s} {'vs FAISS':>9s} {'multiply':>9s} "
          f"{'top-k':>7s} {'agreement':>9s}")
    configs = [default_tiles(len(xq))] + [(qt, b) for qt in map(int, a.qtiles.split(","))
                                          for b in map(int, a.blocks.split(","))]
    for i, (qtile, block) in enumerate(configs):   # the first is the default (default_tiles)
        if i == 1:
            print("  (sweep)")
        t, found = timed(lambda: siann_l2(xb, norms, xq, K, qtile, block))
        split = [0.0, 0.0]
        siann_l2(xb, norms, xq, K, qtile, block, split)
        agree = agreement(xb, xq, found, ref, a.metric)
        print(f"  {qtile:10,d} {block:10,d} {t:7.3f}s {t_faiss / t:8.2f}x {split[0]:8.3f}s {split[1]:6.3f}s "
              f"{agree:9.4f}", flush=True)


if __name__ == "__main__":
    main()
