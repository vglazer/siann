"""FAISS's flat L2 search by dimension, with its default BLAS threshold and with the threshold at 20,
and the same on GloVe-100. FAISS 1.16 defaults `distance_compute_blas_threshold` to 128,000, which
sends L2 below 128 dimensions down a path 4-7x slower than BLAS. BENCHMARKS.md, "FAISS's BLAS threshold".

    uv run --no-sync --group proto python faiss_blas.py
"""
import argparse
import time

import faiss
import numpy as np

from datasets import load


def timed_search(xb: np.ndarray, xq: np.ndarray, k: int = 10) -> tuple[float, np.ndarray]:
    index = faiss.IndexFlatL2(xb.shape[1])
    index.add(xb)
    index.search(xq[:50], k)   # warm up
    start = time.perf_counter()
    found = index.search(xq, k)[1]
    return time.perf_counter() - start, found


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500_000)
    p.add_argument("--q", type=int, default=1000)
    p.add_argument("--dims", default="32,64,96,112,127,128,192,256")
    p.add_argument("--no-glove", action="store_true")
    a = p.parse_args()
    default = faiss.cvar.distance_compute_blas_threshold
    print(f"faiss {faiss.__version__}, default BLAS threshold {default:,}; "
          f"{a.n:,} Gaussian vectors, {a.q:,} queries")
    rng = np.random.default_rng(0)
    print(f"  {'dimensions':>10s} {'default':>9s} {'threshold 20':>13s}")
    for d in map(int, a.dims.split(",")):
        xb = rng.standard_normal((a.n, d), dtype=np.float32)
        xq = rng.standard_normal((a.q, d), dtype=np.float32)
        faiss.cvar.distance_compute_blas_threshold = default
        t_default, _ = timed_search(xb, xq)
        faiss.cvar.distance_compute_blas_threshold = 20
        t_20, _ = timed_search(xb, xq)
        print(f"  {d:10d} {t_default:8.3f}s {t_20:12.3f}s")
    if not a.no_glove:
        xb, xq = load("glove", a.q)
        faiss.cvar.distance_compute_blas_threshold = default
        t_default, r_default = timed_search(xb, xq)
        faiss.cvar.distance_compute_blas_threshold = 20
        t_20, r_20 = timed_search(xb, xq)
        same = np.array_equal(np.sort(r_default, axis=1), np.sort(r_20, axis=1))
        print(f"GloVe-100 ({len(xb):,} vectors): default {t_default:.3f} s, threshold 20 {t_20:.3f} s, "
              f"same neighbors {same}")


if __name__ == "__main__":
    main()
