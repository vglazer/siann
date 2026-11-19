"""Exact k-nearest neighbors, MLX on the GPU against FAISS's flat index.

Synthetic Gaussian data by default. Prints the time for all queries and the agreement with FAISS's
results (the fraction of FAISS's k neighbors that MLX also returns).

    uv run --group proto python proto/exact.py --n 1000000 --d 128 --q 1000
"""
import argparse
import time

import faiss
import mlx.core as mx
import numpy as np

from knn_kernels import ip_knn, knn_tiled, l2_knn


def mlx_knn(xb: mx.array, xq: np.ndarray, k: int, metric: str, tile: int = 256) -> np.ndarray:
    """Exact k nearest neighbours on the GPU, a tile of queries at a time, each tile evaluated before
    the next so that only one tile's work is held in memory."""
    norms = mx.sum(xb * xb, axis=1) if metric == "l2" else None
    out = []
    for start in range(0, len(xq), tile):
        q = mx.array(xq[start:start + tile])
        if metric == "l2":
            result = l2_knn(q, xb, norms, k)
        elif metric == "ip":
            result = ip_knn(q, xb, k)
        else:
            result = knn_tiled(q, xb, k, metric)
        mx.eval(result)
        out.append(np.array(result))
    return np.concatenate(out)


FAISS_METRIC = {"l1": faiss.METRIC_L1, "l2": faiss.METRIC_L2, "linf": faiss.METRIC_Linf,
                "ip": faiss.METRIC_INNER_PRODUCT}


def faiss_knn(xb: np.ndarray, xq: np.ndarray, k: int, metric: str) -> np.ndarray:
    index = faiss.IndexFlatL2(xb.shape[1]) if metric == "l2" else faiss.IndexFlat(xb.shape[1], FAISS_METRIC[metric])
    index.add(xb)
    return index.search(xq, k)[1]


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--q", type=int, default=1000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--metrics", default="l2,l1")
    a = p.parse_args()
    rng = np.random.default_rng(0)
    xb = rng.standard_normal((a.n, a.d), dtype=np.float32)
    xq = rng.standard_normal((a.q, a.d), dtype=np.float32)
    xb_mx = mx.array(xb)
    mx.eval(xb_mx)
    print(f"{a.n:,} base vectors, {a.d} dimensions, {a.q:,} queries, k={a.k}; FAISS on {faiss.omp_get_max_threads()} threads")
    for metric in a.metrics.split(","):
        mlx_knn(xb_mx, xq[:16], a.k, metric)   # warm up: compile the kernels
        ours, t_mlx = timed(lambda: mlx_knn(xb_mx, xq, a.k, metric))
        ref, t_faiss = timed(lambda: faiss_knn(xb, xq, a.k, metric))
        agree = np.mean([len(set(r) & set(o)) / a.k for r, o in zip(ref, ours)])
        print(f"{metric}: MLX {t_mlx:.2f} s, FAISS {t_faiss:.2f} s, MLX {t_faiss / t_mlx:.1f}x faster; "
              f"agreement with FAISS {agree:.4f}", flush=True)


if __name__ == "__main__":
    main()
