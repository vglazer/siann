"""LSH for L1 on SIFT1M, as recall at 10 against the share of the database
that must be re-ranked. Cauchy (1-stable) random projections, h(x) = floor((a.x + b) / w), m per
table concatenated into one key, L tables, candidates re-ranked by exact L1.

The share re-ranked is the cost that matters: exact L1 on the GPU already scans all of it, so LSH
must reach high recall while re-ranking a small share.

    uv run --group bench python lsh_l1.py
"""
import itertools
import time

from pathlib import Path

import h5py
import mlx.core as mx
import numpy as np

from siann.knn_kernels import dist, knn_tiled

K, QUERIES = 10, 500
f = h5py.File(Path(__file__).resolve().parent.parent / "data" / "sift-128-euclidean.hdf5", "r")
xb, xq = f["train"][:], f["test"][:QUERIES]
n, d = xb.shape
xb_mx = mx.array(xb)
mx.eval(xb_mx)

start = time.perf_counter()
truth = np.concatenate([np.array(knn_tiled(mx.array(xq[i:i + 256]), xb_mx, K, "l1")) for i in range(0, QUERIES, 256)])
t_exact = time.perf_counter() - start
r10 = float(np.median([np.abs(xb[t[-1]] - q).sum() for t, q in zip(truth, xq)]))
print(f"SIFT1M: {n:,} vectors, {d} dimensions; {QUERIES} queries; exact L1 on the GPU {t_exact:.2f} s "
      f"({QUERIES / t_exact:,.0f} queries/s); median L1 distance to the 10th neighbour {r10:,.0f}", flush=True)

rng = np.random.default_rng(0)
mults = rng.integers(1, 2**61, size=64, dtype=np.int64) | 1


def keys(proj: np.ndarray, m: int, tables: int) -> np.ndarray:
    """One int64 key per row and table, mixing the table's m hash values."""
    h = proj.reshape(len(proj), tables, m).astype(np.int64)
    return (h * mults[:m]).sum(axis=2)   # wraps around: a hash, not an exact encoding


for w_factor, m, tables in itertools.product((0.5, 1.0, 2.0), (4, 8, 12), (8, 32)):
    w = w_factor * r10
    a = rng.standard_cauchy((d, m * tables)).astype(np.float32)
    b = rng.uniform(0, w, m * tables).astype(np.float32)
    base = keys(np.floor((np.array(xb_mx @ mx.array(a)) + b) / w), m, tables)
    query = keys(np.floor((xq @ a + b) / w), m, tables)
    order = np.argsort(base, axis=0, kind="stable")
    sorted_keys = np.take_along_axis(base, order, axis=0)
    recall, shown = 0.0, 0
    for i in range(QUERIES):
        parts = []
        for t in range(tables):
            lo, hi = np.searchsorted(sorted_keys[:, t], query[i, t], "left"), np.searchsorted(sorted_keys[:, t], query[i, t], "right")
            parts.append(order[lo:hi, t])
        cand = np.unique(np.concatenate(parts)) if parts else np.empty(0, np.int64)
        shown += len(cand)
        if len(cand):
            dist = np.abs(xb[cand] - xq[i]).sum(axis=1)
            found = cand[np.argsort(dist)[:K]]
            recall += len(set(found) & set(truth[i])) / K
    print(f"w = {w_factor:.1f} x r10, m = {m:2d}, L = {tables:2d}: recall@10 {recall / QUERIES:.3f}, "
          f"re-ranked {100 * shown / QUERIES / n:6.2f}% of the database", flush=True)
