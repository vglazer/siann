"""Exact L2 (and inner product) on the CPU without FAISS: a BLAS matrix multiply (NumPy's `@`, which on
a Mac is Apple's Accelerate) for the products q.x, and siann's Rust top-k (`L2TopK`), which ranks by
|x|^2 - 2 q.x and keeps each query's best k across blocks of the base. With zero norms it ranks by
-q.x, inner product.
"""
import time

import numpy as np

from ._cpu import L2TopK


def default_tiles(nq: int) -> tuple[int, int]:
    """The default query tile and base block: up to 512 queries, and a product of about 2^22 values
    (16 MB), the block rounded down to a power of two. Products of 4-8 million values were fast on
    SIFT and GloVe; smaller ones pay per-call overhead, larger ones fall out of cache (BENCHMARKS.md)."""
    qtile = min(nq, 512)
    return qtile, 1 << ((1 << 22) // qtile).bit_length() - 1


def siann_l2(xb, norms, xq, k, qtile, block, split=None):
    """Exact L2: for each tile of queries and block of the base, one multiply, then the top-k update."""
    out = []
    for s in range(0, len(xq), qtile):
        q = xq[s:s + qtile]
        top = L2TopK(len(q), k)
        for b in range(0, len(xb), block):
            t = time.perf_counter()
            prod = q @ xb[b:b + block].T
            if split is not None:
                split[0] += time.perf_counter() - t
            t = time.perf_counter()
            top.push_l2(prod, norms[b:b + block], b)
            if split is not None:
                split[1] += time.perf_counter() - t
        out.append(np.array(top.result(), dtype=np.int64).reshape(len(q), k))
    return np.concatenate(out)
