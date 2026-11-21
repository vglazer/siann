"""The IVF's lists, built once for every backend: centroids (siann.kmeans, or given), each base vector's
list, and the order and offsets that store the lists contiguously. The GPU IVFs (MLX, PyTorch) and the
CPU IVF (Rust) all take their lists from here, so they hold the same lists, near-ties included.

Lists are assigned by the search metric, except for inner product, whose lists are assigned by L2: by
inner product, centroids pointing towards long vectors attract much of the data and the lists fall out
of balance (BENCHMARKS.md, "Inner product"). Queries probe by the search metric.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Lists:
    """``centroids`` (nlist x d, float32); ``assign`` (n,), each base vector's list; ``order`` (n,), the
    base ids sorted by list (stably); ``offsets`` (nlist + 1,), where each list starts in ``order``."""
    centroids: np.ndarray
    assign: np.ndarray
    order: np.ndarray
    offsets: np.ndarray

    @property
    def nlist(self) -> int:
        return len(self.centroids)

    def sizes(self) -> np.ndarray:
        return np.diff(self.offsets)


def assign_metric(metric: str) -> str:
    """The metric that assigns base vectors to lists for a search by ``metric``."""
    return "l2" if metric == "ip" else metric


def nearest(x: np.ndarray, c: np.ndarray, metric: str, device: str = "auto", tile: int = 4096) -> np.ndarray:
    """Each row of x's nearest row of c by ``metric`` (int64), exactly, on the GPU or the CPU."""
    from ._index import _gpu_available
    gpu = _gpu_available() if device == "auto" else device == "gpu"
    if gpu:
        import mlx.core as mx

        from .exact import knn
        cm = mx.array(c)
        cn = mx.sum(cm * cm, axis=1) if metric == "l2" else None
        out = []
        for s in range(0, len(x), tile):
            r = knn(mx.array(x[s:s + tile]), cm, cn, 1, metric)
            mx.eval(r)
            out.append(np.array(r)[:, 0])
        return np.concatenate(out).astype(np.int64)
    if metric in ("l1", "linf"):
        from ._cpu import FlatIndex
        return np.asarray(FlatIndex(c, c.shape[1], metric).search(x, 1), dtype=np.int64)
    from .blas_l2 import default_tiles, siann_l2
    norms = (np.einsum("ij,ij->i", c, c) if metric == "l2" else np.zeros(len(c))).astype(np.float32)
    qtile, block = default_tiles(len(x))
    return siann_l2(c, norms, x, 1, qtile, min(block, len(c)))[:, 0]


def build_lists(xb: np.ndarray, nlist: int, metric: str, centroids: np.ndarray | None = None,
                assign: str | None = None, seed: int = 1, sample_per_list: int = 100,
                device: str = "auto") -> Lists:
    """The lists of an IVF over ``xb``: k-means centroids (10 iterations on ``sample_per_list`` vectors
    per list, unless ``centroids`` are given), then each base vector assigned to its nearest centroid by
    ``assign`` (by default ``assign_metric(metric)``)."""
    xb = np.ascontiguousarray(xb, dtype=np.float32)
    if centroids is None:
        from .kmeans import kmeans
        rng = np.random.default_rng(0)
        sample = xb[rng.choice(len(xb), min(len(xb), sample_per_list * nlist), replace=False)]
        centroids = kmeans(sample, nlist, niter=10, seed=seed, device=device)
    centroids = np.ascontiguousarray(centroids, dtype=np.float32)
    a = nearest(xb, centroids, assign or assign_metric(metric), device)
    order = np.argsort(a, kind="stable")
    counts = np.bincount(a, minlength=len(centroids))
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return Lists(centroids, a, order.astype(np.int64), offsets)
