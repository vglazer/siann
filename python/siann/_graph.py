"""A graph index for GPU batches: NVIDIA cuVS's CAGRA, built and searched on the Apple GPU through mlxvs
(cuVS's methods on MLX). An IVF index keeps one alongside its lists where mlxvs is installed and the
metric is L2, cosine or inner product, and routes to it the batch sizes it measured faster; the IVF keeps
the rest (single queries and small batches on the CPU, where mlxvs's GPU search is slow). The graph's
search width (``itopk_size``) is calibrated to the recall target as nprobe is."""
from __future__ import annotations

import functools
import time

import numpy as np

DEGREE = 32              # links per vector, pruned from INTERMEDIATE, as in mlxvs's and cuVS's defaults
INTERMEDIATE = 64
MAX_ITOPK = 1024         # the widest search calibration tries
METRICS = {"l2": "l2", "ip": "inner_product"}     # the IVF's metric (cosine arrives as L2 on unit vectors)


@functools.cache
def mlxvs_version() -> str | None:
    """mlxvs's version where it is installed and usable (it needs MLX's GPU), else None."""
    try:
        import mlx.core as mx
        import mlxvs
        if not mx.metal.is_available():
            return None
        return getattr(mlxvs, "__version__", "?")
    except Exception:  # noqa: BLE001 (not installed, or failing to load: no graph)
        return None


def why_not(metric: str, gpu_backend: str | None) -> str | None:
    """Why this index cannot have a graph, or None where it can."""
    if metric not in METRICS:
        return "no graph index: CAGRA has no L1 or L-infinity"
    if gpu_backend != "mlx":
        return "no graph index: mlxvs runs on MLX" + ("" if gpu_backend is None else ", not on PyTorch")
    if mlxvs_version() is None:
        return "no graph index: mlxvs is not installed (pip install siann[graph])"
    return None


class Graph:
    """A CAGRA graph over ``x`` (the IVF's vectors), searched by ``metric`` ("l2" or "ip"), its rows stored as
    bytes where the IVF's lists are, else as float16; from ``graph`` (n, DEGREE) where given, as when loading."""

    def __init__(self, x: np.ndarray, metric: str, storage: str, graph: np.ndarray | None = None, cagra=None):
        import mlx.core as mx
        import mlxvs
        self.metric, self.storage = metric, storage
        self.dtype = "uint8" if storage == "uint8" else "float16"
        self.itopk = None
        t = time.perf_counter()
        if cagra is not None:
            self._cagra = cagra
        elif graph is None:
            self._cagra = mlxvs.CagraIndex.build(x, graph_degree=DEGREE, metric=METRICS[metric],
                                                 intermediate_graph_degree=INTERMEDIATE, dtype=self.dtype)
        else:
            rows = mx.array(x.astype(np.uint8) if self.dtype == "uint8" else x).astype(getattr(mx, self.dtype))
            self._cagra = mlxvs.CagraIndex(mx.contiguous(rows), mx.array(np.asarray(graph, dtype=np.int32)),
                                           METRICS[metric])
        self.build_seconds = time.perf_counter() - t

    def search(self, q: np.ndarray, k: int, itopk: int | None = None) -> np.ndarray:
        """The k nearest rows of x to each query (int64, -1 where fewer than k were found), searching
        ``itopk`` candidates wide (the calibrated width by default), in chunks within the GPU's memory budget."""
        import mlx.core as mx
        width = max(int(itopk or self.itopk or 64), k)
        from ._memory import budget
        step = max(1024, budget("mlx") // (64 * width + 4 * DEGREE * width) // 1024 * 1024)
        out = []
        for s in range(0, len(q), step):
            ids, _ = self._cagra.search(mx.array(q[s:s + step]), k, itopk_size=width)
            mx.eval(ids)
            out.append(np.asarray(ids, dtype=np.int64))
        return np.concatenate(out) if len(out) > 1 else out[0]

    def extended(self, rows: np.ndarray) -> Graph:
        """A new graph with ``rows`` added after the current ones (mlxvs links them without a rebuild); this one
        is left as it was, for searches still using it."""
        import mlxvs
        copy = mlxvs.CagraIndex(self._cagra._dataset, self._cagra._graph, METRICS[self.metric])
        copy.extend(rows.astype(np.uint8) if self.dtype == "uint8" else rows)
        g = Graph(None, self.metric, self.storage, cagra=copy)
        g.itopk = self.itopk
        return g

    def graph_array(self) -> np.ndarray:
        """The graph's links, (n, DEGREE) int32: what a saved index keeps."""
        return np.asarray(self._cagra._graph, dtype=np.int32)

    def gpu_bytes(self) -> int:
        """The graph's memory on the GPU: its rows and its links."""
        n = self._cagra._graph.shape[0]
        d = self._cagra._dataset.shape[1]
        return n * d * (1 if self.dtype == "uint8" else 2) + n * DEGREE * 4
