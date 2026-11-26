"""A graph index: a CAGRA graph (NVIDIA cuVS's graph, 32 links per vector), built by mlxvs (cuVS's methods on
MLX) where it is installed and by siann's own builder elsewhere (siann._graph_build: on any machine), searched on
the GPU through mlxvs for batches and on the CPU by siann's own graph search (siann._cpu.GraphIndex) anywhere,
Linux included. An IVF index keeps one alongside its lists where the metric is L2, cosine or inner product, and
routes to it the batch sizes it measured faster. The search widths (``itopk_size`` on the GPU, ``ef`` on the
CPU) are calibrated to the recall target as nprobe is. A saved index keeps the graph."""
from __future__ import annotations

import functools
import threading
import time

import numpy as np

DEGREE = 32              # links per vector, pruned from INTERMEDIATE, as in mlxvs's and cuVS's defaults
INTERMEDIATE = 64
MAX_ITOPK = 1024         # the widest GPU search calibration tries
MAX_EF = 1024            # the widest CPU search
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


def why_not(metric: str, gpu_backend: str | None = None) -> str | None:
    """Why this index cannot have a graph, or None where it can (any machine can build one)."""
    if metric not in METRICS:
        return "no graph index: CAGRA has no L1 or L-infinity"
    return None


def gpu_search_reason(gpu_backend: str | None) -> str | None:
    """Why the GPU cannot search (or build) a graph here, or None where it can: that needs mlxvs, on MLX."""
    if gpu_backend != "mlx":
        return "mlxvs runs on MLX" + ("" if gpu_backend is None else ", not on PyTorch")
    if mlxvs_version() is None:
        return "mlxvs is not installed (pip install siann[graph])"
    return None


class Graph:
    """A CAGRA graph over ``x`` (the IVF's vectors), searched by ``metric`` ("l2" or "ip"), its rows stored as
    bytes where the IVF's lists are, else as float16 on the GPU and float32 on the CPU. Built (unless ``links``
    (n, DEGREE) are given, as when loading, or ``cagra``, an mlxvs index already built) by mlxvs where ``gpu`` is
    "mlx" and mlxvs is installed, otherwise by siann's own builder, which reuses ``ivf`` (the index's IVF over
    ``x``) for each vector's nearest, on ``gpu`` where given, else on the CPU. ``builder`` says which."""

    def __init__(self, x: np.ndarray, metric: str, storage: str, links: np.ndarray | None = None, cagra=None,
                 ivf=None, gpu: str | None = None):
        self.x, self.metric, self.storage = x, metric, storage
        self.dtype = "uint8" if storage == "uint8" else "float16"
        self.itopk = self.ef = None                    # the GPU's and the CPU's calibrated search widths
        self._links = None if links is None else np.ascontiguousarray(links, dtype=np.int32)
        self._cagra, self._cpu = cagra, None
        self._lock = threading.Lock()
        self.build_seconds = 0.0
        self.builder = f"mlxvs {mlxvs_version()}" if cagra is not None else None
        if cagra is None and links is None:
            t = time.perf_counter()
            if gpu_search_reason(gpu) is None:
                import mlxvs
                self._cagra = mlxvs.CagraIndex.build(x, graph_degree=DEGREE, metric=METRICS[metric],
                                                     intermediate_graph_degree=INTERMEDIATE, dtype=self.dtype)
                self.builder = f"mlxvs {mlxvs_version()}"
            else:
                from ._graph_build import build_links
                self._links, _ = build_links(x, metric, DEGREE, INTERMEDIATE, device="gpu" if gpu else "cpu",
                                             gpu=gpu, ivf=ivf)
                self.builder = "siann"
            self.build_seconds = time.perf_counter() - t

    @property
    def on_gpu(self) -> bool:
        """Whether the GPU can search it: mlxvs is installed."""
        return self._cagra is not None or mlxvs_version() is not None

    @property
    def extendable(self) -> bool:
        """Whether new rows can be linked in without a rebuild: through mlxvs."""
        return mlxvs_version() is not None

    def links(self) -> np.ndarray:
        """The graph's links, (n, DEGREE) int32: what a saved index keeps."""
        if self._links is None:
            self._links = np.asarray(self._cagra._graph, dtype=np.int32)
        return self._links

    def _gpu(self):
        if self._cagra is None:
            with self._lock:
                if self._cagra is None:
                    import mlx.core as mx
                    import mlxvs
                    x = self.x.astype(np.uint8) if self.dtype == "uint8" else self.x
                    rows = mx.contiguous(mx.array(x).astype(getattr(mx, self.dtype)))
                    self._cagra = mlxvs.CagraIndex(rows, mx.array(self.links()), METRICS[self.metric])
        return self._cagra

    def _cpu_index(self):
        if self._cpu is None:
            with self._lock:
                if self._cpu is None:
                    from ._cpu import GraphIndex
                    links = np.ascontiguousarray(self.links(), dtype=np.int64)
                    d = self.x.shape[1]
                    if self.storage == "uint8":
                        self._cpu = GraphIndex(self.x, links, d, self.metric,
                                               np.ascontiguousarray(self.x.astype(np.uint8)), "uint8")
                    else:
                        self._cpu = GraphIndex(self.x, links, d, self.metric)
        return self._cpu

    def search(self, q: np.ndarray, k: int, itopk: int | None = None) -> np.ndarray:
        """The k nearest rows of x to each query on the GPU (int64, -1 where fewer than k were found), searching
        ``itopk`` candidates wide (the calibrated width by default), in chunks within the GPU's memory budget."""
        import mlx.core as mx
        width = max(int(itopk or self.itopk or 64), k)
        from ._memory import budget
        step = max(1024, budget("mlx") // (64 * width + 4 * DEGREE * width) // 1024 * 1024)
        cagra = self._gpu()
        out = []
        for s in range(0, len(q), step):
            ids, _ = cagra.search(mx.array(q[s:s + step]), k, itopk_size=width)
            mx.eval(ids)
            out.append(np.asarray(ids, dtype=np.int64))
        return np.concatenate(out) if len(out) > 1 else out[0]

    def search_cpu(self, q: np.ndarray, k: int, ef: int | None = None) -> np.ndarray:
        """The k nearest rows of x to each query on the CPU, by siann's own graph search with ``ef`` candidates
        (the calibrated number by default); -1 where fewer than k were found."""
        width = max(int(ef or self.ef or 64), k)
        found = np.asarray(self._cpu_index().search(np.ascontiguousarray(q, dtype=np.float32), k, width),
                           dtype=np.int64).reshape(len(q), k)
        found[found >= len(self.x)] = -1
        return found

    def extended(self, x: np.ndarray, rows: np.ndarray) -> Graph:
        """A new graph over ``x``, which is this graph's vectors followed by ``rows``, the rows linked in by mlxvs
        without a rebuild; this one is left as it was, for searches still using it."""
        import mlxvs
        old = self._gpu()
        copy = mlxvs.CagraIndex(old._dataset, old._graph, METRICS[self.metric])
        copy.extend(rows.astype(np.uint8) if self.dtype == "uint8" else rows)
        g = Graph(x, self.metric, self.storage, cagra=copy)
        g.itopk, g.ef = self.itopk, self.ef
        return g
