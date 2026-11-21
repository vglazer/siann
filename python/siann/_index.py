"""siann.Index: exact search over dense vectors, on the GPU (through siann's GPU backend) or the CPU (BLAS through NumPy, and
siann's Rust kernels), with natural scores and a plan that says what was chosen and why."""
from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

METRICS = ("l2", "l1", "linf", "cosine", "ip")
DISTANCES = ("l2", "l1", "linf")          # smaller is nearer; cosine and ip are similarities


@dataclass(frozen=True)
class SearchResult:
    """The k best neighbors of each query, best first: ``indices`` (int64, -1 where fewer than k
    exist) and ``scores`` (float32: the distance for L2, L1 and L-infinity, the similarity for cosine
    and inner product; infinity, or minus infinity for similarities, where an index is -1)."""
    indices: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True)
class Plan:
    """What an index chose and why; ``str(plan)`` explains it."""
    n: int
    d: int
    metric: str
    kind: str
    device: str
    reasons: tuple[str, ...]

    def __str__(self) -> str:
        lines = [f"siann.Index: {self.n:,} vectors, {self.d} dimensions, metric {self.metric}",
                 f"  kind: {self.kind}", f"  device: {self.device}"]
        lines += [f"  {r}" for r in self.reasons]
        return "\n".join(lines)


@functools.cache
def _gpu_available() -> bool:
    """Whether siann's GPU backend finds a GPU (checked once per process)."""
    try:
        import mlx.core as mx
        return bool(mx.metal.is_available())
    except Exception:
        return False


def _is_sparse(x) -> bool:
    try:
        import scipy.sparse
        return scipy.sparse.issparse(x)
    except ImportError:
        return hasattr(x, "tocsr") and hasattr(x, "nnz")


def _as_matrix(x, name: str) -> np.ndarray:
    if _is_sparse(x):
        raise NotImplementedError(f"{name} is a sparse matrix: siann searches dense vectors; "
                                  "pass a NumPy array (x.toarray() for a small one)")
    a = np.asarray(x)
    if not np.issubdtype(a.dtype, np.number) or np.iscomplexobj(a):
        raise TypeError(f"{name} must hold real numbers, not {a.dtype}")
    a = np.ascontiguousarray(a, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array (rows are vectors), not {a.ndim}-D")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} holds infinity or NaN")
    return a


def _normalized(a: np.ndarray, name: str) -> np.ndarray:
    lengths = np.linalg.norm(a, axis=1, keepdims=True)
    if (lengths == 0).any():
        raise ValueError(f"cosine is undefined for a zero vector, and {name} holds "
                         f"{int((lengths == 0).sum())}")
    return np.ascontiguousarray(a / lengths, dtype=np.float32)


class Index:
    """An index over base vectors ``xb`` (n, d), searched by ``metric``: "l2", "l1", "linf", "cosine"
    or "ip" (inner product).

    Search is exact: the results are the true k nearest neighbors. ``device`` is "auto" (where a GPU is
    available, each search goes to the GPU or the CPU by its batch size, from ``profile``'s threshold for the
    metric; otherwise the CPU), "gpu" or "cpu". ``profile`` is a ``siann.Profile`` (from ``siann.tune``),
    by default the one set by ``siann.use_profile`` or siann's own; it is fixed when the index is built.
    Each device's copy of the vectors is built the first time a search needs it. ``index.plan`` says
    what was chosen and why.
    """

    def __init__(self, xb, metric: str = "l2", recall: float | None = None, *, device: str = "auto",
                 profile=None):
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {', '.join(METRICS)}, not {metric!r}")
        if recall is not None:
            raise NotImplementedError("recall targets (approximate search) are not supported in this "
                                      "version: use recall=None for exact search")
        if device not in ("auto", "gpu", "cpu"):
            raise ValueError(f"device must be 'auto', 'gpu' or 'cpu', not {device!r}")
        x = _as_matrix(xb, "xb")
        if len(x) == 0:
            raise ValueError("xb holds no vectors")
        gpu = _gpu_available()
        if device == "gpu" and not gpu:
            raise RuntimeError("device='gpu', but no GPU is available")
        self.metric = metric
        self._x = _normalized(x, "xb") if metric == "cosine" else x
        # cosine ranks as inner product on unit vectors
        self._kernel_metric = "ip" if metric == "cosine" else metric
        from ._profile import Profile, current_profile
        if profile is not None and not isinstance(profile, Profile):
            raise TypeError("profile must be a siann.Profile (from siann.tune or siann.Profile.load)")
        self._profile = profile if profile is not None else current_profile()
        self._device = device if device != "auto" else ("auto" if gpu else "cpu")
        self._layouts: dict[str, object] = {}
        reasons = ["exact search (no recall target): the results are the true nearest neighbors"]
        if device != "auto":
            reasons.append(f"device={device!r} given")
        elif gpu:
            reasons.append(f"GPU found: {self._profile.describe(metric)} "
                           f"(profile: {self._profile.source})")
        else:
            reasons.append("no GPU found: exact search on the CPU")
        cpu_path = ("siann's Rust kernel" if self._kernel_metric in ("l1", "linf")
                    else "a BLAS matrix multiply through NumPy and siann's Rust top-k")
        reasons.append(f"on the CPU: {cpu_path}")
        if metric == "cosine":
            reasons.append("cosine: vectors normalized to unit length and ranked by inner product")
        self.plan = Plan(len(self._x), self._x.shape[1], metric, "exact", self._device, tuple(reasons))

    @property
    def n(self) -> int:
        return len(self._x)

    @property
    def d(self) -> int:
        return self._x.shape[1]

    @property
    def profile(self):
        """The ``siann.Profile`` whose thresholds this index routes by."""
        return self._profile

    def route(self, batch: int) -> str:
        """The device ("gpu" or "cpu") a search of ``batch`` queries goes to."""
        if self._device != "auto":
            return self._device
        t = self._profile.threshold(self.metric)
        return "cpu" if t is None or batch < t else "gpu"

    def __repr__(self) -> str:
        return f"siann.Index({self.n:,} vectors, {self.d} dimensions, metric={self.metric!r}, device={self._device!r})"

    def search(self, xq, k: int = 10, *, device: str | None = None) -> SearchResult:
        """The k nearest base vectors to each query in ``xq`` (m, d), or to a single query (d,),
        returned as a batch of one. ``device`` ("gpu" or "cpu") overrides the index's device."""
        q = np.asarray(xq)
        if q.ndim == 1:
            q = q[None, :]
        q = _as_matrix(q, "xq")
        if q.shape[1] != self.d:
            raise ValueError(f"xq has {q.shape[1]} dimensions, the index {self.d}")
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError(f"k must be a positive integer, not {k!r}")
        dev = device or self.route(len(q))
        if dev not in ("gpu", "cpu"):
            raise ValueError(f"device must be 'gpu' or 'cpu', not {dev!r}")
        if dev == "gpu" and not _gpu_available():
            raise RuntimeError("device='gpu', but no GPU is available")
        if self.metric == "cosine":
            q = _normalized(q, "xq")
        kk = min(int(k), self.n)
        if len(q) == 0:
            idx = np.empty((0, kk), dtype=np.int64)
        else:
            idx = (self._search_gpu if dev == "gpu" else self._search_cpu)(q, kk)
        if kk < k:
            idx = np.concatenate([idx, np.full((len(q), k - kk), -1, dtype=np.int64)], axis=1)
        return SearchResult(idx, self._scores(q, idx))

    def _search_gpu(self, q: np.ndarray, k: int) -> np.ndarray:
        import mlx.core as mx

        from .exact import mlx_knn
        if "gpu" not in self._layouts:
            x = mx.array(self._x)
            mx.eval(x)
            self._layouts["gpu"] = x
        return np.asarray(mlx_knn(self._layouts["gpu"], q, k, self._kernel_metric), dtype=np.int64)

    def _search_cpu(self, q: np.ndarray, k: int) -> np.ndarray:
        m = self._kernel_metric
        if m in ("l1", "linf"):
            from ._cpu import FlatIndex
            if "cpu" not in self._layouts:
                self._layouts["cpu"] = FlatIndex(self._x, self.d, m)
            flat = self._layouts["cpu"].search(q, k)
            return np.asarray(flat, dtype=np.int64).reshape(len(q), k)
        from .blas_l2 import default_tiles, siann_l2
        if "cpu" not in self._layouts:
            norms = (np.einsum("ij,ij->i", self._x, self._x) if m == "l2"
                     else np.zeros(self.n, dtype=np.float32))
            self._layouts["cpu"] = np.ascontiguousarray(norms, dtype=np.float32)
        qtile, block = default_tiles(len(q))
        return np.asarray(siann_l2(self._x, self._layouts["cpu"], q, k, qtile, block), dtype=np.int64)

    def _scores(self, q: np.ndarray, idx: np.ndarray, chunk: int = 256) -> np.ndarray:
        """Each returned neighbor's natural score, computed exactly from the vectors."""
        out = np.empty(idx.shape, dtype=np.float32)
        missing = idx < 0
        for s in range(0, len(q), chunk):
            i = np.where(missing[s:s + chunk], 0, idx[s:s + chunk])
            v = self._x[i]                                   # (c, k, d)
            qq = q[s:s + chunk, None, :]
            if self.metric in ("cosine", "ip"):
                out[s:s + chunk] = (v * qq).sum(axis=2)
            elif self.metric == "l2":
                out[s:s + chunk] = np.sqrt(((v - qq) ** 2).sum(axis=2))
            elif self.metric == "l1":
                out[s:s + chunk] = np.abs(v - qq).sum(axis=2)
            else:
                out[s:s + chunk] = np.abs(v - qq).max(axis=2)
        out[missing] = np.inf if self.metric in DISTANCES else -np.inf
        return out
