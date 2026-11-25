"""siann.Index: nearest-neighbor search over dense vectors, exact or to a recall target (an IVF calibrated to
it), on the GPU (through siann's GPU backend) or the CPU (siann's Rust kernels, and BLAS through NumPy), routed
by batch size; vectors added and removed; saved and loaded; natural scores; and a plan that says what was
chosen and why."""
from __future__ import annotations

import functools
import json
import platform
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._ivf import DEFAULT_ROUTES, NEVER

METRICS = ("l2", "l1", "linf", "cosine", "ip")
DISTANCES = ("l2", "l1", "linf")          # smaller is nearer; cosine and ip are similarities
FORMAT = "siann-index"
FORMAT_VERSION = 3                        # files of a later version are refused; 2 adds ids, 3 a graph
# Changing an index (BENCHMARKS, DESIGN): added vectors wait in a buffer, searched exactly, until it holds
# BUFFER_SHARE of the index or BUFFER_MIN vectors, whichever is more; removed vectors are masked until DEAD_MAX
# are: a search fetches FETCH_EXTRA more and drops them, and searches again, fetching as many more as are
# masked, only the queries left short; an IVF is re-clustered when its largest list against the mean has
# grown REBALANCE times since its last clustering, or the index REGROW times.
BUFFER_SHARE, BUFFER_MIN, DEAD_MAX, FETCH_EXTRA, REBALANCE, REGROW = 0.01, 10_000, 512, 8, 2.0, 2.0


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


GPUS = ("auto", "mlx", "torch")
GPU_NAMES = {"mlx": "MLX", "torch": "PyTorch (MPS)"}


@functools.cache
def _mlx_gpu() -> bool:
    """Whether MLX finds a GPU (checked once per process)."""
    try:
        import mlx.core as mx
        return bool(mx.metal.is_available())
    except Exception:  # noqa: BLE001 (any failure to load or query MLX means no usable GPU)
        return False


@functools.cache
def _torch_gpu() -> bool:
    """Whether PyTorch is installed and finds an Apple GPU through its MPS backend (checked once)."""
    try:
        import torch
        return bool(torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001 (any failure to load or query PyTorch means no usable GPU)
        return False


def _default_gpu() -> str | None:
    """The GPU backend ``gpu="auto"`` takes: MLX where it finds a GPU, otherwise PyTorch's MPS backend,
    otherwise none. PyTorch is only imported when MLX finds no GPU."""
    return "mlx" if _mlx_gpu() else ("torch" if _torch_gpu() else None)


def _gpu_available() -> bool:
    """Whether a GPU backend finds a GPU."""
    return _default_gpu() is not None


def _resolve_gpu(gpu: str) -> str | None:
    """The GPU backend for ``gpu=``: "auto" (``_default_gpu``), "mlx" or "torch" (refused if it finds no GPU)."""
    if gpu not in GPUS:
        raise ValueError(f"gpu must be one of {', '.join(map(repr, GPUS))}, not {gpu!r}")
    if gpu == "auto":
        return _default_gpu()
    if not (_mlx_gpu() if gpu == "mlx" else _torch_gpu()):
        raise RuntimeError(f"gpu={gpu!r}, but {GPU_NAMES[gpu]} finds no GPU"
                           + (" (is PyTorch installed? pip install siann[torch])" if gpu == "torch" else ""))
    return gpu


@functools.cache
def _machine() -> str:
    """This machine, as routing depends on it: the processor's name, its architecture and the OS (not its
    version, so that routes survive an OS update)."""
    name = platform.processor() or "unknown processor"
    if platform.system() == "Darwin":
        try:
            name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], check=True, capture_output=True,
                                  text=True, encoding="utf-8").stdout.strip() or name
        except (OSError, subprocess.CalledProcessError):
            pass
    return f"{name} ({platform.machine()}, {platform.system()})"


def _is_sparse(x) -> bool:
    try:
        import scipy.sparse
        return scipy.sparse.issparse(x)
    except ImportError:
        return hasattr(x, "tocsr") and hasattr(x, "nnz")


def _as_matrix(x, name: str) -> np.ndarray:
    if type(x) is np.ndarray and x.dtype == np.float32 and x.ndim == 2 and x.flags.c_contiguous:
        if not np.isfinite(x).all():               # the common case, checked without conversions
            raise ValueError(f"{name} holds infinity or NaN")
        return x
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


_COPY_LOCK = threading.Lock()   # a device's copy of the vectors is made once, by whichever search needs it first


def _exact_gpu(x: np.ndarray, layouts: dict, q: np.ndarray, k: int, metric: str, backend: str = "mlx") -> np.ndarray:
    """Exact search of x on the GPU, through ``backend`` ("mlx" or "torch"); x's copy there is made on first
    use and kept in ``layouts``."""
    if backend == "torch":
        import torch

        from .torch_kernels import torch_knn
        with _COPY_LOCK:
            if "gpu" not in layouts:
                xt = torch.from_numpy(x).to("mps")
                layouts["gpu"] = (xt, (xt * xt).sum(1) if metric == "l2" else None)
        xt, norms = layouts["gpu"]
        return torch_knn(xt, norms, q, k, metric, tile=_exact_tile(len(x), backend))
    import mlx.core as mx

    from ._memory import cap_mlx_cache
    from .exact import mlx_knn
    with _COPY_LOCK:
        if "gpu" not in layouts:
            cap_mlx_cache()
            a = mx.array(x)
            mx.eval(a)
            layouts["gpu"] = a
    return np.asarray(mlx_knn(layouts["gpu"], q, k, metric, tile=_exact_tile(len(x), backend)), dtype=np.int64)


EXACT_TILE = 256          # queries per exact GPU call, fewer for large collections
EXACT_BYTES = 6           # its working memory per query and base vector: 6 bytes for L2 on SIFT1M, 4 for L1


def _exact_tile(n: int, backend: str) -> int:
    """Queries per exact GPU call: EXACT_TILE, or fewer where a tile's distances to all n vectors would
    exceed the memory budget (above about 2.8 million vectors on a machine with a 4 GB budget)."""
    from ._memory import budget
    return int(max(1, min(EXACT_TILE, budget(backend) // (EXACT_BYTES * max(n, 1)))))


def _exact_cpu(x: np.ndarray, layouts: dict, q: np.ndarray, k: int, metric: str) -> np.ndarray:
    """Exact search of x on the CPU: siann's Rust kernel for L1 and L-infinity, BLAS for L2 and inner product."""
    if metric in ("l1", "linf"):
        from ._cpu import FlatIndex
        with _COPY_LOCK:
            if "cpu" not in layouts:
                layouts["cpu"] = FlatIndex(x, x.shape[1], metric)
        flat = layouts["cpu"].search(q, k)
        return np.asarray(flat, dtype=np.int64).reshape(len(q), k)
    from .blas_l2 import default_tiles, siann_l2
    with _COPY_LOCK:
        if "cpu" not in layouts:
            from ._cpu import sq_norms
            norms = sq_norms(x) if metric == "l2" else np.zeros(len(x), dtype=np.float32)
            layouts["cpu"] = np.ascontiguousarray(norms, dtype=np.float32)
    qtile, block = default_tiles(len(q))
    return np.asarray(siann_l2(x, layouts["cpu"], q, k, qtile, block), dtype=np.int64)


def _normalized(a: np.ndarray, name: str) -> np.ndarray:
    from ._cpu import normalize
    out, zeros = normalize(np.ascontiguousarray(a, dtype=np.float32))
    if zeros:
        raise ValueError(f"cosine is undefined for a zero vector, and {name} holds {zeros}")
    return out


def _changing(method):
    """Runs a method that changes the index under its lock: one change at a time, and none during a search's
    snapshot."""
    @functools.wraps(method)
    def locked(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return locked


def _check_device(device: str) -> None:
    if device not in ("auto", "gpu", "cpu"):
        raise ValueError(f"device must be 'auto', 'gpu' or 'cpu', not {device!r}")


def _check_profile(profile) -> None:
    from ._profile import Profile
    if profile is not None and not isinstance(profile, Profile):
        raise TypeError("profile must be a siann.Profile (from siann.tune or siann.Profile.load)")


class Index:
    """An index over base vectors ``xb`` (n, d), searched by ``metric``: "l2", "l1", "linf", "cosine"
    or "ip" (inner product).

    Without ``recall``, search is exact: the results are the true k nearest neighbors. With a target such as
    ``recall=0.95``, siann builds an IVF (``kind="auto"``), calibrates how many lists each query probes
    (nprobe) so that a sample of the index's own vectors reaches that recall@10, times it against exact
    search on that sample, and keeps whichever is faster; ``kind="exact"`` or ``"ivf"`` overrides the
    choice, ``nlist`` the number of lists and ``storage`` ("float32", "float16", "uint8") the lists'
    type, and ``search(..., nprobe=)`` the calibrated nprobe. The target holds for queries like the
    calibration sample: pass ``calibration_queries`` (representative queries, up to 1,000 used) when the
    queries differ from the base vectors, as they often do. ``device`` is "auto" (where a GPU is
    available, each search goes to the GPU or the CPU by its batch size, from ``profile``'s threshold for the
    metric; otherwise the CPU), "gpu" or "cpu". ``profile`` is a ``siann.Profile`` (from ``siann.tune``),
    by default the one set by ``siann.use_profile`` or siann's own; it is fixed when the index is built.
    ``gpu`` is the GPU backend: "auto" (MLX where it finds a GPU, otherwise PyTorch's MPS backend), "mlx" or
    "torch". Each device's copy of the vectors is built the first time a search needs it; an index too
    large for the GPU's memory stays on the CPU. ``index.plan`` says what was chosen and why.

    ``xb`` may be empty, of shape (0, d), for an index grown by ``add``: added vectors are searched exactly,
    and with a recall target, the choice between an IVF and exact search is made at the first merge and
    made again each time the index has doubled, while exact search stays the faster.
    """

    def __init__(self, xb, metric: str = "l2", recall: float | None = None, *, device: str = "auto",
                 gpu: str = "auto", profile=None, kind: str = "auto", nlist: int | None = None,
                 storage: str = "auto", calibration_queries=None):
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {', '.join(METRICS)}, not {metric!r}")
        if recall is not None and not 0 < recall <= 1:
            raise ValueError(f"recall must be a target between 0 and 1, not {recall!r}")
        if kind not in ("auto", "exact", "ivf", "graph"):
            raise ValueError(f"kind must be 'auto', 'exact', 'ivf' or 'graph', not {kind!r}")
        if kind in ("ivf", "graph") and recall is None:
            raise ValueError(f"kind={kind!r} needs a recall target, such as recall=0.95")
        if storage not in ("auto", "float32", "float16", "uint8"):
            raise ValueError(f"storage must be 'auto', 'float32', 'float16' or 'uint8', not {storage!r}")
        _check_device(device)
        _check_profile(profile)
        x = _as_matrix(xb, "xb")
        if x.shape[1] == 0:
            raise ValueError("xb's vectors have no dimensions")
        from ._profile import current_profile
        gpu = self._setup(_normalized(x, "xb") if metric == "cosine" else x, metric, recall, device, gpu,
                          profile if profile is not None else current_profile(), kind, nlist, storage)
        if calibration_queries is not None:
            self._cal_given = self._checked_queries(calibration_queries)
        reasons = []
        if recall is None or kind == "exact":
            reasons.append("exact search (no recall target): the results are the true nearest neighbors"
                           if recall is None else "exact search (kind='exact' given)")
        elif len(self._x) == 0:
            reasons.append(self._empty_line())
        else:
            reasons += self._build_ivf(recall, kind, nlist, storage, gpu, self._cal_given)
            self._tested_n = len(self._x)
        self._build_reasons = tuple(reasons)
        self._finish_plan(list(reasons), device, gpu)

    def _setup(self, x: np.ndarray, metric: str, recall, device: str, gpu: str, profile, kind: str, nlist,
               storage: str, ids=None, clustered=(None, None)) -> bool:
        """The state an index starts from, built or loaded: ``x`` its vectors (normalized for cosine), the
        options it was built with, the GPU backend resolved and checked (refused with ``device="gpu"`` where
        there is no GPU, or the index would not fit on it), nothing searched or changed yet. Returns whether
        searches may use the GPU."""
        self._lock = threading.RLock()                     # held by changes; searches take a snapshot under it
        self._gpu, self._gpu_arg = _resolve_gpu(gpu), gpu
        if device == "gpu" and self._gpu is None:
            raise RuntimeError("device='gpu', but no GPU is available")
        self.metric = metric
        self._kernel_metric = "ip" if metric == "cosine" else metric      # cosine: inner product on unit vectors
        self._x = x
        self._too_big = self._gpu_too_big()
        if self._too_big and device == "gpu":
            raise RuntimeError(f"device='gpu', but {self._too_big}")
        usable = self._gpu is not None and not self._too_big
        self._profile = profile
        self._device_arg = device
        self._device = device if device != "auto" else ("auto" if usable else "cpu")
        self._layouts: dict[str, object] = {}
        self.recall = recall
        self._ivf = self._nprobe = self._storage = None
        self._routes = {}
        self._cal_q = self._cal_ids = self._cal_given = None
        self._kind_arg, self._nlist_arg, self._storage_arg = kind, nlist, storage
        self._init_changes(len(x), ids, *clustered)
        self._tested_n = 0                                 # the size at the last choice between the IVF and exact
        return usable

    def _init_changes(self, n: int, ids=None, clustered_n=None, clustered_imbalance=None) -> None:
        """No vector added or removed yet: ids are row numbers unless given."""
        self._ids = ids                                    # each row's id; None: the row number
        self._next_id = int(ids.max()) + 1 if ids is not None and len(ids) else n
        self._buf = None                                   # added vectors not yet merged: x, ids, lists
        self._buf_store = None                             # the buffer's arrays, with room to grow (views: _buf)
        self._buf_layouts: dict[str, object] = {}
        self._dead = None                                  # rows removed, masked until compacted
        self._ndead = 0                                    # how many (the mask's sum, kept: it is searched often)
        self._added = self._removed = self._merges = self._reclusters = 0
        self._last_merge = self._last_recluster = None
        self._clustered = (clustered_n, clustered_imbalance)

    def _gpu_too_big(self) -> str | None:
        """Why the index cannot go on the GPU, where its copy there (float32, as exact search holds it) would
        exceed half the GPU's recommended working set; None where it fits or there is no GPU."""
        if self._gpu is None:
            return None
        from ._memory import fits, gb, working_set
        nbytes = self._x.nbytes
        if fits(nbytes, self._gpu):
            return None
        return (f"the index is too large for the GPU: its copy there would take {gb(nbytes)}, more than half the "
                f"{gb(working_set(self._gpu))} the GPU can use")

    def _checked_queries(self, queries) -> np.ndarray:
        cq = _as_matrix(queries, "calibration_queries")
        if cq.shape[1] != self.d or len(cq) == 0:
            raise ValueError(f"calibration_queries must be a non-empty (m, {self.d}) array")
        return cq

    def _empty_line(self) -> str:
        what = "an IVF is built" if self._kind_arg == "ivf" else "the choice between an IVF and exact search is made"
        return (f"empty: added vectors are searched exactly; {what} at the first merge, when "
                f"{BUFFER_MIN:,} have been added")

    def _choose_again(self) -> None:
        """At a merge of an exact index with a recall target: builds the IVF and keeps it if it is faster than
        exact search (as at a build), once the index is no longer empty or has doubled since the last choice."""
        n = len(self._x)
        if (self.recall is None or self._kind_arg == "exact" or n == 0
                or (self._tested_n and n < REGROW * self._tested_n)):
            return
        before = self._tested_n
        nlist = self._nlist_arg if self._nlist_arg and self._nlist_arg <= n else None
        lines = self._build_ivf(self.recall, self._kind_arg, nlist, self._storage_arg, self._gpu is not None,
                                self._cal_given, grown=True)
        self._tested_n = n
        if not before:
            head = (f"grown from empty: at the merge at {n:,} vectors, "
                    + ("an IVF built (kind='ivf' given)" if self._kind_arg == "ivf" else
                       "the choice between an IVF and exact search made"))
        else:
            head = (f"grown: the choice between an IVF and exact search made again at the merge at {n:,} vectors "
                    f"(last made at {before:,})")
        self._build_reasons = (head,) + tuple(lines)

    @_changing
    def calibrate(self, queries) -> int | None:
        """Calibrates on ``queries`` (m, d), representative queries (up to 1,000 used), kept for every later
        calibration. An IVF recalibrates nprobe for its recall target now, re-measures routing and returns the
        new nprobe; an exact index keeps them for its next choice between an IVF and exact search, and returns
        None."""
        if self.recall is None:
            raise ValueError("calibrate needs an index built with a recall target, such as recall=0.95")
        self._cal_given = self._checked_queries(queries)
        if self._ivf is None:
            self._refresh_plan()
            return None
        from ._ivf import CALIBRATION_K
        dev = "cpu" if self._build_device() == "cpu" else "gpu"
        q, ids, k1, source = self._calibration_sample(self._cal_given)
        self._cal_q, self._cal_ids = q, ids
        self._nprobe, reached, _, _ = self._calibrate(self._ivf, q, ids, k1, self.recall, dev)
        routing = self._route()
        line = (f"calibrated: nprobe {self._nprobe} reaches recall@{CALIBRATION_K} {reached:.3f} "
                f"(target {self.recall}) on {source}, given to index.calibrate()")
        self._build_reasons = tuple(line if r.startswith("calibrated: ") else routing if r.startswith("routing ")
                                    else r for r in self._build_reasons)
        self._refresh_plan()
        return self._nprobe

    def _finish_plan(self, reasons: list[str], device: str, gpu: bool) -> None:
        """The plan: the build's reasons, then the device's."""
        metric = self.metric
        if self._too_big:
            reasons.append(f"GPU found, but {self._too_big}: every search on the CPU")
        elif device != "auto":
            reasons.append(f"device={device!r} given")
        elif gpu and self._ivf is not None:
            reasons.append("GPU found: approximate searches routed as above")
        elif gpu:
            reasons.append(f"GPU found: {self._profile.describe(metric)} "
                           f"(profile: {self._profile.source})")
        else:
            reasons.append("no GPU found: every search on the CPU")
        if self._ivf is None:
            cpu_path = ("siann's Rust kernel" if self._kernel_metric in ("l1", "linf")
                        else "a BLAS matrix multiply through NumPy and siann's Rust top-k")
            reasons.append(f"on the CPU: {cpu_path}")
        if self._gpu is not None:
            reasons.append(f"GPU backend: {GPU_NAMES[self._gpu]}"
                           + (f" (gpu={self._gpu_arg!r} given)" if self._gpu_arg != "auto" else ""))
        if self._gpu is not None and not self._too_big and self._device != "cpu":
            from ._memory import budget, gb, working_set
            ws = working_set(self._gpu)
            reasons.append(f"memory: {gb(self._x.nbytes)} of vectors, copied to the GPU when a search first needs "
                           f"them; each GPU call works within {gb(budget(self._gpu))}"
                           + (f" (the GPU can use {gb(ws)})" if ws else ""))
        routes = {**DEFAULT_ROUTES, **self._routes}
        firsts = {paths[0] for _, paths in routes.get("table") or []}
        if self._ivf is not None and self._ivf.graph is not None and (
                any(p.endswith("graph") for p in firsts) if routes.get("table") else
                routes["cpu_ivf_from"] > 1 or (self._device != "cpu" and routes["graph_from"] < 10 ** 9)):
            reasons.append("neighbors depend on the path: each path (the graph on the CPU or the GPU, the IVF) is "
                           "calibrated to the recall target on its own, and they find different neighbors")
        if metric == "cosine":
            reasons.append("cosine: vectors normalized to unit length and ranked by inner product")
        from . import _build
        if self._device != "gpu" and self._kernel_metric in ("l2", "ip") and _build.openblas_spins():
            reasons.append("NumPy's OpenBLAS was loaded before siann, so its threads spin after each call and "
                           "slow siann's CPU searches: import siann before NumPy, or set "
                           "OPENBLAS_THREAD_TIMEOUT=4 before starting Python (TUNING.md)")
        self.plan = Plan(self.n, self.d, metric, "ivf" if self._ivf else "exact", self._device, tuple(reasons))

    def _change_lines(self) -> list[str]:
        """The plan's lines on what changed since the index was built or loaded."""
        lines = []
        if self._added or self._removed:
            line = f"changed since built: {self._added:,} added, {self._removed:,} removed"
            if self._buf is not None:
                line += (f"; {len(self._buf['x']):,} in a buffer searched exactly, merged into the "
                         f"{'lists' if self._ivf is not None else 'index'} at {self._buffer_limit():,}")
            if self._dead is not None:
                line += f"; {self._ndead:,} removed rows masked, compacted past {DEAD_MAX}"
            lines.append(line)
        if self._last_merge:
            lines.append(f"merged {self._merges} time(s); at the last, {self._last_merge}")
        if self._last_recluster:
            lines.append(f"re-clustered {self._reclusters} time(s); the last {self._last_recluster}")
        return lines

    def _refresh_plan(self) -> None:
        self._finish_plan(list(self._build_reasons) + self._change_lines(), self._device_arg, self._gpu is not None)

    def _build_ivf(self, recall: float, kind: str, nlist: int | None, storage: str, gpu: bool,
                   calibration_queries=None, grown: bool = False) -> list[str]:
        """Builds the IVF, calibrates nprobe and, with kind "auto", keeps it only if it is faster than exact
        search on the calibration sample. Returns the plan's lines. ``grown``: at a merge, where storage="uint8"
        given for vectors that are not all bytes falls back to float."""
        from ._ivf import CALIBRATION_K, IVF, default_nlist, default_storage
        n = self.n
        ivf_metric = "l2" if self.metric == "cosine" else self.metric      # cosine: L2 on unit vectors
        nlist = int(nlist) if nlist else default_nlist(n)
        if not 1 <= nlist <= n:
            raise ValueError(f"nlist must be between 1 and the number of vectors ({n:,}), not {nlist}")
        st = default_storage(self._x) if storage == "auto" else storage
        if st == "uint8" and default_storage(self._x) != "uint8":
            if not grown:
                raise ValueError("storage='uint8' needs every coordinate to be an integer from 0 to 255")
            st = "float32"
        self._storage = st
        lines = []
        build_device = self._build_device()
        ivf = IVF(self._x, ivf_metric, nlist, "uint8" if st == "uint8" else "float", build_device, gpu=self._gpu)
        lines.append(f"approximate search: an IVF of {ivf.nlist:,} lists"
                     + (" (about 4 sqrt(n))" if not nlist or nlist == default_nlist(n) else "")
                     + f", built in {ivf.build_seconds:.1f} s (siann's k-means, 10 iterations)")
        lines.append(self._storage_line(st, build_device))
        dev = "cpu" if build_device == "cpu" else "gpu"
        q, ids, k1, source = self._calibration_sample(calibration_queries)
        nprobe, reached, exact_seconds, ivf_seconds = self._calibrate(ivf, q, ids, k1, recall, dev)
        lines.append(f"calibrated: nprobe {nprobe} reaches recall@{CALIBRATION_K} {reached:.3f} (target {recall}) "
                     f"on {source}")
        sample = f"{len(q):,} calibration queries on the {dev.upper()}"
        lines.append(f"IVF chosen: {ivf_seconds * 1e3:.1f} ms for {sample}, against {exact_seconds * 1e3:.1f} ms "
                     "for exact search" + (f" (kind={kind!r} given)" if kind in ("ivf", "graph") else ""))
        self._ivf, self._nprobe = ivf, nprobe
        self._cal_q, self._cal_ids = q, ids
        sizes = ivf.built.sizes()
        self._clustered = (len(self._x), float(sizes.max() / sizes.mean()))
        lines += self._build_graph(kind, k1, dev)
        lines.append(self._route())
        r = self._routes
        exact_everywhere = (all(paths[0].endswith("exact") for _, paths in r["table"]) if r.get("table") else
                            r.get("exact_from", NEVER) <= 1 and str(r.get("single", "-exact")).endswith("exact"))
        if kind == "auto" and exact_everywhere:
            had_graph = self._ivf.graph is not None
            self._ivf = self._nprobe = None                      # exact search wins at every batch size measured
            self._routes, self._clustered = {}, (None, None)
            lines.append("kept exact search: faster than the IVF" + (" and the graph" if had_graph else "")
                         + " at every batch size measured, with recall 1.0")
        return lines

    def _graph_line(self, g, reached_gpu, reached_cpu: float, built_by: str | None = None) -> str:
        from ._graph import DEGREE, mlxvs_version
        made = built_by or f"built in {g.build_seconds:.1f} s by mlxvs {mlxvs_version()}"
        gpu = (f"on the GPU, search width {g.itopk} reaches recall@10 {reached_gpu:.3f}; "
               if reached_gpu is not None else "")
        return (f"graph index: CAGRA ({DEGREE} links per vector), {made}; {gpu}on the CPU, ef {g.ef} reaches "
                f"{reached_cpu:.3f} (target {self.recall}, on the same sample)")

    def _calibrate_graph(self, g, k1: int, dev: str):
        """The graph's search widths for the target: on the GPU (where mlxvs can search it) and on the CPU.
        Returns their recalls (None for the GPU where it cannot)."""
        reached_gpu = None
        if g.on_gpu and dev == "gpu":
            g.itopk, reached_gpu, _, _ = self._calibrate(self._ivf, self._cal_q, self._cal_ids, k1, self.recall, dev,
                                                         graph="gpu")
        g.ef, reached_cpu, _, _ = self._calibrate(self._ivf, self._cal_q, self._cal_ids, k1, self.recall, dev,
                                                  graph="cpu")
        return reached_gpu, reached_cpu

    def _build_graph(self, kind: str, k1: int, dev: str) -> list[str]:
        """With kind "auto" or "graph": a CAGRA graph alongside the IVF's lists, its search width calibrated to the
        target on the calibration sample, for the batch sizes routing finds it faster. Returns the plan's lines."""
        from ._graph import Graph, why_not
        if kind not in ("auto", "graph"):
            return []
        on_gpu = self._gpu if not self._too_big and self._device != "cpu" else None
        reason = why_not(self._ivf.metric, on_gpu)
        if reason:
            if kind == "graph":
                raise RuntimeError(f"kind='graph', but {reason.removeprefix('no graph index: ')}")
            return [reason] if self._ivf.metric in ("l2", "ip") else []
        g = Graph(self._x, self._ivf.metric, self._storage)
        self._ivf.graph = g
        reached_gpu, reached_cpu = self._calibrate_graph(g, k1, dev)
        worst = min(r for r in (reached_gpu, reached_cpu) if r is not None)
        if worst < self.recall:
            self._ivf.graph = None
            if kind == "graph":
                raise RuntimeError(f"kind='graph', but the graph's widest search reached recall {worst:.3f}, below "
                                   f"the target {self.recall}")
            return [f"no graph index: its widest search reached recall@10 {worst:.3f}, below the target"]
        return [self._graph_line(g, reached_gpu, reached_cpu)]

    def _build_device(self) -> str:
        """Where building runs (k-means, list assignment): "gpu" (MLX), "torch" or "cpu"."""
        if self._gpu is None or self._device == "cpu":
            return "cpu"
        return "torch" if self._gpu == "torch" else "gpu"

    @staticmethod
    def _storage_line(st: str, build_device: str) -> str:
        if st == "uint8":
            return "lists stored as bytes (every coordinate an integer 0-255: lossless)"
        return ("lists stored as float16 on the GPU (recall moves by at most about 0.001) and float32 on the CPU"
                if build_device != "cpu" else "lists stored as float32")

    def _calibration_sample(self, calibration_queries=None):
        """The calibration sample: up to 1,000 of the given queries, or of the index's own vectors (each one's
        own match then left out). Returns (queries, their ids or None, k to search, the plan's words)."""
        from ._ivf import CALIBRATION_K, CALIBRATION_SAMPLE
        rng = np.random.default_rng(0)
        n = len(self._x)
        if calibration_queries is not None:
            cq = self._checked_queries(calibration_queries)
            cq = cq[np.sort(rng.choice(len(cq), size=min(len(cq), CALIBRATION_SAMPLE), replace=False))]
            q = _normalized(cq, "calibration_queries") if self.metric == "cosine" else cq
            return q, None, min(CALIBRATION_K, n), f"{len(q):,} of the given calibration queries"
        ids = np.sort(rng.choice(n, size=min(n, CALIBRATION_SAMPLE), replace=False))
        source = (f"{len(ids):,} of its own vectors (the target holds for queries like them; pass "
                  "calibration_queries= for others)")
        return self._x[ids], ids, min(CALIBRATION_K + 1, n), source

    def _calibrate(self, ivf, q: np.ndarray, ids, k1: int, recall: float, dev: str, graph: str | None = None):
        """nprobe for the target on the sample, its recall, and exact search's and the IVF's time on it; with
        ``graph`` ("gpu" or "cpu"), the graph's search width on that device instead of nprobe."""
        from ._graph import MAX_EF, MAX_ITOPK
        from ._ivf import CALIBRATION_K, calibrate
        exact_search = self._search_gpu if dev == "gpu" else self._search_cpu
        exact = exact_search(q, k1)                                       # the truth, and a warm-up
        t = time.perf_counter()
        exact_search(q, k1)
        exact_seconds = time.perf_counter() - t
        if ids is not None:                                               # leave the vector itself out
            from ._cpu import without_own
            truth = without_own(np.ascontiguousarray(exact, dtype=np.int64), np.ascontiguousarray(ids, dtype=np.int64),
                                CALIBRATION_K)
        else:
            truth = exact
        larger = self.metric in ("cosine", "ip")
        tscores = self._scores(q, truth)
        kth = tscores.min(axis=1) if larger else tscores.max(axis=1)
        if graph == "gpu":
            def search(p):
                return ivf.graph.search(q, k1, max(p, k1))
            bound = MAX_ITOPK
        elif graph == "cpu":
            def search(p):
                return ivf.graph.search_cpu(q, k1, max(p, k1))
            bound = MAX_EF
        else:
            path = "gpu-list" if dev == "gpu" else "cpu-list"

            def search(p):
                return ivf.search(q, k1, p, path)
            bound = ivf.nlist
        value, reached = calibrate(search, lambda f: self._scores(q, f), kth, ids, recall, bound, larger)
        if graph:
            value = max(value, k1)
        search(value)                                                     # warm the path before timing it
        t = time.perf_counter()
        search(value)
        return value, reached, exact_seconds, time.perf_counter() - t

    def _route(self) -> str:
        """Sets the IVF's routes, from the profile or measured on the calibration sample; returns the plan's line."""
        from ._ivf import ROUTE_BATCHES, describe_routes, measure_routes
        on_gpu = self._gpu is not None and self._device != "cpu"
        if self._profile.ivf:
            self._routes = dict(self._profile.ivf)
            return f"routing from the profile: {describe_routes(self._routes, on_gpu)}"
        # the calibration queries, topped up with distinct base vectors to the largest batch timed
        q, n = self._cal_q, self.n
        want = min(n, max(ROUTE_BATCHES))
        rng = np.random.default_rng(1)
        extra = self._x[np.sort(rng.choice(n, size=max(0, want - len(q)), replace=False))]
        t = time.perf_counter()
        # on PyTorch, two rounds, each path's best rate kept: its GPU was measured slow in the first round of some
        # builds, and a transient slowdown can only understate a path
        exact = None
        if self._kind_arg == "auto":                        # exact search as one more route, per batch size
            from ._ivf import CALIBRATION_K
            exact = {"cpu-exact": lambda qq: self._search_cpu(qq, CALIBRATION_K)}
            if on_gpu and not self._too_big:
                exact["gpu-exact"] = lambda qq: self._search_gpu(qq, CALIBRATION_K)
        self._routes, _ = measure_routes(self._ivf, np.concatenate([q, extra])[:want], self._nprobe, on_gpu,
                                         rounds=2 if self._gpu == "torch" else 1, exact=exact)
        if self._kind_arg == "graph" and self._ivf.graph is not None:
            self._routes["cpu_ivf_from"] = 10 ** 9         # every batch through the graph, on either device
            for key in ("single", "exact_from", "table"):  # the thresholds, forced, decide
                self._routes.pop(key, None)
            if self._ivf.graph.on_gpu:
                self._routes["graph_from"] = 1
        return f"routing measured in {time.perf_counter() - t:.1f} s: {describe_routes(self._routes, on_gpu)}"

    @_changing
    def save(self, path) -> None:
        """Writes the index to ``path`` (a NumPy ``.npz`` file, written at exactly that path): the base
        vectors, the IVF's centroids and lists, the calibration and its sample, the routes, the profile and
        the plan. ``siann.Index.load(path)`` reads it back on any machine. Added vectors still in the buffer
        are merged, and removed ones compacted, first."""
        from . import __version__
        if self._buf is not None or self._dead is not None:
            self._merge("the index was saved")
            self._refresh_plan()
        meta = {"format": FORMAT, "format_version": FORMAT_VERSION, "siann": __version__,
                "saved": time.strftime("%Y-%m-%d"), "machine": _machine(), "metric": self.metric,
                "recall": self.recall, "kind": self.kind, "nprobe": self._nprobe, "storage": self._storage,
                "ivf_metric": self._ivf.metric if self._ivf else None, "routes": self._routes,
                "profile": self._profile.to_dict(), "build_reasons": list(self._build_reasons),
                "calibration": None if self._ivf is None else ("own" if self._cal_ids is not None else "given"),
                "clustered": list(self._clustered), "next_id": self._next_id,
                "build_args": {"kind": self._kind_arg, "nlist": self._nlist_arg, "storage": self._storage_arg},
                "tested_n": self._tested_n}
        graph = self._ivf.graph if self._ivf is not None else None
        if graph is not None:
            from ._graph import DEGREE, mlxvs_version
            meta["graph"] = {"itopk": graph.itopk, "ef": graph.ef, "degree": DEGREE,
                             "builder": f"mlxvs {mlxvs_version()}"}
        bytes_ok = self._storage == "uint8"                 # every coordinate an integer 0-255: stored as bytes
        arrays = {"meta": np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
                  "x": self._x.astype(np.uint8) if bytes_ok else self._x}
        if self._ids is not None:
            arrays["ids"] = self._ids
        if self._ivf is not None:
            arrays["centroids"] = self._ivf.built.centroids
            arrays["assign"] = self._ivf.built.assign.astype(np.int32)
            arrays["calibration_queries"] = self._cal_q
            if self._cal_ids is not None:
                arrays["calibration_ids"] = self._cal_ids.astype(np.int64)
            if graph is not None:
                arrays["graph"] = graph.links()
        elif self._cal_given is not None:
            arrays["calibration_queries"] = self._cal_given
        with open(path, "wb") as f:
            np.savez(f, **arrays)

    @classmethod
    def load(cls, path, *, device: str = "auto", gpu: str = "auto", profile=None,
             remeasure: bool | None = None) -> Index:
        """An index saved by ``index.save(path)``, returning the same neighbors on any machine: its lists,
        its calibrated nprobe (which depends only on the data) and its choice between the IVF and exact
        search are kept. Its routes and profile, which only decide where each search runs, depend on the
        machine: kept where the file was saved on this machine (as named by its processor), otherwise
        re-measured on the saved calibration sample, and the profile replaced by this process's
        (``siann.use_profile``, or siann's own). ``remeasure`` forces either; ``profile`` gives one."""
        from ._ivf import IVF
        from ._profile import Profile, current_profile
        from .ivf_build import lists_from_assign
        _check_device(device)
        _check_profile(profile)
        try:
            data = np.load(path, allow_pickle=False)
            meta = json.loads(bytes(data["meta"]).decode("utf-8"))
        except (ValueError, KeyError, OSError, json.JSONDecodeError) as e:
            raise ValueError(f"{path} is not a saved siann index ({e})") from None
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path} is not a saved siann index")
        if meta["format_version"] > FORMAT_VERSION:
            raise ValueError(f"{path} was saved by siann {meta['siann']} in format {meta['format_version']}; "
                             f"this siann reads format {FORMAT_VERSION} and earlier: upgrade siann")
        same = meta["machine"] == _machine()
        redo = (not same) if remeasure is None else remeasure
        args = meta.get("build_args") or {}                   # format 2 and earlier: inferred from the plan
        given_exact = any(r.startswith("exact search (kind='exact'") for r in meta["build_reasons"])
        self = cls.__new__(cls)
        gpu = self._setup(np.ascontiguousarray(data["x"], dtype=np.float32), meta["metric"], meta["recall"], device,
                          gpu, profile if profile is not None else
                          (current_profile() if redo else Profile.from_dict(meta["profile"])),
                          args.get("kind") or ("exact" if given_exact else "auto"), args.get("nlist"),
                          args.get("storage") or "auto",
                          ids=np.asarray(data["ids"], dtype=np.int64) if "ids" in data else None,
                          clustered=tuple(meta.get("clustered") or (None, None)))
        self._storage, self._nprobe, self._routes = meta["storage"], meta["nprobe"], dict(meta["routes"])
        if meta["kind"] != "ivf" and "calibration_queries" in data:
            self._cal_given = np.ascontiguousarray(data["calibration_queries"], dtype=np.float32)
        self._next_id = max(self._next_id, int(meta.get("next_id") or 0))
        tested = meta.get("tested_n")
        self._tested_n = len(self._x) if tested is None else int(tested)
        reasons = [r if same or not r.startswith(("IVF chosen", "kept exact search")) else
                   r + " (timed on the machine that saved it, and kept, so that results do not change)"
                   for r in meta["build_reasons"] if not r.startswith("routing ")]
        where = "this machine" if same else meta["machine"]
        reasons.insert(0, f"loaded from {Path(path).name}, saved by siann {meta['siann']} on {meta['saved']} "
                          f"on {where}")
        if meta["kind"] == "ivf":
            built = lists_from_assign(data["centroids"], data["assign"])
            self._ivf = IVF(self._x, meta["ivf_metric"], len(built.centroids),
                            "uint8" if self._storage == "uint8" else "float", "cpu", built=built, gpu=self._gpu)
            self._cal_q = np.ascontiguousarray(data["calibration_queries"], dtype=np.float32)
            self._cal_ids = data.get("calibration_ids")
            if self._cal_ids is None and meta.get("calibration") == "given":
                self._cal_given = self._cal_q
            if self._clustered[1] is None:             # format 1: the balance as loaded
                sizes = built.sizes()
                self._clustered = (len(self._x), float(sizes.max() / sizes.mean()))
            if meta.get("graph"):
                from ._graph import Graph, why_not
                g = Graph(self._x, meta["ivf_metric"], self._storage, links=data["graph"])
                g.itopk, g.ef = meta["graph"].get("itopk"), meta["graph"].get("ef")
                self._ivf.graph = g
                if g.ef is None:                            # a file from before the CPU graph search
                    from ._ivf import CALIBRATION_K
                    k1 = (CALIBRATION_K + 1) if self._cal_ids is not None else min(CALIBRATION_K, len(self._x))
                    g.ef, _, _, _ = self._calibrate(self._ivf, self._cal_q, self._cal_ids, k1, self.recall,
                                                    "gpu" if gpu and self._device != "cpu" else "cpu", graph="cpu")
                reason = why_not(meta["ivf_metric"], self._gpu if gpu and self._device != "cpu" else None)
                if reason:
                    reasons.append(f"the graph index searched on the CPU only: {reason.removeprefix('no graph index: ')}")
            if redo:
                reasons.append(self._route().replace("routing measured", "routing re-measured on this machine", 1))
            else:
                from ._ivf import describe_routes
                reasons.append(f"routing as saved, measured on this machine: "
                               f"{describe_routes(self._routes, gpu and self._device != 'cpu')}")
        if profile is None and not same and not redo:
            reasons.append("profile as saved, from another machine (remeasure=False given)")
        elif profile is None and redo and not same:
            reasons.append("profile: this process's, as the saved one was measured on another machine")
        self._build_reasons = tuple(r for r in reasons if not r.startswith("loaded from"))
        self._finish_plan(reasons, device, gpu)
        return self

    def _ivf_path(self, batch: int, device: str | None, approximate: bool = False) -> str:
        """The path a batch takes: by the device (given, or chosen by the profile) and the batch size; exact
        search where measured faster ("cpu-exact", "gpu-exact"), unless ``approximate``."""
        r = {**DEFAULT_ROUTES, **self._routes}
        if approximate:
            r.pop("exact_from", None)
            if str(r.get("single", "")).endswith("exact"):
                r.pop("single")
        graph = self._ivf.graph
        gpu_ok = self._gpu is not None and not self._too_big
        on = device or (self._device if self._device != "auto" else None)
        if r.get("table"):                                   # the measured table: the first path this search may use
            from ._ivf import table_entry
            for path in table_entry(r["table"], batch):
                if (approximate and path.endswith("exact")) or (on and not path.startswith(on)):
                    continue
                if path.startswith("gpu") and not gpu_ok:
                    continue
                if path.endswith("graph") and (graph is None or (path.startswith("gpu") and not graph.on_gpu)):
                    continue
                return path
            r = {k: v for k, v in r.items() if k not in ("table", "single", "exact_from")}   # none: the thresholds
        if batch >= r.get("exact_from", NEVER):              # exact search measured faster from here
            dev = on or ("gpu" if gpu_ok and batch >= r["cpu_below"] else "cpu")
            if dev == "cpu" or gpu_ok:
                return f"{dev}-exact"
        single = r.get("single") if batch == 1 else None
        if single:                                           # single queries: the path measured fastest
            usable = (not single.endswith("graph") or graph is not None) and (
                single.startswith("cpu") or (gpu_ok and (not single.endswith("graph") or graph.on_gpu)))
            if usable and (on is None or single.startswith(on)):
                return single
        dev = device or (self._device if self._device != "auto" else
                         ("cpu" if batch < r["cpu_below"] else "gpu"))
        if dev == "gpu":
            if graph is not None and graph.on_gpu and batch >= r["graph_from"]:
                return "gpu-graph"
            return "gpu-list" if batch >= r["gpu_list_from"] else "gpu-query"
        if graph is not None and batch < r["cpu_ivf_from"]:
            return "cpu-graph"
        if batch == 1:
            return "cpu-split"
        if batch >= r["cpu_list_from"]:
            return "cpu-list"
        return "cpu-query-blas" if batch >= r["cpu_blas_from"] else "cpu-query"

    @property
    def kind(self) -> str:
        """"exact" or "ivf": what the index searches with."""
        return "ivf" if self._ivf is not None else "exact"

    @property
    def nprobe(self) -> int | None:
        """The calibrated number of lists each query probes (None for exact search)."""
        return self._nprobe

    @property
    def n(self) -> int:
        """The number of vectors searched: those built from and added, less those removed."""
        return len(self._x) - self._ndead + (len(self._buf["x"]) if self._buf is not None else 0)

    @property
    def d(self) -> int:
        """The number of dimensions of each vector."""
        return self._x.shape[1]

    @property
    def profile(self):
        """The ``siann.Profile`` whose thresholds this index routes by."""
        return self._profile

    def route(self, batch: int) -> str:
        """The device ("gpu" or "cpu") a search of ``batch`` queries goes to."""
        if self._ivf is not None:
            return self._ivf_path(batch, None).split("-")[0]
        if self._device != "auto":
            return self._device
        t = self._profile.threshold(self.metric)
        return "cpu" if t is None or batch < t else "gpu"

    def __repr__(self) -> str:
        return f"siann.Index({self.n:,} vectors, {self.d} dimensions, metric={self.metric!r}, device={self._device!r})"

    def search(self, xq, k: int = 10, *, device: str | None = None, nprobe: int | None = None) -> SearchResult:
        """The k nearest base vectors to each query in ``xq`` (m, d), or to a single query (d,),
        returned as a batch of one. ``device`` ("gpu" or "cpu") overrides the index's device; ``nprobe``
        overrides the calibrated number of lists an IVF search probes. Searches may run from several threads
        at once, and alongside changes: each searches the index as it was when it began."""
        with self._lock:
            view = self._snapshot()
        return view._search(xq, k, device, nprobe)

    def _snapshot(self) -> Index:
        """A shallow copy: the index's state at this moment. Changes replace the arrays and objects they change
        rather than writing into them, so the copy stays as it was while searched."""
        view = object.__new__(Index)
        view.__dict__.update(self.__dict__)
        return view

    def _search(self, xq, k: int, device: str | None, nprobe: int | None) -> SearchResult:
        q = np.asarray(xq)
        if q.ndim == 1:
            q = q[None, :]
        q = _as_matrix(q, "xq")
        if q.shape[1] != self.d:
            raise ValueError(f"xq has {q.shape[1]} dimensions, the index {self.d}")
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError(f"k must be a positive integer, not {k!r}")
        if nprobe is not None and (not isinstance(nprobe, (int, np.integer)) or nprobe < 1):
            raise ValueError(f"nprobe must be a positive integer, not {nprobe!r}")
        if device not in (None, "gpu", "cpu"):
            raise ValueError(f"device must be 'gpu' or 'cpu', not {device!r}")
        path = self._ivf_path(len(q), device, approximate=nprobe is not None) if self._ivf is not None else None
        dev = device or (path.split("-")[0] if path else self.route(len(q)))
        if dev not in ("gpu", "cpu"):
            raise ValueError(f"device must be 'gpu' or 'cpu', not {dev!r}")
        if dev == "gpu" and self._gpu is None:
            raise RuntimeError("device='gpu', but no GPU is available")
        if dev == "gpu" and self._too_big:
            raise RuntimeError(f"device='gpu', but {self._too_big}")
        if self.metric == "cosine":
            q = _normalized(q, "xq")
        kk = min(int(k), self.n)
        if len(q) == 0 or kk == 0:
            idx, sc = np.empty((len(q), kk), dtype=np.int64), np.empty((len(q), kk), dtype=np.float32)
        elif self._buf is not None or self._dead is not None:
            idx, sc = self._search_changed(q, kk, path, dev, nprobe)
        else:
            if self._ivf is not None and not path.endswith("exact"):
                idx = self._ivf.search(q, kk, int(nprobe or self._nprobe), path)
            else:
                idx = (self._search_gpu if dev == "gpu" else self._search_cpu)(q, kk)
            # best first by the exact scores returned, ties by row
            from ._cpu import rank
            idx, sc = rank(self._x, np.ascontiguousarray(q, dtype=np.float32), np.ascontiguousarray(idx, dtype=np.int64),
                           self._kernel_metric)
            if self._ids is not None:
                idx = np.where(idx >= 0, self._ids[np.maximum(idx, 0)], -1)
        if kk < k:
            pad = (len(q), k - kk)
            idx = np.concatenate([idx, np.full(pad, -1, dtype=np.int64)], axis=1)
            sc = np.concatenate([sc, np.full(pad, self._worst(), dtype=np.float32)], axis=1)
        return SearchResult(idx, sc)

    def _worst(self) -> float:
        return -np.inf if self.metric in ("cosine", "ip") else np.inf

    def _search_changed(self, q: np.ndarray, kk: int, path, dev: str, nprobe):
        """Search with added vectors in the buffer or removed ones masked: the main vectors (FETCH_EXTRA more
        than k, the masked dropped), the buffer exactly, the two merged by score in Rust; queries that lost more
        than FETCH_EXTRA to masked rows are searched again, fetching as many more as are masked."""
        from ._cpu import merge_results
        q = np.ascontiguousarray(q, dtype=np.float32)
        nm, nd = len(self._x), self._ndead

        def main_rows(qq, fetch):
            if nm - nd <= 0:
                return np.full((len(qq), 0), -1, dtype=np.int64)
            if self._ivf is not None and not path.endswith("exact"):
                return self._ivf.search(qq, fetch, int(nprobe or self._nprobe), path)
            return (self._search_gpu if dev == "gpu" else self._search_cpu)(qq, fetch)

        buf = self._buf
        if buf is not None:
            # on the CPU whatever the main search's device: for buffers of 1,000-10,000 vectors, faster than
            # the GPU at nearly every batch size measured (`add_remove.py`)
            brows = _exact_cpu(buf["x"], self._buf_layouts, q, min(kk, len(buf["x"])), self._kernel_metric)

        def merged(qq, rows, br):
            extra = {} if buf is None else {"xb": buf["x"], "brows": np.ascontiguousarray(br), "bids": buf["ids"]}
            return merge_results(self._x, qq, np.ascontiguousarray(rows, dtype=np.int64), kk, self._kernel_metric,
                                 dead=self._dead, ids=self._ids, **extra)

        fetch = min(kk + min(nd, FETCH_EXTRA), nm)
        ids, s, hits = merged(q, main_rows(q, fetch), None if buf is None else brows)
        if nd > FETCH_EXTRA:
            short = np.flatnonzero(hits > fetch - kk)
            if len(short):
                qs = np.ascontiguousarray(q[short])
                ids[short], s[short], _ = merged(qs, main_rows(qs, min(kk + nd, nm)),
                                                 None if buf is None else brows[short])
        return ids, s

    def _row_ids(self) -> np.ndarray:
        return np.arange(len(self._x), dtype=np.int64) if self._ids is None else self._ids

    def _buffer_limit(self) -> int:
        return max(int(BUFFER_SHARE * len(self._x)), BUFFER_MIN)

    @_changing
    def add(self, x, ids=None) -> np.ndarray:
        """Adds vectors ``x`` (m, d), or one (d,), and returns their ids: ``ids`` if given (int64, distinct,
        none already in the index), otherwise continuing from the largest so far. Searches return these ids.
        The vectors are searched exactly until enough of them are added, then merged into the index; an
        IVF finds each one's list now, recalibrates nprobe at each merge, and re-clusters when its lists fall
        out of balance or the index has doubled since its last clustering. ``index.plan`` says what changed."""
        a = np.asarray(x)
        a = _as_matrix(a[None, :] if a.ndim == 1 else a, "x")
        if a.shape[1] != self.d:
            raise ValueError(f"x has {a.shape[1]} dimensions, the index {self.d}")
        if self.metric == "cosine":
            a = _normalized(a, "x")
        if ids is None:
            new = np.arange(self._next_id, self._next_id + len(a), dtype=np.int64)
        else:
            new = np.asarray(ids)
            if new.ndim != 1 or len(new) != len(a) or not np.issubdtype(new.dtype, np.integer):
                raise ValueError(f"ids must be {len(a)} integers, one per vector")
            new = new.astype(np.int64)
            if len(np.unique(new)) != len(new):
                raise ValueError("ids must be distinct")
            # every id so far is below _next_id, so only lower ids need checking against the index's
            taken = new[np.isin(new, self._live_ids())] if len(new) and new.min() < self._next_id else new[:0]
            if len(taken):
                raise ValueError(f"ids already in the index: {', '.join(map(str, taken[:5]))}"
                                 + (", ..." if len(taken) > 5 else ""))
        if len(a) == 0:
            return new
        assign = self._ivf.nearest_lists(a, self._build_device()) if self._ivf is not None else None
        self._buf_append(a, new, assign)
        self._next_id = max(self._next_id, int(new.max()) + 1)
        self._added += len(a)
        if len(self._buf["x"]) >= self._buffer_limit():
            self._merge(f"the buffer reached {self._buffer_limit():,} vectors")
        self._refresh_plan()
        return new

    def _buf_append(self, a: np.ndarray, new: np.ndarray, assign) -> None:
        """Appends to the buffer, into arrays with room to grow (doubling), so that adding one vector at a time
        does not copy the buffer each time."""
        nb = 0 if self._buf is None else len(self._buf["x"])
        need = nb + len(a)
        st = self._buf_store
        if st is None or need > len(st["x"]):
            cap = max(need, 2 * (len(st["x"]) if st is not None else 0), 256)
            grown = {"x": np.empty((cap, self.d), dtype=np.float32), "ids": np.empty(cap, dtype=np.int64),
                     "assign": None if assign is None else np.empty(cap, dtype=np.int64)}
            if nb:
                for key in ("x", "ids", "assign"):
                    if grown[key] is not None:
                        grown[key][:nb] = self._buf[key]
            st = self._buf_store = grown
        st["x"][nb:need], st["ids"][nb:need] = a, new
        if assign is not None:
            st["assign"][nb:need] = assign
        self._buf = {key: None if st[key] is None else st[key][:need] for key in ("x", "ids", "assign")}
        self._buf_layouts = {}

    def _live_ids(self) -> np.ndarray:
        main = self._row_ids() if self._dead is None else self._row_ids()[~self._dead]
        return main if self._buf is None else np.concatenate([main, self._buf["ids"]])

    @_changing
    def remove(self, ids) -> int:
        """Removes the vectors with these ids, and returns how many were removed. An id not in the index is
        refused, and nothing is removed then. Removed vectors are masked and never returned; past a few, the
        index is compacted."""
        gone = np.unique(np.asarray(ids, dtype=np.int64).ravel())
        missing = gone[~np.isin(gone, self._live_ids())]
        if len(missing):
            raise KeyError(f"not in the index: {', '.join(map(str, missing[:5]))}" + (", ..." if len(missing) > 5 else ""))
        main = np.isin(self._row_ids(), gone)
        if main.any():
            self._dead = main if self._dead is None else (self._dead | main)
            self._ndead = int(self._dead.sum())
        if self._buf is not None:
            keep = ~np.isin(self._buf["ids"], gone)
            b = self._buf
            self._buf = None if not keep.any() else {
                "x": b["x"][keep], "ids": b["ids"][keep], "assign": None if b["assign"] is None else b["assign"][keep]}
            self._buf_store = None
            self._buf_layouts = {}
        self._removed += len(gone)
        if self._ndead > DEAD_MAX:
            self._merge(f"more than {DEAD_MAX} removed vectors were masked")
        self._refresh_plan()
        return len(gone)

    @_changing
    def recluster(self) -> None:
        """Merges added vectors, compacts removed ones and, for an IVF, rebuilds its lists from scratch
        (k-means on every vector, the default number of lists for its size), then recalibrates nprobe and
        re-measures routing. siann does this by itself when the lists fall out of balance."""
        self._merge("index.recluster() was called", force=True)
        self._refresh_plan()

    def _merge(self, why: str, force: bool = False) -> None:
        """Folds the buffer into the index and drops removed rows; for an IVF, re-clusters if due (or forced)
        and recalibrates nprobe."""
        live = np.ones(len(self._x), dtype=bool) if self._dead is None else ~self._dead
        had_dead = self._dead is not None
        b = self._buf
        x = self._x[live] if b is None else np.concatenate([self._x[live], b["x"]])
        ids = self._row_ids()[live] if b is None else np.concatenate([self._row_ids()[live], b["ids"]])
        assign = None
        if self._ivf is not None:
            assign = self._ivf.built.assign[live]
            if b is not None:
                assign = np.concatenate([assign, b["assign"]])
        self._x = np.ascontiguousarray(x, dtype=np.float32)
        self._ids = None if np.array_equal(ids, np.arange(len(ids))) else ids
        self._buf, self._buf_store, self._dead, self._ndead = None, None, None, 0
        self._layouts, self._buf_layouts = {}, {}
        self._merges += 1
        if not self._too_big:
            self._too_big = self._gpu_too_big()
            if self._too_big:                          # grown past what the GPU can hold: on the CPU from now
                self._device = "cpu"
        if self._ivf is not None and len(self._x) == 0:
            self._to_empty()
        if self._ivf is None:
            self._last_merge = why
            self._choose_again()
            return
        from ._ivf import IVF, default_nlist, default_storage
        from .ivf_build import build_lists, lists_from_assign
        if self._storage == "uint8" and default_storage(self._x) != "uint8":
            self._storage = "float32"                  # an added vector is not bytes: the lists become float
        storage = "uint8" if self._storage == "uint8" else "float"
        build_dev = self._build_device()
        dev = "cpu" if build_dev == "cpu" else "gpu"
        built = lists_from_assign(self._ivf.built.centroids, assign.astype(np.int64))
        sizes = built.sizes()
        imbalance = float(sizes.max() / sizes.mean())
        n0, imb0 = self._clustered
        n0, imb0 = n0 or len(self._x), imb0 or imbalance
        reason = None
        if force:
            reason = why
        elif imbalance >= REBALANCE * imb0:
            reason = f"as its largest list reached {imbalance:.1f} times the mean, against {imb0:.1f} when clustered"
        elif len(self._x) >= REGROW * n0:
            reason = f"as the index grew to {len(self._x):,} vectors from {n0:,} when clustered"
        bytes_lost = storage != "uint8" and any(r.startswith("lists stored as bytes") for r in self._build_reasons)
        if reason:
            nlist = default_nlist(len(self._x))
            t = time.perf_counter()
            built = build_lists(self._x, nlist, self._ivf.metric, device=build_dev)
            seconds = time.perf_counter() - t
            sizes = built.sizes()
            self._clustered = (len(self._x), float(sizes.max() / sizes.mean()))
            self._reclusters += 1
            self._last_recluster = f"{reason}: {nlist:,} lists in {seconds:.1f} s"
        old_graph = self._ivf.graph
        self._ivf = IVF(self._x, self._ivf.metric, built.nlist, storage, build_dev, built=built, gpu=self._gpu)
        if self._cal_ids is not None:                  # calibrated on its own vectors: a fresh sample of them
            self._cal_q, self._cal_ids, k1, _ = self._calibration_sample()
        else:
            k1 = min(10, len(self._x))
        self._nprobe, reached, _, _ = self._calibrate(self._ivf, self._cal_q, self._cal_ids, k1, self.recall, dev)
        graph_reached = graph_dropped = None
        if old_graph is not None:
            from ._graph import Graph, why_not
            cannot = why_not(self._ivf.metric, self._gpu if dev == "gpu" else None)
            if cannot:
                graph_dropped = f"the graph index dropped at a merge: {cannot.removeprefix('no graph index: ')}"
            else:
                if reason or had_dead or bytes_lost or b is None:
                    g = Graph(self._x, self._ivf.metric, self._storage)   # rows gone or moved: a new graph
                else:
                    g = old_graph.extended(self._x, b["x"])                # new rows linked in, after the others
                self._ivf.graph = g
                graph_reached = self._calibrate_graph(g, k1, dev)
        self._last_merge = f"{why}; nprobe recalibrated"
        routing = self._route() if reason else None

        def now(line: str) -> str:
            """A build line, as the index is now."""
            if reason and line.startswith("approximate search: an IVF of "):
                return (f"approximate search: an IVF of {built.nlist:,} lists (about 4 sqrt(n)), re-clustered in "
                        f"{seconds:.1f} s (siann's k-means, 10 iterations)")
            if line.startswith("calibrated: nprobe ") and " on " in line:
                return (f"calibrated: nprobe {self._nprobe} reaches recall@10 {reached:.3f} (target {self.recall}) "
                        f"on {line.split(' on ', 1)[1]}")
            if bytes_lost and line.startswith("lists stored as bytes"):
                return self._storage_line("float", build_dev) + " (added vectors are not all bytes)"
            if routing and line.startswith("routing "):
                return routing
            if line.startswith("graph index: CAGRA"):
                if graph_dropped:
                    return graph_dropped
                if graph_reached is not None:
                    return self._graph_line(self._ivf.graph, *graph_reached)
            return line
        self._build_reasons = tuple(now(r) for r in self._build_reasons)

    def _to_empty(self) -> None:
        """Every vector removed from an IVF: back to an empty index, the choice made again once it grows."""
        if self._cal_ids is None and self._cal_q is not None:
            self._cal_given = self._cal_q
        self._ivf = self._nprobe = self._cal_q = self._cal_ids = None
        self._routes, self._clustered, self._tested_n = {}, (None, None), 0
        self._build_reasons = (self._empty_line(),)

    def _search_gpu(self, q: np.ndarray, k: int) -> np.ndarray:
        return _exact_gpu(self._x, self._layouts, q, k, self._kernel_metric, self._gpu or "mlx")

    def _search_cpu(self, q: np.ndarray, k: int) -> np.ndarray:
        return _exact_cpu(self._x, self._layouts, q, k, self._kernel_metric)

    def _scores(self, q: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Each returned neighbor's natural score, computed from the vectors in float32 (in Rust, for every
        path alike); infinity, or minus infinity for similarities, where an index is -1."""
        from ._cpu import scores
        return scores(self._x, np.ascontiguousarray(q, dtype=np.float32), np.ascontiguousarray(idx, dtype=np.int64),
                      self._kernel_metric)
