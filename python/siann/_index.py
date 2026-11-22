"""siann.Index: nearest-neighbor search over dense vectors, exact or to a recall target (an IVF calibrated to
it), on the GPU (through siann's GPU backend) or the CPU (siann's Rust kernels, and BLAS through NumPy), routed
by batch size; vectors added and removed; saved and loaded; natural scores; and a plan that says what was
chosen and why."""
from __future__ import annotations

import functools
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._ivf import DEFAULT_ROUTES

METRICS = ("l2", "l1", "linf", "cosine", "ip")
DISTANCES = ("l2", "l1", "linf")          # smaller is nearer; cosine and ip are similarities
FORMAT = "siann-index"
FORMAT_VERSION = 2                        # files of a later version are refused; 2 adds the ids of added vectors
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


@functools.cache
def _gpu_available() -> bool:
    """Whether siann's GPU backend finds a GPU (checked once per process)."""
    try:
        import mlx.core as mx
        return bool(mx.metal.is_available())
    except Exception:  # noqa: BLE001 (any failure to load or query MLX means no usable GPU)
        return False


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


def _exact_gpu(x: np.ndarray, layouts: dict, q: np.ndarray, k: int, metric: str) -> np.ndarray:
    """Exact search of x on the GPU; x's copy there is made on first use and kept in ``layouts``."""
    import mlx.core as mx

    from .exact import mlx_knn
    if "gpu" not in layouts:
        a = mx.array(x)
        mx.eval(a)
        layouts["gpu"] = a
    return np.asarray(mlx_knn(layouts["gpu"], q, k, metric), dtype=np.int64)


def _exact_cpu(x: np.ndarray, layouts: dict, q: np.ndarray, k: int, metric: str) -> np.ndarray:
    """Exact search of x on the CPU: siann's Rust kernel for L1 and L-infinity, BLAS for L2 and inner product."""
    if metric in ("l1", "linf"):
        from ._cpu import FlatIndex
        if "cpu" not in layouts:
            layouts["cpu"] = FlatIndex(x, x.shape[1], metric)
        flat = layouts["cpu"].search(q, k)
        return np.asarray(flat, dtype=np.int64).reshape(len(q), k)
    from .blas_l2 import default_tiles, siann_l2
    if "cpu" not in layouts:
        norms = np.einsum("ij,ij->i", x, x) if metric == "l2" else np.zeros(len(x), dtype=np.float32)
        layouts["cpu"] = np.ascontiguousarray(norms, dtype=np.float32)
    qtile, block = default_tiles(len(q))
    return np.asarray(siann_l2(x, layouts["cpu"], q, k, qtile, block), dtype=np.int64)


def _normalized(a: np.ndarray, name: str) -> np.ndarray:
    lengths = np.linalg.norm(a, axis=1, keepdims=True)
    if (lengths == 0).any():
        raise ValueError(f"cosine is undefined for a zero vector, and {name} holds "
                         f"{int((lengths == 0).sum())}")
    return np.ascontiguousarray(a / lengths, dtype=np.float32)


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
    Each device's copy of the vectors is built the first time a search needs it. ``index.plan`` says
    what was chosen and why.
    """

    def __init__(self, xb, metric: str = "l2", recall: float | None = None, *, device: str = "auto",
                 profile=None, kind: str = "auto", nlist: int | None = None, storage: str = "auto",
                 calibration_queries=None):
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {', '.join(METRICS)}, not {metric!r}")
        if recall is not None and not 0 < recall <= 1:
            raise ValueError(f"recall must be a target between 0 and 1, not {recall!r}")
        if kind not in ("auto", "exact", "ivf"):
            raise ValueError(f"kind must be 'auto', 'exact' or 'ivf', not {kind!r}")
        if kind == "ivf" and recall is None:
            raise ValueError("kind='ivf' needs a recall target, such as recall=0.95")
        if storage not in ("auto", "float32", "float16", "uint8"):
            raise ValueError(f"storage must be 'auto', 'float32', 'float16' or 'uint8', not {storage!r}")
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
        self.recall = recall
        self._ivf = None
        self._nprobe = None
        self._routes = {}
        self._storage = None
        self._cal_q = self._cal_ids = None
        self._device_arg = device
        self._init_changes(len(self._x))
        reasons = []
        if recall is None or kind == "exact":
            reasons.append("exact search (no recall target): the results are the true nearest neighbors"
                           if recall is None else "exact search (kind='exact' given)")
        else:
            reasons += self._build_ivf(recall, kind, nlist, storage, gpu, calibration_queries)
        self._build_reasons = tuple(reasons)
        self._finish_plan(list(reasons), device, gpu)

    def _init_changes(self, n: int, ids=None, clustered_n=None, clustered_imbalance=None) -> None:
        """No vector added or removed yet: ids are row numbers unless given."""
        self._ids = ids                                    # each row's id; None: the row number
        self._next_id = int(ids.max()) + 1 if ids is not None and len(ids) else n
        self._buf = None                                   # added vectors not yet merged: x, ids, lists
        self._buf_layouts: dict[str, object] = {}
        self._dead = None                                  # rows removed, masked until compacted
        self._ndead = 0                                    # how many (the mask's sum, kept: it is searched often)
        self._added = self._removed = self._merges = self._reclusters = 0
        self._last_merge = self._last_recluster = None
        self._clustered = (clustered_n, clustered_imbalance)

    def _finish_plan(self, reasons: list[str], device: str, gpu: bool) -> None:
        """The plan: the build's reasons, then the device's."""
        metric = self.metric
        if device != "auto":
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
        self._finish_plan(list(self._build_reasons) + self._change_lines(), self._device_arg, _gpu_available())

    def _build_ivf(self, recall: float, kind: str, nlist: int | None, storage: str, gpu: bool,
                   calibration_queries=None) -> list[str]:
        """Builds the IVF, calibrates nprobe and, with kind "auto", keeps it only if it is faster than exact
        search on the calibration sample. Returns the plan's lines."""
        from ._ivf import CALIBRATION_K, IVF, default_nlist, default_storage
        n = self.n
        ivf_metric = "l2" if self.metric == "cosine" else self.metric      # cosine: L2 on unit vectors
        nlist = int(nlist) if nlist else default_nlist(n)
        if not 1 <= nlist <= n:
            raise ValueError(f"nlist must be between 1 and the number of vectors ({n:,}), not {nlist}")
        st = default_storage(self._x) if storage == "auto" else storage
        if st == "uint8" and default_storage(self._x) != "uint8":
            raise ValueError("storage='uint8' needs every coordinate to be an integer from 0 to 255")
        self._storage = st
        lines = []
        build_device = "gpu" if gpu and self._device != "cpu" else "cpu"
        ivf = IVF(self._x, ivf_metric, nlist, "uint8" if st == "uint8" else "float", build_device)
        lines.append(f"approximate search: an IVF of {ivf.nlist:,} lists"
                     + (" (about 4 sqrt(n))" if not nlist or nlist == default_nlist(n) else "")
                     + f", built in {ivf.build_seconds:.1f} s (siann's k-means, 10 iterations)")
        lines.append("lists stored as bytes (every coordinate an integer 0-255: lossless)" if st == "uint8" else
                     "lists stored as float16 on the GPU (recall moves by at most about 0.001) and float32 on the CPU"
                     if build_device == "gpu" else "lists stored as float32")
        dev = "gpu" if build_device == "gpu" else "cpu"
        q, ids, k1, source = self._calibration_sample(calibration_queries)
        nprobe, reached, exact_seconds, ivf_seconds = self._calibrate(ivf, q, ids, k1, recall, dev)
        lines.append(f"calibrated: nprobe {nprobe} reaches recall@{CALIBRATION_K} {reached:.3f} (target {recall}) "
                     f"on {source}")
        sample = f"{len(q):,} calibration queries on the {dev.upper()}"
        if kind == "auto" and ivf_seconds >= exact_seconds:
            lines.append(f"kept exact search: the IVF took {ivf_seconds * 1e3:.1f} ms for {sample}, exact search "
                         f"{exact_seconds * 1e3:.1f} ms, with recall 1.0")
            return lines
        lines.append(f"IVF chosen: {ivf_seconds * 1e3:.1f} ms for {sample}, against {exact_seconds * 1e3:.1f} ms "
                     "for exact search" + (" (kind='ivf' given)" if kind == "ivf" else ""))
        self._ivf, self._nprobe = ivf, nprobe
        self._cal_q, self._cal_ids = q, ids
        sizes = ivf.built.sizes()
        self._clustered = (len(self._x), float(sizes.max() / sizes.mean()))
        lines.append(self._route())
        return lines

    def _calibration_sample(self, calibration_queries=None):
        """The calibration sample: up to 1,000 of the given queries, or of the index's own vectors (each one's
        own match then left out). Returns (queries, their ids or None, k to search, the plan's words)."""
        from ._ivf import CALIBRATION_K, CALIBRATION_SAMPLE
        rng = np.random.default_rng(0)
        n = len(self._x)
        if calibration_queries is not None:
            cq = _as_matrix(calibration_queries, "calibration_queries")
            if cq.shape[1] != self.d or len(cq) == 0:
                raise ValueError(f"calibration_queries must be a non-empty (m, {self.d}) array")
            cq = cq[np.sort(rng.choice(len(cq), size=min(len(cq), CALIBRATION_SAMPLE), replace=False))]
            q = _normalized(cq, "calibration_queries") if self.metric == "cosine" else cq
            return q, None, min(CALIBRATION_K, n), f"{len(q):,} of the given calibration queries"
        ids = np.sort(rng.choice(n, size=min(n, CALIBRATION_SAMPLE), replace=False))
        source = (f"{len(ids):,} of its own vectors (the target holds for queries like them; pass "
                  "calibration_queries= for others)")
        return self._x[ids], ids, min(CALIBRATION_K + 1, n), source

    def _calibrate(self, ivf, q: np.ndarray, ids, k1: int, recall: float, dev: str):
        """nprobe for the target on the sample, its recall, and exact search's and the IVF's time on it."""
        from ._ivf import CALIBRATION_K, calibrate
        exact_search = self._search_gpu if dev == "gpu" else self._search_cpu
        exact = exact_search(q, k1)                                       # the truth, and a warm-up
        t = time.perf_counter()
        exact_search(q, k1)
        exact_seconds = time.perf_counter() - t
        if ids is not None:
            rows = [r[r != own][:CALIBRATION_K] for r, own in zip(exact, ids)]
            truth = np.array([np.concatenate([r, np.full(CALIBRATION_K - len(r), -1)]) for r in rows], dtype=np.int64)
        else:
            truth = exact
        larger = self.metric in ("cosine", "ip")
        tscores = self._scores(q, truth)
        kth = tscores.min(axis=1) if larger else tscores.max(axis=1)
        path = "gpu-list" if dev == "gpu" else "cpu-list"
        nprobe, reached = calibrate(lambda p: ivf.search(q, k1, p, path), lambda f: self._scores(q, f), kth, ids,
                                    recall, ivf.nlist, larger)
        ivf.search(q, k1, nprobe, path)                                   # warm the path before timing it
        t = time.perf_counter()
        ivf.search(q, k1, nprobe, path)
        return nprobe, reached, exact_seconds, time.perf_counter() - t

    def _route(self) -> str:
        """Sets the IVF's routes, from the profile or measured on the calibration sample; returns the plan's line."""
        from ._ivf import ROUTE_BATCHES, describe_routes, measure_routes
        on_gpu = _gpu_available() and self._device != "cpu"
        if self._profile.ivf:
            self._routes = dict(self._profile.ivf)
            return f"routing from the profile: {describe_routes(self._routes, on_gpu)}"
        # the calibration queries, topped up with distinct base vectors to the largest batch timed
        q, n = self._cal_q, self.n
        want = min(n, max(ROUTE_BATCHES))
        rng = np.random.default_rng(1)
        extra = self._x[np.sort(rng.choice(n, size=max(0, want - len(q)), replace=False))]
        t = time.perf_counter()
        self._routes, _ = measure_routes(self._ivf, np.concatenate([q, extra])[:want], self._nprobe, on_gpu)
        return f"routing measured in {time.perf_counter() - t:.1f} s: {describe_routes(self._routes, on_gpu)}"

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
                "clustered": list(self._clustered), "next_id": self._next_id}
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
        with open(path, "wb") as f:
            np.savez(f, **arrays)

    @classmethod
    def load(cls, path, *, device: str = "auto", profile=None, remeasure: bool | None = None) -> Index:
        """An index saved by ``index.save(path)``, returning the same neighbors on any machine: its lists,
        its calibrated nprobe (which depends only on the data) and its choice between the IVF and exact
        search are kept. Its routes and profile, which only decide where each search runs, depend on the
        machine: kept where the file was saved on this machine (as named by its processor), otherwise
        re-measured on the saved calibration sample, and the profile replaced by this process's
        (``siann.use_profile``, or siann's own). ``remeasure`` forces either; ``profile`` gives one."""
        from ._ivf import IVF
        from ._profile import Profile, current_profile
        from .ivf_build import lists_from_assign
        if device not in ("auto", "gpu", "cpu"):
            raise ValueError(f"device must be 'auto', 'gpu' or 'cpu', not {device!r}")
        if profile is not None and not isinstance(profile, Profile):
            raise TypeError("profile must be a siann.Profile (from siann.tune or siann.Profile.load)")
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
        gpu = _gpu_available()
        if device == "gpu" and not gpu:
            raise RuntimeError("device='gpu', but no GPU is available")
        same = meta["machine"] == _machine()
        redo = (not same) if remeasure is None else remeasure
        self = cls.__new__(cls)
        self.metric = meta["metric"]
        self._x = np.ascontiguousarray(data["x"], dtype=np.float32)
        self._kernel_metric = "ip" if self.metric == "cosine" else self.metric
        self._profile = profile if profile is not None else (current_profile() if redo else
                                                             Profile.from_dict(meta["profile"]))
        self._device = device if device != "auto" else ("auto" if gpu else "cpu")
        self._layouts = {}
        self.recall = meta["recall"]
        self._storage = meta["storage"]
        self._nprobe = meta["nprobe"]
        self._routes = dict(meta["routes"])
        self._ivf = self._cal_q = self._cal_ids = None
        self._device_arg = device
        clustered = meta.get("clustered") or [None, None]
        self._init_changes(len(self._x), np.asarray(data["ids"], dtype=np.int64) if "ids" in data else None,
                           clustered[0], clustered[1])
        self._next_id = max(self._next_id, int(meta.get("next_id") or 0))
        reasons = [r if same or not r.startswith(("IVF chosen", "kept exact search")) else
                   r + " (timed on the machine that saved it, and kept, so that results do not change)"
                   for r in meta["build_reasons"] if not r.startswith("routing ")]
        where = "this machine" if same else meta["machine"]
        reasons.insert(0, f"loaded from {Path(path).name}, saved by siann {meta['siann']} on {meta['saved']} "
                          f"on {where}")
        if meta["kind"] == "ivf":
            built = lists_from_assign(data["centroids"], data["assign"])
            self._ivf = IVF(self._x, meta["ivf_metric"], len(built.centroids),
                            "uint8" if self._storage == "uint8" else "float", "cpu", built=built)
            self._cal_q = np.ascontiguousarray(data["calibration_queries"], dtype=np.float32)
            self._cal_ids = data.get("calibration_ids")
            if self._clustered[1] is None:             # format 1: the balance as loaded
                sizes = built.sizes()
                self._clustered = (len(self._x), float(sizes.max() / sizes.mean()))
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

    def _ivf_path(self, batch: int, device: str | None) -> str:
        """The IVF path a batch takes: by the device (given, or chosen by the profile) and the batch size."""
        r = {**DEFAULT_ROUTES, **self._routes}
        dev = device or (self._device if self._device != "auto" else
                         ("cpu" if batch < r["cpu_below"] else "gpu"))
        if dev == "gpu":
            return "gpu-list" if batch >= r["gpu_list_from"] else "gpu-query"
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
        overrides the calibrated number of lists an IVF search probes."""
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
        path = self._ivf_path(len(q), device) if self._ivf is not None else None
        dev = device or (path.split("-")[0] if path else self.route(len(q)))
        if dev not in ("gpu", "cpu"):
            raise ValueError(f"device must be 'gpu' or 'cpu', not {dev!r}")
        if dev == "gpu" and not _gpu_available():
            raise RuntimeError("device='gpu', but no GPU is available")
        if self.metric == "cosine":
            q = _normalized(q, "xq")
        kk = min(int(k), self.n)
        if len(q) == 0:
            idx, sc = np.empty((0, kk), dtype=np.int64), np.empty((0, kk), dtype=np.float32)
        elif self._buf is not None or self._dead is not None:
            idx, sc = self._search_changed(q, kk, path, dev, nprobe)
        else:
            if self._ivf is not None:
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
            if self._ivf is not None:
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
        live = self._live_ids()
        if ids is None:
            new = np.arange(self._next_id, self._next_id + len(a), dtype=np.int64)
        else:
            new = np.asarray(ids)
            if new.ndim != 1 or len(new) != len(a) or not np.issubdtype(new.dtype, np.integer):
                raise ValueError(f"ids must be {len(a)} integers, one per vector")
            new = new.astype(np.int64)
            if len(np.unique(new)) != len(new):
                raise ValueError("ids must be distinct")
            taken = new[np.isin(new, live)]
            if len(taken):
                raise ValueError(f"ids already in the index: {', '.join(map(str, taken[:5]))}"
                                 + (", ..." if len(taken) > 5 else ""))
        if len(a) == 0:
            return new
        assign = None
        if self._ivf is not None:
            from .ivf_build import assign_metric, nearest
            dev = "gpu" if _gpu_available() and self._device != "cpu" else "cpu"
            assign = nearest(a, self._ivf.built.centroids, assign_metric(self._ivf.metric), dev)
        if self._buf is None:
            self._buf = {"x": a, "ids": new, "assign": assign}
        else:
            b = self._buf
            self._buf = {"x": np.concatenate([b["x"], a]), "ids": np.concatenate([b["ids"], new]),
                         "assign": None if assign is None else np.concatenate([b["assign"], assign])}
        self._buf_layouts = {}
        self._next_id = max(self._next_id, int(new.max()) + 1)
        self._added += len(a)
        if len(self._buf["x"]) >= self._buffer_limit():
            self._merge(f"the buffer reached {self._buffer_limit():,} vectors")
        self._refresh_plan()
        return new

    def _live_ids(self) -> np.ndarray:
        main = self._row_ids() if self._dead is None else self._row_ids()[~self._dead]
        return main if self._buf is None else np.concatenate([main, self._buf["ids"]])

    def remove(self, ids) -> int:
        """Removes the vectors with these ids, and returns how many were removed. An id not in the index is
        refused, and nothing is removed then. Removed vectors are masked and never returned; past a few, the
        index is compacted."""
        gone = np.unique(np.asarray(ids, dtype=np.int64).ravel())
        missing = gone[~np.isin(gone, self._live_ids())]
        if len(missing):
            raise KeyError(f"not in the index: {', '.join(map(str, missing[:5]))}" + (", ..." if len(missing) > 5 else ""))
        if len(gone) >= self.n:
            raise ValueError("removing every vector would leave the index empty; build a new one instead")
        main = np.isin(self._row_ids(), gone)
        if main.any():
            self._dead = main if self._dead is None else (self._dead | main)
            self._ndead = int(self._dead.sum())
        if self._buf is not None:
            keep = ~np.isin(self._buf["ids"], gone)
            b = self._buf
            self._buf = None if not keep.any() else {
                "x": b["x"][keep], "ids": b["ids"][keep], "assign": None if b["assign"] is None else b["assign"][keep]}
            self._buf_layouts = {}
        self._removed += len(gone)
        if self._ndead > DEAD_MAX:
            self._merge(f"more than {DEAD_MAX} removed vectors were masked")
        self._refresh_plan()
        return len(gone)

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
        self._buf, self._dead, self._ndead = None, None, 0
        self._layouts, self._buf_layouts = {}, {}
        self._merges += 1
        if self._ivf is None:
            self._last_merge = why
            return
        from ._ivf import IVF, default_nlist, default_storage
        from .ivf_build import build_lists, lists_from_assign
        if self._storage == "uint8" and default_storage(self._x) != "uint8":
            self._storage = "float32"                  # an added vector is not bytes: the lists become float
        storage = "uint8" if self._storage == "uint8" else "float"
        dev = "gpu" if _gpu_available() and self._device != "cpu" else "cpu"
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
            built = build_lists(self._x, nlist, self._ivf.metric, device=dev)
            seconds = time.perf_counter() - t
            sizes = built.sizes()
            self._clustered = (len(self._x), float(sizes.max() / sizes.mean()))
            self._reclusters += 1
            self._last_recluster = f"{reason}: {nlist:,} lists in {seconds:.1f} s"
        self._ivf = IVF(self._x, self._ivf.metric, built.nlist, storage, dev, built=built)
        if self._cal_ids is not None:                  # calibrated on its own vectors: a fresh sample of them
            self._cal_q, self._cal_ids, k1, _ = self._calibration_sample()
        else:
            k1 = min(10, len(self._x))
        self._nprobe, reached, _, _ = self._calibrate(self._ivf, self._cal_q, self._cal_ids, k1, self.recall, dev)
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
                return ("lists stored as float16 on the GPU and float32 on the CPU" if dev == "gpu" else
                        "lists stored as float32") + " (added vectors are not all bytes)"
            if routing and line.startswith("routing "):
                return routing
            return line
        self._build_reasons = tuple(now(r) for r in self._build_reasons)

    def _search_gpu(self, q: np.ndarray, k: int) -> np.ndarray:
        return _exact_gpu(self._x, self._layouts, q, k, self._kernel_metric)

    def _search_cpu(self, q: np.ndarray, k: int) -> np.ndarray:
        return _exact_cpu(self._x, self._layouts, q, k, self._kernel_metric)

    def _scores(self, q: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Each returned neighbor's natural score, computed from the vectors in float32 (in Rust, for every
        path alike); infinity, or minus infinity for similarities, where an index is -1."""
        from ._cpu import scores
        return scores(self._x, np.ascontiguousarray(q, dtype=np.float32), np.ascontiguousarray(idx, dtype=np.int64),
                      self._kernel_metric)
