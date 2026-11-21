"""siann.Index: exact search over dense vectors, on the GPU (through siann's GPU backend) or the CPU (BLAS through NumPy, and
siann's Rust kernels), with natural scores and a plan that says what was chosen and why."""
from __future__ import annotations

import functools
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

METRICS = ("l2", "l1", "linf", "cosine", "ip")
DISTANCES = ("l2", "l1", "linf")          # smaller is nearer; cosine and ip are similarities
FORMAT = "siann-index"
FORMAT_VERSION = 1                        # files of a later version are refused


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
        reasons = []
        if recall is None or kind == "exact":
            reasons.append("exact search (no recall target): the results are the true nearest neighbors"
                           if recall is None else "exact search (kind='exact' given)")
        else:
            reasons += self._build_ivf(recall, kind, nlist, storage, gpu, calibration_queries)
        self._build_reasons = tuple(reasons)
        self._finish_plan(list(reasons), device, gpu)

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
        self.plan = Plan(len(self._x), self._x.shape[1], metric, "ivf" if self._ivf else "exact", self._device,
                         tuple(reasons))

    def _build_ivf(self, recall: float, kind: str, nlist: int | None, storage: str, gpu: bool,
                   calibration_queries=None) -> list[str]:
        """Builds the IVF, calibrates nprobe and, with kind "auto", keeps it only if it is faster than exact
        search on the calibration sample. Returns the plan's lines."""
        from ._ivf import (
            CALIBRATION_K,
            CALIBRATION_SAMPLE,
            IVF,
            calibrate,
            default_nlist,
            default_storage,
        )
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
        # calibration: the given queries, or a sample of the index's own vectors (each one's own match
        # left out)
        rng = np.random.default_rng(0)
        if calibration_queries is not None:
            cq = _as_matrix(calibration_queries, "calibration_queries")
            if cq.shape[1] != self.d or len(cq) == 0:
                raise ValueError(f"calibration_queries must be a non-empty (m, {self.d}) array")
            cq = cq[np.sort(rng.choice(len(cq), size=min(len(cq), CALIBRATION_SAMPLE), replace=False))]
            q, ids = (_normalized(cq, "calibration_queries") if self.metric == "cosine" else cq), None
            k1 = min(CALIBRATION_K, n)
            source = f"{len(q):,} of the given calibration queries"
        else:
            ids = np.sort(rng.choice(n, size=min(n, CALIBRATION_SAMPLE), replace=False))
            q = self._x[ids]
            k1 = min(CALIBRATION_K + 1, n)
            source = (f"{len(q):,} of its own vectors (the target holds for queries like them; pass "
                      "calibration_queries= for others)")
        dev = "gpu" if build_device == "gpu" else "cpu"
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
        ivf_seconds = time.perf_counter() - t
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
        lines.append(self._route())
        return lines

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
        the plan. ``siann.Index.load(path)`` reads it back on any machine."""
        from . import __version__
        meta = {"format": FORMAT, "format_version": FORMAT_VERSION, "siann": __version__,
                "saved": time.strftime("%Y-%m-%d"), "machine": _machine(), "metric": self.metric,
                "recall": self.recall, "kind": self.kind, "nprobe": self._nprobe, "storage": self._storage,
                "ivf_metric": self._ivf.metric if self._ivf else None, "routes": self._routes,
                "profile": self._profile.to_dict(), "build_reasons": list(self._build_reasons),
                "calibration": None if self._ivf is None else ("own" if self._cal_ids is not None else "given")}
        bytes_ok = self._storage == "uint8"                 # every coordinate an integer 0-255: stored as bytes
        arrays = {"meta": np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
                  "x": self._x.astype(np.uint8) if bytes_ok else self._x}
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
        from ._ivf import DEFAULT_ROUTES
        r = {**DEFAULT_ROUTES, **self._routes}
        dev = device or (self._device if self._device != "auto" else
                         ("cpu" if batch < r["cpu_below"] else "gpu"))
        if dev == "gpu":
            return "gpu-list" if batch >= r["gpu_list_from"] else "gpu-query"
        return "cpu-split" if batch == 1 else ("cpu-list" if batch >= r["cpu_list_from"] else "cpu-query")

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
        elif self._ivf is not None:
            idx = self._ivf.search(q, kk, int(nprobe or self._nprobe), self._ivf_path(len(q), device))
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
