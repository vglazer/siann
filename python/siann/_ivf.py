"""The IVF behind siann.Index when a recall target is given: lists built once (siann.ivf_build), each
device's copy made on first use, nprobe calibrated to the target on sample queries (the given calibration
queries, or the index's own vectors), and each search routed by its batch size along routes timed on them."""
from __future__ import annotations

import time

import numpy as np

CALIBRATION_SAMPLE = 1000
CALIBRATION_K = 10
# For L2, cosine and inner product, the CPU's coarse step for a batch can go through BLAS: one matrix
# multiply of the batch against the centroids, whose products Rust ranks as its own coarse step ranks
# distances, with the same results. The list-major scan always takes it; the query-major one takes it as
# the path "cpu-query-blas", from the batch size routing measures it faster (route cpu_blas_from), as that
# depends on the machine: on 4 threads it pays from about 32 queries, on 16 hardly at all (BENCHMARKS,
# "The coarse step"). For L1 and L-infinity each query finds its lists in Rust.
BLAS_CHUNK = 2048        # queries per matrix multiply: products of at most 2,048 x nlist floats


def default_nlist(n: int) -> int:
    """About 4 sqrt(n) lists (4,000 for a million vectors, as every measurement used), with at least 39
    vectors per list (k-means needs enough to place each centroid) and at least one list."""
    return int(max(1, min(round(4 * np.sqrt(n)), n // 39)))


def default_storage(x: np.ndarray) -> str:
    """uint8 where every coordinate is an integer from 0 to 255 (lossless), else float."""
    return "uint8" if (x.min() >= 0 and x.max() <= 255 and np.array_equal(np.rint(x), x)) else "float"


class IVF:
    """The lists and each device's copy of them. ``metric`` is the IVF's: "l1", "l2", "linf" or "ip"
    (cosine arrives as "l2" on unit vectors)."""

    def __init__(self, x: np.ndarray, metric: str, nlist: int, storage: str, device_for_build: str,
                 built=None):
        """Builds the lists, or takes them from ``built`` (a ``siann.ivf_build.Lists``, as when loading)."""
        from .ivf_build import build_lists
        self.x, self.metric, self.storage = x, metric, storage
        t = time.perf_counter()
        self.built = built if built is not None else build_lists(x, nlist, metric, device=device_for_build)
        self.build_seconds = time.perf_counter() - t
        self._gpu = self._cpu = self._coarse = None

    @property
    def nlist(self) -> int:
        return self.built.nlist

    def gpu(self):
        if self._gpu is None:
            from .ivf_gpu import GpuIVF
            lists = "uint8" if self.storage == "uint8" else "float16"      # float16 for GPU batches
            self._gpu = GpuIVF(self.x, self.nlist, self.metric, lists, built=self.built)
        return self._gpu

    def cpu(self):
        if self._cpu is None:
            from ._cpu import IvfIndex
            d = self.x.shape[1]
            if self.storage == "uint8":
                self._cpu = IvfIndex(self.x, self.built.centroids, d, self.metric,
                                     np.ascontiguousarray(self.x.astype(np.uint8)), "uint8", lists=self.built.assign)
            else:
                self._cpu = IvfIndex(self.x, self.built.centroids, d, self.metric, lists=self.built.assign)
        return self._cpu

    def centroids(self) -> np.ndarray:
        """The centroids as C-contiguous float32, for the coarse step's matrix multiply."""
        if self._coarse is None:
            self._coarse = np.ascontiguousarray(self.built.centroids, dtype=np.float32)
        return self._coarse

    def search(self, q: np.ndarray, k: int, nprobe: int, path: str, coarse: str = "auto") -> np.ndarray:
        """The k nearest by one path: "gpu-query", "gpu-list", "cpu-query", "cpu-query-blas" (its coarse
        step through BLAS), "cpu-list" or "cpu-split" (one query at a time, spread over 4 threads); -1 where
        fewer than k were found. ``coarse`` ("auto", "blas" or "rust") overrides how the CPU's batch paths
        find each query's lists ("auto": as the path says; the list-major scan through BLAS).
        All return the same neighbors."""
        nprobe = min(nprobe, self.nlist)
        if path == "gpu-query":
            return np.asarray(self.gpu().search(q, k, nprobe), dtype=np.int64)
        if path == "gpu-list":
            return np.asarray(self.gpu().search_list_major(q, k, nprobe), dtype=np.int64)
        index = self.cpu()
        blas = path != "cpu-split" and self.metric in ("l2", "ip") and (
            coarse == "blas" or (coarse == "auto" and path in ("cpu-query-blas", "cpu-list")))
        if blas:
            # the coarse step's products by BLAS, BLAS_CHUNK queries at a time (each query's result
            # depends on it alone), ranked in Rust as its own coarse step ranks distances
            c = self.centroids()
            scan = index.search_list_major if path == "cpu-list" else index.search
            flat = np.concatenate([scan(q[s:s + BLAS_CHUNK], k, nprobe, products=q[s:s + BLAS_CHUNK] @ c.T)
                                   for s in range(0, len(q), BLAS_CHUNK)])
        elif path == "cpu-list":
            flat = index.search_list_major(q, k, nprobe)
        else:
            flat = index.search(q, k, nprobe, split=4 if path == "cpu-split" else 1)
        out = np.asarray(flat, dtype=np.int64).reshape(len(q), k)
        out[out >= len(self.x)] = -1                                     # the Rust IVF's "none found"
        return out


def recall_of(found: np.ndarray, truth_scores_kth: np.ndarray, found_scores: np.ndarray, larger_better: bool) -> float:
    """The share of found neighbors at least as near as the true k-th nearest (ties count as found)."""
    ok = found >= 0
    if larger_better:
        ok &= found_scores >= truth_scores_kth[:, None] - 1e-6 * np.abs(truth_scores_kth[:, None]) - 1e-9
    else:
        ok &= found_scores <= truth_scores_kth[:, None] + 1e-6 * np.abs(truth_scores_kth[:, None]) + 1e-9
    return float(ok.mean())


def calibrate(search, scores, truth_kth: np.ndarray, sample_ids: np.ndarray, target: float, nlist: int,
              larger_better: bool, k: int = CALIBRATION_K):
    """The smallest nprobe at which ``search(nprobe)`` reaches ``target`` recall@k on the sample: nprobe
    doubles until the target is reached, then a bisection narrows it. With ``sample_ids`` (the sample is
    the index's own vectors), the search returns k + 1 neighbors and each vector itself is left out.
    Returns (nprobe, recall at it)."""
    def recall_at(nprobe):
        found = search(nprobe)
        if sample_ids is not None:
            rows = []
            for r, own in zip(found, sample_ids):
                r = r[r != own][:k]                                      # leave the vector itself out
                rows.append(np.concatenate([r, np.full(k - len(r), -1)]))
            found = np.array(rows, dtype=np.int64)
        return recall_of(found, truth_kth, scores(found), larger_better)

    hi, r_hi = 1, recall_at(1)
    lo = 0
    while r_hi < target and hi < nlist:
        lo, hi = hi, min(2 * hi, nlist)
        r_hi = recall_at(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        r_mid = recall_at(mid)
        if r_mid >= target:
            hi, r_hi = mid, r_mid
        else:
            lo = mid
    return hi, r_hi


ROUTE_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
NEVER = 10 ** 9          # a threshold no batch reaches
CPU_GIVE_UP = 0.3        # seconds: a CPU call this slow at one batch size is not timed at larger ones


DEFAULT_ROUTES = {"cpu_below": 2, "gpu_list_from": 1000, "cpu_list_from": NEVER, "cpu_blas_from": NEVER}
MARGIN = 1.1             # a measured path replaces the default one only where it is this much faster


def crossover_with_margin(a: dict[int, float], b: dict[int, float], default: int) -> int:
    """The batch size from which path b is taken over path a, where by default b is taken from ``default``
    up. At a measured batch, b wins from ``default`` up unless a is ``MARGIN`` times faster, and below it
    only if b is ``MARGIN`` times faster. b is taken from the smallest measured batch from which it wins at
    every larger one; between that batch and the measured one below it (or past the largest measured
    batch, where a wins), the default decides."""
    batches = sorted(a)
    wins = {x: (b[x] * MARGIN >= a[x]) if x >= default else (b[x] >= MARGIN * a[x]) for x in batches}
    i = next((i for i in range(len(batches)) if all(wins[y] for y in batches[i:])), len(batches))
    if i == 0:
        return batches[0]
    upper = batches[i] if i < len(batches) else NEVER
    return max(batches[i - 1] + 1, min(upper, default))


def measure_routes(ivf: IVF, q: np.ndarray, nprobe: int, gpu: bool, k: int = CALIBRATION_K) -> tuple[dict, dict]:
    """Where this IVF's searches should run, measured: each path timed by itself (the median call after a
    warm-up) at batch sizes doubling from 1 to 4,096 queries of ``q``. The CPU is no
    longer timed once the GPU has beaten it at two sizes in a row, or once one of its calls takes 0.3 s.
    Returns the thresholds (``cpu_below``: from which batch size the GPU's best scan beats the CPU's at every
    larger size measured; ``gpu_list_from`` and ``cpu_list_from``: from which the list-major scan beats the
    query-major one; ``cpu_blas_from``, for L2, cosine and inner product: from which the CPU's query-major
    scan does its coarse step through BLAS), each departing from ``DEFAULT_ROUTES`` only where the other path is at least 10%
    faster (about the timings' run-to-run spread), and the rates, as {(path, batch): rate}."""

    def time_paths(b, paths):
        """Each path's rate at batch b and its slowest call. Each path is timed by itself, its calls back to
        back as in a stream of searches (between calls on the other device, a GPU call can run 40% slower):
        a warm-up call, then calls until there are at least 5 (3 from 64 queries) taking 20 ms in all (at
        most 25 calls); the rate is from the median call."""
        qq = q[:b]
        rates, slowest = {}, {}
        for path in paths:
            ivf.search(qq, k, nprobe, path)
            times = []
            while len(times) < (5 if b < 64 else 3) or (sum(times) < 0.02 and len(times) < 25):
                start = time.perf_counter()
                ivf.search(qq, k, nprobe, path)
                times.append(time.perf_counter() - start)
            rates[path], slowest[path] = b / sorted(times)[len(times) // 2], max(times)
        return rates, slowest

    batches = [b for b in ROUTE_BATCHES if b <= len(q)]
    if gpu:
        # a warm-up pass over every batch size: a fresh index's first GPU calls at a size run up to 30% slower
        # than later ones even after a call at that size
        for b in batches:
            for path in ("gpu-query", "gpu-list"):
                ivf.search(q[:b], k, nprobe, path)
    rates = {}
    cpu_on, gpu_wins = True, 0
    blas = ("cpu-query-blas",) if ivf.metric in ("l2", "ip") else ()
    for b in batches:
        paths = (("cpu-split",) if b == 1 else ("cpu-query",) + blas + ("cpu-list",)) if cpu_on else ()
        # the GPU's paths first: after the CPU's, a GPU left idle runs its next calls slower
        measured, slowest = time_paths(b, (("gpu-query", "gpu-list") if gpu else ()) + paths)
        rates.update({(path, b): r for path, r in measured.items()})
        if cpu_on:
            if gpu:
                cpu_best = max(measured[p] for p in paths)
                gpu_best = max(measured["gpu-query"], measured["gpu-list"])
                gpu_wins = gpu_wins + 1 if gpu_best > cpu_best else 0
            cpu_on = gpu_wins < 2 and max(slowest[p] for p in paths) < CPU_GIVE_UP
    cpu_batches = sorted({b for (p, b) in rates if p.startswith("cpu")})
    many = [b for b in cpu_batches if b > 1]
    cq = {b: rates[("cpu-query", b)] for b in many}
    cl = {b: rates[("cpu-list", b)] for b in many}
    routes = {"cpu_blas_from": NEVER}
    if blas and cq:
        cqb = {b: rates[("cpu-query-blas", b)] for b in many}
        routes["cpu_blas_from"] = crossover_with_margin(cq, cqb, DEFAULT_ROUTES["cpu_blas_from"])
        cq = {b: cqb[b] if b >= routes["cpu_blas_from"] else cq[b] for b in many}    # the query-major scan as routed
    routes["cpu_list_from"] = crossover_with_margin(cq, cl, DEFAULT_ROUTES["cpu_list_from"]) if cq else NEVER
    if gpu:
        gq = {b: rates[("gpu-query", b)] for b in batches}
        gl = {b: rates[("gpu-list", b)] for b in batches}
        cpu = {b: rates[("cpu-split", b)] if b == 1 else max(cq[b], cl[b]) for b in cpu_batches}
        best_gpu = {b: max(gq[b], gl[b]) for b in cpu_batches}
        routes["cpu_below"] = crossover_with_margin(cpu, best_gpu, DEFAULT_ROUTES["cpu_below"])
        routes["gpu_list_from"] = crossover_with_margin(gq, gl, DEFAULT_ROUTES["gpu_list_from"])
    else:
        routes["cpu_below"], routes["gpu_list_from"] = NEVER, NEVER
    return routes, rates


def describe_routes(routes: dict, gpu: bool) -> str:
    """The routes, in words."""
    c, gl, cl = routes["cpu_below"], routes["gpu_list_from"], routes["cpu_list_from"]
    cb = routes.get("cpu_blas_from", NEVER)
    notes = ([] if cb >= NEVER or cb >= min(cl, c) else [f"coarse step through BLAS from {cb:,}"]) + \
        ([] if cl >= NEVER else [f"list-major from {cl:,}"])
    cpu_list = f" ({'; '.join(notes)})" if notes else ""
    if not gpu or c >= NEVER:
        return f"single queries split over 4 threads, batches on the CPU{cpu_list}"
    parts = ["single queries on the CPU, split over 4 threads"] if c <= 2 else \
        [f"below {c:,} queries on the CPU (single queries split over 4 threads){cpu_list}"]
    if gl >= NEVER:
        parts.append("larger batches on the GPU, query-major")
    elif gl <= c:
        parts.append("larger batches on the GPU, list-major")
    else:
        parts.append(f"on the GPU query-major up to {gl - 1:,} queries, list-major from {gl:,}")
    return "; ".join(parts)
