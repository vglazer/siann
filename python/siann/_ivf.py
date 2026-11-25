"""The IVF behind siann.Index when a recall target is given: lists built once (siann.ivf_build), each
device's copy made on first use, nprobe calibrated to the target on sample queries (the given calibration
queries, or the index's own vectors), and each search routed by its batch size along routes timed on them."""
from __future__ import annotations

import threading
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
FEW_ON_CPU = 32          # adding fewer vectors finds their lists on the CPU: faster than a GPU call for a few
BLAS_CHUNK = 2048        # queries per matrix multiply: products of at most 2,048 x nlist floats
# A GPU search's working memory grows with its batch: the coarse step's distances to every centroid, and the
# list-major scan's candidates, 16 lanes of k per probed list. So a batch is searched in equal chunks sized to
# keep that within siann._memory's budget per call; each query's result depends on it alone.


def default_nlist(n: int) -> int:
    """About 4 sqrt(n) lists (4,000 for a million vectors, as every measurement used), with at least 39
    vectors per list (k-means needs enough to place each centroid) and at least one list."""
    return int(max(1, min(round(4 * np.sqrt(n)), n // 39)))


def default_storage(x: np.ndarray) -> str:
    """uint8 where every coordinate is an integer from 0 to 255 (lossless), else float."""
    from ._cpu import is_bytes
    return "uint8" if is_bytes(np.ascontiguousarray(x, dtype=np.float32)) else "float"


class IVF:
    """The lists and each device's copy of them. ``metric`` is the IVF's: "l1", "l2", "linf" or "ip"
    (cosine arrives as "l2" on unit vectors)."""

    def __init__(self, x: np.ndarray, metric: str, nlist: int, storage: str, device_for_build: str,
                 built=None, gpu: str | None = "mlx"):
        """Builds the lists, or takes them from ``built`` (a ``siann.ivf_build.Lists``, as when loading).
        ``device_for_build`` is "gpu" (MLX), "torch" or "cpu"; ``gpu`` the GPU backend that searches ("mlx",
        "torch")."""
        self.gpu_backend = gpu or "mlx"
        from .ivf_build import build_lists
        self.x, self.metric, self.storage = x, metric, storage
        t = time.perf_counter()
        self.built = built if built is not None else build_lists(x, nlist, metric, device=device_for_build)
        self.build_seconds = time.perf_counter() - t
        self._gpu = self._cpu = self._coarse = None
        self.graph = None                              # a CAGRA graph for GPU batches (siann._graph), if built
        self._lock = threading.Lock()                  # each device's copy is made once, by the first search to need it
        self._centroid_flat = None                     # the centroids in Rust's exact kernel, for nearest_lists

    @property
    def nlist(self) -> int:
        return self.built.nlist

    def gpu(self):
        if self._gpu is None:
            with self._lock:
                if self._gpu is None:
                    self._gpu = self._make_gpu()
        return self._gpu

    def _make_gpu(self):
        if self.gpu_backend == "torch":
            from .ivf_torch import TorchIVF
            lists = "uint8" if self.storage == "uint8" else "float16"      # as on MLX
            return TorchIVF(self.x, self.built.centroids, self.metric, built=self.built, lists=lists)
        from ._memory import cap_mlx_cache
        from .ivf_gpu import GpuIVF
        cap_mlx_cache()
        lists = "uint8" if self.storage == "uint8" else "float16"      # float16 for GPU batches
        return GpuIVF(self.x, self.nlist, self.metric, lists, built=self.built)

    def cpu(self):
        if self._cpu is None:
            with self._lock:
                if self._cpu is None:
                    from ._cpu import IvfIndex
                    d = self.x.shape[1]
                    if self.storage == "uint8":
                        self._cpu = IvfIndex(self.x, self.built.centroids, d, self.metric,
                                             np.ascontiguousarray(self.x.astype(np.uint8)), "uint8",
                                             lists=self.built.assign)
                    else:
                        self._cpu = IvfIndex(self.x, self.built.centroids, d, self.metric, lists=self.built.assign)
        return self._cpu

    def nearest_lists(self, x: np.ndarray, device: str) -> np.ndarray:
        """Each vector's list: its nearest centroid by the metric the lists were assigned by. Below
        FEW_ON_CPU vectors, or without a GPU, siann's Rust exact kernel on the centroids (kept), faster for a
        few than a GPU call; otherwise the GPU's exact kernels."""
        from .ivf_build import assign_metric, nearest
        metric = assign_metric(self.metric)
        if device != "cpu" and len(x) >= FEW_ON_CPU:
            return nearest(x, self.built.centroids, metric, device)
        if self._centroid_flat is None:
            from ._cpu import FlatIndex
            c = self.centroids()
            self._centroid_flat = FlatIndex(c, c.shape[1], metric)
        return np.asarray(self._centroid_flat.search(np.ascontiguousarray(x, dtype=np.float32), 1),
                          dtype=np.int64)

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
        if path == "gpu-graph":
            return self.graph.search(q, k)
        if path == "cpu-graph":
            return self.graph.search_cpu(q, k)
        if path in ("gpu-query", "gpu-list"):
            scan = self.gpu().search if path == "gpu-query" else self.gpu().search_list_major
            from ._memory import budget
            chunks = -(-len(q) // gpu_chunk(k, nprobe, self.nlist, budget(self.gpu_backend)))
            if chunks == 1:
                return np.asarray(scan(q, k, nprobe), dtype=np.int64)
            step = -(-len(q) // chunks)                                 # equal chunks: none left small
            return np.concatenate([np.asarray(scan(q[s:s + step], k, nprobe), dtype=np.int64)
                                   for s in range(0, len(q), step)])
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


def gpu_chunk(k: int, nprobe: int, nlist: int, budget: int) -> int:
    """Queries per GPU call: ``budget`` bytes over a query's working memory, its distances to the centroids (with the
    probe selection's copies) and its candidates in the list-major scan (a value and an index, 16 lanes of k
    per probed list, held twice as they are selected), in multiples of 1,024: on SIFT1M (4,000 lists, nprobe
    46), within 10% of the memory MLX reports (`memory.py`)."""
    per_query = 12 * nlist + 16 * 16 * k * nprobe
    return max(1024, budget // per_query // 1024 * 1024)


def recall_of(found: np.ndarray, truth_scores_kth: np.ndarray, found_scores: np.ndarray, larger_better: bool) -> float:
    """The share of found neighbors at least as near as the true k-th nearest (ties count as found)."""
    from ._cpu import recall_of as rust_recall
    return rust_recall(np.ascontiguousarray(found, dtype=np.int64), np.ascontiguousarray(found_scores, dtype=np.float32),
                       np.ascontiguousarray(truth_scores_kth, dtype=np.float32), larger_better)


def calibrate(search, scores, truth_kth: np.ndarray, sample_ids: np.ndarray, target: float, nlist: int,
              larger_better: bool, k: int = CALIBRATION_K):
    """The smallest nprobe at which ``search(nprobe)`` reaches ``target`` recall@k on the sample: nprobe
    doubles until the target is reached, then a bisection narrows it. With ``sample_ids`` (the sample is
    the index's own vectors), the search returns k + 1 neighbors and each vector itself is left out.
    Returns (nprobe, recall at it)."""
    def recall_at(nprobe):
        found = search(nprobe)
        if sample_ids is not None:                                       # leave the vector itself out
            from ._cpu import without_own
            found = without_own(np.ascontiguousarray(found, dtype=np.int64), np.ascontiguousarray(sample_ids, dtype=np.int64), k)
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
DIP = 0.8                # a rate below this share of what its neighbors imply is timed again


DEFAULT_ROUTES = {"cpu_below": 2, "gpu_list_from": 1000, "cpu_list_from": NEVER, "cpu_blas_from": NEVER,
                  "graph_from": NEVER, "cpu_ivf_from": 1}
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


def measure_routes(ivf: IVF, q: np.ndarray, nprobe: int, gpu: bool, k: int = CALIBRATION_K,
                   rounds: int = 1, exact: dict | None = None) -> tuple[dict, dict]:
    """Where this IVF's searches should run, measured: each path timed by itself (the median call after a
    warm-up) at batch sizes doubling from 1 to 4,096 queries of ``q``. The CPU is no
    longer timed once the GPU has beaten it at two sizes in a row, or once one of its calls takes 0.3 s.
    Returns the thresholds (``cpu_below``: from which batch size the GPU's best scan beats the CPU's at every
    larger size measured; ``gpu_list_from`` and ``cpu_list_from``: from which the list-major scan beats the
    query-major one; ``cpu_blas_from``, for L2, cosine and inner product: from which the CPU's query-major
    scan does its coarse step through BLAS), each departing from ``DEFAULT_ROUTES`` only where the other path is at least 10%
    faster (about the timings' run-to-run spread), and the rates, as {(path, batch): rate}. Where the IVF has a
    graph, its GPU search is timed with the GPU's scans (``graph_from``: the batch size from which it beats both)
    and its CPU search with the CPU's (``cpu_ivf_from``: from which the CPU's IVF beats it). Single queries are
    decided on their own (``single``: the path measured fastest at one query, where it is at least 10% faster
    than the one the thresholds give it): the crossovers are one threshold each, and a path that wins at one
    query, as the IVF split over 4 threads often does, can lose at every batch from 2 up. With ``exact``
    ({"cpu-exact": search, "gpu-exact": search}, each taking a batch of queries), exact search is timed beside
    them, and routed to (``exact_from``) from the batch size where it beats the approximate path routed there
    at every larger size measured: it meets any recall target. The routes also hold ``table``: at each batch
    size measured, the paths measured there, fastest first (the default path first unless another is 10%
    faster); a search takes the entry of the largest measured batch size at or below its own, and in it the first
    path it may use. The table is what siann routes by; the thresholds remain for profiles and for files saved
    without a table."""

    def time_paths(b, paths):
        """Each path's rate at batch b and its slowest call. Each path is timed by itself, its calls back to
        back as in a stream of searches (between calls on the other device, a GPU call can run 40% slower):
        a warm-up call, then calls until there are at least 5 (3 from 64 queries) taking 20 ms in all (at
        most 25 calls); the rate is from the median call."""
        qq = q[:b]
        rates, slowest = {}, {}
        for path in paths:
            run = exact[path] if exact and path in exact else (lambda x, path=path: ivf.search(x, k, nprobe, path))
            if path.startswith("gpu"):
                clear_gpu_cache(ivf.gpu_backend)   # each GPU path from the same state: another's buffers slow it
            run(qq)
            times = []
            while len(times) < (5 if b < 64 else 3) or (sum(times) < 0.02 and len(times) < 25):
                start = time.perf_counter()
                run(qq)
                times.append(time.perf_counter() - start)
            rates[path], slowest[path] = b / sorted(times)[len(times) // 2], max(times)
        return rates, slowest

    batches = [b for b in ROUTE_BATCHES if b <= len(q)]
    gpu_paths = ("gpu-query", "gpu-list") + (("gpu-graph",) if gpu and ivf.graph is not None and ivf.graph.on_gpu
                                             else ())
    cpu_graph = ("cpu-graph",) if ivf.graph is not None else ()
    gpu_exact = ("gpu-exact",) if exact and gpu and "gpu-exact" in exact else ()
    if gpu:
        # a warm-up pass over every batch size: a fresh index's first GPU calls at a size run up to 30% slower
        # than later ones even after a call at that size
        for b in batches:
            for path in gpu_paths:
                ivf.search(q[:b], k, nprobe, path)
            if gpu_exact:
                exact["gpu-exact"](q[:b])
    rates = {}
    blas = ("cpu-query-blas",) if ivf.metric in ("l2", "ip") else ()
    for _ in range(rounds):
        cpu_on, gpu_wins, cpu_exact = True, 0, ("cpu-exact",) if exact and "cpu-exact" in exact else ()
        exact_gpu_wins = 0
        for b in batches:
            paths = ((("cpu-split",) if b == 1 else ("cpu-query",) + blas + ("cpu-list",)) + cpu_graph) if cpu_on else ()
            # the GPU's paths first: after the CPU's, a GPU left idle runs its next calls slower
            measured, slowest = time_paths(b, (gpu_paths if gpu else ()) + gpu_exact + paths + cpu_exact)
            if cpu_exact:                     # as the CPU's IVF: no longer timed once the GPU has clearly won
                exact_gpu_wins = exact_gpu_wins + 1 if gpu_exact and measured["gpu-exact"] > measured["cpu-exact"] else 0
                if slowest["cpu-exact"] > CPU_GIVE_UP or exact_gpu_wins >= 2:
                    cpu_exact = ()
            for path, r in measured.items():                      # over rounds, each path's best rate
                rates[(path, b)] = max(r, rates.get((path, b), 0.0))
            if cpu_on:
                if gpu:
                    cpu_best = max(measured[p] for p in paths)
                    gpu_best = max(measured[p] for p in gpu_paths)
                    gpu_wins = gpu_wins + 1 if gpu_best > cpu_best else 0
                cpu_on = gpu_wins < 2 and max(slowest[p] for p in paths) < CPU_GIVE_UP
    # a second look at dips: a rate well below the one its neighbors (half and double the batch) imply is
    # retimed, the better reading kept (a transient slowdown can only understate a path); GPU paths first
    dips = []
    for (path, b), r in rates.items():
        lo, hi = rates.get((path, b // 2)), rates.get((path, b * 2))
        if lo and hi and r < DIP * (lo * hi) ** 0.5:
            dips.append((not path.startswith("gpu"), b, path))
    for _, b, path in sorted(dips):
        again, _ = time_paths(b, (path,))
        rates[(path, b)] = max(rates[(path, b)], again[path])
    cpu_batches = sorted({b for (p, b) in rates if p.startswith("cpu") and p != "cpu-exact"})   # the CPU's IVF
    many = [b for b in cpu_batches if b > 1]
    cq = {b: rates[("cpu-query", b)] for b in many}
    cl = {b: rates[("cpu-list", b)] for b in many}
    routes = {"cpu_blas_from": NEVER}
    if blas and cq:
        cqb = {b: rates[("cpu-query-blas", b)] for b in many}
        routes["cpu_blas_from"] = crossover_with_margin(cq, cqb, DEFAULT_ROUTES["cpu_blas_from"])
        cq = {b: cqb[b] if b >= routes["cpu_blas_from"] else cq[b] for b in many}    # the query-major scan as routed
    routes["cpu_list_from"] = crossover_with_margin(cq, cl, DEFAULT_ROUTES["cpu_list_from"]) if cq else NEVER
    cpu = {b: rates[("cpu-split", b)] if b == 1 else max(cq[b], cl[b]) for b in cpu_batches}    # the IVF as routed
    routes["cpu_ivf_from"] = DEFAULT_ROUTES["cpu_ivf_from"]
    if cpu_graph:
        cg = {b: rates[("cpu-graph", b)] for b in cpu_batches}
        routes["cpu_ivf_from"] = crossover_with_margin(cg, cpu, DEFAULT_ROUTES["cpu_ivf_from"])
        cpu = {b: cpu[b] if b >= routes["cpu_ivf_from"] else cg[b] for b in cpu_batches}
    if gpu:
        gq = {b: rates[("gpu-query", b)] for b in batches}
        gl = {b: rates[("gpu-list", b)] for b in batches}
        routes["gpu_list_from"] = crossover_with_margin(gq, gl, DEFAULT_ROUTES["gpu_list_from"])
        ivf_gpu = {b: gl[b] if b >= routes["gpu_list_from"] else gq[b] for b in batches}   # the IVF as routed
        routes["graph_from"] = NEVER
        if "gpu-graph" in gpu_paths:
            gg = {b: rates[("gpu-graph", b)] for b in batches}
            routes["graph_from"] = crossover_with_margin(ivf_gpu, gg, DEFAULT_ROUTES["graph_from"])
            ivf_gpu = {b: gg[b] if b >= routes["graph_from"] else ivf_gpu[b] for b in batches}
        best_gpu = {b: max(rates[(p, b)] for p in gpu_paths) for b in cpu_batches}
        routes["cpu_below"] = crossover_with_margin(cpu, best_gpu, DEFAULT_ROUTES["cpu_below"])
    else:
        routes["cpu_below"], routes["gpu_list_from"], routes["graph_from"] = NEVER, NEVER, NEVER
    graph_gpu, graph_cpu = ivf.graph is not None and "gpu-graph" in gpu_paths, ivf.graph is not None
    routes["exact_from"] = NEVER
    if exact:
        r = {**DEFAULT_ROUTES, **routes}
        ex, ap = {}, {}
        for b in batches:
            e = [rates[(p, b)] for p in ("gpu-exact", "cpu-exact") if (p, b) in rates]
            a = rates.get((routed_path(r, b, graph_gpu, graph_cpu), b))
            if e and a:
                ex[b], ap[b] = max(e), a
        if ex:
            routes["exact_from"] = crossover_with_margin(ap, ex, NEVER)
    routes["table"] = route_table(rates, batches, graph_gpu, graph_cpu)
    one = {p: r for (p, b), r in rates.items() if b == 1}
    given = routed_path({**DEFAULT_ROUTES, **routes}, 1, graph_gpu, graph_cpu)
    best = max(one, key=one.get) if one else None
    if best and given in one and best != given and one[best] >= MARGIN * one[given]:
        routes["single"] = best
    return routes, rates


def routed_path(r: dict, b: int, graph_gpu: bool, graph_cpu: bool) -> str:
    """The path the thresholds give a batch of ``b`` (no device given; the single-query route aside), as
    Index._ivf_path takes it: exact search from ``exact_from``, else by device, graph and scan."""
    if b >= r.get("exact_from", NEVER):
        return "gpu-exact" if b >= r["cpu_below"] else "cpu-exact"
    if b >= r["cpu_below"]:
        if graph_gpu and b >= r["graph_from"]:
            return "gpu-graph"
        return "gpu-list" if b >= r["gpu_list_from"] else "gpu-query"
    if graph_cpu and b < r["cpu_ivf_from"]:
        return "cpu-graph"
    if b == 1:
        return "cpu-split"
    if b >= r["cpu_list_from"]:
        return "cpu-list"
    return "cpu-query-blas" if b >= r["cpu_blas_from"] else "cpu-query"


def route_table(rates: dict, batches, graph_gpu: bool, graph_cpu: bool) -> list:
    """[[batch, [paths fastest first]], ...] at each batch size measured: the default path (the one routing
    takes before any measurement) first unless another is ``MARGIN`` times faster, the others by rate."""
    table = []
    for b in batches:
        here = {p: r for (p, bb), r in rates.items() if bb == b}
        if not here:
            continue
        order = sorted(here, key=lambda p: -here[p])
        default = routed_path(DEFAULT_ROUTES, b, graph_gpu, graph_cpu)
        if default in here and here[order[0]] < MARGIN * here[default]:
            order.remove(default)
            order.insert(0, default)
        table.append([b, order])
    return table


def table_entry(table: list, batch: int) -> list:
    """The paths of the largest measured batch size at or below ``batch`` (the first entry below it)."""
    entry = table[0][1]
    for b, paths in table:
        if b > batch:
            break
        entry = paths
    return entry


PATH_WORDS = {"cpu-split": "the CPU, split over 4 threads", "cpu-query": "the CPU, query-major",
              "cpu-query-blas": "the CPU, query-major (coarse step through BLAS)", "cpu-list": "the CPU, list-major",
              "cpu-graph": "the CPU, through the graph", "cpu-exact": "the CPU, by exact search",
              "gpu-query": "the GPU, query-major", "gpu-list": "the GPU, list-major",
              "gpu-graph": "the GPU, through the graph", "gpu-exact": "the GPU, by exact search"}


def describe_table(table: list) -> str:
    """The table's first choices, as ranges of batch sizes."""
    parts, i = [], 0
    while i < len(table):
        j = i
        while j + 1 < len(table) and table[j + 1][1][0] == table[i][1][0]:
            j += 1
        lo = table[i][0]
        span = (f"{lo:,} and up" if j + 1 == len(table) else
                f"{lo:,}" if table[j + 1][0] - 1 == lo else f"{lo:,}-{table[j + 1][0] - 1:,}")
        parts.append(f"{span}: {PATH_WORDS.get(table[i][1][0], table[i][1][0])}")
        i = j + 1
    return "queries per search " + "; ".join(parts)


def single_path(r: dict, graph_gpu: bool, graph_cpu: bool) -> str:
    """The path the thresholds alone give a single query (no device given)."""
    return routed_path(r, 1, graph_gpu, graph_cpu)


SINGLE_WORDS = {"cpu-split": "on the CPU, split over 4 threads", "cpu-graph": "on the CPU, through the graph",
                "cpu-exact": "on the CPU, by exact search", "gpu-exact": "on the GPU, by exact search",
                "gpu-query": "on the GPU, query-major", "gpu-list": "on the GPU, list-major",
                "gpu-graph": "on the GPU, through the graph"}


def clear_gpu_cache(backend: str | None) -> None:
    """Empties the GPU backend's buffer cache. Timing a GPU path just after another left the cache full of
    other sizes ran up to 2x slower at some batch sizes (exact search after the IVF's scans, on GloVe-100)."""
    import contextlib
    with contextlib.suppress(Exception):                # no GPU backend loaded: nothing to clear
        if backend == "torch":
            import torch
            torch.mps.empty_cache()
        else:
            import mlx.core as mx
            mx.clear_cache()


def describe_routes(routes: dict, gpu: bool) -> str:
    """The routes, in words."""
    c, gl, cl = routes["cpu_below"], routes["gpu_list_from"], routes["cpu_list_from"]
    cb = routes.get("cpu_blas_from", NEVER)
    gg = routes.get("graph_from", NEVER)
    ci = routes.get("cpu_ivf_from", 1)
    notes = ([] if cb >= NEVER or cb >= min(cl, c) else [f"coarse step through BLAS from {cb:,}"]) + \
        ([] if cl >= NEVER else [f"list-major from {cl:,}"])
    if ci > 1:
        notes.insert(0, "through the graph" + ("" if ci >= min(c, NEVER) else f" below {ci:,} queries"))
    cpu_list = f" ({'; '.join(notes)})" if notes else ""
    single = "through the graph" if ci > 1 else "split over 4 threads"
    if routes.get("table"):
        return describe_table(routes["table"])
    measured = routes.get("single")
    ef = routes.get("exact_from", NEVER)
    if ef < NEVER:                                      # exact search from some batch size on: said last
        rest = describe_routes({k: v for k, v in routes.items() if k != "exact_from"}, gpu)
        return f"{rest}; exact search from {ef:,} queries (faster there, recall 1.0)"
    if measured:                                        # decided on its own: said last, as the exception
        rest = describe_routes({k: v for k, v in routes.items() if k != "single"}, gpu)
        for said in (f"single queries on the CPU, {single}; ", f"; single queries {single}"):
            rest = rest.replace(said, "")                   # the thresholds' own word on single queries
        return f"{rest}; single queries {SINGLE_WORDS.get(measured, measured)} (measured fastest)"
    if not gpu or c >= NEVER:
        return f"on the CPU{cpu_list}; single queries {single}"
    parts = [f"single queries on the CPU, {single}"] if c <= 2 else \
        [f"below {c:,} queries on the CPU{cpu_list}"]
    if gg <= c:
        parts.append("larger batches on the GPU through the graph")
        return "; ".join(parts)
    if gl >= NEVER:
        parts.append("larger batches on the GPU, query-major")
    elif gl <= c:
        parts.append("larger batches on the GPU, list-major")
    else:
        parts.append(f"on the GPU query-major up to {gl - 1:,} queries, list-major from {gl:,}")
    if gg < NEVER:
        parts.append(f"through the graph from {gg:,}")
    return "; ".join(parts)
