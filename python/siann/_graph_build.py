"""siann's own CAGRA graph build, on any machine: each vector's KNN_K nearest found by siann's IVF searching the
base vectors themselves (on the GPU where there is one, on the CPU otherwise), then CAGRA's optimization in safe
Rust (siann._cpu.optimize_graph: links reordered by their detours, the best kept, the second half of each list
given to reverse links). Ranks, not distances, decide the optimization, so any of siann's metrics can be built
this way, L1 and L-infinity included."""
from __future__ import annotations

import time

import numpy as np

KNN_K = 64                 # candidates per vector in the k-nearest-neighbor graph, as CAGRA's intermediate degree
# lists each vector's search probes: a rough graph holding 78% of the true 64 nearest prunes to one about as good
# as at 64 probes (SIFT1M, benchmarks/graph_build.py), in under half the time
KNN_NPROBE = 16


def knn_graph(x: np.ndarray, metric: str, k: int = KNN_K, nprobe: int = KNN_NPROBE, device: str = "auto",
              gpu: str | None = "mlx", ivf=None) -> tuple[np.ndarray, dict]:
    """Each row's ``k`` nearest other rows (n, k) int64, nearest first, -1 where fewer were found, by an IVF
    over ``x`` (``ivf``, or one built here) searching ``x`` itself; ``metric`` is the IVF's ("l1", "l2", "linf",
    "ip"; cosine as "l2" on unit vectors). Returns the graph and the seconds each part took."""
    from ._ivf import IVF, default_nlist, default_storage
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = len(x)
    on_gpu = device == "gpu" or (device == "auto" and gpu is not None and _gpu_available())
    times = {}
    t = time.perf_counter()
    if ivf is None:
        storage = "uint8" if default_storage(x) == "uint8" else "float"
        ivf = IVF(x, metric, default_nlist(n), storage, "gpu" if on_gpu and gpu == "mlx" else
                  ("torch" if on_gpu else "cpu"), gpu=gpu)
    times["IVF"] = time.perf_counter() - t
    kk, probes = min(k + 1, n), min(nprobe, ivf.nlist)
    path = fastest_path(ivf, x, kk, probes, on_gpu)
    t = time.perf_counter()
    found = ivf.search(x, kk, probes, path)
    times["search"], times["path"] = time.perf_counter() - t, path
    from ._cpu import without_own
    out = np.asarray(without_own(np.ascontiguousarray(found, dtype=np.int64),
                                 np.arange(n, dtype=np.int64), min(k, n - 1)), dtype=np.int64)
    return out, times


SAMPLE = 4096             # rows each candidate scan is timed on before the whole search


def fastest_path(ivf, x: np.ndarray, k: int, nprobe: int, on_gpu: bool) -> str:
    """The scan that finds each row's k nearest fastest here, timed on a sample: at k = 64 the query-major scans
    beat the list-major ones on every machine measured, and which device and coarse step wins differs (SIFT1M:
    the CPU's query-major scan through BLAS on an M3 Max, 6.2 s against 9.0 s on its GPU; without BLAS on a
    4-CPU Linux machine, 22 s against 39 s)."""
    paths = ["cpu-query"] + (["cpu-query-blas"] if ivf.metric in ("l2", "ip") else []) + (["gpu-query"] if on_gpu else [])
    if len(x) <= 4 * SAMPLE or len(paths) == 1:
        return paths[-1] if on_gpu else paths[0]
    q = x[np.random.default_rng(0).choice(len(x), SAMPLE, replace=False)]
    rates = {}
    for p in paths:
        ivf.search(q[:256], k, nprobe, p)                               # warm-up: each device's copy built
        t = time.perf_counter()
        ivf.search(q, k, nprobe, p)
        rates[p] = SAMPLE / (time.perf_counter() - t)
    return max(rates, key=rates.get)


def build_links(x: np.ndarray, metric: str, degree: int = 32, k: int = KNN_K, nprobe: int = KNN_NPROBE,
                device: str = "auto", gpu: str | None = "mlx", ivf=None) -> tuple[np.ndarray, dict]:
    """A CAGRA graph's links over ``x`` (n, ``degree``) int32: the k-nearest-neighbor graph, then CAGRA's
    optimization. Returns the links and the seconds each part took ("IVF", "search", "optimize")."""
    from ._cpu import optimize_graph
    knn, times = knn_graph(x, metric, k, nprobe, device, gpu, ivf)
    t = time.perf_counter()
    links = optimize_graph(np.ascontiguousarray(knn, dtype=np.int64), degree)
    times["optimize"] = time.perf_counter() - t
    return np.ascontiguousarray(links, dtype=np.int32), times


def _gpu_available() -> bool:
    from ._index import _gpu_available as available
    return available()
