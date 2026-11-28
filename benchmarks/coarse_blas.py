"""The CPU IVF's coarse step for a batch through BLAS (NumPy's matrix multiply against the centroids, ranked
by siann's Rust top-k) against siann's per-query Rust coarse step, for L2, cosine and inner product.

--check compares the two coarse steps' neighbors directly. --save and --compare record and check every CPU output (the IVF's three CPU paths at batches of 1 to
1,000, and exact search), so that the change is checked to return the same neighbors. --ab times the CPU
IVF path by path with the coarse step done each way (alternating, median of 5 rounds), and siann.Index
(its routes measured at build, the BLAS coarse step among them) against FAISS's IVF-Flat on the same lists
at the same nprobe, calibrated (nprobe and routes) on 1,000 of the dataset's queries other than those timed. Run with --threads 4 to see a 4-core
machine (the Rust thread pool, BLAS and FAISS all held to that many).

    uv run --group bench python coarse_blas.py --save before.npz
    uv run --group bench python coarse_blas.py --compare before.npz
    uv run --group bench python coarse_blas.py --ab --threads 4
    uv run --group bench python coarse_blas.py --check
"""
import argparse
import os
import sys

if "--threads" in sys.argv:                 # before NumPy or rayon start their thread pools
    t = sys.argv[sys.argv.index("--threads") + 1]
    os.environ.update(RAYON_NUM_THREADS=t, VECLIB_MAXIMUM_THREADS=t, OPENBLAS_NUM_THREADS=t, OMP_NUM_THREADS=t)

import time

import siann

import numpy as np
from datasets import load

CASES = [("sift", "l2"), ("glove", "cosine"), ("music100", "ip"), ("sift", "l1")]
BATCHES = (1, 2, 4, 10, 32, 100, 1000)
PATHS = ("cpu-query", "cpu-list", "cpu-split")
K = 10


def build(name: str, metric: str, base: int, queries: int, calibrate_on_queries: bool = False):
    """The base, the queries and the CPU IVF; with ``calibrate_on_queries``, nprobe and the routes come
    from the next ``queries`` of the dataset's own queries rather than from the base's vectors."""
    xb, xq = load(name, 2 * queries if calibrate_on_queries else queries)
    xb = np.ascontiguousarray(xb[:base])
    cal = np.ascontiguousarray(xq[queries:]) if calibrate_on_queries else None
    index = siann.Index(xb, metric, recall=0.95, kind="ivf", device="cpu", calibration_queries=cal)
    return xb, np.ascontiguousarray(xq[:queries]), index


def outputs(cases, base: int, queries: int) -> dict[str, np.ndarray]:
    out = {}
    for name, metric in cases:
        xb, xq, index = build(name, metric, base, queries)
        q = xq if metric != "cosine" else xq / np.linalg.norm(xq, axis=1, keepdims=True)
        q = np.ascontiguousarray(q, dtype=np.float32)
        for p in PATHS:
            for b in BATCHES:
                out[f"{name}_{metric}_{p}_{b}"] = np.concatenate(
                    [index._ivf.search(q[s:s + b], K, index.nprobe, p) for s in range(0, len(q), b)])
        out[f"{name}_{metric}_exact"] = siann.Index(xb, metric, device="cpu").search(xq, K).indices
        print(f"  {name} {metric}: nprobe {index.nprobe}, {index._ivf.nlist} lists", flush=True)
    return out


def check(cases, base: int, queries: int) -> None:
    """The BLAS coarse step against the Rust one: the same neighbors, on both batch scans, at every batch size."""
    for name, metric in cases:
        if metric not in ("l2", "cosine", "ip"):
            continue
        _, xq, index = build(name, metric, base, queries, calibrate_on_queries=True)
        q = xq if metric != "cosine" else xq / np.linalg.norm(xq, axis=1, keepdims=True)
        q = np.ascontiguousarray(q, dtype=np.float32)
        ivf, nprobe = index._ivf, index.nprobe
        same = []
        for p in ("cpu-query", "cpu-list"):
            for b in BATCHES[1:]:
                run = [np.concatenate([ivf.search(q[s:s + b], K, nprobe, p, coarse=c) for s in range(0, len(q), b)])
                       for c in ("rust", "blas")]
                same.append(np.array_equal(*run))
        print(f"  {name} {metric}, nprobe {nprobe}: identical at {sum(same)} of {len(same)} (scan, batch) pairs",
              flush=True)


def ab(cases, base: int, queries: int, threads: str) -> None:
    import faiss
    if threads:
        faiss.omp_set_num_threads(int(threads))
    print(f"{threads or os.cpu_count()} threads; queries per second, Rust coarse step / BLAS (ratio), "
          "median of 5 alternating rounds")
    for name, metric in cases:
        if metric not in ("l2", "cosine", "ip"):
            continue
        xb, xq, index = build(name, metric, base, queries, calibrate_on_queries=True)
        q = xq if metric != "cosine" else xq / np.linalg.norm(xq, axis=1, keepdims=True)
        q = np.ascontiguousarray(q, dtype=np.float32)
        ivf, nprobe = index._ivf, index.nprobe
        print(f"  {name} {metric}, {len(xb):,} vectors, {ivf.nlist} lists, nprobe {nprobe}; routes {index._routes}")
        for p in ("cpu-query", "cpu-list"):
            cells = []
            for b in BATCHES[1:]:
                runs = {c: (lambda c=c: [ivf.search(q[s:s + b], K, nprobe, p, coarse=c)
                                         for s in range(0, len(q), b)]) for c in ("rust", "blas")}
                for f in runs.values():
                    f()
                times = {c: [] for c in runs}
                for _ in range(5):
                    for c, f in runs.items():
                        t = time.perf_counter()
                        f()
                        times[c].append(time.perf_counter() - t)
                rust, blas = (len(q) / sorted(times[c])[2] for c in ("rust", "blas"))
                cells.append(f"{b:>5,}: {rust:8,.0f} / {blas:8,.0f} ({blas / rust:.2f})")
            print(f"    {p}\n      " + "\n      ".join(cells), flush=True)
        fm = faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2
        flat = faiss.IndexFlat(xb.shape[1], fm)
        flat.add(np.ascontiguousarray(ivf.built.centroids, dtype=np.float32))
        fivf = faiss.IndexIVFFlat(flat, xb.shape[1], ivf.nlist, fm)
        fivf.is_trained = True
        xs = xb if metric != "cosine" else xb / np.linalg.norm(xb, axis=1, keepdims=True)
        xs = np.ascontiguousarray(xs, dtype=np.float32)
        assign = np.ascontiguousarray(ivf.built.assign, dtype=np.int64)
        fivf.add_core(len(xs), faiss.swig_ptr(xs), None, faiss.swig_ptr(assign))   # siann's lists
        fivf.nprobe = nprobe
        cells = []
        for b in BATCHES:
            n = min(len(q), max(100, 100 * b))
            runs = (lambda: [index.search(xq[s:s + b], K) for s in range(0, n, b)],
                    lambda: [fivf.search(q[s:s + b], K) for s in range(0, n, b)])
            times = ([], [])
            for f in runs:
                f()
            for _ in range(3):
                for f, ts in zip(runs, times):
                    t = time.perf_counter()
                    f()
                    ts.append(time.perf_counter() - t)
            ours, theirs = (n / sorted(ts)[1] for ts in times)
            cells.append(f"{b:>5,}: {ours:8,.0f} / {theirs:8,.0f} ({ours / theirs:.2f}, {index._ivf_path(b, None)})")
        print("    siann.Index / FAISS IVF-Flat, same lists and nprobe, median of 3 alternating passes "
              "(siann's path)\n      " + "\n      ".join(cells), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=int, default=200_000)
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--threads", default="", help="hold every thread pool to this many")
    p.add_argument("--save", help="write every output here (.npz)")
    p.add_argument("--compare", help="check every output against this file (.npz)")
    p.add_argument("--ab", action="store_true")
    p.add_argument("--check", action="store_true", help="the BLAS coarse step's neighbors against the Rust one's")
    a = p.parse_args()
    if a.ab:
        ab(CASES, a.base, a.queries, a.threads)
        return
    if a.check:
        check(CASES, a.base, a.queries)
        return
    out = outputs(CASES, a.base, a.queries)
    if a.save:
        np.savez(a.save, **out)
    if a.compare:
        ref = np.load(a.compare)
        print(f"\nagainst {os.path.basename(a.compare)}:")
        for key in out:
            same = (out[key] == ref[key]).all(axis=1)
            if same.all():
                continue
            print(f"  {key}: {int((~same).sum())} of {len(same)} queries differ", flush=True)
        n_bad = sum(not np.array_equal(out[key], ref[key]) for key in out)
        print(f"  {len(out) - n_bad} of {len(out)} outputs identical")


if __name__ == "__main__":
    main()
