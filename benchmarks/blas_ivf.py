"""IVF on the CPU with BLAS, for L2 (and cosine): the lists stored sorted by list in NumPy, as the GPU
IVF stores them; the coarse step as exact L2 against the centroids (blas_l2.py's path); queries
grouped by the lists they probe; for each list, one matrix multiply of its queries with its vectors
(NumPy's `@`, Accelerate), and the products ranked in Rust (`L2TopK.push_l2_lists`, a group of lists at
a time, in parallel, merged into each query's best k). Against siann's Rust IVF on the same lists
(list-major for the batch of 10,000, query-major for batches of 100, as in ivf_cpu.py). Recall at 10
against exact search; queries per second, median of three after a warm-up.

    uv run --group bench python blas_ivf.py --dataset sift
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

from siann._cpu import FlatIndex, IvfIndex, L2TopK
from siann.blas_l2 import default_tiles, siann_l2
from siann.ivf_gpu import GpuIVF

import numpy as np
from backends import agreement
from datasets import load
from timing import throughput

K = 10


class BlasIVF:
    def __init__(self, xb: np.ndarray, centroids: np.ndarray):
        self.centroids = centroids
        self.cnorms = np.einsum("ij,ij->i", centroids, centroids).astype(np.float32)
        # each vector's nearest centroid, by the same exact search the Rust IVF uses, so that the two
        # IVFs hold identical lists (the BLAS form rounds differently and flips a few near-ties)
        assign = np.array(FlatIndex(centroids, centroids.shape[1], "l2").search(xb, 1), dtype=np.int64)
        order = np.argsort(assign, kind="stable")
        self.ids = order.astype(np.int64)
        self.x = np.ascontiguousarray(xb[order])
        self.norms = np.einsum("ij,ij->i", self.x, self.x).astype(np.float32)
        counts = np.bincount(assign, minlength=len(centroids))
        self.offsets = np.concatenate([[0], np.cumsum(counts)])

    def search(self, xq: np.ndarray, k: int, nprobe: int, group: int = 256, workers: int = 1,
               phases: list | None = None) -> np.ndarray:
        """``workers`` threads compute a group's products at once (NumPy releases the GIL in `@`);
        ``phases``, if given, accumulates seconds spent in [coarse and grouping, products, ranking]."""
        t0 = time.perf_counter()
        nq = len(xq)
        qt, blk = default_tiles(nq)
        probes = siann_l2(self.centroids, self.cnorms, xq, nprobe, qt, min(blk, len(self.centroids)))
        flat = probes.ravel()
        qids = np.repeat(np.arange(nq, dtype=np.int64), nprobe)
        order = np.argsort(flat, kind="stable")
        lists, qids = flat[order], qids[order]
        bounds = np.searchsorted(lists, np.arange(len(self.centroids) + 1))
        top = L2TopK(nq, k)
        work = [c for c in range(len(self.centroids))
                if bounds[c] < bounds[c + 1] and self.offsets[c] < self.offsets[c + 1]]
        t1 = time.perf_counter()
        t_prod = t_rank = 0.0

        def product(c):
            return xq[qids[bounds[c]:bounds[c + 1]]] @ self.x[self.offsets[c]:self.offsets[c + 1]].T

        pool = ThreadPoolExecutor(workers) if workers > 1 else None
        for g in range(0, len(work), group):
            cs = work[g:g + group]
            t = time.perf_counter()
            prods = list(pool.map(product, cs)) if pool else [product(c) for c in cs]
            t_prod += time.perf_counter() - t
            t = time.perf_counter()
            top.push_l2_lists(prods, [self.norms[self.offsets[c]:self.offsets[c + 1]] for c in cs],
                              [self.ids[self.offsets[c]:self.offsets[c + 1]] for c in cs],
                              [qids[bounds[c]:bounds[c + 1]] for c in cs])
            t_rank += time.perf_counter() - t
        if pool:
            pool.shutdown()
        if phases is not None:
            phases[0] += t1 - t0
            phases[1] += t_prod
            phases[2] += t_rank
        return np.array(top.result(), dtype=np.int64).reshape(nq, k)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sift", choices=["sift", "glove"])
    p.add_argument("--nlist", type=int, default=4000)
    p.add_argument("--queries", type=int, default=10_000)
    p.add_argument("--small", type=int, default=1000, help="queries for batches of 100")
    p.add_argument("--nprobes", default="16,64")
    p.add_argument("--workers", default="1,8", help="threads computing the products")
    a = p.parse_args()
    xb, xq = load(a.dataset, a.queries)
    d = xb.shape[1]
    truth = np.array(FlatIndex(xb, d, "l2").search(xq, K), dtype=np.int64).reshape(-1, K)
    centroids = np.ascontiguousarray(np.array(GpuIVF(xb, a.nlist, "l2").centroids), dtype=np.float32)
    rust = IvfIndex(xb, centroids, d, "l2")
    blas = BlasIVF(xb, centroids)
    assert rust.list_sizes() == np.diff(blas.offsets).tolist(), "the two IVFs' lists differ"
    small, full = xq[:a.small], xq
    print(f"{a.dataset}, l2: {len(xb):,} vectors, {d} dimensions, {a.nlist} lists; batches of 100 over "
          f"{a.small:,} queries, and {a.queries:,} at once\n")
    print(f"  {'method':40s} {'recall@10':>9s} {'q/s, 100':>10s} {'q/s, full':>10s}")
    for nprobe in map(int, a.nprobes.split(",")):
        rows = (
            ("Rust IVF (query-major / list-major)",
             lambda q, n=nprobe: np.array(rust.search(q, K, n), dtype=np.int64).reshape(-1, K),
             lambda q, n=nprobe: np.array(rust.search_list_major(q, K, n), dtype=np.int64).reshape(-1, K)),
        ) + tuple(
            (f"BLAS IVF, {w} workers", lambda q, n=nprobe, w=w: blas.search(q, K, n, workers=w),
             lambda q, n=nprobe, w=w: blas.search(q, K, n, workers=w))
            for w in map(int, a.workers.split(",")))
        for name, f100, ffull in rows:
            qps100, _ = throughput(f100, small, 100)
            qps_full, found = throughput(ffull, full, len(full))
            r = agreement(xb, xq, found, truth, "l2")
            print(f"  {f'{name}, nprobe {nprobe}':40s} {r:9.3f} {qps100:10,.0f} {qps_full:10,.0f}", flush=True)
        for w in map(int, a.workers.split(",")):
            ph = [0.0, 0.0, 0.0]
            blas.search(full, K, nprobe, workers=w, phases=ph)
            print(f"    BLAS IVF, {w} workers, batch of {len(full):,}: coarse and grouping {ph[0]:.3f} s, "
                  f"products {ph[1]:.3f} s, ranking {ph[2]:.3f} s", flush=True)


if __name__ == "__main__":
    main()
