"""Where a small batch's time goes in siann's Rust IVF: thread scaling, the coarse step's share, the
top-k's share, and the two-level coarse step, in batches of 100.

SIFT-like data (uint8 lists, integer queries on the integer path) or float32 lists otherwise; 4,000 and
16,000 lists (k-means centroids as in ivf_cpu.py). For each, at a working nprobe: queries per second in
batches of 100 (over --small queries, median of three after a warm-up) as the batch's thread pool
grows (`search(..., threads=n)`, one query per thread; 12 is the performance cores, 16 all cores),
with nprobe 1 (almost only the coarse step), with k = 1 and 100 against 10, and with the exact coarse
step against the two-level one. Single-query throughput on one thread is the reference for scaling.
With --gprobes, also the groups probed by the two-level step, which changes the centroid bytes a query
reads without changing its lists (recall at 10 shown, counting ties as correct).

    uv run --group bench python batch_profile.py --dataset sift
"""
import argparse
import time

import faiss
import numpy as np
from siann._cpu import IvfIndex

from backends import agreement
from datasets import load
from siann._cpu import FlatIndex
from siann.ivf_gpu import GpuIVF

K = 10
NPROBE = {4000: 64, 16000: 128}   # recall about 0.97 on SIFT


def qps(fn, xq, batch, repeats=3):
    """Queries per second searching `batch` queries per call: median of `repeats` after a warm-up."""
    def run():
        for s in range(0, len(xq), batch):
            fn(xq[s:s + batch])
    run()
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        run()
        times.append(time.perf_counter() - t)
    return len(xq) / sorted(times)[len(times) // 2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sift", choices=["sift", "glove"])
    p.add_argument("--nlists", default="4000,16000")
    p.add_argument("--small", type=int, default=1000)
    p.add_argument("--threads", default="1,2,4,8,12,16")
    p.add_argument("--gprobes", default="", help="groups probed by the two-level step, for a sweep")
    a = p.parse_args()
    xb, xq = load(a.dataset, a.small)
    n, d = xb.shape
    integer = np.array_equal(xb.astype(np.uint8).astype(np.float32), xb)
    storage = "uint8" if integer else "float32"
    stored = xb.astype(np.uint8) if integer else None
    print(f"{a.dataset}, l2: {n:,} vectors, {d} dimensions, {storage} lists; batches of 100 over {a.small:,} "
          f"queries; queries per second")
    for nlist in map(int, a.nlists.split(",")):
        cents = np.ascontiguousarray(np.array(GpuIVF(xb, nlist, "l2").centroids), dtype=np.float32)
        index = IvfIndex(xb, cents, d, "l2", stored, storage)
        ngroups = max(1, nlist // 64)
        km = faiss.Kmeans(d, ngroups, niter=20, seed=1)
        km.train(cents)
        index.set_coarse_groups(np.ascontiguousarray(km.centroids, dtype=np.float32),
                                km.index.search(cents, 1)[1][:, 0].astype(np.int64))
        g, nprobe = max(1, ngroups // 4), NPROBE.get(nlist, 64)

        def search(q, k=K, n=nprobe, gp=g, threads=0):
            return index.search(q, k, n, gprobe=gp, threads=threads)
        one = qps(lambda q: search(q), xq, 1)
        print(f"\n  {nlist} lists, nprobe {nprobe}, two-level coarse step ({g} of {ngroups} groups)")
        print(f"    single queries, one thread: {one:,.0f}")
        for t in map(int, a.threads.split(",")):
            r = qps(lambda q, t=t: search(q, threads=t), xq, 100)
            print(f"    batches of 100, {t:2d} threads: {r:9,.0f}  ({r / one:4.1f} times one thread's single queries)",
                  flush=True)
        base = qps(lambda q: search(q), xq, 100)
        print(f"    batches of 100, global pool:  {base:9,.0f}")
        print(f"    nprobe 1 (the coarse step and one list): {qps(lambda q: search(q, n=1), xq, 100):9,.0f}")
        for k in (1, 100):
            print(f"    k = {k:3d}: {qps(lambda q, k=k: search(q, k=k), xq, 100):9,.0f}")
        print(f"    exact coarse step: {qps(lambda q: search(q, gp=0), xq, 100):9,.0f}", flush=True)
        if a.gprobes:
            truth = np.array(FlatIndex(xb, d, "l2").search(xq, K), dtype=np.int64).reshape(-1, K)
            gsizes = np.bincount(km.index.search(cents, 1)[1][:, 0], minlength=ngroups)
            for gp in map(int, a.gprobes.split(",")):
                found = np.array(search(xq, gp=gp), dtype=np.int64).reshape(-1, K)
                cent_kb = (ngroups + gsizes.mean() * gp) * d * 4 / 1024
                print(f"    {gp:3d} of {ngroups} groups probed (about {cent_kb:5,.0f} KB of centroids per query): "
                      f"{qps(lambda q, gp=gp: search(q, gp=gp), xq, 100):9,.0f}, recall "
                      f"{agreement(xb, xq, found, truth, 'l2'):.3f}", flush=True)


if __name__ == "__main__":
    main()
