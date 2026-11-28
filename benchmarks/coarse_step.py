"""How much of a single IVF query is the coarse step (comparing the query with every centroid), and does
a cheaper coarse step pay at matched recall?

1. The breakdown: single-query latency of siann's Rust IVF (float32 lists, one thread and split over 4)
   at nprobe 1 to 16, fitted by a line, latency = intercept + slope x nprobe; the intercept is the
   coarse step plus the cost of the call, the slope the cost of one list. The call alone is timed on a
   tiny index (8 vectors, one centroid).
2. Fewer lists: the same IVF with 1,000, 2,000 and 4,000 lists (fewer centroids make the coarse step
   cheaper, but each list longer), latency against recall at 10, to compare at matched recall with
   FAISS's HNSW (single_query.py: on SIFT, 29 microseconds at recall 0.84 with uint8 vectors, 53 with
   float32).
3. A two-level coarse step (`set_coarse_groups`, `search(..., gprobe=g)`): the centroids clustered by
   k-means into groups (--two-level lists:groups); a query is compared with the groups' centers, then
   only with the centroids of its g nearest groups. Latency against recall for 4,000 and 8,000 lists,
   and the median time of a call of --batch queries (one thread per query, the CPU's small-batch path).

Latency is the median over --queries single calls after a warm-up (single_query.latency); recall
counts ties as correct (backends.agreement). L2.

    uv run --group bench python coarse_step.py --dataset sift
    uv run --group bench python coarse_step.py --dataset sift --nlists ""   # the two-level step only
"""
import argparse

from siann._cpu import FlatIndex, IvfIndex
from siann.ivf_gpu import GpuIVF

import faiss
import numpy as np
from backends import agreement
from datasets import load
from single_query import latency

K = 10


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sift", choices=["sift", "glove"])
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--nlists", default="1000,2000,4000")
    p.add_argument("--nprobes", default="1,2,4,8,16,32,64")
    p.add_argument("--splits", default="1,4")
    p.add_argument("--two-level", default="4000:64,8000:128", help="lists:groups for the two-level step")
    p.add_argument("--gprobes", default="4,8,16")
    p.add_argument("--batch", type=int, default=8, help="queries per call for the batch column (0: none)")
    p.add_argument("--two-level-nprobes", default="8,16,32")
    p.add_argument("--storage", default="float32", choices=["float32", "float16", "uint8"],
                   help="list storage for the two-level runs (uint8 only for integer data, SIFT)")
    a = p.parse_args()
    xb, xq = load(a.dataset, a.queries)
    d = xb.shape[1]
    truth = np.array(FlatIndex(xb, d, "l2").search(xq, K), dtype=np.int64).reshape(-1, K)
    tiny = IvfIndex(np.ascontiguousarray(xb[:8]), np.ascontiguousarray(xb[:1]), d, "l2")
    print(f"{a.dataset}, l2: {len(xb):,} vectors, {d} dimensions, {a.queries:,} single queries; "
          f"latency in microseconds (median)")
    for split in map(int, a.splits.split(",")):
        med, _, _ = latency(lambda q, s=split: np.array(tiny.search(q, min(K, 8), 1, split=s)), xq)
        print(f"  call overhead (8 vectors, 1 centroid), split={split}: {med:.0f}")

    for nlist in [int(n) for n in a.nlists.split(",") if n]:
        centroids = np.ascontiguousarray(np.array(GpuIVF(xb, nlist, "l2").centroids), dtype=np.float32)
        index = IvfIndex(xb, centroids, d, "l2")
        sizes = np.array(index.list_sizes())
        print(f"\n  {nlist} lists (mean length {sizes.mean():.0f}, largest {sizes.max()}):")
        print(f"    {'nprobe':>6s} {'recall@10':>9s} " + " ".join(f"{f'split={s}':>8s}" for s in a.splits.split(",")))
        fits = {}
        nprobes = [n for n in map(int, a.nprobes.split(",")) if n <= nlist]
        rows = {}
        for nprobe in nprobes:
            meds = []
            for split in map(int, a.splits.split(",")):
                med, _, found = latency(
                    lambda q, n=nprobe, s=split: np.array(index.search(q, K, n, split=s), dtype=np.int64).reshape(-1, K), xq)
                meds.append(med)
                fits.setdefault(split, []).append((nprobe, med))
            rows[nprobe] = (agreement(xb, xq, found, truth, "l2"), meds)
            print(f"    {nprobe:6d} {rows[nprobe][0]:9.3f} " + " ".join(f"{m:8.0f}" for m in meds), flush=True)
        for split, pts in fits.items():
            small = [(n, m) for n, m in pts if n <= 16]
            slope, intercept = np.polyfit([n for n, _ in small], [m for _, m in small], 1)
            print(f"    split={split}: fit over nprobe 1-16: {intercept:.0f} + {slope:.1f} x nprobe "
                  f"(coarse step and call; per list)")

    splits = list(map(int, a.splits.split(",")))
    for spec in [s for s in a.two_level.split(",") if s]:
        nlist, ngroups = map(int, spec.split(":"))
        centroids = np.ascontiguousarray(np.array(GpuIVF(xb, nlist, "l2").centroids), dtype=np.float32)
        stored = {"float32": None, "float16": xb.astype(np.float16).view(np.uint16),
                  "uint8": xb.astype(np.uint8)}[a.storage]
        index = IvfIndex(xb, centroids, d, "l2", stored, a.storage)
        km = faiss.Kmeans(d, ngroups, niter=20, seed=1)
        km.train(centroids)
        assign = km.index.search(centroids, 1)[1][:, 0].astype(np.int64)
        index.set_coarse_groups(np.ascontiguousarray(km.centroids, dtype=np.float32), assign)
        print(f"\n  two-level: {nlist} lists in {ngroups} groups (largest {np.bincount(assign).max()} lists), "
              f"{a.storage} lists:")
        bcol = f" {f'batch {a.batch}':>9s}" if a.batch else ""
        print(f"    {'nprobe':>6s} {'gprobe':>6s} {'recall@10':>9s} " + " ".join(f"{f'split={s}':>8s}" for s in splits)
              + bcol)
        for nprobe in map(int, a.two_level_nprobes.split(",")):
            for g in [0] + list(map(int, a.gprobes.split(","))):
                meds = []
                for split in splits:
                    med, _, found = latency(
                        lambda q, n=nprobe, s=split, g=g: np.array(index.search(q, K, n, split=s, gprobe=g),
                                                                   dtype=np.int64).reshape(-1, K), xq)
                    meds.append(med)
                cell = ""
                if a.batch:
                    med, _, _ = latency(
                        lambda q, n=nprobe, g=g: index.search(q[0], K, n, gprobe=g),
                        np.stack([xq[s:s + a.batch] for s in range(0, len(xq) - a.batch + 1, a.batch)]))
                    cell = f" {med:9.0f}"
                gl = "exact" if g == 0 else str(g)
                print(f"    {nprobe:6d} {gl:>6s} {agreement(xb, xq, found, truth, 'l2'):9.3f} "
                      + " ".join(f"{m:8.0f}" for m in meds) + cell, flush=True)


if __name__ == "__main__":
    main()
