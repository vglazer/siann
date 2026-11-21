"""Distance computations per query at matched recall: siann's IVF against FAISS's HNSW.

In batches of about 100, both are limited by memory traffic, so the work is the number of vectors a
query touches. For the IVF that is the centroids compared in the coarse step plus the vectors in the
probed lists, counted exactly from each query's probes; with --two-level, the coarse step compares
the groups' centres and the centroids of the probed groups instead (groups of about 64 centroids, a
quarter of them probed). For HNSW it is FAISS's own count (`faiss.cvar.hnsw_stats.ndis`). Bytes per
query are the memory a query reads that no other query shares: the vectors of its probed lists, or
HNSW's visited vectors, times their storage (uint8 where the data allows, as for SIFT, else float32;
HNSW's links not counted). The centroids are shared by every query and stay in cache, so they count
as distances but not as bytes. Recall at 10 against exact search, counting ties as correct.

    uv run --group bench python dist_counts.py --dataset sift
"""
import argparse

import faiss
import numpy as np
from backends import agreement
from datasets import load

from siann._cpu import FlatIndex, IvfIndex
from siann.ivf_gpu import GpuIVF

K = 10


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sift", choices=["sift", "glove"])
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--nlists", default="4000,8000,16000")
    p.add_argument("--nprobes", default="8,16,32,64,128,256")
    p.add_argument("--efs", default="16,32,64,128,256")
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args()
    faiss.omp_set_num_threads(a.threads)
    xb, xq = load(a.dataset, a.queries)
    n, d = xb.shape
    integer = np.array_equal(xb.astype(np.uint8).astype(np.float32), xb)
    width = 1 if integer else 4
    truth = np.array(FlatIndex(xb, d, "l2").search(xq, K), dtype=np.int64).reshape(-1, K)
    print(f"{a.dataset}, l2: {n:,} vectors, {d} dimensions ({'uint8' if integer else 'float32'}, "
          f"{d * width} bytes each), {len(xq):,} queries")
    print(f"\n  {'method':52s} {'recall@10':>9s} {'distances/query':>15s} {'list KB/query':>13s}")

    for nlist in map(int, a.nlists.split(",")):
        cents = np.ascontiguousarray(np.array(GpuIVF(xb, nlist, "l2").centroids), dtype=np.float32)
        sizes = np.bincount(np.array(FlatIndex(cents, d, "l2").search(xb, 1), dtype=np.int64), minlength=nlist)
        index = IvfIndex(xb, cents, d, "l2")
        # the two-level coarse step: about 64 centroids per group, a quarter of the groups probed
        ngroups = max(1, nlist // 64)
        km = faiss.Kmeans(d, ngroups, niter=20, seed=1)
        km.train(cents)
        gassign = km.index.search(cents, 1)[1][:, 0].astype(np.int64)
        index.set_coarse_groups(np.ascontiguousarray(km.centroids, dtype=np.float32), gassign)
        gprobe = max(1, ngroups // 4)
        gsizes = np.bincount(gassign, minlength=ngroups)
        gnear = np.array(FlatIndex(np.ascontiguousarray(km.centroids, dtype=np.float32), d, "l2")
                         .search(xq, gprobe), dtype=np.int64).reshape(-1, gprobe)
        coarse_two = ngroups + gsizes[gnear].sum(axis=1).mean()
        for nprobe in [p for p in map(int, a.nprobes.split(",")) if p <= nlist]:
            probes = np.array(FlatIndex(cents, d, "l2").search(xq, nprobe), dtype=np.int64).reshape(-1, nprobe)
            scanned = sizes[probes].sum(axis=1).mean()
            for label, coarse, g in (("exact coarse", nlist, 0), (f"two-level, {gprobe}/{ngroups} groups", coarse_two, gprobe)):
                if g and nlist < 8000:
                    continue
                found = np.array(index.search(xq, K, nprobe, gprobe=g), dtype=np.int64).reshape(-1, K)
                r = agreement(xb, xq, found, truth, "l2")
                dist = coarse + scanned
                kb = scanned * d * width / 1024   # the probed lists; the centroids stay in cache
                print(f"  {f'IVF {nlist} lists, nprobe {nprobe}, {label}':52s} {r:9.3f} {dist:15,.0f} {kb:13,.0f}",
                      flush=True)

    hnsw = faiss.IndexHNSWFlat(d, 32)
    hnsw.hnsw.efConstruction = 40
    hnsw.add(xb)
    for ef in map(int, a.efs.split(",")):
        hnsw.hnsw.efSearch = ef
        faiss.cvar.hnsw_stats.reset()
        found = hnsw.search(xq, K)[1]
        ndis = faiss.cvar.hnsw_stats.ndis / len(xq)
        r = agreement(xb, xq, found, truth, "l2")
        print(f"  {f'FAISS HNSW (M 32), efSearch {ef}':52s} {r:9.3f} {ndis:15,.0f} {ndis * d * width / 1024:13,.0f}",
              flush=True)


if __name__ == "__main__":
    main()
