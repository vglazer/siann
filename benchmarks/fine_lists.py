"""Finer IVF partitions for small batches: do more lists, with the two-level coarse step, close the
gap to FAISS's HNSW on SIFT-like data?

The IVF with 4,000, 8,000 and 16,000 lists (k-means centroids as in ivf_cpu.py), uint8 lists where the
data allows (SIFT; integer queries take the exact integer path) and float32 otherwise, the Rust IVF's
coarse step two-level (groups of about 64 centroids, a quarter of the groups probed). Queries per
second: single queries (one call each, one thread), batches of 100 (Rust: one thread per query; GPU:
the query-major scan) and one batch of 10,000 (Rust's list-major scan, exact coarse step; the GPU's
list-major scan). Against FAISS's HNSW (M = 32) in the same storage (IndexHNSWSQ with QT_8bit_direct for
uint8, data padded to a multiple of 8 dimensions), at the nearest recall. Recall at 10 against exact
search, counting ties as correct.

    uv run --group bench python fine_lists.py --dataset sift
"""
import argparse

import faiss
import mlx.core as mx
import numpy as np
from backends import agreement
from datasets import load
from single_query import pad8
from timing import throughput

from siann._cpu import FlatIndex, IvfIndex
from siann.ivf_gpu import GpuIVF

K = 10
NPROBES = {4000: (16, 32, 64, 128), 8000: (32, 64, 128, 256), 16000: (32, 64, 128, 256, 512)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sift", choices=["sift", "glove"])
    p.add_argument("--nlists", default="4000,8000,16000")
    p.add_argument("--queries", type=int, default=10_000)
    p.add_argument("--small", type=int, default=1000)
    p.add_argument("--efs", default="16,32,64,128,256")
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args()
    faiss.omp_set_num_threads(a.threads)
    xb, xq = load(a.dataset, a.queries)
    n, d = xb.shape
    integer = np.array_equal(xb.astype(np.uint8).astype(np.float32), xb)
    storage = "uint8" if integer else "float32"
    stored = xb.astype(np.uint8) if integer else None
    truth = np.array(FlatIndex(xb, d, "l2").search(xq, K), dtype=np.int64).reshape(-1, K)
    small, full = xq[:a.small], xq
    print(f"{a.dataset}, l2: {n:,} vectors, {d} dimensions, {storage} lists; batches: 1 and 100 over "
          f"{a.small:,} queries, {a.queries:,} at once")
    print(f"\n  {'method':44s} {'recall@10':>9s} {'q/s, single':>12s} {'q/s, 100':>10s} {'q/s, full':>10s}")

    def row(name, f1, f100, ffull, queries=(small, full), recall_from="full"):
        qps1, _ = throughput(f1, queries[0], 1, repeats=1) if f1 else (float("nan"), None)
        qps100, found100 = throughput(f100, queries[0], 100)
        qps_full, found = throughput(ffull, queries[1], len(queries[1]))
        r = agreement(xb, xq, found, truth, "l2")
        r100 = agreement(xb, xq[:a.small], found100, truth[:a.small], "l2")
        s1 = f"{qps1:12,.0f}" if f1 else f"{'':>12s}"
        print(f"  {name:44s} {r:9.3f} {s1} {qps100:10,.0f} {qps_full:10,.0f}"
              + (f"   (recall in batches of 100: {r100:.3f})" if abs(r100 - r) > 0.01 else ""), flush=True)

    for nlist in map(int, a.nlists.split(",")):
        gpu = GpuIVF(xb, nlist, "l2", storage)
        cents = np.ascontiguousarray(np.array(gpu.centroids), dtype=np.float32)
        rust = IvfIndex(xb, cents, d, "l2", stored, storage)
        ngroups = max(1, nlist // 64)
        km = faiss.Kmeans(d, ngroups, niter=20, seed=1)
        km.train(cents)
        assign = km.index.search(cents, 1)[1][:, 0].astype(np.int64)
        rust.set_coarse_groups(np.ascontiguousarray(km.centroids, dtype=np.float32), assign)
        g = max(1, ngroups // 4)
        for nprobe in NPROBES.get(nlist, (16, 32, 64, 128)):
            def rq(q, n=nprobe):   # query-major, two-level coarse step
                return np.array(rust.search(q, K, n, gprobe=g), dtype=np.int64).reshape(-1, K)

            def rl(q, n=nprobe):   # list-major, exact coarse step
                return np.array(rust.search_list_major(q, K, n), dtype=np.int64).reshape(-1, K)
            row(f"Rust IVF, {nlist} lists, nprobe {nprobe}", rq, rq, rl)
            row(f"GPU IVF, {nlist} lists, nprobe {nprobe}", None,
                lambda q, n=nprobe, ivf=gpu: ivf.search(q, K, n), lambda q, n=nprobe, ivf=gpu: ivf.search_list_major(q, K, n))
        del gpu
        mx.clear_cache()

    xbp, xqp = pad8(xb), pad8(xq)
    if integer:
        hnsw = faiss.IndexHNSWSQ(xbp.shape[1], faiss.ScalarQuantizer.QT_8bit_direct, 32)
        hnsw.train(xbp[:10_000])
    else:
        hnsw = faiss.IndexHNSWFlat(xbp.shape[1], 32)
    hnsw.hnsw.efConstruction = 40
    hnsw.add(xbp)
    for ef in map(int, a.efs.split(",")):
        hnsw.hnsw.efSearch = ef

        def h(q):
            return hnsw.search(q, K)[1]
        row(f"FAISS HNSW ({storage}), efSearch {ef}", h, h, h, queries=(xqp[:a.small], xqp))


if __name__ == "__main__":
    main()
