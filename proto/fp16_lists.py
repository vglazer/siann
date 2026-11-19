"""float16 (and uint8) lists against float32 lists in siann's IVFs: does cutting the memory traffic pay?

The same lists (GpuIVF's k-means centroids, each vector in its nearest centroid's list) stored as float32,
as float16 and, with --storages, as uint8 (SIFT only: its coordinates are integers from 0 to 255), searched by the Rust CPU IVF (query-major for single queries and batches of 100,
list-major for one batch of all the queries) and MLX's GPU IVF (query-major for batches of 100,
list-major for all), as in ivf_cpu.py. With --faiss, also FAISS's own compressed indexes in the same
storages, the strongest alternatives, on data padded to a multiple of 8 dimensions (pad8): IVF over the same centroids (IndexIVFFlat, and
IndexIVFScalarQuantizer with QT_fp16 or QT_8bit_direct, not by residual, so that it stores exactly the
vectors siann's lists do; L2 only), and HNSW (IndexHNSWFlat, IndexHNSWSQ; M = 32). Recall at 10 against exact search on the original float32
vectors, counting ties as correct (backends.agreement). Queries per second, median of three after a
warm-up (single queries: one pass).

    uv run --no-sync --group proto python fp16_lists.py --dataset sift --metric l2
    uv run --no-sync --group proto python fp16_lists.py --dataset sift --metric l2 --storages float32,float16,uint8
    uv run --no-sync --group proto python fp16_lists.py --dataset sift --storages float32,float16,uint8 --faiss
"""
import argparse
import copy

import faiss
import mlx.core as mx
import numpy as np
from siann_cpu import FlatIndex, IvfIndex

from backends import agreement
from datasets import load
from ivf_gpu import GpuIVF
from timing import throughput

K = 10


def pad8(x: np.ndarray) -> np.ndarray:
    """x with zero columns appended up to a multiple of 8 dimensions, which leaves L2 distances unchanged.
    FAISS's scalar quantizer has a fast (SIMD) path only for such dimensions: at 100, its float16 search
    is 6.5 times slower than at 104 (0.118 against 0.016 s for 100 queries on 200,000 vectors), so
    FAISS's indexes are given padded data, as a careful user would."""
    extra = -x.shape[1] % 8
    return np.ascontiguousarray(np.pad(x, ((0, 0), (0, extra)))) if extra else x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sift", choices=["sift", "glove"])
    p.add_argument("--metric", default="l2", choices=["l1", "l2", "linf"])
    p.add_argument("--nlist", type=int, default=4000)
    p.add_argument("--queries", type=int, default=10_000, help="for the full batch")
    p.add_argument("--small", type=int, default=1000, help="queries for single queries and batches of 100")
    p.add_argument("--nprobes", default="16,64")
    p.add_argument("--storages", default="float32,float16", help="any of float32, float16, uint8")
    p.add_argument("--faiss", action="store_true", help="also FAISS's IVF and HNSW in the same storages (L2)")
    p.add_argument("--efs", default="16,32,64,128,256", help="HNSW efSearch values, with --faiss")
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args()
    faiss.omp_set_num_threads(a.threads)
    xb, xq = load(a.dataset, a.queries)
    d = xb.shape[1]
    truth = np.array(FlatIndex(xb, d, a.metric).search(xq, K), dtype=np.int64).reshape(-1, K)
    x16 = xb.astype(np.float16)
    print(f"{a.dataset}, {a.metric}: {len(xb):,} vectors, {d} dimensions, {a.nlist} lists; float16 changes "
          f"{np.mean(x16.astype(np.float32) != xb):.1%} of coordinates (largest error {np.abs(x16.astype(np.float32) - xb).max():.2e}); "
          f"batches: 1 and 100 over {a.small:,} queries, {a.queries:,} at once")

    gpu32 = GpuIVF(xb, a.nlist, a.metric)
    centroids = np.ascontiguousarray(np.array(gpu32.centroids), dtype=np.float32)
    storages = a.storages.split(",")
    if "uint8" in storages and not np.array_equal(xb.astype(np.uint8).astype(np.float32), xb):
        raise SystemExit("uint8 lists need integer coordinates from 0 to 255 (SIFT)")
    gpu, rust = {}, {}
    for lists in storages:
        gpu[lists] = copy.copy(gpu32)   # the same lists, in this storage
        gpu[lists].x = gpu32.x.astype(getattr(mx, lists))
        mx.eval(gpu[lists].x)
        stored = {"float32": None, "float16": x16.view(np.uint16), "uint8": xb.astype(np.uint8)}[lists]
        rust[lists] = IvfIndex(xb, centroids, d, a.metric, stored, lists)
    fivf, fhnsw = {}, {}
    if a.faiss:
        if a.metric != "l2":
            raise SystemExit("--faiss: FAISS's scalar quantizer supports L2 (and inner product) only")
        qtype = {"float16": faiss.ScalarQuantizer.QT_fp16, "uint8": faiss.ScalarQuantizer.QT_8bit_direct}
        xbp = pad8(xb)
        dp = xbp.shape[1]
        quantizer = faiss.IndexFlatL2(dp)
        quantizer.add(pad8(centroids))
        for lists in storages:
            if lists == "float32":
                ix = faiss.IndexIVFFlat(quantizer, dp, a.nlist)
                h = faiss.IndexHNSWFlat(dp, 32)
            else:
                ix = faiss.IndexIVFScalarQuantizer(quantizer, dp, a.nlist, qtype[lists], faiss.METRIC_L2)
                ix.by_residual = False   # store the vectors themselves, as siann's lists do
                ix.train(xbp[:10_000])   # trains nothing for these types; the quantizer has the centroids
                h = faiss.IndexHNSWSQ(dp, qtype[lists], 32)
                h.train(xbp[:10_000])
            ix.add(xbp)
            h.hnsw.efConstruction = 40
            h.add(xbp)
            fivf[lists], fhnsw[lists] = ix, h
    small, full = xq[:a.small], xq

    def recall(found):
        return agreement(xb, xq[:len(found)], found, truth[:len(found)], a.metric)

    print(f"\n  {'method':30s} {'recall@10':>9s} {'q/s, single':>12s} {'q/s, 100':>10s} {'q/s, full':>10s}")

    smallp, fullp = pad8(small), pad8(full)   # FAISS's queries, padded beforehand so that padding is not timed

    def row(name, search):
        qps1, _ = throughput(search, smallp, 1, repeats=1)
        qps100, _ = throughput(search, smallp, 100)
        qps_full, found = throughput(search, fullp, len(fullp))
        print(f"  {name:30s} {recall(found):9.3f} {qps1:12,.0f} {qps100:10,.0f} {qps_full:10,.0f}", flush=True)
    for nprobe in map(int, a.nprobes.split(",")):
        for lists in storages:
            r = rust[lists]
            qps1, found1 = throughput(lambda q: np.array(r.search(q, K, nprobe), dtype=np.int64).reshape(-1, K),
                                      small, 1, repeats=1)
            qps100, _ = throughput(lambda q: np.array(r.search(q, K, nprobe), dtype=np.int64).reshape(-1, K),
                                   small, 100)
            qps_full, found = throughput(
                lambda q: np.array(r.search_list_major(q, K, nprobe), dtype=np.int64).reshape(-1, K), full, len(full))
            print(f"  {f'Rust IVF {lists}, nprobe {nprobe}':30s} {recall(found):9.3f} {qps1:12,.0f} {qps100:10,.0f} "
                  f"{qps_full:10,.0f}", flush=True)
        for lists in storages:
            g = gpu[lists]
            qps100, _ = throughput(lambda q: g.search(q, K, nprobe), small, 100)
            qps_full, found = throughput(lambda q: g.search_list_major(q, K, nprobe), full, len(full))
            print(f"  {f'GPU IVF {lists}, nprobe {nprobe}':30s} {recall(found):9.3f} {'':>12s} {qps100:10,.0f} "
                  f"{qps_full:10,.0f}", flush=True)
        for lists, ix in fivf.items():
            ix.nprobe = nprobe
            row(f"FAISS IVF {lists}, nprobe {nprobe}", lambda q, ix=ix: ix.search(q, K)[1])
    for ef in ([int(e) for e in a.efs.split(",") if e] if a.faiss else []):
        for lists, h in fhnsw.items():
            h.hnsw.efSearch = ef
            row(f"FAISS HNSW {lists}, ef {ef}", lambda q, h=h: h.search(q, K)[1])


if __name__ == "__main__":
    main()
