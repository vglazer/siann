"""Single-query latency: siann's Rust IVF on one thread, and split (`split=n`: the coarse step and the
probed lists spread over a pool of n threads of its own), against FAISS's HNSW, which answers a single
query on one thread. With --faiss (L2 only), also FAISS's IVF over the same centroids in the same
storages (IndexIVFFlat, IndexIVFScalarQuantizer with QT_fp16 or QT_8bit_direct, not by residual), on
one thread and split its own way (parallel_mode 1: one query's lists over n OpenMP threads), and HNSW
in the compressed storages too (IndexHNSWSQ); FAISS's indexes get data padded to a multiple of 8
dimensions (pad8).

The lists are ivf_cpu.py's (GpuIVF's k-means centroids, 4,000 lists), stored as float32, float16 and,
on SIFT, uint8. Each query is one call; latency is the median and the 99th percentile over --queries
queries after a warm-up pass. Recall at 10 against exact search, counting ties as correct
(backends.agreement). A split query occupies several cores for its duration, so it buys latency, not
throughput: queries per second on a loaded machine are better served one per thread (ivf_cpu.py).

    uv run --group bench python single_query.py --dataset sift --metric l2
    uv run --group bench python single_query.py --dataset sift --splits 1,4 --faiss
    uv run --group bench python single_query.py --efs "" --splits 1,2,4,8 --storages float32,uint8
                                                     # the split sweep, without HNSW
"""
import argparse
import time

import faiss
import numpy as np
from siann._cpu import FlatIndex, IvfIndex

from backends import agreement
from datasets import load
from exact import FAISS_METRIC
from siann.ivf_gpu import GpuIVF

K = 10


def pad8(x: np.ndarray) -> np.ndarray:
    """x with zero columns appended up to a multiple of 8 dimensions, which leaves L2 distances unchanged.
    FAISS's scalar quantizer has a fast (SIMD) path only for such dimensions: at 100, its float16 search
    is 6.5 times slower than at 104 (0.118 against 0.016 s for 100 queries on 200,000 vectors), so
    FAISS's indexes are given padded data, as a careful user would."""
    extra = -x.shape[1] % 8
    return np.ascontiguousarray(np.pad(x, ((0, 0), (0, extra)))) if extra else x


def latency(search, xq: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Median and 99th-percentile latency in microseconds, one query per call, after a warm-up pass."""
    for q in xq[:100]:
        search(q[None])
    times, found = [], []
    for q in xq:
        start = time.perf_counter()
        found.append(search(q[None]))
        times.append(time.perf_counter() - start)
    t = np.array(times) * 1e6
    return float(np.median(t)), float(np.percentile(t, 99)), np.concatenate(found)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sift", choices=["sift", "glove"])
    p.add_argument("--metric", default="l2", choices=["l1", "l2", "linf"])
    p.add_argument("--nlist", type=int, default=4000)
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--nprobes", default="4,16,64,128")
    p.add_argument("--efs", default="16,32,64,128,256")
    p.add_argument("--splits", default="1,4", help="tasks per query; 1 is one thread")
    p.add_argument("--storages", default="float32,float16,uint8", help="uint8 only where lossless (SIFT)")
    p.add_argument("--faiss", action="store_true", help="also FAISS's IVF and compressed HNSW (L2)")
    p.add_argument("--threads", type=int, default=16, help="FAISS's threads (HNSW answers one query on one)")
    a = p.parse_args()
    faiss.omp_set_num_threads(a.threads)
    xb, xq = load(a.dataset, a.queries)
    d = xb.shape[1]
    truth = np.array(FlatIndex(xb, d, a.metric).search(xq, K), dtype=np.int64).reshape(-1, K)
    centroids = np.ascontiguousarray(np.array(GpuIVF(xb, a.nlist, a.metric).centroids), dtype=np.float32)
    wanted = a.storages.split(",")
    storages = {"float32": (None, "float32"), "float16": (xb.astype(np.float16).view(np.uint16), "float16")}
    if np.array_equal(xb.astype(np.uint8).astype(np.float32), xb):
        storages["uint8"] = (xb.astype(np.uint8), "uint8")
    storages = {name: v for name, v in storages.items() if name in wanted}
    rust = {name: IvfIndex(xb, centroids, d, a.metric, stored, kind) for name, (stored, kind) in storages.items()}
    efs = [int(e) for e in a.efs.split(",") if e]
    qtype = {"float16": faiss.ScalarQuantizer.QT_fp16, "uint8": faiss.ScalarQuantizer.QT_8bit_direct}
    if a.faiss and a.metric != "l2":
        raise SystemExit("--faiss: FAISS's scalar quantizer supports L2 (and inner product) only")
    xbp, xqp = pad8(xb), pad8(xq)   # FAISS's data, padded; L2 distances unchanged
    dp = xbp.shape[1]
    hnsws = {}
    for name in (storages if a.faiss else ["float32"]) if efs else []:
        if name == "float32":
            h = faiss.IndexHNSWFlat(dp, 32, FAISS_METRIC[a.metric])
        else:
            h = faiss.IndexHNSWSQ(dp, qtype[name], 32)
            h.train(xbp[:10_000])
        h.hnsw.efConstruction = 40
        h.add(xbp)
        hnsws[name] = h
    fivf = {}
    if a.faiss:
        quantizer = faiss.IndexFlatL2(dp)
        quantizer.add(pad8(centroids))
        for name in storages:
            if name == "float32":
                ix = faiss.IndexIVFFlat(quantizer, dp, a.nlist)
            else:
                ix = faiss.IndexIVFScalarQuantizer(quantizer, dp, a.nlist, qtype[name], faiss.METRIC_L2)
                ix.by_residual = False
                ix.train(xbp[:10_000])
            ix.add(xbp)
            fivf[name] = ix
    print(f"{a.dataset}, {a.metric}: {len(xb):,} vectors, {d} dimensions, {a.nlist} lists; {a.queries:,} single "
          f"queries; 'n threads' is split=n, a pool of n threads per query; latency in microseconds")

    def recall(found):
        return agreement(xb, xq, found, truth, a.metric)

    print(f"\n  {'method':34s} {'recall@10':>9s} {'median':>8s} {'p99':>8s}")
    for nprobe in map(int, a.nprobes.split(",")):
        for name, r in rust.items():
            for split in map(int, a.splits.split(",")):
                med, p99, found = latency(
                    lambda q, r=r, s=split, n=nprobe: np.array(r.search(q, K, n, split=s), dtype=np.int64).reshape(-1, K), xq)
                how = "1 thread" if split == 1 else f"{split} threads"
                print(f"  {f'Rust IVF {name}, {how}, nprobe {nprobe}':34s} {recall(found):9.3f} {med:8.0f} {p99:8.0f}",
                      flush=True)
        for name, ix in fivf.items():
            ix.nprobe = nprobe
            for split in map(int, a.splits.split(",")):
                # FAISS's own split: parallel_mode 1 spreads one query's lists over the OpenMP threads
                ix.parallel_mode = 0 if split == 1 else 1
                faiss.omp_set_num_threads(split)
                med, p99, found = latency(lambda q, ix=ix: ix.search(q, K)[1], xqp)
                how = "1 thread" if split == 1 else f"{split} threads"
                print(f"  {f'FAISS IVF {name}, {how}, nprobe {nprobe}':34s} {recall(found):9.3f} {med:8.0f} "
                      f"{p99:8.0f}", flush=True)
        faiss.omp_set_num_threads(a.threads)
    for ef in efs:
        for name, h in hnsws.items():
            h.hnsw.efSearch = ef
            med, p99, found = latency(lambda q, h=h: h.search(q, K)[1], xqp)
            print(f"  {f'FAISS HNSW {name}, efSearch {ef}':34s} {recall(found):9.3f} {med:8.0f} {p99:8.0f}", flush=True)


if __name__ == "__main__":
    main()
