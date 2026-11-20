"""Can the GPU IVF reach FAISS HNSW's speed in batches of about 100? A feasibility check on the current
kernels: the fixed cost of a call, the query-major scan at different threads per query, the list-major
scan on small batches, and the time per stage, each with its effective bandwidth (bytes of the probed
lists read per second).

SIFT1M (or GloVe-100), L2, 4,000 lists (k-means as in ivf_gpu.py), uint8 lists where the data allows,
nprobe 64 (recall about 0.97 on SIFT). Batches of --batch queries over --small queries, median of three
after a warm-up. The stage times force each stage to finish (mx.eval) before the next starts, so they
include one synchronization each and add up to more than a whole call.

    uv run --group bench python gpu_small_batch.py --dataset sift
"""
import argparse
import time

import mlx.core as mx
import numpy as np

from datasets import load
from siann.ivf_gpu import SCANS, GpuIVF, coarse_dist
from siann.knn_kernels import _finish, select_k

K = 10


def timed(fn, repeats=3):
    fn()
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return sorted(times)[len(times) // 2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sift", choices=["sift", "glove"])
    p.add_argument("--nlist", type=int, default=4000)
    p.add_argument("--nprobe", type=int, default=64)
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--small", type=int, default=1000)
    a = p.parse_args()
    xb, xq = load(a.dataset, a.small)
    n, d = xb.shape
    integer = np.array_equal(xb.astype(np.uint8).astype(np.float32), xb)
    lists = "uint8" if integer else "float32"
    width = 1 if integer else 4
    ivf = GpuIVF(xb, a.nlist, "l2", lists)
    batches = [xq[s:s + a.batch] for s in range(0, len(xq), a.batch)]
    sizes = np.diff(np.array(ivf.offsets))
    # bytes of lists a query reads at this nprobe, from its probes
    probes = np.concatenate([np.array(select_k(coarse_dist(mx.array(b), ivf.centroids, ivf.cnorms, "l2"),
                                               a.nprobe, threads=32)) for b in batches])
    mb = sizes[probes].sum(axis=1).mean() * d * width / 1e6
    print(f"{a.dataset}: {n:,} vectors, {d} dimensions, {a.nlist} lists ({lists}), nprobe {a.nprobe}; "
          f"batches of {a.batch}; a query reads {mb:.2f} MB of lists")

    def rate(fn):
        t = timed(lambda: [fn(b) for b in batches])
        q = len(xq) / t
        return q, q * mb / 1e3   # queries per second, GB/s

    empty = timed(lambda: [mx.eval(mx.array(np.zeros(4, np.float32)) + 1) for _ in range(100)]) / 100
    one = timed(lambda: [ivf.search(b[:1], K, 1) for b in batches]) / len(batches)
    print(f"\n  fixed cost: an empty GPU round trip {empty * 1e3:.3f} ms; a one-query, nprobe-1 search "
          f"{one * 1e3:.3f} ms")

    print(f"\n  {'scan':36s} {'q/s':>10s} {'GB/s':>7s}")
    for threads in (32, 64, 128, 256, 512):
        q, gb = rate(lambda b, t=threads: ivf.search(b, K, a.nprobe, threads=t))
        print(f"  {f'query-major, {threads} threads per query':36s} {q:10,.0f} {gb:7.0f}", flush=True)
    q, gb = rate(lambda b: ivf.search_list_major(b, K, a.nprobe))
    print(f"  {'list-major':36s} {q:10,.0f} {gb:7.0f}")

    # the stages of a query-major call, each finished before the next
    threads = 64
    qa = [mx.array(b) for b in batches]
    mx.eval(*qa)

    def coarse(q):
        pr = select_k(coarse_dist(q, ivf.centroids, ivf.cnorms, "l2"), a.nprobe, threads=32).astype(mx.int32)
        mx.eval(pr)
        return pr
    prs = [coarse(q) for q in qa]

    def scan(q, pr):
        nq = q.shape[0]
        vals, idx = SCANS["l2"](inputs=[q, ivf.x, ivf.offsets, pr],
                                template=[("Q", nq), ("D", d), ("P", a.nprobe), ("T", threads), ("K", K)],
                                grid=(threads, nq, 1), threadgroup=(min(threads, 256), 1, 1),
                                output_shapes=[(nq * threads * K,), (nq * threads * K,)],
                                output_dtypes=[mx.float32, mx.int32])
        mx.eval(vals, idx)
        return vals, idx
    outs = [scan(q, pr) for q, pr in zip(qa, prs)]

    def merge(q, vi):
        r = mx.take(ivf.ids, mx.maximum(_finish(vi[0], vi[1], q.shape[0], K), 0))
        mx.eval(r)
    t_coarse = timed(lambda: [coarse(q) for q in qa]) / len(qa)
    t_scan = timed(lambda: [scan(q, pr) for q, pr in zip(qa, prs)]) / len(qa)
    t_merge = timed(lambda: [merge(q, vi) for q, vi in zip(qa, outs)]) / len(qa)
    t_scan_gb = a.batch * mb / 1e3 / t_scan
    print(f"\n  per call of {a.batch} queries (query-major, {threads} threads): coarse step and selection "
          f"{t_coarse * 1e3:.3f} ms, scan {t_scan * 1e3:.3f} ms ({t_scan_gb:.0f} GB/s), merge {t_merge * 1e3:.3f} ms")

    # the coarse step's two parts, and other ways to select each query's nprobe nearest centroids
    dists = [coarse_dist(q, ivf.centroids, ivf.cnorms, "l2") for q in qa]
    mx.eval(*dists)
    t_dist = timed(lambda: [mx.eval(coarse_dist(q, ivf.centroids, ivf.cnorms, "l2")) for q in qa]) / len(qa)
    print(f"\n  coarse distances alone (one fused multiply): {t_dist * 1e3:.3f} ms per call")
    for st in (8, 16, 32, 64, 128):
        t = timed(lambda st=st: [mx.eval(select_k(dd, a.nprobe, threads=st)) for dd in dists]) / len(qa)
        print(f"  selection, siann's two-stage select_k, {st:3d} threads per query: {t * 1e3:.3f} ms per call")
    t = timed(lambda: [mx.eval(mx.argpartition(dd, kth=a.nprobe - 1, axis=1)[:, :a.nprobe]) for dd in dists]) / len(qa)
    print(f"  selection, MLX argpartition (unordered nprobe): {t * 1e3:.3f} ms per call")
    for st in (128, 256):
        def scan_t(q, pr, st=st):
            nq = q.shape[0]
            vals, idx = SCANS["l2"](inputs=[q, ivf.x, ivf.offsets, pr],
                                    template=[("Q", nq), ("D", d), ("P", a.nprobe), ("T", st), ("K", K)],
                                    grid=(st, nq, 1), threadgroup=(min(st, 256), 1, 1),
                                    output_shapes=[(nq * st * K,), (nq * st * K,)],
                                    output_dtypes=[mx.float32, mx.int32])
            mx.eval(vals, idx)
        t = timed(lambda st=st: [scan_t(q, pr) for q, pr in zip(qa, prs)]) / len(qa)
        print(f"  scan alone, {st} threads per query: {t * 1e3:.3f} ms per call ({a.batch * mb / 1e3 / t:.0f} GB/s)")

    # both changes in one call (one evaluation, as ivf.search does): the probes chosen by argpartition
    # (the scan needs them in no order) and more threads per query; recall against exact search
    from siann._cpu import FlatIndex
    from backends import agreement
    truth = np.array(FlatIndex(xb, d, "l2").search(xq, K), dtype=np.int64).reshape(-1, K)

    def fast(b, st):
        q = mx.array(b)
        nq = q.shape[0]
        dd = coarse_dist(q, ivf.centroids, ivf.cnorms, "l2")
        pr = mx.argpartition(dd, kth=a.nprobe - 1, axis=1)[:, :a.nprobe].astype(mx.int32)
        vals, idx = SCANS["l2"](inputs=[q, ivf.x, ivf.offsets, pr],
                                template=[("Q", nq), ("D", d), ("P", a.nprobe), ("T", st), ("K", K)],
                                grid=(st, nq, 1), threadgroup=(min(st, 256), 1, 1),
                                output_shapes=[(nq * st * K,), (nq * st * K,)],
                                output_dtypes=[mx.float32, mx.int32])
        r = mx.take(ivf.ids, mx.maximum(_finish(vals, idx, nq, K), 0))
        mx.eval(r)
        return np.array(r)
    print()
    for st in (64, 128, 256):
        q, gb = rate(lambda b, st=st: fast(b, st))
        found = np.concatenate([fast(b, st) for b in batches])
        print(f"  argpartition probes, {st} threads per query: {q:10,.0f} q/s ({gb:.0f} GB/s), recall "
              f"{agreement(xb, xq, found, truth, 'l2'):.3f}", flush=True)
    found = np.concatenate([ivf.search(b, K, a.nprobe) for b in batches])
    print(f"  (current search, recall {agreement(xb, xq, found, truth, 'l2'):.3f})")


if __name__ == "__main__":
    main()
