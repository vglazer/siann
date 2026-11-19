"""Inner product (maximum inner product search) with an IVF: assigning by inner product against the
reduction to L2.

1. List balance: an IVF assigning each vector to the centroid of largest inner product (as
   `ivf_cpu.py --metric ip` and FAISS's METRIC_INNER_PRODUCT IVF do) against one assigning by L2.
2. The reduction (Bachrach et al., 2014): each base vector x gets one more coordinate,
   sqrt(M^2 - |x|^2), M the largest |x|, so that every base vector has length M; each query gets 0.
   Then |q - x|^2 = |q|^2 + M^2 - 2 q.x, so L2 on the extended vectors ranks exactly as inner product
   on the originals, and an ordinary L2 IVF answers inner-product queries. siann's Rust and GPU IVFs
   on the extended vectors, against FAISS's HNSW with inner product on the originals and with L2 on
   the extended vectors (the same trick, available to FAISS users).

Recall at 10 is against exact inner-product search on the original vectors (backends.agreement,
"ip"); queries per second as in ivf_cpu.py (timing.throughput): single queries, batches of 100, and
one batch of all the queries.

    uv run --no-sync --group proto python ip_reduction.py --dataset glove-raw
"""
import argparse

import faiss
import numpy as np
from siann_cpu import FlatIndex, IvfIndex

from backends import agreement
from datasets import load
from ivf_gpu import GpuIVF
from timing import throughput

K = 10


def extend(xb: np.ndarray, xq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The reduction to L2: base vectors extended to length M, queries with a zero coordinate."""
    n2 = np.einsum("ij,ij->i", xb.astype(np.float64), xb.astype(np.float64))
    extra = np.sqrt(np.maximum(n2.max() - n2, 0.0)).astype(np.float32)
    return (np.ascontiguousarray(np.hstack([xb, extra[:, None]])),
            np.ascontiguousarray(np.hstack([xq, np.zeros((len(xq), 1), np.float32)])))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="glove-raw", choices=["sift", "glove", "glove-raw"])
    p.add_argument("--nlist", type=int, default=4000)
    p.add_argument("--queries", type=int, default=10_000)
    p.add_argument("--small", type=int, default=1000)
    p.add_argument("--nprobes", default="4,16,32,64,128")
    p.add_argument("--efs", default="16,32,64,128,256")
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args()
    faiss.omp_set_num_threads(a.threads)
    xb, xq = load(a.dataset, a.queries)
    d = xb.shape[1]
    truth = np.array(FlatIndex(xb, d, "ip").search(xq, K), dtype=np.int64).reshape(-1, K)
    xe, qe = extend(xb, xq)
    norms = np.linalg.norm(xb, axis=1)
    print(f"{a.dataset}, inner product: {len(xb):,} vectors, {d} dimensions (lengths {norms.min():.2f} to "
          f"{norms.max():.2f}), {a.nlist} lists")

    # 1. List balance, the same centroids assigned by inner product and by L2
    centroids = np.ascontiguousarray(np.array(GpuIVF(xb, a.nlist, "l2").centroids), dtype=np.float32)
    for m in ("ip", "l2"):
        sizes = np.sort(np.array(IvfIndex(xb, centroids, d, m).list_sizes()))[::-1]
        print(f"  lists assigned by {m}: largest {sizes[0]:,}, 10 largest hold {sizes[:10].sum() / len(xb):.0%} of "
              f"the vectors, {np.mean(sizes == 0):.0%} empty")

    # 2. The reduction: L2 IVFs on the extended vectors
    gpu = GpuIVF(xe, a.nlist, "l2")
    ce = np.ascontiguousarray(np.array(gpu.centroids), dtype=np.float32)
    rust = IvfIndex(xe, ce, d + 1, "l2")
    sizes = np.sort(np.array(rust.list_sizes()))[::-1]
    print(f"  extended vectors, lists by L2: largest {sizes[0]:,}, 10 largest hold {sizes[:10].sum() / len(xb):.1%}")
    hnsw_ip = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw_ip.hnsw.efConstruction = 40
    hnsw_ip.add(xb)
    hnsw_l2 = faiss.IndexHNSWFlat(d + 1, 32)
    hnsw_l2.hnsw.efConstruction = 40
    hnsw_l2.add(xe)

    small, full = a.small, a.queries

    def row(name, search, queries):
        qps1, _ = throughput(search, queries[:small], 1, repeats=1)
        qps100, _ = throughput(search, queries[:small], 100)
        qps_full, found = throughput(search, queries[:full], full)
        r = agreement(xb, xq, found, truth, "ip")
        print(f"  {name:34s} {r:9.3f} {qps1:12,.0f} {qps100:10,.0f} {qps_full:10,.0f}", flush=True)

    print(f"\n  {'method':34s} {'recall@10':>9s} {'q/s, single':>12s} {'q/s, 100':>10s} {'q/s, full':>10s}")
    for nprobe in map(int, a.nprobes.split(",")):
        row(f"Rust IVF, extended, nprobe {nprobe}",
            lambda q, n=nprobe: np.array(rust.search(q, K, n) if len(q) < 1000 else rust.search_list_major(q, K, n),
                                         dtype=np.int64).reshape(-1, K), qe)
        row(f"GPU IVF, extended, nprobe {nprobe}",
            lambda q, n=nprobe: gpu.search(q, K, n) if len(q) < 1000 else gpu.search_list_major(q, K, n), qe)
    for ef in map(int, a.efs.split(",")):
        hnsw_ip.hnsw.efSearch = ef
        row(f"FAISS HNSW, inner product, ef {ef}", lambda q: hnsw_ip.search(q, K)[1], xq)
        hnsw_l2.hnsw.efSearch = ef
        row(f"FAISS HNSW, extended L2, ef {ef}", lambda q: hnsw_l2.search(q, K)[1], qe)


if __name__ == "__main__":
    main()
