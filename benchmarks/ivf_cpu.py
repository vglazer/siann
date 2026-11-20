"""IVF on the CPU in Rust (rust_cpu) against FAISS's IVF and HNSW, and siann's GPU IVF (ivf_gpu.py),
as recall at 10 against queries per second, at three batch sizes: single queries (one call per query),
batches of 100, and one batch of all the queries.

All three IVFs use the same k-means centroids (GpuIVF's: FAISS k-means, 10 iterations, seed 1), so
that they differ only in how they search. Recall counts a returned neighbor as correct when it is no
farther than the exact k-th nearest, so that ties (common in L1 on SIFT's integer data) are not
misses. Exact neighbors from the Rust flat index.

    uv run --group bench python ivf_cpu.py --dataset sift --metric l2
"""
import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import faiss
import numpy as np
from siann._cpu import FlatIndex, IvfIndex

from backends import agreement
from datasets import load
from exact import FAISS_METRIC
from siann.ivf_gpu import GpuIVF
from timing import throughput

K = 10


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sift", choices=["sift", "glove", "glove-raw", "music100", "dbpedia"])
    p.add_argument("--metric", default="l2", choices=["l1", "l2", "linf", "ip"])
    p.add_argument("--nlist", type=int, default=4000)
    p.add_argument("--queries", type=int, default=10_000, help="for the full batch")
    p.add_argument("--base", type=int, default=0, help="use only the first N base vectors (0: all)")
    p.add_argument("--small", type=int, default=1000, help="queries for single queries and batches of 100")
    p.add_argument("--nprobes", default="4,16,32,64,128")
    p.add_argument("--efs", default="16,32,64,128,256")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--no-torch", action="store_true", help="skip the PyTorch IVF (ivf_torch.py)")
    a = p.parse_args()
    faiss.omp_set_num_threads(a.threads)
    faiss.cvar.distance_compute_blas_threshold = 20   # see faiss_blas.py
    xb, xq = load(a.dataset, a.queries)
    if a.base:
        xb = np.ascontiguousarray(xb[:a.base])
    d = xb.shape[1]
    fm = FAISS_METRIC[a.metric]
    print(f"{a.dataset}, {a.metric}: {len(xb):,} vectors, {d} dimensions; {a.nlist} lists; "
          f"{a.threads} CPU threads; batches: 1 and 100 over {a.small:,} queries, {a.queries:,} at once")

    start = time.perf_counter()
    truth = np.array(FlatIndex(xb, d, a.metric).search(xq, K), dtype=np.int64).reshape(-1, K)
    print(f"exact neighbors (Rust flat): {time.perf_counter() - start:.1f} s")

    def recall(found: np.ndarray) -> float:
        return agreement(xb, xq[:len(found)], found, truth[:len(found)], a.metric)

    start = time.perf_counter()
    gpu = GpuIVF(xb, a.nlist, a.metric)
    print(f"GPU IVF built in {time.perf_counter() - start:.1f} s")
    centroids = np.ascontiguousarray(np.array(gpu.centroids), dtype=np.float32)

    start = time.perf_counter()
    rust = IvfIndex(xb, centroids, d, a.metric)
    print(f"Rust IVF built in {time.perf_counter() - start:.1f} s (from the same centroids)")

    start = time.perf_counter()
    quantizer = faiss.IndexFlat(d, fm)
    quantizer.add(centroids)
    fivf = faiss.IndexIVFFlat(quantizer, d, a.nlist, fm)
    fivf.train(centroids)   # a no-op: the quantizer already holds the nlist centroids
    fivf.add(xb)
    print(f"FAISS IVF built in {time.perf_counter() - start:.1f} s (from the same centroids)")

    start = time.perf_counter()
    hnsw = faiss.IndexHNSWFlat(d, 32, fm)
    hnsw.hnsw.efConstruction = 40
    hnsw.add(xb)
    print(f"FAISS HNSW (M=32) built in {time.perf_counter() - start:.1f} s", flush=True)

    small, full = xq[:a.small], xq
    print(f"\n  {'method':24s} {'recall@10':>9s} {'q/s, single':>12s} {'q/s, 100':>10s} {'q/s, full':>10s}")

    def row(name: str, search) -> None:
        qps1, found1 = throughput(search, small, 1, repeats=1)
        qps100, _ = throughput(search, small, 100)
        qps_full, found = throughput(search, full, len(full))
        r = recall(found)
        assert abs(recall(found1) - recall(found[:len(found1)])) < 0.01   # same results batched or not
        print(f"  {name:24s} {r:9.3f} {qps1:12,.0f} {qps100:10,.0f} {qps_full:10,.0f}", flush=True)

    for nprobe in map(int, a.nprobes.split(",")):
        row(f"Rust IVF, nprobe {nprobe}",
            lambda q, n=nprobe: np.array(rust.search(q, K, n), dtype=np.int64).reshape(-1, K))
        row(f"Rust IVF list-major, {nprobe}",
            lambda q, n=nprobe: np.array(rust.search_list_major(q, K, n), dtype=np.int64).reshape(-1, K))
        fivf.nprobe = nprobe
        row(f"FAISS IVF, nprobe {nprobe}", lambda q: fivf.search(q, K)[1])

        def gpu_search(q, n=nprobe):
            # the query-major scan for small batches, the list-major scan for large ones (ivf_gpu.py)
            return gpu.search(q, K, n) if len(q) < 1000 else gpu.search_list_major(q, K, n)
        row(f"GPU IVF, nprobe {nprobe}", gpu_search)
    for ef in map(int, a.efs.split(",")):
        hnsw.hnsw.efSearch = ef
        row(f"FAISS HNSW, efSearch {ef}", lambda q: hnsw.search(q, K)[1])

    if not a.no_torch:
        # PyTorch cannot share this process with FAISS, so its IVF runs in its own, from the same
        # centroids. Free this process's indexes first, so that both fit in memory.
        del gpu, rust, fivf, hnsw
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "centroids.npy"
            np.save(path, centroids)
            subprocess.run([sys.executable, str(Path(__file__).with_name("ivf_torch.py")),
                            "--centroids", str(path), "--dataset", a.dataset, "--metric", a.metric,
                            "--queries", str(a.queries), "--small", str(a.small), "--nprobes", a.nprobes],
                           check=True)


if __name__ == "__main__":
    main()
