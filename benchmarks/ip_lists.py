"""IVF for inner product with the lists assigned by L2 instead of by inner product. Assigned by inner
product (as FAISS's METRIC_INNER_PRODUCT IVF assigns), centroids pointing towards long vectors attract
much of the data and the lists are unbalanced; assigned by L2 they balance. Three ways to build the
same IVF (the same k-means centroids, 4,000 lists), each scanned by inner product on the GPU:

- assign ip, probe ip: siann's IVF for inner product until this measurement;
- assign l2, probe ip: lists by L2, a query's lists chosen by inner product with the centroids;
- assign l2, probe l2: lists by L2, a query's lists chosen by L2 distance to the centroids.

Recall@10 against exact inner product search and queries per second (batches of 100 over --small
queries, query-major scan; one batch of --queries, list-major scan), median of three after a warm-up.

    uv run --group bench python ip_lists.py --dataset music100
"""
import argparse

import numpy as np
from backends import agreement
from datasets import load
from timing import throughput

from siann._cpu import FlatIndex
from siann.ivf_gpu import GpuIVF

K = 10


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="music100", choices=["music100", "glove-raw"])
    p.add_argument("--nlist", type=int, default=4000)
    p.add_argument("--nprobes", default="4,16,32,64,128")
    p.add_argument("--queries", type=int, default=10_000)
    p.add_argument("--small", type=int, default=1000)
    p.add_argument("--lists", default="float16", choices=["float32", "float16"])
    p.add_argument("--variants", default="ip/ip,l2/ip,l2/l2", help="assign/probe pairs")
    a = p.parse_args()
    xb, xq = load(a.dataset, a.queries)
    truth = np.array(FlatIndex(xb, xb.shape[1], "ip").search(xq, K), dtype=np.int64).reshape(-1, K)
    small = xq[:a.small]
    print(f"{a.dataset}, ip: {len(xb):,} vectors, {a.nlist} lists ({a.lists}); queries per second")
    for assign, probe in (v.split("/") for v in a.variants.split(",")):
        ivf = GpuIVF(xb, a.nlist, "ip", a.lists, assign=assign, probe=probe)
        sizes = np.diff(np.array(ivf.offsets))
        top = np.sort(sizes)[::-1]
        print(f"\n  assign {assign}, probe {probe}: largest list {top[0]:,}, 10 largest hold "
              f"{top[:10].sum() / sizes.sum():.0%}, empty {np.mean(sizes == 0):.0%}")
        print(f"  {'nprobe':>8s} {'recall@10':>9s} {'100':>9s} {f'{len(xq):,}':>9s}")
        for nprobe in map(int, a.nprobes.split(",")):
            q100, _ = throughput(lambda q, n=nprobe: ivf.search(q, K, n), small, 100)
            qf, found = throughput(lambda q, n=nprobe: ivf.search_list_major(q, K, n), xq, len(xq))
            r = agreement(xb, xq, found, truth, "ip")
            print(f"  {nprobe:8d} {r:9.3f} {q100:9,.0f} {qf:9,.0f}", flush=True)


if __name__ == "__main__":
    main()
