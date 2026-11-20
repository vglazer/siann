"""The PyTorch GPU IVF (siann.ivf_torch) on a dataset, from centroids saved by ivf_cpu.py (which calls
it): recall and queries per second at each nprobe, printed as rows of ivf_cpu.py's table. FAISS and
PyTorch cannot share a process on macOS, so this runs on its own.

    uv run --group bench python ivf_torch.py --centroids C.npy --dataset sift --metric l2
"""
import argparse

import numpy as np
from siann._cpu import FlatIndex
from siann.ivf_torch import TorchIVF

from backends import agreement
from datasets import load
from timing import throughput


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--centroids", required=True)
    p.add_argument("--dataset", default="sift", choices=["sift", "glove", "glove-raw", "music100"])
    p.add_argument("--metric", default="l2", choices=["l1", "l2", "linf", "ip"])
    p.add_argument("--queries", type=int, default=10_000)
    p.add_argument("--small", type=int, default=1000)
    p.add_argument("--nprobes", default="4,16,32,64,128")
    a = p.parse_args()
    k = 10
    xb, xq = load(a.dataset, a.queries)
    truth = np.array(FlatIndex(xb, xb.shape[1], a.metric).search(xq, k), dtype=np.int64).reshape(-1, k)
    index = TorchIVF(xb, np.load(a.centroids), a.metric)
    small, full = xq[:a.small], xq
    for nprobe in map(int, a.nprobes.split(",")):
        def search(q, n=nprobe):
            return index.search(q, k, n) if len(q) < 1000 else index.search_list_major(q, k, n)
        qps1, found1 = throughput(search, small, 1, repeats=1)
        qps100, _ = throughput(search, small, 100)
        qps_full, found = throughput(search, full, len(full))
        r = agreement(xb, xq, found, truth, a.metric)
        r1 = agreement(xb, small, found1, truth[:len(small)], a.metric)
        assert abs(r1 - agreement(xb, small, found[:len(small)], truth[:len(small)], a.metric)) < 0.01
        print(f"  {f'PyTorch IVF, nprobe {nprobe}':24s} {r:9.3f} {qps1:12,.0f} {qps100:10,.0f} {qps_full:10,.0f}",
              flush=True)


if __name__ == "__main__":
    main()
