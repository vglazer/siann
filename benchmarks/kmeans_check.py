"""siann's own k-means (siann.kmeans) against FAISS's, as the IVF's centroids: the same sample (100
vectors per list, seed 0), 10 iterations, 4,000 lists, the same GPU IVF built on each set of centroids,
then recall@10 at each nprobe (list-major scan, every query) and the lists' balance, with training times.
siann's k-means runs with two seeds to show how much recall moves with the starting centroids alone.

    uv run --group bench python kmeans_check.py --dataset sift
"""
import argparse
import time

from siann._cpu import FlatIndex
from siann.ivf_gpu import GpuIVF
from siann.kmeans import kmeans

import faiss
import numpy as np
from backends import agreement
from datasets import load

K = 10
METRIC = {"sift": "l2", "glove": "l2", "music100": "ip", "dbpedia": "l2"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sift", choices=sorted(METRIC))
    p.add_argument("--base", type=int, default=0)
    p.add_argument("--nlist", type=int, default=4000)
    p.add_argument("--queries", type=int, default=10_000)
    p.add_argument("--nprobes", default="8,16,32,64,128")
    a = p.parse_args()
    metric = METRIC[a.dataset]
    xb, xq = load(a.dataset, a.queries)
    if a.base:
        xb = np.ascontiguousarray(xb[:a.base])
    n, d = xb.shape
    truth = np.array(FlatIndex(xb, d, metric).search(xq, K), dtype=np.int64).reshape(-1, K)
    sample = xb[np.random.default_rng(0).choice(n, min(n, 100 * a.nlist), replace=False)]
    print(f"{a.dataset}, {metric}: {n:,} vectors, {d} dimensions, {a.nlist} lists, k-means on {len(sample):,} "
          f"sample vectors, 10 iterations", flush=True)
    runs = {}
    t = time.perf_counter()
    km = faiss.Kmeans(d, a.nlist, niter=10, seed=1)
    km.train(sample)
    runs["FAISS k-means"] = (km.centroids, time.perf_counter() - t)
    for seed in (1, 2):
        t = time.perf_counter()
        c = kmeans(sample, a.nlist, niter=10, seed=seed)
        runs[f"siann k-means, seed {seed}"] = (c, time.perf_counter() - t)
    nprobes = list(map(int, a.nprobes.split(",")))
    print(f"\n  {'centroids':26s} {'train s':>8s} {'largest':>8s} {'empty':>6s} " + " ".join(f"{f'np {p}':>7s}" for p in nprobes))
    for name, (c, secs) in runs.items():
        ivf = GpuIVF(xb, a.nlist, metric, "float32", centroids=c)
        sizes = np.diff(np.array(ivf.offsets))
        recalls = [agreement(xb, xq, ivf.search_list_major(xq, K, p), truth, metric) for p in nprobes]
        print(f"  {name:26s} {secs:8.1f} {sizes.max():8,d} {np.mean(sizes == 0):6.1%} "
              + " ".join(f"{r:7.3f}" for r in recalls), flush=True)
        del ivf


if __name__ == "__main__":
    main()
