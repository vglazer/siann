"""FAISS's approximate options, HNSW and IVF, as recall at 10 against queries per second, with the
build time. Ground truth from the GPU's exact search.

    uv run --group bench python faiss_ann.py [--dataset sift] [--metric l1]
"""
import argparse
import time

import faiss
import mlx.core as mx
import numpy as np

from datasets import load
from siann.knn_kernels import knn_tiled, l2_knn

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="sift")
p.add_argument("--metric", default="l1")
p.add_argument("--queries", type=int, default=1000)
a = p.parse_args()
K = 10
# FAISS 1.16's default (128,000) sends flat L2 below 128 dimensions down a path 4-7x slower than BLAS.
faiss.cvar.distance_compute_blas_threshold = 20
xb, xq = load(a.dataset, a.queries)
xb_mx = mx.array(xb)
norms = mx.sum(xb_mx * xb_mx, axis=1)
exact = (lambda q: knn_tiled(q, xb_mx, K, "l1")) if a.metric == "l1" else (lambda q: l2_knn(q, xb_mx, norms, K))
truth = np.concatenate([np.array(exact(mx.array(xq[i:i + 256]))) for i in range(0, len(xq), 256)])
metric = faiss.METRIC_L1 if a.metric == "l1" else faiss.METRIC_L2


def recall(found):
    return np.mean([len(set(r) & set(t)) / K for r, t in zip(found, truth)])


def sweep(name, index, knob, values):
    for v in values:
        knob(v)
        start = time.perf_counter()
        _, found = index.search(xq, K)
        t = time.perf_counter() - start
        print(f"  {name} {v}: recall@10 {recall(found):.3f}, {len(xq) / t:,.0f} queries/s", flush=True)


d = xb.shape[1]
start = time.perf_counter()
hnsw = faiss.IndexHNSWFlat(d, 32, metric)
hnsw.hnsw.efConstruction = 40
hnsw.add(xb)
print(f"{a.dataset}: HNSW (M=32, {a.metric}) on {len(xb):,} vectors: built in {time.perf_counter() - start:.1f} s", flush=True)
sweep("efSearch", hnsw, lambda v: setattr(hnsw.hnsw, "efSearch", v), (16, 32, 64, 128, 256))

start = time.perf_counter()
nlist = 4000
ivf = faiss.IndexIVFFlat(faiss.IndexFlat(d, metric), d, nlist, metric)
ivf.train(xb[np.random.default_rng(0).choice(len(xb), min(len(xb), 100 * nlist), replace=False)])
ivf.add(xb)
print(f"{a.dataset}: IVF ({nlist} lists, {a.metric}) on {len(xb):,} vectors: built in {time.perf_counter() - start:.1f} s", flush=True)
sweep("nprobe", ivf, lambda v: setattr(ivf, "nprobe", v), (1, 4, 16, 64))
