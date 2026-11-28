"""siann.Index with a recall target on real data and real queries: whether calibrating nprobe on a sample
of the index's own vectors gives the target recall on the dataset's own queries, what the index chose, and
its speed by batch size on the paths it routes to.

    uv run --group bench python index_recall.py --dataset sift --recall 0.95
    uv run --group bench python index_recall.py --dataset sift --calibrate-on queries   # held-out queries
    uv run --group bench python index_recall.py --dataset sift --calibrate-on queries --gpu torch
"""
import argparse
import time

import siann

import numpy as np
from backends import agreement
from datasets import load

METRIC = {"sift": "l2", "glove": "cosine", "music100": "ip", "dbpedia": "cosine"}
K = 10


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sift", choices=sorted(METRIC))
    p.add_argument("--recall", type=float, default=0.95)
    p.add_argument("--base", type=int, default=0)
    p.add_argument("--queries", type=int, default=10_000)
    p.add_argument("--gpu", default="auto", choices=["auto", "mlx", "torch"], help="siann.Index's GPU backend")
    p.add_argument("--calibrate-on", default="own", choices=["own", "queries"],
                   help="calibrate on the index's own vectors, or on the first 1,000 queries (then measure on the rest)")
    a = p.parse_args()
    metric = METRIC[a.dataset]
    xb, xq = load(a.dataset, a.queries)
    if a.base:
        xb = np.ascontiguousarray(xb[:a.base])
    exact = siann.Index(xb, metric)
    truth = exact.search(xq, K).indices
    t = time.perf_counter()
    if a.calibrate_on == "queries":
        cal, xq, truth = xq[:1000], xq[1000:], truth[1000:]
        index = siann.Index(xb, metric, recall=a.recall, calibration_queries=cal, gpu=a.gpu)
    else:
        index = siann.Index(xb, metric, recall=a.recall, gpu=a.gpu)
    print(f"{a.dataset}, {metric}: {len(xb):,} vectors, target {a.recall}; built in {time.perf_counter() - t:.1f} s\n"
          f"{index.plan}\n", flush=True)
    found = index.search(xq, K).indices
    # recall on the dataset's queries, ties counted as found (cosine and L2 rank alike on these unit vectors)
    r = agreement(xb, xq, found, truth, "ip" if metric in ("ip", "cosine") and a.dataset == "music100" else "l2")
    print(f"  recall@{K} on the dataset's {len(xq):,} queries: {r:.3f} (target {a.recall})")
    for b in (1, 10, 100, 1000, 10_000):
        q = xq[:min(b, len(xq))]
        reps = max(1, 2000 // b) if b < 1000 else 3
        warm = time.perf_counter()      # steady state: a device idle before this batch starts slower
        while time.perf_counter() - warm < 0.2:
            index.search(q[:1] if b == 1 else q, K)
        t = time.perf_counter()
        for i in range(reps):
            index.search(xq[i:i + 1] if b == 1 else q, K)
        rate = reps * len(q) / (time.perf_counter() - t)
        print(f"  batch {b:>6,}: {index._ivf_path(len(q), None) if index.kind == 'ivf' else index.route(len(q)):10s}"
              f" {rate:10,.0f} queries/s", flush=True)


if __name__ == "__main__":
    main()
