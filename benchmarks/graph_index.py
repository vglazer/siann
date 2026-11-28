"""siann.Index with its graph (CAGRA through mlxvs, for the GPU batches routing finds it faster) against the
same index without it (kind="ivf"), FAISS's HNSW at matched recall and mlxvs's CAGRA on its own, on the
datasets where CAGRA beat siann's IVF: SIFT1M (L2), Music-100 (inner product) and DBpedia OpenAI's first
100,000 vectors (1,536 dimensions, served normalized, so L2 ranks as cosine). siann at recall target 0.95,
calibrated on 1,000 of the dataset's queries (half of them where there are fewer than 2,000); recall@10
and queries per second on the others. FAISS's HNSW (M = 32,
efConstruction 40, 16 threads) at the smallest efSearch whose recall reaches siann's at that batch size
(by bisection);
on SIFT also with byte vectors (IndexHNSWSQ, QT_8bit_direct), its fastest there. mlxvs's CAGRA, at the
smallest itopk_size that reaches siann's recall.

    uv run --group bench python graph_index.py --dataset sift
"""
import argparse
import time

import siann
from siann._cpu import FlatIndex

import faiss
import mlx.core as mx
import mlxvs
import numpy as np
from backends import agreement
from datasets import load

K = 10
WIDEST = 1024           # the largest efSearch or itopk_size tried
METRIC = {"sift": "l2", "music100": "ip", "dbpedia": "l2"}


def rate(fn, q, b):
    """Queries per second searching b queries per call (best of three passes over up to 1,000 queries, or one
    call where b is larger)."""
    q = q[:max(b, min(len(q), 100 * b))]

    def run():
        for s in range(0, len(q), b):
            fn(q[s:s + b])
    run()
    best = float("inf")
    for _ in range(3):
        t = time.perf_counter()
        run()
        best = min(best, time.perf_counter() - t)
    return len(q) / best


def smallest(recall_at, want):
    """The smallest setting from 10 to WIDEST whose recall reaches ``want`` (by bisection, recall rising with
    the setting), and its recall; WIDEST where none does."""
    cache = {}

    def at(v):
        if v not in cache:
            cache[v] = recall_at(v)
        return cache[v]
    lo, hi = 10, WIDEST
    if at(lo) >= want:
        return lo, at(lo)
    if at(hi) < want:
        return hi, at(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if at(mid) >= want else (mid, hi)
    return hi, at(hi)


def batched(fn, q, b):
    return np.concatenate([fn(q[s:s + b]) for s in range(0, len(q), b)])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sift", choices=sorted(METRIC))
    p.add_argument("--base", type=int, default=0, help="the first N base vectors (0: all)")
    p.add_argument("--queries", type=int, default=10_000)
    p.add_argument("--batches", default="1,100,0", help="queries per call; 0: every test query in one call")
    a = p.parse_args()
    metric = METRIC[a.dataset]
    xb, xq = load(a.dataset, a.queries)
    if a.base:
        xb = np.ascontiguousarray(xb[:a.base])
    xb, xq = np.ascontiguousarray(xb), np.ascontiguousarray(xq)
    cal, test = xq[:min(1000, len(xq) // 2)], xq[min(1000, len(xq) // 2):]
    batches = [b if b else len(test) for b in map(int, a.batches.split(",")) if b <= len(test)]
    n, d = xb.shape
    truth = np.array(FlatIndex(xb, d, metric).search(test, K), dtype=np.int64).reshape(-1, K)
    rec = lambda found: agreement(xb, test, found, truth, metric)
    print(f"{a.dataset}, {metric}: {n:,} vectors, {d} dimensions; {len(cal):,} calibration queries, recall@10 "
          f"on {len(test):,} others; queries per second")

    rows = []
    for label, kind in (("siann, with the graph", "auto"), ("siann, IVF only", "ivf")):
        t = time.perf_counter()
        index = siann.Index(xb, metric, recall=0.95, kind=kind, calibration_queries=cal)
        built = time.perf_counter() - t
        cells, recalls = [], []
        for b in batches:
            found = batched(lambda q: index.search(q, K).indices, test, b)
            recalls.append(rec(found))
            cells.append((index._ivf_path(b, None), rate(lambda q: index.search(q, K), test, b)))
        rows.append((label, built, recalls, cells))
        if kind == "auto":
            target = recalls
            graph_line = next((r for r in index.plan.reasons if r.startswith(("graph index", "no graph"))), "")
            route_line = next((r for r in index.plan.reasons if r.startswith("routing")), "")

    # FAISS's HNSW: built once per storage, efSearch matched to siann's recall at each batch size
    xb_f, xq_f = (xb, test)
    faiss_metric = faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2
    storages = [("float32", faiss.IndexHNSWFlat(d, 32, faiss_metric))]
    if a.dataset == "sift":
        storages.append(("uint8", faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit_direct, 32, faiss_metric)))
    for name, h in storages:
        t = time.perf_counter()
        if name != "float32":
            h.train(xb_f)
        h.add(xb_f)
        built = time.perf_counter() - t
        def hnsw_recall(ef, h=h):
            h.hnsw.efSearch = ef
            return rec(h.search(xq_f, K)[1].astype(np.int64))
        cells, recalls = [], []
        for b, want in zip(batches, target):
            ef, got = smallest(hnsw_recall, want)
            h.hnsw.efSearch = ef
            recalls.append(got)
            cells.append((f"ef {ef}", rate(lambda q: h.search(q, K), xq_f, b)))
        rows.append((f"FAISS HNSW, {name}", built, recalls, cells))

    # mlxvs's CAGRA on its own, itopk matched to siann's recall
    t = time.perf_counter()
    cg = mlxvs.CagraIndex.build(xb, graph_degree=32, metric="inner_product" if metric == "ip" else "l2",
                                dtype="uint8" if a.dataset == "sift" else "float16")
    built = time.perf_counter() - t
    q_mx = mx.array(test)
    def cagra_recall(it):
        return rec(np.asarray(cg.search(q_mx, K, itopk_size=it)[0]).astype(np.int64))
    cells, recalls = [], []
    for b, want in zip(batches, target):
        it, got = smallest(cagra_recall, want)
        recalls.append(got)
        cells.append((f"itopk {it}", rate(lambda q, it=it: mx.eval(cg.search(mx.array(q), K, itopk_size=it)[0]),
                                          test, b)))
    rows.append(("mlxvs CAGRA", built, recalls, cells))

    print(f"  {graph_line}\n  {route_line}\n")
    head = "".join(f"  batch {b:,}: recall, path, q/s".rjust(40) for b in batches)
    print(f"  {'':24s} {'build':>7s}{head}")
    for label, built, recalls, cells in rows:
        print(f"  {label:24s} {built:6.1f}s" + "".join(
            f"{r:9.3f} {path:>14s} {qps:13,.0f}" for r, (path, qps) in zip(recalls, cells)), flush=True)


if __name__ == "__main__":
    main()
