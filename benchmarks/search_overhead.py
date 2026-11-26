"""Where a search's time goes on the CPU: siann.Index.search against the parts of it, at small batches, where
Python's share is largest. For the path the index routes each batch to: the raw Rust search (siann._cpu's
GraphIndex or IvfIndex), the same search through siann's Python wrapper (the IVF's or the graph's ``search``,
with its reshaping), the ranking of the results by their exact scores (siann._cpu.rank), and the rest of
Index.search (its checks, the snapshot, picking the path). Each part the median of back-to-back calls (at least
0.3 s and 20 calls), in microseconds per call.

    uv run --group bench python search_overhead.py --dataset sift
"""
import argparse
import time

import numpy as np
from datasets import load

import siann
from siann._cpu import rank

METRIC = {"sift": "l2", "music100": "ip", "glove": "cosine"}
K = 10


def per_call(fn) -> float:
    """Microseconds per call: the median of back-to-back calls after a warm-up."""
    fn()
    times, start = [], time.perf_counter()
    while len(times) < 20 or time.perf_counter() - start < 0.3:
        t = time.perf_counter_ns()
        fn()
        times.append(time.perf_counter_ns() - t)
    return float(np.median(times)) / 1e3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sift", choices=sorted(METRIC))
    p.add_argument("--recall", type=float, default=0.95)
    p.add_argument("--batches", default="1,10,100")
    a = p.parse_args()
    xb, xq = load(a.dataset, 10_000)
    metric = METRIC[a.dataset]
    t = time.perf_counter()
    index = siann.Index(xb, metric, recall=a.recall, calibration_queries=xq[:1000])
    print(f"{a.dataset}: {len(xb):,} x {xb.shape[1]}, {metric}, target {a.recall}: built in {time.perf_counter() - t:.0f} s "
          f"({'graph by ' + index._ivf.graph.builder if index._ivf is not None and index._ivf.graph else 'no graph'}); "
          f"microseconds per call, the median of back-to-back calls", flush=True)
    print(f"  {'batch':>5} {'path':>15} {'Index.search':>13} {'Rust search':>12} {'wrapper':>8} {'rank':>6} "
          f"{'the rest':>9} {'Python share':>13}", flush=True)
    test = np.ascontiguousarray(xq[1000:], dtype=np.float32)
    ivf = index._ivf
    for b in map(int, a.batches.split(",")):
        q = np.ascontiguousarray(test[:b])
        qn = q / np.linalg.norm(q, axis=1, keepdims=True) if metric == "cosine" else q
        qn = np.ascontiguousarray(qn, dtype=np.float32)
        path = index._ivf_path(b, None) if ivf is not None else f"exact-{index.route(b)}"
        total = per_call(lambda q=q: index.search(q, K))
        if path == "cpu-graph":
            g = ivf.graph
            raw = per_call(lambda: g._cpu_index().search(qn, K, max(g.ef, K)))
            wrapped = per_call(lambda: g.search_cpu(qn, K))
            rows = g.search_cpu(qn, K)
        elif path.startswith("cpu-") and not path.endswith("exact"):
            cpu = ivf.cpu()
            split = 4 if path == "cpu-split" else 1
            raw = (per_call(lambda: cpu.search(qn, K, index.nprobe, split=split)) if path in ("cpu-query", "cpu-split")
                   else float("nan"))
            wrapped = per_call(lambda: ivf.search(qn, K, index.nprobe, path))
            rows = ivf.search(qn, K, index.nprobe, path)
        else:
            raw = wrapped = float("nan")
            rows = index.search(q, K).indices
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        ranked = per_call(lambda: rank(index._x, qn, rows, index._kernel_metric))
        rest = total - (0 if np.isnan(wrapped) else wrapped) - ranked
        python = float("nan") if np.isnan(raw) else (total - raw) / total
        print(f"  {b:>5} {path:>15} {total:>13.1f} {raw:>12.1f} {wrapped - raw:>8.1f} {ranked:>6.1f} {rest:>9.1f} "
              f"{python:>12.0%}", flush=True)


if __name__ == "__main__":
    main()
