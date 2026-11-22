"""Searching a large batch on the GPU in chunks (siann._memory.GPU_BUDGET) against in one call: the same
neighbors, the speed, and MLX's peak memory. An IVF at recall 0.95 (calibrated on 1,000 of the dataset's
queries) on SIFT1M and on GloVe-100, both GPU scans, batches of 1,000 to 100,000 queries (the dataset's
queries, then base vectors).

    uv run --group bench python gpu_chunks.py
"""
import argparse
import time

import mlx.core as mx
import numpy as np
from datasets import load

import siann
from siann import _ivf, _memory

MB = 1 << 20


def timed(index, q, path):
    """The result, the best of three times, and MLX's peak memory above what was allocated before."""
    index._ivf.search(q[:1000], 10, index.nprobe, path)                       # warm
    mx.clear_cache()
    mx.reset_peak_memory()
    active = mx.get_active_memory()
    times = []
    for _ in range(3):
        t = time.perf_counter()
        out = index._ivf.search(q, 10, index.nprobe, path)
        times.append(time.perf_counter() - t)
    return out, min(times), (mx.get_peak_memory() - active) / MB


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="sift,glove")
    p.add_argument("--batches", default="1000,10000,100000")
    p.add_argument("--budgets", default="", help="GPU_BUDGET values to compare, in MB (default: siann's)")
    a = p.parse_args()
    budgets = [int(m) * MB for m in a.budgets.split(",")] if a.budgets else [_memory.budget("mlx")]
    for name in a.datasets.split(","):
        xb, xq = load(name, 10_000)
        metric = "cosine" if name == "glove" else "l2"
        index = siann.Index(xb, metric, recall=0.95, calibration_queries=xq[:1000])
        sizes = []
        for budget in budgets:
            sizes.append(f"{budget // MB:,} MB: {_ivf.gpu_chunk(10, index.nprobe, index._ivf.nlist, budget):,}")
        print(f"\n{name}, {metric}: {len(xb):,} vectors, {index._ivf.nlist:,} lists, nprobe {index.nprobe}; "
              f"queries per chunk by budget: {', '.join(sizes)}")
        print(f"  {'path':9s} {'batch':>8s}  {'one call: q/s, MB':>20s}"
              + "".join(f"  {f'{b // MB:,} MB: q/s, MB':>20s}" for b in budgets) + "  identical")
        q = np.concatenate([xq, xb[:max(0, max(map(int, a.batches.split(","))) - len(xq))]])
        if metric == "cosine":
            q = q / np.linalg.norm(q, axis=1, keepdims=True)
        q = np.ascontiguousarray(q, dtype=np.float32)
        for path in ("gpu-query", "gpu-list"):
            for b in map(int, a.batches.split(",")):
                one = None
                if path == "gpu-query" or b * index.nprobe * 16 * 10 < 2 ** 31:   # one call overflows MLX past it
                    _memory.GPU_BUDGET, _memory.BUDGET_SHARE = 1 << 62, 1           # one call: the whole working set
                    one, t1, m1 = timed(index, q[:b], path)
                cells, same = [f"{b / t1:11,.0f} {m1:8,.0f}" if one is not None else f"{'overflows':>20s}"], True
                for budget in budgets:
                    _memory.GPU_BUDGET, _memory.BUDGET_SHARE = budget, 1
                    chunked, t2, m2 = timed(index, q[:b], path)
                    cells.append(f"{b / t2:11,.0f} {m2:8,.0f}")
                    same = same and (one is None or np.array_equal(one, chunked))
                print(f"  {path:9s} {b:8,}  " + "  ".join(cells) + f"  {same}", flush=True)
        _memory.GPU_BUDGET, _memory.BUDGET_SHARE = budgets[0], 1


if __name__ == "__main__":
    main()
