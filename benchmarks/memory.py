"""How much memory a siann.Index search takes, by path, batch size and index size: whether it stays bounded as
the batch grows. Indexes on SIFT1M's first --sizes vectors (exact L2 and L1, and an IVF at recall 0.95) are
saved once; each case then runs in a fresh process, which loads one, warms the device's copy with a search of
10 queries, and runs one search of the batch (base vectors as queries past the dataset's 10,000) while a
thread samples the process's resident memory every half millisecond. Reports the resident memory after the
warm-up, the search's peak above it, and on the GPU, MLX's own peak allocation above what was allocated
before the search and its buffer cache after it. Also an add of 1% of the index that triggers a merge.

    uv run --group bench python memory.py
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

MB = 1 << 20


def sampled_peak(fn):
    """fn's result and the peak resident memory while it ran, sampled every half millisecond."""
    import psutil
    proc, peak, done = psutil.Process(), [0], threading.Event()

    def watch():
        while not done.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.0005)
    t = threading.Thread(target=watch)
    t.start()
    try:
        out = fn()
    finally:
        done.set()
        t.join()
    return out, max(peak[0], proc.memory_info().rss)


def case(path: str, device: str, batch: int, merge: bool) -> dict:
    """One measurement, in this process (run as a child)."""
    import psutil
    from datasets import load

    import siann
    index = siann.Index.load(path, device=device if device == "cpu" else "auto")
    xb, xq = load("sift", 10_000)
    q = np.ascontiguousarray(np.concatenate([xq, xb[:max(0, batch - len(xq))]])[:batch])
    index.search(q[:10], 10, device=device)                           # build the device's copy
    gpu = device == "gpu"
    if gpu:
        import mlx.core as mx
        mx.reset_peak_memory()
        active = mx.get_active_memory()
    base = psutil.Process().memory_info().rss
    if merge:
        extra = np.ascontiguousarray(xb[:max(10_000, index.n // 100)] + 0.5)     # new vectors at every size
        _, peak = sampled_peak(lambda: index.add(extra))
    else:
        _, peak = sampled_peak(lambda: index.search(q, 10, device=device))
    out = {"rss": base / MB, "peak": (peak - base) / MB}
    if gpu:
        out["mlx_peak"] = (mx.get_peak_memory() - active) / MB
        out["mlx_cache"] = mx.get_cache_memory() / MB
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="200000,1000000")
    p.add_argument("--batches", default="1,100,1000,10000,100000")
    p.add_argument("--cpu-exact-max", type=int, default=10_000, help="largest batch for exact search on the CPU")
    p.add_argument("--child", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.child:
        print(json.dumps(case(**json.loads(a.child))))
        return
    from datasets import load

    import siann
    xb, xq = load("sift", 1000)
    work = tempfile.mkdtemp()
    batches = [int(b) for b in a.batches.split(",")]
    print("siann.Index memory, MB: resident after the warm-up / the search's peak above it; on the GPU also MLX's "
          "peak allocation during the search / its cache after")
    for n in map(int, a.sizes.split(",")):
        x = np.ascontiguousarray(xb[:n])
        print(f"\nSIFT, {n:,} vectors ({x.nbytes / MB:,.0f} MB as float32):", flush=True)
        built = {"exact l2": siann.Index(x, "l2"), "exact l1": siann.Index(x, "l1"),
                 "ivf l2": siann.Index(x, "l2", recall=0.95, calibration_queries=xq)}
        paths = {}
        for name, index in built.items():
            paths[name] = os.path.join(work, f"{name.replace(' ', '_')}_{n}.siann")
            index.save(paths[name])
        del built
        print(f"  {'case':18s} {'batch':>8s} {'resident':>9s} {'peak':>8s} {'MLX peak':>9s} {'MLX cache':>10s}")
        runs = [(name, dev, b, False) for name in paths for dev in ("cpu", "gpu") for b in batches
                if not (name.startswith("exact") and dev == "cpu" and b > a.cpu_exact_max)]
        runs += [("ivf l2", dev, 10, True) for dev in ("cpu", "gpu")]
        for name, dev, b, merge in runs:
            if dev == "gpu" and not siann.gpu_available():
                continue
            arg = json.dumps({"path": paths[name], "device": dev, "batch": b, "merge": merge})
            r = subprocess.run([sys.executable, __file__, "--child", arg], capture_output=True, text=True,
                               encoding="utf-8", check=True)
            m = json.loads(r.stdout.strip().splitlines()[-1])
            label = f"{name}, {dev}" + (", merge" if merge else "")
            gpu = f"{m['mlx_peak']:9,.0f} {m['mlx_cache']:10,.0f}" if "mlx_peak" in m else f"{'':9s} {'':10s}"
            print(f"  {label:18s} {'' if merge else f'{b:,}':>8s} {m['rss']:9,.0f} {m['peak']:8,.0f} {gpu}", flush=True)


if __name__ == "__main__":
    main()
