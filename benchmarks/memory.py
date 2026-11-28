"""How much memory a siann.Index search takes, by path, batch size and index size: whether it stays bounded as
the batch grows. Indexes on SIFT1M's first --sizes vectors (exact L2 and L1, and an IVF at recall 0.95) are
saved once; each case then runs in a fresh process, which loads one, warms the device's copy with a search of
10 queries, and runs one search of the batch (base vectors as queries past the dataset's 10,000) while a
thread samples the process's resident memory every half millisecond. Reports the resident memory after the
warm-up, the search's peak above it, and on the GPU, MLX's own peak allocation above what was allocated
before the search and its buffer cache after it. Also an add of 1% of the index that triggers a merge.
With --gpu torch, the GPU cases on PyTorch's MPS backend instead, which reports no peak: its driver's
allocation (cached buffers included) is sampled with the resident memory, and reported above what it was
before the search, and after it.

    uv run --group bench python memory.py
    uv run --group bench python memory.py --gpu torch
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


def sampled_peak(fn, extra=None):
    """fn's result, the peak resident memory while it ran and the peak of ``extra()`` (another counter, or 0),
    both sampled every half millisecond."""
    import psutil
    proc, peak, peak2, done = psutil.Process(), [0], [0], threading.Event()
    extra = extra or (lambda: 0)

    def watch():
        while not done.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            peak2[0] = max(peak2[0], extra())
            time.sleep(0.0005)
    t = threading.Thread(target=watch)
    t.start()
    try:
        out = fn()
    finally:
        done.set()
        t.join()
    return out, max(peak[0], proc.memory_info().rss), max(peak2[0], extra())


def case(path: str, device: str, batch: int, merge: bool, backend: str = "mlx") -> dict:
    """One measurement, in this process (run as a child)."""
    import siann

    import psutil
    from datasets import load
    index = siann.Index.load(path, device=device if device == "cpu" else "auto", gpu=backend)
    xb, xq = load("sift", 10_000)
    q = np.ascontiguousarray(np.concatenate([xq, xb[:max(0, batch - len(xq))]])[:batch])
    index.search(q[:10], 10, device=device)                           # build the device's copy
    gpu = device == "gpu"
    counter = None
    if gpu and backend == "torch":
        import torch
        counter = torch.mps.driver_allocated_memory
        torch.mps.synchronize()
        before = counter()
    elif gpu:
        import mlx.core as mx
        mx.reset_peak_memory()
        active = mx.get_active_memory()
    base = psutil.Process().memory_info().rss
    if merge:
        extra = np.ascontiguousarray(xb[:max(10_000, index.n // 100)] + 0.5)     # new vectors at every size
        _, peak, peak2 = sampled_peak(lambda: index.add(extra), counter)
    else:
        _, peak, peak2 = sampled_peak(lambda: index.search(q, 10, device=device), counter)
    out = {"rss": base / MB, "peak": (peak - base) / MB}
    if gpu and backend == "torch":
        out["gpu_peak"], out["gpu_after"] = (peak2 - before) / MB, counter() / MB
    elif gpu:
        out["gpu_peak"] = (mx.get_peak_memory() - active) / MB
        out["gpu_after"] = mx.get_cache_memory() / MB
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="200000,1000000")
    p.add_argument("--batches", default="1,100,1000,10000,100000")
    p.add_argument("--cpu-exact-max", type=int, default=10_000, help="largest batch for exact search on the CPU")
    p.add_argument("--gpu", default="mlx", choices=["mlx", "torch"],
                   help="the GPU backend; with torch, the GPU cases only")
    p.add_argument("--child", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.child:
        print(json.dumps(case(**json.loads(a.child))))
        return
    import siann

    from datasets import load
    xb, xq = load("sift", 1000)
    work = tempfile.mkdtemp()
    batches = [int(b) for b in a.batches.split(",")]
    torch = a.gpu == "torch"
    gpu_cols = ("PyTorch peak", "after") if torch else ("MLX peak", "MLX cache")
    print("siann.Index memory, MB: resident after the warm-up / the search's peak above it; on the GPU also "
          + ("PyTorch's driver allocation, its peak during the search above it before / after"
             if torch else "MLX's peak allocation during the search / its cache after"))
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
        print(f"  {'case':18s} {'batch':>8s} {'resident':>9s} {'peak':>8s} {gpu_cols[0]:>12s} {gpu_cols[1]:>10s}")
        devices = ("gpu",) if torch else ("cpu", "gpu")
        runs = [(name, dev, b, False) for name in paths for dev in devices for b in batches
                if not (name.startswith("exact") and dev == "cpu" and b > a.cpu_exact_max)]
        runs += [("ivf l2", dev, 10, True) for dev in devices]
        for name, dev, b, merge in runs:
            if dev == "gpu" and not siann.gpu_available():
                continue
            arg = json.dumps({"path": paths[name], "device": dev, "batch": b, "merge": merge, "backend": a.gpu})
            r = subprocess.run([sys.executable, __file__, "--child", arg], capture_output=True, text=True,
                               encoding="utf-8", check=True)
            m = json.loads(r.stdout.strip().splitlines()[-1])
            label = f"{name}, {dev}" + (", merge" if merge else "")
            gpu = f"{m['gpu_peak']:12,.0f} {m['gpu_after']:10,.0f}" if "gpu_peak" in m else f"{'':12s} {'':10s}"
            print(f"  {label:18s} {'' if merge else f'{b:,}':>8s} {m['rss']:9,.0f} {m['peak']:8,.0f} {gpu}", flush=True)


if __name__ == "__main__":
    main()
