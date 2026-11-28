"""An index's measured routing against fixed thresholds, on the same index and the same queries, the two
timed side by side at each batch size: three rounds, the two paths in turn, each timed back to back after a
0.1 s warm-up (at least 5 calls and 50 ms per round), the median call of each. The fixed thresholds are the ones
siann used before measuring: single queries on the CPU split over 4 threads, the GPU's query-major scan
below 1,000 queries and its list-major scan from 1,000.

    uv run --group bench python index_routing_ab.py --dataset sift
    uv run --group bench python index_routing_ab.py --dataset music100 --calibrate-on queries
"""
import argparse
import time

import siann

import numpy as np
from datasets import load

METRIC = {"sift": "l2", "glove": "cosine", "music100": "ip"}
K = 10


def fixed_path(b: int) -> str:
    return "cpu-split" if b == 1 else ("gpu-query" if b < 1000 else "gpu-list")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sift", choices=sorted(METRIC))
    p.add_argument("--recall", type=float, default=0.95)
    p.add_argument("--calibrate-on", default="own", choices=["own", "queries"])
    a = p.parse_args()
    xb, xq = load(a.dataset, 10_000)
    cal = xq[:1000] if a.calibrate_on == "queries" else None
    t = time.perf_counter()
    index = siann.Index(xb, METRIC[a.dataset], recall=a.recall, calibration_queries=cal)
    print(f"{a.dataset}, {METRIC[a.dataset]}, target {a.recall}, calibrated on {a.calibrate_on}: built in "
          f"{time.perf_counter() - t:.1f} s, nprobe {index.nprobe}")
    print("\n".join(line for line in str(index.plan).splitlines() if "routing" in line))
    q = xq[1000:] if cal is not None else xq
    ivf, nprobe = index._ivf, index.nprobe
    print(f"\n  {'batch':>6s} {'fixed path':>10s} {'q/s':>9s}   {'measured path':>13s} {'q/s':>9s}   {'ratio':>6s}")
    for b in (1, 2, 4, 8, 16, 32, 64, 100, 128, 256, 512, 1000, 2048, 4096, 9000):
        if b > len(q):
            break
        qq = q[:b]
        paths = (fixed_path(b), index._ivf_path(b, None))
        times = {path: [] for path in dict.fromkeys(paths)}
        for _ in range(3):                      # three rounds, the paths in turn
            for path in times:
                warm = time.perf_counter()      # back to back, warmed up for 0.1 s: after calls on the other
                while time.perf_counter() - warm < 0.1:     # device, a GPU call can run 40% slower
                    ivf.search(qq, K, nprobe, path)
                run = []
                while len(run) < 5 or sum(run) < 0.05:
                    start = time.perf_counter()
                    ivf.search(qq, K, nprobe, path)
                    run.append(time.perf_counter() - start)
                times[path] += run
        rate = {path: b / np.median(times[path]) for path in paths}
        f, m = paths
        print(f"  {b:6,} {f:>10s} {rate[f]:9,.0f}   {m:>13s} {rate[m]:9,.0f}   {rate[m] / rate[f]:6.2f}", flush=True)


if __name__ == "__main__":
    main()
