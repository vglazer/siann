"""Approximate search's routing thresholds, measured with siann.tune(recall=0.95) on SIFT1M (L2), GloVe-100
(cosine) and Music-100 (inner product), calibrated and timed on each dataset's own queries: the batch size
from which the GPU beats the CPU (cpu_below) and from which each device's list-major scan beats its
query-major scan (gpu_list_from, cpu_list_from). They differ with nprobe, which is why each index measures
its own when it is built.

    uv run --group bench python ivf_routing.py
"""
import argparse

from datasets import load

import siann

PLAN = (("sift", "l2"), ("glove", "cosine"), ("music100", "ip"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--recall", type=float, default=0.95)
    p.add_argument("--budget", type=float, default=120.0)
    a = p.parse_args()
    found = []
    for name, metric in PLAN:
        xb, xq = load(name, 5000)
        prof = siann.tune(xb, xq, metrics=(metric,), batches=(1, 1000), recall=a.recall, budget=a.budget)
        print(f"{name}:\n{prof}\n", flush=True)
        found.append((name, prof.ivf["nprobe"], prof.ivf["cpu_below"], prof.ivf["gpu_list_from"],
                      prof.ivf["cpu_list_from"]))
    print("thresholds by dataset (10**9: never, among the batch sizes measured):")
    print(f"  {'dataset':9s} {'nprobe':>6s} {'cpu_below':>10s} {'gpu_list_from':>14s} {'cpu_list_from':>14s}")
    for name, n, c, gl, cl in found:
        print(f"  {name:9s} {n:6d} {c:10,} {gl:14,} {cl:14,}")


if __name__ == "__main__":
    main()
