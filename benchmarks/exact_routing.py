"""siann's default profile: where exact search is faster on the CPU than on the GPU, by batch size and
metric, measured with siann.tune on SIFT1M (L2, L1, L-infinity), GloVe-100 (cosine, inner product) and
DBpedia OpenAI (cosine, L2; the first 100,000 vectors), k = 10. For each metric, the default threshold is
the smallest crossover across the datasets that measure it, so that the CPU takes only batch sizes where it
was faster on every one, and no measured case is slower than sending every batch to the GPU.

    uv run --group bench python exact_routing.py
"""
import argparse

import numpy as np
from datasets import load

import siann

PLAN = (("sift", 0, ("l2", "l1", "linf")), ("glove", 0, ("cosine", "ip")), ("dbpedia", 100_000, ("cosine", "l2")))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=float, default=120.0, help="seconds per dataset")
    a = p.parse_args()
    found: dict[str, list] = {}
    for name, base, metrics in PLAN:
        xb, xq = load(name, 1024)
        if base:
            xb = np.ascontiguousarray(xb[:base])
        prof = siann.tune(xb, xq, metrics=metrics, budget=a.budget)
        print(f"{name}:\n{prof}\n", flush=True)
        for m in metrics:
            found.setdefault(m, []).append((name, prof.threshold(m)))
    print("default thresholds (the CPU below this many queries per batch; the smallest across datasets):")
    for m in siann.METRICS:
        ts = found.get(m, [])
        values = [t if t is not None else float("inf") for _, t in ts]
        best = min(values) if values else 1
        print(f"  {m:7s} {best if best != float('inf') else None}   ({', '.join(f'{n}: {t}' for n, t in ts)})")


if __name__ == "__main__":
    main()
