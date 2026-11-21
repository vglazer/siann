"""The facts DATASETS.md states about each dataset: sizes, value range, whether the coordinates are
integers (and fit in a byte), vector lengths, and how unevenly long the base vectors are.

    uv run --group bench python dataset_stats.py --datasets sift,glove,glove-raw,music100
"""
import argparse

import numpy as np
from datasets import load


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="sift,glove,glove-raw,music100,dbpedia")
    a = p.parse_args()
    for name in a.datasets.split(","):
        xb, xq = load(name, 10_000)
        lb, lq = np.linalg.norm(xb, axis=1), np.linalg.norm(xq, axis=1)
        integer = bool(np.array_equal(np.rint(xb), xb) and np.array_equal(np.rint(xq), xq))
        byte = integer and xb.min() >= 0 and xb.max() <= 255
        pct = np.percentile(lb, [1, 50, 99])
        print(f"{name}: {xb.shape[0]:,} base vectors, {xq.shape[0]:,} queries, {xb.shape[1]} dimensions")
        print(f"  values {xb.min():.4g} to {xb.max():.4g}; finite {bool(np.isfinite(xb).all() and np.isfinite(xq).all())}; "
              f"integers {integer}; bytes {byte}")
        print(f"  base lengths {lb.min():.4g} to {lb.max():.4g} (1st, 50th, 99th percentile {pct[0]:.4g}, "
              f"{pct[1]:.4g}, {pct[2]:.4g}); zero-length {int((lb == 0).sum())}")
        print(f"  query lengths {lq.min():.4g} to {lq.max():.4g}, median {np.median(lq):.4g}")
        print(f"  longest / median base length {lb.max() / pct[1]:.3g}; share of base vectors within 10% of "
              f"the longest {np.mean(lb >= 0.9 * lb.max()):.4f}", flush=True)


if __name__ == "__main__":
    main()
