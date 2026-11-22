"""A siann.Index grown from empty: SIFT1M's first --n vectors added --step at a time to an empty index (recall
target 0.95, calibrated on 1,000 of the dataset's queries), against an index built from scratch on the same
vectors with the same target and queries. Prints, at each merge, the index's kind, its lists and nprobe and the
time of the add that merged; then recall@10 on 9,000 other queries and queries per second by batch size, for
both.

    uv run --group bench python grow.py
"""
import argparse
import time

import numpy as np
from backends import agreement
from datasets import load

import siann

BATCHES = (1, 100, 1000)


def rate(index, q, b: int) -> float:
    """Queries per second searching b queries per call (median of three passes over up to 1,000 queries)."""
    q = q[:max(b, min(len(q), 100 * b))]

    def run():
        for s in range(0, len(q), b):
            index.search(q[s:s + b], 10)
    run()
    times = []
    for _ in range(3):
        t = time.perf_counter()
        run()
        times.append(time.perf_counter() - t)
    return len(q) / sorted(times)[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sift")
    p.add_argument("--n", type=int, default=200_000)
    p.add_argument("--step", type=int, default=10_000)
    a = p.parse_args()
    xb, xq = load(a.dataset, 10_000)
    x = np.ascontiguousarray(xb[:a.n])
    cal, test = xq[:1000], xq[1000:]
    print(f"{a.dataset}: {a.n:,} vectors added {a.step:,} at a time to an empty index (target 0.95, calibrated "
          f"on 1,000 of the dataset's queries)")
    grown = siann.Index(np.empty((0, x.shape[1]), np.float32), "l2", recall=0.95, calibration_queries=cal)
    total = 0.0
    print(f"  {'vectors':>9s}  {'kind':5s} {'lists':>6s} {'nprobe':>6s} {'add, s':>7s}")
    for s in range(0, a.n, a.step):
        merges = grown._merges
        t = time.perf_counter()
        grown.add(x[s:s + a.step])
        took = time.perf_counter() - t
        total += took
        if grown._merges > merges:
            lists = f"{grown._ivf.nlist:,}" if grown._ivf is not None else "-"
            print(f"  {grown.n:9,}  {grown.kind:5s} {lists:>6s} {grown.nprobe or '-':>6} {took:7.2f}", flush=True)
    print(f"  adding all {a.n:,}: {total:.1f} s")
    t = time.perf_counter()
    fresh = siann.Index(x, "l2", recall=0.95, calibration_queries=cal)
    print(f"  built from scratch on the same vectors: {time.perf_counter() - t:.1f} s, {fresh.kind}, "
          f"{fresh._ivf.nlist if fresh._ivf else '-'} lists, nprobe {fresh.nprobe}")
    truth = siann.Index(x, "l2").search(test, 10, device="cpu").indices
    print(f"\n  {'':22s} {'recall@10':>9s}" + "".join(f"{f'batch {b:,}':>13s}" for b in BATCHES))
    for label, index in (("grown from empty", grown), ("built from scratch", fresh)):
        found = index.search(test, 10).indices
        print(f"  {label:22s} {agreement(x, test, found, truth, 'l2'):9.3f}"
              + "".join(f"{rate(index, test, b):13,.0f}" for b in BATCHES), flush=True)
    print(f"\n{grown.plan}")


if __name__ == "__main__":
    main()
