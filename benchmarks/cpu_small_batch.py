"""siann's Rust exact kernel (FlatIndex) on small batches: queries per second at batch sizes from 1 to 128
on SIFT1M, by L1, L-infinity and L2 (median of 3 calls after a warm-up), and its outputs, saved with
--save and checked bit for bit with --compare, so that a change to how the kernel divides work among
threads can be checked to return identical neighbors. --ab times the two ways of dividing it, in the same
process and alternating (7 rounds, the median of each): one thread per 32 queries (`split=False`), and
small batches also divided among threads by ranges of the base vectors (`split=True`, the default).

    uv run --group bench python cpu_small_batch.py --save before.npz
    uv run --group bench python cpu_small_batch.py --compare before.npz
    uv run --group bench python cpu_small_batch.py --ab
"""
import argparse
import os
import time

import numpy as np
from datasets import load

from siann._cpu import FlatIndex

BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
K = 10


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sift")
    p.add_argument("--metrics", default="l1,linf,l2")
    p.add_argument("--save", help="write every output here (.npz)")
    p.add_argument("--compare", help="check every output against this file (.npz)")
    p.add_argument("--ab", action="store_true", help="split=False against split=True, alternating")
    a = p.parse_args()
    xb, xq = load(a.dataset, max(BATCHES))
    if a.ab:
        ab(xb, xq, a.metrics.split(","))
        return
    print(f"{a.dataset}: {len(xb):,} x {xb.shape[1]}, {os.cpu_count()} CPU threads; queries per second")
    print(f"  {'metric':6s}" + "".join(f"{b:>9,}" for b in BATCHES))
    outputs = {}
    for m in a.metrics.split(","):
        index = FlatIndex(xb, xb.shape[1], m)
        rates = []
        for b in BATCHES:
            q = np.ascontiguousarray(xq[:b])
            outputs[f"{m}_{b}"] = np.asarray(index.search(q, K), dtype=np.int64)
            times = []
            for _ in range(3):
                t = time.perf_counter()
                index.search(q, K)
                times.append(time.perf_counter() - t)
            rates.append(b / sorted(times)[1])
        print(f"  {m:6s}" + "".join(f"{r:9,.0f}" for r in rates), flush=True)
    if a.save:
        np.savez(a.save, **outputs)
    if a.compare:
        ref = np.load(a.compare)
        bad = [key for key in outputs if not np.array_equal(outputs[key], ref[key])]
        print(f"\noutputs identical to {os.path.basename(a.compare)}: "
              f"{'all ' + str(len(outputs)) if not bad else 'NO, differ: ' + ', '.join(bad)}")


def ab(xb: np.ndarray, xq: np.ndarray, metrics: list[str]) -> None:
    print(f"{len(xb):,} x {xb.shape[1]}, {os.cpu_count()} CPU threads; queries per second, split=False / "
          "split=True (ratio), median of 7 alternating rounds")
    for m in metrics:
        index = FlatIndex(xb, xb.shape[1], m)
        cells = []
        for b in BATCHES:
            q = np.ascontiguousarray(xq[:b])
            for split in (False, True):
                index.search(q, K, split=split)
            times = {False: [], True: []}
            for _ in range(7):
                for split in (False, True):
                    t = time.perf_counter()
                    index.search(q, K, split=split)
                    times[split].append(time.perf_counter() - t)
            off, on = (b / sorted(times[s])[3] for s in (False, True))
            cells.append(f"{b:>5,}: {off:6,.0f} / {on:6,.0f} ({on / off:.2f})")
        print(f"  {m}\n    " + "\n    ".join(cells), flush=True)


if __name__ == "__main__":
    main()
