"""What changing a siann.Index costs: an IVF built on 900,000 of SIFT1M's vectors (recall target 0.95,
calibrated on 1,000 of the dataset's queries), then vectors added and removed. Times adding (one call of
many, and many calls of one), searching with 1,000 to 10,000 added vectors in the buffer (held there) and
after they are merged, the merge itself (with its recalibration), searching with 64 and DEAD_MAX removed
vectors masked, and the compaction; then a buffer's exact search on the CPU and on the GPU.

    uv run --group bench python add_remove.py
"""
import argparse
import time

import numpy as np
from datasets import load

import siann
from siann import _index

BATCHES = (1, 100, 1000)


def rate(index, xq, b: int) -> float:
    """Queries per second searching b queries per call (median of three passes over up to 1,000 queries)."""
    q = xq[:max(b, min(len(xq), 100 * b))]
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
    a = p.parse_args()
    xb, xq = load(a.dataset, 2000)
    base, extra = xb[:900_000], xb[900_000:]
    t = time.perf_counter()
    index = siann.Index(base, "l2", recall=0.95, calibration_queries=xq[1000:])
    print(f"{a.dataset}: an IVF on {len(base):,} vectors built in {time.perf_counter() - t:.1f} s, nprobe "
          f"{index.nprobe}; the buffer merges at {index._buffer_limit():,}, removed rows compact past "
          f"{_index.DEAD_MAX}")
    q = xq[:1000]
    head = f"\n  {'queries per second, batch:':32s}" + "".join(f"{b:>10,}" for b in BATCHES)
    print(head)

    def row(label):
        print(f"  {label:32s}" + "".join(f"{rate(index, q, b):10,.0f}" for b in BATCHES), flush=True)
    row("as built")
    limit = _index.BUFFER_MIN
    _index.BUFFER_MIN = 10 ** 9                                     # hold the buffer to time searches with it
    t = time.perf_counter()
    for i in range(100):
        index.add(extra[i])
    one = (time.perf_counter() - t) / 100
    added = 100
    for size in (1_000, 2_500, 5_000, 10_000):
        t = time.perf_counter()
        index.add(extra[added:size])
        many = (time.perf_counter() - t, size - added)
        added = size
        row(f"buffer of {size:,}")
    _index.BUFFER_MIN = limit
    t = time.perf_counter()
    index._merge("timed")
    merge = time.perf_counter() - t
    row("merged")
    done = 0
    for masked in (64, _index.DEAD_MAX):
        index.remove(np.arange(done, masked))                       # ids done..masked-1: masked so far in all
        done = masked
        row(f"{masked} removed, masked")
    t = time.perf_counter()
    index.remove([_index.DEAD_MAX])                                 # one more: the compaction
    compact = time.perf_counter() - t
    assert index._dead is None
    row("compacted")

    print(f"\n  adding one vector per call: {one * 1e6:,.0f} microseconds per call; {many[1]:,} in one call: "
          f"{many[1] / many[0]:,.0f} vectors/s")
    print(f"  merging {added:,} buffered vectors (lists rebuilt, nprobe recalibrated): {merge:.2f} s; "
          f"compacting {_index.DEAD_MAX + 1} removed: {compact:.2f} s")
    print(f"\n{index.plan}")

    from siann._index import _exact_cpu, _exact_gpu
    print("\n  searching a buffer exactly, microseconds per call, CPU / GPU, by batch size:")
    for nb in (1_000, 2_500, 10_000):
        bx, cells = np.ascontiguousarray(extra[:nb]), []
        lc, lg = {}, {}
        for b in BATCHES[:1] + (10,) + BATCHES[1:]:
            qq, ts = np.ascontiguousarray(xq[:b]), []
            for f, layouts in ((_exact_cpu, lc), (_exact_gpu, lg)):
                for _ in range(5):
                    f(bx, layouts, qq, 10, "l2")
                t = time.perf_counter()
                for _ in range(30):
                    f(bx, layouts, qq, 10, "l2")
                ts.append((time.perf_counter() - t) / 30 * 1e6)
            cells.append(f"{b:,}: {ts[0]:,.0f} / {ts[1]:,.0f}")
        print(f"    {nb:>6,} vectors   " + "   ".join(cells), flush=True)


if __name__ == "__main__":
    main()
