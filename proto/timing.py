"""Throughput measurement shared by the IVF comparisons (ivf_cpu.py, ivf_torch.py)."""
import time

import numpy as np


def throughput(search, xq: np.ndarray, batch: int, repeats: int = 3) -> tuple[float, np.ndarray]:
    """Queries per second (median of `repeats`, after a warm-up) and the results, searching `batch`
    queries per call."""
    def run():
        return np.concatenate([search(xq[s:s + batch]) for s in range(0, len(xq), batch)])
    found = run()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        found = run()
        times.append(time.perf_counter() - start)
    return len(xq) / sorted(times)[len(times) // 2], found
