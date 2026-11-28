"""The time siann's CAGRA optimization (siann._cpu.optimize_graph: detour counting, pruning, reverse links)
takes on SIFT1M's rough graph (each vector's 64 nearest by siann's IVF, saved on the first run), the best of
three, and a fingerprint of the links it returns, so that two builds of the Rust code can be compared: the same
fingerprint is the same graph.

    uv run --group bench python graph_optimize_time.py
"""
import hashlib
import os
import time

from siann._cpu import optimize_graph
from siann._graph_build import knn_graph

import numpy as np
from datasets import load

KNN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sift_knn64.npy")


def main() -> None:
    if not os.path.exists(KNN):
        xb, _ = load("sift", 10)
        np.save(KNN, knn_graph(xb, "l2")[0])
    knn = np.ascontiguousarray(np.load(KNN), dtype=np.int64)
    optimize_graph(knn[:10_000] % 10_000, 32)                      # warm-up
    times = []
    for _ in range(3):
        t = time.perf_counter()
        links = optimize_graph(knn, 32)
        times.append(time.perf_counter() - t)
    print(f"optimize_graph on SIFT1M's rough graph ({knn.shape[0]:,} x {knn.shape[1]}): {min(times):.2f} s, the best "
          f"of 3; links {hashlib.sha256(links.tobytes()).hexdigest()[:16]}", flush=True)


if __name__ == "__main__":
    main()
