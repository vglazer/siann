"""Driver for profiling siann's Rust IVF in batches of 100 with a sampling profiler (samply): builds
the SIFT index (4,000 lists, uint8 lists, two-level coarse step) and searches batches of 100 for
--seconds.

    samply record --save-only --unstable-presymbolicate -o profile.json.gz -- \\
        .venv/bin/python profile_batch.py
"""
import argparse
import time

import faiss
import numpy as np
from siann._cpu import IvfIndex

from datasets import load
from siann.ivf_gpu import GpuIVF

p = argparse.ArgumentParser()
p.add_argument("--nlist", type=int, default=4000)
p.add_argument("--nprobe", type=int, default=64)
p.add_argument("--seconds", type=float, default=10)
p.add_argument("--threads", type=int, default=12)
a = p.parse_args()
xb, xq = load("sift", 1000)
d = xb.shape[1]
cents = np.ascontiguousarray(np.array(GpuIVF(xb, a.nlist, "l2").centroids), dtype=np.float32)
index = IvfIndex(xb, cents, d, "l2", xb.astype(np.uint8), "uint8")
ng = a.nlist // 64
km = faiss.Kmeans(d, ng, niter=20, seed=1)
km.train(cents)
index.set_coarse_groups(np.ascontiguousarray(km.centroids, dtype=np.float32),
                        km.index.search(cents, 1)[1][:, 0].astype(np.int64))
print("searching", flush=True)
t0, n = time.perf_counter(), 0
while time.perf_counter() - t0 < a.seconds:
    for s in range(0, len(xq), 100):
        index.search(xq[s:s + 100], 10, a.nprobe, gprobe=ng // 4, threads=a.threads)
        n += 100
print(f"{n / (time.perf_counter() - t0):,.0f} queries per second")
