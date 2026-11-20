"""The GPU IVF's probe selection, before and after: siann's ordered two-stage selection with 64 threads
per query in the query-major scan ("select_k", the earlier default) against MLX's unordered
argpartition with 128 threads per query (the default), at every batch size, so that the new default
is checked to be no slower anywhere. With --merge, the query-major scan's merge done separately (MLX
operations over every thread's candidates) against inside the kernel (a tree merge in threadgroup
memory, the default). (Reading four coordinates per load, `results/gpu_load4_ab.txt`, was measured
with an earlier version of this script and dropped: up to 6% faster on SIFT, 20-45% slower on GloVe.) Same lists, same centroids; the results are compared index for
index (they can differ only where two centroids tie at the nprobe-th place).

Queries per second: single queries (one call each), batches of 100 (query-major scan) and one batch
of 10,000 (list-major scan), median of three after a warm-up (single queries: one pass).

    uv run --group bench python gpu_select_ab.py --dataset sift --lists uint8             # the crossover
    uv run --group bench python gpu_select_ab.py --dataset sift --lists uint8 --defaults  # old against new
"""
import argparse

import numpy as np
from siann._cpu import FlatIndex

from backends import agreement
from datasets import load
from siann.ivf_gpu import GpuIVF
from timing import throughput

K = 10


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sift", choices=["sift", "glove"])
    p.add_argument("--lists", default="float32", choices=["float32", "float16", "uint8"])
    p.add_argument("--nlist", type=int, default=4000)
    p.add_argument("--nprobes", default="8,16,32,64,128")
    p.add_argument("--queries", type=int, default=10_000)
    p.add_argument("--small", type=int, default=1000)
    p.add_argument("--defaults", action="store_true", help="only the earlier default against the current one")
    p.add_argument("--merge", action="store_true", help="the separate merge against the merge in the kernel")
    a = p.parse_args()
    xb, xq = load(a.dataset, a.queries)
    d = xb.shape[1]
    truth = np.array(FlatIndex(xb, d, "l2").search(xq, K), dtype=np.int64).reshape(-1, K)
    ivf = GpuIVF(xb, a.nlist, "l2", a.lists)
    small = xq[:a.small]
    print(f"{a.dataset}, l2: {len(xb):,} vectors, {a.nlist} lists ({a.lists}); queries per second")
    print(f"\n  {'nprobe, selection':40s} {'recall@10':>9s} {'single':>9s} {'100':>9s} {'10,000':>9s}")
    for nprobe in map(int, a.nprobes.split(",")):
        found = {}
        configs = (("select_k, 64 threads (earlier)", "select_k", 64),
                   ("select_k, 128 threads", "select_k", 128),
                   ("argpartition, 128 threads", "argpartition", 128))
        configs = tuple((label, sel, thr, "separate", False) for label, sel, thr in configs)
        if a.defaults:
            configs = (configs[0], ("default (auto selection, 128 threads)", "auto", 128, "separate", False))
        if a.merge:
            configs = (("separate merge (auto, 128 threads)", "auto", 128, "separate", False),
                       ("merge in the kernel (auto, 128 threads)", "auto", 128, "kernel", False))
        for label, sel, thr, mg, l4 in configs:
            def qm(q, n=nprobe, s=sel, t=thr, m=mg):
                return ivf.search(q, K, n, threads=t, select=s, merge=m)

            def lm(q, n=nprobe, s=sel):
                return ivf.search_list_major(q, K, n, select=s)
            q1, _ = throughput(qm, small, 1, repeats=1)
            q100, f100 = throughput(qm, small, 100)
            qf, ff = throughput(lm, xq, len(xq))
            found[(sel, thr, mg, l4)] = (f100, ff)
            r = agreement(xb, xq, ff, truth, "l2")
            print(f"  {f'{nprobe}, {label}':40s} {r:9.3f} {q1:9,.0f} {q100:9,.0f} {qf:9,.0f}", flush=True)
        if a.merge:
            same = np.mean(found[("auto", 128, "separate", False)][0] == found[("auto", 128, "kernel", False)][0])
            print(f"  {'':40s} identical neighbors, the two merges (batches of 100): {same:.4f}")
        elif not a.defaults:
            same = [np.mean(found[("select_k", 128, "separate", False)][i]
                            == found[("argpartition", 128, "separate", False)][i])
                    for i in (0, 1)]
            print(f"  {'':40s} identical neighbors, the two selections at 128 threads: {same[0]:.4f} "
                  f"(query-major), {same[1]:.4f} (list-major)")


if __name__ == "__main__":
    main()
