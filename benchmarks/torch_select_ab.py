"""The PyTorch GPU IVF (ivf_torch.py) with ivf_gpu.py's probe selection, threads per query and merge,
against its earlier settings, with MLX's IVF on the same lists alongside: PyTorch with the ordered
two-stage selection, 64 threads per query and the separate merge ("earlier"); PyTorch with the defaults
(unordered selection, 128 threads per query, the merge in the scan kernel; the list-major scan keeps
the ordered selection up to nprobe 16); MLX with the same defaults. Float32 lists (the PyTorch IVF has
no others), the centroids of ivf_gpu.py's k-means for both, so the lists are the same; the results are
compared index for index (they can differ only where two centroids tie at the nprobe-th place).
The MLX part runs first in a process of its own and saves the centroids and its results for the PyTorch
part.

Queries per second: single queries (one call each), batches of 100 (query-major scan) and one batch
of 10,000 (list-major scan), median of three after a warm-up (single queries: one pass).

    uv run --group bench python torch_select_ab.py --dataset sift
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
from siann._cpu import FlatIndex

from backends import agreement
from datasets import load
from timing import throughput

K = 10


def measure(label, nprobe, qm, lm, small, xq, xb, truth):
    q1, _ = throughput(qm, small, 1, repeats=1)
    q100, f100 = throughput(qm, small, 100)
    qf, ff = throughput(lm, xq, len(xq))
    r = agreement(xb, xq, ff, truth, "l2")
    return f"  {f'{nprobe}, {label}':40s} {r:9.3f} {q1:9,.0f} {q100:9,.0f} {qf:9,.0f}", f100, ff


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sift", choices=["sift", "glove"])
    p.add_argument("--nlist", type=int, default=4000)
    p.add_argument("--nprobes", default="8,16,32,64,128")
    p.add_argument("--queries", type=int, default=10_000)
    p.add_argument("--small", type=int, default=1000)
    p.add_argument("--part", choices=["mlx", "torch"], help=argparse.SUPPRESS)
    p.add_argument("--work", help=argparse.SUPPRESS)
    a = p.parse_args()
    nprobes = list(map(int, a.nprobes.split(",")))
    if a.part is None:
        with tempfile.TemporaryDirectory() as work:
            for part in ("mlx", "torch"):
                subprocess.run([sys.executable, __file__, *sys.argv[1:], "--part", part, "--work", work], check=True)
            rows = {part: np.load(os.path.join(work, f"{part}.npz")) for part in ("mlx", "torch")}
            print(f"{a.dataset}, l2: {a.nlist} lists (float32); queries per second")
            print(f"\n  {'nprobe, backend and settings':40s} {'recall@10':>9s} {'single':>9s} {'100':>9s} {'10,000':>9s}")
            for i, nprobe in enumerate(nprobes):
                print(str(rows["torch"]["rows"][2 * i]))
                print(str(rows["torch"]["rows"][2 * i + 1]))
                print(str(rows["mlx"]["rows"][i]))
                same = [np.mean(rows["torch"][f"{w}{nprobe}"] == rows["mlx"][f"{w}{nprobe}"]) for w in ("qm", "lm")]
                print(f"  {'':40s} identical neighbors, PyTorch and MLX defaults: {same[0]:.4f} (query-major), "
                      f"{same[1]:.4f} (list-major)", flush=True)
        return
    xb, xq = load(a.dataset, a.queries)
    truth = np.array(FlatIndex(xb, xb.shape[1], "l2").search(xq, K), dtype=np.int64).reshape(-1, K)
    small = xq[:a.small]
    out, rows = {}, []
    if a.part == "mlx":
        from siann.ivf_gpu import GpuIVF
        mlx = GpuIVF(xb, a.nlist, "l2", "float32")
        out["centroids"] = np.array(mlx.centroids)
        for nprobe in nprobes:
            row, f100, ff = measure("MLX, defaults", nprobe, lambda q, n=nprobe: mlx.search(q, K, n),
                                    lambda q, n=nprobe: mlx.search_list_major(q, K, n), small, xq, xb, truth)
            rows.append(row)
            out[f"qm{nprobe}"], out[f"lm{nprobe}"] = f100, ff
    else:
        from siann.ivf_torch import TorchIVF
        ivf = TorchIVF(xb, np.load(os.path.join(a.work, "mlx.npz"))["centroids"], "l2")
        for nprobe in nprobes:
            row, _, _ = measure("PyTorch, earlier", nprobe,
                                lambda q, n=nprobe: ivf.search(q, K, n, threads=64, select="select_k", merge="separate"),
                                lambda q, n=nprobe: ivf.search_list_major(q, K, n, select="select_k"),
                                small, xq, xb, truth)
            rows.append(row)
            row, f100, ff = measure("PyTorch, defaults", nprobe, lambda q, n=nprobe: ivf.search(q, K, n),
                                    lambda q, n=nprobe: ivf.search_list_major(q, K, n), small, xq, xb, truth)
            rows.append(row)
            out[f"qm{nprobe}"], out[f"lm{nprobe}"] = f100, ff
    np.savez(os.path.join(a.work, f"{a.part}.npz"), rows=np.array(rows), **out)


if __name__ == "__main__":
    main()
