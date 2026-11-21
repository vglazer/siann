"""What a new batch size costs on the GPU: the first call at each batch size (which compiles the kernels
for that shape, where a size is a compile-time constant) against the same call repeated, for exact search
(siann.Index on the GPU, every metric) and the GPU IVF's two scans (MLX and PyTorch, L2, nprobe 16).
Random vectors, 100,000 x 128, k = 10; each batch size is new to the process when first called.

    uv run --group bench python compile_cost.py
"""
import time

import numpy as np

import siann
from siann.ivf_build import build_lists
from siann.ivf_gpu import GpuIVF

BATCHES = (7, 8, 9, 50, 51, 300, 301, 1000, 1001)


def first_and_again(fn, xq):
    rows = []
    for b in BATCHES:
        q = xq[:b]
        t = time.perf_counter()
        fn(q)
        first = time.perf_counter() - t
        t = time.perf_counter()
        fn(q)
        again = time.perf_counter() - t
        rows.append((b, first * 1e3, again * 1e3))
    return rows


def show(name, rows):
    extra = [f - a for _, f, a in rows]
    print(f"  {name:28s} first call minus repeat: median {np.median(extra):6.1f} ms, max {max(extra):6.1f} ms;"
          f" repeat {np.median([a for _, _, a in rows]):6.2f} ms (median)", flush=True)


def main() -> None:
    rng = np.random.default_rng(0)
    xb = rng.standard_normal((100_000, 128), dtype=np.float32)
    xq = rng.standard_normal((1001, 128), dtype=np.float32)
    print(f"batch sizes {', '.join(map(str, BATCHES))}, each new to the process when first called")
    for metric in siann.METRICS:
        index = siann.Index(xb, metric, device="gpu")
        index.search(xq[:2], 10)                              # builds the GPU copy
        show(f"exact, {metric}", first_and_again(lambda q: index.search(q, 10), xq))
    built = build_lists(xb, 400, "l2")
    ivf = GpuIVF(xb, 400, "l2", built=built)
    show("MLX IVF, query-major", first_and_again(lambda q: ivf.search(q, 10, 16), xq))
    show("MLX IVF, list-major", first_and_again(lambda q: ivf.search_list_major(q, 10, 16), xq))
    try:
        from siann.ivf_torch import TorchIVF
    except ImportError:
        return
    tivf = TorchIVF(xb, built.centroids, "l2", built=built)
    show("PyTorch IVF, query-major", first_and_again(lambda q: tivf.search(q, 10, 16), xq))
    show("PyTorch IVF, list-major", first_and_again(lambda q: tivf.search_list_major(q, 10, 16), xq))


if __name__ == "__main__":
    main()
