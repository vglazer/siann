"""What to do when NumPy was imported before siann and its OpenBLAS threads spin (siann._build): one process per
mode, as import order is fixed once a process has started. Modes: "spin" (NumPy first, as is: the threads spin
after each BLAS call), "one-thread" (NumPy first, OpenBLAS set to one thread at run time, so no thread is left
to spin), "no-blas" (NumPy first, spinning, siann's routes without their BLAS paths), "clean" (siann first: the
threads sleep, BLAS keeps its threads). Each: siann.Index.search at small batches on its routed paths, and exact
search on the CPU (BLAS for inner product), in queries per second, on distinct queries.

    python openblas_modes.py --mode spin
"""
import argparse
import sys

if "--mode" in sys.argv and sys.argv[sys.argv.index("--mode") + 1] == "clean":
    import siann  # first: before NumPy, so that OpenBLAS's threads sleep

import glob
import os
import time

import numpy as np
from datasets import load

import siann


def one_blas_thread() -> str:
    """OpenBLAS set to one thread, through the function NumPy's bundled OpenBLAS exports."""
    import ctypes
    libs = glob.glob(os.path.join(os.path.dirname(np.__file__), "..", "numpy.libs", "*openblas*.so*"))
    lib = ctypes.CDLL(libs[0])
    for name in ("scipy_openblas_set_num_threads64_", "openblas_set_num_threads64_", "openblas_set_num_threads"):
        if hasattr(lib, name):
            getattr(lib, name)(ctypes.c_int(1))
            return f"{os.path.basename(libs[0])}: {name}(1)"
    return "no OpenBLAS function found"


def rate(call, q, b):
    call(q[:b])
    n, t = 0, time.perf_counter()
    for s in range(0, len(q), b):
        call(q[s:s + b])
        n += min(b, len(q) - s)
    return n / (time.perf_counter() - t)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=("spin", "one-thread", "no-blas", "clean"))
    p.add_argument("--dataset", default="music100")
    a = p.parse_args()
    note = one_blas_thread() if a.mode == "one-thread" else ""
    xb, xq = load(a.dataset, 10_000)
    metric = {"music100": "ip", "sift": "l2"}[a.dataset]
    from siann import _build
    index = siann.Index(xb, metric, recall=0.95, calibration_queries=xq[:1000])
    if a.mode == "no-blas" and index._routes.get("table"):
        index._routes["table"] = [[b, [p for p in paths if p != "cpu-query-blas"] or paths]
                                  for b, paths in index._routes["table"]]
    test = np.ascontiguousarray(xq[-2000:], dtype=np.float32)
    row = {b: round(rate(lambda qq: index.search(qq, 10), test, b)) for b in (1, 10, 100)}
    exact = siann.Index(xb, metric)
    ex = round(rate(lambda qq: exact.search(qq, 10, device="cpu"), test[:1000], 1000))
    print(f"{a.mode:>10}: OpenBLAS spins {_build.openblas_spins()!s:5}; Index.search q/s at 1, 10, 100: "
          f"{row[1]:,} / {row[10]:,} / {row[100]:,}; paths {[index._ivf_path(b, None) for b in (1, 10, 100)]}; "
          f"exact on the CPU, batch of 1,000: {ex:,} q/s {note}", flush=True)


if __name__ == "__main__":
    main()
