"""Where exact L2 search spends its time, per tile of queries, in MLX and in PyTorch MPS: the matrix
multiply alone, the distances as `norms - 2 * (q @ x.T)`, the same through a fused `addmm`, and the
custom top-k kernel. Each framework runs in its own process. RESULTS.md, "Exact search on the
GPU".

    uv run --no-sync --group proto python profile_l2.py [--n 1000000] [--tile 256]
"""
import argparse
import subprocess
import sys
import time

import numpy as np


def mlx_steps(n: int, d: int, tile: int, repeats: int) -> dict[str, float]:
    import mlx.core as mx

    from knn_kernels import select_k
    rng = np.random.default_rng(0)
    xb = mx.array(rng.standard_normal((n, d), dtype=np.float32))
    q = mx.array(rng.standard_normal((tile, d), dtype=np.float32))
    norms = mx.sum(xb * xb, axis=1)
    mx.eval(xb, q, norms)
    dist = mx.addmm(norms[None, :], q, xb.T, alpha=-2.0)
    mx.eval(dist)

    def t(f):
        mx.eval(f())
        start = time.perf_counter()
        for _ in range(repeats):
            mx.eval(f())
        return (time.perf_counter() - start) / repeats * 1e3
    return {"matrix multiply alone": t(lambda: q @ xb.T),
            "norms - 2 * (q @ x.T)": t(lambda: norms[None, :] - 2 * (q @ xb.T)),
            "fused addmm": t(lambda: mx.addmm(norms[None, :], q, xb.T, alpha=-2.0)),
            "top-k kernel (select_k)": t(lambda: select_k(dist, 10))}


def torch_steps(n: int, d: int, tile: int, repeats: int) -> dict[str, float]:
    import torch

    from torch_kernels import select_k
    rng = np.random.default_rng(0)
    xb = torch.from_numpy(rng.standard_normal((n, d), dtype=np.float32)).to("mps")
    q = torch.from_numpy(rng.standard_normal((tile, d), dtype=np.float32)).to("mps")
    norms = (xb * xb).sum(1)
    dist = torch.addmm(norms[None, :], q, xb.T, alpha=-2)

    def t(f):
        f()
        torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            f()
        torch.mps.synchronize()
        return (time.perf_counter() - start) / repeats * 1e3
    return {"matrix multiply alone": t(lambda: q @ xb.T),
            "norms - 2 * (q @ x.T)": t(lambda: norms[None, :] - 2 * (q @ xb.T)),
            "fused addmm": t(lambda: torch.addmm(norms[None, :], q, xb.T, alpha=-2)),
            "top-k kernel (select_k)": t(lambda: select_k(dist, 10))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--framework", choices=["mlx", "torch"], help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.framework:
        steps = (mlx_steps if a.framework == "mlx" else torch_steps)(a.n, a.d, a.tile, a.repeats)
        for name, ms in steps.items():
            print(f"{name}\t{ms:.2f}")
        return
    rows = {}
    for fw in ("mlx", "torch"):
        r = subprocess.run([sys.executable, __file__, "--framework", fw, f"--n={a.n}", f"--d={a.d}",
                            f"--tile={a.tile}", f"--repeats={a.repeats}"],
                           check=True, capture_output=True, text=True, encoding="utf-8")
        for line in r.stdout.strip().splitlines():
            name, ms = line.split("\t")
            rows.setdefault(name, {})[fw] = float(ms)
    print(f"{a.tile} queries x {a.n:,} vectors x {a.d} dimensions, mean of {a.repeats}, in ms")
    print(f"  {'step':26s} {'MLX':>8s} {'PyTorch':>8s}")
    for name, v in rows.items():
        print(f"  {name:26s} {v['mlx']:8.2f} {v['torch']:8.2f}")


if __name__ == "__main__":
    main()
