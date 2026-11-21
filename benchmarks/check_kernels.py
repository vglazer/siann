"""Checks siann's kernels against NumPy on small random data, for L1, L2 and L-infinity: the MLX kernels (knn_kernels.py), their
PyTorch MPS port (torch_kernels.py), the Rust CPU kernels (rust_cpu, if built) and the GPU IVFs
(ivf_torch.py, and ivf_gpu.py in a process of its own). The cases cover
dimensions that are not multiples of the kernels' tiles (7, 37, 100) and k from 1 to 32. Exits with
status 1 if any result differs.

    uv run --group bench python check_kernels.py
"""
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

import siann.torch_kernels as tk
from siann.ivf_torch import TorchIVF
from siann.knn_kernels import dist, ip_knn, knn_tiled, l2_knn, select_k

from kernel_variants import knn_fused

try:
    from siann._cpu import FlatIndex, IvfIndex, widen_f16
except ImportError:
    FlatIndex = IvfIndex = widen_f16 = None
    print("siann_cpu not built (crates/siann-py): skipping the Rust kernels")

ok = True
if widen_f16 is not None:
    # Every finite float16 (exponent below 31), including subnormals and both zeros, widened by the
    # Rust lists' conversion, against NumPy's; compared as bits, so that -0 and +0 differ.
    bits = np.arange(1 << 16, dtype=np.uint16)
    bits = bits[(bits >> 10) & 0x1F != 0x1F]
    got = np.array(widen_f16(bits), dtype=np.float32).view(np.uint32)
    want = bits.view(np.float16).astype(np.float32).view(np.uint32)
    same = bool(np.array_equal(got, want))
    ok &= same
    print(f"rust float16 widening, all {len(bits):,} finite values: {same}", flush=True)

rng = np.random.default_rng(1)
for n, d, nq, k in ((5000, 32, 9, 10), (20011, 128, 33, 1), (3000, 7, 5, 32), (3001, 37, 70, 10),
                    (4000, 100, 20, 10)):
    x = rng.standard_normal((n, d), dtype=np.float32)
    q = rng.standard_normal((nq, d), dtype=np.float32)
    diff = np.abs(q[:, None, :] - x[None, :, :])
    exact = {"l1": diff.sum(axis=2), "l2": (diff ** 2).sum(axis=2), "linf": diff.max(axis=2),
             "ip": -(q.astype(np.float64) @ x.T.astype(np.float64))}
    want = {m: np.argsort(v, axis=1, kind="stable")[:, :k] for m, v in exact.items()}
    results_extra = {}
    xm, qm = mx.array(x), mx.array(q)
    xt, qt = torch.from_numpy(x).to("mps"), torch.from_numpy(q).to("mps")
    found = {m: {} for m in exact}
    found["l1"]["mlx select_k"] = np.array(select_k(mx.array(exact["l1"]), k))
    for m in ("l1", "linf"):
        found[m][f"mlx {m} fused"] = np.array(knn_fused(qm, xm, k, m))
        found[m][f"mlx {m} tiled"] = np.array(knn_tiled(qm, xm, k, m))
        found[m][f"torch {m} tiled"] = tk.knn_tiled(qt, xt, k, m).cpu().numpy()
    found["l2"]["mlx l2"] = np.array(l2_knn(qm, xm, mx.sum(xm * xm, axis=1), k))
    found["l2"]["torch l2"] = tk.l2_knn(qt, xt, (xt * xt).sum(1), k).cpu().numpy()
    found["ip"]["mlx ip"] = np.array(ip_knn(qm, xm, k))
    if FlatIndex is not None:
        for m in exact:
            found[m][f"rust {m}"] = np.array(FlatIndex(x, d, m).search(q, k)).reshape(nq, k)
        # The BLAS path (blas_l2.py): NumPy's products, Rust's top-k, over blocks of the base. It ranks
        # by |x|^2 - 2 q.x in float32, whose rounding differs from NumPy's (q - x)^2 sums, so it is
        # checked by the neighbors' exact distances rather than their order.
        from siann.blas_l2 import siann_l2
        norms = np.einsum("ij,ij->i", x, x).astype(np.float32)
        got = siann_l2(x, norms, q, k, qtile=16, block=777)
        results_extra["blas l2 (distances)"] = bool(np.allclose(
            np.sort(np.take_along_axis(exact["l2"], got, 1), axis=1),
            np.take_along_axis(exact["l2"], want["l2"], 1), rtol=1e-4, atol=1e-4))
        # IVF probing every list scans everything, so it must give the exact result. 37 centroids
        # (some lists may be empty) from rows of x.
        cents = np.ascontiguousarray(x[rng.choice(n, 37, replace=False)])
        for m, res in found.items():
            ivf = IvfIndex(x, cents, d, m)
            assert sum(ivf.list_sizes()) == n
            res[f"rust ivf {m}"] = np.array(ivf.search(q, k, 37)).reshape(nq, k)
            res[f"rust ivf list-major {m}"] = np.array(ivf.search_list_major(q, k, 37)).reshape(nq, k)
            # With fewer probes, both scans must agree with each other.
            same = np.array_equal(ivf.search(q, k, 5), ivf.search_list_major(q, k, 5))
            results_extra[f"rust ivf scans agree {m}"] = same
            # A two-level coarse step probing every group sees every centroid, so it must return the
            # exact coarse step's results (7 groups of the 37 centroids, assigned at random).
            ivf.set_coarse_groups(np.ascontiguousarray(cents[:7]), rng.integers(0, 7, 37).astype(np.int64))
            results_extra[f"rust ivf two-level, all groups {m}"] = all(
                np.array_equal(ivf.search(q, k, p), ivf.search(q, k, p, gprobe=7)) for p in (1, 5, 37))
            # A split query (spread over threads) must return exactly what one thread does.
            results_extra[f"rust ivf split {m}"] = all(
                np.array_equal(ivf.search(q, k, p), ivf.search(q, k, p, split=s)) for p in (1, 5, 37) for s in (2, 3, 16))
            # float16 lists: probing every list must be exact search on the rounded vectors. Rounding
            # makes exact ties possible (an L-infinity distance is one coordinate difference), so
            # compare the neighbors' exact distances in order, letting tied neighbors swap.
            x16 = x.astype(np.float16)
            dr = np.abs(q[:, None, :] - x16.astype(np.float32)[None, :, :])
            dm = {"l1": dr.sum(axis=2), "l2": (dr ** 2).sum(axis=2), "linf": dr.max(axis=2),
                  "ip": -(q.astype(np.float64) @ x16.astype(np.float64).T)}[m]
            want16 = np.take_along_axis(dm, np.argsort(dm, axis=1, kind="stable")[:, :k], 1)
            ivf16 = IvfIndex(x, cents, d, m, x16.view(np.uint16), "float16")
            for scan, fn in (("", ivf16.search), (" list-major", ivf16.search_list_major)):
                got16 = np.array(fn(q, k, 37), dtype=np.int64).reshape(nq, k)
                results_extra[f"rust ivf{scan} f16 {m}"] = np.array_equal(np.take_along_axis(dm, got16, 1), want16)
            results_extra[f"rust ivf split f16 {m}"] = all(
                np.array_equal(ivf16.search(q, k, p), ivf16.search(q, k, p, split=s)) for p in (1, 5, 37) for s in (2, 3, 16))
            # uint8 lists need integer data: the same draws scaled to 0-255 and rounded, with the
            # centroids taken from it, and ties compared the same way.
            lo, hi = x.min(), x.max()
            xi = np.rint((x - lo) / (hi - lo) * 255).astype(np.float32)
            qi = np.rint((q - lo) / (hi - lo) * 255).astype(np.float32)
            di = np.abs(qi[:, None, :] - xi[None, :, :]).astype(np.float64)
            dm8 = {"l1": di.sum(axis=2), "l2": (di ** 2).sum(axis=2), "linf": di.max(axis=2),
                   "ip": -(qi.astype(np.float64) @ xi.astype(np.float64).T)}[m]
            want8 = np.take_along_axis(dm8, np.argsort(dm8, axis=1, kind="stable")[:, :k], 1)
            ci = np.ascontiguousarray(xi[rng.choice(n, 37, replace=False)])
            ivf8 = IvfIndex(xi, ci, d, m, xi.astype(np.uint8), "uint8")
            for scan, fn in (("", ivf8.search), (" list-major", ivf8.search_list_major)):
                got8 = np.array(fn(qi, k, 37), dtype=np.int64).reshape(nq, k)
                results_extra[f"rust ivf{scan} u8 {m}"] = np.array_equal(np.take_along_axis(dm8, got8, 1), want8)
            # Integer queries on byte lists take the integer kernel, whose distances equal the float
            # path's exactly: a float32 index on the same integer data must return the same indices.
            # Queries that are not integers fall back to the float path, matching it too.
            ivf32 = IvfIndex(xi, ci, d, m)
            qf = qi + 0.5
            results_extra[f"rust ivf u8 integer kernel = float {m}"] = all(
                np.array_equal(ivf8.search(qq, k, p, split=s), ivf32.search(qq, k, p, split=s))
                for qq in (qi, qf) for p in (1, 5, 37) for s in (1, 3))
            # integer data ties often: split must still match one thread index for index
            results_extra[f"rust ivf split u8 {m}"] = all(
                np.array_equal(ivf8.search(qi, k, p), ivf8.search(qi, k, p, split=s)) for p in (1, 5, 37) for s in (2, 3, 16))
            try:
                IvfIndex(x, cents, d, m, xi.astype(np.uint8), "uint8")
                results_extra[f"rust u8 rejects non-integer data {m}"] = False
            except ValueError:
                results_extra[f"rust u8 rejects non-integer data {m}"] = True
            tivf = TorchIVF(x, cents, m)
            res[f"torch ivf {m}"] = tivf.search(q, k, 37)
            res[f"torch ivf list-major {m}"] = tivf.search_list_major(q, k, 37)
    results = {name: np.array_equal(got, want[m]) for m, res in found.items() for name, got in res.items()}
    results |= results_extra
    errs = {f"{lib} {m}": float(np.abs(fn(m) - exact[m]).max())
            for m in ("l1", "linf")
            for lib, fn in (("mlx", lambda m: np.array(dist(qm, xm, m))),
                            ("torch", lambda m: tk.dist(qt, xt, m).cpu().numpy()))}
    # L-infinity takes no sums, so its distances must be exact, not merely close.
    exact_linf = errs["mlx linf"] == 0.0 and errs["torch linf"] == 0.0
    results["linf distances exact"] = exact_linf
    ok &= all(results.values())
    print(f"n={n} d={d} q={nq} k={k}: " + ", ".join(f"{name} {r}" for name, r in results.items())
          + "; tiled distance error " + ", ".join(f"{name} {e:.1e}" for name, e in errs.items()), flush=True)
# MLX's IVF (ivf_gpu.py --check), in its own process.
ok &= subprocess.run([sys.executable, str(Path(__file__).with_name("ivf_gpu.py")), "--check"]).returncode == 0
sys.exit(0 if ok else 1)
