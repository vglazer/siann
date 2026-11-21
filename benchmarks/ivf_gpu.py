"""The GPU IVF (siann.ivf_gpu) against exact search: --check runs both scans, probing every list,
against NumPy on small random data, for every metric and list storage (run by check_kernels.py in a
process of its own, since FAISS, whose k-means GpuIVF uses, and PyTorch cannot share one); without it,
recall and queries per second at a range of nprobe.

    uv run --group bench python ivf_gpu.py [--metric l1|l2|linf] [--batch 1000]
    uv run --group bench python ivf_gpu.py --check
"""
import argparse
import time

import mlx.core as mx
import numpy as np
from siann.exact import knn as nearest
from siann.ivf_gpu import METRICS, GpuIVF

from datasets import load


def check() -> bool:
    """Both scans, probing every list, against exact search (NumPy) on small random data, for every
    metric and both list storages; with fewer probes, the two scans must agree. Run by check_kernels.py in its own process,
    since FAISS (the k-means here) and PyTorch cannot share one."""
    ok = True
    rng = np.random.default_rng(2)
    for n, d, nq, k, nlist in ((5000, 32, 9, 10, 37), (3001, 37, 70, 10, 16), (4000, 100, 20, 1, 50)):
        x = rng.standard_normal((n, d), dtype=np.float32)
        q = rng.standard_normal((nq, d), dtype=np.float32)
        res = {}
        # float16 lists hold the rounded vectors, exactly; probing every list must then give exact
        # search on them. uint8 lists need integer data: the same draws, scaled to 0-255 and rounded.
        xi = np.rint((x - x.min()) / (x.max() - x.min()) * 255).astype(np.float32)
        qi = np.rint((q - x.min()) / (x.max() - x.min()) * 255).astype(np.float32)
        for lists in ("float32", "float16", "uint8"):
            xs, qs = (xi, qi) if lists == "uint8" else (x, q)
            xr = xs.astype(lists).astype(np.float32)
            diff = np.abs(qs[:, None, :] - xr[None, :, :])
            exact = {"l1": diff.sum(axis=2), "l2": (diff ** 2).sum(axis=2), "linf": diff.max(axis=2),
                     "ip": -(qs.astype(np.float64) @ xr.T.astype(np.float64))}
            tag = {"float32": "", "float16": " f16", "uint8": " u8"}[lists]
            for m in METRICS:
                want = np.argsort(exact[m], axis=1, kind="stable")[:, :k]
                if lists == "float32":
                    def same(a, b):
                        return np.array_equal(a, b)
                else:
                    # rounding (and uint8's integers) puts coordinates on a coarse grid, and an
                    # L-infinity distance is one coordinate difference, so exact ties occur: compare
                    # the neighbors' exact distances, in order, which lets tied neighbors swap
                    def same(a, b, dm=exact[m]):
                        return np.array_equal(np.take_along_axis(dm, a, 1), np.take_along_axis(dm, b, 1))
                index = GpuIVF(xs, nlist, m, lists)
                res[f"query-major {m}{tag}"] = same(index.search(qs, k, nlist), want)
                res[f"list-major {m}{tag}"] = same(index.search_list_major(qs, k, nlist), want)
                res[f"scans agree {m}{tag}"] = same(index.search(qs, k, 3), index.search_list_major(qs, k, 3))
        # one probed list of a few vectors holds fewer than k: the missing neighbors are -1, after the
        # found ones, which are distinct, and both scans agree
        for m in METRICS:
            index = GpuIVF(x[:200], 100, m, "float32")
            qm, lm = index.search(q, 10, 1), index.search_list_major(q, 10, 1)   # k = 10, more than a list holds
            found_first = all(np.all(np.diff((r < 0).astype(int)) >= 0) for r in qm)
            distinct = all(len(set(r[r >= 0])) == int((r >= 0).sum()) for r in qm)
            res[f"fewer than k: -1 {m}"] = bool((qm < 0).any() and found_first and distinct and np.array_equal(qm, lm))
        ok &= all(res.values())
        print(f"mlx ivf n={n} d={d} q={nq} k={k}: " + ", ".join(f"{name} {r}" for name, r in res.items()))
    return ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sift")
    p.add_argument("--metric", default="l1")
    p.add_argument("--batch", type=int, default=1000)
    p.add_argument("--nlist", type=int, default=4000)
    p.add_argument("--threads", type=int, default=64)
    p.add_argument("--check", action="store_true", help="check both scans against exact search, then exit")
    a = p.parse_args()
    if a.check:
        raise SystemExit(0 if check() else 1)
    k = 10
    xb, xq = load(a.dataset, a.batch)
    xb_mx = mx.array(xb)
    norms = mx.sum(xb_mx * xb_mx, axis=1)
    truth = np.concatenate([np.array(nearest(mx.array(xq[i:i + 256]), xb_mx, norms, k, a.metric)) for i in range(0, len(xq), 256)])
    start = time.perf_counter()
    index = GpuIVF(xb, a.nlist, a.metric)
    print(f"{a.dataset}: GPU IVF ({a.nlist} lists, {a.metric}) on {len(xb):,} vectors: built in {time.perf_counter() - start:.1f} s; "
          f"batches of {a.batch} queries, {a.threads} threads per query", flush=True)
    for nprobe in (4, 16, 32, 64, 128):
        row = []
        for name, fn in (("query-major", lambda: index.search(xq, k, nprobe, a.threads)),
                         ("list-major", lambda: index.search_list_major(xq, k, nprobe))):
            found = fn()   # warm up, including compiling for this batch size
            times = []
            for _ in range(3):
                start = time.perf_counter()
                found = fn()
                times.append(time.perf_counter() - start)
            recall = np.mean([len(set(r) & set(g)) / k for r, g in zip(found, truth)])
            row.append(f"{name} recall@10 {recall:.3f}, {len(xq) / sorted(times)[1]:,.0f} queries/s")
        print(f"  nprobe {nprobe:3d}: " + "; ".join(row), flush=True)


if __name__ == "__main__":
    main()
