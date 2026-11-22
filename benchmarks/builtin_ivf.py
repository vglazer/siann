"""What siann's custom GPU kernels add to the IVF, per framework: the IVF written with each framework's built-in
operations only, against the same IVF on siann's custom Metal kernels, on MLX and on PyTorch's MPS backend,
on the same lists (siann's k-means, built once).

The built-in IVF (both frameworks, the same algorithm): the coarse step as a matrix multiply against the
centroids (L2) or a broadcast difference (L1), each query's nprobe nearest lists by a partial selection;
then each query's candidates gathered from its lists, the lists' starts and sizes turned into positions by
a cumulative sum and a sorted search, padded to the most candidates any query of the chunk has, rounded up
to one of a few sizes so that buffers are reused; their
distances (a batched matrix multiply for L2, absolute differences for L1); and the k nearest by a partial
selection. Queries go in chunks whose gathered candidates take about --budget MB (4 GB by default: on SIFT
128 MB was slower than 2 GB for both frameworks, and a chunk's intermediates take a few times its budget). The custom IVF takes
siann's route by batch size: its query-major scan below 1,000 queries, its list-major scan from 1,000.

Queries per second at batches of 1 to 10,000, median of three after a warm-up (single queries and batches
of 10: one pass over 200 and 1,000 queries); recall@10 against exact search; and how many of each built-in
IVF's neighbors are as near as the custom one's on its framework (ties, common on SIFT's integer data,
broken either way count as the same).

    uv run --group bench python builtin_ivf.py --dataset sift --metric l2 --nprobes 16,64
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
from backends import agreement
from datasets import load
from timing import throughput

from siann._cpu import FlatIndex
from siann.ivf_build import Lists

K = 10
BATCHES = (1, 10, 100, 1000, 10_000)


def bucket(width: int) -> int:
    """``width`` rounded up to a size of a ladder with 8 steps per doubling (at most 12.5% more), so that a
    chunk's buffers come in few sizes and a framework's buffer cache can reuse them: with every chunk's own
    width, nearly every call allocates afresh, which on MLX costs more than the search."""
    step = 1 << max(0, width.bit_length() - 4)
    return -(-width // step) * step


def chunk_size(budget_mb: float, nprobe: int, built: Lists, d: int, metric: str) -> int:
    """Queries per chunk: their gathered candidates (about nprobe lists of the mean size, twice over for the
    longest in the chunk) in about budget_mb of float32; for L1, whose differences and their absolute values
    are as large as the gather, a third of that."""
    per_query = 2 * nprobe * (built.sizes().mean()) * d * 4 * (3 if metric == "l1" else 1)
    return int(max(1, budget_mb * 1e6 // per_query))


class MlxBuiltin:
    def __init__(self, xb: np.ndarray, built: Lists, metric: str, budget_mb: float):
        import mlx.core as mx
        self.mx, self.metric, self.built, self.budget = mx, metric, built, budget_mb
        # MLX keeps freed buffers for reuse. Capped at twice the budget, a chunk's buffers did not all fit,
        # and large batches ran slower than batches of 100; four times holds them
        mx.set_cache_limit(int(4 * budget_mb * 1e6))
        self.x = mx.array(np.ascontiguousarray(xb[built.order]))
        self.xn = mx.sum(self.x * self.x, axis=1)
        self.ids = mx.array(built.order.astype(np.int32))
        self.starts = mx.array(built.offsets[:-1].astype(np.int32))
        self.sizes = mx.array(built.sizes().astype(np.int32))
        self.c = mx.array(built.centroids)
        self.cn = mx.sum(self.c * self.c, axis=1)
        mx.eval(self.x, self.xn, self.ids, self.starts, self.sizes, self.c, self.cn)

    def _dist(self, q, x, xn=None):
        """Distances from each query (c, d) to its own rows of x (c, m, d), or to shared rows x (m, d)."""
        mx = self.mx
        if x.ndim == 2:
            if self.metric == "l2":
                return xn[None, :] - 2 * (q @ x.T)
            return mx.sum(mx.abs(q[:, None, :] - x[None, :, :]), axis=2)
        if self.metric == "l2":
            return xn - 2 * (x @ q[:, :, None])[:, :, 0]
        return mx.sum(mx.abs(x - q[:, None, :]), axis=2)

    def search(self, xq: np.ndarray, k: int, nprobe: int) -> np.ndarray:
        mx = self.mx
        out = []
        step = chunk_size(self.budget, nprobe, self.built, xq.shape[1], self.metric)
        for s in range(0, len(xq), step):
            q = mx.array(xq[s:s + step])
            nq = q.shape[0]
            probes = mx.argpartition(self._dist(q, self.c, self.cn), nprobe - 1, axis=1)[:, :nprobe]
            sz = self.sizes[probes]
            cum = mx.cumsum(sz, axis=1)
            total = cum[:, -1]
            width = bucket(max(int(mx.max(total).item()), k))
            rows = mx.arange(nq)
            pos = mx.arange(width)
            flat = (cum + rows[:, None] * width).reshape(-1)
            j = mx.searchsorted(flat, (pos[None, :] + rows[:, None] * width).reshape(-1), side="right")
            j = mx.minimum(j, nq * nprobe - 1).reshape(nq, width)
            first = (cum - sz).reshape(-1)[j]
            valid = pos[None, :] < total[:, None]
            row = mx.where(valid, self.starts[probes.reshape(-1)[j]] + pos[None, :] - first, 0)
            dist = mx.where(valid, self._dist(q, self.x[row], self.xn[row]), mx.inf)
            pick = mx.argpartition(dist, k - 1, axis=1)[:, :k]
            pd = mx.take_along_axis(dist, pick, axis=1)
            order = mx.argsort(pd, axis=1)
            pick, pd = mx.take_along_axis(pick, order, axis=1), mx.take_along_axis(pd, order, axis=1)
            found = mx.where(pd < mx.inf, self.ids[mx.take_along_axis(row, pick, axis=1)], -1)
            mx.eval(found)
            out.append(np.array(found, dtype=np.int64))
            del q, probes, sz, cum, total, flat, j, first, valid, row, dist, pick, pd, order, found
        return np.concatenate(out)


class TorchBuiltin:
    def __init__(self, xb: np.ndarray, built: Lists, metric: str, budget_mb: float):
        import torch
        self.t, self.metric, self.built, self.budget = torch, metric, built, budget_mb
        dev = "mps"
        self.x = torch.from_numpy(np.ascontiguousarray(xb[built.order])).to(dev)
        self.xn = (self.x * self.x).sum(1)
        self.ids = torch.from_numpy(built.order.astype(np.int64)).to(dev)
        self.starts = torch.from_numpy(built.offsets[:-1].astype(np.int64)).to(dev)
        self.sizes = torch.from_numpy(built.sizes().astype(np.int64)).to(dev)
        self.c = torch.from_numpy(built.centroids).to(dev)
        self.cn = (self.c * self.c).sum(1)

    def _dist(self, q, x, xn=None):
        t = self.t
        if x.ndim == 2:
            if self.metric == "l2":
                return t.addmm(xn[None, :], q, x.T, alpha=-2)
            return t.cdist(q, x, p=1)
        if self.metric == "l2":
            return xn - 2 * t.bmm(x, q[:, :, None])[:, :, 0]
        return (x - q[:, None, :]).abs().sum(2)

    def search(self, xq: np.ndarray, k: int, nprobe: int) -> np.ndarray:
        t = self.t
        out = []
        step = chunk_size(self.budget, nprobe, self.built, xq.shape[1], self.metric)
        for s in range(0, len(xq), step):
            q = t.from_numpy(np.ascontiguousarray(xq[s:s + step])).to("mps")
            nq = q.shape[0]
            probes = t.topk(self._dist(q, self.c, self.cn), nprobe, dim=1, largest=False, sorted=False).indices
            sz = self.sizes[probes]
            cum = t.cumsum(sz, 1)
            total = cum[:, -1]
            width = bucket(max(int(total.max().item()), k))
            rows = t.arange(nq, device="mps")
            pos = t.arange(width, device="mps")
            flat = (cum + rows[:, None] * width).reshape(-1)
            j = t.searchsorted(flat, (pos[None, :] + rows[:, None] * width).reshape(-1), right=True)
            j = j.clamp(max=nq * nprobe - 1).reshape(nq, width)
            first = (cum - sz).reshape(-1)[j]
            valid = pos[None, :] < total[:, None]
            row = t.where(valid, self.starts[probes.reshape(-1)[j]] + pos[None, :] - first, 0)
            dist = t.where(valid, self._dist(q, self.x[row], self.xn[row]), float("inf"))
            pd, pick = t.topk(dist, k, dim=1, largest=False, sorted=True)
            found = t.where(pd < float("inf"), self.ids[t.gather(row, 1, pick)], -1)
            out.append(found.cpu().numpy())
        return np.concatenate(out)


def rates(search, xq: np.ndarray, empty_cache) -> tuple[list[float], dict[int, np.ndarray]]:
    """Queries per second at each batch size, and the neighbors found at each, from an empty buffer cache: a
    cache full of another implementation's buffers slowed MLX's custom list-major scan."""
    empty_cache()
    out, found = [], {}
    for b in BATCHES:
        qs = xq[:200] if b == 1 else xq[:1000] if b == 10 else xq
        r, f = throughput(search, qs, b, repeats=1 if b <= 10 else 3)
        out.append(r)
        found[b] = f
    return out, found


def run_part(a, work: str) -> None:
    w = np.load(os.path.join(work, "lists.npz"))
    built = Lists(w["centroids"], w["assign"], w["order"], w["offsets"])
    xb, xq = load(a.dataset, a.queries)
    rows, found = {}, {}
    for nprobe in map(int, a.nprobes.split(",")):
        if a.part == "mlx":
            from siann.ivf_gpu import GpuIVF
            custom = GpuIVF(xb, built.nlist, a.metric, "float32", built=built)
            builtin = MlxBuiltin(xb, built, a.metric, a.budget_mlx or a.budget)
        else:
            from siann.ivf_torch import TorchIVF
            custom = TorchIVF(xb, built.centroids, a.metric, built=built)
            builtin = TorchBuiltin(xb, built, a.metric, a.budget_torch or a.budget)

        def routed(q, n=nprobe, c=custom):
            return c.search_list_major(q, K, n) if len(q) >= 1000 else c.search(q, K, n)
        if a.part == "mlx":
            import mlx.core as mx
            empty = mx.clear_cache
        else:
            import torch
            empty = torch.mps.empty_cache
        for name, s in (("custom", routed), ("built-in", lambda q, n=nprobe, bi=builtin: bi.search(q, K, n))):
            rows[f"{name}{nprobe}"], f = rates(s, xq, empty)
            for b, v in f.items():
                found[f"{name}{nprobe}_{b}"] = v
    if a.part == "mlx":
        import mlx.core as mx
        found["peak_gb"] = np.array(mx.get_peak_memory() / 1e9)
    else:
        import torch
        found["peak_gb"] = np.array(torch.mps.driver_allocated_memory() / 1e9)
    np.savez(os.path.join(work, f"{a.part}.npz"), **{k: np.array(v) for k, v in rows.items()}, **found)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sift", choices=["sift", "glove"])
    p.add_argument("--metric", default="l2", choices=["l2", "l1"], help="glove is served normalized: l2 ranks as cosine")
    p.add_argument("--nlist", type=int, default=4000)
    p.add_argument("--nprobes", default="16,64")
    p.add_argument("--queries", type=int, default=10_000)
    p.add_argument("--budget", type=float, default=4096, help="MB of gathered candidates per chunk (built-in IVF)")
    p.add_argument("--budget-mlx", type=float, help="the MLX built-in IVF's budget, if not --budget's")
    p.add_argument("--budget-torch", type=float, help="the PyTorch built-in IVF's budget, if not --budget's")
    p.add_argument("--part", choices=["mlx", "torch"], help=argparse.SUPPRESS)
    p.add_argument("--work", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.part:
        run_part(a, a.work)
        return
    from siann.ivf_build import build_lists
    xb, xq = load(a.dataset, a.queries)
    built = build_lists(xb, a.nlist, a.metric, device="gpu")
    truth = np.asarray(FlatIndex(xb, xb.shape[1], a.metric).search(xq, K), dtype=np.int64).reshape(-1, K)
    with tempfile.TemporaryDirectory() as work:
        np.savez(os.path.join(work, "lists.npz"), centroids=built.centroids, assign=built.assign,
                 order=built.order, offsets=built.offsets)
        for part in ("mlx", "torch"):
            subprocess.run([sys.executable, __file__, *sys.argv[1:], "--part", part, "--work", work], check=True)
        res = {part: np.load(os.path.join(work, f"{part}.npz")) for part in ("mlx", "torch")}
        print(f"{a.dataset}, {a.metric}: {len(xb):,} vectors, {a.nlist:,} lists (float32), {len(xq):,} queries; "
              "queries per second by batch size")
        print(f"\n  {'nprobe, framework, kernels':34s} {'recall':>7s}" + "".join(f"{b:>10,}" for b in BATCHES)
              + "   same as custom")
        for nprobe in map(int, a.nprobes.split(",")):
            for part, label in (("mlx", "MLX"), ("torch", "PyTorch")):
                r = res[part]
                for name in ("custom", "built-in"):
                    rec = agreement(xb, xq, r[f"{name}{nprobe}_10000"], truth, a.metric)
                    same = "" if name == "custom" else \
                        f"   {agreement(xb, xq, r[f'built-in{nprobe}_100'], r[f'custom{nprobe}_100'], a.metric):.4f}"
                    print(f"  {f'{nprobe}, {label}, {name}':34s} {rec:7.3f}"
                          + "".join(f"{v:10,.0f}" for v in r[f"{name}{nprobe}"]) + same, flush=True)
                ratio = r[f"custom{nprobe}"] / r[f"built-in{nprobe}"]
                if nprobe == int(a.nprobes.split(",")[-1]):
                    print(f"  {label}: peak GPU memory {float(r['peak_gb']):.1f} GB (MLX's peak; PyTorch's allocated at the end)")
                print(f"  {f'{nprobe}, {label}: custom / built-in':34s} {'':7s}" + "".join(f"{v:10.2f}" for v in ratio))


if __name__ == "__main__":
    main()
