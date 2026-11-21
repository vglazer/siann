"""siann.ivf_gpu's GPU IVF, ported to PyTorch's MPS backend through torch.mps.compile_shader: the same
Metal code for the query-major and list-major scans, the same coarse step (the exact kernels from
torch_kernels, then the same probe selection) and the same merges (inside the query-major scan's
kernel; the two-stage selection after the list-major scan). It takes its centroids from outside (the
benchmarks pass GpuIVF's), and assigns each base vector to its nearest centroid by exact search, so its
lists match GpuIVF's.
"""
import numpy as np
import torch

from . import torch_kernels as tk
from ._metal import COMBINE, list_scan, scan, scan_merged
from .ivf_build import build_lists


def _lib(source: str, metric: str, **sizes: int):
    return tk._lib(COMBINE[metric] + source, **sizes)


class TorchIVF:
    def __init__(self, xb: np.ndarray, centroids: np.ndarray, metric: str, built=None):
        """The lists come from ``built`` (a ``siann.ivf_build.Lists``, the same for every backend), or are
        built from ``centroids``, assigned as siann.ivf_build assigns them."""
        self.metric = metric
        dev = "mps"
        if built is None:
            built = build_lists(xb, len(centroids), metric, centroids=centroids, device="cpu")
        self.built = built
        self.centroids = torch.from_numpy(built.centroids).to(dev)
        self.cnorms = (self.centroids * self.centroids).sum(1)
        self.nlist = built.nlist
        self.ids = torch.from_numpy(built.order).to(dev)
        self.x = torch.from_numpy(np.ascontiguousarray(xb[built.order])).to(dev)
        self.offsets = torch.from_numpy(built.offsets.astype(np.int32)).to(dev)

    def _to_ids(self, local: torch.Tensor) -> np.ndarray:
        """Positions in the lists to base ids; -1 (fewer than k found) stays -1."""
        return torch.where(local >= 0, self.ids[local.clamp(min=0).long()], -1).cpu().numpy()

    def _nearest(self, q: torch.Tensor, k: int, threads: int = 512, metric: str | None = None) -> torch.Tensor:
        """The k nearest centroids of each query in `metric` (the index's by default), by the exact kernels."""
        metric = metric or self.metric
        if metric != "l2":
            return tk.select_k(tk.dist(q, self.centroids, metric), k, threads)
        return tk.select_k(torch.addmm(self.cnorms[None, :], q, self.centroids.T, alpha=-2), k, threads)

    def probe_lists(self, q: torch.Tensor, nprobe: int, select: str = "argpartition") -> torch.Tensor:
        """Each query's nprobe nearest lists: unordered by torch.topk (sorted=False, as ivf_gpu.py's
        argpartition), or in increasing order by the two-stage selection with 32 threads ("select_k")."""
        if select == "select_k":
            return self._nearest(q, nprobe, threads=32).to(torch.int32).contiguous()
        if self.metric != "l2":
            dist = tk.dist(q, self.centroids, self.metric)
        else:
            dist = torch.addmm(self.cnorms[None, :], q, self.centroids.T, alpha=-2)
        if nprobe >= self.nlist:
            return torch.arange(self.nlist, device="mps", dtype=torch.int32).expand(len(q), -1).contiguous()
        return torch.topk(dist, nprobe, dim=1, largest=False, sorted=False).indices.to(torch.int32).contiguous()

    def search(self, xq: np.ndarray, k: int, nprobe: int, threads: int = 128, select: str = "auto",
               merge: str = "kernel") -> np.ndarray:
        """The query-major scan: `threads` threads per query walk its lists. select "auto" is the unordered
        selection; merge "kernel" merges each query's candidates in the scan kernel (for threads a power of
        two up to 256), "separate" with torch.topk over every thread's candidates."""
        q = torch.from_numpy(xq).to("mps")
        nq, d = q.shape
        probes = self.probe_lists(q, nprobe, "argpartition" if select == "auto" else select)
        if merge == "kernel" and threads <= 256 and threads & (threads - 1) == 0 and threads * k * 8 <= 32768:
            vals = torch.empty(nq * k, device="mps")
            idx = torch.empty(nq * k, device="mps", dtype=torch.int32)
            _lib(scan_merged("float").torch_source(), self.metric, D=d, T=threads, K=k).ivf_scan_merged(
                q, self.x, self.offsets, probes, tk.sz(nq, nprobe), vals, idx, threads=[threads, nq],
                group_size=[threads, 1])
            local = idx.view(nq, k)
            return self._to_ids(local)
        vals = torch.empty(nq * threads * k, device="mps")
        idx = torch.empty(nq * threads * k, device="mps", dtype=torch.int32)
        _lib(scan("float").torch_source(), self.metric, D=d, T=threads, K=k).ivf_scan(
            q, self.x, self.offsets, probes, tk.sz(nq, nprobe), vals, idx, threads=[threads, nq],
            group_size=[min(threads, 256), 1])
        best = torch.topk(vals.view(nq, -1), k, dim=1, largest=False, sorted=True).indices
        local = torch.gather(idx.view(nq, -1), 1, best)
        return self._to_ids(local)

    def search_list_major(self, xq: np.ndarray, k: int, nprobe: int, select: str = "auto") -> np.ndarray:
        """The list-major scan: queries grouped by list on the GPU, one threadgroup per list. select "auto" is
        the ordered selection up to nprobe 16 (faster for large batches) and the unordered one above."""
        q = torch.from_numpy(xq).to("mps")
        nq, d = q.shape
        lanes = 16
        if select == "auto":
            select = "select_k" if nprobe <= 16 else "argpartition"
        probes = self.probe_lists(q, nprobe, select).reshape(-1).long()   # pair id = query * nprobe + slot
        pair_ids = torch.argsort(probes).to(torch.int32)
        counts = torch.bincount(probes, minlength=self.nlist).to(torch.int32)
        first = (torch.cumsum(counts, 0) - counts).to(torch.int32)
        vals = torch.empty(nq * nprobe * lanes * k, device="mps")
        idx = torch.empty(nq * nprobe * lanes * k, device="mps", dtype=torch.int32)
        _lib(list_scan("float").torch_source(), self.metric, D=d, K=k).ivf_list_scan(
            q, self.x, self.offsets, pair_ids, first, counts, tk.sz(nprobe), vals, idx,
            threads=[256 * self.nlist], group_size=[256])
        pick = tk.select_k(vals.view(nq, -1), k, threads=64)
        pick = pick.long()   # -1 where fewer than k candidates were found
        local = torch.where(pick >= 0, torch.gather(idx.view(nq, -1), 1, pick.clamp(min=0)), -1)
        return self._to_ids(local)
