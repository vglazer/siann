"""ivf_gpu.py's GPU IVF, ported to PyTorch's MPS backend through torch.mps.compile_shader: the same
Metal code for the query-major and list-major scans, the same coarse step (the exact kernels from
torch_kernels.py, then the same probe selection) and the same merges (inside the query-major scan's
kernel; the two-stage selection after the list-major scan).

FAISS and PyTorch cannot share a process on macOS, so this runs on its own, from centroids saved by
ivf_cpu.py (which calls it), and prints its rows in ivf_cpu.py's table. Each base vector goes to its
nearest centroid by exact search here, as in ivf_gpu.py, so the lists match the other IVFs' and so
should the recall at each nprobe.

    uv run --no-sync --group proto python ivf_torch.py --centroids C.npy --dataset sift --metric l2
"""
import argparse

import numpy as np
import torch
from siann_cpu import FlatIndex

import torch_kernels as tk
from backends import agreement
from datasets import load
from timing import throughput

from knn_kernels import COMBINE

SCAN = """
kernel void ivf_scan(device const float* q, device const float* x, device const int* offsets,
                     device const int* probes, device float* out_val, device int* out_idx,
                     uint2 pos [[thread_position_in_grid]]) {
    uint t = pos.x, i = pos.y;
    if (t >= T || i >= Q) return;
    float best[K];
    int where[K];
    for (uint a = 0; a < K; ++a) { best[a] = INFINITY; where[a] = -1; }
    for (uint p = 0; p < P; ++p) {
        int l = probes[i * P + p];
        for (int j = offsets[l] + int(t); j < offsets[l + 1]; j += int(T)) {
            float v = 0.0f;
            for (uint d = 0; d < D; ++d) { v = combine(v, q[i * D + d], x[uint(j) * D + d]); }
            if (v < best[K - 1]) {
                int s = K - 1;
                while (s > 0 && best[s - 1] > v) { best[s] = best[s - 1]; where[s] = where[s - 1]; --s; }
                best[s] = v;
                where[s] = j;
            }
        }
    }
    for (uint a = 0; a < K; ++a) {
        out_val[(i * T + t) * K + a] = best[a];
        out_idx[(i * T + t) * K + a] = where[a];
    }
}
"""

# The query-major scan with each query's candidates merged inside the kernel, as ivf_gpu.py's
# SCAN_MERGED: one threadgroup of T threads per query (T a power of two up to 256), each thread's
# sorted top-k written to threadgroup memory, then a tree merge in log2(T) steps.
SCAN_MERGED = SCAN.split("    for (uint a = 0; a < K; ++a) {\n        out_val")[0].replace(
    "    if (t >= T || i >= Q) return;\n", "").replace("kernel void ivf_scan(", "kernel void ivf_scan_merged(") + """
    threadgroup float tv[T * K];
    threadgroup int ti[T * K];
    for (uint a = 0; a < K; ++a) { tv[t * K + a] = best[a]; ti[t * K + a] = where[a]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = T / 2; stride > 0; stride /= 2) {
        if (t < stride) {
            float mv[K];
            int mi[K];
            uint a0 = t * K, b0 = (t + stride) * K;
            uint ia = 0, ib = 0;
            for (uint c = 0; c < K; ++c) {
                float va = tv[a0 + ia], vb = tv[b0 + ib];
                if (va <= vb) { mv[c] = va; mi[c] = ti[a0 + ia]; ++ia; }
                else { mv[c] = vb; mi[c] = ti[b0 + ib]; ++ib; }
            }
            for (uint c = 0; c < K; ++c) { tv[a0 + c] = mv[c]; ti[a0 + c] = mi[c]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0) {
        for (uint a = 0; a < K; ++a) { out_val[i * K + a] = tv[a]; out_idx[i * K + a] = ti[a]; }
    }
}
"""

LIST_SCAN = """
kernel void ivf_list_scan(device const float* q, device const float* x, device const int* offsets,
                          device const int* pair_ids, device const int* list_first,
                          device const int* list_count, device float* out_val, device int* out_idx,
                          uint l [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]]) {
    const uint QB = 16, LANES = 16, XB = 64, S = 16, R = XB / LANES;
    threadgroup float qs[QB][S + 1];
    threadgroup float xs[XB][S + 1];
    uint qa = tid / LANES, lane = tid % LANES;
    int x0 = offsets[l], x1 = offsets[l + 1];
    int first = list_first[l], total = list_count[l];
    for (int c0 = 0; c0 < total; c0 += int(QB)) {
    int start = first + c0, count = min(int(QB), total - c0);
    bool active = int(qa) < count;
    int pair = active ? pair_ids[start + int(qa)] : 0;
    float best[K];
    int where[K];
    for (uint a = 0; a < K; ++a) { best[a] = INFINITY; where[a] = -1; }
    for (int base = x0; base < x1; base += int(XB)) {
        float acc[R];
        for (uint r = 0; r < R; ++r) acc[r] = 0.0f;
        for (uint d0 = 0; d0 < D; d0 += S) {
            for (uint e = tid; e < QB * S; e += 256) {
                uint r = e / S, c = e % S;
                qs[r][c] = (int(r) < count && d0 + c < D) ? q[uint(pair_ids[start + int(r)] / int(P)) * D + d0 + c] : 0.0f;
            }
            for (uint e = tid; e < XB * S; e += 256) {
                uint r = e / S, c = e % S;
                int j = base + int(r);
                xs[r][c] = (j < x1 && d0 + c < D) ? x[uint(j) * D + d0 + c] : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint c = 0; c < S; ++c) {
                float qv = qs[qa][c];
                for (uint r = 0; r < R; ++r) acc[r] = combine(acc[r], qv, xs[lane + r * LANES][c]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint r = 0; r < R; ++r) {
            int j = base + int(lane + r * LANES);
            float v = acc[r];
            if (active && j < x1 && v < best[K - 1]) {
                int s = K - 1;
                while (s > 0 && best[s - 1] > v) { best[s] = best[s - 1]; where[s] = where[s - 1]; --s; }
                best[s] = v;
                where[s] = j;
            }
        }
    }
    if (active) {
        for (uint a = 0; a < K; ++a) {
            out_val[(uint(pair) * LANES + lane) * K + a] = best[a];
            out_idx[(uint(pair) * LANES + lane) * K + a] = where[a];
        }
    }
    }
}
"""


def _lib(source: str, metric: str, **sizes: int):
    return tk._lib(COMBINE[metric] + source, **sizes)


class TorchIVF:
    def __init__(self, xb: np.ndarray, centroids: np.ndarray, metric: str):
        self.metric = metric
        dev = "mps"
        self.centroids = torch.from_numpy(np.ascontiguousarray(centroids)).to(dev)
        self.cnorms = (self.centroids * self.centroids).sum(1)
        self.nlist = len(centroids)
        assign = np.concatenate([self._nearest(torch.from_numpy(xb[i:i + 4096]).to(dev), 1)[:, 0].cpu().numpy()
                                 for i in range(0, len(xb), 4096)])
        order = np.argsort(assign, kind="stable")
        self.ids = torch.from_numpy(order.astype(np.int64)).to(dev)
        self.x = torch.from_numpy(np.ascontiguousarray(xb[order])).to(dev)
        counts = np.bincount(assign, minlength=self.nlist)
        self.offsets = torch.from_numpy(np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)).to(dev)

    def _nearest(self, q: torch.Tensor, k: int, threads: int = 512) -> torch.Tensor:
        """The k nearest centroids of each query, by the exact kernels."""
        if self.metric != "l2":
            return tk.select_k(tk.dist(q, self.centroids, self.metric), k, threads)
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
            _lib(SCAN_MERGED, self.metric, Q=nq, D=d, P=nprobe, T=threads, K=k).ivf_scan_merged(
                q, self.x, self.offsets, probes, vals, idx, threads=[threads, nq], group_size=[threads, 1])
            local = idx.view(nq, k)
            return self.ids[local.clamp(min=0).long()].cpu().numpy()
        vals = torch.empty(nq * threads * k, device="mps")
        idx = torch.empty(nq * threads * k, device="mps", dtype=torch.int32)
        _lib(SCAN, self.metric, Q=nq, D=d, P=nprobe, T=threads, K=k).ivf_scan(
            q, self.x, self.offsets, probes, vals, idx, threads=[threads, nq], group_size=[min(threads, 256), 1])
        best = torch.topk(vals.view(nq, -1), k, dim=1, largest=False, sorted=True).indices
        local = torch.gather(idx.view(nq, -1), 1, best)
        return self.ids[local.clamp(min=0).long()].cpu().numpy()

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
        _lib(LIST_SCAN, self.metric, D=d, P=nprobe, K=k).ivf_list_scan(
            q, self.x, self.offsets, pair_ids, first, counts, vals, idx,
            threads=[256 * self.nlist], group_size=[256])
        pick = tk.select_k(vals.view(nq, -1), k, threads=64)
        local = torch.gather(idx.view(nq, -1), 1, pick.long())
        return self.ids[local.clamp(min=0).long()].cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--centroids", required=True)
    p.add_argument("--dataset", default="sift", choices=["sift", "glove", "glove-raw"])
    p.add_argument("--metric", default="l2", choices=["l1", "l2", "linf", "ip"])
    p.add_argument("--queries", type=int, default=10_000)
    p.add_argument("--small", type=int, default=1000)
    p.add_argument("--nprobes", default="4,16,32,64,128")
    a = p.parse_args()
    k = 10
    xb, xq = load(a.dataset, a.queries)
    truth = np.array(FlatIndex(xb, xb.shape[1], a.metric).search(xq, k), dtype=np.int64).reshape(-1, k)
    index = TorchIVF(xb, np.load(a.centroids), a.metric)
    small, full = xq[:a.small], xq
    for nprobe in map(int, a.nprobes.split(",")):
        def search(q, n=nprobe):
            return index.search(q, k, n) if len(q) < 1000 else index.search_list_major(q, k, n)
        qps1, found1 = throughput(search, small, 1, repeats=1)
        qps100, _ = throughput(search, small, 100)
        qps_full, found = throughput(search, full, len(full))
        r = agreement(xb, xq, found, truth, a.metric)
        r1 = agreement(xb, small, found1, truth[:len(small)], a.metric)
        assert abs(r1 - agreement(xb, small, found[:len(small)], truth[:len(small)], a.metric)) < 0.01
        print(f"  {f'PyTorch IVF, nprobe {nprobe}':24s} {r:9.3f} {qps1:12,.0f} {qps100:10,.0f} {qps_full:10,.0f}",
              flush=True)


if __name__ == "__main__":
    main()
