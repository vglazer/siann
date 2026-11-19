"""IVF on the GPU for L1 (and L2, L-infinity), against FAISS's HNSW and IVF on SIFT1M.

Build: k-means centroids (FAISS's, in L2), each vector assigned to its nearest centroid in the search
metric by the exact GPU kernel, vectors stored grouped by list. Search, a batch of queries at a time:
the nearest nprobe centroids by the exact kernel, then a scan kernel in which T threads per query walk
its lists, each keeping its best k in registers, and a small final merge.

    uv run --group proto python ivf_gpu.py [--metric l1|l2|linf] [--batch 1000]
    uv run --group proto python ivf_gpu.py --check
"""
import argparse
import time

import faiss
import mlx.core as mx
import numpy as np

from datasets import load
from knn_kernels import COMBINE, _finish, dist, ip_knn, knn_tiled, l2_knn, select_k

SCAN = """
    uint t = thread_position_in_grid.x;
    uint i = thread_position_in_grid.y;
    if (t >= T || i >= Q) return;
    float best[K];
    int where[K];
    for (int a = 0; a < K; ++a) { best[a] = INFINITY; where[a] = -1; }
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
    for (int a = 0; a < K; ++a) {
        out_val[(i * T + t) * K + a] = best[a];
        out_idx[(i * T + t) * K + a] = where[a];
    }
"""
METRICS = ("l1", "l2", "linf", "ip")
SCANS = {m: mx.fast.metal_kernel(name=f"ivf_scan_{m}", input_names=["q", "x", "offsets", "probes"],
                                 output_names=["out_val", "out_idx"], source=SCAN, header=COMBINE[m])
         for m in METRICS}


# List-major scan: one threadgroup per work item, a list and up to QB of the queries probing it.
# 256 threads = QB queries x LANES lanes; the queries and XB of the list's vectors at a time are staged
# through threadgroup memory S dimensions at a time, so each list vector is read once per QB queries.
# Each thread keeps the best K for its query; output is per (query, probe slot, lane).
LIST_MAJOR = """
    const uint QB = 16, LANES = 16, XB = 64, S = 16, R = XB / LANES;
    threadgroup float qs[QB][S + 1];
    threadgroup float xs[XB][S + 1];
    uint l = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint qa = tid / LANES, lane = tid % LANES;
    int x0 = offsets[l], x1 = offsets[l + 1];
    int first = list_first[l], total = list_count[l];
    for (int c0 = 0; c0 < total; c0 += int(QB)) {
    int start = first + c0, count = min(int(QB), total - c0);
    bool active = int(qa) < count;
    int pair = active ? pair_ids[start + int(qa)] : 0;
    float best[K];
    int where[K];
    for (int a = 0; a < K; ++a) { best[a] = INFINITY; where[a] = -1; }
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
        for (int a = 0; a < K; ++a) {
            out_val[(uint(pair) * LANES + lane) * K + a] = best[a];
            out_idx[(uint(pair) * LANES + lane) * K + a] = where[a];
        }
    }
    }
"""
LIST_SCANS = {m: mx.fast.metal_kernel(
    name=f"ivf_list_scan_{m}", input_names=["q", "x", "offsets", "pair_ids", "list_first", "list_count"],
    output_names=["out_val", "out_idx"], source=LIST_MAJOR, header=COMBINE[m])
    for m in METRICS}


def nearest(q, x, norms, k, metric):
    if metric == "l2":
        return l2_knn(q, x, norms, k)
    if metric == "ip":
        return ip_knn(q, x, k)
    return knn_tiled(q, x, k, metric)


def coarse_dist(q, centroids, cnorms, metric):
    """The distances (squared for L2; the negated inner product for "ip") from each query to every
    centroid."""
    if metric == "l2":
        return mx.addmm(cnorms[None, :], q, centroids.T, alpha=-2.0)
    if metric == "ip":
        return mx.addmm(mx.zeros_like(cnorms)[None, :], q, centroids.T, alpha=-1.0)
    return dist(q, centroids, metric)


class GpuIVF:
    def __init__(self, xb: np.ndarray, nlist: int, metric: str, lists: str = "float32"):
        """``lists`` is the storage of the inverted lists: "float32"; "float16", half the memory
        traffic, distances still computed in float32 from the rounded vectors; or "uint8", a quarter,
        for vectors whose coordinates are integers from 0 to 255 (SIFT's), so that nothing is lost."""
        if lists == "uint8" and not np.array_equal(xb.astype(np.uint8).astype(np.float32), xb):
            raise ValueError("uint8 lists need coordinates that are integers from 0 to 255")
        self.metric = metric
        sample = xb[np.random.default_rng(0).choice(len(xb), min(len(xb), 100 * nlist), replace=False)]
        km = faiss.Kmeans(xb.shape[1], nlist, niter=10, seed=1)
        km.train(sample)
        self.centroids = mx.array(km.centroids)
        self.cnorms = mx.sum(self.centroids * self.centroids, axis=1)
        assign = np.concatenate([np.array(nearest(mx.array(xb[i:i + 4096]), self.centroids, self.cnorms, 1, metric))[:, 0]
                                 for i in range(0, len(xb), 4096)])
        order = np.argsort(assign, kind="stable")
        self.ids = mx.array(order.astype(np.int32))
        self.x = mx.array(xb[order]).astype(getattr(mx, lists))
        counts = np.bincount(assign, minlength=nlist)
        self.offsets = mx.array(np.concatenate([[0], np.cumsum(counts)]).astype(np.int32))
        mx.eval(self.x, self.ids, self.offsets)

    def search(self, xq: np.ndarray, k: int, nprobe: int, threads: int = 64) -> np.ndarray:
        q = mx.array(xq)
        # few threads per row: with k = nprobe, each thread's k survivors add up quickly
        dist = coarse_dist(q, self.centroids, self.cnorms, self.metric)
        probes = select_k(dist, nprobe, threads=32).astype(mx.int32)
        nq, d = q.shape
        vals, idx = SCANS[self.metric](
            inputs=[q, self.x, self.offsets, probes],
            template=[("Q", nq), ("D", d), ("P", nprobe), ("T", threads), ("K", k)],
            grid=(threads, nq, 1), threadgroup=(min(threads, 256), 1, 1),
            output_shapes=[(nq * threads * k,), (nq * threads * k,)], output_dtypes=[mx.float32, mx.int32])
        local = _finish(vals, idx, nq, k)
        result = mx.take(self.ids, mx.maximum(local, 0))
        mx.eval(result)
        return np.array(result)


def search_list_major(self, xq: np.ndarray, k: int, nprobe: int) -> np.ndarray:
    """The same search with the list-major scan, grouped by list on the GPU."""
    q = mx.array(xq)
    nq, d = q.shape
    nlist = self.offsets.shape[0] - 1
    dist = coarse_dist(q, self.centroids, self.cnorms, self.metric)
    probes = select_k(dist, nprobe, threads=32).reshape(-1)   # pair id = query * nprobe + slot
    pair_ids = mx.argsort(probes).astype(mx.int32)
    counts = mx.zeros((nlist,), dtype=mx.int32).at[probes].add(1)
    first = (mx.cumsum(counts) - counts).astype(mx.int32)
    lanes = 16
    vals, idx = LIST_SCANS[self.metric](
        inputs=[q, self.x, self.offsets, pair_ids, first, counts],
        template=[("D", d), ("P", nprobe), ("K", k)],
        grid=(256 * nlist, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(nq * nprobe * lanes * k,), (nq * nprobe * lanes * k,)], output_dtypes=[mx.float32, mx.int32])
    pick = select_k(vals.reshape(nq, -1), k, threads=64)
    local = mx.take_along_axis(idx.reshape(nq, -1), pick, axis=1)
    result = mx.take(self.ids, mx.maximum(local, 0))
    mx.eval(result)
    return np.array(result)


GpuIVF.search_list_major = search_list_major


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
