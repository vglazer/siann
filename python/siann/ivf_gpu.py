"""IVF on the GPU (MLX) for L1, L2, L-infinity and inner product.

Build: k-means centroids (in L2), each vector assigned to its nearest centroid in the search metric by
the exact GPU kernel, vectors stored grouped by list (float32, float16 or uint8). Search, a batch of
queries at a time: each query's nprobe nearest centroids, then either the query-major scan (128
threads per query walk its lists and merge their best k inside the kernel) or the list-major scan
(queries grouped by list, each list read once for all the queries that probe it).
"""
import mlx.core as mx
import numpy as np

from .knn_kernels import COMBINE, _finish, dist, ip_knn, knn_tiled, l2_knn, select_k

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
# The query-major scan with the merge inside the kernel: each query's T threads form one threadgroup
# (T a power of two up to 256); after scanning, each thread's sorted best K goes to threadgroup memory,
# and the lists are merged pairwise in log2(T) steps (two sorted lists of K into one, the left first
# on ties), so the kernel writes only K results per query, sorted.
SCAN_MERGED = SCAN.split("    for (int a = 0; a < K; ++a) {\n        out_val")[0].replace(
    "    if (t >= T || i >= Q) return;\n", "") + """
    threadgroup float tv[T * K];
    threadgroup int ti[T * K];
    for (int a = 0; a < K; ++a) { tv[t * K + a] = best[a]; ti[t * K + a] = where[a]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = T / 2; stride > 0; stride /= 2) {
        if (t < stride) {
            float mv[K];
            int mi[K];
            uint a0 = t * K, b0 = (t + stride) * K;
            int ia = 0, ib = 0;
            for (int c = 0; c < K; ++c) {
                float va = tv[a0 + ia], vb = tv[b0 + ib];
                if (va <= vb) { mv[c] = va; mi[c] = ti[a0 + ia]; ++ia; }
                else { mv[c] = vb; mi[c] = ti[b0 + ib]; ++ib; }
            }
            for (int c = 0; c < K; ++c) { tv[a0 + c] = mv[c]; ti[a0 + c] = mi[c]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0) {
        for (int a = 0; a < K; ++a) { out_val[i * K + a] = tv[a]; out_idx[i * K + a] = ti[a]; }
    }
"""
METRICS = ("l1", "l2", "linf", "ip")
SCANS = {m: mx.fast.metal_kernel(name=f"ivf_scan_{m}", input_names=["q", "x", "offsets", "probes"],
                                 output_names=["out_val", "out_idx"], source=SCAN, header=COMBINE[m])
         for m in METRICS}
SCANS_MERGED = {m: mx.fast.metal_kernel(name=f"ivf_scan_merged_{m}", input_names=["q", "x", "offsets", "probes"],
                                        output_names=["out_val", "out_idx"], source=SCAN_MERGED, header=COMBINE[m])
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


def probe_lists(dist: mx.array, nprobe: int, select: str = "argpartition") -> mx.array:
    """Each query's nprobe nearest lists (int32), from its distances to the centroids. "argpartition"
    returns them in no order, which the scans do not need; "select_k", siann's two-stage kernel with
    32 threads per query, returns them in order."""
    if select == "argpartition":
        if nprobe >= dist.shape[1]:
            return mx.broadcast_to(mx.arange(dist.shape[1], dtype=mx.int32), dist.shape)
        return mx.argpartition(dist, kth=nprobe - 1, axis=1)[:, :nprobe].astype(mx.int32)
    return select_k(dist, nprobe, threads=32).astype(mx.int32)


class GpuIVF:
    def __init__(self, xb: np.ndarray, nlist: int, metric: str, lists: str = "float32"):
        """``lists`` is the storage of the inverted lists: "float32"; "float16", half the memory
        traffic, distances still computed in float32 from the rounded vectors; or "uint8", a quarter,
        for vectors whose coordinates are integers from 0 to 255 (SIFT's), so that nothing is lost."""
        if lists == "uint8" and not np.array_equal(xb.astype(np.uint8).astype(np.float32), xb):
            raise ValueError("uint8 lists need coordinates that are integers from 0 to 255")
        self.metric = metric
        sample = xb[np.random.default_rng(0).choice(len(xb), min(len(xb), 100 * nlist), replace=False)]
        import faiss   # k-means only

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

    def search(self, xq: np.ndarray, k: int, nprobe: int, threads: int = 128,
               select: str = "auto", merge: str = "kernel") -> np.ndarray:
        """The query-major scan: `threads` threads per query walk its lists (128 is fastest at every
        measured nprobe). `select` chooses each query's probes (probe_lists); "auto" is argpartition,
        fastest for this scan at every nprobe (BENCHMARKS.md, "Where a GPU call's time goes")."""
        select = "argpartition" if select == "auto" else select
        q = mx.array(xq)
        dist = coarse_dist(q, self.centroids, self.cnorms, self.metric)
        probes = probe_lists(dist, nprobe, select)
        nq, d = q.shape
        if merge == "kernel" and threads <= 256 and threads & (threads - 1) == 0 and threads * k * 8 <= 32768:
            # the merge inside the scan kernel: K sorted results per query
            _, idx = SCANS_MERGED[self.metric](
                inputs=[q, self.x, self.offsets, probes],
                template=[("Q", nq), ("D", d), ("P", nprobe), ("T", threads), ("K", k)],
                grid=(threads, nq, 1), threadgroup=(threads, 1, 1),
                output_shapes=[(nq * k,), (nq * k,)], output_dtypes=[mx.float32, mx.int32])
            local = idx.reshape(nq, k)
        else:
            vals, idx = SCANS[self.metric](
                inputs=[q, self.x, self.offsets, probes],
                template=[("Q", nq), ("D", d), ("P", nprobe), ("T", threads), ("K", k)],
                grid=(threads, nq, 1), threadgroup=(min(threads, 256), 1, 1),
                output_shapes=[(nq * threads * k,), (nq * threads * k,)], output_dtypes=[mx.float32, mx.int32])
            local = _finish(vals, idx, nq, k)
        result = mx.where(local >= 0, mx.take(self.ids, mx.maximum(local, 0)), -1)   # -1: fewer than k found
        mx.eval(result)
        return np.array(result)


def search_list_major(self, xq: np.ndarray, k: int, nprobe: int, select: str = "auto") -> np.ndarray:
    """The same search with the list-major scan, grouped by list on the GPU. `select` "auto" uses
    select_k up to nprobe 16 and argpartition above: for batches of thousands select_k is faster at
    small nprobe (938,000 against 581,000 queries per second at nprobe 8 on SIFT), argpartition from 32."""
    if select == "auto":
        select = "select_k" if nprobe <= 16 else "argpartition"
    q = mx.array(xq)
    nq, d = q.shape
    nlist = self.offsets.shape[0] - 1
    dist = coarse_dist(q, self.centroids, self.cnorms, self.metric)
    probes = probe_lists(dist, nprobe, select).reshape(-1)   # pair id = query * nprobe + slot
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
    # pick is -1 where fewer than k candidates were found; take_along_axis would read it as the last column
    local = mx.where(pick >= 0, mx.take_along_axis(idx.reshape(nq, -1), mx.maximum(pick, 0), axis=1), -1)
    result = mx.where(local >= 0, mx.take(self.ids, mx.maximum(local, 0)), -1)   # -1: fewer than k found
    mx.eval(result)
    return np.array(result)


GpuIVF.search_list_major = search_list_major
