"""IVF on the GPU (MLX) for L1, L2, L-infinity and inner product.

Build (siann.ivf_build): siann's k-means centroids (in L2), each vector assigned to its nearest centroid
by the exact kernels (in the search metric; in L2 for inner product), vectors stored grouped by list
(float32, float16 or uint8). Search, a batch of
queries at a time: each query's nprobe nearest centroids, then either the query-major scan (128
threads per query walk its lists and merge their best k inside the kernel) or the list-major scan
(queries grouped by list, each list read once for all the queries that probe it).
"""
import mlx.core as mx
import numpy as np

from ._metal import COMBINE, list_scan, scan, scan_merged
from .ivf_build import build_lists
from .knn_kernels import _finish, dist, mlx_kernel, select_k, sz

METRICS = ("l1", "l2", "linf", "ip")
# The scans' bodies are siann._metal's, shared with the PyTorch port; MLX infers the lists' element type
# (float32, float16 or uint8) from the array passed.
SCANS = {m: mlx_kernel(scan("float"), COMBINE[m], f"_{m}") for m in METRICS}
SCANS_MERGED = {m: mlx_kernel(scan_merged("float"), COMBINE[m], f"_{m}") for m in METRICS}
LIST_SCANS = {m: mlx_kernel(list_scan("float"), COMBINE[m], f"_{m}") for m in METRICS}


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
    def __init__(self, xb: np.ndarray, nlist: int, metric: str, lists: str = "float32",
                 assign: str | None = None, probe: str | None = None, centroids: np.ndarray | None = None,
                 built=None):
        """``lists`` is the storage of the inverted lists: "float32"; "float16", half the memory
        traffic, distances still computed in float32 from the rounded vectors; or "uint8", a quarter,
        for vectors whose coordinates are integers from 0 to 255 (SIFT's), so that nothing is lost.
        ``assign`` is the metric that puts each vector in its list and ``probe`` the one that picks a
        query's lists (the coarse step); both default to ``metric``, which the scan always uses, except
        that for inner product the lists are assigned by L2 (by inner product, centroids pointing
        towards long vectors attract much of the data: BENCHMARKS.md, "Inner product"). ``built`` is a
        ``siann.ivf_build.Lists`` to take the lists from (the same lists for every backend);
        otherwise they are built here, from ``centroids`` if given."""
        if lists == "uint8" and not np.array_equal(xb.astype(np.uint8).astype(np.float32), xb):
            raise ValueError("uint8 lists need coordinates that are integers from 0 to 255")
        self.metric = metric
        self.probe_metric = probe or metric   # the coarse step's metric; the scan uses `metric`
        if built is None:
            built = build_lists(xb, nlist, metric, centroids=centroids, assign=assign, device="gpu")
        self.built = built
        self.centroids = mx.array(built.centroids)
        self.cnorms = mx.sum(self.centroids * self.centroids, axis=1)
        self.ids = mx.array(built.order.astype(np.int32))
        self.x = mx.array(xb[built.order]).astype(getattr(mx, lists))
        self.offsets = mx.array(built.offsets.astype(np.int32))
        mx.eval(self.x, self.ids, self.offsets)

    def search(self, xq: np.ndarray, k: int, nprobe: int, threads: int = 128,
               select: str = "auto", merge: str = "kernel") -> np.ndarray:
        """The query-major scan: `threads` threads per query walk its lists (128 is fastest at every
        measured nprobe). `select` chooses each query's probes (probe_lists); "auto" is argpartition,
        fastest for this scan at every nprobe (BENCHMARKS.md, "Where a GPU call's time goes")."""
        select = "argpartition" if select == "auto" else select
        q = mx.array(xq)
        dist = coarse_dist(q, self.centroids, self.cnorms, self.probe_metric)
        probes = probe_lists(dist, nprobe, select)
        nq, d = q.shape
        if merge == "kernel" and threads <= 256 and threads & (threads - 1) == 0 and threads * k * 8 <= 32768:
            # the merge inside the scan kernel: K sorted results per query
            _, idx = SCANS_MERGED[self.metric](
                inputs=[q, self.x, self.offsets, probes, sz(nq, nprobe)],
                template=[("D", d), ("T", threads), ("K", k)],
                grid=(threads, nq, 1), threadgroup=(threads, 1, 1),
                output_shapes=[(nq * k,), (nq * k,)], output_dtypes=[mx.float32, mx.int32])
            local = idx.reshape(nq, k)
        else:
            vals, idx = SCANS[self.metric](
                inputs=[q, self.x, self.offsets, probes, sz(nq, nprobe)],
                template=[("D", d), ("T", threads), ("K", k)],
                grid=(threads, nq, 1), threadgroup=(min(threads, 256), 1, 1),
                output_shapes=[(nq * threads * k,), (nq * threads * k,)], output_dtypes=[mx.float32, mx.int32])
            local = _finish(vals, idx, nq, k)
        return self._to_ids(local)

    def search_list_major(self, xq: np.ndarray, k: int, nprobe: int, select: str = "auto") -> np.ndarray:
        """The same search with the list-major scan, grouped by list on the GPU. `select` "auto" uses
        select_k up to nprobe 16 and argpartition above: for batches of thousands select_k is faster at
        small nprobe (938,000 against 581,000 queries per second at nprobe 8 on SIFT), argpartition from 32."""
        if select == "auto":
            select = "select_k" if nprobe <= 16 else "argpartition"
        q = mx.array(xq)
        nq, d = q.shape
        nlist = self.offsets.shape[0] - 1
        dist = coarse_dist(q, self.centroids, self.cnorms, self.probe_metric)
        probes = probe_lists(dist, nprobe, select).reshape(-1)   # pair id = query * nprobe + slot
        pair_ids = mx.argsort(probes).astype(mx.int32)
        counts = mx.zeros((nlist,), dtype=mx.int32).at[probes].add(1)
        first = (mx.cumsum(counts) - counts).astype(mx.int32)
        lanes = 16
        vals, idx = LIST_SCANS[self.metric](
            inputs=[q, self.x, self.offsets, pair_ids, first, counts, sz(nprobe)],
            template=[("D", d), ("K", k)],
            grid=(256 * nlist, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(nq * nprobe * lanes * k,), (nq * nprobe * lanes * k,)], output_dtypes=[mx.float32, mx.int32])
        pick = select_k(vals.reshape(nq, -1), k, threads=64)
        # pick is -1 where fewer than k candidates were found; take_along_axis would read it as the last column
        local = mx.where(pick >= 0, mx.take_along_axis(idx.reshape(nq, -1), mx.maximum(pick, 0), axis=1), -1)
        return self._to_ids(local)

    def _to_ids(self, local: mx.array) -> np.ndarray:
        """Positions in the lists to base ids; -1 (fewer than k found) stays -1."""
        result = mx.where(local >= 0, mx.take(self.ids, mx.maximum(local, 0)), -1)
        mx.eval(result)
        return np.array(result)
