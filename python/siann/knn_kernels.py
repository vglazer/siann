"""GPU kernels for exact k-nearest neighbors through mx.fast.metal_kernel.

Selection is two-stage. Each of T threads per query keeps the best k of a strided slice of the
row in registers (neighbouring threads read neighbouring elements); a small argpartition over the
T * k survivors then gives the final k. For L1 the distance is computed inside the same kernel, so
no distance matrix is ever written.

L1 and L-infinity (the largest coordinate difference) share their kernels: each distance is built up
one coordinate pair (q, x) at a time by `combine` (COMBINE below), a sum of |q - x| for L1 and a
running maximum of it for L-infinity. Exact L2 and inner product are matrix multiplies instead, but
the IVF scans use `combine` for every metric, inner product included (a running sum of -q x, so that
smaller is nearer, as for the distances).
"""
import mlx.core as mx

# How each metric folds one coordinate pair (q, x) into a running distance a, as a Metal header.
# Inner product ("ip") is negated, so that, as for the distances, smaller is nearer.
COMBINE = {
    "l1": "inline float combine(float a, float q, float x) { return a + metal::abs(q - x); }\n",
    "l2": "inline float combine(float a, float q, float x) { float e = q - x; return a + e * e; }\n",
    "linf": "inline float combine(float a, float q, float x) { return metal::max(a, metal::abs(q - x)); }\n",
    "ip": "inline float combine(float a, float q, float x) { return a - q * x; }\n",
}

_KEEP_BEST = """
    float best[K];
    int where[K];
    for (int a = 0; a < K; ++a) { best[a] = INFINITY; where[a] = -1; }
"""
_INSERT = """
        if (v < best[K - 1]) {
            int p = K - 1;
            while (p > 0 && best[p - 1] > v) { best[p] = best[p - 1]; where[p] = where[p - 1]; --p; }
            best[p] = v;
            where[p] = int(j);
        }
"""
_WRITE = """
    for (int a = 0; a < K; ++a) {
        out_val[(i * T + t) * K + a] = best[a];
        out_idx[(i * T + t) * K + a] = where[a];
    }
"""

_select = mx.fast.metal_kernel(
    name="select_k", input_names=["dist"], output_names=["out_val", "out_idx"],
    source="""
    uint t = thread_position_in_grid.x;
    uint i = thread_position_in_grid.y;
    if (t >= T || i >= Q) return;
""" + _KEEP_BEST + """
    for (uint j = t; j < N; j += T) {
        float v = dist[i * N + j];
""" + _INSERT + """
    }
""" + _WRITE)

_FUSED_SELECT = """
    uint t = thread_position_in_grid.x;
    uint i = thread_position_in_grid.y;
    if (t >= T || i >= Q) return;
""" + _KEEP_BEST + """
    for (uint j = t; j < N; j += T) {
        float v = 0.0f;
        for (uint d = 0; d < D; ++d) { v = combine(v, q[i * D + d], x[j * D + d]); }
""" + _INSERT + """
    }
""" + _WRITE
_fused_select = {m: mx.fast.metal_kernel(name=f"{m}_select_k", input_names=["q", "x"],
                                         output_names=["out_val", "out_idx"], source=_FUSED_SELECT,
                                         header=COMBINE[m])
                 for m in ("l1", "linf")}


# A tiled L1 (or L-infinity) distance matrix, like a matrix multiply: each threadgroup of 16 x 16
# threads computes a 64 x 64 block of distances (4 x 4 per thread, in registers), staging 16 dimensions
# of 64 queries and 64 base vectors at a time in threadgroup memory, so each base vector is read once
# per 64 queries. Padding is zero in both, a difference of 0 that changes neither metric.
_TILED = """
    const uint B = 64, S = 16, R = 4;
    threadgroup float qs[B][S + 1];
    threadgroup float xs[B][S + 1];
    uint tx = thread_position_in_threadgroup.x, ty = thread_position_in_threadgroup.y;
    uint col0 = threadgroup_position_in_grid.x * B, row0 = threadgroup_position_in_grid.y * B;
    uint lane = ty * 16 + tx;
    float acc[R][R];
    for (uint a = 0; a < R; ++a) for (uint b = 0; b < R; ++b) acc[a][b] = 0.0f;
    for (uint d0 = 0; d0 < D; d0 += S) {
        for (uint e = lane; e < B * S; e += 256) {
            uint r = e / S, c = e % S;
            uint qi = row0 + r, xi = col0 + r, dd = d0 + c;
            qs[r][c] = (qi < Q && dd < D) ? q[qi * D + dd] : 0.0f;
            xs[r][c] = (xi < N && dd < D) ? x[xi * D + dd] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = 0; c < S; ++c) {
            float qv[R], xv[R];
            for (uint a = 0; a < R; ++a) { qv[a] = qs[ty * R + a][c]; xv[a] = xs[tx * R + a][c]; }
            for (uint a = 0; a < R; ++a) for (uint b = 0; b < R; ++b) acc[a][b] = combine(acc[a][b], qv[a], xv[b]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint a = 0; a < R; ++a) for (uint b = 0; b < R; ++b) {
        uint qi = row0 + ty * R + a, xi = col0 + tx * R + b;
        if (qi < Q && xi < N) out[qi * N + xi] = acc[a][b];
    }
"""
_tiled = {m: mx.fast.metal_kernel(name=f"{m}_tiled", input_names=["q", "x"], output_names=["out"],
                                  source=_TILED, header=COMBINE[m])
          for m in ("l1", "linf")}


def dist(q: mx.array, x: mx.array, metric: str) -> mx.array:
    """The L1 or L-infinity ("linf") distance matrix between the rows of q and of x, tiled."""
    nq, d = q.shape
    n = x.shape[0]
    groups = ((n + 63) // 64, (nq + 63) // 64)
    return _tiled[metric](inputs=[q, x], template=[("N", n), ("Q", nq), ("D", d)],
                          grid=(groups[0] * 16, groups[1] * 16, 1), threadgroup=(16, 16, 1),
                          output_shapes=[(nq, n)], output_dtypes=[mx.float32])[0]


def knn_tiled(q: mx.array, x: mx.array, k: int, metric: str, threads: int = 512) -> mx.array:
    """The k nearest rows of x to each row of q in L1 or L-infinity distance: the tiled distance
    matrix, then select_k."""
    return select_k(dist(q, x, metric), k, threads)


def l1_dist(q: mx.array, x: mx.array) -> mx.array:
    """The L1 distance matrix between the rows of q and of x, tiled."""
    return dist(q, x, "l1")


def l1_knn_tiled(q: mx.array, x: mx.array, k: int, threads: int = 512) -> mx.array:
    """The k nearest rows of x to each row of q in L1 distance: the tiled distance matrix, then select_k."""
    return knn_tiled(q, x, k, "l1", threads)


def _finish(vals: mx.array, idx: mx.array, q: int, k: int) -> mx.array:
    vals, idx = vals.reshape(q, -1), idx.reshape(q, -1)
    part = mx.argpartition(vals, kth=k - 1, axis=1)[:, :k]
    order = mx.argsort(mx.take_along_axis(vals, part, axis=1), axis=1)
    return mx.take_along_axis(idx, mx.take_along_axis(part, order, axis=1), axis=1)


def select_k(dist: mx.array, k: int, threads: int = 512) -> mx.array:
    """The column indices of the k smallest entries of each row, in increasing order."""
    q, n = dist.shape
    vals, idx = _select(inputs=[dist], template=[("N", n), ("Q", q), ("T", threads), ("K", k)],
                        grid=(threads, q, 1), threadgroup=(256, 1, 1),
                        output_shapes=[(q * threads * k,), (q * threads * k,)], output_dtypes=[mx.float32, mx.int32])
    return _finish(vals, idx, q, k)


def knn_fused(q: mx.array, x: mx.array, k: int, metric: str, threads: int = 256) -> mx.array:
    """The k nearest rows of x to each row of q in L1 or L-infinity distance, fused: no distance matrix."""
    nq, d = q.shape
    n = x.shape[0]
    vals, idx = _fused_select[metric](inputs=[q, x], template=[("N", n), ("Q", nq), ("D", d), ("T", threads), ("K", k)],
                           grid=(threads, nq, 1), threadgroup=(256, 1, 1),
                           output_shapes=[(nq * threads * k,), (nq * threads * k,)], output_dtypes=[mx.float32, mx.int32])
    return _finish(vals, idx, nq, k)


def l1_knn(q: mx.array, x: mx.array, k: int, threads: int = 256) -> mx.array:
    """The k nearest rows of x to each row of q in L1 distance, fused: no distance matrix."""
    return knn_fused(q, x, k, "l1", threads)


def ip_knn(q: mx.array, x: mx.array, k: int, threads: int = 512) -> mx.array:
    """The k rows of x with the largest inner product with each row of q, largest first: the negated
    products from one fused multiply (addmm with alpha -1 and a zero row), then select_k."""
    zero = mx.zeros((1, x.shape[0]), dtype=mx.float32)
    return select_k(mx.addmm(zero, q, x.T, alpha=-1.0), k, threads)


def l2_knn(q: mx.array, x: mx.array, norms: mx.array, k: int, threads: int = 512) -> mx.array:
    """The k nearest rows of x to each row of q in L2 distance: a matrix multiply, then select_k.
    ``norms`` holds the squared norms of the rows of x."""
    # addmm fuses the multiply, scaling and addition: 6.9 ms against 17.8 ms for 256 x 1M.
    return select_k(mx.addmm(norms[None, :], q, x.T, alpha=-2.0), k, threads)
