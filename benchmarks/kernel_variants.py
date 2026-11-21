"""GPU kernel variants measured and not kept in siann, for the checks and benchmarks that cite them.

The fused L1 (and L-infinity) kernel computes each query's distances and selection in one pass, with no
distance matrix: it reads the whole base once per query and is limited by memory speed, at a third of
the tiled kernel's speed (ALGORITHMS.md, "What does not pay").
"""
import mlx.core as mx
from siann.knn_kernels import _INSERT, _KEEP_BEST, _WRITE, COMBINE, _finish

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


def knn_fused(q: mx.array, x: mx.array, k: int, metric: str, threads: int = 256) -> mx.array:
    """The k nearest rows of x to each row of q in L1 or L-infinity distance, fused: no distance matrix."""
    nq, d = q.shape
    n = x.shape[0]
    vals, idx = _fused_select[metric](inputs=[q, x], template=[("N", n), ("Q", nq), ("D", d), ("T", threads), ("K", k)],
                                      grid=(threads, nq, 1), threadgroup=(256, 1, 1),
                                      output_shapes=[(nq * threads * k,), (nq * threads * k,)],
                                      output_dtypes=[mx.float32, mx.int32])
    return _finish(vals, idx, nq, k)
