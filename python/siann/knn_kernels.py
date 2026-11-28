"""GPU kernels for exact k-nearest neighbors through mx.fast.metal_kernel.

Selection is two-stage. Each of T threads per query keeps the best k of a strided slice of the
row in registers (neighboring threads read neighboring elements); a small argpartition over the
T * k survivors then gives the final k.

L1 and L-infinity (the largest coordinate difference) share their kernels: each distance is built up
one coordinate pair (q, x) at a time by `combine` (COMBINE, in siann._metal, with the kernels' bodies), a sum of |q - x| for L1 and a
running maximum of it for L-infinity. Exact L2 and inner product are matrix multiplies instead, but
the IVF scans use `combine` for every metric, inner product included (a running sum of -q x, so that
smaller is nearer, as for the distances).
"""
import mlx.core as mx

from ._metal import _INSERT, _KEEP_BEST, _WRITE, COMBINE, SELECT, TILED, Kernel  # noqa: F401 (re-exported)


def mlx_kernel(kernel: Kernel, header: str = "", suffix: str = ""):
    """An MLX kernel from a shared body; its run-time sizes go in the last input, ``sz``."""
    return mx.fast.metal_kernel(name=kernel.name + suffix, input_names=[n for n, _ in kernel.inputs] + ["sz"],
                                output_names=[n for n, _ in kernel.outputs], source=kernel.mlx_source(),
                                header=header)


def sz(*sizes: int) -> mx.array:
    """The run-time sizes a kernel reads from ``sz``, in its order."""
    return mx.array(sizes, dtype=mx.uint32)


_select = mlx_kernel(SELECT)
_tiled = {m: mlx_kernel(TILED, COMBINE[m], f"_{m}") for m in ("l1", "linf")}


def dist(q: mx.array, x: mx.array, metric: str) -> mx.array:
    """The L1 or L-infinity ("linf") distance matrix between the rows of q and of x, tiled."""
    nq, d = q.shape
    n = x.shape[0]
    groups = ((n + 63) // 64, (nq + 63) // 64)
    return _tiled[metric](inputs=[q, x, sz(n, nq)], template=[("D", d)],
                          grid=(groups[0] * 16, groups[1] * 16, 1), threadgroup=(16, 16, 1),
                          output_shapes=[(nq, n)], output_dtypes=[mx.float32])[0]


def knn_tiled(q: mx.array, x: mx.array, k: int, metric: str, threads: int = 512) -> mx.array:
    """The k nearest rows of x to each row of q in L1 or L-infinity distance: the tiled distance
    matrix, then select_k."""
    return select_k(dist(q, x, metric), k, threads)


def _finish(vals: mx.array, idx: mx.array, q: int, k: int) -> mx.array:
    vals, idx = vals.reshape(q, -1), idx.reshape(q, -1)
    part = mx.argpartition(vals, kth=k - 1, axis=1)[:, :k]
    order = mx.argsort(mx.take_along_axis(vals, part, axis=1), axis=1)
    return mx.take_along_axis(idx, mx.take_along_axis(part, order, axis=1), axis=1)


def select_k(dist: mx.array, k: int, threads: int = 512) -> mx.array:
    """The column indices of the k smallest entries of each row, in increasing order."""
    q, n = dist.shape
    vals, idx = _select(inputs=[dist, sz(n, q)], template=[("T", threads), ("K", k)],
                        grid=(threads, q, 1), threadgroup=(256, 1, 1),
                        output_shapes=[(q * threads * k,), (q * threads * k,)], output_dtypes=[mx.float32, mx.int32])
    return _finish(vals, idx, q, k)


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
