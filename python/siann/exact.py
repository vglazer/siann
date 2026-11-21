"""Exact k nearest neighbors on the GPU (MLX), a tile of queries at a time."""
import mlx.core as mx
import numpy as np

from .knn_kernels import ip_knn, knn_tiled, l2_knn


def knn(q: mx.array, x: mx.array, norms: mx.array | None, k: int, metric: str) -> mx.array:
    """The k nearest rows of x to each row of q (lazy): a matrix multiply for L2 (``norms``, the rows'
    squared norms) and inner product, the tiled kernel for L1 and L-infinity; then siann's top-k."""
    if metric == "l2":
        return l2_knn(q, x, norms, k)
    if metric == "ip":
        return ip_knn(q, x, k)
    return knn_tiled(q, x, k, metric)


def mlx_knn(xb: mx.array, xq: np.ndarray, k: int, metric: str, tile: int = 256) -> np.ndarray:
    """Exact k nearest neighbours on the GPU, a tile of queries at a time, each tile evaluated before
    the next so that only one tile's work is held in memory."""
    norms = mx.sum(xb * xb, axis=1) if metric == "l2" else None
    out = []
    for start in range(0, len(xq), tile):
        result = knn(mx.array(xq[start:start + tile]), xb, norms, k, metric)
        mx.eval(result)
        out.append(np.array(result))
    return np.concatenate(out)
