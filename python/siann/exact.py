"""Exact k nearest neighbors on the GPU (MLX), a tile of queries at a time."""
import mlx.core as mx
import numpy as np

from .knn_kernels import ip_knn, knn_tiled, l2_knn


def mlx_knn(xb: mx.array, xq: np.ndarray, k: int, metric: str, tile: int = 256) -> np.ndarray:
    """Exact k nearest neighbours on the GPU, a tile of queries at a time, each tile evaluated before
    the next so that only one tile's work is held in memory."""
    norms = mx.sum(xb * xb, axis=1) if metric == "l2" else None
    out = []
    for start in range(0, len(xq), tile):
        q = mx.array(xq[start:start + tile])
        if metric == "l2":
            result = l2_knn(q, xb, norms, k)
        elif metric == "ip":
            result = ip_knn(q, xb, k)
        else:
            result = knn_tiled(q, xb, k, metric)
        mx.eval(result)
        out.append(np.array(result))
    return np.concatenate(out)
