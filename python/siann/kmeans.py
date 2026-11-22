"""k-means for the IVF's centroids, on the GPU through siann's exact kernels where there is one,
otherwise on the CPU through BLAS: Lloyd's iterations from k randomly chosen vectors, in L2 (the
centroids only need to spread over the data, whatever the search metric). An empty cluster takes half of
a large one, as in FAISS: the large cluster's centroid is copied and the two are nudged apart."""
from __future__ import annotations

import numpy as np

from .ivf_build import nearest


def kmeans(x: np.ndarray, k: int, niter: int = 10, seed: int = 1, device: str = "auto") -> np.ndarray:
    """``k`` centroids (float32, k x d) of the rows of ``x`` by ``niter`` iterations of Lloyd's
    algorithm, starting from ``k`` rows chosen at random with ``seed``."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    n, d = x.shape
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= {n} (the rows given), not {k}")
    rng = np.random.default_rng(seed)
    c = x[rng.choice(n, size=k, replace=False)].copy()
    from ._cpu import centroid_means
    for _ in range(niter):
        assign = nearest(x, c, "l2", device)
        counts = centroid_means(x, np.ascontiguousarray(assign, dtype=np.int64), c)   # c updated in place
        # empty clusters: split a large one (chosen with probability growing with its size)
        for j in np.flatnonzero(counts == 0):
            weights = np.maximum(counts - 1, 0).astype(np.float64)
            i = rng.choice(k, p=weights / weights.sum())
            nudge = np.where(rng.random(d) < 0.5, 1.0, -1.0).astype(np.float32) / 1024
            c[j] = c[i] * (1 + nudge)
            c[i] = c[i] * (1 - nudge)
            counts[j] = counts[i] // 2
            counts[i] -= counts[j]
    return c
