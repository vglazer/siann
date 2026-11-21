"""The IVF behind siann.Index when a recall target is given: lists built once (siann.ivf_build), each
device's copy made on first use, nprobe calibrated to the target on a sample of the index's own
vectors, and each search routed by its batch size."""
from __future__ import annotations

import time

import numpy as np

CALIBRATION_SAMPLE = 1000
CALIBRATION_K = 10


def default_nlist(n: int) -> int:
    """About 4 sqrt(n) lists (4,000 for a million vectors, as every measurement used), with at least 39
    vectors per list (k-means needs enough to place each centroid) and at least one list."""
    return int(max(1, min(round(4 * np.sqrt(n)), n // 39)))


def default_storage(x: np.ndarray) -> str:
    """uint8 where every coordinate is an integer from 0 to 255 (lossless), else float."""
    return "uint8" if (x.min() >= 0 and x.max() <= 255 and np.array_equal(np.rint(x), x)) else "float"


class IVF:
    """The lists and each device's copy of them. ``metric`` is the IVF's: "l1", "l2", "linf" or "ip"
    (cosine arrives as "l2" on unit vectors)."""

    def __init__(self, x: np.ndarray, metric: str, nlist: int, storage: str, device_for_build: str):
        from .ivf_build import build_lists
        self.x, self.metric, self.storage = x, metric, storage
        t = time.perf_counter()
        self.built = build_lists(x, nlist, metric, device=device_for_build)
        self.build_seconds = time.perf_counter() - t
        self._gpu = self._cpu = None

    @property
    def nlist(self) -> int:
        return self.built.nlist

    def gpu(self):
        if self._gpu is None:
            from .ivf_gpu import GpuIVF
            lists = "uint8" if self.storage == "uint8" else "float16"      # float16 for GPU batches
            self._gpu = GpuIVF(self.x, self.nlist, self.metric, lists, built=self.built)
        return self._gpu

    def cpu(self):
        if self._cpu is None:
            from ._cpu import IvfIndex
            d = self.x.shape[1]
            if self.storage == "uint8":
                self._cpu = IvfIndex(self.x, self.built.centroids, d, self.metric,
                                     np.ascontiguousarray(self.x.astype(np.uint8)), "uint8", lists=self.built.assign)
            else:
                self._cpu = IvfIndex(self.x, self.built.centroids, d, self.metric, lists=self.built.assign)
        return self._cpu

    def search(self, q: np.ndarray, k: int, nprobe: int, path: str) -> np.ndarray:
        """The k nearest by one path: "gpu-query", "gpu-list", "cpu-query", "cpu-list" or "cpu-split"
        (one query at a time, spread over 4 threads); -1 where fewer than k were found."""
        nprobe = min(nprobe, self.nlist)
        if path == "gpu-query":
            return np.asarray(self.gpu().search(q, k, nprobe), dtype=np.int64)
        if path == "gpu-list":
            return np.asarray(self.gpu().search_list_major(q, k, nprobe), dtype=np.int64)
        index = self.cpu()
        if path == "cpu-list":
            flat = index.search_list_major(q, k, nprobe)
        else:
            flat = index.search(q, k, nprobe, split=4 if path == "cpu-split" else 1)
        out = np.asarray(flat, dtype=np.int64).reshape(len(q), k)
        out[out >= len(self.x)] = -1                                     # the Rust IVF's "none found"
        return out


def recall_of(found: np.ndarray, truth_scores_kth: np.ndarray, found_scores: np.ndarray, larger_better: bool) -> float:
    """The share of found neighbors at least as near as the true k-th nearest (ties count as found)."""
    ok = found >= 0
    if larger_better:
        ok &= found_scores >= truth_scores_kth[:, None] - 1e-6 * np.abs(truth_scores_kth[:, None]) - 1e-9
    else:
        ok &= found_scores <= truth_scores_kth[:, None] + 1e-6 * np.abs(truth_scores_kth[:, None]) + 1e-9
    return float(ok.mean())


def calibrate(search, scores, truth_kth: np.ndarray, sample_ids: np.ndarray, target: float, nlist: int,
              larger_better: bool, k: int = CALIBRATION_K):
    """The smallest nprobe at which ``search(nprobe)`` reaches ``target`` recall@k on the sample: nprobe
    doubles until the target is reached, then a bisection narrows it. With ``sample_ids`` (the sample is
    the index's own vectors), the search returns k + 1 neighbors and each vector itself is left out.
    Returns (nprobe, recall at it)."""
    def recall_at(nprobe):
        found = search(nprobe)
        if sample_ids is not None:
            rows = []
            for r, own in zip(found, sample_ids):
                r = r[r != own][:k]                                      # leave the vector itself out
                rows.append(np.concatenate([r, np.full(k - len(r), -1)]))
            found = np.array(rows, dtype=np.int64)
        return recall_of(found, truth_kth, scores(found), larger_better)

    hi, r_hi = 1, recall_at(1)
    lo = 0
    while r_hi < target and hi < nlist:
        lo, hi = hi, min(2 * hi, nlist)
        r_hi = recall_at(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        r_mid = recall_at(mid)
        if r_mid >= target:
            hi, r_hi = mid, r_mid
        else:
            lo = mid
    return hi, r_hi
