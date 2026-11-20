"""ann-benchmarks datasets in ../data: SIFT1M (sift) and GloVe-100 (glove, cosine distance, served
normalized so that L2 ranks like cosine; glove-raw, the same vectors unnormalized, for inner product,
where their lengths, 0.04 to 18.7, matter)."""
from pathlib import Path

import h5py
import numpy as np

FILES = {"sift": "sift-128-euclidean.hdf5", "glove": "glove-100-angular.hdf5",
         "glove-raw": "glove-100-angular.hdf5"}


def load(name: str, queries: int) -> tuple[np.ndarray, np.ndarray]:
    f = h5py.File(Path(__file__).resolve().parent.parent / f"data/{FILES[name]}", "r")
    xb, xq = f["train"][:].astype(np.float32), f["test"][:queries].astype(np.float32)
    if name == "glove":
        xb /= np.linalg.norm(xb, axis=1, keepdims=True) + 1e-12
        xq /= np.linalg.norm(xq, axis=1, keepdims=True) + 1e-12
    return xb, xq
