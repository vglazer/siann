"""Datasets in ../data: from ann-benchmarks, SIFT1M (sift) and GloVe-100 (glove, cosine distance,
served normalized so that L2 ranks like cosine; glove-raw, the same vectors unnormalized, for inner
product, where their lengths, 0.04 to 18.7, matter); and Music-100 (music100, inner product), from
Morozov and Babenko's ip-nsw, in data/music-100/ as raw float32."""
from pathlib import Path

import h5py
import numpy as np

FILES = {"sift": "sift-128-euclidean.hdf5", "glove": "glove-100-angular.hdf5",
         "glove-raw": "glove-100-angular.hdf5"}


DATA = Path(__file__).resolve().parent.parent / "data"


def load(name: str, queries: int) -> tuple[np.ndarray, np.ndarray]:
    if name == "music100":
        xb = np.fromfile(DATA / "music-100" / "database_music100.bin", dtype=np.float32).reshape(-1, 100)
        xq = np.fromfile(DATA / "music-100" / "query_music100.bin", dtype=np.float32).reshape(-1, 100)
        return xb, np.ascontiguousarray(xq[:queries])
    f = h5py.File(DATA / FILES[name], "r")
    xb, xq = f["train"][:].astype(np.float32), f["test"][:queries].astype(np.float32)
    if name == "glove":
        xb /= np.linalg.norm(xb, axis=1, keepdims=True) + 1e-12
        xq /= np.linalg.norm(xq, axis=1, keepdims=True) + 1e-12
    return xb, xq
