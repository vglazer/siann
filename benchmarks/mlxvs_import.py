"""An IVF built by mlxvs, searched by siann: mlxvs's IVF-Flat (4,000 lists, its own k-means) built on a
dataset, its centroids and lists written into a siann index file, loaded with siann.Index.load (routes
re-measured, as for a file from another machine) and calibrated with index.calibrate on 1,000 of the
dataset's queries; recall measured on the other 9,000. Against mlxvs's own search of the same lists at the
same nprobe, and against an index siann builds itself on the same data with the same target and queries.
GloVe is the normalized set, searched by L2 (cosine's ranking).

    uv run --group bench python mlxvs_import.py --dataset sift
"""
import argparse
import json
import os
import tempfile
import time

import mlx.core as mx
import mlxvs
import numpy as np
from backends import agreement
from datasets import load

import siann
from siann import _index
from siann._cpu import FlatIndex
from siann._ivf import default_storage

K = 10
MLXVS_MAX_PROBES = 256                  # mlxvs's IVF-Flat search refuses more ("k must be in [1, 256]")
BATCHES = (1, 100, 10_000)


def write_siann_file(path, x, centers, indices, offsets, recall, cal_q, builder):
    """A siann index file (format 2) holding an IVF whose lists another library built: the vectors, its
    centroids and each vector's list, the calibration queries; nprobe and routes left for the loading
    machine (Index.load re-measures routes for a file from elsewhere; index.calibrate sets nprobe)."""
    n = len(x)
    nlist = len(centers)
    assign = np.empty(n, dtype=np.int32)
    for c in range(nlist):
        assign[indices[offsets[c]:offsets[c + 1]]] = c
    sizes = np.diff(offsets)
    storage = default_storage(x)
    meta = {
        "format": _index.FORMAT, "format_version": _index.FORMAT_VERSION, "siann": siann.__version__,
        "saved": time.strftime("%Y-%m-%d"), "machine": f"built by {builder}", "metric": "l2", "recall": recall,
        "kind": "ivf", "nprobe": 1, "storage": storage, "ivf_metric": "l2", "routes": {},
        "profile": siann.DEFAULT_PROFILE.to_dict(),
        "build_reasons": [f"approximate search: an IVF of {nlist:,} lists built by {builder}, imported",
                          _index.Index._storage_line(storage, "gpu"),
                          "calibrated: nprobe 1 (not yet calibrated) on the given calibration queries"],
        "calibration": "given", "clustered": [n, float(sizes.max() / sizes.mean())], "next_id": n,
        "build_args": {"kind": "ivf", "nlist": None, "storage": "auto"}, "tested_n": n}
    arrays = {"meta": np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
              "x": x.astype(np.uint8) if storage == "uint8" else x,
              "centroids": np.ascontiguousarray(centers, dtype=np.float32), "assign": assign,
              "calibration_queries": cal_q}
    with open(path, "wb") as f:
        np.savez(f, **arrays)


def rate(fn, q, b):
    """Queries per second searching b queries per call (best of three passes over up to 1,000 queries,
    or one batch of 10,000)."""
    q = q[:max(b, min(len(q), 100 * b))]

    def run():
        for s in range(0, len(q), b):
            fn(q[s:s + b])
    run()
    best = float("inf")
    for _ in range(3):
        t = time.perf_counter()
        run()
        best = min(best, time.perf_counter() - t)
    return len(q) / best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sift", choices=["sift", "glove"])
    p.add_argument("--nlist", type=int, default=4000)
    p.add_argument("--nprobes", default="16,32,64")
    a = p.parse_args()
    xb, xq = load(a.dataset, 10_000)
    xb, xq = np.ascontiguousarray(xb), np.ascontiguousarray(xq)
    cal, test = xq[:1000], xq[1000:]
    truth = np.array(FlatIndex(xb, xb.shape[1], "l2").search(test, K), dtype=np.int64).reshape(-1, K)
    print(f"{a.dataset}, L2: {len(xb):,} vectors; mlxvs {getattr(mlxvs, '__version__', '?')} IVF-Flat, "
          f"{a.nlist:,} lists; recall@10 on {len(test):,} queries; queries per second")

    t = time.perf_counter()
    vs = mlxvs.IvfFlatIndex.build(xb, n_lists=a.nlist, metric="l2")
    mx.eval(vs._centers, vs._indices, vs._offsets)
    print(f"  mlxvs built it in {time.perf_counter() - t:.1f} s")
    path = os.path.join(tempfile.mkdtemp(), "imported.siann")
    write_siann_file(path, xb, np.array(vs._centers), np.array(vs._indices).astype(np.int64),
                     np.array(vs._offsets).astype(np.int64), 0.95, cal, f"mlxvs {getattr(mlxvs, '__version__', '')}")
    t = time.perf_counter()
    imported = siann.Index.load(path)
    print(f"  siann loaded it in {time.perf_counter() - t:.1f} s (routes re-measured)")

    print("\n  the same lists at the same nprobe: recall, mlxvs's search / siann's; siann's CPU and GPU "
          "(ids identical, and scores identical: where the ids differ but not the scores, ties at the same\n"
          "  distance; where the scores differ, the GPU's float16 lists)")
    for nprobe in map(int, a.nprobes.split(",")):
        ids_vs = np.array(vs.search(test, K, n_probes=nprobe)[0]).astype(np.int64)
        g = imported.search(test, K, device="gpu", nprobe=nprobe)
        c = imported.search(test, K, device="cpu", nprobe=nprobe)
        gpu, cpu = g.indices, c.indices
        print(f"    nprobe {nprobe:3d}: {agreement(xb, test, ids_vs, truth, 'l2'):.3f} / "
              f"{agreement(xb, test, gpu, truth, 'l2'):.3f}; CPU and GPU: ids identical "
              f"{(cpu == gpu).mean():.4f}, scores identical {(c.scores == g.scores).mean():.4f}", flush=True)

    t = time.perf_counter()
    imported.calibrate(cal)
    print(f"\n  calibrated on 1,000 queries in {time.perf_counter() - t:.1f} s: nprobe {imported.nprobe}")
    t = time.perf_counter()
    own = siann.Index(xb, "l2", recall=0.95, calibration_queries=cal)
    print(f"  siann's own index built in {time.perf_counter() - t:.1f} s: {own._ivf.nlist:,} lists, "
          f"nprobe {own.nprobe}")
    print(f"\n  {'':28s} {'recall@10':>9s}" + "".join(f"{f'batch {b:,}':>13s}" for b in BATCHES))
    nv = min(imported.nprobe, MLXVS_MAX_PROBES)
    rows = (("mlxvs's lists, in siann", lambda q: imported.search(q, K), imported.search(test, K).indices),
            ("siann's own lists", lambda q: own.search(q, K), own.search(test, K).indices),
            (f"mlxvs's search, nprobe {nv}" + (" (its most)" if nv < imported.nprobe else ""),
             lambda q: mx.eval(vs.search(q, K, n_probes=nv)[0]),
             np.array(vs.search(test, K, n_probes=nv)[0]).astype(np.int64)))
    for label, fn, found in rows:
        print(f"  {label:28s} {agreement(xb, test, found, truth, 'l2'):9.3f}"
              + "".join(f"{rate(fn, test, b):13,.0f}" for b in BATCHES), flush=True)
    print(f"\n{imported.plan}")


if __name__ == "__main__":
    main()
