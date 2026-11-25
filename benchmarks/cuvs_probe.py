"""What NVIDIA cuVS offers siann, measured on an NVIDIA GPU (Colab or Kaggle, a T4 or newer).

--capabilities: which metrics cuVS's brute force accepts, and which data types and metrics its IVF-Flat and
CAGRA accept, each checked against exact search in NumPy on small Gaussian data (a metric's names tried in
turn: cuVS spells L1 and L-infinity in several ways). A few seconds per case.

--cagra: a CAGRA graph built by cuVS on SIFT1M (ann-benchmarks' file, downloaded; 32 links per vector pruned
from 64, as siann's graphs), its own search's recall against ann-benchmarks' ground truth at a few search
widths, and the index saved in cuVS's file format (without the vectors) plus its links as a NumPy file, for
siann to read and to compare with its own graphs. A few minutes, most of it the download and the build.

Every line prints as it is measured, with its number out of the total and the minutes so far.

    python cuvs_probe.py --capabilities
    python cuvs_probe.py --cagra
"""
import argparse
import os
import time
import urllib.request

import numpy as np

START = time.perf_counter()
SIFT_URL = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"


def elapsed():
    return f"{(time.perf_counter() - START) / 60:.1f} min so far"


def distances(xb, xq, metric):
    """Every query's distance to every base vector (smaller is nearer: inner product and cosine negated)."""
    if metric == "l2":
        d = ((xq[:, None, :] - xb[None, :, :]) ** 2).sum(2)
    elif metric == "l1":
        d = np.abs(xq[:, None, :] - xb[None, :, :]).sum(2)
    elif metric == "linf":
        d = np.abs(xq[:, None, :] - xb[None, :, :]).max(2)
    elif metric == "ip":
        d = -(xq @ xb.T)
    else:                                                  # cosine
        d = -(xq / np.linalg.norm(xq, axis=1, keepdims=True)) @ (xb / np.linalg.norm(xb, axis=1, keepdims=True)).T
    return d


def recall_tied(found, dist, k):
    """recall@k up to ties: the share of found neighbors no farther than each query's k-th nearest (byte data
    has many tied distances, which any order may break)."""
    found = np.asarray(found)
    kth = np.sort(dist, axis=1)[:, k - 1:k]
    ok = (found >= 0) & (found < dist.shape[1])
    near = np.take_along_axis(dist, np.where(ok, found, 0), 1) <= kth + 1e-6 * np.abs(kth) + 1e-6
    return float(np.mean(ok & near))


def recall(found, truth):
    found = np.asarray(found)
    return float(np.mean([len(set(f) & set(t)) / len(t) for f, t in zip(found, truth, strict=True)]))


def capabilities() -> None:
    import cupy as cp
    import cuvs
    from cuvs.neighbors import brute_force, cagra, ivf_flat
    print(f"cuVS {getattr(cuvs, '__version__', '?')}, CuPy {cp.__version__}", flush=True)
    rng = np.random.default_rng(0)
    n, d, nq, k = 5000, 32, 50, 10
    xb = rng.standard_normal((n, d), dtype=np.float32)
    xq = rng.standard_normal((nq, d), dtype=np.float32)
    xbb = rng.integers(0, 256, (n, d)).astype(np.float32)   # integer data for the byte types
    xqb = rng.integers(0, 256, (nq, d)).astype(np.float32)

    # each metric's candidate names in cuVS
    names = {"l2": ("sqeuclidean", "euclidean", "l2"), "ip": ("inner_product",), "cosine": ("cosine",),
             "l1": ("l1", "cityblock", "manhattan"), "linf": ("chebyshev", "linf")}
    cases = ([("brute force", m, None) for m in names]
             + [("IVF-Flat", m, dt) for m in ("l2", "ip", "cosine", "l1", "linf")
                for dt in ("float32", "float16", "int8", "uint8")]
             + [("CAGRA", m, dt) for m in ("l2", "ip", "cosine") for dt in ("float32", "float16", "int8", "uint8")])
    print(f"  {len(cases)} cases; recall@{k} up to ties against NumPy on {nq} queries, {n:,} x {d} (integer "
          f"data for int8 and uint8); IVF-Flat probes every list, so it should be exact", flush=True)
    for i, (what, metric, dt) in enumerate(cases, 1):
        if dt in ("int8", "uint8"):
            base, qs = (xbb - 128 if dt == "int8" else xbb), (xqb - 128 if dt == "int8" else xqb)
        else:
            base, qs = xb, xq
        dist = distances(base, qs, metric)
        cast = {"float32": np.float32, "float16": np.float16, "int8": np.int8, "uint8": np.uint8, None: np.float32}[dt]
        b, q = cp.asarray(base.astype(cast)), cp.asarray(qs.astype(cast))
        label = f"[{i}/{len(cases)}] {what}, {metric}" + (f", {dt}" if dt else "")
        results = []
        for name in names[metric]:
            try:
                if what == "brute force":
                    index = brute_force.build(b, metric=name)
                    _, found = brute_force.search(index, q, k)
                elif what == "IVF-Flat":
                    index = ivf_flat.build(ivf_flat.IndexParams(n_lists=16, metric=name), b)
                    _, found = ivf_flat.search(ivf_flat.SearchParams(n_probes=16), index, q, k)   # every list
                else:
                    index = cagra.build(cagra.IndexParams(metric=name, graph_degree=32,
                                                          intermediate_graph_degree=64), b)
                    _, found = cagra.search(cagra.SearchParams(itopk_size=128), index, q, k)
                results.append(f"'{name}': recall {recall_tied(cp.asnumpy(found), dist, k):.4f}")
                break                                       # the first name that works is enough
            except Exception as e:  # noqa: BLE001 (not supported under this name: try the next, then report)
                results.append(f"'{name}': {type(e).__name__}: {str(e).splitlines()[0][:90] if str(e) else ''}")
        print(f"  {label:34s} " + "; ".join(results) + f"   ({elapsed()})", flush=True)


def load_sift(path: str):
    import h5py
    if not os.path.exists(path):
        print(f"  downloading {SIFT_URL} (about 500 MB)", flush=True)
        t = time.perf_counter()
        urllib.request.urlretrieve(SIFT_URL, path)
        print(f"  downloaded in {time.perf_counter() - t:.0f} s ({elapsed()})", flush=True)
    with h5py.File(path, "r") as f:
        return (np.asarray(f["train"], dtype=np.float32), np.asarray(f["test"], dtype=np.float32),
                np.asarray(f["neighbors"])[:, :10])


def cagra_export(path: str, out: str) -> None:
    import cupy as cp
    import cuvs
    from cuvs.neighbors import cagra
    steps = 5
    print(f"cuVS {getattr(cuvs, '__version__', '?')}: CAGRA on SIFT1M, {steps} steps", flush=True)
    xb, xq, gt = load_sift(path)
    print(f"  [1/{steps}] SIFT1M loaded: {xb.shape[0]:,} x {xb.shape[1]}, {len(xq):,} queries ({elapsed()})", flush=True)
    b, q = cp.asarray(xb), cp.asarray(xq)
    params = cagra.IndexParams(metric="sqeuclidean", graph_degree=32, intermediate_graph_degree=64)
    t = time.perf_counter()
    index = cagra.build(params, b)
    cp.cuda.Device().synchronize()
    build = time.perf_counter() - t
    print(f"  [2/{steps}] built: {build:.1f} s (graph degree 32, intermediate 64, cuVS's default build "
          f"algorithm) ({elapsed()})", flush=True)
    print(f"  [3/{steps}] cuVS's own search, 10,000 queries, recall@10 against ann-benchmarks' ground truth:", flush=True)
    for width in (32, 64, 128, 256):
        sp = cagra.SearchParams(itopk_size=width)
        cagra.search(sp, index, q, 10)                     # warm-up
        cp.cuda.Device().synchronize()
        t = time.perf_counter()
        _, found = cagra.search(sp, index, q, 10)
        cp.cuda.Device().synchronize()
        sec = time.perf_counter() - t
        print(f"      itopk_size {width:>4}: recall {recall(cp.asnumpy(found), gt):.4f}, "
              f"{len(xq) / sec:,.0f} queries/s ({elapsed()})", flush=True)
    os.makedirs(out, exist_ok=True)
    fname = os.path.join(out, "cagra_sift.cuvs")
    cagra.save(fname, index, include_dataset=False)
    print(f"  [4/{steps}] saved {fname}: {os.path.getsize(fname):,} bytes, cuVS's format without the vectors "
          f"({elapsed()})", flush=True)
    graph = getattr(index, "graph", None)
    if graph is None:
        print(f"  [5/{steps}] this cuVS version gives no access to the links (index.graph): only the cuVS file",
              flush=True)
    else:
        links = cp.asnumpy(cp.asarray(graph)).astype(np.int32)
        np.save(os.path.join(out, "cagra_sift_links.npy"), links)
        print(f"  [5/{steps}] saved {os.path.join(out, 'cagra_sift_links.npy')}: links {links.shape}, "
              f"{links.dtype} ({elapsed()})", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--capabilities", action="store_true")
    p.add_argument("--cagra", action="store_true")
    p.add_argument("--sift", default="sift-128-euclidean.hdf5", help="where SIFT1M is, or is downloaded to")
    p.add_argument("--out", default=".", help="where the CAGRA files are saved")
    a = p.parse_args()
    if a.capabilities:
        capabilities()
    if a.cagra:
        cagra_export(a.sift, a.out)


if __name__ == "__main__":
    main()
