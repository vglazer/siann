"""siann's GPU paths against mlxvs (NVIDIA cuVS's methods on MLX, https://github.com/mikepapadim/mlxvs),
the strongest GPU alternative on the Mac, on the same machine, data and queries.

- Exact search: siann's GPU kernels (siann.exact.mlx_knn) against mlxvs.brute_force_knn.
- IVF: siann's GPU IVF (its best list storage for the data: uint8 where lossless, else float16) against
  mlxvs's IvfFlatIndex, both with --nlist lists, at the same nprobe.
- Graphs: mlxvs's CagraIndex (graph degree 32 pruned from 64, as in mlxvs's own benchmarks), float32 and
  float16, over a sweep of itopk_size.

Queries per second: single queries (one call each, one pass over --small queries), batches of 100 over
--small queries, and one batch of --queries (median of three after a warm-up; siann's IVF runs its
query-major scan below 1,000 queries and its list-major scan above). Recall@10 against exact search
(the Rust flat index), ties counted as correct (backends.agreement). Results converted to NumPy inside
the timing, for both libraries.

    uv run --group bench python mlxvs_compare.py --dataset sift
    uv run --group bench python mlxvs_compare.py --dataset sift --base 100000 --queries 1000 --small 100   # smoke test
"""
import argparse
import time

from siann._cpu import FlatIndex
from siann.exact import mlx_knn
from siann.ivf_gpu import GpuIVF

import mlx.core as mx
import mlxvs
import numpy as np
from backends import agreement
from datasets import load
from timing import throughput

K = 10
METRIC = {"sift": "l2", "glove": "l2", "music100": "ip", "dbpedia": "l2"}   # glove, dbpedia normalized: l2 ranks as cosine
MLXVS_METRIC = {"l2": "l2", "ip": "inner_product"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sift", choices=sorted(METRIC))
    p.add_argument("--base", type=int, default=0, help="use only the first N base vectors (0: all)")
    p.add_argument("--queries", type=int, default=10_000)
    p.add_argument("--small", type=int, default=1000)
    p.add_argument("--nlist", type=int, default=4000)
    p.add_argument("--nprobes", default="16,32,64,128")
    p.add_argument("--itopk", default="32,64,128,256")
    p.add_argument("--skip", default="", help="comma-separated parts to skip: exact, ivf, cagra")
    a = p.parse_args()
    skip = set(filter(None, a.skip.split(",")))
    metric = METRIC[a.dataset]
    xb, xq = load(a.dataset, a.queries)
    if a.base:
        xb = np.ascontiguousarray(xb[:a.base])
    small = xq[:a.small]
    n, d = xb.shape
    truth = np.array(FlatIndex(xb, d, metric).search(xq, K), dtype=np.int64).reshape(-1, K)
    integer = bool(np.array_equal(np.rint(xb), xb) and xb.min() >= 0 and xb.max() <= 255)
    storage = "uint8" if integer else "float16"
    print(f"{a.dataset}, {metric}: {n:,} vectors, {d} dimensions, {len(xq):,} queries; mlxvs "
          f"{getattr(mlxvs, '__version__', '?')}, mlx {mx.__version__}; queries per second", flush=True)
    columns = f"  {'method':44s} {'recall@10':>9s} {'single':>9s} {'100':>9s} {f'{len(xq):,}':>9s}"

    def header(title):
        print(f"\n{title}\n{columns}", flush=True)

    def row(label, search, batch_fn=None):
        q1, _ = throughput(search, small, 1, repeats=1)
        q100, _ = throughput(search, small, 100)
        qf, found = throughput(batch_fn or search, xq, len(xq))
        r = agreement(xb, xq, found, truth, metric)
        print(f"  {label:44s} {r:9.3f} {q1:9,.0f} {q100:9,.0f} {qf:9,.0f}", flush=True)

    if "exact" not in skip:
        header("Exact search")
        xb_mx = mx.array(xb)
        mx.eval(xb_mx)
        row("siann exact (GPU kernels)", lambda q: mlx_knn(xb_mx, q, K, metric))
        row("mlxvs brute_force_knn",
            lambda q: np.asarray(mlxvs.brute_force_knn(xb_mx, mx.array(q), K, MLXVS_METRIC[metric])[0]))

    if "ivf" not in skip:
        t = time.perf_counter()
        ivf = GpuIVF(xb, a.nlist, metric, storage)
        t_siann = time.perf_counter() - t
        t = time.perf_counter()
        vs_ivf = mlxvs.IvfFlatIndex.build(xb, n_lists=a.nlist, metric=MLXVS_METRIC[metric])
        t_vs = time.perf_counter() - t
        header(f"IVF, {a.nlist} lists: siann built in {t_siann:.1f} s ({storage} lists), mlxvs IVF-Flat in {t_vs:.1f} s")
        for nprobe in map(int, a.nprobes.split(",")):
            row(f"siann GPU IVF ({storage}), nprobe {nprobe}",
                lambda q, n=nprobe: ivf.search(q, K, n) if len(q) < 1000 else ivf.search_list_major(q, K, n))
            row(f"mlxvs IVF-Flat, n_probes {nprobe}",
                lambda q, n=nprobe: np.asarray(vs_ivf.search(mx.array(q), k=K, n_probes=n)[0]))

    if "cagra" not in skip:
        for dtype in ("float32", "float16"):
            t = time.perf_counter()
            cg = mlxvs.CagraIndex.build(xb, graph_degree=32, metric=MLXVS_METRIC[metric], dtype=dtype)
            header(f"mlxvs CAGRA ({dtype}), graph degree 32: built in {time.perf_counter() - t:.1f} s")
            for it in map(int, a.itopk.split(",")):
                row(f"mlxvs CAGRA ({dtype}), itopk_size {it}",
                    lambda q, i=it: np.asarray(cg.search(mx.array(q), k=K, itopk_size=i)[0]))
            del cg


if __name__ == "__main__":
    main()
