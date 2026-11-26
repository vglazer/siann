"""siann's own CAGRA graph build against mlxvs's and cuVS's: the same 32 links per vector, each graph searched by
siann's own CPU search (siann._cpu.GraphIndex) at a range of search widths (ef), recall@10 against exact search
and queries per second on a batch of test queries (all cores, a thread per query), and the width and speed at
which each graph reaches the recall target. siann's graph: each vector's 64 nearest by siann's IVF searching the
base vectors (on the GPU), then CAGRA's optimization in Rust (siann._graph_build). mlxvs's: mlxvs.CagraIndex's
default build. cuVS's: the links a Kaggle T4 built (benchmarks/colab/cuvs_cagra.ipynb; SIFT1M only), read from
../cagra_sift_links.npy where present. Also the rough graph's own quality: the share of each vector's true 64
nearest it holds, on a sample.

    uv run --group bench python graph_build.py --dataset sift
    uv run --group bench python graph_build.py --dataset music100 --n 200000
"""
import argparse
import itertools
import time
from pathlib import Path

import numpy as np
from datasets import load

import siann
from siann._cpu import GraphIndex
from siann._graph_build import KNN_K, KNN_NPROBE, build_links, knn_graph

METRIC = {"sift": "l2", "glove": "l2", "music100": "ip", "dbpedia": "l2"}   # glove, dbpedia: normalized, so L2
EFS = (16, 24, 32, 48, 64, 96, 128, 192, 256)
K = 10
CUVS_LINKS = Path(__file__).resolve().parent.parent / "cagra_sift_links.npy"


def recall(found, truth):
    return float(np.mean([len(set(f) & set(t)) / len(t) for f, t in zip(found, truth, strict=True)]))


def curve(xb, links, metric, q, truth, stored):
    d = xb.shape[1]
    g = (GraphIndex(xb, np.ascontiguousarray(links, dtype=np.int64), d, metric, stored, "uint8") if stored is not None
         else GraphIndex(xb, np.ascontiguousarray(links, dtype=np.int64), d, metric))
    out = []
    for ef in EFS:
        g.search(q[:100], K, ef)                                       # warm-up
        t = time.perf_counter()
        found = np.asarray(g.search(q, K, ef), dtype=np.int64).reshape(len(q), K)
        sec = time.perf_counter() - t
        out.append((ef, recall(found, truth), len(q) / sec))
    return out


def at_target(points, target):
    """The width and speed at which the curve first reaches the target, interpolated between widths."""
    for (e0, r0, s0), (e1, r1, s1) in itertools.pairwise(points):
        if r0 >= target:
            return e0, s0
        if r1 >= target:
            f = (target - r0) / (r1 - r0)
            return e0 + f * (e1 - e0), s0 * (s1 / s0) ** f
    return None, None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sift", choices=sorted(METRIC))
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--recall", type=float, default=0.95)
    p.add_argument("--queries", type=int, default=2000)
    p.add_argument("--knn-nprobe", type=int, default=None, help="the rough graph's probes (siann's default if not given)")
    p.add_argument("--only-siann", action="store_true", help="siann's graph alone (no mlxvs or cuVS rows)")
    a = p.parse_args()
    xb, xq = load(a.dataset, 10_000)
    if a.n:
        xb = np.ascontiguousarray(xb[:a.n])
    metric = METRIC[a.dataset]
    q = np.ascontiguousarray(xq[-a.queries:], dtype=np.float32)
    n = len(xb)
    stored = xb.astype(np.uint8) if a.dataset == "sift" else None
    print(f"{a.dataset}: {n:,} x {xb.shape[1]}, {metric}; graphs of 32 links per vector searched by siann's CPU "
          f"search on {len(q):,} test queries", flush=True)
    truth = siann.Index(xb, metric).search(q, K).indices

    # the rough graph's quality on a sample: the share of each vector's true 64 nearest it holds
    t = time.perf_counter()
    kw = {"nprobe": a.knn_nprobe} if a.knn_nprobe else {}
    knn, times = knn_graph(xb, metric, **kw)
    sample = np.random.default_rng(0).choice(n, 1000, replace=False)
    true64 = siann.Index(xb, metric).search(xb[sample], KNN_K + 1).indices[:, 1:]
    print(f"  rough graph ({KNN_K} per vector by siann's IVF, nprobe {kw.get('nprobe', KNN_NPROBE)}): {time.perf_counter() - t:.1f} s "
          f"(IVF {times['IVF']:.1f} s, search {times['search']:.1f} s); holds {recall(knn[sample], true64):.3f} of "
          f"each vector's true {KNN_K} nearest", flush=True)

    graphs = {}
    t = time.perf_counter()
    links, times = build_links(xb, metric, **kw)
    graphs["siann"] = (links, time.perf_counter() - t,
                       f"IVF {times['IVF']:.1f} s, search {times['search']:.1f} s, optimize {times['optimize']:.1f} s")
    try:
        if a.only_siann:
            raise RuntimeError("--only-siann")
        from siann._graph import Graph
        g = Graph(xb, metric, "uint8" if stored is not None else "float32")
        graphs["mlxvs"] = (g.links(), g.build_seconds, "mlxvs's default build")
    except Exception as e:  # noqa: BLE001 (mlxvs missing: its row skipped)
        print(f"  (mlxvs skipped: {type(e).__name__}: {e})", flush=True)
    if a.dataset == "sift" and n == 1_000_000 and CUVS_LINKS.exists() and not a.only_siann:
        graphs["cuVS"] = (np.load(CUVS_LINKS), 7.0, "built on a Kaggle T4 (results/cuvs_cagra_t4.txt)")

    print(f"\n  {'graph':6s} {'built':>8s}   recall@10 / queries per second at ef " + " ".join(f"{e:>5}" for e in EFS),
          flush=True)
    summary = []
    for name, (lk, built, how) in graphs.items():
        pts = curve(xb, lk, metric, q, truth, stored)
        print(f"  {name:6s} {built:7.1f}s   " + "  ".join(f"{r:.3f}/{s / 1000:,.0f}k" for _, r, s in pts)
              + f"   ({how})", flush=True)
        summary.append((name, at_target(pts, a.recall)))
    print(f"\n  at recall {a.recall}:", flush=True)
    for name, (ef, speed) in summary:
        print(f"    {name:6s} " + (f"ef {ef:5.1f}, {speed:,.0f} queries/s" if ef else "not reached"), flush=True)


if __name__ == "__main__":
    main()
