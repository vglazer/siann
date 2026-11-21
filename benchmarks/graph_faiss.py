"""Graph against graph, with the same search code: a CAGRA graph built on the GPU by mlxvs (graph degree
32 pruned from 64, as in its benchmarks), loaded into a FAISS HNSW index as its only layer (every vector
at level 0, its 32 CAGRA neighbors as its links, entered at the vector nearest the mean), searched by
FAISS's own HNSW code, against FAISS's own HNSW graphs: M = 32 (FAISS's usual setting, 64 links per vector
at level 0) and M = 16 (32 links, CAGRA's degree). The same CAGRA graph exported to hnswlib
(`save_hnswlib`) checks that the loaded copy behaves alike. graph_cpu.py compared graphs through hnswlib,
whose search is slow on Apple silicon; this removes that difference.

Single queries, one call each, on one thread: median latency in microseconds after a warm-up pass, and
queries per second from that pass; batch throughput with every thread for reference. Recall@10 against
exact search, ties counted as correct (backends.agreement).

    uv run --group bench python graph_faiss.py --dataset sift
    uv run --group bench python graph_faiss.py --dataset dbpedia --base 100000 --queries 1000
"""
import argparse
import os
import tempfile
import time

import faiss
import hnswlib
import mlxvs
import numpy as np
from backends import agreement
from datasets import load
from graph_cpu import single

from siann._cpu import FlatIndex

K = 10
METRIC = {"sift": "l2", "glove": "l2", "music100": "ip", "dbpedia": "l2"}   # glove, dbpedia normalized
FAISS_METRIC = {"l2": faiss.METRIC_L2, "ip": faiss.METRIC_INNER_PRODUCT}
MLXVS_METRIC = {"l2": "l2", "ip": "inner_product"}


def faiss_from_graph(xb: np.ndarray, graph: np.ndarray, metric: str, entry: int) -> faiss.IndexHNSWFlat:
    """A FAISS HNSW index whose only layer is ``graph`` (n x degree neighbor ids, -1 for none)."""
    n, d = xb.shape
    degree = graph.shape[1]
    index = faiss.IndexHNSWFlat(d, degree // 2, FAISS_METRIC[metric])
    if index.hnsw.nb_neighbors(0) != degree:
        raise ValueError(f"level 0 holds {index.hnsw.nb_neighbors(0)} links, the graph {degree}")
    index.storage.add(xb)
    index.ntotal = n
    h = index.hnsw
    faiss.copy_array_to_vector(np.ones(n, dtype=np.int32), h.levels)            # every vector at level 0
    faiss.copy_array_to_vector(np.arange(n + 1, dtype=np.uint64) * degree, h.offsets)
    faiss.copy_array_to_vector(np.ascontiguousarray(graph, dtype=np.int32).ravel(), h.neighbors)
    h.entry_point = entry
    h.max_level = 0
    return index


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sift", choices=sorted(METRIC))
    p.add_argument("--base", type=int, default=0, help="use only the first N base vectors (0: all)")
    p.add_argument("--queries", type=int, default=10_000)
    p.add_argument("--small", type=int, default=1000)
    p.add_argument("--efs", default="16,32,64,128,256,512")
    a = p.parse_args()
    metric = METRIC[a.dataset]
    xb, xq = load(a.dataset, a.queries)
    if a.base:
        xb = np.ascontiguousarray(xb[:a.base])
    n, d = xb.shape
    small = xq[:a.small]
    truth = np.array(FlatIndex(xb, d, metric).search(xq, K), dtype=np.int64).reshape(-1, K)
    threads = os.cpu_count()
    faiss.omp_set_num_threads(threads)
    print(f"{a.dataset}, {metric}: {n:,} vectors, {d} dimensions; single queries on one thread over "
          f"{len(small):,}, batch of {len(xq):,} on {threads} threads", flush=True)

    t = time.perf_counter()
    cagra = mlxvs.CagraIndex.build(xb, graph_degree=32, metric=MLXVS_METRIC[metric])
    graph = np.array(cagra.graph)
    t_cagra = time.perf_counter() - t
    path = os.path.join(tempfile.mkdtemp(), "cagra.hnsw")
    cagra.save_hnswlib(path)
    del cagra
    entry = int(np.argmin(((xb - xb.mean(axis=0)) ** 2).sum(axis=1)))
    indexes = {"CAGRA graph, FAISS search": ("faiss", faiss_from_graph(xb, graph, metric, entry))}
    hl = hnswlib.Index(space="l2" if metric == "l2" else "ip", dim=d)
    hl.load_index(path)
    indexes["CAGRA graph, hnswlib search"] = ("hnswlib", hl)
    builds = {}
    for m in (32, 16):
        t = time.perf_counter()
        fh = faiss.IndexHNSWFlat(d, m, FAISS_METRIC[metric])
        fh.add(xb)
        builds[m] = time.perf_counter() - t
        indexes[f"FAISS HNSW (M {m})"] = ("faiss", fh)
    print(f"\n  builds: CAGRA graph on the GPU {t_cagra:.1f} s, FAISS HNSW M 32 {builds[32]:.1f} s, "
          f"M 16 {builds[16]:.1f} s")
    print(f"\n  {'method':38s} {'recall@10':>9s} {'median us':>9s} {'single/s':>9s} {f'{len(xq):,}/s':>10s}")
    for ef in map(int, a.efs.split(",")):
        for name, (kind, index) in indexes.items():
            if kind == "faiss":
                index.hnsw.efSearch = ef

                def search(q, index=index):
                    return index.search(q, K)[1].astype(np.int64)

                def set_threads(t):
                    faiss.omp_set_num_threads(t)
            else:
                index.set_ef(max(ef, K))

                def search(q, index=index):
                    return index.knn_query(q, k=K)[0].astype(np.int64)

                def set_threads(t, index=index):
                    index.set_num_threads(t)
            set_threads(1)
            us, qps1, _ = single(search, small)
            set_threads(threads)
            search(xq)
            t = time.perf_counter()
            found = search(xq)
            qps = len(xq) / (time.perf_counter() - t)
            r = agreement(xb, xq, found, truth, metric)
            print(f"  {f'{name}, ef {ef}':38s} {r:9.3f} {us:9.0f} {qps1:9,.0f} {qps:10,.0f}", flush=True)


if __name__ == "__main__":
    main()
