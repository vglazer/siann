"""Can a graph built on the GPU serve single queries on the CPU as well as HNSW? mlxvs builds a CAGRA
graph on the GPU (graph degree 32 pruned from 64, as in its own benchmarks) and exports it in
hnswlib's format (`save_hnswlib`: the graph as hnswlib's bottom layer, no upper layers, entered at the
medoid); hnswlib then searches it on the CPU. Against FAISS's HNSW (M = 32), built and searched as
usual, and, with --native, hnswlib's own HNSW (M = 16, ef_construction 200, as in mlxvs's benchmarks),
so that the same search code runs on both kinds of graph.

Single queries, one call each, on one thread (as HNSW answers a single query): median latency in
microseconds over --small queries, after a warm-up pass, and queries per second from that pass. Batch
throughput with every thread (one call for all --queries) for reference. Recall@10 against exact search
(the Rust flat index), ties counted as correct (backends.agreement). Build times on all threads.

    uv run --group bench python graph_cpu.py --dataset sift
    uv run --group bench python graph_cpu.py --dataset dbpedia --base 100000 --queries 1000
"""
import argparse
import os
import tempfile
import time

import faiss
import hnswlib
import mlxvs
import numpy as np
from siann._cpu import FlatIndex

from backends import agreement
from datasets import load

K = 10
METRIC = {"sift": "l2", "glove": "l2", "music100": "ip", "dbpedia": "l2"}   # glove, dbpedia normalized
HNSWLIB_SPACE = {"l2": "l2", "ip": "ip"}
MLXVS_METRIC = {"l2": "l2", "ip": "inner_product"}
FAISS_METRIC = {"l2": faiss.METRIC_L2, "ip": faiss.METRIC_INNER_PRODUCT}


def single(search, xq):
    """Median latency (microseconds) over one pass of single queries after a warm-up pass, queries per
    second from that pass, and the results."""
    for q in xq[:100]:
        search(q[None, :])
    lat, found = [], []
    for q in xq:
        t = time.perf_counter()
        found.append(search(q[None, :]))
        lat.append(time.perf_counter() - t)
    return np.median(lat) * 1e6, len(xq) / sum(lat), np.concatenate(found)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sift", choices=sorted(METRIC))
    p.add_argument("--base", type=int, default=0, help="use only the first N base vectors (0: all)")
    p.add_argument("--queries", type=int, default=10_000)
    p.add_argument("--small", type=int, default=1000)
    p.add_argument("--efs", default="16,32,64,128,256,512")
    p.add_argument("--native", action="store_true", help="also hnswlib's own HNSW (M = 16)")
    a = p.parse_args()
    metric = METRIC[a.dataset]
    xb, xq = load(a.dataset, a.queries)
    if a.base:
        xb = np.ascontiguousarray(xb[:a.base])
    n, d = xb.shape
    small = xq[:a.small]
    truth = np.array(FlatIndex(xb, d, metric).search(xq, K), dtype=np.int64).reshape(-1, K)
    threads = os.cpu_count()
    print(f"{a.dataset}, {metric}: {n:,} vectors, {d} dimensions; single queries on one thread over "
          f"{len(small):,}, batch of {len(xq):,} on {threads} threads", flush=True)

    graphs = {}
    t = time.perf_counter()
    cagra = mlxvs.CagraIndex.build(xb, graph_degree=32, metric=MLXVS_METRIC[metric])
    path = os.path.join(tempfile.mkdtemp(), "cagra.hnsw")
    cagra.save_hnswlib(path)
    del cagra
    g = hnswlib.Index(space=HNSWLIB_SPACE[metric], dim=d)
    g.load_index(path)
    graphs["CAGRA (GPU-built), hnswlib search"] = (g, time.perf_counter() - t)
    if a.native:
        t = time.perf_counter()
        h = hnswlib.Index(space=HNSWLIB_SPACE[metric], dim=d)
        h.init_index(max_elements=n, M=16, ef_construction=200, random_seed=0)
        h.set_num_threads(threads)
        h.add_items(xb, np.arange(n))
        graphs["hnswlib HNSW (M 16)"] = (h, time.perf_counter() - t)
    t = time.perf_counter()
    faiss.omp_set_num_threads(threads)
    fh = faiss.IndexHNSWFlat(d, 32, FAISS_METRIC[metric])
    fh.add(xb)
    t_faiss = time.perf_counter() - t

    print(f"\n  builds: " + ", ".join(f"{name} {s:.1f} s" for name, (_, s) in graphs.items())
          + f", FAISS HNSW (M 32) {t_faiss:.1f} s")
    print(f"\n  {'method':42s} {'recall@10':>9s} {'median us':>9s} {'single/s':>9s} {f'{len(xq):,}/s':>10s}")
    for ef in map(int, a.efs.split(",")):
        for name, (g, _) in graphs.items():
            g.set_ef(max(ef, K))

            def search(q, g=g):
                return g.knn_query(q, k=K)[0].astype(np.int64)
            g.set_num_threads(1)
            us, qps1, _ = single(search, small)
            g.set_num_threads(threads)
            search(xq)
            t = time.perf_counter()
            found = search(xq)
            qps = len(xq) / (time.perf_counter() - t)
            r = agreement(xb, xq, found, truth, metric)
            print(f"  {f'{name}, ef {ef}':42s} {r:9.3f} {us:9.0f} {qps1:9,.0f} {qps:10,.0f}", flush=True)
        fh.hnsw.efSearch = ef

        def fsearch(q):
            return fh.search(q, K)[1].astype(np.int64)
        faiss.omp_set_num_threads(1)
        us, qps1, _ = single(fsearch, small)
        faiss.omp_set_num_threads(threads)
        fsearch(xq)
        t = time.perf_counter()
        found = fsearch(xq)
        qps = len(xq) / (time.perf_counter() - t)
        r = agreement(xb, xq, found, truth, metric)
        print(f"  {f'FAISS HNSW (M 32), efSearch {ef}':42s} {r:9.3f} {us:9.0f} {qps1:9,.0f} {qps:10,.0f}", flush=True)


if __name__ == "__main__":
    main()
