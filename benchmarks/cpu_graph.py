"""A CAGRA graph built on the Mac's GPU by mlxvs, searched on the CPU by siann anywhere, Linux included.

--export DIR, on a Mac with mlxvs: builds the dataset's CAGRA graph (degree 32, pruned from 64) and saves the
vectors, the queries and the graph to DIR/<dataset>_cagra.npz, a file that needs neither the GPU nor mlxvs.

--run DIR, on any machine: loads it and compares, on the CPU, siann's search of the graph
(siann._cpu.GraphIndex), FAISS's search of the same graph (the graph loaded into a FAISS HNSW index as its
only layer: the same graph, FAISS's code), siann's CPU IVF (siann.Index, its nprobe set by hand) and FAISS's
own HNSW (M = 32, built here; on SIFT also with byte vectors, its fastest there). Each is set to the smallest
ef or nprobe whose recall@10 on the queries reaches each target, then timed at batch sizes of 1, 10 and 100
(queries per second, the best of three passes). All threads of the machine.

    uv run --group bench python cpu_graph.py --export /tmp/graphs --dataset sift   # on the Mac
    python cpu_graph.py --run /tmp/graphs --dataset sift                          # anywhere

--export-index DIR builds and saves a whole siann.Index (recall target 0.95, its graph built by mlxvs) on the
Mac; --run-index DIR loads it anywhere and searches it at the routes it re-measures there, against FAISS's
HNSW built there at the efSearch matching its recall at each batch size.
"""
import argparse
import os
import time

import faiss
import numpy as np

import siann
from siann._cpu import FlatIndex, GraphIndex, recall_of, scores

K = 10
METRIC = {"sift": "l2", "music100": "ip"}
FAISS_METRIC = {"l2": faiss.METRIC_L2, "ip": faiss.METRIC_INNER_PRODUCT}
TARGETS = (0.90, 0.95, 0.98)
BATCHES = (1, 10, 100)


def export(path, dataset):
    import mlx.core as mx
    import mlxvs
    from datasets import load
    xb, xq = load(dataset, 10_000)
    integer = bool(np.array_equal(np.rint(xb), xb) and xb.min() >= 0 and xb.max() <= 255)
    t = time.perf_counter()
    g = mlxvs.CagraIndex.build(xb, graph_degree=32, metric="inner_product" if METRIC[dataset] == "ip" else "l2",
                               dtype="uint8" if integer else "float16")
    mx.eval(g._graph)
    seconds = time.perf_counter() - t
    np.savez(path, xb=xb.astype(np.uint8) if integer else xb, xq=xq, graph=np.asarray(g._graph, dtype=np.int32),
             build_seconds=seconds)
    print(f"{dataset}: graph built by mlxvs on the GPU in {seconds:.1f} s, saved to {path}")


def faiss_from_graph(xb, graph, metric, entry):
    """A FAISS HNSW index whose only layer is ``graph`` (n x degree neighbor ids), entered at ``entry``."""
    n, d = xb.shape
    degree = graph.shape[1]
    index = faiss.IndexHNSWFlat(d, degree // 2, FAISS_METRIC[metric])
    index.storage.add(xb)
    index.ntotal = n
    h = index.hnsw
    faiss.copy_array_to_vector(np.ones(n, dtype=np.int32), h.levels)
    faiss.copy_array_to_vector(np.arange(n + 1, dtype=np.uint64) * degree, h.offsets)
    faiss.copy_array_to_vector(np.ascontiguousarray(graph, dtype=np.int32).ravel(), h.neighbors)
    h.entry_point = entry
    h.max_level = 0
    return index


def agreement(xb, xq, found, truth, metric):
    """Recall@10 counting ties as found: a neighbor at least as near as the true 10th (siann's own measure)."""
    found = np.ascontiguousarray(np.where((found >= 0) & (found < len(xb)), found, -1), dtype=np.int64)
    kth = scores(xb, xq, np.ascontiguousarray(truth[:, -1:]), metric)[:, 0]
    return recall_of(found, scores(xb, xq, found, metric), np.ascontiguousarray(kth), metric == "ip")


def rate(fn, q, b):
    q = q[:max(b, min(len(q), 200 * b))]

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


def smallest(recall_at, want, lo, hi):
    """The smallest setting in [lo, hi] whose recall reaches ``want`` (bisection), and its recall."""
    cache = {}

    def at(v):
        if v not in cache:
            cache[v] = recall_at(v)
        return cache[v]
    if at(lo) >= want:
        return lo, at(lo)
    if at(hi) < want:
        return hi, at(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if at(mid) >= want else (mid, hi)
    return hi, at(hi)


def cpus() -> int:
    """The CPUs this process may run on (a container's pinned set, not the host's count)."""
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()


def run(path, dataset, nq, only_graph=False, targets=TARGETS, float32=False):
    faiss.omp_set_num_threads(cpus())            # FAISS's threads as many as siann's (Rust follows the pinning)
    data = np.load(path)
    xb = np.ascontiguousarray(data["xb"], dtype=np.float32)
    xq = np.ascontiguousarray(data["xq"][:nq], dtype=np.float32)
    graph = data["graph"]
    integer = data["xb"].dtype == np.uint8 and not float32
    metric = METRIC[dataset]
    n, d = xb.shape
    truth = np.array(FlatIndex(xb, d, metric).search(xq, K), dtype=np.int64).reshape(-1, K)
    rec = lambda found: agreement(xb, xq, found, truth, metric)
    print(f"{dataset}, {metric}: {n:,} vectors, {d} dimensions, {len(xq):,} queries, {cpus()} CPUs; graph "
          f"built on the Mac's GPU by mlxvs in {float(data['build_seconds']):.1f} s; queries per second")
    contenders = {}
    t = time.perf_counter()
    stored = (data["xb"],) if integer else ()
    g = GraphIndex(xb, np.ascontiguousarray(graph, dtype=np.int64), d, metric, *stored,
                   **({"storage": "uint8"} if integer else {}))
    contenders["siann, CAGRA graph"] = (time.perf_counter() - t, "ef",
                                        lambda q, p: np.asarray(g.search(q, K, p), dtype=np.int64).reshape(-1, K),
                                        (K, 1024))
    mean = xb.mean(axis=0)
    entry = int(np.argmin(((xb - mean) ** 2).sum(axis=1)))
    fg = faiss_from_graph(xb, graph, metric, entry)

    def faiss_search(index):
        def search(q, p):
            index.hnsw.efSearch = p
            return index.search(q, K)[1].astype(np.int64)
        return search
    contenders["FAISS, the same graph"] = (0.0, "ef", faiss_search(fg), (K, 1024))
    if only_graph:
        return compare(contenders, rec, xq, targets)
    t = time.perf_counter()
    ivf = siann.Index(xb, metric, recall=0.95, kind="ivf", device="cpu", calibration_queries=xq[:1000])
    contenders["siann, CPU IVF"] = (time.perf_counter() - t, "nprobe",
                                    lambda q, p: ivf.search(q, K, nprobe=p).indices, (1, 1024))
    for name, make in (("float32", lambda: faiss.IndexHNSWFlat(d, 32, FAISS_METRIC[metric])),) + (
            (("uint8", lambda: faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit_direct, 32,
                                                 FAISS_METRIC[metric])),) if integer else ()):
        h = make()
        t = time.perf_counter()
        if name != "float32":
            h.train(xb)
        h.add(xb)
        contenders[f"FAISS HNSW, {name}"] = (time.perf_counter() - t, "ef", faiss_search(h), (K, 1024))
    print("  build here: " + ", ".join(f"{k} {v[0]:.1f} s" for k, v in contenders.items() if v[0]))
    compare(contenders, rec, xq, targets)


def compare(contenders, rec, xq, targets):
    """Each contender at the smallest setting reaching each target, timed at each batch size."""
    for target in targets:
        print(f"\n  recall {target}: {'setting':>12s} {'recall':>7s}" + "".join(f"{f'batch {b}':>11s}" for b in BATCHES))
        for label, (_, knob, search, (lo, hi)) in contenders.items():
            p, got = smallest(lambda v, search=search: rec(search(xq, v)), target, lo, hi)
            cells = "".join(f"{rate(lambda q, p=p, search=search: search(q, p), xq, b):11,.0f}" for b in BATCHES)
            print(f"  {label:24s} {f'{knob} {p}':>12s} {got:7.3f}{cells}", flush=True)


def export_index(folder, dataset):
    """A siann.Index built on the Mac (recall target 0.95, calibrated on 1,000 of the queries; its graph built on
    the GPU by mlxvs), saved with the dataset's other queries for testing elsewhere."""
    from datasets import load
    xb, xq = load(dataset, 10_000)
    t = time.perf_counter()
    index = siann.Index(xb, METRIC[dataset], recall=0.95, calibration_queries=xq[:1000])
    print(f"{dataset}: siann.Index built on the Mac in {time.perf_counter() - t:.1f} s\n{index.plan}")
    index.save(os.path.join(folder, f"{dataset}.siann"))
    np.save(os.path.join(folder, f"{dataset}_test_queries.npy"), np.ascontiguousarray(xq[1000:]))


def run_index(folder, dataset, queries=2000):
    """The index saved by export_index, loaded here (or with ``folder`` None, built here, graph and all, as a user
    of this machine would build it) and searched at its own routes, against FAISS's HNSW built here at the
    efSearch that matches its recall at each batch size."""
    faiss.omp_set_num_threads(cpus())
    t = time.perf_counter()
    if folder is None:
        from datasets import load
        xb0, xq0 = load(dataset, 10_000)
        index = siann.Index(xb0, METRIC[dataset], recall=0.95, calibration_queries=xq0[:1000])
        xq = np.ascontiguousarray(xq0[-queries:], dtype=np.float32)
    else:
        index = siann.Index.load(os.path.join(folder, f"{dataset}.siann"))
        xq = np.load(os.path.join(folder, f"{dataset}_test_queries.npy"))
    loaded = time.perf_counter() - t
    xb = np.ascontiguousarray(index._x)
    metric = METRIC[dataset]
    d = xb.shape[1]
    truth = np.array(FlatIndex(xb, d, metric).search(xq, K), dtype=np.int64).reshape(-1, K)
    rec = lambda found: agreement(xb, xq, found, truth, metric)
    print(f"{dataset}: the index {'built here' if folder is None else 'built on the Mac, loaded here'} in "
          f"{loaded:.1f} s ({cpus()} CPUs); recall@10 on "
          f"{len(xq):,} queries; queries per second\n{index.plan}\n")
    hnsw = [("float32", faiss.IndexHNSWFlat(d, 32, FAISS_METRIC[metric]))]
    if index._storage == "uint8":
        hnsw.append(("uint8", faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit_direct, 32, FAISS_METRIC[metric])))
    for name, h in hnsw:
        if name != "float32":
            h.train(xb)
        h.add(xb)
    print(f"  {'batch':>6s} {'siann: path':>14s} {'recall':>7s} {'q/s':>9s}" +
          "".join(f"  {f'FAISS HNSW {name}: ef, recall, q/s':>36s}" for name, _ in hnsw))
    for b in BATCHES:
        found = np.concatenate([index.search(xq[s:s + b], K).indices for s in range(0, len(xq), b)])
        got = rec(found)
        line = (f"  {b:6d} {index._ivf_path(b, None):>14s} {got:7.3f} "
                f"{rate(lambda q: index.search(q, K), xq, b):9,.0f}")
        for _, h in hnsw:
            def at(ef, h=h):
                h.hnsw.efSearch = ef
                return rec(h.search(xq, K)[1].astype(np.int64))
            ef, r_h = smallest(at, got, K, 1024)
            h.hnsw.efSearch = ef
            line += f"  {ef:>18d} {r_h:7.3f} {rate(lambda q, h=h: h.search(q, K), xq, b):9,.0f}"
        print(line, flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sift", choices=sorted(METRIC))
    p.add_argument("--export", help="build the graph on the Mac and save it in this folder")
    p.add_argument("--run", help="search the saved graph from this folder, on this machine's CPU")
    p.add_argument("--export-index", help="build and save a siann.Index (with its graph) in this folder, on the Mac")
    p.add_argument("--run-index", help="load the saved siann.Index from this folder and search it here")
    p.add_argument("--build-index", action="store_true", help="build the siann.Index here and search it here")
    p.add_argument("--queries", type=int, default=2000)
    p.add_argument("--only-graph", action="store_true", help="only siann's and FAISS's search of the same graph")
    p.add_argument("--targets", default=",".join(map(str, TARGETS)))
    p.add_argument("--float32", action="store_true", help="siann reads float32 rows even for byte data")
    a = p.parse_args()
    if a.export:
        os.makedirs(a.export, exist_ok=True)
        export(os.path.join(a.export, f"{a.dataset}_cagra.npz"), a.dataset)
    if a.export_index:
        os.makedirs(a.export_index, exist_ok=True)
        export_index(a.export_index, a.dataset)
    if a.run_index:
        run_index(a.run_index, a.dataset)
    if a.build_index:
        run_index(None, a.dataset, a.queries)
    if a.run:
        run(os.path.join(a.run, f"{a.dataset}_cagra.npz"), a.dataset, a.queries, a.only_graph,
            tuple(map(float, a.targets.split(","))), a.float32)


if __name__ == "__main__":
    main()
