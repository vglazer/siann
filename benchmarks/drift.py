"""How inserts degrade an IVF, and when it needs re-clustering: an IVF trained on part of a collection, the
rest inserted the way an online index would insert them (each vector into the list of its nearest centroid,
the centroids unchanged), against an IVF retrained from scratch on the same contents.

Two splits of a random subset of the collection: "random" (trained on a random 10%, so the collection grows
tenfold without drift) and "regions" (the subset clustered into 50 coarse regions, trained on all but 10 of
them, which are inserted: data from regions the centroids never saw). After inserting 0%, 50% and 100% of the
rest: recall@10 on the dataset's queries at the nprobe calibrated to the target when trained, the nprobe that
reaches the target again, and the largest list against the mean, for both IVFs, with the same number of
lists (about 4 sqrt(n) of the training set, at least 39 vectors per list), so that only the centroids
differ. For "regions", recall is also given for the queries that fall in the held-out regions. Then the
speed of both at their nprobe for the target (batches of 1,000 on the GPU), and what inserting costs:
assigning the vectors to their lists, re-clustering from scratch, and FAISS's HNSW (M 32) adding the same
vectors to one built on the training set.

    uv run --group bench python drift.py --dataset glove --base 200000
"""
import argparse
import time

import faiss
import numpy as np
from backends import agreement
from datasets import load

import siann
from siann._ivf import default_nlist
from siann.ivf_build import build_lists, lists_from_assign, nearest
from siann.ivf_gpu import GpuIVF
from siann.kmeans import kmeans

K = 10
STEPS = (0.0, 0.5, 1.0)


def needed_nprobe(recall_at, target: float, nlist: int) -> int:
    """The smallest nprobe whose recall reaches the target (doubling, then bisection)."""
    hi = 1
    while recall_at(hi) < target and hi < nlist:
        hi = min(2 * hi, nlist)
    lo = hi // 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if recall_at(mid) >= target else (mid, hi)
    return hi


def evaluate(x: np.ndarray, lists, xq: np.ndarray, truth: np.ndarray, nprobe: int, target: float, mask=None):
    """Recall at nprobe (and on the queries in ``mask``), the nprobe for the target, the largest list / mean."""
    ivf = GpuIVF(x, lists.nlist, "l2", "float32", built=lists)

    def recall_at(p, q=xq, t=truth):
        return agreement(x, q, ivf.search(q, K, p), t, "l2")
    sizes = lists.sizes()
    return {"recall": recall_at(nprobe), "masked": recall_at(nprobe, xq[mask], truth[mask]) if mask is not None else None,
            "needed": needed_nprobe(recall_at, target, lists.nlist), "imbalance": sizes.max() / sizes.mean(), "ivf": ivf}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="glove")
    p.add_argument("--base", type=int, default=200_000)
    p.add_argument("--queries", type=int, default=10_000)
    p.add_argument("--target", type=float, default=0.95)
    a = p.parse_args()
    xb, xq = load(a.dataset, a.queries)
    rng = np.random.default_rng(0)
    x = np.ascontiguousarray(xb[np.sort(rng.choice(len(xb), min(a.base, len(xb)), replace=False))])
    print(f"{a.dataset} (served normalized for cosine where it is), a random {len(x):,} of its {len(xb):,} vectors, "
          f"{len(xq):,} queries, recall@10 target {a.target}", flush=True)

    regions = kmeans(x, 50, device="gpu")
    region_of = nearest(x, regions, "l2", "gpu")
    held = rng.choice(50, size=10, replace=False)
    splits = {"random": rng.permutation(len(x)) < len(x) // 10, "regions": ~np.isin(region_of, held)}
    query_new = np.isin(nearest(xq, regions, "l2", "gpu"), held)

    for name, trained in splits.items():
        train, rest = np.where(trained)[0], rng.permutation(np.where(~trained)[0])
        nlist = default_nlist(len(train))
        built = build_lists(x[train], nlist, "l2", device="gpu")
        print(f"\n{name}: trained on {len(train):,} vectors ({len(train) / len(x):.0%}), {nlist:,} lists; "
              f"inserting {len(rest):,}" + (f"; {query_new.mean():.1%} of the queries fall in the held-out regions"
                                            if name == "regions" else ""), flush=True)
        exact0 = siann.Index(x[train], "l2", device="gpu").search(xq, K).indices
        ivf0 = GpuIVF(x[train], nlist, "l2", "float32", built=built)
        nprobe = needed_nprobe(lambda q: agreement(x[train], xq, ivf0.search(xq, K, q), exact0, "l2"), a.target, nlist)
        print(f"  calibrated when trained: nprobe {nprobe} for recall {a.target}")
        head = f"  {'inserted':>9s} {'vectors':>9s}   {'recall at nprobe ' + str(nprobe):>25s}"
        head += f"   {'new regions':>15s}" if name == "regions" else ""
        print(head + f"   {'nprobe for ' + str(a.target):>17s}   {'largest list / mean':>19s}")
        print(f"  {'':9s} {'':9s}   {'inserted / retrained':>25s}" + (f"   {'ins. / retr.':>15s}" if name == "regions" else "")
              + f"   {'ins. / retr.':>17s}   {'ins. / retr.':>19s}")
        for step in STEPS:
            ids = np.sort(np.concatenate([train, rest[:int(step * len(rest))]]))
            cur = np.ascontiguousarray(x[ids])
            truth = siann.Index(cur, "l2", device="gpu").search(xq, K).indices
            inserted = lists_from_assign(built.centroids, nearest(cur, built.centroids, "l2", "gpu"))
            retrained = build_lists(cur, nlist, "l2", device="gpu")
            mask = query_new if name == "regions" else None
            ins = evaluate(cur, inserted, xq, truth, nprobe, a.target, mask)
            ret = evaluate(cur, retrained, xq, truth, nprobe, a.target, mask)
            row = f"  {step:9.0%} {len(cur):9,}   {ins['recall']:>12.3f} / {ret['recall']:.3f}"
            row += f"   {ins['masked']:>7.3f} / {ret['masked']:.3f}" if mask is not None else ""
            row += f"   {ins['needed']:>9d} / {ret['needed']:<5d}   {ins['imbalance']:>11.1f} / {ret['imbalance']:.1f}"
            print(row, flush=True)
        q = xq[:1000]
        rates = []
        for ev in (ins, ret):
            ev["ivf"].search_list_major(q, K, ev["needed"])
            t = time.perf_counter()
            for _ in range(3):
                ev["ivf"].search_list_major(q, K, ev["needed"])
            rates.append(3 * len(q) / (time.perf_counter() - t))
        print(f"  at the target, all inserted, batches of 1,000 on the GPU: inserted {rates[0]:,.0f} queries/s, "
              f"retrained {rates[1]:,.0f} ({rates[0] / rates[1]:.2f}x)")
        t = time.perf_counter()
        nearest(x[rest], built.centroids, "l2", "gpu")
        t_assign = time.perf_counter() - t
        t = time.perf_counter()
        build_lists(x, nlist, "l2", device="gpu")
        t_retrain = time.perf_counter() - t
        faiss.omp_set_num_threads(16)
        hnsw = faiss.IndexHNSWFlat(x.shape[1], 32)
        hnsw.add(x[train])
        t = time.perf_counter()
        hnsw.add(x[rest])
        t_hnsw = time.perf_counter() - t
        print(f"  inserting {len(rest):,} vectors: siann's assignment {len(rest) / t_assign:,.0f} vectors/s; "
              f"re-clustering all {len(x):,} from scratch {t_retrain:.2f} s; "
              f"FAISS HNSW add (16 threads) {len(rest) / t_hnsw:,.0f} vectors/s", flush=True)


if __name__ == "__main__":
    main()
