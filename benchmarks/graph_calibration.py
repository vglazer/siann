"""Does a recall target calibrated on one set of queries hold on others? siann calibrates each search width
(the IVF's nprobe, the graph's ef on the CPU and itopk_size on the GPU) as the smallest that reaches the target
on its calibration queries. This script measures, at each width around the calibrated one, recall@10 on those
1,000 calibration queries and on 8,000 other queries from the same set, against exact search, with each
measurement's spread (the standard error of the mean over queries): how far recall on new queries falls below
the calibration's, for the IVF and the graph.

It also tests two remedies for the graph: calibrating on half of the calibration queries and checking the other
half (raising the width until both halves reach the target), and calibrating to the target plus a margin.

    uv run --group bench python graph_calibration.py --dataset music100
"""
import argparse

import siann

import numpy as np
from datasets import load

METRIC = {"sift": "l2", "glove": "cosine", "music100": "ip", "dbpedia": "cosine"}
K = 10
CAL = 1000


def per_query(found: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.array([len(set(f) & set(t)) / K for f, t in zip(found, truth, strict=True)])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="music100", choices=sorted(METRIC))
    p.add_argument("--recall", type=float, default=0.95)
    p.add_argument("--n", type=int, default=None)
    a = p.parse_args()
    xb, xq = load(a.dataset, 10_000)
    if a.n:
        xb = np.ascontiguousarray(xb[:a.n])
    metric = METRIC[a.dataset]
    cal, test = xq[:CAL], xq[CAL:]
    exact = siann.Index(xb, metric)
    truth_cal = exact.search(cal, K, device="gpu").indices
    truth_test = exact.search(test, K, device="gpu").indices
    index = siann.Index(xb, metric, recall=a.recall, calibration_queries=cal, kind="graph")
    ivf, g = index._ivf, index._ivf.graph
    qn = (lambda q: q / np.linalg.norm(q, axis=1, keepdims=True)) if metric == "cosine" else (lambda q: q)
    cq, tq = np.ascontiguousarray(qn(cal), dtype=np.float32), np.ascontiguousarray(qn(test), dtype=np.float32)
    print(f"{a.dataset}: {len(xb):,} x {xb.shape[1]}, {metric}, target {a.recall}; calibrated: nprobe "
          f"{index.nprobe}, ef {g.ef}, itopk {g.itopk}. recall@10 (id overlap) on the {CAL:,} calibration queries "
          f"and {len(test):,} others, each +- its standard error", flush=True)

    searches = {"IVF, nprobe": (index.nprobe, lambda q, w: ivf.search(q, K, w, "gpu-query")),
                "graph on the CPU, ef": (g.ef, lambda q, w: g.search_cpu(q, K, w)),
                "graph on the GPU, itopk": (g.itopk, lambda q, w: g.search(q, K, w))}
    for name, (w0, search) in searches.items():
        print(f"\n  {name} (calibrated {w0})", flush=True)
        for w in sorted({max(1, int(w0 * f)) for f in (0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0)}):
            rc, rt = per_query(search(cq, w), truth_cal), per_query(search(tq, w), truth_test)
            print(f"    {w:>5}: calibration {rc.mean():.4f} +- {rc.std() / len(rc) ** 0.5:.4f}   others "
                  f"{rt.mean():.4f} +- {rt.std() / len(rt) ** 0.5:.4f}   gap {rc.mean() - rt.mean():+.4f}"
                  + ("   <- calibrated" if w == w0 else ""), flush=True)

    # the remedies, for the graph on the CPU: the width each picks, and its recall on the others
    print("\n  remedies for the graph on the CPU (ef picked; recall on the others)", flush=True)

    def smallest(qs, truth, target):
        def recall_at(w):
            return per_query(g.search_cpu(qs, K, max(w, K)), truth).mean()
        w = 1
        while recall_at(w) < target and w < 1024:
            w *= 2
        lo, hi = w // 2, w
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if recall_at(mid) >= target else (mid, hi)
        return max(hi, K)

    picks = {"as siann does (all 1,000)": smallest(cq, truth_cal, a.recall)}
    half = CAL // 2
    w = smallest(cq[:half], truth_cal[:half], a.recall)
    while per_query(g.search_cpu(cq[half:], K, w), truth_cal[half:]).mean() < a.recall and w < 1024:
        w = int(w * 1.1) + 1
    picks["half, checked on the other half"] = w
    for m in (0.005, 0.01):
        picks[f"target + {m}"] = smallest(cq, truth_cal, min(a.recall + m, 1.0))
    for name, w in picks.items():
        rt = per_query(g.search_cpu(tq, K, w), truth_test)
        print(f"    {name:34s} ef {w:>4}: others {rt.mean():.4f} +- {rt.std() / len(rt) ** 0.5:.4f}", flush=True)


if __name__ == "__main__":
    main()
