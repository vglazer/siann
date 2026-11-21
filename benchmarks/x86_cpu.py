"""siann's CPU path against FAISS on any machine, x86-64 included, with nothing but NumPy, FAISS and siann
installed: exact search by L2, inner product, L1 and L-infinity, and approximate search (siann's IVF at a
recall target against FAISS's IVF-Flat and HNSW at matched recall), at batch sizes from 1 to 1,000, on
synthetic vectors of low intrinsic dimension (the datasets are too large for a CI runner). siann's
neighbors are checked against FAISS's exact search. The workflow x86_bench.yml runs it on GitHub's x86-64 runners, once per
build of siann (baseline x86-64, as the published wheels are built, and x86-64-v3, with AVX2).

    uv run --group bench python x86_cpu.py [--n 200000] [--d 128] [--siann-only]
"""
import argparse
import os
import platform
import time

import numpy as np

import siann

BATCHES = (1, 10, 100, 1000)
K = 10


def machine() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            name = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        name = platform.processor() or platform.machine()
    return f"{name}, {os.cpu_count()} threads, {platform.system()}"


INTRINSIC = 16   # real embeddings have low intrinsic dimension; i.i.d. noise in d dimensions makes an IVF too easy


def clustered(n: int, rng: np.random.Generator, centers: np.ndarray, w: np.ndarray) -> np.ndarray:
    """100 Gaussian clusters in 16 dimensions, projected to d by ``w``, plus a little noise in all d: an IVF
    at recall 0.95 probes about 9 of 1,789 lists on 200,000 x 128, as on SIFT."""
    z = centers[rng.integers(0, len(centers), n)] + rng.standard_normal((n, INTRINSIC), dtype=np.float32)
    return z @ w + 0.1 * rng.standard_normal((n, w.shape[1]), dtype=np.float32)


def rate(search, xq: np.ndarray, batch: int) -> float:
    """Queries per second searching ``batch`` queries per call: the median of 3 passes after a warm-up,
    over at least 100 queries (single queries) and up to all of xq."""
    q = xq[:max(min(len(xq), 100 * batch), min(len(xq), 100))]
    def run():
        for s in range(0, len(q), batch):
            search(q[s:s + batch])
    run()
    times = []
    for _ in range(3):
        t = time.perf_counter()
        run()
        times.append(time.perf_counter() - t)
    return len(q) / sorted(times)[1]


def recall(found: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean([len(set(a) & set(b)) / K for a, b in zip(found, truth)]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200_000)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--target", type=float, default=0.95, help="siann's recall target")
    p.add_argument("--siann-only", action="store_true", help="skip FAISS's timings (still its exact search)")
    a = p.parse_args()
    import faiss
    faiss.cvar.distance_compute_blas_threshold = 20   # FAISS's slow L2 path below 128 dimensions (BENCHMARKS)
    rng = np.random.default_rng(0)
    w = rng.standard_normal((INTRINSIC, a.d), dtype=np.float32)
    centers = 2 * rng.standard_normal((100, INTRINSIC), dtype=np.float32)
    xb, xq, cq = (clustered(n, rng, centers, w) for n in (a.n, 1000, 1000))
    print(f"{machine()}; siann {siann.__version__}, FAISS {faiss.__version__}")
    print(f"{a.n:,} x {a.d} clustered vectors, 1,000 queries, k = {K}; queries per second by batch size\n")

    faiss_metric = {"l2": faiss.METRIC_L2, "ip": faiss.METRIC_INNER_PRODUCT, "l1": faiss.METRIC_L1,
                    "linf": faiss.METRIC_Linf}
    print("exact search, siann / FAISS (ratio); siann's neighbors against FAISS's")
    print(f"  {'metric':6s}" + "".join(f"{b:>26,}" for b in BATCHES) + "   agree")
    truth_l2 = None
    for m in ("l2", "ip", "l1", "linf"):
        index = siann.Index(xb, m, device="cpu")
        flat = faiss.IndexFlat(a.d, faiss_metric[m])
        flat.add(xb)
        truth = flat.search(xq, K)[1]
        truth_l2 = truth if m == "l2" else truth_l2
        agree = recall(index.search(xq, K).indices, truth)
        cells = []
        for b in BATCHES:
            ours = rate(lambda q: index.search(q, K), xq, b)
            if a.siann_only:
                cells.append(f"{ours:>26,.0f}")
            else:
                theirs = rate(lambda q: flat.search(q, K), xq, b)
                cells.append(f"{f'{ours:,.0f} / {theirs:,.0f} ({ours / theirs:.2f})':>26s}")
        print(f"  {m:6s}" + "".join(cells) + f"   {agree:.4f}", flush=True)

    print(f"\napproximate search, L2: siann.Index(recall={a.target}) calibrated on 1,000 separate queries")
    t = time.perf_counter()
    index = siann.Index(xb, "l2", recall=a.target, kind="ivf", device="cpu", calibration_queries=cq)
    built = time.perf_counter() - t
    ours_recall = recall(index.search(xq, K).indices, truth_l2)
    ours = [rate(lambda q: index.search(q, K), xq, b) for b in BATCHES]
    nlist = index._ivf.nlist
    print(f"  siann IVF ({nlist:,} lists, nprobe {index.nprobe}, built in {built:.1f} s): recall {ours_recall:.3f}")
    print("  " + "".join(f"{b:>10,}" for b in BATCHES))
    print("  " + "".join(f"{r:10,.0f}" for r in ours), flush=True)
    if a.siann_only:
        return

    def matched(name, search, settings, set_setting):
        """The first setting whose recall reaches siann's, and its speed by batch size."""
        for s in settings:
            set_setting(s)
            r = recall(search(xq), truth_l2)
            if r >= ours_recall:
                rates = [rate(search, xq, b) for b in BATCHES]
                print(f"  {name} at {s} (recall {r:.3f}): "
                      + ", ".join(f"{b:,}: {x:,.0f} (siann {o / x:.2f}x)" for b, x, o in zip(BATCHES, rates, ours)),
                      flush=True)
                return
        print(f"  {name}: recall {r:.3f} at most, below siann's", flush=True)

    quantizer = faiss.IndexFlatL2(a.d)
    ivf = faiss.IndexIVFFlat(quantizer, a.d, nlist)
    t = time.perf_counter()
    ivf.train(xb)
    ivf.add(xb)
    print(f"  FAISS IVF-Flat ({nlist:,} lists) built in {time.perf_counter() - t:.1f} s")
    matched("FAISS IVF-Flat, nprobe", lambda q: ivf.search(q, K)[1], [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96, 128, 256, 512],
            lambda s: setattr(ivf, "nprobe", s))
    hnsw = faiss.IndexHNSWFlat(a.d, 32)
    t = time.perf_counter()
    hnsw.add(xb)
    print(f"  FAISS HNSW (M 32) built in {time.perf_counter() - t:.1f} s")
    matched("FAISS HNSW, efSearch", lambda q: hnsw.search(q, K)[1], [10, 12, 14, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256, 512],
            lambda s: setattr(hnsw.hnsw, "efSearch", s))


if __name__ == "__main__":
    main()
