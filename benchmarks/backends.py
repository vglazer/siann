"""Exact k-nearest neighbors on each backend with its built-in operations, against FAISS's flat index
and siann's custom MLX kernels.

Built-in means what a user of the framework would write: L2 as norms minus twice a matrix multiply,
L1 as a broadcast absolute difference summed over dimensions (or `torch.cdist`), L-infinity ("linf")
the same with a maximum in place of the sum, then the framework's own top-k. Prints each backend's time for all queries and its agreement with FAISS
(the fraction of FAISS's k neighbors it also returns).

Each backend is warmed up on the full query set, then timed as the median of --repeats runs.
Each backend runs in its own process: FAISS and PyTorch each link their own OpenMP runtime, and the
two cannot be loaded into one process on macOS. It also keeps one backend from disturbing another's
timing.

    uv run --group bench python benchmarks/backends.py --n 200000 --q 1000
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

TILE = 256          # queries per tile
L1_TILE = 16        # queries per tile for the broadcast L1 (its intermediate is q x chunk x d)
L1_CHUNK = 25_000   # base vectors per chunk for the broadcast L1

BACKENDS = ["faiss-cpu", "siann-rust-cpu", "mlx-custom-gpu", "torch-custom-mps",
            "mlx-builtin-gpu", "mlx-compiled-gpu", "torch-builtin-mps",
            "mlx-builtin-cpu", "torch-builtin-cpu"]


# --- MLX, built-in operations ---------------------------------------------------------------------

_MLX = {}


def _mlx_fns(k: int, compiled: bool):
    """The tile functions, made once per process; with ``compiled``, under mx.compile, which compiles
    (and fuses where it can) the whole function instead of running it op by op."""
    key = (k, compiled)
    if key not in _MLX:
        import mlx.core as mx

        def l2_tile(q, xb, norms):
            dist = norms[None, :] - 2 * (q @ xb.T)
            idx = mx.argpartition(dist, k - 1, axis=1)[:, :k]
            vals = mx.take_along_axis(dist, idx, axis=1)
            return mx.take_along_axis(idx, mx.argsort(vals, axis=1), axis=1)

        def l1_chunk(q, b):
            return mx.sum(mx.abs(q[:, None, :] - b[None, :, :]), axis=2)

        def linf_chunk(q, b):
            return mx.max(mx.abs(q[:, None, :] - b[None, :, :]), axis=2)

        def topk_smallest(dist):
            idx = mx.argpartition(dist, k - 1, axis=1)[:, :k]
            vals = mx.take_along_axis(dist, idx, axis=1)
            return mx.take_along_axis(idx, mx.argsort(vals, axis=1), axis=1)

        fns = (l2_tile, l1_chunk, linf_chunk, topk_smallest)
        _MLX[key] = tuple(mx.compile(f) for f in fns) if compiled else fns
    return _MLX[key]


def mlx_builtin(xb, xq: np.ndarray, k: int, metric: str, device, compiled: bool = False) -> np.ndarray:
    import mlx.core as mx
    l2_tile, l1_chunk, linf_chunk, topk_smallest = _mlx_fns(k, compiled)
    chunk_fn = linf_chunk if metric == "linf" else l1_chunk
    out = []
    with mx.stream(device):
        if metric in ("l2", "ip"):
            # inner product: the same tile with zero norms (-2 q.x ranks as -q.x)
            norms = mx.sum(xb * xb, axis=1) if metric == "l2" else mx.zeros((xb.shape[0],))
            for s in range(0, len(xq), TILE):
                q = mx.array(xq[s:s + TILE])
                result = l2_tile(q, xb, norms)
                mx.eval(result)
                out.append(np.array(result))
        else:
            for s in range(0, len(xq), L1_TILE):
                q = mx.array(xq[s:s + L1_TILE])
                parts = []
                for c in range(0, xb.shape[0], L1_CHUNK):
                    d = chunk_fn(q, xb[c:c + L1_CHUNK])
                    mx.eval(d)
                    parts.append(d)
                result = topk_smallest(mx.concatenate(parts, axis=1))
                mx.eval(result)
                out.append(np.array(result))
    return np.concatenate(out)


# --- PyTorch, built-in operations -----------------------------------------------------------------

def torch_builtin(xb, xq: np.ndarray, k: int, metric: str) -> np.ndarray:
    import torch
    out = []
    with torch.no_grad():
        if metric == "ip":
            for s in range(0, len(xq), TILE):
                q = torch.from_numpy(xq[s:s + TILE]).to(xb.device)
                out.append(torch.topk(q @ xb.T, k, dim=1, largest=True).indices.cpu().numpy())
            return np.concatenate(out)
        norms = (xb * xb).sum(1) if metric == "l2" else None
        # On MPS, cdist builds the full broadcast (tile x n x d) and fails beyond 2^31 elements.
        tile = L1_TILE if metric != "l2" and xb.device.type == "mps" else TILE
        p = {"l1": 1.0, "linf": float("inf")}.get(metric)
        for s in range(0, len(xq), tile):
            q = torch.from_numpy(xq[s:s + tile]).to(xb.device)
            dist = norms[None, :] - 2 * (q @ xb.T) if metric == "l2" else torch.cdist(q, xb, p=p)
            out.append(torch.topk(dist, k, dim=1, largest=False).indices.cpu().numpy())
    return np.concatenate(out)


def torch_custom(xb, xq: np.ndarray, k: int, metric: str) -> np.ndarray:
    """siann's custom kernels (knn_kernels.py), ported to MPS in torch_kernels.py."""
    from siann.torch_kernels import knn_tiled, l2_knn

    import torch
    out = []
    with torch.no_grad():
        norms = (xb * xb).sum(1) if metric == "l2" else None
        for s in range(0, len(xq), TILE):
            q = torch.from_numpy(xq[s:s + TILE]).to(xb.device)
            result = l2_knn(q, xb, norms, k) if metric == "l2" else knn_tiled(q, xb, k, metric)
            out.append(result.cpu().numpy())
    return np.concatenate(out)


def make_runner(backend: str, xb: np.ndarray, k: int, threads: int):
    """A function (queries, metric) -> neighbor indices, with the base vectors already on the device."""
    if backend == "siann-rust-cpu":
        from siann._cpu import FlatIndex  # crates/siann-py, built with maturin
        indexes = {m: FlatIndex(xb, xb.shape[1], m) for m in ("l1", "l2", "linf", "ip")}

        def run(q, m):
            return np.array(indexes[m].search(q, k), dtype=np.int64).reshape(len(q), k)
        return run, "siann_cpu (Rust)"
    if backend == "faiss-cpu":
        import faiss
        faiss.omp_set_num_threads(threads)
        # FAISS 1.16 defaults this to 128,000, which sends L2 below 128 dimensions down a path 4-7x
        # slower than BLAS (GloVe-100: 2.9 s against 0.42 s). 20 was the earlier default.
        faiss.cvar.distance_compute_blas_threshold = 20
        from exact import faiss_knn
        return lambda q, m: faiss_knn(xb, q, k, m), f"faiss {faiss.__version__}"
    if backend.startswith("mlx"):
        import mlx.core as mx
        xb_mx = mx.array(xb)
        mx.eval(xb_mx)
        if backend == "mlx-custom-gpu":
            from siann.exact import mlx_knn
            return lambda q, m: mlx_knn(xb_mx, q, k, m), f"mlx {mx.__version__}"
        device = mx.gpu if backend.endswith("gpu") else mx.cpu
        compiled = "compiled" in backend
        return (lambda q, m: mlx_builtin(xb_mx, q, k, m, device, compiled)), f"mlx {mx.__version__}"
    if backend.startswith("torch"):
        import torch
        torch.set_num_threads(threads)
        xb_t = torch.from_numpy(xb).to("mps" if backend.endswith("mps") else "cpu")
        if backend == "torch-custom-mps":
            return lambda q, m: torch_custom(xb_t, q, k, m), f"torch {torch.__version__}"
        return lambda q, m: torch_builtin(xb_t, q, k, m), f"torch {torch.__version__}"
    raise ValueError(backend)


def data(a) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian vectors (n x d), or an ann-benchmarks dataset (datasets.py; n and d ignored)."""
    if a.dataset != "gaussian":
        from datasets import load
        return load(a.dataset, a.q)
    rng = np.random.default_rng(0)
    return (rng.standard_normal((a.n, a.d), dtype=np.float32),
            rng.standard_normal((a.q, a.d), dtype=np.float32))


def run_one(a, out: Path) -> None:
    """Child process: run one backend on every metric, save its results and times."""
    xb, xq = data(a)
    run, version = make_runner(a.backend, xb, a.k, a.threads)
    times = {}
    for metric in a.metrics.split(","):
        result = run(xq, metric)   # warm up on the full query set: every tile shape compiles here
        if result is None:         # this backend does not support this metric
            continue
        samples = []
        for _ in range(a.repeats):
            start = time.perf_counter()
            result = run(xq, metric)
            samples.append(time.perf_counter() - start)
        times[metric] = sorted(samples)[len(samples) // 2]   # the median
        np.save(out / f"{a.backend}-{metric}.npy", result)
    (out / f"{a.backend}.json").write_text(json.dumps({"version": version, "times": times}),
                                           encoding="utf-8")


def distances(xb: np.ndarray, xq: np.ndarray, idx: np.ndarray, metric: str) -> np.ndarray:
    """The exact distances (in float64) from each query to its returned neighbors."""
    diff = xb[idx].astype(np.float64) - xq[:, None, :].astype(np.float64)
    if metric == "l1":
        return np.abs(diff).sum(2)
    if metric == "linf":
        return np.abs(diff).max(2)
    if metric == "ip":   # negated, so that, as for the distances, smaller is nearer
        return -np.einsum("qkd,qd->qk", xb[idx].astype(np.float64), xq.astype(np.float64))
    return (diff * diff).sum(2)


def agreement(xb, xq, ours: np.ndarray, ref: np.ndarray, metric: str) -> float:
    """The fraction of returned neighbors no farther than the reference's k-th nearest. Unlike
    comparing indices, this counts a tie broken differently (common on SIFT's integer data) as
    correct. A missing neighbor (-1, or an id past the base, where fewer than k were found) is a miss."""
    ours = np.asarray(ours, dtype=np.int64)
    valid = (ours >= 0) & (ours < len(xb))
    kth = distances(xb, xq, ref, metric).max(axis=1, keepdims=True)
    mine = distances(xb, xq, np.where(valid, ours, 0), metric)
    return float(np.mean(valid & (mine <= kth + 1e-6 * np.abs(kth) + 1e-9)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200_000)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--q", type=int, default=1000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--metrics", default="l2,l1", help="any of l2, l1, linf, ip (inner product)")
    p.add_argument("--dataset", default="gaussian", choices=["gaussian", "sift", "glove", "glove-raw", "music100", "dbpedia"],
                   help="glove is normalized, so L2 ranks as cosine")
    p.add_argument("--repeats", type=int, default=3, help="timed runs after a warm-up; the median is reported")
    p.add_argument("--backends", default=",".join(BACKENDS))
    p.add_argument("--backend", help=argparse.SUPPRESS)   # set in the child process
    p.add_argument("--out", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.backend:
        run_one(a, Path(a.out))
        return

    backends = a.backends.split(",")
    if "faiss-cpu" not in backends:
        backends.insert(0, "faiss-cpu")   # the reference
    xb, xq = data(a)
    print(f"{a.dataset}: {len(xb):,} base vectors, {xb.shape[1]} dimensions, {len(xq):,} queries, "
          f"k={a.k}, {a.threads} CPU threads")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        info = {}
        for b in backends:
            args = [sys.executable, __file__, "--backend", b, "--out", tmp] + [
                f"--{f}={getattr(a, f)}" for f in ("n", "d", "q", "k", "threads", "metrics", "repeats", "dataset")]
            r = subprocess.run(args, check=False, capture_output=True, text=True, encoding="utf-8")
            if r.returncode != 0:
                print(f"  {b}: failed\n{r.stderr[-2000:]}")
                continue
            info[b] = json.loads((out / f"{b}.json").read_text(encoding="utf-8"))
        print("versions: " + ", ".join(sorted({v["version"] for v in info.values()})))
        for metric in a.metrics.split(","):
            ref, t_ref = np.load(out / f"faiss-cpu-{metric}.npy"), info["faiss-cpu"]["times"][metric]
            print(f"\n{metric.upper()}")
            for b, v in info.items():
                if metric not in v["times"]:
                    continue
                t = v["times"][metric]
                ours = np.load(out / f"{b}-{metric}.npy")
                print(f"  {b:20s} {t:8.3f} s  {t_ref / t:7.2f}x FAISS  agreement {agreement(xb, xq, ours, ref, metric):.4f}")


if __name__ == "__main__":
    main()
