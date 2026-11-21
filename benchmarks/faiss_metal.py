"""FAISS's Metal GPU backend (faiss-cpu 1.16 on Apple silicon): exact search (IndexFlat) and IVF-Flat moved
to the GPU with `index_cpu_to_gpu`, their neighbors checked against the same index on the CPU, and the
speed of both measured. Queries go in calls of `--chunk` (100 by default).

FAISS's GPU memory is counted outside the process, and an exact search holds chunk x n distances: calls of
10,000 queries against a million vectors (over 40 GB) froze this machine twice. So exact search refuses
calls of more than about 1.1 billion distances (1,000 queries against a million vectors); larger batches
are measured as calls of 1,000. The IVF holds only its probed lists' distances, so its calls may be larger.

    uv run --group bench python faiss_metal.py --stage tiny [--metric l1]      # 1,000 x 32 vectors, 1 query
    uv run --group bench python faiss_metal.py --stage small                   # 100,000 Gaussian vectors
    uv run --group bench python faiss_metal.py --stage exact --dataset sift --chunk 1000 --queries 10000
    uv run --group bench python faiss_metal.py --stage ivf --dataset glove --chunk 10000 --queries 10000
"""
import argparse
import os
import resource
import subprocess
import time

import faiss
import numpy as np

K = 10
MAX_DISTANCES = 1.1e9            # per exact-search call: about 4.4 GB of float32 distances
MIN_FREE_GB = 16                 # stop before a GPU call if the machine's free memory is below this
METRICS = {"l2": faiss.METRIC_L2, "ip": faiss.METRIC_INNER_PRODUCT, "l1": faiss.METRIC_L1,
           "linf": faiss.METRIC_Linf}
DEFAULT_METRIC = {"sift": "l2", "glove": "l2", "music100": "ip"}   # glove is served normalized: cosine


def free_gb() -> float:
    """Free and inactive memory, in GB (vm_stat's pages free + inactive + speculative)."""
    out = subprocess.run(["vm_stat"], check=True, capture_output=True, text=True, encoding="utf-8").stdout
    page = int(out.split("page size of ")[1].split(" ")[0])
    pages = {line.split(":")[0]: int(line.split(":")[1].strip().rstrip(".")) for line in out.splitlines()[1:]
             if ":" in line and line.split(":")[1].strip().rstrip(".").isdigit()}
    return sum(pages.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages speculative")) * page / 1e9


def report(label: str) -> None:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9     # bytes on macOS
    print(f"  [{label}] free memory {free_gb():.1f} GB, this process's peak {rss:.1f} GB", flush=True)


def search_chunks(index, xq: np.ndarray, chunk: int, params=None, guard: bool = False) -> tuple[np.ndarray, float]:
    """All of xq's neighbors, chunk queries per call; returns (ids, seconds). With ``guard`` (GPU calls),
    the machine's free memory is checked before each call, outside the timing, and the run stops if it is
    below MIN_FREE_GB."""
    out, spent = [], 0.0
    for s in range(0, len(xq), chunk):
        if guard and (free := free_gb()) < MIN_FREE_GB:
            raise SystemExit(f"stopped: {free:.1f} GB free before a GPU call, below {MIN_FREE_GB} GB")
        t = time.perf_counter()
        out.append(index.search(xq[s:s + chunk], K, params=params)[1])
        spent += time.perf_counter() - t
    return np.concatenate(out), spent


def compare(name: str, cpu, gpu, xb: np.ndarray, xq: np.ndarray, chunk: int, metric: str) -> None:
    """The GPU index against the CPU one: neighbors (ties counted as found) and speed."""
    from backends import agreement
    search_chunks(gpu, xq[:min(chunk, len(xq))], chunk, guard=True)    # warm-up
    search_chunks(cpu, xq[:min(chunk, len(xq))], chunk)
    report(f"{name}, after a warm-up call")
    found_gpu, t_gpu = search_chunks(gpu, xq, chunk, guard=True)
    found_cpu, t_cpu = search_chunks(cpu, xq, chunk)
    agree = agreement(xb, xq, found_gpu, found_cpu.astype(np.int64), metric)
    print(f"  {name}: GPU {len(xq) / t_gpu:,.0f} queries/s, CPU {len(xq) / t_cpu:,.0f} queries/s, "
          f"GPU's neighbors agree with the CPU's {agree:.4f}", flush=True)
    report(f"{name}, after the search")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["tiny", "small", "exact", "ivf", "sift-exact", "sift-ivf"])
    p.add_argument("--dataset", default="sift", choices=sorted(DEFAULT_METRIC))
    p.add_argument("--metric", choices=sorted(METRICS), help="default: the dataset's (l2 for tiny and small)")
    p.add_argument("--chunk", type=int, default=100, help="queries per call")
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--nlist", type=int, default=4000)
    p.add_argument("--nprobes", default="8,16,32,64")
    a = p.parse_args()
    stage = {"sift-exact": "exact", "sift-ivf": "ivf"}.get(a.stage, a.stage)
    metric = a.metric or (DEFAULT_METRIC[a.dataset] if stage in ("exact", "ivf") else "l2")
    faiss.omp_set_num_threads(os.cpu_count())
    # FAISS 1.16's default sends L2 below 128 dimensions down a slow path (BENCHMARKS, "FAISS's BLAS threshold")
    faiss.cvar.distance_compute_blas_threshold = 20
    print(f"FAISS {faiss.__version__}, stage {stage}, metric {metric}, {a.chunk} queries per call", flush=True)
    report("start")
    ngpu = faiss.get_num_gpus()
    print(f"  GPUs found: {ngpu}", flush=True)
    if ngpu < 1:
        return
    res = faiss.StandardGpuResources()
    report("GPU resources made")
    rng = np.random.default_rng(0)
    if stage in ("tiny", "small"):
        n, d, nq = (1000, 32, 1) if stage == "tiny" else (100_000, 128, 100)
        xb = rng.standard_normal((n, d), dtype=np.float32)
        xq = rng.standard_normal((nq, d), dtype=np.float32)
        cpu = faiss.IndexFlat(d, METRICS[metric])
        cpu.add(xb)
        compare(f"exact {metric}, {n:,} x {d}, {nq} queries", cpu, faiss.index_cpu_to_gpu(res, 0, cpu), xb, xq,
                min(a.chunk, nq), metric)
        return
    from datasets import load
    xb, xq = load(a.dataset, a.queries)
    n, d = xb.shape
    if stage == "exact":
        if a.chunk * n > MAX_DISTANCES:
            raise SystemExit(f"{a.chunk} queries per call against {n:,} vectors holds {a.chunk * n / 1e9:.1f} "
                             f"billion distances; at most {MAX_DISTANCES / 1e9:.1f} billion (lower --chunk)")
        cpu = faiss.IndexFlat(d, METRICS[metric])
        cpu.add(xb)
        compare(f"exact {metric}, {a.dataset}, {len(xq):,} queries", cpu, faiss.index_cpu_to_gpu(res, 0, cpu),
                xb, xq, a.chunk, metric)
        return
    quantizer = faiss.IndexFlat(d, METRICS[metric])
    cpu = faiss.IndexIVFFlat(quantizer, d, a.nlist, METRICS[metric])
    t = time.perf_counter()
    cpu.train(xb[np.random.default_rng(0).choice(n, min(n, 100 * a.nlist), replace=False)])
    cpu.add(xb)
    print(f"  IVF-Flat ({a.nlist:,} lists) built on the CPU in {time.perf_counter() - t:.1f} s", flush=True)
    gpu = faiss.index_cpu_to_gpu(res, 0, cpu)
    report("IVF copied to the GPU")
    exact = faiss.IndexFlat(d, METRICS[metric])
    exact.add(xb)
    truth, _ = search_chunks(exact, xq, 1000)
    truth = truth.astype(np.int64)
    from backends import agreement
    for nprobe in map(int, a.nprobes.split(",")):
        # the Metal index is an opaque faiss.Index: nprobe goes in per-search parameters, which it honors
        cpu.nprobe = nprobe
        params = faiss.SearchParametersIVF(nprobe=nprobe)
        search_chunks(gpu, xq[:a.chunk], a.chunk, params, guard=True)
        found_gpu, t_gpu = search_chunks(gpu, xq, a.chunk, params, guard=True)
        found_cpu, t_cpu = search_chunks(cpu, xq, a.chunk)
        print(f"  IVF-Flat nprobe {nprobe:3d}: recall@10 GPU {agreement(xb, xq, found_gpu, truth, metric):.3f}, "
              f"CPU {agreement(xb, xq, found_cpu.astype(np.int64), truth, metric):.3f}; "
              f"GPU {len(xq) / t_gpu:,.0f} queries/s, CPU {len(xq) / t_cpu:,.0f} queries/s", flush=True)
        report(f"nprobe {nprobe}")


if __name__ == "__main__":
    main()
