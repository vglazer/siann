"""FAISS's Metal GPU backend (faiss-cpu 1.16 on Apple silicon), in stages from tiny to a million vectors:
exact search (IndexFlat) and IVF-Flat moved to the GPU with `index_cpu_to_gpu`, their neighbors checked
against the same index on the CPU, and their speed measured. Queries go in chunks (`--chunk`, 100 by
default), so that no call holds more than chunk x n distances.

Earlier runs of FAISS's Metal path with 10,000 queries against a million vectors in one call froze this
machine twice; run one stage at a time, in order, and check each before the next:

    uv run --group bench python faiss_metal.py --stage tiny         # 1,000 x 32 vectors, 1 query
    uv run --group bench python faiss_metal.py --stage small        # 100,000 Gaussian vectors, 100 queries
    uv run --group bench python faiss_metal.py --stage sift-exact   # SIFT1M, 1,000 queries in chunks
    uv run --group bench python faiss_metal.py --stage sift-ivf     # IVF-Flat on SIFT1M, 4,000 lists
"""
import argparse
import os
import resource
import subprocess
import time

import faiss
import numpy as np

K = 10


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


def search_chunks(index, xq: np.ndarray, chunk: int, params=None) -> tuple[np.ndarray, float]:
    """All of xq's neighbors, chunk queries per call; returns (ids, seconds)."""
    out = []
    t = time.perf_counter()
    for s in range(0, len(xq), chunk):
        out.append(index.search(xq[s:s + chunk], K, params=params)[1])
    return np.concatenate(out), time.perf_counter() - t


def compare(name: str, cpu, gpu, xb: np.ndarray, xq: np.ndarray, chunk: int, metric: str = "l2") -> None:
    """The GPU index against the CPU one: neighbors (ties counted as found) and speed."""
    from backends import agreement
    search_chunks(gpu, xq[:min(chunk, len(xq))], chunk)                # warm-up
    report(f"{name}, after a warm-up call")
    found_gpu, t_gpu = search_chunks(gpu, xq, chunk)
    found_cpu, t_cpu = search_chunks(cpu, xq, chunk)
    agree = agreement(xb, xq, found_gpu, found_cpu.astype(np.int64), metric)
    print(f"  {name}: GPU {len(xq) / t_gpu:,.0f} queries/s, CPU {len(xq) / t_cpu:,.0f} queries/s, "
          f"GPU's neighbors agree with the CPU's {agree:.4f}", flush=True)
    report(f"{name}, after the search")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["tiny", "small", "sift-exact", "sift-ivf"])
    p.add_argument("--chunk", type=int, default=100, help="queries per call")
    p.add_argument("--queries", type=int, default=1000)
    a = p.parse_args()
    faiss.omp_set_num_threads(os.cpu_count())
    print(f"FAISS {faiss.__version__}, stage {a.stage}, {a.chunk} queries per call", flush=True)
    report("start")
    ngpu = faiss.get_num_gpus()
    print(f"  GPUs found: {ngpu}", flush=True)
    if ngpu < 1:
        return
    res = faiss.StandardGpuResources()
    report("GPU resources made")
    rng = np.random.default_rng(0)
    if a.stage == "tiny":
        xb = rng.standard_normal((1000, 32), dtype=np.float32)
        xq = rng.standard_normal((1, 32), dtype=np.float32)
        cpu = faiss.IndexFlatL2(32)
        cpu.add(xb)
        compare("exact L2, 1,000 x 32, 1 query", cpu, faiss.index_cpu_to_gpu(res, 0, cpu), xb, xq, 1)
        return
    if a.stage == "small":
        xb = rng.standard_normal((100_000, 128), dtype=np.float32)
        xq = rng.standard_normal((100, 128), dtype=np.float32)
        cpu = faiss.IndexFlatL2(128)
        cpu.add(xb)
        compare("exact L2, 100,000 x 128, 100 queries", cpu, faiss.index_cpu_to_gpu(res, 0, cpu), xb, xq,
                a.chunk)
        return
    from datasets import load
    xb, xq = load("sift", a.queries)
    d = xb.shape[1]
    if a.stage == "sift-exact":
        cpu = faiss.IndexFlatL2(d)
        cpu.add(xb)
        compare(f"exact L2, SIFT1M, {len(xq):,} queries", cpu, faiss.index_cpu_to_gpu(res, 0, cpu), xb, xq,
                a.chunk)
        return
    quantizer = faiss.IndexFlatL2(d)
    cpu = faiss.IndexIVFFlat(quantizer, d, 4000)
    t = time.perf_counter()
    cpu.train(xb[np.random.default_rng(0).choice(len(xb), 400_000, replace=False)])
    cpu.add(xb)
    print(f"  IVF-Flat built on the CPU in {time.perf_counter() - t:.1f} s", flush=True)
    gpu = faiss.index_cpu_to_gpu(res, 0, cpu)
    report("IVF copied to the GPU")
    exact = faiss.IndexFlatL2(d)
    exact.add(xb)
    truth = exact.search(xq, K)[1].astype(np.int64)
    from backends import agreement
    for nprobe in (8, 16, 32, 64):
        # the Metal index is an opaque faiss.Index: nprobe goes in per-search parameters, which it honors
        cpu.nprobe = nprobe
        params = faiss.SearchParametersIVF(nprobe=nprobe)
        search_chunks(gpu, xq[:a.chunk], a.chunk, params)
        found_gpu, t_gpu = search_chunks(gpu, xq, a.chunk, params)
        found_cpu, t_cpu = search_chunks(cpu, xq, a.chunk)
        print(f"  IVF-Flat nprobe {nprobe:3d}: recall@10 GPU {agreement(xb, xq, found_gpu, truth, 'l2'):.3f}, "
              f"CPU {agreement(xb, xq, found_cpu.astype(np.int64), truth, 'l2'):.3f}; "
              f"GPU {len(xq) / t_gpu:,.0f} queries/s, CPU {len(xq) / t_cpu:,.0f} queries/s", flush=True)
        report(f"nprobe {nprobe}")


if __name__ == "__main__":
    main()
