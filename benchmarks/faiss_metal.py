"""FAISS's own Metal GPU path (faiss-cpu 1.16 on macOS) against siann's prototype kernels.

DO NOT RUN. On 2026-11-16 this script crashed the whole machine twice (FAISS's Metal search with
10,000 queries first failed with "GPU search failed (pipeline or dispatch error)", then the system
went down). It is kept only as a record of the comparison that was attempted.
"""
raise SystemExit("faiss_metal.py is disabled: FAISS's Metal path crashed this machine; see the docstring")
import time

import faiss
import mlx.core as mx
import numpy as np

from datasets import load
from siann.knn_kernels import l1_knn_tiled, l2_knn


def timed(fn, reps=3):
    fn()
    ts = []
    for _ in range(reps):
        s = time.perf_counter(); r = fn(); ts.append(time.perf_counter() - s)
    return r, sorted(ts)[reps // 2]


res = faiss.StandardGpuResources()
print("faiss", faiss.__version__, "GPUs", faiss.get_num_gpus(), flush=True)

# exact search, synthetic 1M x 128, 1,000 and 10,000 queries
rng = np.random.default_rng(0)
xb = rng.standard_normal((1_000_000, 128), dtype=np.float32)
xb_mx = mx.array(xb)
norms = mx.sum(xb_mx * xb_mx, axis=1)
mx.eval(xb_mx, norms)
for metric in ("l2", "l1"):
    cpu = faiss.IndexFlatL2(128) if metric == "l2" else faiss.IndexFlat(128, faiss.METRIC_L1)
    cpu.add(xb)
    try:
        gpu = faiss.index_cpu_to_gpu(res, 0, cpu)
    except Exception as e:
        print(f"exact {metric}: FAISS GPU unavailable: {str(e).splitlines()[0][:120]}", flush=True)
        continue
    for nq in (1000, 10000):
        xq = rng.standard_normal((nq, 128), dtype=np.float32)
        def faiss_search(index=gpu, xq=xq):
            try:
                return index.search(xq, 10)[1], "one batch"
            except RuntimeError:   # the Metal backend fails on large batches: fall back to chunks
                return np.concatenate([index.search(xq[i:i + 1000], 10)[1] for i in range(0, len(xq), 1000)]), "chunks of 1,000"
        (ref, how), t_gpu = timed(faiss_search)
        def ours():
            out = []
            for i in range(0, nq, 256):
                q = mx.array(xq[i:i + 256])
                r = l2_knn(q, xb_mx, norms, 10) if metric == "l2" else l1_knn_tiled(q, xb_mx, 10)
                mx.eval(r); out.append(np.array(r))
            return np.concatenate(out)
        mine, t_ours = timed(ours)
        agree = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ref, mine)])
        print(f"exact {metric}, {nq:,} queries: FAISS GPU ({how}) {t_gpu:.3f} s, siann {t_ours:.3f} s, "
              f"siann {t_gpu / t_ours:.1f}x faster; agreement {agree:.4f}", flush=True)

# IVF on SIFT1M, L2, batches of 10,000
xb, xq = load("sift", 10000)
xb_mx = mx.array(xb)
norms = mx.sum(xb_mx * xb_mx, axis=1)
truth = np.concatenate([np.array(l2_knn(mx.array(xq[i:i + 256]), xb_mx, norms, 10)) for i in range(0, len(xq), 256)])
quant = faiss.IndexFlatL2(128)
ivf = faiss.IndexIVFFlat(quant, 128, 4000)
ivf.train(xb[np.random.default_rng(0).choice(len(xb), 400_000, replace=False)])
ivf.add(xb)
try:
    givf = faiss.index_cpu_to_gpu(res, 0, ivf)
    for nprobe in (16, 32, 64, 128):
        givf.nprobe = nprobe
        def ivf_search():
            try:
                return givf.search(xq, 10)[1], "one batch"
            except RuntimeError:
                return np.concatenate([givf.search(xq[i:i + 1000], 10)[1] for i in range(0, len(xq), 1000)]), "chunks of 1,000"
        (found, how), t = timed(ivf_search)
        rec = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(found, truth)])
        print(f"SIFT1M L2, FAISS GPU IVF ({how}) nprobe {nprobe:3d}: recall@10 {rec:.3f}, {len(xq) / t:,.0f} queries/s", flush=True)
except Exception as e:
    print(f"FAISS GPU IVF unavailable: {str(e).splitlines()[0][:160]}")
