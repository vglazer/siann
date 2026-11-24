"""A probe of siann's GPU kernels in Triton (NVIDIA and AMD GPUs): exact L1 search with the selection fused
into the distance kernel, siann's two-stage design. Stage 1: each program takes BLOCK_Q queries and BLOCK_N
base vectors, accumulates their L1 distances BLOCK_D coordinates at a time, sorts each query's BLOCK_N
distances (as one 64-bit key: the distance's bits, then the vector's index, so that ties order by index) and
keeps its best k. Stage 2: torch.topk over each query's survivors, k per block of the base.

--check: the neighbors must equal exact search's, ties included; on a GPU, or on any machine through Triton's
interpreter (TRITON_INTERPRET=1, on the CPU). --bench, on an NVIDIA GPU: queries per second against torch.cdist with
torch.topk, FAISS's GPU brute force and cuVS's brute force, all by L1, on Gaussian vectors.

    TRITON_INTERPRET=1 python triton_probe.py --check          # anywhere, on the CPU
    python triton_probe.py --bench                              # on a CUDA GPU (Colab, Kaggle)
"""
import argparse
import time

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _l1_block_topk(q_ptr, x_ptr, out_ptr, nq, n, d, nblocks,
                   BLOCK_Q: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, K: tl.constexpr):
    pq = tl.program_id(0)
    pn = tl.program_id(1)
    rq = pq * BLOCK_Q + tl.arange(0, BLOCK_Q)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_Q, BLOCK_N), dtype=tl.float32)
    for d0 in range(0, d, BLOCK_D):
        rd = d0 + tl.arange(0, BLOCK_D)
        q = tl.load(q_ptr + rq[:, None] * d + rd[None, :], mask=(rq[:, None] < nq) & (rd[None, :] < d), other=0.0)
        x = tl.load(x_ptr + rn[:, None] * d + rd[None, :], mask=(rn[:, None] < n) & (rd[None, :] < d), other=0.0)
        acc += tl.sum(tl.abs(q[:, None, :] - x[None, :, :]), axis=2)
    # one sortable key per candidate: the (non-negative) distance's bits above, the index below
    bits = acc.to(tl.int32, bitcast=True).to(tl.int64)
    key = (bits << 32) | rn[None, :].to(tl.int64)
    key = tl.where(rn[None, :] < n, key, 0x7FFFFFFFFFFFFFFF)
    key = tl.sort(key, dim=1)
    cols = tl.arange(0, BLOCK_N)
    out = out_ptr + rq[:, None] * (nblocks * K) + pn * K + cols[None, :]
    tl.store(out, key, mask=(rq[:, None] < nq) & (cols[None, :] < K))


SURVIVOR_BYTES = 1 << 30   # stage 1's output per call at most: larger batches go in chunks of queries


def l1_knn(x: torch.Tensor, q: torch.Tensor, k: int, block_q=16, block_n=128, block_d=8) -> torch.Tensor:
    """Each query's k nearest rows of x by L1 (int64 indices, nearest first, ties by index)."""
    nq, d = q.shape
    n = x.shape[0]
    nblocks = triton.cdiv(n, block_n)
    step = max(block_q, SURVIVOR_BYTES // (nblocks * k * 8) // block_q * block_q)
    out = []
    for s in range(0, nq, step):
        qs = q[s:s + step]
        survivors = torch.empty((len(qs), nblocks * k), dtype=torch.int64, device=x.device)
        grid = (triton.cdiv(len(qs), block_q), nblocks)
        _l1_block_topk[grid](qs, x, survivors, len(qs), n, d, nblocks, BLOCK_Q=block_q, BLOCK_N=block_n,
                             BLOCK_D=block_d, K=k)
        out.append(torch.topk(survivors, k, dim=1, largest=False, sorted=True).values & 0xFFFFFFFF)
    return torch.cat(out)


def check() -> None:
    """Against exact search by NumPy, ties broken by index, on sizes that leave partial blocks: on the GPU where
    there is one (the compiled kernel), otherwise on the CPU through Triton's interpreter."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  on {dev}")
    rng = np.random.default_rng(0)
    for n, d, nq, k in ((1000, 24, 37, 10), (777, 13, 5, 3), (300, 8, 20, 10)):
        x = rng.integers(0, 4, (n, d)).astype(np.float32)          # small integers: many ties
        q = rng.integers(0, 4, (nq, d)).astype(np.float32)
        dist = np.abs(q[:, None, :] - x[None, :, :]).sum(axis=2)
        truth = np.lexsort((np.broadcast_to(np.arange(n), dist.shape), dist), axis=1)[:, :k]
        got = l1_knn(torch.from_numpy(x).to(dev), torch.from_numpy(q).to(dev), k).cpu().numpy()
        same = np.array_equal(got, truth)
        print(f"  n {n}, d {d}, {nq} queries, k {k}: identical to exact search, ties included: {same}")
        assert same


def bench(n: int, d: int, batches) -> None:
    dev = torch.device("cuda")
    rng = np.random.default_rng(0)
    xb = rng.standard_normal((n, d), dtype=np.float32)
    xq = rng.standard_normal((max(batches), d), dtype=np.float32)
    x, q = torch.from_numpy(xb).to(dev), torch.from_numpy(xq).to(dev)
    k = 10
    print(f"{torch.cuda.get_device_name()}: {n:,} Gaussian vectors, {d} dimensions, L1, k = {k}; queries per second")

    def timed(fn, qb):
        fn(qb)
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(3):
            t = time.perf_counter()
            fn(qb)
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t)
        return len(qb) / best

    def cdist(qb):
        out = []
        for s in range(0, len(qb), 64):                            # cdist's broadcast exceeds memory beyond
            out.append(torch.topk(torch.cdist(qb[s:s + 64], x, p=1), k, largest=False).indices)
        return torch.cat(out)
    contenders = {"siann's kernel, Triton": lambda qb: l1_knn(x, qb, k), "torch.cdist + topk": cdist}
    try:
        import faiss
        import faiss.contrib.torch_utils  # lets FAISS take PyTorch tensors
        res = faiss.StandardGpuResources()
        contenders["FAISS GPU brute force"] = lambda qb: faiss.knn_gpu(res, qb, x, k, metric=faiss.METRIC_L1)
    except Exception as e:  # noqa: BLE001 (FAISS's GPU build missing: skip it)
        print(f"  (FAISS GPU skipped: {type(e).__name__}: {e})")
    try:
        from cuvs.neighbors import brute_force
        bf = brute_force.build(x, metric="l1")
        contenders["cuVS brute force"] = lambda qb: brute_force.search(bf, qb, k)
    except Exception as e:  # noqa: BLE001 (cuVS missing: skip it)
        print(f"  (cuVS skipped: {type(e).__name__}: {e})")
    truth = l1_knn(x, q[:100], k).cpu().numpy()
    if "FAISS GPU brute force" in contenders:
        _, fi = contenders["FAISS GPU brute force"](q[:100])
        fi = torch.as_tensor(fi).cpu().numpy()
        print(f"  the same neighbors as FAISS's on 100 queries: {(np.sort(fi, 1) == np.sort(truth, 1)).mean():.4f}")
    print(f"  {'':26s}" + "".join(f"{f'batch {b:,}':>14s}" for b in batches), flush=True)
    for name, fn in contenders.items():
        print(f"  {name:26s}", end="", flush=True)
        for b in batches:
            try:
                print(f"{timed(fn, q[:b]):14,.0f}", end="", flush=True)
            except Exception as e:  # noqa: BLE001 (report and go on: one failure should not hide the rest)
                print(f"\n    batch {b:,} failed: {type(e).__name__}: {str(e)[:200]}", end="")
        print(flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--batches", default="100,1000,10000")
    a = p.parse_args()
    if a.check:
        check()
    if a.bench:
        bench(a.n, a.d, tuple(map(int, a.batches.split(","))))


if __name__ == "__main__":
    main()
