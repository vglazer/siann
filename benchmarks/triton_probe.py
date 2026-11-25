"""A probe of siann's GPU kernels in Triton (NVIDIA and AMD GPUs): exact L1 search with the selection fused
into the distance kernel, siann's two-stage design. Stage 1: each program takes BLOCK_Q queries and BLOCK_N
base vectors, accumulates their L1 distances BLOCK_D coordinates at a time, and keeps each query's best k of
the BLOCK_N (one 64-bit key each: the distance's bits, then the vector's index, so that ties order by index),
by sorting them or by k passes of a minimum. Triton's auto-tuner picks block sizes, selection and warps. Stage 2: torch.topk over each query's survivors, k per block of the base.

--check: the neighbors must equal exact search's, ties included; on a GPU, or on any machine through Triton's
interpreter (TRITON_INTERPRET=1, on the CPU). --bench, on an NVIDIA GPU: queries per second against torch.cdist with
torch.topk, FAISS's GPU brute force and cuVS's brute force, all by L1, on Gaussian vectors, and the memory each
takes: PyTorch's peak (exact, but blind to FAISS's and cuVS's own allocators), the device's peak above the level
before the call (NVIDIA's management library, polled every 10 ms: short spikes may be missed), the device memory
each keeps between calls (FAISS's scratch memory, cuVS's index), and the host process's peak.

    TRITON_INTERPRET=1 python triton_probe.py --check          # anywhere, on the CPU
    python triton_probe.py --bench                              # on a CUDA GPU (Colab, Kaggle)
"""
import argparse
import gc
import threading
import time

import numpy as np
import torch
import triton
import triton.language as tl

MAX_KEY = 0x7FFFFFFFFFFFFFFF


@triton.jit
def _l1_block_topk(q_ptr, x_ptr, out_ptr, nq, n, d, stride,
                   BLOCK_Q: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, K: tl.constexpr,
                   SORT: tl.constexpr):
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
    _keep_best(acc, rq, rn, pn, out_ptr, nq, n, stride, BLOCK_N, K, SORT)


@triton.jit
def _l1_tile_topk(qt_ptr, xt_ptr, out_ptr, nq, n, d, stride,
                  BLOCK_Q: tl.constexpr, BLOCK_N: tl.constexpr, K: tl.constexpr, SORT: tl.constexpr):
    """The same search with the queries and base vectors stored transposed (one dimension at a time): each step
    reads one dimension of BLOCK_Q queries and of BLOCK_N base vectors, both contiguous, and adds their
    differences into a 2D tile of running distances, as a matrix multiply's rank-1 update does."""
    pq = tl.program_id(0)
    pn = tl.program_id(1)
    rq = pq * BLOCK_Q + tl.arange(0, BLOCK_Q)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_Q, BLOCK_N), dtype=tl.float32)
    for j in range(d):
        qj = tl.load(qt_ptr + j * nq + rq, mask=rq < nq, other=0.0)
        xj = tl.load(xt_ptr + j * n + rn, mask=rn < n, other=0.0)
        acc += tl.abs(qj[:, None] - xj[None, :])
    _keep_best(acc, rq, rn, pn, out_ptr, nq, n, stride, BLOCK_N, K, SORT)


@triton.jit
def _keep_best(acc, rq, rn, pn, out_ptr, nq, n, stride, BLOCK_N: tl.constexpr, K: tl.constexpr,
               SORT: tl.constexpr):
    """Each query's best K of the block, written to its slots of the survivors."""
    # one sortable key per candidate: the (non-negative) distance's bits above, the index below
    bits = acc.to(tl.int32, bitcast=True).to(tl.int64)
    key = (bits << 32) | rn[None, :].to(tl.int64)
    key = tl.where(rn[None, :] < n, key, 0x7FFFFFFFFFFFFFFF)
    if SORT:                                    # the whole block sorted, its first K kept
        key = tl.sort(key, dim=1)
        cols = tl.arange(0, BLOCK_N)
        out = out_ptr + rq[:, None] * stride + pn * K + cols[None, :]
        tl.store(out, key, mask=(rq[:, None] < nq) & (cols[None, :] < K))
    else:                                       # K passes, each taking the smallest key (keys are unique)
        for j in tl.static_range(K):
            m = tl.min(key, axis=1)
            tl.store(out_ptr + rq * stride + pn * K + j, m, mask=rq < nq)
            key = tl.where(key == m[:, None], 0x7FFFFFFFFFFFFFFF, key)


# Tiles of about 16,000 values (BLOCK_Q x BLOCK_N x BLOCK_D), larger ones spilling out of the registers; both
# selections; 4 or 8 warps. Triton's auto-tuner times each once per shape and keeps the fastest.
TILES = ((16, 128, 8), (32, 128, 4), (64, 64, 4), (32, 64, 8), (16, 256, 4), (64, 128, 2))
CONFIGS = [triton.Config({"BLOCK_Q": bq, "BLOCK_N": bn, "BLOCK_D": bd, "SORT": sort}, num_warps=w)
           for bq, bn, bd in TILES for sort in (True, False) for w in (4, 8)]
# the 2D-tile kernel: no third dimension in registers, so larger tiles
TILES_2D = ((32, 128), (64, 128), (128, 128), (32, 256), (64, 256), (128, 64))
CONFIGS_2D = [triton.Config({"BLOCK_Q": bq, "BLOCK_N": bn, "SORT": sort}, num_warps=w)
              for bq, bn in TILES_2D for sort in (True, False) for w in (4, 8)]
MIN_BLOCK_N = min([bn for _, bn, _ in TILES] + [bn for _, bn in TILES_2D])
# each trial of the auto-tuner restores the survivors: configurations with other block sizes write other slots
_l1_tuned = triton.autotune(configs=CONFIGS, key=["n", "d", "K"], restore_value=["out_ptr"])(_l1_block_topk)
_tiles_tuned = triton.autotune(configs=CONFIGS_2D, key=["n", "d", "K"], restore_value=["out_ptr"])(_l1_tile_topk)


SURVIVOR_BYTES = 1 << 30   # stage 1's output per call at most: larger batches go in chunks of queries


def l1_knn(x: torch.Tensor, q: torch.Tensor, k: int, config: dict | None = None, xt: torch.Tensor | None = None):
    """Each query's k nearest rows of x by L1 (int64 indices, nearest first, ties by index): with the block sizes
    and selection of ``config`` ({"BLOCK_Q", "BLOCK_N", "BLOCK_D", "SORT"}, plus "num_warps"), or else the
    fastest Triton's auto-tuner finds. With ``xt`` (x transposed, d x n, contiguous), the 2D-tile kernel."""
    nq, d = q.shape
    n = x.shape[0]
    tiles = xt is not None
    kernel, tuned = (_l1_tile_topk, _tiles_tuned) if tiles else (_l1_block_topk, _l1_tuned)
    base = xt if tiles else x
    width = triton.cdiv(n, MIN_BLOCK_N) * k         # room for the smallest blocks' survivors; unused slots stay MAX
    step = max(64, SURVIVOR_BYTES // (width * 8) // 64 * 64)
    out = []
    for s in range(0, nq, step):
        qs = q[s:s + step]
        survivors = torch.full((len(qs), width), MAX_KEY, dtype=torch.int64, device=x.device)
        m = len(qs)
        qa = qs.T.contiguous() if tiles else qs
        if config is None:
            def grid(meta, m=m):
                return (triton.cdiv(m, meta["BLOCK_Q"]), triton.cdiv(n, meta["BLOCK_N"]))
            tuned[grid](qa, base, survivors, m, n, d, width, K=k)
        else:
            cfg = dict(config)
            warps = cfg.pop("num_warps", 4)
            grid = (triton.cdiv(m, cfg["BLOCK_Q"]), triton.cdiv(n, cfg["BLOCK_N"]))
            kernel[grid](qa, base, survivors, m, n, d, width, K=k, num_warps=warps, **cfg)
        out.append(torch.topk(survivors, k, dim=1, largest=False, sorted=True).values & 0xFFFFFFFF)
    return torch.cat(out)


def check(both: bool = False) -> None:
    """Against exact search by NumPy, ties broken by index, on sizes that leave partial blocks: on the GPU where
    there is one (the compiled kernel), otherwise on the CPU through Triton's interpreter."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  on {dev}", flush=True)
    rng = np.random.default_rng(0)
    for n, d, nq, k in ((1000, 24, 37, 10), (777, 13, 5, 3), (300, 8, 20, 10)):
        x = rng.integers(0, 4, (n, d)).astype(np.float32)          # small integers: many ties
        q = rng.integers(0, 4, (nq, d)).astype(np.float32)
        dist = np.abs(q[:, None, :] - x[None, :, :]).sum(axis=2)
        truth = np.lexsort((np.broadcast_to(np.arange(n), dist.shape), dist), axis=1)[:, :k]
        xd, qd = torch.from_numpy(x).to(dev), torch.from_numpy(q).to(dev)
        xtd = xd.T.contiguous()
        same = []
        for c in CONFIGS_2D:                                       # every configuration the auto-tuner may pick
            cfg = {**c.kwargs, "num_warps": c.num_warps}
            same.append(np.array_equal(l1_knn(xd, qd, k, cfg, xt=xtd).cpu().numpy(), truth))
            print("." if same[-1] else "x", end="", flush=True)            # one mark per configuration
        if both:
            for c in CONFIGS:
                cfg = {**c.kwargs, "num_warps": c.num_warps}
                same.append(np.array_equal(l1_knn(xd, qd, k, cfg).cpu().numpy(), truth))
                print("." if same[-1] else "x", end="", flush=True)
        print(f"\n  n {n}, d {d}, {nq} queries, k {k}: identical to exact search, ties included, in "
              f"{sum(same)} of {len(same)} configurations")
        assert all(same)


class Peaks:
    """Peak memory during a block, in MB above the level at its start: the device's (all allocators, polled
    through NVIDIA's management library) and the host process's; None where a library is missing."""

    def __init__(self, interval: float = 0.01):
        self.interval = interval
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        except Exception:  # noqa: BLE001 (no NVML: device peaks unknown)
            self._nvml = None
        try:
            import psutil
            self._proc = psutil.Process()
        except Exception:  # noqa: BLE001
            self._proc = None

    def device_used(self):
        return None if self._nvml is None else self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used / 2**20

    def _host(self):
        return None if self._proc is None else self._proc.memory_info().rss / 2**20

    def __enter__(self):
        self.d0, self.h0 = self.device_used(), self._host()
        self.dmax, self.hmax = self.d0, self.h0
        self._stop = threading.Event()

        def poll():
            while not self._stop.is_set():
                d, h = self.device_used(), self._host()
                if d is not None:
                    self.dmax = max(self.dmax, d)
                if h is not None:
                    self.hmax = max(self.hmax, h)
                time.sleep(self.interval)
        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        d, h = self.device_used(), self._host()             # one last reading, after the work
        self.device = None if d is None else max(self.dmax, d) - self.d0
        self.host = None if h is None else max(self.hmax, h) - self.h0


def _mb(v):
    return "?" if v is None else f"{v:,.0f}"


SLOW = 20.0      # seconds: a call slower than this is not repeated, and that contender's larger batches skipped


def bench(n: int, d: int, batches, config_2d: dict | None = None) -> None:
    dev = torch.device("cuda")
    rng = np.random.default_rng(0)
    xb = rng.standard_normal((n, d), dtype=np.float32)
    xq = rng.standard_normal((max(batches), d), dtype=np.float32)
    x, q = torch.from_numpy(xb).to(dev), torch.from_numpy(xq).to(dev)
    k = 10
    print(f"{torch.cuda.get_device_name()}: {n:,} Gaussian vectors, {d} dimensions, L1, k = {k}; queries per second", flush=True)

    start = time.perf_counter()

    def elapsed():
        return f"{(time.perf_counter() - start) / 60:.1f} min so far"

    def timed(fn, qb):
        """Queries per second, best of two calls after a warm-up; a slow contender (a warm-up past SLOW s) is
        timed by its warm-up alone, and flagged so that its larger batches are skipped."""
        t = time.perf_counter()
        fn(qb)
        torch.cuda.synchronize()
        warm = time.perf_counter() - t
        if warm > SLOW:
            return len(qb) / warm, True
        best = warm
        for _ in range(2):
            t = time.perf_counter()
            fn(qb)
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t)
        return len(qb) / best, False

    def cdist(qb):
        out = []
        for s in range(0, len(qb), 64):                            # cdist's broadcast exceeds memory beyond
            out.append(torch.topk(torch.cdist(qb[s:s + 64], x, p=1), k, largest=False).indices)
        return torch.cat(out)
    xt = x.T.contiguous()                                          # the 2D-tile kernel's layout, made once
    # the 2D kernel's configurations timed one by one, each printed as it is measured (Triton's auto-tuner says
    # nothing until all are done); the 3D kernel at the configuration its auto-tuner chose on the T4
    best_cfg, best_t = None, float("inf")
    configs = [{**c.kwargs, "num_warps": c.num_warps} for c in CONFIGS_2D]
    if config_2d is not None:
        configs = [config_2d]                                       # tuning skipped: a configuration given
    print(f"  tuning the 2D-tile kernel on 1,000 queries: {len(configs)} configurations, each compiled, run once "
          f"and timed once (seconds per call; a configuration past {SLOW} s on its first call is not timed)",
          flush=True)
    for i, cfg in enumerate(configs, 1):
        try:
            t = time.perf_counter()
            l1_knn(x, q[:1000], k, cfg, xt=xt)                     # compiles
            torch.cuda.synchronize()
            first = time.perf_counter() - t
            if first > SLOW:
                print(f"    [{i}/{len(configs)}] {cfg}: too slow ({first:.1f} s with compiling); {elapsed()}",
                      flush=True)
                continue
            t = time.perf_counter()
            l1_knn(x, q[:1000], k, cfg, xt=xt)
            torch.cuda.synchronize()
            t = time.perf_counter() - t
        except Exception as e:  # noqa: BLE001 (a configuration the GPU cannot run: skip it)
            print(f"    [{i}/{len(configs)}] {cfg}: failed ({type(e).__name__}); {elapsed()}", flush=True)
            continue
        print(f"    [{i}/{len(configs)}] {cfg}: {t:.3f}; {elapsed()}", flush=True)
        if t < best_t:
            best_cfg, best_t = cfg, t
    print(f"  fastest 2D-tile configuration: {best_cfg}", flush=True)
    best_3d = {"BLOCK_Q": 64, "BLOCK_N": 128, "BLOCK_D": 2, "SORT": True, "num_warps": 4}
    contenders = {"siann, 2D tiles, tuned": lambda qb: l1_knn(x, qb, k, best_cfg, xt=xt),
                  "siann, 3D tiles, tuned": lambda qb: l1_knn(x, qb, k, best_3d),
                  "torch.cdist + topk": cdist}
    only_small = {"torch.cdist + topk"}                             # ~90 queries/s on a T4: batch 100 only
    try:
        import faiss
        import faiss.contrib.torch_utils  # lets FAISS take PyTorch tensors
        res = faiss.StandardGpuResources()
        contenders["FAISS GPU brute force"] = lambda qb: faiss.knn_gpu(res, qb, x, k, metric=faiss.METRIC_L1)
    except Exception as e:  # noqa: BLE001 (FAISS's GPU build missing: skip it)
        print(f"  (FAISS GPU skipped: {type(e).__name__}: {e})")
    try:
        from cuvs.neighbors import brute_force
        built = {}

        def cuvs_search(qb):
            if "index" not in built:                                # built at the first call: counted as kept
                built["index"] = brute_force.build(x, metric="l1")
            return brute_force.search(built["index"], qb, k)
        contenders["cuVS brute force"] = cuvs_search
    except Exception as e:  # noqa: BLE001 (cuVS missing: skip it)
        print(f"  (cuVS skipped: {type(e).__name__}: {e})")
    if "FAISS GPU brute force" in contenders:
        _, fi = contenders["FAISS GPU brute force"](q[:100])
        fi = torch.as_tensor(fi).cpu().numpy()
        for name, ours in (("2D tiles", l1_knn(x, q[:100], k, best_cfg, xt=xt)),
                           ("3D tiles", l1_knn(x, q[:100], k, best_3d))):
            same = (np.sort(fi, 1) == np.sort(ours.cpu().numpy(), 1)).mean()
            print(f"  {name}: the same neighbors as FAISS's on 100 queries: {same:.4f}")
    peaks = Peaks()

    def settle():
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()                                    # PyTorch's cached blocks back to the device

    def memory(fn, qb):
        """PyTorch's peak, the device's and the host's during one call, MB above the start."""
        settle()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.memory_allocated()
        with peaks as p:
            fn(qb)
            torch.cuda.synchronize()
        return (torch.cuda.max_memory_allocated() - start) / 2**20, p.device, p.host

    print("  queries per second; then memory per call, MB: PyTorch's peak / the device's peak / the host's peak",
          flush=True)
    print(f"  {len(contenders)} contenders x {len(batches)} batch sizes; '-' where a batch was skipped",
          flush=True)
    print(f"  {'':32s}" + "".join(f"{f'batch {b:,}':>20s}" for b in batches), flush=True)
    for row, (name, fn) in enumerate(contenders.items(), 1):
        settle()
        before = peaks.device_used()
        print(f"  [{row}/{len(contenders)}] {name:26s}", end="", flush=True)
        mems, slow = [], False
        for b in batches:
            if slow or (name in only_small and b > batches[0]):
                print(f"{'-':>20s}", end="", flush=True)
                mems.append(None)
                continue
            try:
                qps, slow = timed(fn, q[:b])
                print(f"{qps:20,.0f}", end="", flush=True)
                mems.append(None if slow else memory(fn, q[:b]))
            except Exception as e:  # noqa: BLE001 (report and go on: one failure should not hide the rest)
                print(f"\n    batch {b:,} failed: {type(e).__name__}: {str(e)[:200]}", end="")
                mems.append(None)
        print(f"   ({elapsed()})", flush=True)
        cells = ["" if m is None else f"{_mb(m[0])} / {_mb(m[1])} / {_mb(m[2])}" for m in mems]
        print(f"  {'      memory':32s}" + "".join(f"{c:>20s}" for c in cells), flush=True)
        settle()
        after = peaks.device_used()
        kept = None if before is None else after - before
        print(f"  {'      kept on the device':32s}{_mb(kept):>20s} MB", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--all", action="store_true", help="with --check, every configuration of both kernels")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--batches", default="100,1000,10000")
    p.add_argument("--config", help="the 2D kernel's configuration, skipping its tuning: BLOCK_Q,BLOCK_N,sort|min,warps")
    a = p.parse_args()
    if a.check:
        check(a.all)
    if a.bench:
        cfg = None
        if a.config:
            bq, bn, sel, w = a.config.split(",")
            cfg = {"BLOCK_Q": int(bq), "BLOCK_N": int(bn), "SORT": sel == "sort", "num_warps": int(w)}
        bench(a.n, a.d, tuple(map(int, a.batches.split(","))), cfg)


if __name__ == "__main__":
    main()
