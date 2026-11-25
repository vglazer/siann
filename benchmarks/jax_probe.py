"""A probe of exact search on a Google TPU through JAX: built-in operations for every metric, and a custom kernel
for L1 in Pallas (JAX's kernel language, which compiles for TPUs and NVIDIA GPUs, and which PyTorch reaches on
TPUs through TorchTPU).

Built-in: L2 and inner product as a matrix multiply followed by lax.top_k, or by lax.approx_min_k (the TPU's own
approximate top-k, at a recall target of 0.95), at the TPU's default matrix-multiply precision (bfloat16 inputs)
and at full float32 (HIGHEST); L1 as a broadcast difference summed over the coordinates, a chunk of the base at a
time, with the best k kept across chunks (lax.scan).

Pallas, L1: a grid of (query block, base block) programs; each computes its tile of L1 distances in the TPU's
on-chip memory and merges it into its query block's best k, kept in the output across the base blocks, by k
passes of a minimum (ties by index, as siann's other kernels), so no distance matrix ever reaches memory.

Every result is checked against exact search in NumPy on a sample of queries (recall@k and the share of queries
whose neighbors are exactly NumPy's). Memory: the device's bytes in use and its peak so far, where it reports them.

    python jax_probe.py                                   # on a TPU (Colab: Runtime, Change runtime type, TPU)
    python jax_probe.py --check                           # the Pallas kernel against NumPy, small shapes
    python jax_probe.py --check --interpret               # the same anywhere, through Pallas's TPU interpreter
"""
import argparse
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CHUNK = 256                     # queries per matrix multiply (256 x 1,000,000 float32 scores is 1 GB)
L1_CHUNK = 16_384               # base vectors per step of the built-in L1 scan
KP = 128                        # the kept best-k, padded to the TPU's 128 lanes
BIG = 3.0e38                    # "no candidate" (finite: inf - inf would be NaN in a padded row)
PAD = 1.0e30                    # padding rows of the base: never nearer than a real row
SLOW = 20.0                     # seconds: a call slower than this is not repeated, and its row's larger batches skipped
PALLAS_BLOCKS = ((8, 512), (16, 512))                                       # --full: the five below
PALLAS_BLOCKS_FULL = ((8, 512), (8, 1024), (16, 512), (32, 256), (8, 2048))


def exact_numpy(xb, xq, k, metric):
    """NumPy's k nearest (int64 indices, ties by index): the reference."""
    out = []
    sq = (xb.astype(np.float64) ** 2).sum(axis=1) if metric == "l2" else None
    for q in xq:
        if metric == "l2":
            d = sq - 2 * (xb @ q).astype(np.float64)             # the rank of |x - q|^2, without a copy of xb
        elif metric == "ip":
            d = -(xb @ q)
        else:
            d = np.abs(xb - q).sum(axis=1)
        part = np.argpartition(d, k)[:k + 1] if len(d) > k + 1 else np.arange(len(d))
        cand = np.flatnonzero(d <= d[part].max())
        out.append(cand[np.lexsort((cand, d[cand]))][:k])
    return np.array(out)


# --- Pallas: L1 with the selection fused -------------------------------------------------------------------------

def _l1_kernel(q_ref, x_ref, v_ref, i_ref, *, k, block_n):
    j = pl.program_id(1)

    @pl.when(j == 0)
    def _():
        v_ref[...] = jnp.full(v_ref.shape, BIG, jnp.float32)
        i_ref[...] = jnp.full(i_ref.shape, jnp.iinfo(jnp.int32).max, jnp.int32)

    q, x = q_ref[...], x_ref[...]
    d = jnp.sum(jnp.abs(q[:, None, :] - x[None, :, :]), axis=2)              # (BQ, BN), on chip
    di = j * block_n + lax.broadcasted_iota(jnp.int32, d.shape, 1)
    kv, ki = v_ref[...], i_ref[...]                                            # (BQ, KP), the best so far
    lane = lax.broadcasted_iota(jnp.int32, kv.shape, 1)
    nv = jnp.full(kv.shape, BIG, jnp.float32)
    ni = jnp.full(ki.shape, jnp.iinfo(jnp.int32).max, jnp.int32)
    for p in range(k):                                                         # k passes of a minimum
        m = jnp.minimum(jnp.min(kv, axis=1, keepdims=True), jnp.min(d, axis=1, keepdims=True))
        top = jnp.iinfo(jnp.int32).max
        mi = jnp.minimum(jnp.min(jnp.where(kv == m, ki, top), axis=1, keepdims=True),
                         jnp.min(jnp.where(d == m, di, top), axis=1, keepdims=True))   # ties: the smaller index
        nv = jnp.where(lane == p, m, nv)
        ni = jnp.where(lane == p, mi, ni)
        kv = jnp.where((kv == m) & (ki == mi), BIG, kv)
        d = jnp.where((d == m) & (di == mi), BIG, d)
    v_ref[...] = nv
    i_ref[...] = ni


@functools.partial(jax.jit, static_argnames=("k", "block_q", "block_n", "interpret"))
def pallas_l1(x, q, k, block_q=8, block_n=512, interpret=False):
    """Each query's k nearest rows of x by L1 (int32 indices, nearest first, ties by index), through the kernel.
    x must already be padded to a multiple of block_n (pad_base)."""
    nq, d = q.shape
    n = x.shape[0]
    mq = -(-nq // block_q) * block_q
    q = jnp.pad(q, ((0, mq - nq), (0, 0)))
    grid = (mq // block_q, n // block_n)
    params = pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary"))
    _, i = pl.pallas_call(
        functools.partial(_l1_kernel, k=k, block_n=block_n),
        grid=grid,
        in_specs=[pl.BlockSpec((block_q, d), lambda a, b: (a, 0)),
                  pl.BlockSpec((block_n, d), lambda a, b: (b, 0))],
        out_specs=[pl.BlockSpec((block_q, KP), lambda a, b: (a, 0)),
                   pl.BlockSpec((block_q, KP), lambda a, b: (a, 0))],
        out_shape=[jax.ShapeDtypeStruct((mq, KP), jnp.float32), jax.ShapeDtypeStruct((mq, KP), jnp.int32)],
        compiler_params=params,
        interpret=pltpu.InterpretParams() if interpret else False,
    )(q, x)
    return i[:nq, :k]


def pad_base(x, block_n):
    n = x.shape[0]
    m = -(-n // block_n) * block_n
    return jnp.pad(x, ((0, m - n), (0, 0)), constant_values=PAD)


# --- built-in operations -----------------------------------------------------------------------------------------

def builtin_searches(x, k, full=False):
    sq = jnp.sum(x * x, axis=1)

    def scores(q, metric, precision):
        s = jnp.matmul(q, x.T, precision=precision)
        return sq[None, :] - 2 * s if metric == "l2" else -s

    def exact(q, metric, precision):
        return lax.top_k(-scores(q, metric, precision), k)[1]

    def approx(q, metric, precision):
        return lax.approx_min_k(scores(q, metric, precision), k, recall_target=0.95)[1]

    precisions = {"default": lax.Precision.DEFAULT, "highest": lax.Precision.HIGHEST}
    if full:
        plan = [(m, how, p) for m in ("l2", "ip") for p in precisions for how in ("top_k", "approx_min_k")]
    else:                                   # the short default: what differs, once each
        plan = [("l2", "top_k", "default"), ("l2", "top_k", "highest"), ("l2", "approx_min_k", "default"),
                ("ip", "top_k", "default")]
    out = {}
    for metric, how, pname in plan:
        fn = exact if how == "top_k" else approx
        out[(metric, f"matmul + {how}, precision {pname}")] = jax.jit(
            functools.partial(fn, metric=metric, precision=precisions[pname]))
    return out


def builtin_l1(x, k):
    n, d = x.shape
    m = -(-n // L1_CHUNK) * L1_CHUNK
    xs = jnp.pad(x, ((0, m - n), (0, 0)), constant_values=PAD).reshape(m // L1_CHUNK, L1_CHUNK, d)

    @jax.jit
    def search(q):
        def step(carry, t):
            bv, bi = carry
            xc, base = t
            dist = jnp.sum(jnp.abs(q[:, None, :] - xc[None, :, :]), axis=2)
            v, i = lax.top_k(-dist, k)
            v, i = jnp.concatenate([bv, -v], 1), jnp.concatenate([bi, i + base], 1)
            top = lax.top_k(-v, k)[1]
            return (jnp.take_along_axis(v, top, 1), jnp.take_along_axis(i, top, 1)), None
        init = (jnp.full((q.shape[0], k), BIG, jnp.float32), jnp.zeros((q.shape[0], k), jnp.int32))
        (_, bi), _ = lax.scan(step, init, (xs, jnp.arange(m // L1_CHUNK, dtype=jnp.int32) * L1_CHUNK))
        return bi
    return search


def run_chunks(fn, q, size):
    """fn over q in chunks of one fixed size (the last one padded), so each jitted fn compiles once."""
    out = []
    for s in range(0, q.shape[0], size):
        qs = q[s:s + size]
        if qs.shape[0] < size:
            qs = jnp.concatenate([qs, jnp.broadcast_to(qs[:1], (size - qs.shape[0], qs.shape[1]))])
        out.append(fn(qs)[:min(size, q.shape[0] - s)])
    return jnp.concatenate(out)


def memory():
    """The device's bytes in use and its peak so far, MB (None where it does not say)."""
    try:
        st = jax.devices()[0].memory_stats() or {}
    except Exception:  # noqa: BLE001
        st = {}
    get = st.get
    return tuple(None if get(key) is None else get(key) / 2**20 for key in ("bytes_in_use", "peak_bytes_in_use"))


def check(interpret: bool) -> None:
    rng = np.random.default_rng(1)
    print(f"  on {jax.devices()[0].platform}{', through the TPU interpreter' if interpret else ''}", flush=True)
    for n, d, nq, k, bq, bn in ((1000, 128, 37, 10, 8, 128), (777, 128, 5, 3, 8, 256), (2048, 128, 20, 10, 16, 512)):
        x = rng.integers(0, 4, (n, d)).astype(np.float32)              # small integers: many ties
        q = rng.integers(0, 4, (nq, d)).astype(np.float32)
        truth = exact_numpy(x, q, k, "l1")
        got = np.asarray(pallas_l1(pad_base(jnp.asarray(x), bn), jnp.asarray(q), k, bq, bn, interpret))
        same = np.array_equal(got, truth)
        print(f"  n {n}, d {d}, {nq} queries, k {k}, blocks {bq} x {bn}: identical to exact search, ties "
              f"included: {same}", flush=True)
        assert same


def bench(n: int, d: int, batches, l1_batch: int, check_queries: int, full: bool = False) -> None:
    k = 10
    dev = jax.devices()[0]
    rng = np.random.default_rng(0)
    xb = rng.standard_normal((n, d), dtype=np.float32)
    xq = rng.standard_normal((max(batches), d), dtype=np.float32)
    print(f"JAX {jax.__version__} on {dev.device_kind} ({dev.platform}): {n:,} Gaussian vectors, {d} dimensions, "
          f"k = {k}", flush=True)
    t = time.perf_counter()
    truth = {m: exact_numpy(xb, xq[:check_queries], k, m) for m in ("l2", "ip")}
    truth["l1"] = exact_numpy(xb, xq[:min(check_queries, l1_batch)], k, "l1")
    print(f"  NumPy's exact neighbors for the check: {time.perf_counter() - t:.1f} s", flush=True)
    x, q = jnp.asarray(xb), jnp.asarray(xq)

    start = time.perf_counter()

    def elapsed():
        return f"{(time.perf_counter() - start) / 60:.1f} min so far"

    def timed(call, qb):
        """Queries per second, best of two after a warm-up; a call past SLOW s is timed by its warm-up alone and
        flagged slow (its row's larger batches are skipped)."""
        t = time.perf_counter()
        call(qb).block_until_ready()                                   # compiles, warms up
        warm = time.perf_counter() - t
        if warm > SLOW:
            return qb.shape[0] / warm, True
        best = warm
        for _ in range(2):
            t = time.perf_counter()
            call(qb).block_until_ready()
            best = min(best, time.perf_counter() - t)
        return qb.shape[0] / best, False

    def agreement(found, ref):
        found = np.asarray(found)[:len(ref)]
        recall = np.mean([len(set(f) & set(r)) / len(r) for f, r in zip(found, ref, strict=True)])
        return recall, np.mean([np.array_equal(f, r) for f, r in zip(found, ref, strict=True)])

    def mem():
        used, peak = memory()
        return f"{'?' if used is None else f'{used:,.0f}'} / {'?' if peak is None else f'{peak:,.0f}'}"

    searches = builtin_searches(x, k, full)
    blocks = PALLAS_BLOCKS_FULL if full else PALLAS_BLOCKS
    rows = len(searches) + 1 + len(blocks)
    print(f"  queries per second; then the device's memory after the row, MB: in use / peak so far; then recall@10 "
          f"and exact matches against NumPy. {rows} rows; '-' where a batch was skipped (a call past {SLOW:.0f} s "
          f"skips its row's larger batches)", flush=True)
    print(f"  {'':50s}" + "".join(f"{f'batch {b:,}':>14s}" for b in batches), flush=True)
    row = 0
    for (metric, how), fn in searches.items():
        def call(qb, fn=fn):
            return run_chunks(fn, qb, CHUNK)
        row += 1
        print(f"  {f'[{row}/{rows}] {metric}, {how}':50s}", end="", flush=True)
        try:
            slow = False
            for b in batches:
                if slow:
                    print(f"{'-':>14s}", end="", flush=True)
                    continue
                qps, slow = timed(call, q[:b])
                print(f"{qps:14,.0f}", end="", flush=True)
            recall, same = agreement(call(q[:check_queries]), truth[metric])
            print(f"   ({elapsed()})\n      memory {mem()}; recall@10 {recall:.4f}, exact matches {same:.4f}",
                  flush=True)
        except Exception as e:  # noqa: BLE001 (report and go on: one failure should not hide the rest)
            print(f"\n      failed: {type(e).__name__}: {str(e)[:300]}", flush=True)

    qb = q[:l1_batch]
    print(f"\n  L1, batch {l1_batch} (queries per second)", flush=True)
    search = builtin_l1(x, k)
    row += 1
    try:
        def call_builtin(qq):
            return run_chunks(search, qq, 16)
        qps, _ = timed(call_builtin, qb)
        recall, same = agreement(call_builtin(qb), truth["l1"])
        print(f"    [{row}/{rows}] built-in tiles (lax.scan){qps:18,.0f}   memory {mem()}; recall@10 {recall:.4f}, "
              f"exact matches {same:.4f}; {elapsed()}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"    built-in tiles failed: {type(e).__name__}: {str(e)[:300]}", flush=True)
    for bq, bn in blocks:
        row += 1
        try:
            xp = pad_base(x, bn)

            def call_pallas(qq, xp=xp, bq=bq, bn=bn):
                return pallas_l1(xp, qq, k, bq, bn)
            qps, _ = timed(call_pallas, qb)
            recall, same = agreement(call_pallas(qb), truth["l1"])
            print(f"    [{row}/{rows}] Pallas, blocks {bq:>2} x {bn:<5}{qps:18,.0f}   memory {mem()}; recall@10 "
                  f"{recall:.4f}, exact matches {same:.4f}; {elapsed()}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"    [{row}/{rows}] Pallas, blocks {bq} x {bn} failed: {type(e).__name__}: {str(e)[:300]}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--interpret", action="store_true", help="with --check: through Pallas's TPU interpreter")
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--batches", default="100,1000")
    p.add_argument("--l1-batch", type=int, default=100)
    p.add_argument("--check-queries", type=int, default=20)
    p.add_argument("--full", action="store_true", help="every search, precision and Pallas block shape")
    a = p.parse_args()
    if a.check:
        check(a.interpret)
    else:
        bench(a.n, a.d, tuple(map(int, a.batches.split(","))), a.l1_batch, a.check_queries, a.full)


if __name__ == "__main__":
    main()
