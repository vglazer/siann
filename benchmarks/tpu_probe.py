"""A probe of exact search on a Google TPU through PyTorch's built-in operations (no custom kernels: Triton has
no TPU target): L2 and inner product as a matrix multiply followed by a top-k, the top-k taken once over each
query's whole row or in two stages (the best k of each block of the base, then the best k of those, siann's
two-stage design), and L1 as a broadcast difference summed over the coordinates, kept to the best k per chunk of
the base. Each runs eagerly and through torch.compile, and the matrix multiplies at the TPU's default precision
(bfloat16 inputs, one pass) and at full float32 ("highest", six passes).

Every result is checked against exact search in NumPy on a sample of queries (recall@k, and the share of
queries whose k neighbors are exactly NumPy's). Speeds are queries per second on Gaussian vectors; memory is the
accelerator's peak where its backend reports one, and the host process's peak.

The TPU is reached through TorchTPU (``torch_tpu``, the device "tpu") where it is installed, else through
PyTorch/XLA (``torch_xla``). ``--device cpu`` runs the same code on PyTorch's CPU device, to check it anywhere.

    python tpu_probe.py                                  # on a TPU (Colab: Runtime, Change runtime type, TPU)
    python tpu_probe.py --device cpu --n 20000 --batches 100,300
"""
import argparse
import contextlib
import threading
import time

import numpy as np
import torch

CHUNK = 256          # queries per matrix multiply: 256 x 1,000,000 float32 scores is 1 GB
BLOCK = 1000         # the first stage's block of the base (n must be a multiple)
L1_Q, L1_N = 8, 31_250   # L1's tile: 8 queries x 31,250 vectors x 128 coordinates is 128 MB
SLOW = 20.0               # seconds: a call slower than this is not repeated, and its row's larger batches skipped


class Backend:
    """The accelerator: its device, how to wait for it, its torch.compile backend and its memory counters."""

    def __init__(self, want: str):
        self.name, self.compile_backend, self._xm = "cpu", None, None
        self.device = torch.device("cpu")
        if want in ("auto", "tpu"):
            try:
                try:
                    import torch_tpu  # noqa: F401 (registers the "tpu" device where autoloading does not)
                except ImportError:
                    pass
                torch.ones(1, device="tpu")
                self.name, self.device, self.compile_backend = "torchtpu", torch.device("tpu"), "tpu"
                return
            except Exception as e:  # noqa: BLE001 (TorchTPU missing or unusable here: try PyTorch/XLA)
                print(f"  (TorchTPU not usable: {type(e).__name__}: {str(e)[:200]})", flush=True)
            try:
                import torch_xla
                import torch_xla.core.xla_model as xm
                self.name, self.device, self.compile_backend, self._xm = "torch_xla", torch_xla.device(), "openxla", xm
                return
            except Exception as e:  # noqa: BLE001
                print(f"  (PyTorch/XLA not usable: {type(e).__name__}: {str(e)[:200]})", flush=True)
            if want == "tpu":
                raise SystemExit("no TPU backend: install torch_tpu (or torch_xla) on a TPU runtime")

    def describe(self) -> str:
        if self.name == "torchtpu":
            try:
                p = torch.tpu.get_device_properties(0)
                return f"TorchTPU, {getattr(p, 'name', '?')} ({getattr(p, 'version', '?')}), " \
                       f"{getattr(p, 'hbm_bytes_per_chip', 0) / 2**30:.0f} GB HBM"
            except Exception:  # noqa: BLE001
                return "TorchTPU"
        if self.name == "torch_xla":
            try:
                import torch_xla
                return f"PyTorch/XLA {torch_xla.__version__}, {self._xm.xla_device_hw(self.device)}"
            except Exception:  # noqa: BLE001
                return "PyTorch/XLA"
        return "PyTorch's CPU device"

    def sync(self) -> None:
        if self.name == "torchtpu":
            torch.accelerator.synchronize()
        elif self.name == "torch_xla":
            import torch_xla
            torch_xla.sync()
            self._xm.wait_device_ops()

    def reset_peak(self) -> None:
        if self.name == "torchtpu":
            with contextlib.suppress(Exception):          # no peak counter: the peaks read None
                torch.accelerator.reset_peak_memory_stats()

    def peak_mb(self):
        """The accelerator's peak allocation since the last reset (TorchTPU), or since the start (PyTorch/XLA)."""
        try:
            if self.name == "torchtpu":
                return torch.accelerator.max_memory_allocated() / 2**20
            if self.name == "torch_xla":
                info = self._xm.get_memory_info(self.device)
                peak = info.get("peak_bytes_used", info.get("bytes_used"))
                return None if peak is None else peak / 2**20
        except Exception:  # noqa: BLE001
            return None
        return None


class HostPeak:
    """The host process's peak resident memory during a block, MB above its start (polled every 10 ms)."""

    def __enter__(self):
        try:
            import psutil
            self._proc = psutil.Process()
        except ImportError:
            self._proc, self.mb = None, None
            return self
        self.start = self.top = self._proc.memory_info().rss
        self._stop = threading.Event()

        def poll():
            while not self._stop.is_set():
                self.top = max(self.top, self._proc.memory_info().rss)
                time.sleep(0.01)
        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._proc is not None:
            self._stop.set()
            self._thread.join()
            self.mb = (max(self.top, self._proc.memory_info().rss) - self.start) / 2**20


def exact_numpy(xb, xq, k, metric):
    """NumPy's k nearest (int64 indices, ties by index) for a sample of queries: the reference."""
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


def make_searches(x, k, full=False):
    """Each search maps a fixed-size chunk of queries to its k nearest indices (int64, on the device)."""
    n = x.shape[0]
    sq = (x * x).sum(dim=1)

    def scores(q, metric):
        s = q @ x.T
        return sq[None, :] - 2 * s if metric == "l2" else -s

    def one_stage(q, metric):
        return torch.topk(scores(q, metric), k, dim=1, largest=False).indices

    def two_stage(q, metric):
        s = scores(q, metric).view(q.shape[0], n // BLOCK, BLOCK)
        v, i = torch.topk(s, k, dim=2, largest=False)               # the best k of each block
        i = i + (torch.arange(n // BLOCK, device=x.device) * BLOCK)[None, :, None]
        v, i = v.reshape(q.shape[0], -1), i.reshape(q.shape[0], -1)
        best = torch.topk(v, k, dim=1, largest=False).indices
        return torch.gather(i, 1, best)

    def l1_tile(q, xs):
        d = (q[:, None, :] - xs[None, :, :]).abs().sum(dim=2)
        return torch.topk(d, k, dim=1, largest=False)

    searches = {
        ("l2", "one-stage top-k"): lambda q: one_stage(q, "l2"),
        ("l2", "two-stage top-k"): lambda q: two_stage(q, "l2"),
        ("ip", "one-stage top-k"): lambda q: one_stage(q, "ip"),
        ("ip", "two-stage top-k"): lambda q: two_stage(q, "ip"),
    }
    if not full:                                                      # the short default
        del searches[("ip", "two-stage top-k")]
    return searches, l1_tile


def run_chunks(fn, q, size):
    """fn over q in chunks of a fixed size (the last one padded), so a compiled fn sees a single shape."""
    out = []
    for s in range(0, len(q), size):
        qs = q[s:s + size]
        if len(qs) < size:
            qs = torch.cat([qs, qs[:1].expand(size - len(qs), -1)])
        out.append(fn(qs)[:min(size, len(q) - s)])
    return torch.cat(out)


def run_l1(tile, x, q, k):
    """L1: each tile's best k, merged as the tiles go."""
    out = []
    n = x.shape[0]
    for s in range(0, len(q), L1_Q):
        qs = q[s:s + L1_Q]
        if len(qs) < L1_Q:
            qs = torch.cat([qs, qs[:1].expand(L1_Q - len(qs), -1)])
        bv = bi = None
        for t in range(0, n, L1_N):
            v, i = tile(qs, x[t:t + L1_N])
            i = i + t
            if bv is None:
                bv, bi = v, i
            else:
                v, i = torch.cat([bv, v], 1), torch.cat([bi, i], 1)
                top = torch.topk(v, k, dim=1, largest=False).indices
                bv, bi = torch.gather(v, 1, top), torch.gather(i, 1, top)
        out.append(bi[:min(L1_Q, len(q) - s)])
    return torch.cat(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto", choices=("auto", "tpu", "cpu"))
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--batches", default="100,1000")
    p.add_argument("--l1-batch", type=int, default=100, help="L1's single batch size (its tiles are slow)")
    p.add_argument("--check-queries", type=int, default=20)
    p.add_argument("--full", action="store_true", help="every search in every mode (eager and compiled, both precisions)")
    a = p.parse_args()
    if a.n % BLOCK or a.n % L1_N and a.n > L1_N:
        raise SystemExit(f"--n must be a multiple of {BLOCK} and of {L1_N} (or below {L1_N})")
    be = Backend(a.device)
    batches = tuple(map(int, a.batches.split(",")))
    k = 10
    rng = np.random.default_rng(0)
    xb = rng.standard_normal((a.n, a.d), dtype=np.float32)
    xq = rng.standard_normal((max(batches), a.d), dtype=np.float32)
    print(f"{be.describe()}: {a.n:,} Gaussian vectors, {a.d} dimensions, k = {k}", flush=True)
    t = time.perf_counter()
    truth = {m: exact_numpy(xb, xq[:a.check_queries], k, m) for m in ("l2", "ip")}
    truth["l1"] = exact_numpy(xb, xq[:min(a.check_queries, a.l1_batch)], k, "l1")
    print(f"  NumPy's exact neighbors for {a.check_queries} queries: {time.perf_counter() - t:.1f} s", flush=True)
    x, q = torch.from_numpy(xb).to(be.device), torch.from_numpy(xq).to(be.device)
    be.sync()

    def agreement(found, ref):
        found = found[:len(ref)]
        recall = np.mean([len(set(f) & set(r)) / len(r) for f, r in zip(found, ref, strict=True)])
        return recall, np.mean([np.array_equal(f, r) for f, r in zip(found, ref, strict=True)])

    start = time.perf_counter()

    def elapsed():
        return f"{(time.perf_counter() - start) / 60:.1f} min so far"

    def measure(call, qb):
        """Queries per second (best of two after a warm-up), the accelerator's peak and the host's; a call past
        SLOW s is timed by its warm-up alone and flagged slow (no memory run, no larger batches)."""
        t = time.perf_counter()
        call(qb)                                                      # compiles, warms up
        be.sync()
        warm = time.perf_counter() - t
        if warm > SLOW:
            return len(qb) / warm, None, None, True
        best = warm
        for _ in range(2):
            t = time.perf_counter()
            call(qb)
            be.sync()
            best = min(best, time.perf_counter() - t)
        be.reset_peak()
        with HostPeak() as h:
            call(qb)
            be.sync()
        return len(qb) / best, be.peak_mb(), h.mb, False

    def fmt_mem(dev, host):
        return f"{'?' if dev is None else f'{dev:,.0f}'} / {'?' if host is None else f'{host:,.0f}'}"

    modes = [(p, c) for p in ("medium", "highest") for c in (False, True) if not (c and be.name == "cpu")]
    if not a.full:                       # the short default: eager and compiled at the default, highest compiled
        modes = [m for m in modes if m != ("highest", False)] or modes[:1]
    per_mode = 4 if a.full else 3
    rows = len(modes) * per_mode + sum(1 for p, _ in modes if p == "medium")
    print(f"  queries per second, then memory per call, MB: the accelerator's peak / the host's peak; recall@10 and "
          f"exact matches against NumPy. {rows} rows; '-' where a batch was skipped (a call past {SLOW:.0f} s "
          f"skips its row's larger batches)", flush=True)
    row = 0
    for precision in ("medium", "highest"):
        torch.set_float32_matmul_precision(precision)
        searches, l1_tile = make_searches(x, k, a.full)
        for compiled in (False, True):
            if (precision, compiled) not in modes:
                continue
            if compiled:
                torch._dynamo.reset()
            mode = f"precision {precision}, {'compiled' if compiled else 'eager'}"
            print(f"\n  {mode}" + "".join(f"{f'batch {b:,}':>22s}" for b in batches), flush=True)
            for (metric, how), fn in searches.items():
                fn = torch.compile(fn, backend=be.compile_backend) if compiled else fn

                def call(qb, fn=fn):
                    return run_chunks(fn, qb, CHUNK)
                row += 1
                label = f"[{row}/{rows}] {metric}, {how}"
                print(f"    {label:30s}", end="", flush=True)
                mems, slow = [], False
                try:
                    found = call(q[:a.check_queries]).cpu().numpy()
                    for b in batches:
                        if slow:
                            print(f"{'-':>22s}", end="", flush=True)
                            mems.append("")
                            continue
                        qps, dev, host, slow = measure(call, q[:b])
                        print(f"{qps:22,.0f}", end="", flush=True)
                        mems.append(fmt_mem(dev, host))
                    recall, same = agreement(found, truth[metric])
                    print(f"   ({elapsed()})", flush=True)
                    print(f"    {'  memory':30s}" + "".join(f"{m:>22s}" for m in mems), flush=True)
                    print(f"      against NumPy on {len(truth[metric])} queries: recall@10 {recall:.4f}, "
                          f"exact matches {same:.4f}", flush=True)
                except Exception as e:  # noqa: BLE001 (report and go on: one failure should not hide the rest)
                    print(f"\n      failed: {type(e).__name__}: {str(e)[:300]}", flush=True)
            if precision == "medium":                                 # L1 has no matrix multiply: once per mode
                tile = torch.compile(l1_tile, backend=be.compile_backend) if compiled else l1_tile
                row += 1
                print(f"    {f'[{row}/{rows}] l1, tiles':30s}", end="", flush=True)
                try:
                    qb = q[:a.l1_batch]
                    qps, dev, host, _ = measure(lambda qq: run_l1(tile, x, qq, k), qb)
                    recall, same = agreement(run_l1(tile, x, qb, k).cpu().numpy(), truth["l1"])
                    print(f"{qps:22,.0f}  (batch {a.l1_batch}; memory {fmt_mem(dev, host)}; recall@10 {recall:.4f}, "
                          f"exact matches {same:.4f}; {elapsed()})", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"\n      failed: {type(e).__name__}: {str(e)[:300]}", flush=True)


if __name__ == "__main__":
    main()
