"""Profiles: the routing thresholds siann uses, measured on some data and machine, and siann.tune, which
measures them on the user's own data and hardware ("tuning"; an index's own "calibration" is finding the
settings that reach its recall target)."""
from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
IVF_BATCHES = (1, 4, 16, 64, 256, 1024, 4096)


@dataclass(frozen=True)
class Profile:
    """Where siann sends each search, and the measurements behind it.

    ``cpu_below[metric]`` is the batch size below which exact search by that metric runs on the CPU
    when a GPU is available: 1 means always the GPU, ``None`` always the CPU. ``rates`` holds the
    measured queries per second, as ``(metric, device, batch, rate)``. ``ivf``, if set, holds approximate
    search's thresholds for every index given the profile: ``cpu_below``, the batch size below which an IVF
    search runs on the CPU (a single query split over 4 threads), and ``gpu_list_from`` and
    ``cpu_list_from``, the batch sizes from which a scan reads each list once for all the queries that
    probe it, and ``cpu_blas_from``, the batch size from which the CPU's other scan finds each query's lists
    through BLAS (L2, cosine and inner product). Without them (siann's default), each index measures its own when it is built: they depend
    on how many lists its queries probe.
    """
    cpu_below: dict[str, int | None]
    source: str
    rates: tuple[tuple[str, str, int, float], ...] = ()
    meta: dict = field(default_factory=dict)
    ivf: dict | None = None

    def threshold(self, metric: str) -> int | None:
        """The batch size below which ``metric`` runs on the CPU (``None``: always the CPU)."""
        return self.cpu_below.get(metric, 1)

    def describe(self, metric: str) -> str:
        t = self.threshold(metric)
        if t is None:
            return "always on the CPU"
        if t <= 1:
            return "always on the GPU"
        return f"on the CPU below {t} queries per batch, on the GPU from {t}"

    def describe_ivf(self) -> str:
        if not self.ivf:
            return "measured by each index when it is built"
        from ._ivf import describe_routes
        return describe_routes(self.ivf, True)

    def __str__(self) -> str:
        lines = [f"siann.Profile ({self.source})"]
        lines += [f"  {m:7s} exact search {self.describe(m)}" for m in self.cpu_below]
        lines.append(f"  approximate search (IVF): {self.describe_ivf()}")
        if any(m != "ivf" for m, _, _, _ in self.rates):
            batches = sorted({b for m, _, b, _ in self.rates if m != "ivf"})
            rate = {(m, dev, b): r for m, dev, b, r in self.rates}
            lines.append("  queries per second, CPU / GPU, by batch size:")
            lines.append("    " + " " * 7 + "".join(f"{b:>16,}" for b in batches))
            for m in self.cpu_below:
                cells = [f"{rate.get((m, 'cpu', b), float('nan')):,.0f} / {rate.get((m, 'gpu', b), float('nan')):,.0f}"
                         for b in batches]
                lines.append(f"    {m:7s}" + "".join(f"{c:>16s}" for c in cells))
        ivf_rates = [r for r in self.rates if r[0] == "ivf"]
        if ivf_rates:
            batches = sorted({b for _, _, b, _ in ivf_rates})
            rate = {(p, b): r for _, p, b, r in ivf_rates}
            lines.append(f"  approximate search, queries per second by path and batch size "
                         f"({self.ivf.get('metric')}, recall {self.ivf.get('recall')}, nprobe {self.ivf.get('nprobe')}):")
            lines.append("    " + " " * 14 + "".join(f"{b:>10,}" for b in batches))
            for p in ("cpu-split", "cpu-query", "cpu-query-blas", "cpu-list", "gpu-query", "gpu-list"):
                if any((p, b) in rate for b in batches):
                    lines.append(f"    {p:14s}" + "".join(f"{rate[(p, b)]:>10,.0f}" if (p, b) in rate else f"{'':>10s}"
                                                       for b in batches))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """The profile as plain data (what ``save`` writes, as JSON)."""
        return {"source": self.source, "cpu_below": self.cpu_below, "meta": self.meta,
                "rates": [list(r) for r in self.rates], "ivf": self.ivf}

    @classmethod
    def from_dict(cls, data: dict) -> Profile:
        return cls(dict(data["cpu_below"]), data["source"],
                   tuple((m, d, int(b), float(r)) for m, d, b, r in data.get("rates", [])), data.get("meta", {}),
                   data.get("ivf"))

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path) -> Profile:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# Tuned by benchmarks/exact_routing.py on an Apple M3 Max (results/exact_routing.txt): for each metric,
# the smallest threshold across SIFT1M, GloVe-100 and DBpedia OpenAI (100,000 vectors), each moving from
# this default only where the other device was at least 10% faster. Single L2 queries go to the CPU;
# L1 and L-infinity are level on the two devices at 1-4 queries on a million vectors and stay on the GPU.
# Crossovers move with the collection's size: on smaller ones the CPU wins more small batches, which
# siann.tune measures (TUNING.md).
DEFAULT = Profile(
    cpu_below={"l2": 2, "l1": 1, "linf": 1, "cosine": 1, "ip": 1},
    source="siann's default, tuned on an Apple M3 Max with about a million vectors",
)

_current: Profile | None = None


def use_profile(profile: Profile | None) -> None:
    """Make ``profile`` the one every index uses unless given its own; ``None`` restores siann's
    default."""
    global _current
    if profile is not None and not isinstance(profile, Profile):
        raise TypeError("use_profile takes a siann.Profile (or None)")
    _current = profile


def current_profile() -> Profile:
    return _current if _current is not None else DEFAULT


def _rate(index, q: np.ndarray, k: int, device: str, deadline: float) -> float:
    index.search(q, k, device=device)                  # builds the device's copy, compiles for the shape
    times = []
    for _ in range(3):
        start = time.perf_counter()
        index.search(q, k, device=device)
        times.append(time.perf_counter() - start)
        if time.perf_counter() > deadline:
            break
    return len(q) / sorted(times)[len(times) // 2]


def threshold_with_margin(cpu: dict[int, float], gpu: dict[int, float], default: int | None) -> int | None:
    """The batch size from which exact search goes to the GPU, from measured rates: it moves from
    ``default`` (siann's default threshold for the metric; ``None``, always the CPU) only where the other
    device was at least 10% faster, as approximate search's routes do (``siann._ivf.MARGIN``). Between the
    batch sizes measured, and past the largest, the default holds; ``None`` means always the CPU."""
    from ._ivf import NEVER, crossover_with_margin
    t = crossover_with_margin(cpu, gpu, NEVER if default is None else default)
    return None if t >= NEVER else t


def _tune_ivf(xb, xq, given_xq: bool, metric: str, recall: float, k: int, gpu: str = "auto"):
    """Approximate search's thresholds: an IVF calibrated to ``recall`` (on ``xq`` if given), its routes
    measured as an index measures its own; returns (thresholds, rates as (label, path, batch, rate))."""
    from ._index import Index
    from ._ivf import measure_routes
    index = Index(xb, metric, recall=recall, kind="ivf", calibration_queries=xq[:1000] if given_xq else None,
                  profile=Profile({metric: 1}, "tuning"), gpu=gpu)
    routes, rates = measure_routes(index._ivf, xq, index.nprobe, index._gpu is not None, k,
                                   rounds=2 if index._gpu == "torch" else 1)
    thresholds = dict(routes, recall=recall, nprobe=index.nprobe, metric=metric)
    return thresholds, [("ivf", p, b, r) for (p, b), r in rates.items()]


def tune(xb, xq=None, metrics=None, batches=BATCHES, k: int = 10, budget: float = 60.0,
         seed: int = 0, recall: float | None = None, gpu: str = "auto") -> Profile:
    """Measures where exact search is faster, on the CPU or the GPU, for this data on this machine.

    ``xb`` is the user's base vectors (all of them: the crossover depends on the collection's size);
    ``xq`` the queries to time with, by default a sample of ``xb``. ``metrics`` defaults to all five.
    Each metric and batch size is timed on both devices (median of three after a warm-up that also
    compiles the GPU kernels for that shape), within about ``budget`` seconds in all. A threshold moves
    from siann's default only where the other device was at least 10% faster, about the timings' spread. With ``recall``,
    approximate search's thresholds are measured too: an IVF over ``xb`` (by the first metric) calibrated to
    that recall, on ``xq`` if given, its paths timed at batch sizes from 1 to 4,096. ``gpu`` chooses the GPU
    backend timed, as for ``siann.Index``. Returns a Profile;
    pass it to ``siann.Index(..., profile=p)`` or ``siann.use_profile(p)``.
    """
    from ._index import METRICS, Index, _resolve_gpu
    metrics = tuple(metrics) if metrics is not None else METRICS
    for m in metrics:
        if m not in METRICS:
            raise ValueError(f"unknown metric {m!r}")
    batches = sorted({int(b) for b in batches})
    xb = np.asarray(xb)
    given_xq = xq is not None
    if xq is None:
        rng = np.random.default_rng(seed)
        want = max(max(batches), max(IVF_BATCHES) if recall else 0)
        xq = xb[rng.choice(len(xb), size=min(len(xb), want), replace=len(xb) < want)]
    xq = np.asarray(xq)
    if len(xq) < max(batches):
        batches = [b for b in batches if b <= len(xq)]
    meta = {"n": len(xb), "d": int(xb.shape[1]), "k": int(k), "machine": platform.platform(),
            "date": time.strftime("%Y-%m-%d")}
    if _resolve_gpu(gpu) is None:
        # every exact search on the CPU; approximate search still chooses between the CPU's two scans
        ivf, ivf_rates = (_tune_ivf(xb, xq, given_xq, metrics[0], recall, k, gpu) if recall is not None
                          else (None, []))
        return Profile({m: None for m in metrics}, "tuned: no GPU found, every search on the CPU",
                       tuple(ivf_rates), meta, ivf)
    deadline = time.perf_counter() + budget
    rates, cpu_below = [], {}
    per_metric = budget / (len(metrics) + (2 if recall else 0))
    for m in metrics:
        index = Index(xb, m, profile=Profile({m: 1}, "tuning"), gpu=gpu)
        stop = min(deadline, time.perf_counter() + per_metric)
        on_cpu, on_gpu = {}, {}
        for b in batches:
            q = xq[:b]
            on_cpu[b] = _rate(index, q, k, "cpu", stop)
            on_gpu[b] = _rate(index, q, k, "gpu", stop)
            rates += [(m, "cpu", b, on_cpu[b]), (m, "gpu", b, on_gpu[b])]
        cpu_below[m] = threshold_with_margin(on_cpu, on_gpu, DEFAULT.threshold(m))
    ivf = None
    if recall is not None:
        ivf, ivf_rates = _tune_ivf(xb, np.asarray(xq), given_xq, metrics[0], recall, k, gpu)
        rates += ivf_rates
    source = f"tuned on {meta['n']:,} x {meta['d']} vectors, k = {k}, {meta['date']}"
    return Profile(cpu_below, source, tuple(rates), meta, ivf)
