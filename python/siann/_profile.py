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
# Approximate search's routing (see Profile.ivf). Single queries on the CPU: on SIFT1M at recall 0.925 a
# single query takes siann's Rust IVF on one thread 0.14 ms (7,228 per second) against the GPU's 0.49 ms
# (2,055); the GPU leads in batches of 100 (BENCHMARKS.md). List-major scans from 1,000 queries, where the GPU IVF
# switches in every measurement. Where between 1 and 100 the devices cross is measured by siann.tune.
IVF_DEFAULTS = {"cpu_below": 2, "list_major_from": 1000}


@dataclass(frozen=True)
class Profile:
    """Where siann sends each search, and the measurements behind it.

    ``cpu_below[metric]`` is the batch size below which exact search by that metric runs on the CPU
    when a GPU is available: 1 means always the GPU, ``None`` always the CPU. ``rates`` holds the
    measured queries per second, as ``(metric, device, batch, rate)``. ``ivf`` holds approximate
    search's thresholds: ``cpu_below``, the batch size below which an IVF search runs on the CPU (a single
    query split over 4 threads), and ``list_major_from``, the batch size from which a scan reads each list
    once for all the queries that probe it (on either device).
    """
    cpu_below: dict[str, int | None]
    source: str
    rates: tuple[tuple[str, str, int, float], ...] = ()
    meta: dict = field(default_factory=dict)
    ivf: dict = field(default_factory=lambda: dict(IVF_DEFAULTS))

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
        c, lm = self.ivf.get("cpu_below", 2), self.ivf.get("list_major_from", 1000)
        cpu = "single queries" if c <= 2 else f"batches below {c:,}"
        return (f"{cpu} on the CPU (one query split over 4 threads), the GPU above; "
                f"list-major scans from {lm:,} queries")

    def __str__(self) -> str:
        lines = [f"siann.Profile ({self.source})"]
        lines += [f"  {m:7s} exact search {self.describe(m)}" for m in self.cpu_below]
        lines.append(f"  approximate search (IVF): {self.describe_ivf()}")
        if self.rates:
            batches = sorted({b for _, _, b, _ in self.rates})
            rate = {(m, dev, b): r for m, dev, b, r in self.rates}
            lines.append("  queries per second, CPU / GPU, by batch size:")
            lines.append("    " + " " * 7 + "".join(f"{b:>16,}" for b in batches))
            for m in self.cpu_below:
                cells = [f"{rate.get((m, 'cpu', b), float('nan')):,.0f} / {rate.get((m, 'gpu', b), float('nan')):,.0f}"
                         for b in batches]
                lines.append(f"    {m:7s}" + "".join(f"{c:>16s}" for c in cells))
        return "\n".join(lines)

    def save(self, path) -> None:
        data = {"source": self.source, "cpu_below": self.cpu_below, "meta": self.meta,
                "rates": [list(r) for r in self.rates], "ivf": self.ivf}
        Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path) -> Profile:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(dict(data["cpu_below"]), data["source"],
                   tuple((m, d, int(b), float(r)) for m, d, b, r in data.get("rates", [])), data.get("meta", {}),
                   dict(data.get("ivf", IVF_DEFAULTS)))


# Tuned by benchmarks/exact_routing.py on an Apple M3 Max (results/exact_routing.txt): for each metric,
# the smallest crossover across SIFT1M, GloVe-100 and DBpedia OpenAI (100,000 vectors), so that no
# measured case is slower than with the GPU for every batch. Only single L2 queries were faster on the
# CPU (155 against 142 per second on SIFT1M). Crossovers move with the collection's size: on smaller
# ones the CPU wins more small batches, which siann.tune measures (TUNING.md).
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


def crossover(cpu: dict[int, float], gpu: dict[int, float]) -> int | None:
    """The smallest measured batch from which the GPU is at least as fast as the CPU at every larger
    measured batch; ``None`` if the GPU never is."""
    batches = sorted(cpu)
    for i, b in enumerate(batches):
        if all(gpu[c] >= cpu[c] for c in batches[i:]):
            return b
    return None


def tune(xb, xq=None, metrics=None, batches=BATCHES, k: int = 10, budget: float = 60.0,
              seed: int = 0) -> Profile:
    """Measures where exact search is faster, on the CPU or the GPU, for this data on this machine.

    ``xb`` is the user's base vectors (all of them: the crossover depends on the collection's size);
    ``xq`` the queries to time with, by default a sample of ``xb``. ``metrics`` defaults to all five.
    Each metric and batch size is timed on both devices (median of three after a warm-up that also
    compiles the GPU kernels for that shape), within about ``budget`` seconds in all. Returns a Profile;
    pass it to ``siann.Index(..., profile=p)`` or ``siann.use_profile(p)``.
    """
    from ._index import METRICS, Index, _gpu_available
    metrics = tuple(metrics) if metrics is not None else METRICS
    for m in metrics:
        if m not in METRICS:
            raise ValueError(f"unknown metric {m!r}")
    batches = sorted(set(int(b) for b in batches))
    xb = np.asarray(xb)
    if xq is None:
        rng = np.random.default_rng(seed)
        xq = xb[rng.choice(len(xb), size=min(len(xb), max(batches)), replace=len(xb) < max(batches))]
    xq = np.asarray(xq)
    if len(xq) < max(batches):
        batches = [b for b in batches if b <= len(xq)]
    meta = {"n": int(len(xb)), "d": int(xb.shape[1]), "k": int(k), "machine": platform.platform(),
            "date": time.strftime("%Y-%m-%d")}
    if not _gpu_available():
        return Profile({m: None for m in metrics}, "tuned: no GPU found, every search on the CPU", (), meta)
    deadline = time.perf_counter() + budget
    rates, cpu_below = [], {}
    per_metric = budget / len(metrics)
    for m in metrics:
        index = Index(xb, m, profile=Profile({m: 1}, "tuning"))
        stop = min(deadline, time.perf_counter() + per_metric)
        cpu, gpu = {}, {}
        for b in batches:
            q = xq[:b]
            cpu[b] = _rate(index, q, k, "cpu", stop)
            gpu[b] = _rate(index, q, k, "gpu", stop)
            rates += [(m, "cpu", b, cpu[b]), (m, "gpu", b, gpu[b])]
        cpu_below[m] = crossover(cpu, gpu)
    source = f"tuned on {meta['n']:,} x {meta['d']} vectors, k = {k}, {meta['date']}"
    return Profile(cpu_below, source, tuple(rates), meta)
