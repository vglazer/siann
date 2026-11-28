"""How close siann's own choices come to the best one, lever by lever. A nearest-neighbor search is made fast
by three levers: (1) trading recall for speed, (2) an index (an inverted index or a graph) instead of exact
search, (3) for large enough batches, the GPU instead of the CPU. With a recall target, siann.Index pulls all
three itself: it calibrates how hard to search to just meet the target (1), picks exact search, the IVF or the
graph (2), and routes each batch size to a device and a scan (3). This script checks each choice against every
alternative, for each dataset, recall target and batch size.

The alternatives, each forced through the same Index.search, so that every path pays the same overhead (the
scores, the ranking, the ids): exact search on the CPU and on the GPU; every IVF path ("cpu-split", one query
over 4 threads; "cpu-query"; "cpu-query-blas"; "cpu-list"; "gpu-query"; "gpu-list") at the IVF's calibrated
nprobe; the graph on the CPU and on the GPU (where mlxvs is installed and the metric is L2, cosine or inner
product) at its calibrated search widths. They come from one index built with kind="graph" (an IVF with a graph
beside it, each calibrated to the target on its own): siann's own kind="auto" index, as a user would build it,
which keeps its IVF and graph beside exact search; only where siann kept exact search alone are the alternatives
built apart (kind="graph", or kind="ivf" where no graph can be built). The same index for both keeps two builds'
different graphs (each build's graph differs, and with it its calibrated width) out of the comparison. Both calibrate on the first CAL
queries; the others are the test queries.

Recall: each path's recall@10 on all the test queries (about 9,000: a standard error near 0.002) against exact
search. A path counts if it is within TOLERANCE of the target. The target holds on average: calibrated on 1,000
queries, recall on new ones lands a few thousandths either side of it (graph_calibration.py). Timing: siann's choice and every path that counts, at each batch size, in ROUNDS
rounds, in a rotated order each round (the GPU runs slower right after the CPU), each call back to back for at
least 0.1 s and 3 calls after a warm-up; the median over rounds. A path past SLOW s per call is not timed at
larger batches.

Each row is classed by the lever that went wrong, if any:
- recall (lever 1): the path siann chose misses the target (by more than TOLERANCE) on the test queries;
- index (lever 2): the best path that counts is a different kind (exact, IVF, graph) and more than MARGIN faster;
- device or scan (lever 3): the best is the same kind, on another device or scan, and more than MARGIN faster.
The regret is the best path's speed over the choice's (1.00: siann chose the best).

    uv run --group bench python choice_bench.py --datasets sift --recalls 0.95
    uv run --group bench python choice_bench.py --datasets sift,glove,music100,dbpedia

Each dataset's finished targets are kept in results/choice_bench_<name>.json as they finish, and the text file
rewritten from them: a run that is stopped picks up where it was, at the next target, when started again.
"""
import argparse
import json
import time
from pathlib import Path

import siann

import numpy as np
from datasets import load

METRIC = {"sift": "l2", "glove": "cosine", "music100": "ip", "dbpedia": "cosine"}
K = 10
CAL = 1000
TOLERANCE = 0.01
RECALL_ON = "all test queries, one index"   # in the checkpoint: results measured otherwise are not reused
MARGIN = 1.10
ROUNDS = 3
SLOW = 5.0
BATCHES = (1, 10, 100, 1000, 5000)
RECALLS = (0.90, 0.95, 0.99)
RESULTS = Path(__file__).resolve().parent / "results"
IVF_PATHS = ("cpu-split", "cpu-query", "cpu-query-blas", "cpu-list", "gpu-query", "gpu-list")
GRAPH_PATHS = ("cpu-graph", "gpu-graph")


def kind_of(path: str) -> str:
    return "exact" if "exact" in path else ("graph" if path.endswith("graph") else "IVF")   # "gpu-exact" too


def calls_of(call, q) -> float:
    """Seconds per call, back to back for at least 0.1 s and 3 calls; the median."""
    times, start = [], time.perf_counter()
    while len(times) < 3 or time.perf_counter() - start < 0.1:
        t = time.perf_counter()
        call(q)
        times.append(time.perf_counter() - t)
        if times[-1] > SLOW:
            break
    return float(np.median(times))


def timed_rounds(calls: dict, q) -> dict:
    """Each call's seconds, the median over ROUNDS rounds, in a rotated order each round, after a warm-up."""
    names = list(calls)
    for name in names:
        calls[name](q)
    per = {name: [] for name in names}
    for r in range(ROUNDS):
        for name in names[r % len(names):] + names[:r % len(names)]:
            per[name].append(calls_of(calls[name], q))
    return {name: float(np.median(v)) for name, v in per.items()}


def forced(index: siann.Index, path: str):
    """A view of ``index`` that sends every search down ``path``."""
    view = index._snapshot()
    view._ivf_path = lambda batch, device, approximate=False, path=path: path
    return view


def recall(found: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean([len(set(f) & set(t)) / K for f, t in zip(found, truth, strict=True)]))


class Out:
    """Lines printed as they come and kept, for the checkpoint."""

    def __init__(self):
        self.lines = []

    def __call__(self, line: str = "") -> None:
        print(line, flush=True)
        self.lines.append(line)


def run_target(xb, metric, cal, test, truth, exact, target, batches, say) -> list:
    t = time.perf_counter()
    auto = siann.Index(xb, metric, recall=target, calibration_queries=cal)
    say(f"\n  target {target}: siann's choice built in {time.perf_counter() - t:.1f} s, kind {auto.kind!r}"
          + (f", nprobe {auto.nprobe}" if auto.nprobe else ""))
    t = time.perf_counter()
    if auto._ivf is not None:                            # its own IVF and graph: the same lists, graph and widths
        every = auto
        has_graph = every._ivf.graph is not None
        say(f"    every path from siann's own index (an IVF{' and a graph' if has_graph else ''})")
    else:                                                # exact search kept: the alternatives built apart
        try:
            every = siann.Index(xb, metric, recall=target, calibration_queries=cal, kind="graph")
            has_graph = every._ivf is not None and every._ivf.graph is not None
        except (RuntimeError, ValueError) as e:          # no graph for this metric or target, or no mlxvs
            say(f"    no graph index ({str(e)[:120]})")
            every, has_graph = siann.Index(xb, metric, recall=target, calibration_queries=cal, kind="ivf"), False
        say(f"    every path's index (kind={'graph' if has_graph else 'ivf'}): built in {time.perf_counter() - t:.1f} s")

    paths = {"exact, cpu": lambda q: exact.search(q, K, device="cpu"),
             "exact, gpu": lambda q: exact.search(q, K, device="gpu")}
    for p in IVF_PATHS + (GRAPH_PATHS if has_graph else ()):
        if p == "gpu-graph" and not every._ivf.graph.on_gpu:
            continue
        view = forced(every, p)
        paths[p] = lambda q, view=view: view.search(q, K)
    rec = {p: recall(call(test).indices, truth) for p, call in paths.items()}
    counts = {p for p, r in rec.items() if r >= target - TOLERANCE}
    say(f"    recall@10, {len(test):,} test queries: " + ", ".join(f"{p} {r:.3f}" for p, r in rec.items()))

    def chosen(b):
        return auto._ivf_path(b, None) if auto._ivf is not None else f"exact, {auto.route(b)}"

    choice_recall = {}
    for b in batches:
        c = chosen(b)
        if c not in choice_recall:
            call = (lambda q: auto.search(q, K, device=c.split(", ")[1])) if c.startswith("exact") else \
                forced(auto, c).search
            choice_recall[c] = recall(call(test).indices if c.startswith("exact") else
                                      call(test, K).indices, truth)

    say(f"    {'batch':>6} {'siann chose':>16} {'recall':>7} {'q/s':>10}   {'best path':>16} {'q/s':>10}  "
        f"{'regret':>6}  lever")
    rows, slow = [], set()
    for b in batches:
        q = np.ascontiguousarray(test[:b])
        calls = {p: paths[p] for p in paths if p in counts and p not in slow}
        calls["siann's choice"] = lambda qq: auto.search(qq, K)
        sec = timed_rounds(calls, q)
        slow |= {p for p, s in sec.items() if s > SLOW and p != "siann's choice"}
        speeds = {p: b / s for p, s in sec.items() if p != "siann's choice"}
        choice, c = b / sec["siann's choice"], chosen(b)
        best = max(speeds, key=speeds.get) if speeds else None
        regret = speeds[best] / choice if best else float("nan")
        if choice_recall[c] < target - TOLERANCE:
            lever = "recall (1)"
        elif best and regret > MARGIN:
            lever = "index (2)" if kind_of(best) != kind_of(c) else "device or scan (3)"
        else:
            lever = "-"
        rows.append((target, b, c, lever, regret))
        others = ", ".join(f"{p} {v:,.0f}" for p, v in sorted(speeds.items(), key=lambda kv: -kv[1])[:4])
        say(f"    {b:>6,} {c:>16} {choice_recall[c]:7.3f} {choice:>10,.0f}   {best or '-':>16} "
            f"{speeds.get(best, float('nan')):>10,.0f}  {regret:6.2f}  {lever}   (fastest: {others})")
    return rows


def run(name: str, n: int | None, targets, batches, fresh: bool = False) -> None:
    """One dataset, target by target. Each finished target is saved to results/choice_bench_<name>.json at once
    (its printed lines and rows), and the text file rewritten from it; a rerun reprints the targets already
    saved and runs only the missing ones (--fresh: all)."""
    ckpt = RESULTS / f"choice_bench_{name}.json"
    done = {} if fresh or not ckpt.exists() else json.loads(ckpt.read_text(encoding="utf-8"))
    if done.get("setup") != [n, list(batches), RECALL_ON]:
        done = {}                                        # another size or other batches: start over
    done["setup"] = [n, list(batches), RECALL_ON]
    left = [t for t in targets if str(t) not in done]
    if not left:
        print(f"{name}: every target already measured ({ckpt.name}); --fresh to measure again", flush=True)
    if not left:
        write_text(name, done, targets)
        return
    xb, xq = load(name, 10_000)
    if n:
        xb = np.ascontiguousarray(xb[:n])
    metric = METRIC[name]
    cal, test = xq[:CAL], xq[CAL:]
    batches = [b for b in batches if b <= len(test)]
    done["header"] = (f"{name}: {len(xb):,} x {xb.shape[1]}, {metric}, recall@{K} targets "
                      f"{', '.join(map(str, targets))}; {len(test):,} test queries (calibration on {CAL:,} others); "
                      f"{ROUNDS} timing rounds")
    print(done["header"], flush=True)
    resumed = [t for t in targets if str(t) in done]
    if resumed:
        print(f"  resumed from {ckpt.name}: targets {', '.join(map(str, resumed))} already measured; "
              f"left: {', '.join(map(str, left))}", flush=True)
    exact = siann.Index(xb, metric)
    truth = exact.search(test, K, device="gpu").indices
    for target in left:
        say = Out()
        rows = run_target(xb, metric, cal, test, truth, exact, target, batches, say)
        done[str(target)] = {"lines": say.lines, "rows": rows}
        ckpt.write_text(json.dumps(done, indent=1) + "\n", encoding="utf-8")
        write_text(name, done, targets)
    print("\n" + summary(name, done, targets), flush=True)


def summary(name: str, done: dict, targets) -> str:
    rows = [tuple(r) for t in targets if str(t) in done for r in done[str(t)]["rows"]]
    misses = [r for r in rows if r[3] != "-"]
    missing = [t for t in targets if str(t) not in done]
    return (f"  {name}: {len(rows) - len(misses)} of {len(rows)} choices within {MARGIN:.2f}x of the best and on "
            f"target; " + ("misses: " + "; ".join(f"target {t}, batch {b:,}: {lv}, {c}, regret {g:.2f}"
                                               for t, b, c, lv, g in misses) if misses else "no misses")
            + (f" (targets not yet measured: {', '.join(map(str, missing))})" if missing else ""))


def write_text(name: str, done: dict, targets) -> None:
    """results/choice_bench_<name>.txt from the checkpoint: the targets measured so far, and the summary."""
    lines = [done.get("header", name)]
    for t in targets:
        if str(t) in done:
            lines += done[str(t)]["lines"]
    lines += ["", summary(name, done, targets)]
    (RESULTS / f"choice_bench_{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="sift")
    p.add_argument("--recalls", default=",".join(map(str, RECALLS)))
    p.add_argument("--batches", default=",".join(map(str, BATCHES)))
    p.add_argument("--n", type=int, default=None, help="the first n base vectors (all by default)")
    p.add_argument("--n-dbpedia", type=int, default=100_000, help="DBpedia's base vectors (1,536-d)")
    p.add_argument("--fresh", action="store_true", help="ignore the checkpoints: measure every target again")
    a = p.parse_args()
    targets = tuple(map(float, a.recalls.split(",")))
    batches = tuple(map(int, a.batches.split(",")))
    for name in a.datasets.split(","):
        run(name, a.n_dbpedia if name == "dbpedia" else a.n, targets, batches, a.fresh)
        print(flush=True)


if __name__ == "__main__":
    main()
