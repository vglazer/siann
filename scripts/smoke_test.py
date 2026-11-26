"""Checks an installed siann (a built wheel) on the CPU: exact search by every metric against NumPy, an
IVF reaching its recall target, a graph built by siann's own builder reaching it too, and a save and load that
return the same neighbors. The release workflow
runs it on each wheel, on the wheel's own platform; run it from anywhere but the repository's python/
folder, so that the installed package is the one imported.

    python scripts/smoke_test.py
"""
import tempfile
from pathlib import Path

import numpy as np

import siann


def brute_force(x: np.ndarray, q: np.ndarray, metric: str, k: int) -> np.ndarray:
    if metric == "cosine":
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
    if metric in ("cosine", "ip"):
        d = -(q @ x.T)
    else:
        diff = np.abs(q[:, None, :] - x[None, :, :])
        d = {"l1": diff.sum(-1), "l2": (diff ** 2).sum(-1), "linf": diff.max(-1)}[metric]
    return np.argsort(d, axis=1, kind="stable")[:, :k]


def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3_000, 16), dtype=np.float32)
    q = rng.standard_normal((20, 16), dtype=np.float32)
    for metric in siann.METRICS:
        found = siann.Index(x, metric).search(q, 5, device="cpu").indices
        truth = brute_force(x, q, metric, 5)
        assert (found == truth).mean() > 0.99, (metric, (found == truth).mean())

    centers = rng.standard_normal((50, 24), dtype=np.float32)
    xc = centers[rng.integers(0, 50, 30_000)] + 0.3 * rng.standard_normal((30_000, 24), dtype=np.float32)
    qc = xc[:200] + 0.05
    index = siann.Index(xc, "l2", recall=0.9, kind="ivf", device="cpu")
    truth = siann.Index(xc, "l2").search(qc, 10, device="cpu").indices
    found = index.search(qc, 10).indices
    recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(found, truth)])
    assert recall >= 0.85, recall

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "index.siann"
        index.save(path)
        loaded = siann.Index.load(path, device="cpu")
        assert (loaded.search(qc, 10).indices == found).all()

    xg = rng.standard_normal((20_000, 16), dtype=np.float32)
    qg = rng.standard_normal((200, 16), dtype=np.float32)
    graph = siann.Index(xg, "l1", recall=0.9, kind="graph", device="cpu", calibration_queries=qg[:100])
    truth = siann.Index(xg, "l1").search(qg[100:], 10, device="cpu").indices
    got = graph.search(qg[100:], 10).indices
    graph_recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(got, truth)])
    assert graph._ivf.graph is not None and graph_recall >= 0.85, graph_recall
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "graph.siann"
        graph.save(path)
        assert (siann.Index.load(path, device="cpu").search(qg[100:], 10).indices == got).all()

    print(f"siann {siann.__version__}: exact search by {', '.join(siann.METRICS)}, IVF recall {recall:.3f}, "
          f"graph recall {graph_recall:.3f} (built by {graph._ivf.graph.builder}), save and load: ok "
          f"(GPU {'found' if siann.gpu_available() else 'not found'})")


if __name__ == "__main__":
    main()
