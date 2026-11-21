"""Draws the README's figures into assets/ (run: uv run --group dev python examples/make_figures.py)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import siann  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "assets"


def metrics(out: Path) -> None:
    """One query's 25 nearest of 600 points under L2, L1 and L-infinity, found by siann, with the
    boundary through the 25th: a circle, a diamond and a square."""
    rng = np.random.default_rng(1)
    pts = rng.standard_normal((600, 2), dtype=np.float32)
    q = np.array([[0.9, 0.5]], dtype=np.float32)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    for ax, metric, title in zip(axes, ("l2", "l1", "linf"), ("L2 (Euclidean)", "L1 (Manhattan)", "L-infinity (Chebyshev)")):
        res = siann.Index(pts, metric, device="cpu").search(q, 25)
        r = res.scores[0, -1]
        ax.scatter(*pts.T, s=10, c="0.75")
        ax.scatter(*pts[res.indices[0]].T, s=22, c="tab:blue")
        ax.scatter(*q.T, s=260, marker="*", c="tab:red", zorder=3)
        if metric == "l2":
            t = np.linspace(0, 2 * np.pi, 400)
            ax.plot(q[0, 0] + r * np.cos(t), q[0, 1] + r * np.sin(t), c="tab:blue", lw=1.5)
        elif metric == "l1":
            ax.plot(q[0, 0] + r * np.array([1, 0, -1, 0, 1]), q[0, 1] + r * np.array([0, 1, 0, -1, 0]), c="tab:blue", lw=1.5)
        else:
            ax.plot(q[0, 0] + r * np.array([1, -1, -1, 1, 1]), q[0, 1] + r * np.array([1, 1, -1, -1, 1]), c="tab:blue", lw=1.5)
        span = 2.2 * r
        ax.set_xlim(q[0, 0] - span, q[0, 0] + span)
        ax.set_ylim(q[0, 1] - span, q[0, 1] + span)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "metrics.png", dpi=100, facecolor="white")
    plt.close(fig)


def main(out: Path = OUT) -> None:
    out.mkdir(parents=True, exist_ok=True)
    metrics(out)
    print(out / "metrics.png")


if __name__ == "__main__":
    main()
