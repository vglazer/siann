"""Builds and executes examples/quickstart.ipynb (run: uv run --group dev python examples/make_quickstart.py)."""

from pathlib import Path

import nbformat
import nbformat.sign
from nbclient import NotebookClient


def md(text: str):
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbformat.v4.new_code_cell(text.strip())


cells = [
    md("""
# siann quickstart

siann is fast similarity search on Apple silicon for dense vectors. Give it base vectors and a metric, and
it finds each query's nearest neighbors, on the GPU (through MLX) or the CPU (BLAS through NumPy, and
siann's own Rust kernels), choosing the device itself unless you tell it which. This notebook covers exact
search: every metric, the scores it returns, how it explains its choices, and how the devices compare.
"""),
    code("""
import time

import numpy as np
import siann

siann.__version__
"""),
    md("""
## Some vectors

Real embeddings cluster, so instead of uniform noise the base vectors here are 200,000 points in 128
dimensions drawn around 1,000 random centers, and the queries are 1,000 more points drawn the same way.
"""),
    code("""
rng = np.random.default_rng(0)
centers = rng.standard_normal((1000, 128), dtype=np.float32)


def sample(n):
    return (centers[rng.integers(0, len(centers), n)]
            + 0.3 * rng.standard_normal((n, 128), dtype=np.float32))


xb, xq = sample(200_000), sample(1000)
xb.shape, xq.shape
"""),
    md("""
## Exact search

`siann.Index(xb, metric)` builds an index, and `index.search(xq, k)` returns each query's `k` nearest base
vectors, best first: `indices` (int64) and `scores`. Without a recall target the search is exact, so these
are the true nearest neighbors.
"""),
    code("""
index = siann.Index(xb, metric="cosine")
res = index.search(xq, k=5)
res.indices[:3], res.scores[:3]
"""),
    md("""
`index.plan` says what was chosen and why: here, which device runs the search and how each device would
compute it.
"""),
    code("""
print(index.plan)
"""),
    md("""
## The metrics and their scores

siann searches by L2 (Euclidean), L1, L-infinity (the largest coordinate difference), cosine and inner
product. Scores are each metric's natural value: the distance for L2 (not squared), L1 and L-infinity,
smallest first, and the similarity for cosine and inner product, largest first.
"""),
    code("""
for metric in siann.METRICS:
    r = siann.Index(xb, metric).search(xq[:1], k=3)
    print(f"{metric:7s} neighbors {r.indices[0]}  scores {np.round(r.scores[0], 3)}")
"""),
    md("""
The scores are exact: here they are recomputed with NumPy for the L2 search.
"""),
    code("""
r = siann.Index(xb, "l2").search(xq[:4], k=3)
np.allclose(r.scores, np.linalg.norm(xb[r.indices] - xq[:4, None, :], axis=2), rtol=1e-4)
"""),
    md("""
## Choosing the device

`device="cpu"` or `device="gpu"` pins an index to one device, and the same argument to `search` overrides it
for one call. Each device makes its own copy of the vectors the first time a search needs it. Both return the
same neighbors; how fast each is depends on the metric. L2, cosine and inner product are matrix multiplies
on either device, where L1 and L-infinity have no matrix-multiply form, so the GPU's lead is largest there.
The times below are for this machine. Each is the second of two identical calls: the first builds the
device's copy of the vectors and, on the GPU, compiles the kernels for this batch size and k.
"""),
    code("""
def timed(index, device):
    index.search(xq, 10, device=device)                   # builds the copy, compiles for this shape
    start = time.perf_counter()
    index.search(xq, 10, device=device)
    return time.perf_counter() - start


devices = ["cpu", "gpu"] if index.plan.device == "gpu" else ["cpu"]
for metric in siann.METRICS:
    idx = siann.Index(xb, metric)
    times = {d: timed(idx, d) for d in devices}
    print(f"{metric:7s} " + "  ".join(f"{d} {t * 1e3:7.1f} ms" for d, t in times.items()))
"""),
    md("""
## When fewer than k exist

Asking for more neighbors than there are vectors returns all of them, then `-1` in the missing slots, with
the metric's worst score (infinity for a distance, minus infinity for a similarity). A neighbor is never
made up.
"""),
    code("""
small = siann.Index(xb[:3], "l2")
small.search(xq[:1], k=5)
"""),
    md("""
## What siann refuses

siann searches dense vectors, so a sparse matrix is refused with an explanation rather than converted
silently; so is a recall target, which asks for approximate search.
"""),
    code("""
import scipy.sparse

for attempt in (lambda: siann.Index(scipy.sparse.random(100, 20, density=0.1, format="csr"), "ip"),
                lambda: siann.Index(xb, "l2", recall=0.95)):
    try:
        attempt()
    except NotImplementedError as e:
        print("NotImplementedError:", e)
"""),
]


def main() -> None:
    nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {
        "name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"}})
    NotebookClient(nb, timeout=600, kernel_name="python3").execute()
    nbformat.sign.NotebookNotary().sign(nb)
    out = Path(__file__).with_name("quickstart.ipynb")
    nbformat.write(nb, out)
    print(out, f"{out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
