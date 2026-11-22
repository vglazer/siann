"""Builds and executes examples/quickstart.ipynb (run: uv run --group dev python examples/make_quickstart.py)."""

from pathlib import Path

import nbformat
import nbformat.sign
from nbclient import NotebookClient

W = "https://en.wikipedia.org/wiki/"


def md(text: str):
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbformat.v4.new_code_cell(text.strip())


cells = [
    md("""
# siann quickstart

siann is fast similarity search on Apple silicon for dense vectors. Give it base vectors and a metric, and
it finds each query's nearest neighbors, on the GPU or the CPU (BLAS through NumPy, and siann's own Rust
kernels), choosing the device for each search unless you tell it which.

This notebook covers what each metric means and how to choose one, the scores siann returns and how it
explains its choices, how the GPU and the CPU compare as batches grow, tuning siann's routing to your own
data and hardware, exact search as the ground truth, and approximate search to a recall target.
"""),
    code("""
import time

import matplotlib.pyplot as plt
import numpy as np

import siann

siann.__version__
"""),
    md(f"""
## The metrics, in two dimensions

siann searches by five measures of nearness. In two dimensions they are easy to see: below, 600 points,
one query (the orange dot), and each metric's 25 nearest points to it (blue), with the boundary that encloses them.
The L2, L1 and L-infinity panels are zoomed in on the query; the cosine and inner product panels show the
whole cloud, since their regions run out from the origin.

- [**L2**]({W}Euclidean_distance) (Euclidean) is straight-line distance; its ball is a circle. The usual
  choice for vectors whose coordinates are comparable measurements.
- [**L1**]({W}Taxicab_geometry) (Manhattan) adds up the coordinate differences; its ball is a diamond. One
  large difference counts less than under L2, which squares it, so L1 is more robust to a few outlying
  coordinates.
- [**L-infinity**]({W}Chebyshev_distance) (Chebyshev) is the largest single coordinate difference; its ball
  is a square. A neighbor within distance r differs by at most r in every coordinate: a worst-case bound.
- [**Cosine**]({W}Cosine_similarity) compares directions only, so its neighbors fill a wedge from the origin,
  whatever their length. The usual choice for text and image embeddings.
- [**Inner product**]({W}Dot_product) favors vectors that point the query's way *and* are long: its
  neighbors lie beyond a line perpendicular to the query. For embeddings trained to be compared this way.
"""),
    code("""
rng = np.random.default_rng(1)
pts = rng.standard_normal((600, 2), dtype=np.float32)
q = np.array([[0.9, 0.5]], dtype=np.float32)
k = 25

fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
axes = axes.ravel()
axes[-1].axis("off")
for ax, metric in zip(axes, siann.METRICS):
    res = siann.Index(pts, metric, device="cpu").search(q, k)
    near, kth = res.indices[0], res.scores[0, -1]
    ax.scatter(*pts.T, s=6, c="0.75")
    ax.scatter(*pts[near].T, s=14, c="tab:blue")
    ax.scatter(*q.T, s=110, c="tab:orange", edgecolors="black", linewidths=1, zorder=3)
    t = np.linspace(0, 2 * np.pi, 400)
    if metric == "l2":
        ax.plot(q[0, 0] + kth * np.cos(t), q[0, 1] + kth * np.sin(t), c="tab:blue", lw=1)
    elif metric == "l1":
        ax.plot(q[0, 0] + kth * np.array([1, 0, -1, 0, 1]), q[0, 1] + kth * np.array([0, 1, 0, -1, 0]), c="tab:blue", lw=1)
    elif metric == "linf":
        ax.plot(q[0, 0] + kth * np.array([1, -1, -1, 1, 1]), q[0, 1] + kth * np.array([1, 1, -1, -1, 1]), c="tab:blue", lw=1)
    elif metric == "cosine":
        a, w = np.arctan2(q[0, 1], q[0, 0]), np.arccos(np.clip(kth, -1, 1))
        for b in (a - w, a + w):
            ax.plot([0, 4 * np.cos(b)], [0, 4 * np.sin(b)], c="tab:blue", lw=1)
    else:   # the line q.x = kth, perpendicular to q
        u = q[0] / np.linalg.norm(q[0])
        base, along = u * kth / np.linalg.norm(q[0]), np.array([-u[1], u[0]])
        ax.plot(*np.array([base - 5 * along, base + 5 * along]).T, c="tab:blue", lw=1)
    ax.set_title({"l2": "L2", "l1": "L1", "linf": "L-infinity", "cosine": "cosine", "ip": "inner product"}[metric])
    # the balls are small: zoom on the query; cosine and inner product regions run out from the origin
    span = 2.2 * kth if metric in ("l2", "l1", "linf") else 3.2
    cx, cy = (q[0] if metric in ("l2", "l1", "linf") else (0, 0))
    ax.set_xlim(cx - span, cx + span); ax.set_ylim(cy - span, cy + span)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
"""),
    md("""
## Some vectors

Real embeddings cluster, so instead of uniform noise the rest of the notebook uses 200,000 points in 128
dimensions drawn around 1,000 random centers, and up to 10,000 queries drawn the same way.
"""),
    code("""
centers = rng.standard_normal((1000, 128), dtype=np.float32)


def sample(n):
    return (centers[rng.integers(0, len(centers), n)]
            + 0.3 * rng.standard_normal((n, 128), dtype=np.float32))


xb, xq = sample(200_000), sample(10_000)
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
res = index.search(xq[:1000], k=5)
res.indices[:3], res.scores[:3]
"""),
    md("""
`index.plan` says what was chosen and why: here, how a search is routed to the GPU or the CPU by its batch
size, under which profile, and how each device computes it.
"""),
    code("""
print(index.plan)
"""),
    md("""
## Scores

Scores are each metric's natural value: the distance for L2 (not squared), L1 and L-infinity, smallest
first, and the similarity for cosine and inner product, largest first. They are exact: here they are
recomputed with NumPy for an L2 search. The metrics also disagree about who is nearest, even on the same
data: the share of each L2 top 10 that another metric's top 10 shares is below.
"""),
    code("""
l2 = siann.Index(xb, "l2").search(xq[:1000], k=10)
print("L2 scores exact:", np.allclose(l2.scores, np.linalg.norm(xb[l2.indices] - xq[:1000, None, :], axis=2), rtol=1e-4))
for metric in ("l1", "linf", "cosine", "ip"):
    other = siann.Index(xb, metric).search(xq[:1000], k=10).indices
    shared = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(l2.indices, other)])
    print(f"{metric:7s} shares {shared:.0%} of L2's top 10")
"""),
    md("""
## The GPU and the CPU, by batch size

`device="cpu"` or `device="gpu"` pins an index to one device, and the same argument to `search` overrides it
for one call; each device makes its own copy of the vectors the first time a search needs it. Both return
the same neighbors. Which is faster depends on the batch and the metric: a GPU call has a fixed cost that a
large batch spreads thin, and L2, cosine and inner product are matrix multiplies on either device, where
L1 and L-infinity have no matrix-multiply form, so the GPU's lead is largest there.

Below, queries per second on this machine at batch sizes from 1 to 10,000, k = 10. Each point is the
second of two identical calls: the first builds the device's copy and, on the GPU, compiles the kernels for
that batch size.
"""),
    code("""
devices = ["cpu", "gpu"] if siann.gpu_available() else ["cpu"]
batches = [1, 10, 100, 1000, 10_000]
rate = {}
for metric in siann.METRICS:
    idx = siann.Index(xb, metric)
    for device in devices:
        for b in batches:
            idx.search(xq[:b], 10, device=device)
            start = time.perf_counter()
            idx.search(xq[:b], 10, device=device)
            rate[metric, device, b] = b / (time.perf_counter() - start)

fig, axes = plt.subplots(1, 5, figsize=(17, 3.4), sharey=True)
for ax, metric in zip(axes, siann.METRICS):
    for device in devices:
        ax.plot(batches, [rate[metric, device, b] for b in batches], marker="o", label=device.upper())
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("queries per batch")
    ax.set_title({"l2": "L2", "l1": "L1", "linf": "L-infinity", "cosine": "cosine", "ip": "inner product"}[metric])
axes[0].set_ylabel("queries per second"); axes[0].legend()
fig.tight_layout()
"""),
    code("""
for metric in siann.METRICS:
    print(f"{metric:7s} " + "  ".join(
        f"batch {b:>6,}: {rate[metric, 'gpu', b] / rate[metric, 'cpu', b]:5.1f}x" for b in batches)
          if "gpu" in devices else metric)
"""),
    md("""
The table gives the GPU's speed over the CPU's at each batch size; below 1, the CPU is faster. Where the
two cross depends on the metric, the hardware and the size of the collection: the CPU's cost grows with the
number of vectors, the GPU's cost per call does not.
"""),
    md("""
## Tuning siann to your data and machine

`siann.Index` sends each search to the GPU or the CPU by its batch size, from a *profile*'s threshold for
the metric. siann's default profile was tuned on an Apple M3 Max with collections of about a million
vectors. For other hardware or other sizes, `siann.tune` measures both devices on your own vectors and
returns a profile for them; here, for two metrics on this notebook's 200,000 vectors (about half a minute):
"""),
    code("""
print(siann.DEFAULT_PROFILE)
profile = siann.tune(xb, metrics=("l2", "cosine"), budget=30)
print(profile)
"""),
    md("""
A profile is used only where you pass it, so results never depend on a file you forgot about: give it to
one index with `profile=`, or to every index in the process with `siann.use_profile(profile)`. It saves to
JSON with `profile.save(path)`, and `siann.Profile.load(path)` reads it back. The plan names the profile
and its threshold, and `index.route(batch)` says where a batch of that size goes:
"""),
    code("""
tuned = siann.Index(xb, "l2", profile=profile)
print(tuned.plan)
{b: tuned.route(b) for b in (1, 8, 64, 1000)}
"""),
    md("""
## Exact search as ground truth

An approximate index is judged by its recall: the share of the true nearest neighbors it finds. That needs
the true neighbors first, for many queries, which exact search on the GPU provides quickly:
"""),
    code("""
truth_index = siann.Index(xb, "l2")
truth_index.search(xq, 10)                         # builds the copy, compiles for this batch
start = time.perf_counter()
truth = truth_index.search(xq, 10).indices
print(f"true 10 nearest neighbors of {len(xq):,} queries among {len(xb):,}: "
      f"{time.perf_counter() - start:.2f} s on the {truth_index.plan.device.upper()}")


def recall(found, truth):
    return np.mean([len(set(a) & set(b)) / truth.shape[1] for a, b in zip(found, truth)])


recall(truth, truth)
"""),
    md(f"""
## Approximate search

Exact search compares each query with every vector. Approximate search compares it with only some of them,
trading a little accuracy for a lot of speed.

**IVF.** siann does this with an [inverted file]({W}Inverted_index) (IVF). When the index is built, the
vectors are grouped into *lists* by [k-means clustering]({W}K-means_clustering): each list holds the vectors
nearest one cluster centre, its *centroid*. A search compares the query with the centroids, picks the lists
of the nearest few, and scans only their vectors. Most of a query's true neighbors are in those lists; a few
sit just across a list's boundary, in a list that was not scanned, and are missed.

**nprobe** is how many lists a search scans. Scanning more lists finds more of the true neighbors and takes
longer; scanning all of them is exact search.

**Recall** measures what is missed. Recall@10 is the share of each query's true 10 nearest neighbors that the
search returns, averaged over the queries: at 0.95, 95 of every 100 true neighbors are found. It is how
approximate search is judged: a faster search at lower recall is not a better one, so methods are compared
at the same recall, as in siann's benchmarks.

With a recall target, siann builds an IVF, finds the smallest nprobe that reaches the target on a sample of
queries (*calibration*), times that against exact search on the sample, and keeps whichever is faster; it
also times its CPU and GPU scans on the sample, to route each batch. All of it is automatic:
`siann.Index(xb, "l2", recall=0.95)` calibrates on 1,000 of the index's own vectors. Passing a sample of your
own queries as `calibration_queries` is optional, like tuning, and pays where the queries differ from the
base vectors (at the end of this section). Here, 1,000 of the queries are passed. The plan reports every
choice.
"""),
    code("""
approx = siann.Index(xb, "l2", recall=0.95, calibration_queries=xq[:1000])
print(approx.plan)
"""),
    md("""
The target holds on queries it was not calibrated on: here, the other 9,000, against their true neighbors
from above. This synthetic data clusters cleanly, so a couple of lists per query suffice; on SIFT1M the same
target takes about 40 of its 4,000 lists.
"""),
    code("""
rest = xq[1000:]
found = approx.search(rest, 10).indices
print(f"recall@10 on the other {len(rest):,} queries: {recall(found, truth[1000:]):.3f}")
"""),
    md("""
Approximate search pays most where exact search costs most: in large batches. Where each batch runs, on the
CPU or the GPU and with which scan, the index measured when it was built (the plan's "routing" line).
"""),
    code("""
def rate(fn, b):
    q = rest[:b]
    fn(q)                                              # warm-up: copies, kernels for this shape
    calls = max(1, 2000 // b) if b < 1000 else 3
    start = time.perf_counter()
    for c in range(calls):
        fn(rest[c:c + 1] if b == 1 else q)
    return calls * b / (time.perf_counter() - start)


sizes = [1, 10, 100, 1000, 9000]
exact_rate = [rate(lambda q: truth_index.search(q, 10), b) for b in sizes]
approx_rate = [rate(lambda q: approx.search(q, 10), b) for b in sizes]
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(sizes, exact_rate, marker="o", label="exact")
ax.plot(sizes, approx_rate, marker="o", label=f"IVF, recall 0.95 (nprobe {approx.nprobe})")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("queries per batch"); ax.set_ylabel("queries per second"); ax.legend()
fig.tight_layout()
"""),
    md("""
nprobe trades recall for speed. Below, recall against queries per second for a batch of 9,000 queries at
nprobe from 1 to 128 (`search(..., nprobe=)` overrides the calibrated value). Each step doubles the lists
scanned: recall climbs towards 1.0 and speed falls towards exact search's (in red, the dashed line its
speed). The calibrated nprobe is ringed: the fewest lists that reach the target.
"""),
    code("""
curve = []
for p in (1, 2, 4, 8, 16, 32, 64, 128):
    approx.search(rest, 10, nprobe=p)
    start = time.perf_counter()
    found = approx.search(rest, 10, nprobe=p).indices
    curve.append((recall(found, truth[1000:]), len(rest) / (time.perf_counter() - start), p))
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot([r for r, _, _ in curve], [q for _, q, _ in curve], marker="o", c="tab:blue", label="IVF, by nprobe")
for r, q, p in curve:
    ax.annotate(f"nprobe {p}", (r, q), textcoords="offset points", xytext=(-8, 2), ha="right", fontsize=8,
                color="tab:blue")
calibrated = [c for c in curve if c[2] == approx.nprobe]
if calibrated:
    ax.scatter([calibrated[0][0]], [calibrated[0][1]], s=260, facecolors="none", edgecolors="tab:orange",
               linewidths=2.5, zorder=3, label=f"calibrated to recall 0.95 (nprobe {approx.nprobe})")
ax.axhline(exact_rate[-1], color="tab:red", linestyle="--", linewidth=1)
ax.scatter([1.0], [exact_rate[-1]], marker="s", s=90, c="tab:red", zorder=3, label="exact search (recall 1.0)")
ax.set_yscale("log"); ax.set_xlabel("recall@10 (share of the true 10 nearest found)")
ax.set_ylabel("queries per second, batch of 9,000")
ax.legend(loc="center left"); fig.tight_layout()
"""),
    md("""
**Calibrating on your queries (optional).** Without `calibration_queries`, siann calibrates on a sample of
the index's own vectors, and the target holds for queries like them. Here the queries are drawn like the
base vectors, so the two agree:
"""),
    code("""
own = siann.Index(xb, "l2", recall=0.95)
print(f"calibrated on its own vectors: nprobe {own.nprobe}, recall {recall(own.search(rest, 10).indices, truth[1000:]):.3f}")
print(f"calibrated on 1,000 queries:   nprobe {approx.nprobe}, recall {recall(approx.search(rest, 10).indices, truth[1000:]):.3f}")
"""),
    md("""
On real data the queries often differ from the base vectors, in either direction, and calibrating on 1,000
of them pays on each dataset measured (a million vectors, target 0.95, recall on the other 9,000 queries;
BENCHMARKS.md, "Approximate search through siann.Index"):

| Dataset | Calibrated on its own vectors | Calibrated on 1,000 queries |
| --- | --- | --- |
| SIFT1M, L2 | nprobe 38, recall **0.940**: below the target | nprobe 46, recall 0.955 |
| Music-100, inner product | nprobe 129, recall 0.986; 60,076 / 170,253 queries/s in batches of 100 / 10,000 | nprobe 25, recall 0.946; 168,543 / 333,396 queries/s (**2.8x / 2.0x**) |

SIFT's queries are harder than its base vectors, so calibrating on them is what keeps the target, for 5-10%
of the speed. Music-100's are easier, so calibrating on them avoids scanning five times the lists needed,
and lands within 0.005 of the target. The same queries time the routing too: timed on its own vectors, Music-100's index sends
batches of 512 to a scan that runs at 0.77 times the speed of the other on real queries.
"""),
    md("""
## Saving and loading an index

`index.save(path)` writes everything the index built: the vectors, the IVF's lists, the calibration and
its sample, the routes and the profile. `siann.Index.load(path)` reads it back without rebuilding, and
returns the same neighbors. On another machine it keeps the calibration, which depends only on the data,
and re-measures the routes, which depend on the machine; the plan says which it did.
"""),
    code("""
import os
import tempfile

path = os.path.join(tempfile.mkdtemp(), "approx.siann")
start = time.perf_counter()
approx.save(path)
loaded = siann.Index.load(path)
print(f"saved and loaded in {time.perf_counter() - start:.2f} s ({os.path.getsize(path) / 1e6:.0f} MB)")
print(loaded.plan)
assert (loaded.search(rest, 10).indices == approx.search(rest, 10).indices).all()
"""),
    md("""
## Changing an index

`index.add(x)` adds vectors and returns their ids; `index.remove(ids)` removes vectors by id. Searches return
ids: a vector's row number unless `add(x, ids=...)` names it. Added vectors are searched exactly until
10,000 of them (or 1% of the index) have gathered; then they are merged into the index, and an approximate
index recalibrates nprobe, so that the recall target keeps holding. Here, an index built on the first
150,000 vectors gets the other 50,000 in batches of 10,000. Their ids continue from 150,000, which are
their row numbers in `xb`, so the true neighbors from above still apply.
"""),
    code("""
growing = siann.Index(xb[:150_000], "l2", recall=0.95, calibration_queries=xq[:1000])
for s in range(150_000, 200_000, 10_000):
    ids = growing.add(xb[s:s + 10_000])
print(f"ids of the last batch: {ids[0]:,} to {ids[-1]:,}; the index holds {growing.n:,}")
found = growing.search(rest, 10).indices
print(f"recall@10 on the other {len(rest):,} queries: {recall(found, truth[1000:]):.3f}")
print(growing.plan)
"""),
    md("""
Adding to an approximate index puts each vector in the list of its nearest centroid, and the centroids stay
as they were. That keeps recall, but vectors from regions the centroids never saw crowd into a few lists,
which slows searches. When the largest list has doubled against the mean since the index was clustered, or
the index has doubled in size, siann clusters it again (in the `add` that crosses the line), and the plan
says why. Here, 30,000 vectors from 20 new centres:
"""),
    code("""
new_centers = 3 + rng.standard_normal((20, 128), dtype=np.float32)
elsewhere = (new_centers[rng.integers(0, 20, 30_000)]
             + 0.3 * rng.standard_normal((30_000, 128), dtype=np.float32))
start = time.perf_counter()
for s in range(0, 30_000, 10_000):
    growing.add(elsewhere[s:s + 10_000])
print(f"added in {time.perf_counter() - start:.1f} s")
print("\\n".join(line for line in str(growing.plan).splitlines() if "merged" in line or "clustered" in line))
"""),
    md("""
Removed vectors are never returned, and the results stay exact. Vectors added with ids of your own are found
by them:
"""),
    code("""
near = growing.search(xq[:1], 3).indices[0]
growing.remove(near)
print(f"removed {near}; the nearest now: {growing.search(xq[:1], 3).indices[0]}")
mine = growing.add(xq[:2], ids=[10**9, 10**9 + 1])
print(f"added with ids {mine}: the nearest to the first query is {growing.search(xq[:1], 1).indices[0, 0]:,}")
"""),
    md("""
`index.recluster()` clusters the index again when you choose, and `index.save()` merges and compacts first
and keeps the ids.

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
silently.
"""),
    code("""
import scipy.sparse

try:
    siann.Index(scipy.sparse.random(100, 20, density=0.1, format="csr"), "ip")
except NotImplementedError as e:
    print("NotImplementedError:", e)
"""),
]


def main() -> None:
    nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {
        "name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"}})
    NotebookClient(nb, timeout=900, kernel_name="python3").execute()
    nbformat.sign.NotebookNotary().sign(nb)
    out = Path(__file__).with_name("quickstart.ipynb")
    nbformat.write(nb, out)
    print(out, f"{out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
