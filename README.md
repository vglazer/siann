# siann

[![CI](https://github.com/vglazer/siann/actions/workflows/ci.yml/badge.svg)](https://github.com/vglazer/siann/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/siann)](https://pypi.org/project/siann/)
[![Documentation](https://img.shields.io/readthedocs/siann/latest)](https://siann.readthedocs.io)

**Nearest-neighbor search that decides how to search, on Apple silicon.** Tell siann what you need, exact
neighbors or a recall target, for dense NumPy vectors by Euclidean (L2), cosine, inner product, Manhattan
(L1) or Chebyshev (L-infinity) distance, and it chooses the method (exact search, or an IVF calibrated to
the target) and the device for each batch size (the CPU or the Apple GPU), from measurements on your data
and machine. `index.plan` says what it chose and why.

The choice holds as the data changes. An index can start empty and grow, take removals, and be searched from
several threads while it changes; siann merges, re-clusters and recalibrates as it goes, and switches from
exact search to an IVF once that is faster. An index saves to one file that loads on any machine, and the
CPU path runs on Linux and Windows too.

Each path it chooses between is fast in its own right ([IMPLEMENTATION](IMPLEMENTATION.md)): exact search
is 2.6-3.8 times the fastest alternative on the Mac for L2, cosine and inner product (1.1-1.2 times at 1,536
dimensions, where the matrix multiply dominates) and 18-49 times for L1 and L-infinity, which other GPU
libraries lack. Where another library is faster, as graph indexes are for approximate search on some data,
[How it compares](#how-it-compares) says so.

Start with the [quickstart notebook](https://github.com/vglazer/siann/blob/main/examples/quickstart.ipynb).

![One query (an orange dot) among 600 points, and its 25 nearest by three metrics, found by siann: under L2 they fill a circle around the query, under L1 a diamond, under L-infinity a square](https://raw.githubusercontent.com/vglazer/siann/main/assets/metrics.png)

*One query's 25 nearest points by Euclidean (L2), Manhattan (L1) and Chebyshev (L-infinity) distance.*

## Install

siann is [on PyPI](https://pypi.org/project/siann/) and runs on Python 3.12 or later: on the CPU on
Linux, macOS and Windows, and on the GPU too on Macs with Apple silicon, where it installs MLX for it.

```sh
uv add siann            # with uv
pip install siann       # with pip
```

There are wheels (built packages) for Linux on x86-64 and ARM, macOS on Apple silicon and Windows on x86-64.
The x86-64 wheels need a processor with AVX2 (Intel from 2013's Haswell, AMD from 2015's Excavator); on an
older one, `pip install --no-binary siann siann` builds siann from source, which takes a
[Rust](https://rustup.rs) toolchain. On Linux and Windows, import siann before NumPy
([TUNING](TUNING.md#numpys-blas-on-linux-and-windows)).

To run the quickstart notebook, clone the repository and, from its folder,
`uv run --group dev --with jupyterlab jupyter lab examples/quickstart.ipynb`.

## Documentation

| Document | What it covers |
| --- | --- |
| [Quickstart notebook](https://github.com/vglazer/siann/blob/main/examples/quickstart.ipynb) | The metrics, scores, plans, devices and tuning, hands-on |
| [siann.readthedocs.io](https://siann.readthedocs.io) | These pages, online |
| [IMPLEMENTATION](IMPLEMENTATION.md) | Why siann is fast: how it chooses the fastest way to meet a recall target, the search methods, the kernels in detail, and where other libraries are faster |
| [BACKENDS](BACKENDS.md) | MLX, PyTorch MPS, FAISS and siann's Rust code: what each offers, and their pitfalls |
| [DATASETS](DATASETS.md) | SIFT1M, GloVe-100, Music-100, DBpedia OpenAI and Gaussian vectors: sources, preparation and quirks |
| [BENCHMARKS](BENCHMARKS.md) | Which path to use for each workload, and every measurement behind it, with the scripts that reproduce them |
| [TUNING](TUNING.md) | Calibration and tuning: what siann measures, on what, and how to measure it on your own queries, data and hardware |
| [DEVELOPMENT](DEVELOPMENT.md) | Building, testing and linting, the rules for changing the code, and the ideas that do not pay |
| [API reference](https://siann.readthedocs.io/en/latest/API/) | Every public class, method and function, from the docstrings |

## Python

```python
import numpy as np
import siann

xb = np.random.default_rng(0).standard_normal((100_000, 128), dtype=np.float32)
xq = np.random.default_rng(1).standard_normal((1_000, 128), dtype=np.float32)
index = siann.Index(xb, metric="cosine")     # or "l2", "l1", "linf", "ip"
res = index.search(xb[:5], k=10)             # exact, on the GPU or the CPU by batch size
res.indices, res.scores                      # (5, 10) each, best first
print(index.plan)                            # what was chosen, and why
profile = siann.tune(xb)                     # measure the GPU/CPU choice on your data

approx = siann.Index(xb, metric="cosine", recall=0.95)   # approximate: an IVF, calibrated to the target
approx.search(xq, k=10)
approx = siann.Index(xb, metric="cosine", recall=0.95, calibration_queries=xq[:1000])  # optional: on your queries
approx.save("approx.siann")                  # and siann.Index.load("approx.siann"), on any machine
on_torch = siann.Index(xb, metric="cosine", gpu="torch")  # PyTorch's MPS backend instead of MLX (siann[torch])

ids = approx.add(xq)                         # add vectors: returns their ids, which searches return
approx.add(xq[:2], ids=[10**9, 10**9 + 1])   # or name them yourself
approx.remove(ids[:10])                      # remove vectors by id; never returned again
approx.recluster()                           # optional: siann re-clusters by itself when the lists fall out of balance
approx.calibrate(xq[:1000])                  # recalibrate on your queries, once you have them

grown = siann.Index(np.empty((0, 128), np.float32), metric="l2", recall=0.95)  # start empty, grow by add()
```

Scores are each metric's natural value: the distance for L2 (not squared), L1 and L-infinity, the
similarity for cosine and inner product. Every choice siann makes can be overridden: `device="cpu"` or
`"gpu"` for the index or for one search, `kind="exact"` or `"ivf"`, `nlist=`, `storage=`, `nprobe=` for
one search, and a profile from `siann.tune` for the routing ([TUNING](TUNING.md)).

## What siann does for you

**It takes a target, not settings.** Give `recall=0.95` and siann builds an IVF, finds the number of lists
each query must probe to reach the target, times that against exact search and keeps the faster. The target
holds for queries like the ones it calibrated on: the index's own vectors by default, or a sample of your
queries, passed as `calibration_queries=` or later to `index.calibrate()`
([TUNING](TUNING.md#calibration-meeting-the-recall-target-on-your-queries)).

**One index searches on both the CPU and the Apple GPU,** each search sent where its batch size runs
fastest: typically single queries to the CPU and batches to the GPU, but the crossovers move with the data
and the machine. An approximate index times its paths at batch sizes from 1 to 4,096 when it is built;
exact search routes by a profile, siann's own or one `siann.tune` measures on your data and machine
([TUNING](TUNING.md#approximate-search-where-each-batch-runs)). The GPU kernels run through MLX, or through
PyTorch's MPS backend with `gpu="torch"`, with the same results ([BACKENDS](BACKENDS.md)).

**Every common metric is fast.** L1 and L-infinity get kernels of their own, on the GPU and in Rust on the
CPU, rather than being left to plain loops; cosine and inner product return similarities, the distances
are not squared ([IMPLEMENTATION](IMPLEMENTATION.md#distances)).

**The index keeps choosing as it changes.** It can start empty and grow by `add()`, lose vectors by
`remove()`, and keep ids of your own. Added vectors are searched exactly until enough have gathered to merge;
at each merge siann recalibrates, re-clusters when the lists fall out of balance or the index has doubled,
and makes the choice between exact search and an IVF again as the index grows. Searches from other threads
see the index as it was when they began ([IMPLEMENTATION](IMPLEMENTATION.md#changing-an-index)).

**It explains itself.** `index.plan` lists each choice and the measurement behind it: the method, nprobe and
the recall it reached, the storage, the routes, the memory it may use, what changed since the build. Where
another library is faster, [How it compares](#how-it-compares) says so.

**It is safe on unified memory.** On Apple silicon the GPU's memory is the machine's, and a search that
outgrows it stalls the system rather than failing. siann splits GPU searches to stay within a budget, which
never changes their results, and keeps an index too large for the GPU on the CPU
([IMPLEMENTATION](IMPLEMENTATION.md#memory)).

## How it compares

Measured on an Apple M3 Max, with exact search returning exactly the same neighbors and approximate search
compared at matched recall: queries per second of siann's routed path divided by the other's, so above 1
siann is faster and below 1 the other is. SIFT1M (image descriptors, L2), GloVe-100 (word embeddings,
cosine), Music-100 (inner product) and DBpedia OpenAI embeddings (1,536 dimensions) favor different methods
([DATASETS](DATASETS.md)).

**Against the fastest alternative for each workload,** FAISS 1.16 (on the CPU or its Metal GPU path) or
mlxvs 0.16.1 (NVIDIA cuVS's methods on MLX, CAGRA graphs included):

| Workload | Fastest alternative | siann against it |
| --- | --- | --- |
| Exact search, L1 or L-infinity | FAISS on the CPU (neither its GPU path nor mlxvs has these metrics) | **18-49x** |
| Exact search, L2, cosine or inner product, batches | mlxvs or FAISS's GPU path (level with each other) | **2.6-3.8x**; 1.1-1.2x at 1,536 dimensions |
| Approximate, single queries | FAISS's HNSW on the CPU | GloVe **1.5-1.65x**, Music-100 0.85-1.4x, SIFT 0.69-0.77x; split over 4 threads, GloVe **1.5-3.1x**, SIFT **1.2-1.6x** from recall 0.97 |
| Approximate, batches of about 100 | mlxvs's CAGRA, or FAISS's HNSW (about level) | GloVe level with CAGRA, Music-100 level with both, SIFT 0.67-0.83x |
| Approximate, batches of 10,000 | mlxvs's CAGRA | GloVe level, Music-100 0.67-0.69x, SIFT 0.38-0.48x |
| Approximate, 1,536 dimensions (100,000 vectors) | mlxvs's CAGRA; FAISS's HNSW for single queries | 0.35-0.53x; single queries 0.37-0.42x |

**Against FAISS on the Mac,** its faster path for each workload, CPU or GPU:

| Workload | FAISS's faster path | siann against it |
| --- | --- | --- |
| Exact search, L2, cosine or inner product, batches of 100 | its GPU | **2.7-2.9x** |
| Exact search, L2 or cosine, batches of 10,000 | its GPU (in calls of 1,000, as larger ones exhaust memory) | **5.6-6.0x** |
| Exact search, inner product, batches of 10,000 | its CPU | **3.5x** |
| Exact search, L1 or L-infinity | its CPU (its GPU path has neither) | L1 **18.5-25x**, L-infinity **47-49x** |
| Approximate, single queries | HNSW on the CPU | GloVe **1.5-1.65x**, Music-100 0.85-1.4x, SIFT 0.69-0.77x; split over 4 threads, GloVe **1.5-3.1x**, SIFT **1.2-1.6x** from recall 0.97 |
| Approximate, batches of 100 | HNSW on the CPU | GloVe **1.2-1.8x**, Music-100 0.77-1.05x, SIFT 0.67-0.84x below recall 0.99, level at 0.99 |
| Approximate, batches of 10,000 | HNSW on the CPU | GloVe **5.0x**, Music-100 **2.1-2.3x**, SIFT **1.2-1.3x** |

FAISS's GPU IVF is no faster than its own HNSW at matched recall in any case measured, so HNSW is FAISS's
approximate method on the Mac. Without a GPU, siann's CPU path against FAISS's: exact L1 3.8-4.3x and
L-infinity 7.3x (FAISS computes them with plain loops), L2 and cosine 1.7-1.9x on 1,000 queries and
3.0-3.6x on 100, inner product 0.9-2.7x.

[BENCHMARKS](BENCHMARKS.md) has the measurements behind each row and how to rerun them, and
[IMPLEMENTATION](IMPLEMENTATION.md) explains where the speed comes from.

## License

MIT ([LICENSE](https://github.com/vglazer/siann/blob/main/LICENSE))
