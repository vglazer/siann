# siann

[![CI](https://github.com/vglazer/siann/actions/workflows/ci.yml/badge.svg)](https://github.com/vglazer/siann/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/siann)](https://pypi.org/project/siann/)
[![Documentation](https://img.shields.io/readthedocs/siann/latest)](https://siann.readthedocs.io)

**Nearest-neighbor search that decides how to search, on Apple silicon.** Tell siann what you need, exact
neighbors or a recall target, for dense NumPy vectors by Euclidean (L2), cosine, inner product, Manhattan
(L1) or Chebyshev (L-infinity) distance. It chooses the method (exact search; an IVF, an index that
clusters the vectors and scans only the clusters nearest each query; or a graph, which links each vector to
near neighbors and walks the links; each calibrated to the target) and the device for each batch size (the
CPU or the Apple GPU) from measurements on your data and machine, says why in `index.plan`, and keeps
choosing as the index grows and changes.

Each path it chooses between is fast in its own right, through custom GPU kernels and CPU code laid out for
the hardware ([IMPLEMENTATION](IMPLEMENTATION.md)); where another library is faster,
[How it compares](#how-it-compares) says so. Start with the
[quickstart notebook](https://github.com/vglazer/siann/blob/main/examples/quickstart.ipynb).

![One query (an orange dot) among 600 points, and its 25 nearest by three metrics, found by siann: under L2 they fill a circle around the query, under L1 a diamond, under L-infinity a square](https://raw.githubusercontent.com/vglazer/siann/main/assets/metrics.png)

*One query's 25 nearest points by Euclidean (L2), Manhattan (L1) and Chebyshev (L-infinity) distance.*

## Install

```sh
uv add siann            # or: pip install siann
```

siann runs on Python 3.12 or later: on the CPU on Linux, macOS and Windows, and on the GPU too on Macs with
Apple silicon, where it installs [MLX](https://github.com/ml-explore/mlx), Apple's array framework, for it.
Wheels (built packages) cover Linux on x86-64 and ARM, macOS on Apple silicon and Windows on x86-64. The
x86-64 wheels need a processor with AVX2 (Intel from 2013, AMD from 2015); on an older one, `pip install
--no-binary siann siann` builds from source, which takes a [Rust](https://rustup.rs) toolchain. On Linux and
Windows, import siann before NumPy ([TUNING](TUNING.md#numpys-blas-on-linux-and-windows)). For graph
searches on the Apple GPU, `pip install siann[graph]` adds [mlxvs](https://github.com/mikepapadim/mlxvs);
without it, siann builds its graphs itself and searches them on the CPU.

## Python

```python
import numpy as np
import siann

xb = np.random.default_rng(0).standard_normal((100_000, 128), dtype=np.float32)
xq = np.random.default_rng(1).standard_normal((1_000, 128), dtype=np.float32)

index = siann.Index(xb, metric="cosine")     # exact; or "l2", "l1", "linf", "ip"
res = index.search(xq, k=10)                 # res.indices, res.scores: (1000, 10) each, best first
print(index.plan)                            # what siann chose, and why

approx = siann.Index(xb, metric="cosine", recall=0.95, calibration_queries=xq)  # an IVF and a graph
ids = approx.add(xq)                         # add vectors; searches return their ids
approx.remove(ids[:10])                      # remove them by id
approx.save("approx.siann")                  # siann.Index.load("approx.siann"), on any machine

grown = siann.Index(np.empty((0, 128), np.float32), metric="l2", recall=0.95)  # start empty, grow by add()
```

Scores are each metric's natural value: the distance for L2 (not squared), L1 and L-infinity, the similarity
for cosine and inner product. Every choice can be overridden: `device=`, `kind="exact"`, `"ivf"` or `"graph"`,
`nlist=`, `storage=`, `nprobe=` for one search, `gpu="torch"` for [PyTorch](https://pytorch.org)'s MPS
backend (its Apple GPU backend) instead of MLX, and a profile from `siann.tune` for the routing
([TUNING](TUNING.md)).

## What siann does for you

**It takes a target, not settings.** With `recall=0.95`, siann builds an IVF and a graph, finds how many
lists a query must probe and how wide a graph search must run to reach the target, and times every path,
exact search included, at each batch size. Calibrating on a sample of
your queries (`calibration_queries=`, or later `index.calibrate()`) makes the target hold on them
([TUNING](TUNING.md#calibration-meeting-the-recall-target-on-your-queries)).

**One index uses both the CPU and the Apple GPU,** each search sent where its batch size runs fastest on
your data and machine: typically single queries and small batches to the graph on the CPU, large batches to
the GPU, and exact search where it wins ([TUNING](TUNING.md#approximate-search-where-each-batch-runs)).

**Graphs on any machine, by any metric.** siann builds its graphs itself where mlxvs is not installed, on
Linux and Windows too, as good as mlxvs's and cuVS's; a saved index carries its graph to any machine; and
L1 and L-infinity get graphs too, 6.2 times siann's IVF on SIFT1M by L1 at recall 0.95
([IMPLEMENTATION](IMPLEMENTATION.md#graph-approximate-search)).

**Every common metric is fast,** L1 and L-infinity included, with kernels of their own on the GPU and in
Rust on the CPU ([IMPLEMENTATION](IMPLEMENTATION.md#distances)).

**The index keeps choosing as it changes.** It can start empty, grow and shrink, keep ids of your own, and
be searched from several threads while it changes; siann merges, recalibrates, re-clusters, and switches
from exact search to an IVF once that is faster ([IMPLEMENTATION](IMPLEMENTATION.md#changing-an-index)).

**It explains itself.** `index.plan` lists each choice and the measurement behind it: method, nprobe and
the recall reached, storage, routes, memory, what changed since the build.

**It is safe on unified memory.** GPU searches stay within a memory budget, so a large batch cannot stall
the Mac, and an index too large for the GPU stays on the CPU ([IMPLEMENTATION](IMPLEMENTATION.md#memory)).

## How it compares

Measured on an Apple M3 Max: queries per second of siann's routed path divided by the other's, above 1 where
siann is faster, with exact search returning the same neighbors and approximate search compared at matched
recall. Against [FAISS](https://github.com/facebookresearch/faiss) 1.16, Meta's similarity-search library,
the standard on the CPU (its faster path, CPU or Metal GPU); and on the GPU against
[mlxvs](https://github.com/mikepapadim/mlxvs) 0.16.1, NVIDIA's [cuVS](https://github.com/rapidsai/cuvs) GPU
search methods ported to MLX. FAISS's HNSW, mlxvs's CAGRA and siann's own graph are graph indexes: each
vector is linked to near neighbors, and a search walks the links, touching far fewer vectors than an IVF.
siann's approximate rows are its routed path: its graph on the CPU or the GPU, or its IVF, whichever it
measured faster for the batch. SIFT1M (image
descriptors, L2), GloVe-100 (word embeddings, cosine), Music-100 (inner product) and DBpedia OpenAI
embeddings (1,536 dimensions) favor different methods ([DATASETS](DATASETS.md)).

| Workload | Against FAISS | Against mlxvs |
| --- | --- | --- |
| Exact, L1 or L-infinity | L1 **18.5-25x**, L-infinity **47-49x** (its CPU; neither GPU library has these metrics) | |
| Exact, L2, cosine or inner product, batches of 100 | **2.7-2.9x** (its GPU) | **2.6-3.8x** in batches; 1.1-1.2x at 1,536 dimensions |
| Exact, batches of 10,000 | L2 and cosine **5.6-6.0x** (its GPU), inner product **3.5x** (its CPU) | |
| Approximate, single queries | SIFT 0.93x, Music-100 0.72x, GloVe **1.5-3.1x** (HNSW, its fastest variant) | CAGRA on the GPU: SIFT and Music-100 **10-14x** |
| Approximate, batches of 100 | SIFT 0.96x, Music-100 0.90x, GloVe **1.2-1.8x** (HNSW) | CAGRA: SIFT 0.78x, Music-100 0.86x, GloVe level |
| Approximate, batches of 9,000-10,000 | SIFT **3.1x**, Music-100 **2.8x**, GloVe **5.0x** (HNSW) | CAGRA: SIFT 0.86x, Music-100 0.94x, GloVe level |
| Approximate, 1,536 dimensions (a million vectors) | single queries 0.36x; batches of 100 **3.9x**, of 9,000 **9.2x** (HNSW) | CAGRA: single queries level, batches 0.64-0.95x |

FAISS's GPU IVF is no faster than its HNSW at matched recall in any case measured, so HNSW is FAISS's
approximate method on the Mac. Without a GPU, siann's CPU path against FAISS's: exact L1 3.8-4.3x and
L-infinity 7.3x, L2 and cosine 1.7-1.9x on 1,000 queries and 3.0-3.6x on 100, inner product 0.9-2.7x. On
4 Linux CPUs, with siann's index and graph built there, against FAISS's HNSW at single queries, batches of 10
and of 100: SIFT **1.16x**, 0.84x, 0.99x; Music-100 0.78x, 0.94x, **1.78x**.
[BENCHMARKS](BENCHMARKS.md) has the measurements behind each row and how to rerun them.

## Documentation

| Document | What it covers |
| --- | --- |
| [Quickstart notebook](https://github.com/vglazer/siann/blob/main/examples/quickstart.ipynb) | Every feature, hands-on |
| [IMPLEMENTATION](IMPLEMENTATION.md) | How siann chooses, the search methods, the kernels, where other libraries are faster |
| [TUNING](TUNING.md) | What siann measures, on what, and how to measure it on your own queries, data and hardware |
| [BENCHMARKS](BENCHMARKS.md) | Every measurement, with the scripts that reproduce them |
| [BACKENDS](BACKENDS.md) | MLX, PyTorch MPS, FAISS and siann's Rust code: what each offers, and their pitfalls |
| [DATASETS](DATASETS.md) | SIFT1M, GloVe-100, Music-100, DBpedia OpenAI and Gaussian vectors |
| [DEVELOPMENT](DEVELOPMENT.md) | Building, testing, releasing, the rules for changing the code, and the ideas that do not pay |
| [API reference](https://siann.readthedocs.io/en/latest/API/) | Every public class, method and function |

These pages are also at [siann.readthedocs.io](https://siann.readthedocs.io).

## License

MIT ([LICENSE](https://github.com/vglazer/siann/blob/main/LICENSE))
