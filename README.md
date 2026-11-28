# siann

[![CI](https://github.com/vglazer/siann/actions/workflows/ci.yml/badge.svg)](https://github.com/vglazer/siann/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/siann)](https://pypi.org/project/siann/)
[![Documentation](https://img.shields.io/readthedocs/siann/latest)](https://siann.readthedocs.io)

**Nearest-neighbor search that decides how to search.** Give siann dense NumPy vectors, a metric (Euclidean
L2, cosine, inner product, Manhattan L1 or Chebyshev L-infinity) and either "exact" or a recall target. It
picks the method (exact search, an IVF or a graph) and, for each batch size, where to run it (the CPU, or the
GPU on Apple silicon), from measurements on your data and machine, and `index.plan` says why.

Every path is siann's own: Metal kernels on the Apple GPU (through MLX, or PyTorch's MPS backend) and Rust
on the CPU, on Linux, macOS and Windows. Where another library is faster, [How it compares](#how-it-compares)
says so. Start with the [quickstart notebook](https://github.com/vglazer/siann/blob/main/examples/quickstart.ipynb).

![One query (an orange dot) among 600 points, and its 25 nearest by three metrics, found by siann: under L2 they fill a circle around the query, under L1 a diamond, under L-infinity a square](https://raw.githubusercontent.com/vglazer/siann/main/assets/metrics.png)

*One query's 25 nearest points by Euclidean (L2), Manhattan (L1) and Chebyshev (L-infinity) distance.*

## Install

```sh
uv add siann            # or: pip install siann
```

Python 3.12 or later. siann runs on the CPU on Linux, macOS and Windows, and on the GPU too on Macs with Apple
silicon, where it installs [MLX](https://github.com/ml-explore/mlx), Apple's array framework. Wheels (built
packages) cover Linux on x86-64 and ARM, macOS on Apple silicon and Windows on x86-64; the x86-64 ones need a
processor with AVX2 (Intel from 2013, AMD from 2015), and `pip install --no-binary siann siann` builds from
source elsewhere, with a [Rust](https://rustup.rs) toolchain.

**Import siann before NumPy.** On Linux and Windows, siann sets up NumPy's BLAS as it loads, which can make
some CPU searches twice as fast; it warns if NumPy came first ([TUNING](TUNING.md#numpys-blas-on-linux-and-windows)).

## Python

```python
import siann                 # before NumPy: on Linux and Windows it sets NumPy's BLAS up first
import numpy as np

xb = np.random.default_rng(0).standard_normal((100_000, 128), dtype=np.float32)
xq = np.random.default_rng(1).standard_normal((1_000, 128), dtype=np.float32)

index = siann.Index(xb, metric="cosine")     # exact; or "l2", "l1", "linf", "ip"
res = index.search(xq, k=10)                 # res.indices, res.scores: (1000, 10) each, best first
print(index.plan)                            # what siann chose, and why

approx = siann.Index(xb, metric="cosine", recall=0.95, calibration_queries=xq)  # approximate
ids = approx.add(xq)                         # add vectors; searches return their ids
approx.remove(ids[:10])                      # remove them by id
approx.save("approx.siann")                  # siann.Index.load("approx.siann"), on any machine
```

Scores are each metric's natural value: the distance for L2 (not squared), L1 and L-infinity, the similarity
for cosine and inner product. Every choice can be overridden: `kind="exact"`, `"ivf"` or `"graph"`,
`device="cpu"` or `"gpu"`, `nprobe=` for one search, `gpu="torch"` for PyTorch instead of MLX, and a profile
from `siann.tune` ([TUNING](TUNING.md)).

## What siann decides

- **How hard to search.** A recall target is calibrated when the index is built: the fewest IVF lists to scan
  and the narrowest graph search that reach it on a sample of queries (`calibration_queries=`, or the index's
  own vectors). On new queries recall lands within a few thousandths of the target, either way.
- **Which method, on which device, for each batch size.** Every path, exact search included, is timed at batch
  sizes from 1 to 4,096 when the index is built, and each search takes the fastest for its size: typically
  the graph on the CPU for single queries and small batches, the GPU for large ones.
- **Graphs on any machine, by any metric.** siann builds its own graphs (CAGRA graphs, the kind NVIDIA's cuVS
  builds), on Linux and Windows too, for L1 and L-infinity as well: by L1 on SIFT1M, 6.2 times siann's IVF at
  recall 0.95. A graph that cannot reach the target is declined.
- **As the index changes,** it merges added vectors, recalibrates, re-clusters, and switches methods once
  another is faster; it can start empty, keep your own ids, and be searched from several threads while it
  changes.

GPU searches stay within a memory budget, and an index too large for the GPU stays on the CPU
([IMPLEMENTATION](IMPLEMENTATION.md) explains each choice).

## FAISS, cuVS and mlxvs

- **[FAISS](https://github.com/facebookresearch/faiss)** (Meta) is the standard for similarity search on the CPU,
  and the main baseline below. siann does not call it.
- **[cuVS](https://github.com/rapidsai/cuvs)** (NVIDIA) is GPU search on NVIDIA GPUs; CAGRA is its graph index.
  siann's graphs are CAGRA graphs, but siann does not use cuVS, and runs on no NVIDIA GPU.
- **[mlxvs](https://github.com/mikepapadim/mlxvs)** brings cuVS's methods to the Mac, on MLX. siann does not need
  it: with it installed (`pip install siann[graph]`, Apple silicon), siann has mlxvs build its graphs, faster on
  the GPU, and search them in GPU batches; without it, siann builds its graphs itself and searches them on the
  CPU, and sends GPU batches to its IVF or exact search.

## How it compares

On an Apple M3 Max: queries per second of siann's routed path divided by the other library's, above 1 where
siann is faster, exact search returning the same neighbors and approximate search compared at matched recall
(0.95 to 0.96). Against FAISS 1.16's faster path (CPU or Metal GPU) and mlxvs 0.16.1 on the GPU. FAISS's HNSW
and mlxvs's CAGRA are graph indexes. The datasets ([DATASETS](DATASETS.md)): SIFT1M (image descriptors, L2),
GloVe-100 (word embeddings, cosine), Music-100 (inner product), DBpedia OpenAI embeddings (1,536 dimensions).

| Workload | Against FAISS | Against mlxvs |
| --- | --- | --- |
| Exact, L1 or L-infinity | L1 **18.5-25x**, L-infinity **47-49x** (its CPU; neither GPU library has these metrics) | |
| Exact, L2, cosine or inner product, batches of 100 | **2.7-2.9x** (its GPU) | **2.6-3.8x** in batches; 1.1-1.2x at 1,536 dimensions |
| Exact, batches of 10,000 | L2 and cosine **5.6-6.0x** (its GPU), inner product **3.5x** (its CPU) | |
| Approximate, single queries | SIFT 0.93x, Music-100 0.72x, GloVe **1.5-3.1x** (HNSW, its fastest variant) | CAGRA on the GPU: SIFT and Music-100 **10-14x** |
| Approximate, batches of 100 | SIFT 0.96x, Music-100 0.90x, GloVe **1.2-1.8x** (HNSW) | CAGRA: SIFT 0.78x, Music-100 0.86x, GloVe level |
| Approximate, batches of 9,000-10,000 | SIFT **3.1x**, Music-100 **2.8x**, GloVe **5.0x** (HNSW) | CAGRA: SIFT 0.86x, Music-100 0.94x, GloVe level |
| Approximate, 1,536 dimensions (a million vectors) | single queries 0.36x; batches of 100 **3.9x**, of 9,000 **9.2x** (HNSW) | CAGRA: single queries level, batches 0.64-0.95x |

Without a GPU, against FAISS on the CPU: exact L1 3.8-4.3x, L-infinity 7.3x, L2 and cosine 1.7-3.6x, inner
product 0.9-2.7x. On 4 Linux CPUs, approximate, at single queries, batches of 10 and of 100: SIFT 1.16x,
0.84x, 0.99x; Music-100 0.78x, 0.94x, 1.78x. [BENCHMARKS](BENCHMARKS.md) has the measurements and the scripts
that reproduce them.

## Documentation

| Document | What it covers |
| --- | --- |
| [Quickstart notebook](https://github.com/vglazer/siann/blob/main/examples/quickstart.ipynb) | Every feature, hands-on |
| [IMPLEMENTATION](IMPLEMENTATION.md) | How siann chooses, the search methods, the kernels, where other libraries are faster |
| [TUNING](TUNING.md) | What siann measures, on what, and how to measure it on your own queries, data and hardware |
| [BENCHMARKS](BENCHMARKS.md) | Every measurement, with the scripts that reproduce them |
| [BACKENDS](BACKENDS.md) | MLX, PyTorch MPS and siann's Rust code, and FAISS as the baseline: what each offers, and their pitfalls |
| [DATASETS](DATASETS.md) | SIFT1M, GloVe-100, Music-100, DBpedia OpenAI and Gaussian vectors |
| [DEVELOPMENT](DEVELOPMENT.md) | Building, testing, releasing, the rules for changing the code, and the ideas that do not pay |
| [API reference](https://siann.readthedocs.io/en/latest/API/) | Every public class, method and function |

These pages are also at [siann.readthedocs.io](https://siann.readthedocs.io).

## License

MIT ([LICENSE](https://github.com/vglazer/siann/blob/main/LICENSE))
