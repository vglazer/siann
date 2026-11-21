# siann

[![CI](https://github.com/vglazer/siann/actions/workflows/ci.yml/badge.svg)](https://github.com/vglazer/siann/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/readthedocs/siann/latest)](https://siann.readthedocs.io)

**Fast nearest-neighbor search on Apple silicon,** exact or to a recall target, for dense NumPy vectors,
on the GPU and the CPU, by Euclidean (L2), cosine, inner product, Manhattan (L1) and Chebyshev (L-infinity)
distance. Give siann a recall target rather than settings: it calibrates the index to your data, measures
where each batch size runs fastest, on the CPU or the GPU, and `index.plan` says what it chose and why. Each
path it chooses between is fast in its own right, through custom GPU kernels and CPU code laid out for the
hardware ([IMPLEMENTATION](IMPLEMENTATION.md)).

Exact search beats FAISS on the CPU and mlxvs on the GPU almost everywhere, most of all for L1 and
L-infinity. Where another library is faster, as graph indexes are for approximate search on some data,
[Which to use](#which-to-use-and-how-it-compares) says so.

Start with the [quickstart notebook](https://github.com/vglazer/siann/blob/main/examples/quickstart.ipynb).

![One query (an orange dot) among 600 points, and its 25 nearest by three metrics, found by siann: under L2 they fill a circle around the query, under L1 a diamond, under L-infinity a square](https://raw.githubusercontent.com/vglazer/siann/main/assets/metrics.png)

*One query's 25 nearest points by Euclidean (L2), Manhattan (L1) and Chebyshev (L-infinity) distance.*

## Install

siann requires a Mac with Apple silicon, Python 3.12 or later, and a [Rust](https://rustup.rs)
toolchain to build it from source:

```sh
uv add git+https://github.com/vglazer/siann         # with uv
pip install git+https://github.com/vglazer/siann    # with pip
```

To run the quickstart notebook, clone the repository and, from its folder,
`uv run --group dev --with jupyterlab jupyter lab examples/quickstart.ipynb`.

## Documentation

| Document | What it covers |
| --- | --- |
| [Quickstart notebook](https://github.com/vglazer/siann/blob/main/examples/quickstart.ipynb) | The metrics, scores, plans, devices and tuning, hands-on |
| [siann.readthedocs.io](https://siann.readthedocs.io) | These pages, online |
| [IMPLEMENTATION](IMPLEMENTATION.md) | Why siann is fast: how it chooses the fastest way to meet a recall target, the search methods, the kernels in detail, where other libraries are faster, and what does not pay |
| [BACKENDS](BACKENDS.md) | MLX, PyTorch MPS, FAISS and siann's Rust code: what each offers, and their pitfalls |
| [DATASETS](DATASETS.md) | SIFT1M, GloVe-100, Music-100, DBpedia OpenAI and Gaussian vectors: sources, preparation and quirks |
| [BENCHMARKS](BENCHMARKS.md) | Which path to use for each workload, and every measurement behind it, with the scripts that reproduce them |
| [TUNING](TUNING.md) | Calibration and tuning: what siann measures, on what, and how to measure it on your own queries, data and hardware |
| [DEVELOPMENT](DEVELOPMENT.md) | Building, testing and linting, and the rules for changing the code |

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
```

Scores are each metric's natural value: the distance for L2 (not squared), L1 and L-infinity, the
similarity for cosine and inner product. `device="cpu"` or `"gpu"` overrides the choice of device, for the
index or for one search; a profile from `siann.tune` changes it for the indexes it is given to
([TUNING](TUNING.md)). With `recall=`, siann builds an IVF, finds the number of lists to probe that reaches
the target on a sample of the index's own vectors, and keeps exact search instead where that is faster;
the plan reports each choice. Passing `calibration_queries`, a sample of your own queries, is optional, and
keeps the target where they differ from the base vectors ([TUNING](TUNING.md)).

## Which to use, and how it compares

The fastest path for each workload, measured on an Apple M3 Max against the strongest option for the
case: FAISS 1.16 on the CPU (its compressed indexes included) and mlxvs 0.16.1 on the GPU (NVIDIA
cuVS's methods on MLX: exact search, IVF and CAGRA graphs), with exact search returning exactly the same
neighbors and approximate search compared at matched recall. "SIFT-like" and "GloVe-like" refer to the
real datasets measured, SIFT1M (image descriptors) and GloVe-100 (word embeddings), which favor
different methods; Music-100 is the inner-product dataset, and DBpedia OpenAI embeddings (1,536
dimensions) the high-dimensional one ([DATASETS](DATASETS.md)). FAISS's Metal
GPU path is level with mlxvs in small batches and slower in large ones, with no graph index, so mlxvs is
the GPU comparison ([BENCHMARKS](BENCHMARKS.md#faisss-metal-path-on-the-gpu)).

| Workload | Use | Against the strongest alternative |
| --- | --- | --- |
| Exact search, batches, GPU available | **siann's GPU kernels** | 2.6-5.7 times mlxvs at 100-128 dimensions, 1.1-1.2 times at 1,536 (100,000 vectors); L1 and L-infinity, which mlxvs lacks: 18-49 times FAISS on the CPU |
| Exact search on the CPU, L1 or L-infinity | **siann's Rust kernel** | 3.8-4.3 times FAISS (L1), 7.3 times (L-infinity) |
| Exact search on the CPU, L2, cosine or inner product | **siann's BLAS path** (NumPy's multiply, Accelerate, then siann's Rust top-k) | L2 and cosine 1.7-1.9 times FAISS on 1,000 queries, 3.0-3.6 times on 100; inner product 0.9-2.7 times |
| Approximate, large batches (thousands) | GloVe-like data, L1 and L-infinity: **siann's GPU IVF, list-major**; SIFT-like data: **mlxvs's CAGRA** | GloVe: level with mlxvs's CAGRA; SIFT: CAGRA 2.1-2.6 times siann's GPU IVF |
| Approximate, batches of about 100 | GloVe-like data, L1 and L-infinity: **siann's GPU IVF**; SIFT-like data: **mlxvs's CAGRA** or **FAISS HNSW** (about level) | GloVe: level with CAGRA, 1.2-1.8 times FAISS's HNSW; SIFT: CAGRA 1.2-1.5 times siann's GPU IVF up to recall 0.995, HNSW 1.2-1.5 times below 0.99 |
| Approximate, single queries, one core each (throughput) | GloVe-like data: **siann's Rust IVF**, two-level coarse step; SIFT-like data: **FAISS HNSW** | GloVe: 1.5-1.65 times HNSW; SIFT: HNSW 1.3-1.45 times faster; mlxvs on the GPU 2.7-5.2 times slower than siann's CPU path |
| Approximate, single query, lowest latency with cores to spare | **siann's Rust IVF, split over 4 threads**, except SIFT-like data below recall 0.97 | GloVe: 1.5-3.1 times HNSW; SIFT: 1.2-1.6 times from recall 0.97 |
| Approximate search by inner product | Music-100-like data: **siann's IVF** for single queries and batches of about 100, **mlxvs's CAGRA** for thousands; vectors of very uneven length (raw GloVe): **FAISS HNSW**, or **siann's exact GPU search** for recall 1.0 | Music-100: siann's IVF level with HNSW and CAGRA up to batches of 100, 2.1-2.3 times HNSW in batches of 10,000, where CAGRA is 1.45-1.5 times siann; raw GloVe: siann's exact search 0.88-1.35 times HNSW at its recall 0.89 |

For approximate search, store the lists as bytes when every coordinate is an integer from 0 to 255
(lossless), and otherwise as float16 for batches on the GPU (recall moves by at most 0.001) and float32
for single queries. On SIFT-like data, FAISS's HNSW and siann's GPU IVF cross over between 100 and
10,000 queries per batch below recall 0.99, and at about 100 from 0.99; on GloVe-like data, below 100.

**On the CPU, against FAISS** (queries per second at the nearest recall for approximate search):

| Workload | siann against FAISS |
| --- | ---: |
| Exact search, L-infinity (200,000 random vectors) | **7.3x** |
| Exact search, L1 | **3.8-4.3x** |
| Exact search, L2 and cosine | **1.7-1.9x** on 1,000 queries, **3.0-3.6x** on 100 |
| Exact search, inner product | 0.9-2.7x |
| Approximate, single queries on one thread, GloVe (against compressed HNSW) | **1.5-1.65x** |
| Approximate, single queries on one thread, SIFT (against compressed HNSW) | 0.69-0.77x (HNSW faster) |
| Approximate, single queries on one thread, Music-100 (inner product) | level (0.85-1.4x) |
| Approximate, single query split over 4 threads, against HNSW on one | GloVe **1.5-3.1x**; SIFT **1.2-1.6x** from recall 0.97 |

L1 and L-infinity have no matrix-multiply form, so FAISS computes them with plain loops (its L-infinity
takes three times as long as its L1), where siann runs kernels laid out like a matrix multiply. L2,
cosine and inner product are matrix multiplies, which FAISS runs through BLAS too, so siann gains less
there.

**On the GPU, against mlxvs** (queries per second at the nearest recall for approximate search):

| Workload | siann against mlxvs |
| --- | ---: |
| Exact search, every batch size, 100-128 dimensions | **2.6-5.7x** |
| Exact search, 1,536 dimensions (100,000 vectors) | 1.1-1.2x |
| siann's GPU IVF against mlxvs's IVF-Flat | **1.2-4.2x** |
| siann's GPU IVF against CAGRA, GloVe | level (0.9-1.1x) |
| siann's GPU IVF against CAGRA, Music-100 (inner product) | level in batches of 100, 0.67-0.69x in batches of 10,000 |
| siann's GPU IVF against CAGRA, SIFT | 0.38-0.83x (CAGRA faster) |
| siann's GPU IVF against CAGRA, 1,536-dimensional embeddings (100,000 vectors) | 0.35-0.53x (CAGRA faster) |
| L1 and L-infinity | mlxvs has neither; against FAISS on the CPU, the only alternative, **18-49x** exact |

[BENCHMARKS](BENCHMARKS.md) has the measurements behind each row and how to rerun them, and
[IMPLEMENTATION](IMPLEMENTATION.md) explains where the speed comes from.

## License

MIT ([LICENSE](https://github.com/vglazer/siann/blob/main/LICENSE))
