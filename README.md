# siann

[![Documentation](https://img.shields.io/readthedocs/siann/latest)](https://siann.readthedocs.io)

*Pronounced like "cyan".*

**si**licon **a**pproximate **n**earest **n**eighbors: fast similarity search on Apple silicon, for
dense NumPy vectors, across metrics (L1, L2, L-infinity, cosine, inner product). siann runs each search
on the CPU (its own Rust code, and BLAS through NumPy) or the GPU ([MLX](https://github.com/ml-explore/mlx),
or the same kernels on PyTorch's MPS backend), measures every path against the strongest alternative on
the same Mac, [FAISS](https://github.com/facebookresearch/faiss) on the CPU and
[mlxvs](https://github.com/mikepapadim/mlxvs) on the GPU, and recommends the fastest for each
workload, naming the other library where it is faster.

## Which to use, and how it compares

The fastest path for each workload, measured on an Apple M3 Max against the strongest option for the
case: FAISS 1.16 on the CPU (its compressed indexes included) and mlxvs 0.16.1 on the GPU (NVIDIA
cuVS's methods on MLX: exact search, IVF and CAGRA graphs), with exact search returning exactly the same
neighbors and approximate search compared at matched recall. "SIFT-like" and "GloVe-like" refer to the
real datasets measured, SIFT1M (image descriptors) and GloVe-100 (word embeddings), which favor
different methods; Music-100 is the inner-product dataset ([DATASETS](DATASETS.md)). FAISS's Metal
backend is not part of these comparisons.

| Workload | Use | Against the strongest alternative |
| --- | --- | --- |
| Exact search, batches, GPU available | **siann's GPU kernels** (MLX) | L2 and cosine 8 times FAISS; L1 18-25 times; L-infinity 47-49 times (random vectors); inner product 4-18 times; 2.6-5.7 times mlxvs, which has no L1 or L-infinity |
| Exact search on the CPU, L1 or L-infinity | **siann's Rust kernel** | 3.8-4.3 times FAISS (L1), 7.3 times (L-infinity) |
| Exact search on the CPU, L2, cosine or inner product | **siann's BLAS path** (NumPy's multiply, Accelerate, then siann's Rust top-k) | L2 and cosine 1.7-1.9 times FAISS on 1,000 queries, 3.0-3.6 times on 100; inner product 0.9-2.7 times |
| Approximate, large batches (thousands) | GloVe-like data, L1 and L-infinity: **siann's GPU IVF, list-major**; SIFT-like data: **mlxvs's CAGRA** | GloVe: 5.0 times FAISS's HNSW, level with CAGRA; SIFT: CAGRA 2.1-2.6 times siann's GPU IVF, which is 1.2-1.3 times HNSW |
| Approximate, batches of about 100 | GloVe-like data, L1 and L-infinity: **siann's GPU IVF**; SIFT-like data: **mlxvs's CAGRA** or **FAISS HNSW** (about level) | GloVe: 1.2-1.8 times HNSW, level with CAGRA; SIFT: CAGRA 1.2-1.5 times siann's GPU IVF up to recall 0.995, HNSW 1.2-1.5 times below 0.99 |
| Approximate, single queries, one core each (throughput) | GloVe-like data: **siann's Rust IVF**, two-level coarse step; SIFT-like data: **FAISS HNSW** | GloVe: 1.5-1.65 times HNSW; SIFT: HNSW 1.3-1.45 times faster; mlxvs on the GPU 2.7-5.2 times slower than siann's CPU path |
| Approximate, single query, lowest latency with cores to spare | **siann's Rust IVF, split over 4 threads**, except SIFT-like data below recall 0.97 | GloVe: 1.5-3.1 times HNSW; SIFT: 1.2-1.6 times from recall 0.97 |
| Approximate search by inner product | Music-100-like data: **siann's IVF** for single queries and batches of about 100, **mlxvs's CAGRA** for thousands; vectors of very uneven length (raw GloVe): **FAISS HNSW**, or **siann's exact GPU search** for recall 1.0 | Music-100: siann's IVF level with HNSW and CAGRA up to batches of 100, 2.1-2.3 times HNSW in batches of 10,000, where CAGRA is 1.45-1.5 times siann; raw GloVe: siann's exact search 0.88-1.35 times HNSW at its recall 0.89 |

For approximate search, store the lists as bytes when every coordinate is an integer from 0 to 255
(lossless), and otherwise as float16 for batches on the GPU (recall moves by at most 0.001) and float32
for single queries. On SIFT-like data, FAISS's HNSW and siann's GPU IVF cross over between 100 and
10,000 queries per batch below recall 0.99, and at about 100 from 0.99; on GloVe-like data, below 100.

**How much faster, exact search** (the speed-up over FAISS's flat index; 1,000 queries; L-infinity on
200,000 Gaussian vectors, inner product on raw GloVe-100 and Music-100):

| Metric | siann's GPU | siann's CPU |
| --- | ---: | ---: |
| L-infinity | **47-49x** (random vectors only) | **7.3x** |
| L1 | **18-25x** | **3.8-4.3x** |
| L2 and cosine | **8x** | 1.7-3.6x |
| Inner product | **4-18x** | 0.9-2.7x |

L1 and L-infinity have no matrix-multiply form, so FAISS computes them with plain loops (its L-infinity
takes three times as long as its L1), where siann runs kernels laid out like a matrix multiply. L2,
cosine and inner product are matrix multiplies, which FAISS runs through BLAS too, so on the CPU siann
gains less there.

**How much faster, approximate search** (siann's GPU IVF against FAISS's HNSW, queries per second, at
the nearest recall):

| Dataset, batch | Against float32 HNSW | Against compressed HNSW |
| --- | ---: | ---: |
| GloVe, 10,000 queries (recall 0.84) | **6.0x** (191,532 against 32,114) | **5.0x** (199,030 against 40,092) |
| SIFT L2, 10,000 (recall 0.97) | **2.7x** (181,901 against 68,227) | **1.3x** (199,347 against 153,710) |
| SIFT L1, 10,000 (recall 0.96) | **2.1x** (123,855 against 58,059) | (FAISS has no compressed L1) |
| GloVe, batches of 100 (recall 0.84) | **1.4x** (38,744 against 28,088) | **1.8x** (60,559 against 34,348) |
| SIFT L2, batches of 100 | 0.65x at recall 0.97 | 0.84x at 0.97, 0.99x at 0.99, **1.66x** at 0.997 |
| SIFT L1, batches of 100 (recall 0.96) | 0.55x | (FAISS has no compressed L1) |

**Against mlxvs on the GPU** (queries per second at the nearest recall):

| Comparison | siann against mlxvs |
| --- | ---: |
| Exact search, every batch size | **2.6-5.7x** |
| siann's GPU IVF against mlxvs's IVF-Flat | **1.2-4.2x** |
| siann's GPU IVF against CAGRA, GloVe | level (0.9-1.1x) |
| siann's GPU IVF against CAGRA, Music-100 (inner product) | level in batches of 100, 0.67-0.69x in batches of 10,000 |
| siann's GPU IVF against CAGRA, SIFT | 0.38-0.83x (CAGRA faster) |
| L1 and L-infinity | mlxvs has neither |

[BENCHMARKS](BENCHMARKS.md) has the measurements behind each row and how to rerun them, and
[ARCHITECTURE](ARCHITECTURE.md) explains where the speed comes from.

## Documentation

| Document | What it covers |
| --- | --- |
| [ARCHITECTURE](ARCHITECTURE.md) | Why siann is faster than FAISS (and where it is not), with diagrams of how the pieces fit and which path a search takes |
| [ALGORITHMS](ALGORITHMS.md) | How exact search and IVF work on the GPU and the CPU, and what does not pay |
| [BACKENDS](BACKENDS.md) | MLX, PyTorch MPS, FAISS and siann's Rust code: what each offers, and their pitfalls |
| [DATASETS](DATASETS.md) | SIFT1M, GloVe-100, Music-100 and Gaussian vectors: sources, preparation and quirks |
| [BENCHMARKS](BENCHMARKS.md) | Which path to use for each workload, and every measurement behind it, with the scripts that reproduce them |
