# siann

[![Documentation](https://img.shields.io/readthedocs/siann/latest)](https://siann.readthedocs.io)

*Pronounced like "cyan".*

**si**licon **a**pproximate **n**earest **n**eighbors: fast similarity search on Apple silicon, for
dense NumPy vectors (sparse vectors, such as TF-IDF or SPLADE, are not in scope), across metrics (L1,
L2, L-infinity, cosine, inner product), and a benchmark of how much each Mac compute backend buys for
it: on the GPU, [MLX](https://github.com/ml-explore/mlx), PyTorch's MPS backend and
[jax-mps](https://github.com/tillahoffmann/jax-mps); on the CPU, [FAISS](https://github.com/facebookresearch/faiss),
the frameworks' CPU backends and siann's own Rust code.

## Which to use, and how it compares

The fastest path for each workload, measured on an Apple M3 Max against FAISS 1.16 on the CPU, each
comparison against FAISS's strongest option for the case (its compressed indexes included), with
exact search returning exactly FAISS's neighbors and approximate search compared at matched recall.
"SIFT-like" and "GloVe-like" refer to the two real datasets measured, SIFT1M (image descriptors) and
GloVe-100 (word embeddings), which favor different methods ([DATASETS](DATASETS.md)):

| Workload | Use | Against FAISS's strongest option |
| --- | --- | --- |
| Exact search, batches, GPU available | **siann's GPU kernels** (MLX) | L2 and cosine 8 times FAISS; L1 18-25 times; L-infinity 47-49 times (random vectors); inner product 4-18 times |
| Exact search on the CPU, L1 or L-infinity | **siann's Rust kernel** | 3.8-4.3 times (L1), 7.3 times (L-infinity) |
| Exact search on the CPU, L2, cosine or inner product | **siann's BLAS path** (NumPy's multiply, Accelerate, then siann's Rust top-k) | L2 and cosine 1.7-1.9 times on 1,000 queries, 3.0-3.6 times on 100; inner product level to 2.7 times |
| Approximate, large batches (thousands) | **siann's GPU IVF, list-major** | GloVe: 5.0 times HNSW; SIFT: 1.2-1.3 times |
| Approximate, batches of about 100 | GloVe-like data: **siann's GPU IVF**; SIFT-like data: **FAISS HNSW** below recall 0.99, **siann's GPU IVF** from 0.99 | GloVe: 1.2-1.8 times HNSW; SIFT: HNSW 1.2-1.5 times faster below 0.99, level at 0.99, siann 1.7 times at 0.997 |
| Approximate, single queries, one core each (throughput) | GloVe-like data: **siann's Rust IVF**, two-level coarse step; SIFT-like data: **FAISS HNSW** | GloVe: 1.5-1.65 times HNSW; SIFT: HNSW 1.3-1.45 times faster |
| Approximate, single query, lowest latency with cores to spare | **siann's Rust IVF, split over 4 threads**, except SIFT-like data below recall 0.97 | GloVe: 1.5-3.1 times HNSW; SIFT: 1.2-1.6 times from recall 0.97 |
| Inner product on vectors of uneven length | **siann's exact search on the GPU** (an IVF fails there) | recall 1.0 at 0.88-1.36 times the speed of FAISS's HNSW at recall 0.89 |

For approximate search, store the lists as bytes when every coordinate is an integer from 0 to 255
(lossless), and otherwise as float16 for batches on the GPU (recall moves by at most 0.001) and float32
for single queries. On SIFT-like data, HNSW and the GPU cross over between 100 and 10,000 queries per
batch below recall 0.99, and at about 100 from 0.99; on GloVe-like data, below 100.

**How much faster, exact search** (the speed-up over FAISS's flat index; 1,000 queries; L-infinity on
200,000 Gaussian vectors, inner product on raw GloVe-100):

| Metric | siann's GPU | siann's CPU |
| --- | ---: | ---: |
| L-infinity | **47-49x** (random vectors only) | **7.3x** |
| L1 | **18-25x** | **3.8-4.3x** |
| L2 and cosine | **8x** | 1.7-3.6x |
| Inner product | **4-18x** | level to 2.7x |

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

[BENCHMARKS](BENCHMARKS.md) has the measurements behind each row and how to rerun them, and
[ARCHITECTURE](ARCHITECTURE.md) explains where the speed comes from.

## Documentation

| Document | What it covers |
| --- | --- |
| [ARCHITECTURE](ARCHITECTURE.md) | Why siann is faster than FAISS (and where it is not), with diagrams of how the pieces fit and which path a search takes |
| [ALGORITHMS](ALGORITHMS.md) | How exact search and IVF work on the GPU and the CPU, and what does not pay |
| [BACKENDS](BACKENDS.md) | MLX, PyTorch MPS, jax-mps, FAISS and siann's Rust code: what each offers, and their pitfalls |
| [DATASETS](DATASETS.md) | SIFT1M, GloVe-100 and Gaussian vectors: sources, preparation and quirks |
| [BENCHMARKS](BENCHMARKS.md) | Which path to use for each workload, and every measurement behind it, with the scripts that reproduce them |
