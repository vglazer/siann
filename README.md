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
| Approximate, large batches (thousands) | **siann's GPU IVF, list-major** | GloVe: 3.8 times HNSW; SIFT: level at recall 0.97, 1.25 times at 0.84 |
| Approximate, batches of about 100 | SIFT-like data: **FAISS HNSW**; GloVe-like data: **siann's GPU IVF** | SIFT: HNSW about 2 times faster; GloVe: level (1.0-1.07 times) |
| Approximate, single queries, one core each (throughput) | GloVe-like data: **siann's Rust IVF**, two-level coarse step; SIFT-like data: **FAISS HNSW** | GloVe: 1.5-1.65 times HNSW; SIFT: HNSW 1.3-1.45 times faster |
| Approximate, single query, lowest latency with cores to spare | **siann's Rust IVF, split over 4 threads**, except SIFT-like data below recall 0.97 | GloVe: 1.5-3.1 times HNSW; SIFT: 1.2-1.6 times from recall 0.97 |
| Inner product on vectors of uneven length | **exact search on the GPU** (an IVF fails there) | recall 1.0 at 0.88-1.36 times the speed of FAISS's HNSW at recall 0.89 |

For approximate search, store the lists as bytes when every coordinate is an integer from 0 to 255
(lossless), and otherwise as float16 for batches on the GPU (recall moves by at most 0.001) and float32
for single queries. Where HNSW and the GPU cross over depends on the data: between 100 and 10,000
queries per batch on SIFT, and at about 100 on GloVe.

[BENCHMARKS](BENCHMARKS.md) has the measurements behind each row and how to rerun them, and
[ARCHITECTURE](ARCHITECTURE.md) explains where the speed comes from.

## Documentation

| Document | What it covers |
| --- | --- |
| [ARCHITECTURE](ARCHITECTURE.md) | Why siann is faster than FAISS (and where it is not), with diagrams of how the pieces fit and which path a search takes |
| [ALGORITHMS](ALGORITHMS.md) | How exact search and IVF work on the GPU and the CPU, and what does not pay |
| [BACKENDS](BACKENDS.md) | MLX, PyTorch MPS, jax-mps, FAISS and siann's Rust code: what each offers, and their pitfalls |
| [DATASETS](DATASETS.md) | SIFT1M, GloVe-100 and Gaussian vectors: sources, preparation and quirks |
| [BENCHMARKS](BENCHMARKS.md) | Which backend to use for each workload, and the measurements behind it |
| [RESULTS](RESULTS.md) | Every measurement, with the script that produced it |
