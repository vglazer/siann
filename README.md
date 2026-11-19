# siann

*Pronounced like "cyan".*

**si**licon **a**pproximate **n**earest **n**eighbors: fast similarity search on Apple silicon, for
dense NumPy vectors (sparse vectors, such as TF-IDF or SPLADE, are not in scope), across metrics (L1,
L2, L-infinity, cosine, inner product), and a benchmark of how much each Mac compute backend buys for
it: on the GPU, [MLX](https://github.com/ml-explore/mlx), PyTorch's MPS backend and
[jax-mps](https://github.com/tillahoffmann/jax-mps); on the CPU, [FAISS](https://github.com/facebookresearch/faiss),
the frameworks' CPU backends and siann's own Rust code.

## Findings so far

Measured on an Apple M3 Max against FAISS 1.16 on the CPU, with every result checked against exact
search:

- **Exact search on the GPU,** with siann's custom kernels: 8 times FAISS for L2 and 18-25 times for
  L1, on a million vectors in batches of 1,000 queries, on SIFT1M, GloVe-100 and random data.
- **The GPU's speed comes from custom kernels.** With only built-in operations, MLX, PyTorch and JAX
  are all slower on the GPU than FAISS on the CPU. The same custom kernel runs at the same speed in
  MLX and PyTorch; JAX on a Mac has no way to write one.
- **Exact L1 on the CPU,** in safe Rust: 3.8 times FAISS.
- **Exact L2 on the CPU without FAISS:** BLAS through NumPy (Accelerate, the library FAISS uses too)
  with a Rust top-k and cache-sized tiles: 1.7-1.9 times FAISS on 1,000 queries, 3.0-3.6 times on 100.
- **Approximate search (IVF):** on large batches, the GPU is 1.1-2 times FAISS's HNSW on SIFT1M and
  2.6-4.6 times on GloVe-100. For single queries, siann's Rust IVF beats HNSW on GloVe-100 (1.7-1.9
  times) and trails it on SIFT1M (0.9 times with L1, 0.7-0.8 with L2). HNSW wins small batches.
- **L-infinity** (the largest coordinate difference), exact search, on 200,000 random vectors: siann's
  GPU kernels 47-49 times FAISS and its Rust kernel 7.3 times. FAISS takes three times as long for
  L-infinity as for L1; siann's GPU kernels take the same time for both. Measured on random vectors
  only.
- **Smaller IVF lists** (float16, or bytes for data like SIFT's) speed up batches of about 100
  queries by 1.3-2.9 times, with recall unchanged, and slow down single queries. But FAISS's HNSW
  gains as much from the same compression, so the gap to it does not close on SIFT.
- **Single queries split over 4 threads** cut the Rust IVF's latency 1.3-2.0 times. Against FAISS's
  compressed HNSW (on one thread), that wins on GloVe-100 (1.5-3.1 times) but not on SIFT1M, where
  HNSW is faster or level. See [BENCHMARKS](BENCHMARKS.md).
- **Inner product,** exact search on the GPU: 4.1-17.7 times FAISS on raw GloVe-100, whose vector
  lengths vary widely. There an IVF fails (recall below 0.65, FAISS's too), and exact GPU search,
  with recall 1.0, is nearly as fast as FAISS's HNSW at recall 0.89 in batches of 100 (0.88 times) and
  faster in batches of 10,000 (1.36 times).
- **Which to use when:** see [BENCHMARKS](BENCHMARKS.md); **why siann is faster:** see
  [ARCHITECTURE](ARCHITECTURE.md).
- **LSH for dense vectors** did not work on real data.

## Documentation

| Document | What it covers |
| --- | --- |
| [ARCHITECTURE](ARCHITECTURE.md) | Why siann is faster than FAISS (and where it is not), with diagrams of how the pieces fit and which path a search takes |
| [ALGORITHMS](ALGORITHMS.md) | How exact search and IVF work on the GPU and the CPU, and what was dropped |
| [BACKENDS](BACKENDS.md) | MLX, PyTorch MPS, jax-mps, FAISS and siann's Rust code: what each offers, and their pitfalls |
| [DATASETS](DATASETS.md) | SIFT1M, GloVe-100 and Gaussian vectors: sources, preparation and quirks |
| [BENCHMARKS](BENCHMARKS.md) | Which backend to use for each workload, and the measurements behind it |
| [proto/RESULTS.md](proto/RESULTS.md) | Every measurement, with the script that produced it |
