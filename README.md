# siann

[![Documentation](https://img.shields.io/readthedocs/siann/latest)](https://siann.readthedocs.io)

*Pronounced like "cyan".*

**si**licon **a**pproximate **n**earest **n**eighbors: fast similarity search on Apple silicon, for
dense NumPy vectors (sparse vectors, such as TF-IDF or SPLADE, are not in scope), across metrics (L1,
L2, L-infinity, cosine, inner product), and a benchmark of how much each Mac compute backend buys for
it: on the GPU, [MLX](https://github.com/ml-explore/mlx), PyTorch's MPS backend and
[jax-mps](https://github.com/tillahoffmann/jax-mps); on the CPU, [FAISS](https://github.com/facebookresearch/faiss),
the frameworks' CPU backends and siann's own Rust code.

## How it compares

Against FAISS 1.16 on the CPU of an Apple M3 Max, with each comparison against FAISS's strongest index
for the case (its compressed indexes included), exact search returning exactly FAISS's neighbors and
approximate search compared at matched recall:

- **Exact search on the GPU,** in batches: 8 times FAISS for L2 and cosine, 18-25 times for L1, 47-49
  times for L-infinity (on random vectors) and 4-18 times for inner product.
- **Exact search on the CPU:** 3.8-4.3 times FAISS for L1 and 7.3 times for L-infinity (siann's Rust
  kernel); 1.7-3.6 times for L2 and cosine (BLAS with cache-sized tiles); level to 2.7 times for inner
  product.
- **Approximate search in large batches** (thousands of queries): the GPU IVF is 3.8 times FAISS's
  HNSW on GloVe-100, and level with it on SIFT1M at recall 0.97 (1.25 times at 0.84).
- **Batches of about 100:** level with HNSW on GloVe-100; on SIFT1M, HNSW is about twice as fast.
- **Single queries:** on GloVe-100, 1.5-1.65 times HNSW's speed on one thread and 1.5-3.1 times with 4
  threads per query; on SIFT1M, HNSW is 1.3-1.45 times faster on one thread, and siann leads from
  recall 0.97 with 4 threads.
- **Inner product on vectors of uneven length** (raw GloVe-100), where an IVF fails for FAISS and
  siann alike: exact search on the GPU, with recall 1.0, at 0.88-1.36 times the speed of FAISS's HNSW
  at recall 0.89.

[BENCHMARKS](BENCHMARKS.md) has the tables and how to rerun them, and
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
