# Backends

*Part of the siann documentation: [README](README.md), [ALGORITHMS](ALGORITHMS.md),
[ARCHITECTURE](ARCHITECTURE.md), [BACKENDS](BACKENDS.md), [DATASETS](DATASETS.md), [BENCHMARKS](BENCHMARKS.md); measurements in
[RESULTS](RESULTS.md).*

A Mac has several ways to run numerical code on its GPU and CPU. siann measures each of them on the
same searches, with the same algorithms, and checks every result against exact search. This page
describes what each backend offers and the pitfalls found along the way. The comparisons run through
`proto/backends.py`, which starts each backend in its own process.

## Summary

| Backend | Runs on | Custom kernels | Exact search, 1M x 128, 1,000 queries (L2 / L1) |
| --- | --- | --- | ---: |
| FAISS 1.16 | CPU | no (it is a library of indexes) | 0.39 s / 1.8 s |
| MLX 0.32 | GPU (Metal), CPU | yes: `mx.fast.metal_kernel` | 0.046 s / 0.081 s with siann's kernels |
| PyTorch 2.15, MPS backend | GPU (Metal), CPU | yes: `torch.mps.compile_shader` | 0.10 s / 0.082 s with siann's kernels |
| JAX 0.11 with jax-mps 0.11 | GPU (through MLX), CPU (XLA) | no | built-in operations only |
| siann's Rust code | CPU | (it is the kernel) | L1: 0.47 s |

With only each framework's built-in operations, every GPU backend is slower than FAISS on the CPU:
0.25-0.6 times its speed. On the GPU, the three frameworks' built-in operations are within 1.6 times
of each other. **The GPU's advantage comes entirely from custom kernels,** so a comparison of
backends is mostly a comparison of how well each supports them, and of the speed of its built-in
matrix multiply.

## MLX

Apple's array framework for its own chips. It evaluates lazily: operations build a graph, which runs
when a result is needed (`mx.eval`). Custom Metal kernels (Metal is Apple's GPU programming
language) are written as source strings through `mx.fast.metal_kernel`, with sizes passed as
template arguments, so each new shape compiles once.

- **The fastest matrix multiply of the three:** 6.7 ms for 256 x 1M x 128, against PyTorch's 14.2 ms.
  Fused with the norms through `mx.addmm`, 6.9 ms. This is why MLX leads on L2.
- **Evaluate each tile.** Left to the end, lazy evaluation kept every tile's distance matrix alive
  and took 196 s instead of 0.9 s for 10,000 queries.
- **`mx.compile`** (whole-function compilation) speeds up built-in L1 from 7.2 to 4.4 s, and does
  nothing for L2.
- **Its CPU backend is not competitive:** 15 times slower than FAISS.

## PyTorch (MPS)

PyTorch's Metal backend, called MPS after Apple's Metal Performance Shaders.
`torch.mps.compile_shader` (in PyTorch 2.15, the version measured) compiles Metal source into callable kernels, so siann's kernels port with
the Metal code unchanged (`proto/torch_kernels.py`).

- **The same kernel runs at the same speed as in MLX:** L1 0.082 against 0.081 s; the top-k kernel
  3.8 against 4.2 ms per tile. A custom kernel's speed belongs to the kernel, not the framework.
- **IVF too:** the GPU IVF ported the same way (`proto/ivf_torch.py`) is level with MLX's on
  batches of 10,000 and up to a third slower on single queries and small batches, a fixed cost of
  about 0.2 ms per call.
- **The matrix multiply is about half MLX's speed,** and is the whole L2 gap (0.10 against 0.046 s).
  `torch.addmm` still helps: 18.1 against 31.4 ms per tile written out.
- **`torch.cdist(p=1)` builds the whole broadcast** (queries x vectors x dimensions) and fails beyond
  2^31 elements ("tensors requiring 64-bit indexing are not supported"); siann's harness passes it 16
  queries at a time.
- **FAISS and PyTorch cannot share a process on macOS:** each loads its own copy of the OpenMP
  runtime (the library both use for CPU threads), and the second aborts ("libomp.dylib already
  initialized"). Run them in separate processes.

## JAX (jax-mps)

JAX has no official Mac GPU support. [jax-mps](https://github.com/tillahoffmann/jax-mps), a
third-party plugin, translates JAX programs into MLX operations. On the CPU, JAX uses its XLA
compiler.

- **No custom-kernel route:** Pallas, JAX's kernel language, targets CUDA and TPU, and jax-mps does
  not support it. JAX can only be compared on built-in operations.
- **Level with the other frameworks' built-in operations** (L2 0.67 s against MLX's 0.65 s), and
  faster than MLX's own on L1 when MLX runs op by op (4.6 against 7.2 s): JAX compiles the whole
  function. Under `mx.compile`, MLX catches up (4.4 s).
- **Make jitted functions once per process,** with arrays passed as arguments. Made anew on every
  call, with the base vectors captured inside, every call recompiled (0.33 s instead of 0.007 s for
  100 queries on 20,000 vectors).
- **It ran without trouble,** first on a tiny problem, then at full size. It is experimental (it
  prints a warning to that effect).

## FAISS

Meta's similarity-search library, the standard to beat. siann compares against its CPU indexes: the
flat index for exact search, IVF-Flat, and HNSW (a graph index, its strongest CPU method for
approximate search).

- **Its exact L2 is BLAS plus a top-k,** with Accelerate on a Mac, the same BLAS NumPy calls. siann's
  own BLAS path (NumPy's multiply and a Rust top-k) is 1.7-3.6 times as fast, mainly by choosing the
  tile sizes for the cache (see [ALGORITHMS](ALGORITHMS.md)).
- **Set `faiss.cvar.distance_compute_blas_threshold = 20`** for exact L2 below 128 dimensions.
  FAISS 1.16 defaults it to 128,000, so with fewer queries it skips BLAS and runs 4-7 times slower
  (GloVe-100: 2.90 s against 0.42 s). See `proto/faiss_blas.py`.
- **Its Metal GPU path is not run.** FAISS 1.16's Mac builds include a Metal backend (flat and IVF
  indexes). In one measurement it was 2.6 times slower than siann on exact L2; on a larger run it
  failed and crashed the machine, twice.
- **HNSW handles L1 well,** and is the method to beat for single queries and small batches.
- **Pad to a multiple of 8 dimensions for its compressed indexes.** Its scalar quantizer
  (`IndexIVFScalarQuantizer`, `IndexHNSWSQ`) has a fast SIMD path only for such dimensions: at 100
  (GloVe), float16 search is 6.5 times slower than at 104, and slower than float32. Zero columns leave
  L2 distances unchanged.
- **Its compressed indexes are strong.** `QT_8bit_direct` (bytes) and `QT_fp16` lists, not by residual,
  hold exactly the vectors siann's compressed lists do, and compression speeds up its HNSW 1.2-2.3
  times. For bytes, FAISS also **rounds the query to integers** and computes in integer arithmetic:
  on one thread its uint8 IVF is twice as fast as siann's Rust (and faster than its own float32), and
  exact for integer queries like SIFT's, but only approximate for others (`proto/faiss_sq8_query.py`).
- **Its IVF can split one query over threads** (`parallel_mode = 1`), but, like siann's first try on
  rayon's global pool, it pays about 100 microseconds to wake the threads.
- **Its inner product (`IndexFlatIP`) is faster than its L2** (0.29 against 0.46 s for 1,000 queries
  on GloVe), having no norms to add, so siann's CPU BLAS path is only level with it on 1,000 queries
  (2.7 times on 100); siann's GPU is 4.1-17.7 times.
- **Its L-infinity (`METRIC_Linf`) is slow:** three times its own L1 time for exact search (0.97
  against 0.33 s on 200,000 Gaussian vectors), where siann's GPU kernels take the same time for both.

## siann's Rust CPU code

A small extension in safe Rust (`proto/rust_cpu`), exposed to Python through PyO3 (the bridge that
lets Python call Rust) and threaded with rayon (Rust's data-parallelism library): exact search and IVF
for L1, L2 and L-infinity, with the blocked layout described in [ALGORITHMS](ALGORITHMS.md), and IVF
lists stored as float32, float16 or uint8. It exists for the cases where FAISS and the frameworks' CPU
backends fall short: exact L1 at 3.8 times FAISS and L-infinity at 7.3 times, and IVF for single
queries.

The code relies on the compiler to turn plain loops into NEON instructions, without `unsafe` (which
the explicit NEON intrinsics would need). That works, but the compiler's choices need checking in
the assembly, and two went wrong:

- **A maximum vectorized the wrong loop.** `f32::max` (and an integer maximum) is a reduction the
  compiler may reorder, so it vectorized across dimensions, with shuffles and a large stack frame:
  L-infinity took twice L1's time. A comparison and select fixed it.
- **Nested arrays of 16-bit values were loaded one lane at a time.** Widening float16 lists over
  `[u16; 8]` blocks gathered each value separately; the same loop over flat slices is one 16-byte
  load per 8 values.

A third pitfall is not the compiler's: **rayon's global pool is the wrong place for one query's
work.** Spreading a single query over all 16 cores wakes the pool's sleeping threads, which cost about
100 microseconds per query, more than the split saved at low nprobe. Split queries run on a small pool
of their own (rayon's `ThreadPoolBuilder`, safe), made once per size: 4 threads cut latency 1.3-2.0
times.

For exact L2, the Rust code does not multiply matrices itself: calling Accelerate from Rust would need
`unsafe` (every call into a C library is), so NumPy does the multiply and Rust only ranks the products
(`L2TopK`). It reads NumPy's arrays in place through rust-numpy (the PyO3 project's NumPy bindings),
whose safe API keeps the `unsafe` inside the crate, as pyo3's does; copying the products instead would
cost about as much as FAISS's whole search.

A fourth lesson, from the integer path for byte lists: **the layout decides what the compiler can
do.** Integer sums may be reordered, so in the blocked layout (a dimension's 8 bytes per row) plain
loops were vectorized across dimensions, and even explicit SIMD types (the `wide` crate) only reached
the float path's speed. With each vector's bytes contiguous, the same freedom works for us: a plain
loop per vector compiles to 16-byte absolute differences and the dot-product instruction (`udot`),
1.8 times faster, with no SIMD library.

Rust has no stable float16 type (it exists only in nightly builds), so float16 lists are stored as
their 16-bit patterns and widened in safe code (see [ALGORITHMS](ALGORITHMS.md), "List storage").
