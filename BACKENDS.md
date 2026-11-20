# Backends

*Part of the [siann documentation](README.md#documentation).*

A Mac has several ways to run numerical code on its GPU and CPU. siann measures each of them on the
same searches, with the same algorithms, and checks every result against exact search. This page
describes what each backend offers and its pitfalls; [BENCHMARKS](BENCHMARKS.md) has the full
measurements. The comparisons run through `proto/backends.py`, which starts each backend in its own
process.

## Summary

| Backend | Runs on | Custom kernels | Exact search, 1M x 128, 1,000 queries (L2 / L1) |
| --- | --- | --- | ---: |
| FAISS 1.16 | CPU | no (it is a library of indexes) | 0.39 s / 1.8 s |
| MLX 0.32 | GPU (Metal), CPU | yes: `mx.fast.metal_kernel` | 0.046 s / 0.081 s with siann's kernels |
| PyTorch 2.15, MPS backend | GPU (Metal), CPU | yes: `torch.mps.compile_shader` | 0.10 s / 0.082 s with siann's kernels |
| JAX 0.11 with jax-mps 0.11 | GPU (through MLX), CPU (XLA) | no | built-in operations only |
| siann's Rust code | CPU | (it is the kernel) | L1: 0.47 s |

With only each framework's built-in operations, every GPU backend is slower than FAISS on the CPU:
0.25-0.6 times its speed, the three frameworks within 1.6 times of each other. **The GPU's advantage
comes entirely from custom kernels,** so a comparison of backends is mostly a comparison of how well
each supports them, and of the speed of its built-in matrix multiply.

## MLX

Apple's array framework for its own chips, and siann's default GPU backend. It evaluates lazily:
operations build a graph, which runs when a result is needed (`mx.eval`). Custom Metal kernels (Metal
is Apple's GPU programming language) are written as source strings through `mx.fast.metal_kernel`,
with sizes passed as template arguments, so each new shape compiles once.

- **The fastest matrix multiply of the three:** 6.7 ms for 256 x 1M x 128, against PyTorch's 14.2 ms;
  fused with the norms through `mx.addmm`, 6.9 ms. This is why MLX leads on L2.
- **Evaluate each tile.** Evaluated only at the end, a search keeps every tile's distance matrix alive:
  196 s instead of 0.9 s for 10,000 queries.
- **`argpartition` suits choosing a few dozen of a few thousand per row:** 0.30 ms for 100 rows of
  4,000 at k = 64, against 1.0 ms for siann's two-stage selection kernel, whose running top-k per thread
  does not fit in registers at that k. For the top 10 of a million, siann's kernel is the one to use
  (4.1 against 146-152 ms per tile).
- **`mx.compile`** (whole-function compilation) speeds up built-in L1 from 7.2 to 4.4 s, and does
  nothing for L2.
- **Its CPU backend is not competitive:** 15 times slower than FAISS.

## PyTorch (MPS)

PyTorch's Metal backend, called MPS after Apple's Metal Performance Shaders.
`torch.mps.compile_shader` (in PyTorch 2.15) compiles Metal source into callable kernels, so siann's
kernels port with the Metal code unchanged (`proto/torch_kernels.py`, `proto/ivf_torch.py`).

- **The same kernel runs at the same speed as in MLX:** L1 0.082 against 0.081 s; the top-k kernel 3.8
  against 4.2 ms per tile. A custom kernel's speed belongs to the kernel, not the framework.
- **The GPU IVF too** returns the same neighbors as MLX's and runs level with it on batches of 10,000.
  PyTorch's fixed cost of about 0.2 ms per call makes it 0.84-0.97 times MLX's speed on batches of 100
  and 0.50-0.88 times on single queries.
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
  faster than MLX's own on L1 when MLX runs op by op (4.6 against 7.2 s), since JAX compiles the whole
  function. Under `mx.compile`, MLX catches up (4.4 s).
- **Make jitted functions once per process,** with arrays passed as arguments. Made anew on every
  call, with the base vectors captured inside, every call recompiles (0.33 s instead of 0.007 s for
  100 queries on 20,000 vectors).
- **It runs without trouble at full size,** though it is experimental (it prints a warning to that
  effect).

## FAISS

Meta's similarity-search library, the standard to beat. siann compares against its CPU indexes: the
flat index for exact search, IVF-Flat and IVF with scalar quantization, and HNSW (a graph index, its
strongest CPU method for approximate search), plain and compressed.

- **Its exact L2 is BLAS plus a top-k,** with Accelerate on a Mac, the same BLAS NumPy calls. siann's
  own BLAS path (NumPy's multiply and a Rust top-k) is 1.7-3.6 times as fast, mainly by choosing the
  tile sizes for the cache.
- **Set `faiss.cvar.distance_compute_blas_threshold = 20`** for exact L2 below 128 dimensions.
  FAISS 1.16 defaults it to 128,000, so with fewer queries it skips BLAS and runs 4-7 times slower
  (GloVe-100: 2.90 s against 0.42 s).
- **Do not run its Metal GPU path.** FAISS 1.16's Mac builds include a Metal backend (flat and IVF
  indexes). On exact L2 with 1,000 queries it is 2.6 times slower than siann; with 10,000 it fails and
  crashes the machine.
- **HNSW handles L1 well,** and is the method to beat for single queries and small batches.
- **Pad to a multiple of 8 dimensions for its compressed indexes.** Its scalar quantizer
  (`IndexIVFScalarQuantizer`, `IndexHNSWSQ`) has a fast SIMD path only for such dimensions: at 100
  (GloVe), float16 search is 6.5 times slower than at 104, and slower than float32. Zero columns leave
  L2 distances unchanged.
- **Its compressed indexes are strong.** `QT_8bit_direct` (bytes) and `QT_fp16` lists, not by residual,
  hold exactly the vectors siann's compressed lists do, and compression speeds up its HNSW 1.6-2.3
  times on SIFT. For bytes, FAISS also **rounds the query to integers** and computes in integer
  arithmetic: exact for integer queries like SIFT's, approximate for others.
- **Its IVF can split one query over threads** (`parallel_mode = 1`), but it pays about 100
  microseconds to wake the threads, as any split on a shared pool does.
- **Its inner product (`IndexFlatIP`) is faster than its L2** (0.29 against 0.46 s for 1,000 queries
  on GloVe), having no norms to add.
- **Its L-infinity (`METRIC_Linf`) is slow:** three times its own L1 time for exact search (0.97
  against 0.33 s on 200,000 Gaussian vectors).

## siann's Rust CPU code

A small extension in safe Rust (`proto/rust_cpu`), exposed to Python through PyO3 (the bridge that
lets Python call Rust) and threaded with rayon (Rust's data-parallelism library): exact search and IVF
for L1, L2, L-infinity and inner product, with the blocked layout described in
[ALGORITHMS](ALGORITHMS.md), IVF lists stored as float32, float16 or uint8, and `L2TopK`, which ranks
NumPy's BLAS products for exact L2. It exists for the cases where FAISS and the frameworks' CPU
backends fall short: exact L1 at 3.8-4.3 times FAISS, L-infinity at 7.3 times, and IVF for single
queries.

The code relies on the compiler to turn plain loops into NEON instructions, without `unsafe` (which
explicit NEON intrinsics would need); the rules that keep it fast are in
[ARCHITECTURE](ARCHITECTURE.md#rules-for-changing-the-code). Rust has no stable float16 type (it exists
only in nightly builds), so float16 lists are stored as their 16-bit patterns and widened in safe code
(see [ALGORITHMS](ALGORITHMS.md), "List storage").
