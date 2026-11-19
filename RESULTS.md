# Results

*Part of the siann documentation: [README](README.md), [ALGORITHMS](ALGORITHMS.md),
[ARCHITECTURE](ARCHITECTURE.md), [BACKENDS](BACKENDS.md), [DATASETS](DATASETS.md),
[BENCHMARKS](BENCHMARKS.md).*

Every measurement behind siann's documentation, in the order it was made, with the script that
produced it and what it showed, including the ideas that were tried and dropped. The scripts named
below are in [proto/](proto/) and run from there (`uv run --no-sync --group proto python <script>`);
their raw output is in [proto/results/](proto/results/).

# First measurements: the GPU against FAISS

M3 Max (16 cores, 400 GB/s), MLX 0.32.4, FAISS 1.16 (faiss-cpu, 16 threads). Synthetic Gaussian
data, 1,000,000 base vectors of 128 dimensions, k = 10. Every MLX result equals FAISS's.

## Exact search (`exact.py`)

| Case | MLX | FAISS flat | Speed-up | Gate |
| --- | ---: | ---: | ---: | --- |
| L2, 1,000 queries | 0.15 s | 0.40 s | 2.6× | 2×: met |
| L2, 10,000 queries | 0.88 s | 3.32 s | 3.8× | 2×: met |
| L1, 1,000 queries | 0.22 s | 1.86 s | 8.3× | 5×: met |
| L1, 10,000 queries | 0.83 s | 17.43 s | 21× | 5×: met |

- **Top-k selection was the bottleneck,** not distances. For 256 queries × 1M, distances (a matrix
  multiply) take 18 ms; MLX's `argpartition` and `topk` take 146 and 152 ms. A two-stage selection
  kernel (`knn_kernels.select_k`: each of 512 threads per row keeps its best k in registers, then a
  small argpartition) takes 4.1 ms.
- **Lazy evaluation must be bounded:** evaluating only at the end kept every tile's 1 GB distance
  matrix alive (196 s for 10,000 queries). Evaluate each tile.
- **L1 had to be tiled.** The first, fused kernel (`l1_knn`: distance and selection in one pass)
  streamed all base vectors once per query, about 512 GB for 1,000 queries, so it was memory-bound
  at 2.3-2.7x. The tiled kernel (`l1_dist`, like a matrix multiply: 16 x 16 threads compute 64 x 64
  distances, 4 x 4 per thread in registers, through threadgroup memory 16 dimensions at a time)
  reads each base vector once per 64 queries; with `select_k` after it, L1 is as fast as L2.


## Approximate L1 on SIFT1M (`lsh_l1.py`, `faiss_l1.py`)

SIFT1M from ann-benchmarks (1,000,000 vectors, 128 dimensions); ground truth from the GPU's exact L1.

| Method | Recall@10 | Queries/s | Notes |
| --- | ---: | ---: | --- |
| Cauchy LSH, w = 2 r10, m = 4, L = 32 | 0.72 | — | re-ranks 10.5% of the database |
| FAISS HNSW (M 32), efSearch 64 | 0.947 | 56,000 | built in 12 s |
| FAISS HNSW, efSearch 256 | 0.990 | 17,000 | |
| FAISS IVF (4,000 lists), nprobe 64 | 0.958 | 5,800 | built in 19 s |
| Exact L1 on the GPU (`l1_knn_tiled`) | 1.000 | about 12,000 | batches of 10,000 queries |

- **LSH is out.** Real data has low contrast: at w = r10 (the distance to the 10th neighbour), a
  Cauchy hash collides with probability about 0.28 for a neighbour and about 0.26 for a point twice as
  far, so separation needs ever more tables and candidates. Reaching recall 0.72 re-ranks a tenth of
  the database.
- **FAISS's HNSW handles L1 well,** contrary to the plan's assumption: approximate L1 on the CPU is
  already fast. Exact GPU search wins where results must be exact or there is no index to build.

## IVF on the GPU, L1 on SIFT1M (`ivf_gpu.py`)

4,000 lists (FAISS k-means, L1 assignment on the GPU), batches of 1,000 queries; built in 14 s.

| nprobe | Recall@10 | GPU IVF queries/s | FAISS HNSW at similar recall |
| ---: | ---: | ---: | ---: |
| 16 | 0.784 | 107,000 | 157,000 (0.780) |
| 64 | 0.951 | 31,000 | 56,000 (0.947) |

FAISS's CPU IVF at nprobe 64: 0.958 at 5,800 queries/s, so the GPU gives 5x there.

- **The coarse selection needs few threads per row:** with k = nprobe = 64, 512 threads leave
  32,768 survivors per query; 32 threads cut it from 14 ms to about 1 ms.
- **Batch size must not be a compile-time constant:** each new batch size recompiled the kernels
  inside the timed call. Pass sizes at run time.
- **The scan is memory-bound:** query by query, each reads its 64 lists (about 8 MB), about 270 GB/s
  overall. This led to the list-major scan below: each list loaded once into threadgroup memory and
  compared with all the queries that probe it (about 16 per list at 1,000 queries, 160 at 10,000).

### List-major scan

Each list is loaded once, through threadgroup memory, and compared with all the queries probing it,
16 at a time; one threadgroup per list; queries grouped by list on the GPU (argsort and counts);
the final merge through `select_k`.

| Recall@10, batches of 10,000 | GPU IVF, list-major | FAISS HNSW (CPU) | Ratio |
| --- | ---: | ---: | ---: |
| about 0.80 (nprobe 16) | 326,000 q/s | 157,000 q/s (0.780) | 2.1× |
| about 0.90 (nprobe 32) | 152,000 q/s | 98,000 q/s (0.880) | 1.6× |
| about 0.955 (nprobe 64) | 103,000 q/s | 56,000 q/s (0.947) | 1.8× |
| about 0.98 (nprobe 128) | 37,700 q/s | 31,600 q/s (0.977) | 1.2× |

With batches of 1,000 (about 16 queries per list instead of 160) the list-major scan is level with
HNSW: 59,000 queries/s at 0.950.

Where the time went at 10,000 queries and nprobe 64, before the last two fixes: coarse distances
5.5 ms, coarse selection 24.8 ms, grouping on the CPU 51 ms (now on the GPU), scan 44.7 ms,
merge with `argpartition` 29.9 ms (3.4 ms with `select_k`). Remaining headroom: the coarse top-64
selection (a per-thread top-64 does not fit in registers; sort within a threadgroup instead) and the
scan itself, which runs well below the GPU's peak.

## Does it generalize? SIFT1M with L2, GloVe-100 with cosine (`ivf_gpu.py`, `faiss_ann.py`)

Batches of 10,000 queries; GloVe normalized, so L2 ranks like cosine. GPU IVF built in 14 s, HNSW in
10-14 s.

**SIFT1M, L2:**

| Recall@10 | GPU IVF, list-major | FAISS HNSW | Ratio |
| --- | ---: | ---: | ---: |
| about 0.85 | 475,000 q/s (0.839) | 208,000 q/s (0.861) | 2.3× |
| about 0.93 | 196,000 q/s (0.924) | 125,000 q/s (0.936) | 1.6× |
| about 0.975 | 139,000 q/s (0.972) | 72,000 q/s (0.977) | 1.9× |
| 0.992 | 45,000 q/s | 40,000 q/s | 1.1× |

FAISS's CPU IVF at nprobe 64: 0.972 at 13,000 queries/s.

**GloVe-100, cosine:**

| Recall@10 | GPU IVF, list-major | FAISS HNSW | Ratio |
| --- | ---: | ---: | ---: |
| about 0.70 | 487,000 q/s (0.711) | 100,000 q/s (0.699) | 4.9× |
| about 0.78 | 201,000 q/s (0.780) | 58,000 q/s (0.781) | 3.5× |
| about 0.84 | 146,000 q/s (0.839) | 33,000 q/s (0.845) | 4.4× |
| about 0.89 | 46,000 q/s (0.890) | 18,000 q/s (0.892) | 2.6× |

The advantage is largest on the hard dataset: HNSW walks much further through its graph on GloVe,
while IVF's dense scans suit the GPU. It narrows only at recall 0.99 on SIFT.

## Summary of the first measurements

- **Exact search on the GPU:** L2 2.6-3.8x and L1 8-21x FAISS's flat index, identical results.
- **IVF on the GPU, in batches:** 1.1-2.3x FAISS's HNSW on SIFT (L1 and L2) and 2.6-4.9x on GloVe,
  about 10x FAISS's CPU IVF. Level with HNSW at batches of 1,000; HNSW still wins single queries.
- **LSH:** not viable on real data.
- **Headroom:** the coarse top-nprobe selection (about a quarter of the time) and the scan kernel.

## FAISS's Metal GPU path (`faiss_metal.py`, now disabled)

faiss-cpu 1.16 on macOS ships a Metal backend (`gpu_metal`: flat, IVF-Flat, IVF-PQ and binary flat
indexes, reached through `index_cpu_to_gpu`), so the comparisons above are against FAISS's
CPU methods, not its GPU path. One measurement before it failed:

| Case | FAISS on Metal | siann prototype | Ratio |
| --- | ---: | ---: | ---: |
| Exact L2, 1,000 queries, 1M x 128 | 0.222 s | 0.085 s | 2.6x |

With 10,000 queries, FAISS's Metal search failed ("MetalIndexFlat: GPU search failed (pipeline or
dispatch error)"), and the run then crashed the machine, twice. The script now exits immediately.
Its GPU IVF path was never measured. Further comparisons with it should rely on published numbers.

# Every backend

## Exact search on every backend (`backends.py`)

Each backend runs in its own process, is warmed up on the full query set (so that every tile shape
is compiled), then timed as the median of three runs. 1,000,000 x 128 Gaussian vectors, 1,000
queries, k = 10, 16 CPU threads; mlx 0.32.4, torch 2.15.0, jax 0.11.2 with jax-mps 0.11.1, FAISS
1.16. Every backend returns FAISS's neighbors exactly (agreement 1.0000).

- **Custom kernels:** siann's kernels (`knn_kernels.py`), and the same Metal code ported to PyTorch
  through `torch.mps.compile_shader` (`torch_kernels.py`). L2 is a fused `addmm` (norms minus twice
  the matrix multiply), then the two-stage top-k kernel; L1 is the tiled distance kernel, then the
  same top-k.
- **Built-in operations:** what a user of each framework would write. L2 as norms minus twice a
  matrix multiply, L1 as a broadcast absolute difference (MLX, JAX) or `torch.cdist` (PyTorch),
  then the framework's own top-k. JAX's functions are jitted, as JAX code always is; "MLX compiled"
  is MLX's under `mx.compile`.

| Backend | L2 | vs FAISS | L1 | vs FAISS |
| --- | ---: | ---: | ---: | ---: |
| FAISS flat (CPU) | 0.39 s | 1.00x | 1.82 s | 1.00x |
| MLX, custom kernels | 0.046 s | 8.4x | 0.082 s | 22x |
| PyTorch MPS, custom kernels | 0.101 s | 3.9x | 0.082 s | 22x |
| MLX built-in (GPU) | 0.65 s | 0.60x | 7.20 s | 0.25x |
| MLX compiled (GPU) | 0.63 s | 0.61x | 4.37 s | 0.42x |
| PyTorch built-in (MPS) | 0.70 s | 0.55x | 4.72 s | 0.39x |
| JAX built-in (jax-mps) | 0.67 s | 0.58x | 4.56 s | 0.40x |
| PyTorch built-in (CPU) | 1.10 s | 0.35x | 4.52 s | 0.40x |
| JAX built-in (XLA, CPU) | 1.34 s | 0.29x | 3.95 s | 0.46x |
| MLX built-in (CPU) | 6.8 s | 0.06x | 32.5 s | 0.06x |

With 10,000 queries: L2 FAISS 3.23 s, MLX custom 0.42 s (7.6x), PyTorch custom 0.96 s (3.4x, before
MLX's `addmm`); L1 FAISS 17.2 s, MLX and PyTorch custom 0.81 s (21x). The MLX CPU row is from a single
run without the full warm-up.

**Findings** (the per-tile times are from `profile_l2.py`; correctness of every kernel against NumPy
from `check_kernels.py`):

- **The same Metal kernel runs at the same speed in MLX and PyTorch.** L1 is identical (0.082 s),
  and the top-k kernel takes 4.2 against 3.8 ms per tile of 256 x 1M. A custom kernel's speed
  belongs to the kernel, not the framework.
- **The frameworks differ in their matrix multiply.** For 256 x 1M x 128, MLX's takes 6.7 ms and
  PyTorch's 14.2; fused with the norms through `addmm`, 6.9 and 18.1 ms; written as
  `norms - 2 * (q @ x.T)`, 17.7 and 31.4 ms. (A first, one-off measurement gave MLX's multiply alone
  as 9.9 ms; `profile_l2.py` measures 6.7 ms.) That is the whole L2 gap
  between the custom backends (MLX 8.4x, PyTorch 3.9x).
- **`addmm` doubled siann's L2:** written as `norms - 2 * (q @ x.T)`, the custom MLX L2 took 0.087 s
  (4.4x FAISS); the first measurement's 2.6x also had compilation in its timing (warm-up on 16 queries only).
- **Built-in operations lose to FAISS's CPU everywhere,** on the GPU too: 0.25-0.6x. On the GPU the
  three frameworks are level (L2 0.63-0.70 s; L1 4.4-4.7 s, except MLX uncompiled at 7.2 s). The
  GPU's advantage comes entirely from custom kernels.
- **`mx.compile` helps MLX's L1 (7.2 to 4.4 s) but not its L2.**
- **jax-mps translates JAX programs into MLX operations** and has no custom-kernel route: Pallas,
  JAX's kernel language, targets CUDA and TPU only. It ran without trouble, and level with the
  other frameworks' built-in operations.
- **On the CPU, FAISS wins:** JAX's XLA and PyTorch reach 0.3-0.46x, MLX 0.06x.
- **`torch.cdist(p=1)` on MPS builds the whole broadcast** and fails beyond 2^31 elements ("tensors
  requiring 64-bit indexing are not supported"), so queries go 16 at a time.
- **FAISS and PyTorch cannot share a process on macOS:** each links its own OpenMP runtime
  ("libomp.dylib already initialized"). Hence one process per backend.
- **JAX needs its jitted functions made once per process,** with the arrays as arguments: made per
  call, with the base vectors captured, every call recompiled (0.33 s instead of 0.007 s for 100
  queries on 20,000 vectors).
- **At 200,000 vectors the GPU barely pays** (first, single-run measurement): FAISS L2 0.079 s
  against custom MLX 0.099 s before `addmm`; L1 0.300 against 0.106 s.
- **CPU headroom for L1:** FAISS's L1 does 128 G absolute differences in about 1.8 s, some 70 G per
  second. Twelve performance cores with NEON (Apple's vector instructions) should manage several
  times that, so a compiled L1 kernel could beat FAISS on the CPU (it does: see below). L2 is a
  matrix multiply, where FAISS uses BLAS (the tuned linear-algebra library) and is hard to beat.
- `ivf_gpu.py` got the same change for its coarse and fine L2 distances (below, "IVF with `addmm`").

## L1 on the CPU in Rust (`rust_cpu/`, first named `cpu_l1/`)

A PyO3 extension (PyO3 exposes Rust to Python) in safe Rust, with rayon for threads. The base
vectors are stored in blocks of 8, dimension by dimension; the inner loop computes 4 queries
against 8 vectors, 32 independent running sums, which the compiler turns into 8 NEON
absolute-difference instructions (`fabd.4s`) and 8 additions per dimension, with no scalar
fallback. Each thread scans the base in tiles of 1,024 vectors (512 KB) for its queries, and keeps
each query's best k in a small sorted list. Padding vectors hold infinity. Build:
`cd proto/rust_cpu && VIRTUAL_ENV=../../.venv uvx maturin develop --release --uv`.

1,000,000 x 128, 1,000 queries, k = 10, 16 threads (median of three, `backends.py`; the sweep
and build time from `rust_cpu_sweep.py`, correctness from `check_kernels.py`):

| Backend | L1 | vs FAISS |
| --- | ---: | ---: |
| FAISS flat (CPU) | 1.79 s | 1.00x |
| siann, Rust (CPU) | 0.471 s | 3.8x |
| siann, MLX custom kernels (GPU) | 0.081 s | 22x |

- **3.8x FAISS on the first attempt,** identical results (exact match with NumPy on small cases,
  including 7, 37 and 100 dimensions and k from 1 to 32). About 270 G absolute differences per
  second. `rust_cpu_sweep.py` counts the instructions in the built extension: 8 `fabd.4s`, 8
  `fadd.4s`, no scalar `fabd`.
- **Queries per thread task make no measurable difference:** 8: 0.528 s, 16: 0.492, 32: 0.470 and
  0.542 in two runs, 64: 0.471 and 0.481. The spread between runs is as large as between sizes.
  Default 32.
- **Index build** (the transposition into blocks): 0.10-0.13 s for 1M x 128.
- **Untried:** other block shapes (queries x vectors), tile sizes, and splitting the base rather
  than the queries across threads for small batches.

## Real data: SIFT1M and GloVe-100 (`backends.py --dataset`)

1,000 queries, k = 10, median of three. Agreement now counts a neighbor as correct when it is no
farther than FAISS's k-th nearest, so that ties broken differently (SIFT's coordinates are
integers, so its L1 distances often tie) are not counted as errors. Every backend scores 1.0000.

| Backend | SIFT L2 | SIFT L1 | GloVe L2 | GloVe L1 |
| --- | ---: | ---: | ---: | ---: |
| FAISS flat (CPU) | 0.405 s | 2.07 s | 0.431 s | 1.65 s |
| MLX, custom kernels | 8.7x | 25x | 8.2x | 18.5x |
| PyTorch MPS, custom kernels | 4.0x | 25x | 3.9x | 18.4x |
| siann, Rust (CPU) | | 4.3x | | 3.9x |
| MLX compiled (GPU) | 0.65x | 0.48x | 0.54x | 0.34x |
| PyTorch built-in (MPS) | 0.59x | 0.44x | | |
| JAX built-in (jax-mps) | 0.61x | 0.47x | | |
| PyTorch built-in (CPU) | 0.35x | 0.46x | | |
| JAX built-in (XLA, CPU) | 0.32x | 0.57x | | |

The ratios match the Gaussian data's: the rankings hold on real data. GloVe-100 has 100 dimensions,
not a multiple of the kernels' 16-dimension tiles, and the kernels handle it. (The empty GloVe cells
were measured only against the misconfigured FAISS below: built-in operations there were 0.39-0.50x
on L1, as on SIFT.)

### FAISS's BLAS threshold

FAISS 1.16 sets `faiss.cvar.distance_compute_blas_threshold` to 128,000. With fewer queries than
that, flat L2 search below 128 dimensions skips BLAS (the tuned matrix-multiply library) for a path
4-7x slower (`faiss_blas.py`):

| Dimensions (500,000 Gaussian vectors, 1,000 queries) | 32 | 64 | 96 | 112 | 127 | 128 | 192 | 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FAISS flat L2, default | 0.184 s | 0.453 | 0.803 | 1.002 | 1.209 | 0.177 | 0.200 | 0.230 |
| Threshold 20 | 0.156 s | 0.185 | 0.159 | 0.166 | 0.172 | 0.170 | 0.205 | 0.232 |

On GloVe-100, L2 takes 2.90 s by default and 0.419 s with the threshold at 20 (FAISS's earlier
default), with the same neighbors; L1 and 128 dimensions are unaffected. The first GloVe run
therefore showed our L2 at 55x FAISS; against FAISS at its best it is 8.2x. `backends.py` and
`faiss_ann.py` now set the threshold to 20. The earlier comparisons are unaffected: HNSW does not use
flat search, and FAISS's IVF was measured only on SIFT (128 dimensions).

## IVF with `addmm` (`ivf_gpu.py`)

The coarse distances (queries against the 4,000 centroids) now use `addmm`. SIFT1M, L2, batches of
10,000, list-major scan: 524,000 queries/s at recall 0.839 (before `addmm`: 475,000), 200,000 at 0.924
(196,000), 139,000 at 0.972 (139,000), 44,000 at 0.992 (45,000). The coarse step is small next to
the scan, so the gain is 10% at low nprobe and nothing above.

## IVF on the CPU in Rust, against FAISS and the GPU (`ivf_cpu.py`)

The Rust extension (`rust_cpu/`) now has IVF for L1 and L2, with two scans: query-major (one thread
per query, each reading its own lists) and list-major (queries grouped by list, each list scanned
once for all of its queries, 4 at a time, with the exact kernel). Rust, FAISS and GPU IVF share the
same 4,000 k-means centroids, so they differ only in how they search; their recall at each nprobe is
identical. Throughput in queries per second at three batch sizes: single queries (one call per query,
1,000 queries), batches of 100 (over 1,000 queries) and one batch of 10,000. Recall counts ties as
correct (see "Real data" above). The GPU IVF uses the query-major scan below 1,000 queries and the
list-major scan above. Raw output: `results/ivf_sift_l2.txt`, `results/ivf_sift_l1.txt`,
`results/ivf_glove_l2.txt` as of commit 0d671bf; the files now hold a rerun with PyTorch rows added
(below), whose other rows agree with these within a few percent.

**At matched recall,** each IVF setting against the HNSW setting nearest in recall:

| Dataset | Recall: IVF / HNSW | Single queries: best IVF (Rust) / HNSW | Batches of 100: best / HNSW | 10,000: GPU IVF / CPU list-major / HNSW |
| --- | --- | ---: | ---: | ---: |
| SIFT L2 | 0.925 / 0.937 | 7,789 / 11,462 | 33,079 (GPU) / 95,958 | 201,430 / 70,032 / 123,480 |
| SIFT L2 | 0.973 / 0.978 | 4,602 / 5,670 | 25,178 (GPU) / 55,302 | 140,578 / 39,803 / 71,251 |
| SIFT L2 | 0.993 / 0.993 | 2,520 / 3,524 | 12,783 (GPU) / 34,420 | 44,781 / 21,172 / 39,769 |
| SIFT L1 | 0.904 / 0.898 | 7,526 / 7,695 | 26,680 (GPU) / 79,938 | 152,908 / 73,104 / 100,721 |
| SIFT L1 | 0.960 / 0.957 | 4,400 / 4,806 | 19,698 (GPU) / 49,241 | 105,076 / 40,361 / 58,418 |
| SIFT L1 | 0.988 / 0.985 | 2,508 / 2,622 | 12,497 (CPU list-major) / 28,192 | 38,662 / 22,540 / 32,466 |
| GloVe | 0.780 / 0.781 | 8,993 / 5,397 | 34,385 (GPU) / 47,500 | 207,448 / 77,076 / 57,226 |
| GloVe | 0.839 / 0.845 | 5,436 / 2,837 | 27,099 (GPU) / 28,157 | 147,041 / 42,412 / 32,367 |
| GloVe | 0.890 / 0.892 | 3,077 / 1,591 | 13,819 (GPU) / 16,022 | 46,537 / 23,164 / 17,778 |

**Findings:**

- **Single queries:** Rust's query-major IVF beats HNSW on GloVe (1.7-1.9 times), is level on SIFT
  L1 (0.92-0.98 times) and trails on SIFT L2 (0.68-0.81 times). It is 1.4-6.6 times FAISS's IVF with
  the same lists, most at low nprobe, where FAISS's fixed cost per query dominates (SIFT L2 at
  nprobe 4: 3,728 against 20,474 queries/s; at nprobe 128: 1,222 against 2,520). Each single query runs on one thread, in both.
- **Batches of 100 (or many single queries at once): HNSW wins everywhere,** 1.0-3.0 times the best
  IVF. The query-major scan barely scales with threads: at nprobe 16 and above, 16 threads give
  2.3-3.7 times one thread's rate (GloVe nprobe 64: 5,436 alone, 14,958 in batches of 100), because
  every query reads its own lists from memory, about 7.6 MB each at nprobe 64 on GloVe, some 115 GB/s
  in all. HNSW reads far
  less per query.
- **Batches of 10,000: the GPU wins,** 1.1-2 times HNSW on SIFT and 2.6-4.5 times on GloVe. The CPU
  list-major scan is 2.3-4.1 times the query-major scan and FAISS's IVF there at nprobe 16 and above
  (1.2-1.8 times at nprobe 4), since each list is read once per batch; beats HNSW on GloVe (1.3 times), and trails it on SIFT (0.5-0.7 times).
- **The query-major scan is the right choice below some batch size, list-major above.** On the CPU,
  list-major is slower for single queries (it pays for grouping), wins in batches of 100 from
  nprobe 64 (from 32 on SIFT L1), and always wins at 10,000.
- **FAISS's IVF is level with Rust's query-major scan on batches from nprobe 16** (both are limited
  by memory; at nprobe 4, Rust is 1.3-1.5 times faster) and well behind it on single queries.

**What this means for the recommendation** (first draft; the final version, from the rerun's
numbers, is in [BENCHMARKS](BENCHMARKS.md)):

| Workload | Best choice measured |
| --- | --- |
| Exact search, any batch | GPU, siann's custom kernels (MLX; PyTorch on L1) |
| Exact search on the CPU | L1: siann's Rust kernel (3.8x FAISS); L2: FAISS flat, with the BLAS threshold at 20 |
| Approximate, one query at a time | GloVe-like data: Rust IVF; SIFT-like L1: Rust IVF or HNSW; SIFT-like L2: HNSW |
| Approximate, small batches or concurrent queries | FAISS HNSW (CPU) |
| Approximate, large batches (10,000) | GPU IVF, list-major |

The crossover between HNSW and the GPU lies between 100 and 10,000 queries per batch; the first IVF
measurements found them level at 1,000 on SIFT L1.

Measured since, below: splitting a single query over threads ("Single queries split over threads"),
float16 lists ("float16 lists") and the PyTorch port of the GPU list scan ("IVF on PyTorch MPS").

## IVF on PyTorch MPS (`ivf_torch.py`)

The GPU IVF ported to PyTorch through `torch.mps.compile_shader`: the same Metal code for both scans,
the same coarse step and merge, and the same centroids (from `ivf_cpu.py`, which runs it in a separate
process, since FAISS and PyTorch cannot share one). Checked against exact search in
`check_kernels.py`. Raw output: the PyTorch rows of `results/ivf_*.txt`.

| SIFT L2, nprobe | Recall (both) | Single: MLX / PyTorch | Batches of 100 | Batch of 10,000 |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.571 | 2,356 / 1,541 | 155,380 / 110,696 | 1,184,466 / 1,149,359 |
| 16 | 0.840 | 1,287 / 936 | 75,859 / 62,148 | 517,798 / 512,256 |
| 32 | 0.925 | 490 / 442 | 33,313 / 30,719 | 199,112 / 201,188 |
| 64 | 0.973 | 455 / 416 | 25,193 / 23,737 | 141,078 / 143,308 |
| 128 | 0.993 | 220 / 214 | 12,783 / 12,680 | 44,501 / 45,591 |

- **Recall is identical at every nprobe** on all three datasets, as the shared lists require.
- **Large batches are level** (within 3% on SIFT L2, SIFT L1 and GloVe): the kernels do the work,
  and run at the same speed in both frameworks, as in exact search.
- **Single queries and batches of 100 are up to a third slower in PyTorch** at low nprobe, and level
  at high nprobe: a fixed cost per call of the order of 0.2 ms (SIFT L2, nprobe 4: 1,541 against
  2,356 single queries per second).
- **For the recommendation:** for large batches, MLX and PyTorch are interchangeable for IVF; MLX
  keeps its edge on exact L2 (its matrix multiply) and on small calls.

## L-infinity on every backend (`backends.py --metrics l1,linf`)

L-infinity is the largest coordinate difference. It shares L1's kernels: each distance is built one
coordinate difference at a time, as a running maximum of absolute differences where L1 keeps a sum
(`COMBINE` in `knn_kernels.py`, the `Metric` trait in `rust_cpu`). FAISS has it as `METRIC_Linf`.
200,000 Gaussian vectors, 128 dimensions, 1,000 queries, k = 10; raw output in
`proto/results/linf_gaussian.txt`. Every backend agrees with FAISS (agreement 1.0000).

| Backend | L1 | L-infinity |
| --- | ---: | ---: |
| FAISS flat (CPU), time | 0.334 s | 0.972 s |
| siann Rust (CPU) | 3.5x FAISS (0.095 s) | 7.3x FAISS (0.133 s) |
| siann MLX custom kernels (GPU) | 17.4x | 49.0x |
| siann PyTorch MPS custom kernels | 16.6x | 47.2x |

- **The GPU kernels take the same time for L-infinity as for L1** (0.019-0.021 s); FAISS takes three
  times as long for L-infinity as for its own L1, so siann's lead grows to 47-49x.
- **A maximum is exact:** no rounding, unlike a sum, so the tiled kernels' L-infinity distances match
  NumPy's to the bit (`check_kernels.py` requires it).
- **Rust: `f32::max` vectorized the wrong loop.** Written with `f32::max`, the Rust kernel took 0.209 s,
  twice its L1 time. LLVM treats a maximum as a reduction it may reorder, and vectorized across
  dimensions instead of keeping one running maximum per (query, vector) lane, with shuffles and a
  3.7 KB stack frame; an unsigned-integer maximum on the float bits did the same. A plain comparison
  and select keeps the intended layout (two instructions, `fcmgt` and `bit`, where L1 has one `fadd`):
  0.133 s, 1.4 times L1's time.
- **L1 and L2 did not change:** every L1 and L2 output, distances included, was bit-identical before and
  after the change (A/B on fixed seeds). `check_kernels.py` covers L-infinity on every backend and now
  also checks MLX's IVF (`ivf_gpu.py --check`) in its own process.
- **Measured on random vectors only,** not on SIFT1M or GloVe-100.

## float16 lists (`fp16_lists.py`)

The IVF's lists stored as float16 instead of float32: half the bytes per vector, with distances still
computed in float32 from the rounded vectors. Queries stay float32. The same 4,000 lists (GpuIVF's
k-means centroids) for both storages; L2, k = 10; recall at 10 against exact search on the original
float32 vectors. Queries per second; Rust uses its query-major scan for single queries (one thread) and
batches of 100 (16 threads) and its list-major scan for one batch of 10,000; the GPU (MLX) its
query-major scan for batches of 100 and list-major for 10,000. Raw output in
`proto/results/fp16_sift_l2.txt` and `fp16_glove_l2.txt`.

| Dataset, nprobe | Recall: float32 / float16 | Rust, single | Rust, 100 | Rust, 10,000 | GPU, 100 | GPU, 10,000 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SIFT, 16 | 0.840 / 0.840 | 0.62x | **1.61x** | 0.91x | **1.43x** | 1.07x |
| SIFT, 64 | 0.973 / 0.973 | 0.52x | **1.73x** | 0.87x | **1.60x** | 1.04x |
| GloVe, 16 | 0.712 / 0.711 | 0.63x | **1.58x** | 0.91x | **1.29x** | 1.04x |
| GloVe, 64 | 0.839 / 0.839 | 0.55x | **1.76x** | 0.88x | **1.36x** | 1.02x |

Each entry is float16's speed over float32's on the same path. For scale, float32 on SIFT at nprobe 64:
Rust 4,605 (single), 12,567 (100), 39,938 (10,000); GPU 25,262 (100), 140,945 (10,000) queries per second.

- **Recall does not change.** SIFT's coordinates are integers from 0 to 255, exact in float16. GloVe's
  (normalized) all round, by at most 2.4e-4, and recall moves by at most 0.001.
- **float16 pays where the scan is memory-bound, and only there:** batches of 100 on 16 CPU threads
  (1.6-1.8x, close to the 2x that halving the bytes allows) and on the GPU (1.3-1.6x). These are the
  batch sizes where HNSW wins (BENCHMARKS.md).
- **It costs where the scan is not memory-bound.** A single query on one thread reads its lists well
  below the memory system's limit, so widening each value is pure extra work: 0.52-0.63x. The
  list-major scans already read each list once for many queries: the GPU gains 2-7%, Rust loses 9-13%.
  So no storage is best at every batch size.
- **How Rust widens matters.** Rust has no stable float16 type, so the lists hold float16 bits (`u16`),
  widened exactly in safe code (sign, exponent and mantissa moved into float32's positions, then one
  multiply by 2^112; correct for subnormals and both zeros, checked against NumPy on all 63,488 finite
  float16 values in `check_kernels.py`). Widening inside the distance loops, the first version, made
  the compiler load the halves one lane at a time: 1,244 (single), 14,535 (100) and 20,336 (10,000)
  queries per second on SIFT at nprobe 64. Widening each block into a buffer first, in one flat loop,
  gives 2,379, 21,766 and 34,930 (1.3-1.9x faster). Over nested `[u16; 8]` arrays even that loop was
  compiled into lane-by-lane loads; over flat slices it is one 16-byte load per 8 values.
- **Checks.** With every list probed, both scans on float16 lists must return exact search on the
  rounded vectors (`check_kernels.py`, `ivf_gpu.py --check`), compared by the neighbors' exact
  distances, since rounding creates exact ties (an L-infinity distance is a single coordinate
  difference). The float32 path is unchanged: bit-identical outputs before and after.

### uint8 lists on SIFT (`fp16_lists.py --storages float32,float16,uint8`)

*The HNSW figure quoted here is uncompressed HNSW's; compressed, HNSW gains as much (see "Against
FAISS's compressed indexes" below).*

SIFT's coordinates are integers from 0 to 255, so its lists can be stored as bytes, a quarter of
float32's, with nothing lost (both IVFs refuse uint8 lists for other data). Rust widens a byte with one
conversion, cheaper than float16's. SIFT, L2, same setup as above; raw output in
`proto/results/uint8_sift_l2.txt`. Each entry is the storage's speed over float32's in the same run.

| Path | float16, nprobe 16 / 64 | uint8, nprobe 16 / 64 |
| --- | ---: | ---: |
| Rust, batches of 100 | 1.57x / 1.66x | **2.16x / 2.85x** |
| GPU, batches of 100 | 1.43x / 1.58x | **1.66x / 2.02x** |
| Rust, single queries | 0.62x / 0.51x | 0.83x / 0.74x |
| Rust, 10,000 (list-major) | 0.89x / 0.99x | 0.96x / 1.11x |
| GPU, 10,000 (list-major) | 1.06x / 1.06x | 1.08x / 1.09x |

Recall is unchanged (0.840 and 0.973). Quartering the bytes keeps paying in the memory-bound cases:
uint8 on batches of 100 is 1.4-1.7x float16 in Rust and 1.2-1.3x on the GPU. Rust's single queries
still lose, but less (widening a byte costs less than a float16). In absolute terms, at nprobe 64
(recall 0.973) uint8 lists give 36,318 queries per second in Rust and 51,821 on the GPU in batches of
100, against 12,737 and 25,603 with float32; ivf_cpu.py measured FAISS's HNSW at 58,470 for recall 0.978
there (BENCHMARKS.md, a different run). The float32 Rust list-major rate at nprobe 64 came out at 34,127
here against about 39,900 in the float16 runs, so that column carries about 15% noise.


## Single queries split over threads (`single_query.py`)

*The HNSW comparisons in this section are against uncompressed HNSW; against FAISS's compressed HNSW,
see "Against FAISS's compressed indexes" below, which reverses the SIFT result.*

A single query on one thread is not memory-bound, which is why compressed lists slow it down, and
why HNSW, which reads far less per query, wins single queries on SIFT. Splitting one query over
threads attacks its latency instead: `IvfIndex.search(..., split=n)` runs the coarse step (the 4,000
centroids, in n chunks) and the probed lists (n contiguous runs) as n parallel tasks on a thread pool
of n threads of its own, then merges the runs' best k in probe order. The result is the same as one
thread's, index for index, ties included (`check_kernels.py`, for 2, 3 and 16 threads, every metric
and list storage, uint8's tied integer data included).

L2, the same 4,000 lists as ivf_cpu.py; latency per query (one call each, 1,000 queries after a
warm-up), median in microseconds; recall at 10 against exact search. HNSW (M = 32, FAISS) in the same
run, on one thread per query as always. Raw output: `proto/results/single_sift_l2.txt`,
`single_glove_l2.txt`, and the pool-size sweep `single_sift_l2_threads.txt`.

| Dataset, recall (IVF / HNSW) | HNSW | Rust, 1 thread | Rust, 4 threads | Best Rust (storage, threads) |
| --- | ---: | ---: | ---: | ---: |
| SIFT, 0.828 / 0.842 | **53** | 97 | 62 | 62 (float32, 4) |
| SIFT, 0.972 / 0.974 | 161 | 224 | 123 | **116** (uint8, 4) |
| SIFT, 0.994 / 0.992 | 299 | 391 | 200 | **155** (uint8, 8) |
| GloVe, 0.716 / 0.711 | 103 | 88 | **60** | 60 (float32, 4) |
| GloVe, 0.843 / 0.856 | 346 | 203 | **111** | 111 (float32, 4) |
| GloVe, 0.894 / 0.899 | 649 | 338 | **176** | 175 (float16, 8) |

- **Four threads cut the latency 1.3-2.0x** (most at high nprobe), and put siann's IVF ahead of HNSW on
  latency everywhere but SIFT at low recall: 1.4-1.9x on SIFT at recall 0.97 and above, 1.7-3.7x on
  GloVe. HNSW keeps SIFT at recall 0.84 (53 against 62 microseconds).
- **This buys latency, not throughput.** A split query uses 4-8 cores where HNSW uses one; with many
  concurrent queries, one query per thread (ivf_cpu.py's "single" column) is the fair comparison, and
  there HNSW still wins on SIFT.
- **Compressed lists pay again when split:** spread over threads, the scan is memory-bound, so uint8
  lists beat float32 at nprobe 64 and above (116 against 123, 155 against 200 microseconds), while on
  one thread they lose (284 against 224). float16 gains less, its widening costing more.
- **More threads are not better.** Four threads are best at every nprobe for float32; eight help only
  at nprobe 128 with compressed lists. At low nprobe, eight threads are slower than one (72 against 66
  microseconds at nprobe 4): starting and joining tasks costs more than the little scanning saves.
- **The pool matters more than the task count.** The first version ran split queries on rayon's
  global pool (all 16 cores, 4 of them efficiency cores): a query injected into it wakes its sleeping
  threads, and latency had a floor of about 100 microseconds (nprobe 4: 103 against 66 on one thread;
  nprobe 128: 293 against 465). Fewer tasks on the same pool did not help (4 tasks: 130 at nprobe 4);
  a small pool of its own did (4 threads: 45). Each pool size is made once and kept.
- **Noise:** repeated runs of the same configuration differ by up to 15% (one-thread float32 at nprobe
  128: 391-465 microseconds across runs), so differences under that are not findings.

## Against FAISS's compressed indexes (`fp16_lists.py --faiss`, `single_query.py --faiss`)

The float16, uint8 and split-query results above compare compressed siann lists with uncompressed
FAISS. FAISS compresses too, so the strongest alternatives are its own: IVF over the same centroids with
`IndexIVFScalarQuantizer` (`QT_fp16`, `QT_8bit_direct`, with `by_residual = False`, so that it stores
exactly the vectors siann's lists do: on integer data it returns IVF-Flat's distances exactly), and
HNSW with `IndexHNSWSQ` in the same types (M = 32). For a single query, FAISS's IVF can also be split
over threads its own way (`parallel_mode = 1`). L2 only: FAISS's scalar quantizer has no L1 or
L-infinity. Raw output: `proto/results/faiss_sq_sift_l2.txt`, `faiss_sq_glove_l2.txt` (throughput),
`faiss_single_sift_l2.txt`, `faiss_single_glove_l2.txt` (latency).

**FAISS needs dimensions in multiples of 8 for its fast scalar-quantizer path.** At 100 dimensions
(GloVe), its float16 search was 2-6 times slower than its own float32 (IVF on single queries: 786
against 4,452 per second at nprobe 64); on Gaussian data, float16 flat
search took 0.118 s at 100 dimensions against 0.016 s at 104 (100 queries, 200,000 vectors), while
float32 did not care (0.024 s at both). FAISS's indexes were therefore given data zero-padded to a
multiple of 8 (which leaves L2 distances unchanged), as a careful user would; padding costs its float32
IVF about 15% on GloVe (4,452 against 3,748 single queries per second at nprobe 64), so for float32 the
better of the padded and unpadded runs is used below.

**Throughput** (queries per second; the best storage of each method at the nearest recall):

| Dataset, recall | Batch | siann | FAISS HNSW (best storage) |
| --- | --- | ---: | ---: |
| SIFT, 0.973 / 0.978 | 100 | GPU uint8 51,380; Rust uint8 34,223 | **uint8 109,965** |
| SIFT, 0.973 / 0.978 | 10,000 | GPU uint8 145,814 | uint8 152,655 |
| SIFT, 0.840 / 0.862 | 10,000 | **GPU uint8 571,146** | uint8 457,279 |
| GloVe, 0.839 / 0.843 | 100 | GPU float16 36,832; Rust float16 24,632 | float16 34,332 |
| GloVe, 0.711 / 0.704 | 100 | GPU float16 96,566 | float16 95,080 |
| GloVe, 0.839 / 0.843 | 10,000 | **GPU float16 151,665** | float16 40,015 |
| GloVe, 0.839 / 0.845 | 1 (one thread each) | **Rust float32 4,975** | float32 2,770; float16 3,555 |

**Single-query latency** (median microseconds; siann split over 4 threads, HNSW on one):

| Dataset, recall (IVF / HNSW) | siann, best (storage, threads) | FAISS IVF, best | FAISS HNSW, best storage |
| --- | ---: | ---: | ---: |
| SIFT, 0.828 / 0.842 | 64 (float32, 4) | 62 (uint8, 1) | **29** (uint8) |
| SIFT, 0.972 / 0.974 | 117 (uint8, 4) | 134 (uint8, 1) | **92** (uint8) |
| SIFT, 0.994 / 0.992 | 181 (uint8, 4) | 185 (uint8, 4) | 179 (uint8) |
| GloVe, 0.716 / 0.716 | **60** (float32, 4) | 88 (float16, 1) | 89 (float16) |
| GloVe, 0.843 / 0.852 | **111** (float32, 4) | 166 (float16, 4) | 285 (float16) |
| GloVe, 0.894 / 0.894 | **174** (float32, 4) | 227 (float16, 4) | 544 (float16) |

- **HNSW gains as much from compression as IVF does, or more:** on SIFT, uint8 makes it 1.6-2.3 times
  as fast in every batch size (ef 64: 58,970 to 109,965 in batches of 100). So on SIFT the gap in
  batches of 100 stays about 2x, and in batches of 10,000 the GPU's lead (1.1-2.0x against
  uncompressed HNSW in the earlier IVF comparison) shrinks to level at recall 0.97 and 1.25x at 0.84.
- **On SIFT, compressed HNSW on one core beats siann's split IVF on four** at recall 0.84 (29 against
  64 microseconds) and 0.97 (92 against 117), and is level at 0.99. The single-query finding above
  ("ahead of HNSW on SIFT at recall 0.97 and above") held only against uncompressed HNSW.
- **On GloVe, siann's leads hold:** single-query latency 1.5-3.1 times compressed HNSW's, one-thread
  throughput 1.4 times, the GPU 3.8 times in batches of 10,000; batches of 100 are level.
- **FAISS decodes compressed lists far more cheaply than Rust widens them.** On one thread, FAISS's
  uint8 IVF answers a SIFT query at nprobe 64 in 134 microseconds against Rust's 286 (and Rust's float32
  224); its float16 is level with its float32, where Rust's float16 takes 1.8 times its float32. FAISS
  converts in registers inside its SIMD distance loop; Rust widens each block into a buffer first. A
  Rust scan that converts in registers would make compressed lists pay on one thread too.
- **FAISS's own split (`parallel_mode = 1`) has the floor siann's first version had:** about 100
  microseconds at low nprobe (OpenMP waking its threads), so FAISS's IVF answers one query faster on
  one thread than split at nprobe 16 and below.
- **siann's IVF scans are level with FAISS's IVF** on the same lists in batches of 100 (uint8 at nprobe
  64: 34,223 against 34,673). In batches of 10,000, Rust's list-major scan is 2.5-3.2x FAISS's IVF
  with float32 lists, which FAISS scans query by query, but with compressed lists the margin goes:
  level with uint8 (37,690 against 38,997 at nprobe 64), 1.0-1.4x with float16.

The earlier sections' comparisons with HNSW (float16 and uint8 lists, single queries) stand as
measured, but against uncompressed HNSW; these are the comparisons to quote.

## Exact L2 on the CPU without FAISS: BLAS and a Rust top-k (`blas_l2.py`)

FAISS's exact L2 is a BLAS matrix multiply (the products q.x) followed by a top-k selection. siann can
do the same without FAISS: on a Mac, NumPy's `@` calls Apple's Accelerate, the same BLAS FAISS links
(`otool -L` on libfaiss, `numpy.show_config()`), and siann's Rust `L2TopK` ranks by |x|^2 - 2 q.x and
keeps each query's best k across blocks of the base, reading NumPy's product matrix in place (through
rust-numpy, whose safe API keeps the `unsafe` inside the crate, as pyo3's does) with the queries'
rows split over threads. Against FAISS's `IndexFlatL2` with its BLAS threshold set to 20 (its default
sends fewer than 128,000 queries down a 4-7 times slower path; see `faiss_blas.py`). k = 10, all queries
in one call, median of three after a warm-up. Raw output: `proto/results/blas_l2_sift.txt`,
`blas_l2_glove.txt`, `blas_l2_sift_100.txt`, `blas_l2_glove_100.txt`. Agreement with FAISS is 1.0000
in every configuration.

| Dataset, queries | FAISS | siann, default tiles | Speed-up | Best tiles in the sweep |
| --- | ---: | ---: | ---: | ---: |
| SIFT, 1,000 | 0.382 s | 0.220 s (512 x 8,192) | **1.74x** | 1.77x (1,000 x 8,192) |
| GloVe, 1,000 | 0.463 s | 0.239 s (512 x 8,192) | **1.94x** | 1.98x (1,000 x 8,192) |
| SIFT, 100 | 0.107 s | 0.036 s (100 x 32,768) | **3.00x** | 3.29x (100 x 65,536) |
| GloVe, 100 | 0.130 s | 0.036 s (100 x 32,768) | **3.56x** | 3.88x (100 x 65,536) |

- **The tile size decides it.** Each multiply computes a query tile against a base block. Products of
  about 4-8 million values (16-32 MB) are fast; smaller ones pay per-call overhead (128 x 8,192: 0.60x
  FAISS on SIFT), larger ones drop out of cache and the multiply slows by about half (1,000 x 16,384:
  1.11x). The default keeps the product near 2^22 values: a query tile of up to 512, and a block of
  2^22 / tile rounded down to a power of two. One sweep point breaks the pattern (128 x 65,536 on
  SIFT: 0.93x, against 1.56x on GloVe), and repeated runs vary, so this is a default to sweep in
  TUNING, not a fitted formula.
- **The multiply is most of the time** (0.14-0.16 s of 0.22-0.24 s at the default for 1,000 queries);
  the top-k takes the rest, and less as the block grows.
- **This replaces FAISS as the CPU recommendation for exact L2.** The earlier "FAISS flat; siann has
  nothing faster" held for siann's own kernels (the Rust flat L2 kernel, without BLAS: 0.630 s
  against FAISS's 0.446 s on SIFT, 0.71x; `backends.py --dataset sift --metrics l2`, raw output in
  `proto/results/rust_l2_sift.txt`), not for BLAS, which siann can call through NumPy without calling FAISS.
- **Checks:** `check_kernels.py` runs the same path with odd tiles (16 queries, blocks of 777) against
  NumPy, by the neighbors' exact distances (the |x|^2 - 2 q.x form rounds differently from summed
  squares).
- **Not measured:** single queries and small batches below 100 (where a multiply has little to
  reuse and the Rust kernel may win), cosine as such (it is L2 on normalized vectors, so GloVe's
  numbers apply), and BLAS inside the CPU IVF's list-major scan.

## Decoding compressed lists in registers, and why FAISS's uint8 is faster (`single_query.py`, `faiss_sq8_query.py`)

On one thread, FAISS's uint8 IVF answers a SIFT query twice as fast as siann's Rust (134 against 286
microseconds at nprobe 64; "Against FAISS's compressed indexes"). Rust widened each block into an f32
buffer before the distance loop. Converting each value in registers inside the loop instead
(`block_dists_one_direct`) is clean for bytes (two vector conversions, no lane-by-lane loads in the
assembly), but float16's bit operations still make the compiler load the halves one lane at a time,
so float16 keeps the buffer (`Elem::DIRECT`). Single-query latency on SIFT L2, median microseconds,
buffer (`faiss_single_sift_l2.txt`) against in registers (`single_sift_l2_direct.txt`):

| uint8 lists | 1 thread, nprobe 64 | 1 thread, 128 | 4 threads, 64 | 4 threads, 128 |
| --- | ---: | ---: | ---: | ---: |
| buffer | 286 | 517 | 117 | 181 |
| in registers | 263 | 478 | 108 | 166 |

About 8% faster, and float32 is unchanged (bit-identical outputs; 218-224 microseconds). uint8 on one
thread is still slower than float32, so conversion was not the main gap. **FAISS rounds the query:**
for `QT_8bit_direct` lists it rounds the query to integers and computes in integer arithmetic. With
queries offset by 0.3 from integers, its distances equal those of the rounded query exactly (84,810;
the float query's is 84,802.08) (`faiss_sq8_query.py`, raw output `faiss_sq8_query.txt`). That is
exact for SIFT, whose queries are integers, so the comparisons above stand, but for other queries
FAISS's "lossless" uint8 index is approximate. siann could do the same exactly when the queries are
integers (integer absolute differences and squares, 16 lanes at a time in NEON): a new kernel, not
done.

## BLAS in the CPU IVF: tried and dropped (`blas_ivf.py`)

The exact-L2 gain from BLAS (`blas_l2.py`) suggested the CPU IVF's list-major scan could use it too:
lists stored sorted by list in NumPy, the coarse step through `blas_l2.py`'s path, queries grouped by
the lists they probe, one multiply per list (its queries against its vectors, NumPy's `@`), and the
products ranked in Rust a group of 256 lists at a time in parallel (`L2TopK.push_l2_lists`). Built on
identical lists to the Rust IVF (the same assignment), so recall matches exactly. Queries per second;
raw output `proto/results/blas_ivf_sift.txt`, `blas_ivf_glove.txt`.

| Dataset, nprobe | Rust IVF, 100 / 10,000 | BLAS IVF, best workers, 100 / 10,000 |
| --- | ---: | ---: |
| SIFT, 16 | 42,969 / 113,248 | 8,135 / 61,120 |
| SIFT, 64 | 12,429 / 39,640 | 1,950 / 31,394 |
| GloVe, 16 | 48,827 / 126,068 | 7,565 / 62,437 |
| GloVe, 64 | 14,481 / 44,571 | 2,007 / 31,216 |

The BLAS IVF is 0.5-0.8 times Rust's list-major scan on batches of 10,000 and 0.14-0.19 times its
query-major scan on batches of 100. The multiplies are small (about 160 queries by 250 vectors per
list at nprobe 64), so overheads dominate: on SIFT at nprobe 64 with 4 worker threads (NumPy releases
the GIL in `@`), a batch of 10,000 spends 0.116 s on the coarse step and grouping, 0.120 s on the
products and 0.100 s on ranking, against 0.25 s for Rust's whole search, which does the same
arithmetic in cache, four queries at a time. Even with the coarse step and grouping free it would at
best draw level. Dropped; BLAS stays for exact L2.

## The coarse step, and a two-level coarse search (`coarse_step.py`)

Before scanning its lists, an IVF query is compared with every centroid (the coarse step). Fitting
single-query latency (Rust IVF, float32 lists, one thread, SIFT L2) against nprobe from 1 to 16 gives
the coarse step plus the call as the intercept and the cost of a list as the slope; the call alone (a
tiny index) takes about 1 microsecond. Raw output: `proto/results/coarse_sift_l2.txt`.

| Lists | Coarse step (intercept) | Per list | nprobe 16: recall, latency |
| ---: | ---: | ---: | ---: |
| 4,000 | **50 microseconds** | 2.7 | 0.828, 94 |
| 2,000 | 27 | 4.8 | 0.874, 104 |
| 1,000 | 16 | 8.8 | 0.924, 156 |

At siann's default of 4,000 lists and nprobe 16, **the coarse step is half of a single query.**
**Fewer lists do not fix it:** each list grows as the coarse step shrinks, and at matched recall
1,000 lists (nprobe 8: 0.822 in 89 microseconds) are level with 4,000 (nprobe 16: 0.828 in 94); 4,000
win from recall 0.92 up, and fewer lists help only near 0.7.

**A two-level coarse step** (`IvfIndex.set_coarse_groups`, `search(..., gprobe=g)`) clusters the
centroids by k-means into groups (64 for 4,000 lists, 128 for 8,000); a query is compared with the
groups' centres, then only with the centroids of its g nearest groups. With every group probed it
returns exactly the exact coarse step's results (`check_kernels.py`). Single-query latency, median
microseconds, one thread; raw output `coarse_two_level_sift_l2.txt`, `coarse_two_level_sift_l2_uint8.txt`,
`coarse_two_level_glove_l2.txt`:

| Dataset, lists (groups), nprobe | Exact coarse step: recall, latency | 16 groups probed | 8 groups probed |
| --- | ---: | ---: | ---: |
| SIFT, 4,000 (64), 16 | 0.828, 96 | 0.828, **60** | 0.821, 54 |
| SIFT, 4,000 (64), 32 | 0.921, 141 | 0.920, **103** | 0.906, 98 |
| SIFT, 8,000 (128), 32 | 0.879, 152 | 0.876, **71** | 0.851, 65 |
| GloVe, 4,000 (64), 16 | 0.716, 89 | 0.712, **58** | 0.698, 52 |
| GloVe, 4,000 (64), 64 | 0.843, 198 | 0.832, **173** | 0.813, 168 |

- **1.1-2.1 times faster on one thread with 16 groups probed,** most at low nprobe and with many
  lists, where the coarse step is a large share (GloVe at nprobe 64 gains only 1.14x). Recall drops by
  at most 0.011 with 16 of 64 groups; with 8,000 lists, 16 of 128 groups lose more on GloVe (up to
  0.022), and 8 groups up to 0.03.
- **Against FAISS's HNSW** (`faiss_single_*_l2.txt`, one thread each): on SIFT at recall 0.85, siann
  takes 65 microseconds (8,000 lists, 8 groups) against float32 HNSW's 53 and uint8 HNSW's 29 (recall
  0.842); at 0.92, 103 against 88 and 51. On GloVe, 58 microseconds at recall 0.712 against float16
  HNSW's 89 at 0.716 (1.5x), and 173 at 0.832 against 285 at 0.852 (1.65x, at slightly lower recall).
  So on one thread siann now draws level with float32 HNSW on SIFT and beats compressed HNSW on GloVe
  without splitting; compressed HNSW still leads on SIFT by about 2x.
- **What is left on SIFT is the list scan:** uint8 lists are still slower than float32 on one thread
  (4,000 lists, nprobe 16, 16 groups: 71 against 60 microseconds), which is what an integer kernel
  would address. Splitting over 4 threads gains little once the coarse step is cheap (60 to 58).

## An integer path for byte lists (`single_query.py`, `coarse_step.py`, `fp16_lists.py`)

FAISS's uint8 IVF answers a single query about twice as fast as siann's Rust because it rounds the
query to integers and computes in integer arithmetic (above). siann now does the same, exactly: for
uint8 lists, a query whose coordinates are all integers from 0 to 255 (SIFT's are) takes an integer
path; any other query takes the float path, so nothing is approximated. The integer distances are
exact, and below 2^24 for up to 258 dimensions (larger d falls back to float), so they rank exactly
and equal the float path's, whose partial sums of integers are exact too: `check_kernels.py` requires
the uint8 index to return the float32 index's indices on the same integer data, for every metric, one
thread and split, and non-integer queries to fall back. Raw output: `proto/results/int_single_sift_l2.txt`
(with FAISS in the same run), `int_two_level_sift_l2.txt`, `int_uint8_sift_l2.txt`.

**The layout mattered more than the arithmetic.** In the IVF's blocked layout (8 vectors per block,
dimension by dimension), a dimension's 8 bytes fill half a NEON register. An integer kernel there,
written with plain loops, was vectorized across dimensions (integer sums may be reordered, so the
compiler gathered bytes one lane at a time, as with `f32::max` before); written with the `wide` crate's
vector types it widened each 8-byte row with about eight shift-and-shuffle instructions; with
`wide`'s 16-byte loads and single-instruction widening, it came out level with the float path (259
against 263 microseconds at nprobe 64; these versions of `lib.rs` were not kept). FAISS stores each
vector's bytes contiguously instead. So byte lists now also keep a row-major copy (each vector's
coordinates contiguous): a plain loop per vector, which for integers the compiler may reorder, then
compiles to 16-byte absolute differences (`uabd`) and, for L2, the dot-product instruction (`udot`),
16 values per instruction, with no SIMD library. The copy costs a byte per coordinate, so byte lists
take two bytes per coordinate in all, half of float32's.

Single-query latency, SIFT L2, uint8 lists, median microseconds:

| | nprobe 16 | nprobe 64 | nprobe 128 |
| --- | ---: | ---: | ---: |
| float path, converted in registers, 1 thread (`single_sift_l2_direct.txt`) | | 263 | 478 |
| **integer path, row-major, 1 thread** | **77** | **147** | **244** |
| integer path, 4 threads | 52 | 76 | 111 |
| FAISS IVF uint8, 1 thread, same run | 62 | 136 | (249 in `faiss_single_sift_l2.txt`) |
| siann float32 lists, 1 thread | 103 | 237 | 405 |

- **1.8 times the float path on one thread**, level with FAISS's IVF (147 against 136 microseconds at
  nprobe 64; 244 against 249 at 128), and 1.6 times siann's own float32 lists: byte lists now pay on
  one thread too.
- **With the two-level coarse step,** against FAISS's uint8 HNSW in the same run (one thread each):
  siann 42 microseconds at recall 0.851 (8,000 lists, 8 of 128 groups) against HNSW's 29 at 0.842;
  66 at 0.920 (4,000 lists, 16 of 64 groups) against 51 at 0.925. HNSW's single-query lead on SIFT
  falls from about 2 times to 1.3-1.45 times on one thread; with 4 threads siann leads from recall
  0.97 (76 microseconds at 0.972 against 94 at 0.974).
- **Throughput** (`int_uint8_sift_l2.txt`): uint8 single queries one per thread 6,254 per second at
  nprobe 64 (3,362 before), batches of 100 36,945 (34,223 before; memory-bound, so they gain less).
  The list-major scan for large batches still converts bytes to float.

**Not done: the integer path in the list-major scan (large batches).** Rust's list-major scan, for
batches of thousands, still converts bytes to float. It was left alone on purpose: in a batch of
10,000 on SIFT at nprobe 64 it does 37,339 queries per second with uint8 lists, against the GPU IVF's
147,384 (recall 0.973; `int_uint8_sift_l2.txt`) and compressed HNSW's 152,655 on the CPU (recall 0.978;
`faiss_sq_sift_l2.txt`). Even a gain as large as the single-query path's (1.6-1.8 times) would leave
it far behind both, it applies only to integer byte data, and it would need a new kernel (the scan
compares 4 queries with each block at a time).

## Inner product (`backends.py --metrics ip`, `blas_l2.py --metric ip`, `ivf_cpu.py --metric ip`, `ip_reduction.py`)

Inner product ("ip"), largest first, is now a metric everywhere: siann ranks by -q.x, so that, as for
the distances, smaller is nearer. Exact search is a matrix multiply (on the GPU, `addmm` with alpha -1
and a zero row; on the CPU, the BLAS path with zero norms); the Rust kernels and the IVF scans add -q x
per coordinate (the GPU kernels' `combine` now takes both coordinates, with L1, L2 and L-infinity
bit-identical to before). The blocked layouts pad with infinity, harmless for the distances but the
best possible inner product (or NaN against a zero coordinate), so exact search now skips padding
slots by index. The integer path is for the distances only. `check_kernels.py` covers inner product on
every path. The dataset is GloVe-100 unnormalized (`glove-raw`), whose lengths run from 0.04 to 18.7,
so that inner product differs from cosine; this is maximum inner product search, where graph indexes
are known to struggle.

**Exact search** (raw output `proto/results/ip_exact_glove_raw.txt`, `ip_exact_glove_raw_batches.txt`,
`ip_blas_glove_raw.txt`):

| Queries | FAISS IndexFlatIP | siann GPU (MLX) | siann CPU (BLAS, default tiles) | siann Rust kernel |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.136 s | **0.008 s, 17.7x** | 0.040 s, 2.7x (0.108 s in that run) | |
| 1,000 | 0.287 s | **0.049 s, 5.9x** | 0.273 s, 0.94x (0.256 s in that run) | 0.392 s, 0.73x |
| 10,000 | 1.985 s | **0.482 s, 4.1x** | | |

On 1,000 queries the CPU BLAS path is only level with FAISS (0.94-1.08x across tiles), where for L2
it was 1.7-1.9x: FAISS's inner product has no norms to add and takes 0.29 s against 0.46 s for its
L2 on GloVe, leaving less to win back.

**IVF does not work for inner product on this data** (`ip_ivf_glove_raw.txt`, `ip_reduction_glove_raw.txt`):

- **Assigned by inner product** (as FAISS's METRIC_INNER_PRODUCT IVF does), the lists are badly
  unbalanced, because centroids pointing towards long vectors attract most of the data: the largest
  list holds 35,725 vectors, the 10 largest 21% of all of them, and 39% of the lists are empty (by L2:
  2,066, 2%, none). Recall stays at 0.23-0.64 even at nprobe 128, for siann's IVFs and FAISS's alike
  (they share the lists), at 1,700 queries per second in batches of 100 at recall 0.64.
- **Reduced to L2** (Bachrach et al., 2014: each base vector extended by sqrt(M^2 - |x|^2) to length
  M, each query by 0, so that L2 ranks as inner product), the lists balance, but recall falls to
  0.03-0.34 for siann's IVFs, and FAISS's HNSW on the extended vectors does as badly (0.16-0.34): most
  vectors are far shorter than the longest, so the added coordinate dominates the distances.
- **FAISS's HNSW with inner product** is the best approximate method measured: recall 0.888 at 14,224
  queries per second in batches of 100 and 15,275 in batches of 10,000 (efSearch 256).
- **Exact search on the GPU nearly matches or beats it:** recall 1.0 at about 12,500 queries per
  second in batches of 100 (0.88 times HNSW at 0.888) and about 20,700 in batches of 10,000 (1.36
  times).

So for inner product with lengths this uneven, siann's answer is exact search on the GPU. Data whose
vectors have similar lengths (many embedding models produce nearly normalized vectors) is close to
cosine, where the IVF works; that was not measured separately.
