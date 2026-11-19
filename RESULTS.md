# Results

*Part of the siann documentation: [README](README.md), [ALGORITHMS](ALGORITHMS.md),
[ARCHITECTURE](ARCHITECTURE.md), [BACKENDS](BACKENDS.md), [DATASETS](DATASETS.md),
[BENCHMARKS](BENCHMARKS.md).*

The measurements behind siann's documentation, by topic. Each topic gives the numbers, the script
that produces them and what they mean for using or changing siann, including the approaches that do
not pay. The scripts are in [proto/](proto/) and run from there
(`uv run --no-sync --group proto python <script>`); their raw output, where saved, is in
[proto/results/](proto/results/). [BENCHMARKS](BENCHMARKS.md) turns these into a recommendation.

## Setup

- **Machine:** Apple M3 Max (12 performance and 4 efficiency cores, 64 GB, 400 GB/s), 16 CPU threads.
- **Software:** FAISS 1.16 (CPU), MLX 0.32.4, PyTorch 2.15.0, JAX 0.11.2 with jax-mps 0.11.1, NumPy
  with Apple's Accelerate.
- **Data:** SIFT1M, GloVe-100 (normalized, so that L2 ranks as cosine; `glove-raw` unnormalized, for
  inner product) and Gaussian vectors ([DATASETS](DATASETS.md)). k = 10 throughout.
- **Correctness:** every kernel and index is checked against NumPy's exact search (`check_kernels.py`,
  and `ivf_gpu.py --check` for MLX's IVF, in a process of its own since FAISS and PyTorch cannot share
  one). Exact search agrees with FAISS's flat index in every configuration (agreement 1.0000).
- **Recall** at 10 counts a returned neighbor as correct when it is no farther than the exact 10th
  nearest, so that ties broken differently (common on SIFT's integer data) are not misses
  (`agreement` in `backends.py`).
- **Timing:** batch timings are medians of three runs after a warm-up on the full query set;
  single-query throughput is one pass of 1,000 calls; latency is the median of 1,000 single calls after
  a warm-up. Repeated runs of the same configuration differ by up to 15%, so smaller differences are
  not findings.
- **FAISS at its best:** its flat L2 needs `faiss.cvar.distance_compute_blas_threshold = 20` (see
  "FAISS's BLAS threshold"), and its compressed indexes need dimensions in multiples of 8 (see "List
  storage"); every comparison below sets both.

## Exact search on the GPU (`backends.py`, `profile_l2.py`)

1,000 queries; speed over FAISS's flat index on the CPU (`backends.py --dataset sift|glove`):

| Backend | SIFT L2 | SIFT L1 | GloVe L2 (cosine) | GloVe L1 |
| --- | ---: | ---: | ---: | ---: |
| FAISS flat (CPU), time | 0.405 s | 2.07 s | 0.431 s | 1.65 s |
| MLX, siann's kernels | 8.7x | 25x | 8.2x | 18.5x |
| PyTorch MPS, siann's kernels | 4.0x | 25x | 3.9x | 18.4x |

On 1,000,000 Gaussian vectors of 128 dimensions, 1,000 queries: FAISS L2 0.39 s, L1 1.82 s; MLX 0.046
s (8.4x) and 0.082 s (22x); with 10,000 queries, 7.6x and 21x. L-infinity on 200,000 Gaussian vectors
(`backends.py --metrics l1,linf`, `linf_gaussian.txt`): FAISS 0.972 s against its own L1's 0.334 s; MLX
49.0x and PyTorch 47.2x, where for L1 they are 17.4x and 16.6x. Inner product on raw GloVe
(`ip_exact_glove_raw*.txt`): MLX 17.7x FAISS's `IndexFlatIP` on 100 queries (0.008 against 0.136 s),
5.9x on 1,000 (0.049 against 0.287 s) and 4.1x on 10,000 (0.482 against 1.985 s).

What makes the GPU fast:

- **A dedicated top-k kernel.** For 256 queries against a million vectors the distances take 18 ms,
  and the frameworks' selection (`argpartition`, `topk`) 146-152 ms. siann's two-stage selection
  (`knn_kernels.select_k`: each of 512 threads per query keeps its best k in registers, then a small
  selection over the survivors) takes 4.1 ms.
- **L1 and L-infinity tiled like a matrix multiply** (`l1_dist`: 16 x 16 threads compute 64 x 64
  distances, 4 x 4 each in registers, staging 16 dimensions at a time in threadgroup memory), so that
  each base vector is read once per 64 queries. A fused kernel that computes and selects in one pass
  reads the whole base once per query and is memory-bound, at 2.3-2.7x FAISS. With `select_k` after
  it, the tiled L1 is as fast as L2, and L-infinity as fast as L1 (0.019-0.021 s on 200,000 vectors).
- **L2 as one fused multiply:** `addmm` (norms minus twice the products) takes 6.9 ms per tile of 256
  x 1M x 128 in MLX, against 17.7 ms written as `norms - 2 * (q @ x.T)`; it doubles siann's exact L2
  (0.087 to 0.046 s).
- **Evaluate each tile before the next.** MLX evaluates lazily; evaluated only at the end, every
  tile's 1 GB distance matrix stays alive (196 s instead of 0.9 s for 10,000 queries).
- **A maximum is exact:** L-infinity distances match NumPy's to the bit (`check_kernels.py` requires
  it).
- **The kernel, not the framework, sets the speed.** The same Metal kernel runs at the same speed in
  MLX and PyTorch (L1 0.082 s in both; the top-k kernel 4.2 against 3.8 ms per tile). The frameworks
  differ in their matrix multiply (256 x 1M x 128: MLX 6.7 ms, PyTorch 14.2; with `addmm`, 6.9 and 18.1
  ms), which is the whole gap in L2.
- **Small collections:** at 200,000 vectors the GPU barely pays for L2 (FAISS 0.079 s against MLX
  0.099 s without `addmm`, one run) and still does for L1 (0.300 against 0.106 s).

## Framework built-in operations (`backends.py`)

What a user of each framework would write: L2 as norms minus twice a matrix multiply, L1 as a
broadcast absolute difference (MLX, JAX) or `torch.cdist` (PyTorch), then the framework's top-k.
1,000,000 x 128 Gaussian vectors, 1,000 queries:

| Backend | L2 | vs FAISS | L1 | vs FAISS |
| --- | ---: | ---: | ---: | ---: |
| FAISS flat (CPU) | 0.39 s | 1.00x | 1.82 s | 1.00x |
| MLX built-in (GPU) | 0.65 s | 0.60x | 7.20 s | 0.25x |
| MLX compiled (GPU) | 0.63 s | 0.61x | 4.37 s | 0.42x |
| PyTorch built-in (MPS) | 0.70 s | 0.55x | 4.72 s | 0.39x |
| JAX built-in (jax-mps) | 0.67 s | 0.58x | 4.56 s | 0.40x |
| PyTorch built-in (CPU) | 1.10 s | 0.35x | 4.52 s | 0.40x |
| JAX built-in (XLA, CPU) | 1.34 s | 0.29x | 3.95 s | 0.46x |
| MLX built-in (CPU) | 6.8 s | 0.06x | 32.5 s | 0.06x |

The ratios hold on SIFT1M and GloVe-100 (0.34-0.65x on the GPU).

- **Built-in operations lose to FAISS's CPU everywhere,** on the GPU too: the GPU's advantage comes
  entirely from custom kernels.
- **`mx.compile` helps MLX's L1 (7.2 to 4.4 s) but not its L2.**
- **jax-mps has no custom-kernel route** (Pallas, JAX's kernel language, targets CUDA and TPU), and its
  built-in operations are level with the other frameworks'. Make jitted functions once per process,
  with the arrays as arguments: made per call with the base captured, every call recompiles (0.33 s
  instead of 0.007 s for 100 queries on 20,000 vectors).
- **`torch.cdist(p=1)` on MPS builds the whole broadcast** and fails beyond 2^31 elements, so queries go
  16 at a time.
- **FAISS and PyTorch cannot share a process on macOS** (each links its own OpenMP runtime), hence one
  process per backend in `backends.py`.

## Exact search on the CPU (`backends.py`, `blas_l2.py`, `rust_cpu_sweep.py`)

**L1 and L-infinity: siann's Rust kernel.** The base is stored in blocks of 8 vectors, dimension by
dimension; the inner loop compares 4 queries with a block, 32 independent running distances, which the
compiler turns into NEON instructions (8 `fabd.4s` and 8 `fadd.4s` per dimension for L1, counted by
`rust_cpu_sweep.py`); each thread scans the base in tiles of 1,024 vectors for its queries.

| | SIFT L1 | GloVe L1 | Gaussian L1 (1M) | Gaussian L-infinity (200,000) |
| --- | ---: | ---: | ---: | ---: |
| FAISS flat | 2.07 s | 1.65 s | 1.79 s | 0.972 s |
| siann Rust | 4.3x | 3.9x | 3.8x (0.471 s) | 7.3x (0.133 s) |

FAISS's L1 does about 70 G absolute differences per second, siann's about 270 G. Queries per thread task
(8 to 64) make no measurable difference; the default is 32. Building the blocks takes 0.10-0.13 s for
1M x 128. Write a maximum as a comparison and select, not `f32::max`: the compiler treats a maximum as a
reduction it may reorder and vectorizes the wrong loop (0.209 s for L-infinity, twice L1's time; an
unsigned-integer maximum on the float bits does the same); the select keeps one running maximum per
lane (0.133 s, 1.4 times L1's time).

**L2, cosine and inner product: BLAS and a Rust top-k** (`blas_l2.py`). NumPy's `@` calls Accelerate,
the same BLAS FAISS links; siann's Rust `L2TopK` reads the products in place (through rust-numpy) and
ranks by |x|^2 - 2 q.x, keeping each query's best k across blocks of the base. Raw output
`blas_l2_*.txt`, `ip_blas_glove_raw.txt`:

| Dataset, metric, queries | FAISS flat | siann, default tiles | Speed-up | Best tiles in the sweep |
| --- | ---: | ---: | ---: | ---: |
| SIFT, L2, 1,000 | 0.382 s | 0.220 s (512 x 8,192) | **1.74x** | 1.77x (1,000 x 8,192) |
| GloVe, L2, 1,000 | 0.463 s | 0.239 s (512 x 8,192) | **1.94x** | 1.98x (1,000 x 8,192) |
| SIFT, L2, 100 | 0.107 s | 0.036 s (100 x 32,768) | **3.00x** | 3.29x (100 x 65,536) |
| GloVe, L2, 100 | 0.130 s | 0.036 s (100 x 32,768) | **3.56x** | 3.88x (100 x 65,536) |
| raw GloVe, inner product, 1,000 | 0.256 s | 0.273 s | 0.94x | 1.08x (512 x 16,384) |
| raw GloVe, inner product, 100 | 0.108 s | 0.040 s | **2.70x** | 3.11x (100 x 32,768) |

- **The tile size decides it.** Products of 4-8 million values (16-32 MB) per multiply are fast;
  smaller ones pay per-call overhead (128 x 8,192: 0.60x FAISS on SIFT), larger ones fall out of cache
  and the multiply slows by about half (1,000 x 16,384: 1.11x). The default (`default_tiles`) keeps the
  product near 2^22 values: a query tile of up to 512, and a block of 2^22 / tile rounded down to a
  power of two. One sweep point breaks the pattern (128 x 65,536 on SIFT: 0.93x, against 1.56x on
  GloVe), so this is a default to sweep, not a fitted formula.
- **The multiply is most of the time** (0.14-0.16 s of 0.22-0.24 s at the default for 1,000 queries).
- **FAISS's inner product is cheaper than its L2** (0.29 against 0.46 s on GloVe, no norms to add),
  leaving less to win back on 1,000 queries.
- **siann's Rust kernel is no match for BLAS on L2** (0.630 against FAISS's 0.446 s on SIFT, 0.71x;
  `rust_l2_sift.txt`) or inner product (0.73x).
- **Not measured:** batches below 100 queries, where a multiply has little to reuse.

### FAISS's BLAS threshold (`faiss_blas.py`)

FAISS 1.16 sets `faiss.cvar.distance_compute_blas_threshold` to 128,000, so with fewer queries flat L2
below 128 dimensions skips BLAS for a path 4-7x slower:

| Dimensions (500,000 Gaussian vectors, 1,000 queries) | 32 | 64 | 96 | 112 | 127 | 128 | 192 | 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FAISS flat L2, default | 0.184 s | 0.453 | 0.803 | 1.002 | 1.209 | 0.177 | 0.200 | 0.230 |
| Threshold 20 | 0.156 s | 0.185 | 0.159 | 0.166 | 0.172 | 0.170 | 0.205 | 0.232 |

On GloVe-100, 2.90 s by default and 0.419 s at 20, with the same neighbors. L1 and 128 dimensions are
unaffected; HNSW does not use flat search.

### FAISS's Metal GPU path (`faiss_metal.py`, disabled)

faiss-cpu 1.16 on macOS ships a Metal backend (`index_cpu_to_gpu`). For exact L2 on 1M x 128 with
1,000 queries it took 0.222 s against siann's 0.085 s at the time (2.6x); with 10,000 queries it failed
("MetalIndexFlat: GPU search failed") and then crashed the machine, twice. The script exits at once,
and every comparison here is against FAISS's CPU methods.

## IVF: the coarse step (`coarse_step.py`)

The IVF clusters the base by k-means (FAISS's, 10 iterations, 100 sample vectors per list) into 4,000
lists for about a million vectors, assigns each vector to its nearest centroid in the search metric,
and searches a query's nprobe nearest lists. Comparing a query with every centroid, the coarse step,
is half of a single query at nprobe 16. Fitting single-query latency (Rust IVF, float32 lists, one
thread, SIFT L2) against nprobe from 1 to 16 gives the coarse step as the intercept and a list's cost
as the slope; the call itself takes about 1 microsecond (`coarse_sift_l2.txt`):

| Lists | Coarse step | Per list | nprobe 16: recall, latency |
| ---: | ---: | ---: | ---: |
| 4,000 | **50 microseconds** | 2.7 | 0.828, 94 |
| 2,000 | 27 | 4.8 | 0.874, 104 |
| 1,000 | 16 | 8.8 | 0.924, 156 |

**Fewer lists do not help:** each list grows as the coarse step shrinks, so at matched recall 1,000
lists (nprobe 8: 0.822 in 89 microseconds) are level with 4,000 (nprobe 16: 0.828 in 94); 4,000 win from
recall 0.92, and fewer lists help only near 0.7.

**A two-level coarse step does** (`IvfIndex.set_coarse_groups`, `search(..., gprobe=g)`): the
centroids clustered into groups (64 for 4,000 lists, 128 for 8,000), the query compared with the
groups' centres, then only with the centroids of its g nearest groups. With every group probed it
returns exactly the exact coarse step's results. One thread, median microseconds
(`coarse_two_level_*.txt`):

| Dataset, lists (groups), nprobe | Exact coarse step: recall, latency | 16 groups probed | 8 groups probed |
| --- | ---: | ---: | ---: |
| SIFT, 4,000 (64), 16 | 0.828, 96 | 0.828, **60** | 0.821, 54 |
| SIFT, 4,000 (64), 32 | 0.921, 141 | 0.920, **103** | 0.906, 98 |
| SIFT, 8,000 (128), 32 | 0.879, 152 | 0.876, **71** | 0.851, 65 |
| GloVe, 4,000 (64), 16 | 0.716, 89 | 0.712, **58** | 0.698, 52 |
| GloVe, 4,000 (64), 64 | 0.843, 198 | 0.832, **173** | 0.813, 168 |

Probing 16 of 64 groups makes single queries 1.1-2.1x faster, most at low nprobe and with many lists
(GloVe at nprobe 64 gains only 1.14x), and costs at most 0.011 in recall; 16 of 128 groups cost up to
0.022 on GloVe, and 8 groups up to 0.03. On the GPU, the coarse selection needs few threads per query:
with k = nprobe = 64, 512 threads leave 32,768 survivors per query, and 32 threads cut the selection
from 14 ms to about 1 ms.

## IVF: scans and FAISS (`ivf_cpu.py`, `fp16_lists.py --faiss`)

siann's IVFs and FAISS's IVF share the same 4,000 lists, so their recall at each nprobe is identical;
they differ in how they scan. **Query-major:** each query walks its own lists (on the GPU, 64 threads
per query; in Rust, one thread per query). **List-major:** queries grouped by list, each list read once
for all of its queries (on the GPU, one threadgroup per list, 16 queries at a time; in Rust, 4 queries
at a time with the exact kernel). Pass sizes such as the batch size at run time, not as compile-time
constants, or every new batch size recompiles the kernels.

**float32 lists, against FAISS's HNSW with float32 vectors** (M = 32), at the nearest recall; queries
per second (`ivf_sift_l2.txt`, `ivf_sift_l1.txt`, `ivf_glove_l2.txt`):

| Dataset | Recall: IVF / HNSW | Single queries: Rust IVF / HNSW | Batches of 100: best IVF / HNSW | 10,000: GPU IVF / Rust list-major / HNSW |
| --- | --- | ---: | ---: | ---: |
| SIFT L2 | 0.925 / 0.937 | 7,862 / 11,330 | 33,313 (GPU) / 95,579 | 199,112 / 71,867 / 119,917 |
| SIFT L2 | 0.973 / 0.978 | 4,682 / 6,389 | 25,193 (GPU) / 58,470 | 141,078 / 40,433 / 69,749 |
| SIFT L2 | 0.993 / 0.993 | 2,614 / 3,287 | 12,783 (GPU) / 32,131 | 44,501 / 22,089 / 38,921 |
| SIFT L1 | 0.904 / 0.898 | 7,528 / 8,383 | 26,722 (GPU) / 80,580 | 150,092 / 73,073 / 101,050 |
| SIFT L1 | 0.960 / 0.957 | 4,418 / 4,806 | 19,609 (GPU) / 48,898 | 102,057 / 41,344 / 58,297 |
| SIFT L1 | 0.988 / 0.985 | 2,369 / 2,647 | 12,497 (Rust list-major) / 28,248 | 38,694 / 22,536 / 32,515 |
| GloVe | 0.780 / 0.781 | 8,959 / 5,402 | 34,337 (GPU) / 47,519 | 206,607 / 76,521 / 56,891 |
| GloVe | 0.839 / 0.845 | 5,314 / 3,005 | 27,062 (GPU) / 27,960 | 146,998 / 42,512 / 32,304 |
| GloVe | 0.890 / 0.892 | 3,074 / 1,623 | 13,800 (GPU) / 16,004 | 46,409 / 23,159 / 17,721 |

**Compressed lists, against FAISS's compressed HNSW** (`IndexHNSWSQ`; see "List storage"), the
strongest alternative, at the nearest recall (`faiss_sq_*_l2.txt`):

| Dataset, recall | Batch | siann | FAISS HNSW, best storage |
| --- | --- | ---: | ---: |
| SIFT, 0.973 / 0.978 | 100 | GPU uint8 51,380; Rust uint8 34,223 | **uint8 109,965** |
| SIFT, 0.973 / 0.978 | 10,000 | GPU uint8 145,814 | uint8 152,655 |
| SIFT, 0.840 / 0.862 | 10,000 | **GPU uint8 571,146** | uint8 457,279 |
| GloVe, 0.839 / 0.843 | 100 | GPU float16 36,832; Rust float16 24,632 | float16 34,332 |
| GloVe, 0.711 / 0.704 | 100 | GPU float16 96,566 | float16 95,080 |
| GloVe, 0.839 / 0.843 | 10,000 | **GPU float16 151,665** | float16 40,015 |
| GloVe, 0.839 / 0.845 | 1 (one thread each) | **Rust float32 4,975** | float32 2,770; float16 3,555 |

What follows from these:

- **Large batches belong to the GPU's list-major scan.** It is 2.2-3.5x the GPU's query-major scan on
  large batches; against HNSW it leads by 3.8x on GloVe (compressed) and draws on SIFT at recall 0.97
  (1.25x at 0.84; 1.1-2.0x against float32 HNSW). HNSW must walk much further through its graph on
  GloVe for the same recall, while IVF's dense scans do not care.
- **Batches of about 100 favor HNSW on SIFT** (about 2x, compressed or not) and draw on GloVe. The
  query-major scan is memory-bound there: every query reads its own lists (about 7.6 MB each at
  nprobe 64 on GloVe, some 115 GB/s in all, so 16 threads give only 2.3-3.7 times one thread's rate),
  while HNSW reads far less per query.
- **Compression helps HNSW as much as the IVF:** uint8 makes HNSW 1.6-2.3x as fast on SIFT at every
  batch size (ef 64: 58,970 to 109,965 in batches of 100).
- **Against FAISS's own IVF on the same lists:** Rust's query-major scan is 1.3-6.6x faster on single
  queries (most at low nprobe, where FAISS's fixed cost per query dominates: SIFT L2 at nprobe 4, 20,491
  against 3,737 per second) and level in batches of 100 (both memory-bound; 1.3-1.5x at nprobe 4).
  Rust's list-major scan is 2.5-3.2x FAISS's IVF in batches of 10,000 with float32 lists, which FAISS
  scans query by query, level with uint8 lists (37,690 against 38,997 at nprobe 64), and 1.0-1.4x with
  float16.
- **On the CPU, the query-major scan suits single queries and the list-major scan large batches:**
  list-major is slower for single queries (it pays for grouping), wins in batches of 100 from nprobe 64
  (32 on SIFT L1), and always wins at 10,000.

## IVF: list storage (`fp16_lists.py`, `single_query.py`, `faiss_sq8_query.py`)

Lists can be stored in less than float32, with distances still computed in float32 or, for bytes,
exact integers. **float16** (half the bytes) rounds each coordinate: exact on SIFT (small integers),
at most 2.4e-4 on GloVe, moving recall by at most 0.001. **uint8** (a quarter, plus a row-major copy,
so two bytes per coordinate in all) is lossless for data whose coordinates are integers 0-255, like
SIFT's; both IVFs refuse it for other data. Speed over float32 lists on the same path, L2, same run
(`fp16_*_l2.txt`, `uint8_sift_l2.txt`):

| Path | float16, SIFT and GloVe, nprobe 16 / 64 | uint8, SIFT, nprobe 16 / 64 |
| --- | ---: | ---: |
| Rust, batches of 100 | 1.58-1.61x / 1.66-1.76x | **2.16x / 2.85x** |
| GPU, batches of 100 | 1.29-1.43x / 1.36-1.60x | **1.66x / 2.02x** |
| GPU, 10,000 (list-major) | 1.04-1.07x / 1.02-1.06x | 1.08x / 1.09x |
| Rust, 10,000 (list-major) | 0.89-0.91x / 0.87-0.99x | 0.96x / 1.11x |
| Rust, single queries (1 thread) | 0.62-0.63x / 0.51-0.55x | see the integer path below |

- **Smaller lists pay where the scan is memory-bound** (batches of 100, and split single queries) and
  cost where it is not (a single query on one thread, and the list-major scans, which already read each
  list once for many queries). So no storage is best at every batch size.
- **Byte lists with integer queries take an integer path,** exact: absolute differences of bytes,
  accumulated in integers, over the row-major copy. A query whose coordinates are not all integers 0-255
  takes the float path, so nothing is rounded; the integer distances, below 2^24 for up to 258
  dimensions, equal the float path's, and `check_kernels.py` requires the same indices as a float32
  index on the same data. SIFT L2, single queries, median microseconds (`int_single_sift_l2.txt`,
  `single_sift_l2_direct.txt`):

  | | nprobe 16 | nprobe 64 | nprobe 128 |
  | --- | ---: | ---: | ---: |
  | uint8, float path, 1 thread | | 263 | 478 |
  | **uint8, integer path, 1 thread** | **77** | **147** | **244** |
  | uint8, integer path, 4 threads | 52 | 76 | 111 |
  | FAISS IVF uint8, 1 thread | 62 | 136 | 249 |
  | siann float32 lists, 1 thread | 103 | 237 | 405 |

  The integer path is 1.8x the float path on one thread and level with FAISS's IVF; uint8 single
  queries reach 6,254 per second at nprobe 64 against float32's 3,915. It needs each vector's bytes
  contiguous: in the blocked layout a dimension's 8 bytes fill half a NEON register, and integer kernels
  there are no faster than the float path (259 against 263 microseconds at nprobe 64, written with the
  `wide` crate's vector types; with plain loops the compiler gathers bytes one lane at a time, since
  integer sums may be reordered). Row by row, a plain loop compiles to 16-byte absolute differences
  (`uabd`) and, for L2, the dot-product instruction (`udot`).
- **FAISS's uint8 lists round the query** (`QT_8bit_direct`): with queries offset by 0.3 from integers,
  its distances equal the rounded query's exactly (84,810, where the float query's is 84,802.08;
  `faiss_sq8_query.py`). That is exact on SIFT, whose queries are integers, and approximate otherwise.
- **Widening float16 in Rust:** Rust has no stable float16 type, so the lists hold the bits (`u16`),
  widened exactly in safe code (sign, exponent and mantissa moved into float32's positions, then a
  multiply by 2^112; checked against NumPy on all 63,488 finite float16 values). Widen a block into a
  buffer in one flat loop, then run the float32 code: widening inside the distance loops, or over
  nested `[u16; 8]` arrays, makes the compiler load the halves one lane at a time (1.3-1.9x slower on
  SIFT at nprobe 64). Bytes, in the float path, convert in registers inside the loop (8% faster than a
  buffer).
- **FAISS's compressed indexes need dimensions in multiples of 8:** at 100 (GloVe) its float16 search
  is 2-6x slower than its own float32 (IVF single queries at nprobe 64: 786 against 4,452 per second;
  flat float16 search on Gaussian data 0.118 s at 100 dimensions, 0.016 s at 104, float32 0.024 s at
  both). Zero-padding to 104 leaves L2 distances unchanged, and the comparisons here pad; padding costs
  FAISS's float32 IVF about 15% on GloVe, so for float32 the better of padded and unpadded is used.

## IVF: single queries (`single_query.py`, `coarse_step.py`)

A single query on one thread is not limited by memory traffic. It can be split
(`IvfIndex.search(..., split=n)`): the coarse step in n chunks and the probed lists in n contiguous
runs, as n tasks on a thread pool of n threads of its own, the runs' best k merged in probe order, so
that the result is the one-thread result, index for index, ties included. Latency, median
microseconds; HNSW on one thread, as it answers a single query (`faiss_single_*_l2.txt`,
`single_*_l2.txt`, `int_*_sift_l2.txt`):

| Dataset, recall (IVF / HNSW) | siann, one thread | siann, best split (storage, threads) | FAISS HNSW, float32 | FAISS HNSW, compressed |
| --- | ---: | ---: | ---: | ---: |
| SIFT, 0.83-0.85 / 0.842 | 42 (uint8, 8,000 lists, 8 groups; 0.851) | 64 (float32, 4; 0.828) | 53 | **29** |
| SIFT, 0.92 / 0.925 | 66 (uint8, 16 of 64 groups) | | 88 | **51** |
| SIFT, 0.972 / 0.974 | 147 (uint8) | **76** (uint8, 4) | 161 | 94 |
| SIFT, 0.994 / 0.992 | 244 (uint8) | **111** (uint8, 4) | 299 | 179 |
| GloVe, 0.712 / 0.716 | **58** (float32, 16 of 64 groups) | 60 (float32, 4) | 102 | 89 |
| GloVe, 0.83-0.84 / 0.852 | 173 (float32, 16 of 64 groups) | **111** (float32, 4) | 338 | 285 |
| GloVe, 0.894 / 0.894 | 338 (float32) | **174** (float32, 4) | 665 | 544 |

- **On GloVe, siann's IVF answers single queries faster than HNSW,** 1.5-1.65x compressed HNSW on one
  thread (with the two-level coarse step) and 1.5-3.1x with 4 threads.
- **On SIFT, compressed HNSW is faster on one thread** (1.3-1.45x at recall 0.85-0.92), and siann leads
  from recall 0.97 with 4 threads.
- **Splitting buys latency, not throughput:** a split query occupies 4-8 cores where HNSW uses one.
  Under many concurrent queries, one query per thread is the fair comparison.
- **Four threads are best:** 1.3-2.0x one thread's latency at every nprobe for float32 (most at high
  nprobe); eight help only at nprobe 128 with compressed lists, and at low nprobe eight are slower than
  one (72 against 66 microseconds at nprobe 4), since starting and joining tasks costs more than the
  little scanning saves. Split queries make compressed lists pay, since the split scan is
  memory-bound.
- **Give split queries a pool of their own.** On rayon's global pool (all 16 cores), injecting a query
  wakes the pool's sleeping threads, a floor of about 100 microseconds per query (nprobe 4: 103 against
  66 on one thread); fewer tasks on that pool do not help (4 tasks: 130); a pool of 4 threads of its
  own takes 45. FAISS's own split (`parallel_mode = 1`) has the same floor: its IVF answers one query
  faster on one thread at nprobe 16 and below.

## IVF on PyTorch MPS (`ivf_torch.py`)

The GPU IVF's Metal code, run through `torch.mps.compile_shader`, with the same coarse step, merge and
centroids (run by `ivf_cpu.py` in a process of its own). SIFT L2, queries per second (the PyTorch rows
of `ivf_*.txt`):

| nprobe | Recall (both) | Single: MLX / PyTorch | Batches of 100 | Batch of 10,000 |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.571 | 2,356 / 1,541 | 155,380 / 110,696 | 1,184,466 / 1,149,359 |
| 16 | 0.840 | 1,287 / 936 | 75,859 / 62,148 | 517,798 / 512,256 |
| 32 | 0.925 | 490 / 442 | 33,313 / 30,719 | 199,112 / 201,188 |
| 64 | 0.973 | 455 / 416 | 25,193 / 23,737 | 141,078 / 143,308 |
| 128 | 0.993 | 220 / 214 | 12,783 / 12,680 | 44,501 / 45,591 |

Large batches are level (within 3% on SIFT L2, SIFT L1 and GloVe), since the kernels do the work.
Single queries and batches of 100 are up to a third slower in PyTorch at low nprobe: a fixed cost of
about 0.2 ms per call.

## Inner product (`backends.py --metrics ip`, `ivf_cpu.py --metric ip`, `ip_reduction.py`)

siann ranks inner product by -q.x, so that, as for the distances, smaller is nearer. The blocked
layouts pad with infinity, harmless for the distances but the best possible inner product (or NaN
against a zero coordinate), so exact search skips padding slots by index. On raw GloVe-100, whose
vector lengths run from 0.04 to 18.7, exact search is 4.1-17.7x FAISS on the GPU (above), but **IVF
fails** (`ip_ivf_glove_raw.txt`, `ip_reduction_glove_raw.txt`):

- **Assigned by inner product** (as FAISS's `METRIC_INNER_PRODUCT` IVF assigns), the lists are badly
  unbalanced: centroids pointing towards long vectors attract most of the data (the largest list holds
  35,725 vectors, the 10 largest 21% of them, and 39% of the lists are empty; by L2: 2,066, 2%, none).
  Recall stays at 0.23-0.64 up to nprobe 128, for siann's and FAISS's IVF alike, at 1,700 queries per
  second in batches of 100 at recall 0.64.
- **Reduced to L2** (each base vector extended by sqrt(M^2 - |x|^2) to the longest length M, each query
  by 0, so that L2 ranks as inner product), the lists balance but recall falls to 0.03-0.34, and FAISS's
  HNSW on the extended vectors does as badly (0.16-0.34): most vectors are far shorter than the longest,
  so the added coordinate dominates the distances.
- **FAISS's HNSW with inner product** reaches recall 0.888 at 14,224 queries per second in batches of
  100 and 15,275 in batches of 10,000 (efSearch 256).
- **Exact search on the GPU is the answer there:** recall 1.0 at about 12,500 queries per second in
  batches of 100 (0.88x HNSW at 0.888) and about 20,700 in batches of 10,000 (1.36x). Vectors of similar
  length are close to cosine, where the IVF works (not measured separately).

## Approaches that do not pay

- **LSH for dense vectors** (`lsh_l1.py`). Cauchy (1-stable) LSH for L1 on SIFT1M reaches recall 0.72
  only by re-ranking 10.5% of the database (w = 2 r10, m = 4, L = 32), where exact search on the GPU
  already compares everything faster. Real data has low contrast: at w = r10 (the distance to the 10th
  neighbor), a neighbor collides with probability about 0.28 and a point twice as far about 0.26.
  FAISS's HNSW handles L1 well (recall 0.947 at 56,000 queries per second, 0.990 at 17,000).
- **BLAS in the CPU IVF's list-major scan** (`blas_ivf.py`): one multiply per list, its queries against
  its vectors, the products ranked in Rust 256 lists at a time, on lists identical to the Rust IVF's.
  Queries per second (`blas_ivf_*.txt`):

  | Dataset, nprobe | Rust IVF, 100 / 10,000 | BLAS IVF, best workers, 100 / 10,000 |
  | --- | ---: | ---: |
  | SIFT, 16 | 42,969 / 113,248 | 8,135 / 61,120 |
  | SIFT, 64 | 12,429 / 39,640 | 1,950 / 31,394 |
  | GloVe, 16 | 48,827 / 126,068 | 7,565 / 62,437 |
  | GloVe, 64 | 14,481 / 44,571 | 2,007 / 31,216 |

  The multiplies are small (about 160 queries by 250 vectors at nprobe 64), so overheads dominate: on
  SIFT at nprobe 64 with 4 workers, a batch of 10,000 spends 0.116 s on the coarse step and grouping,
  0.120 s on the products and 0.100 s on ranking, against 0.25 s for Rust's whole search. Even with the
  coarse step and grouping free it would at best draw level. BLAS pays for exact L2, where the
  multiplies are large.
- **The integer path in the list-major scan** (large batches): in a batch of 10,000 on SIFT at nprobe
  64, Rust's list-major scan does 37,339 queries per second with uint8 lists, against the GPU IVF's
  147,384 (`int_uint8_sift_l2.txt`) and compressed HNSW's 152,655 on the CPU. Even the single-query
  path's gain (1.6-1.8x) would leave it far behind both, for integer byte data only, at the cost of a
  new kernel.
- **Fewer IVF lists** to cheapen the coarse step, and **splitting a single query on the global thread
  pool**: see "The coarse step" and "Single queries" above.
