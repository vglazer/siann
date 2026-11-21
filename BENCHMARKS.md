# Benchmarks

*Part of the [siann documentation](README.md#documentation).*

For each kind of similarity search on a Mac, which path is fastest, and by how much? Every comparison
is against FAISS's strongest option for the case, its compressed indexes included, with exact search
returning exactly FAISS's neighbors and approximate search compared at matched recall. Each section
names the scripts that reproduce it; they are in [benchmarks/](benchmarks/), and their raw output is in
[benchmarks/results/](benchmarks/results/). The [README](README.md#which-to-use-and-how-it-compares) sums up the
comparison, and [IMPLEMENTATION](IMPLEMENTATION.md) explains where the speed comes from.

## Setup

- **Machine:** Apple M3 Max (12 performance and 4 efficiency cores, 64 GB, 400 GB/s), 16 CPU threads.
- **Software:** FAISS 1.16 (CPU), MLX 0.32.4, PyTorch 2.15.0, NumPy with Apple's Accelerate.
- **Data:** SIFT1M, GloVe-100 (normalized, so that L2 ranks as cosine; `glove-raw` unnormalized, for
  inner product) and Gaussian vectors ([DATASETS](DATASETS.md)). k = 10 throughout.
- **Correctness:** every kernel and index is checked against NumPy's exact search (`check_kernels.py`,
  and `ivf_gpu.py --check` for MLX's IVF). Exact search agrees with FAISS's flat index in every
  configuration.
- **Recall** at 10 counts a returned neighbor as correct when it is no farther than the exact 10th
  nearest, so that ties broken differently (common on SIFT's integer data) are not misses
  (`agreement` in `backends.py`).
- **Timing:** batches are the median of three runs after a warm-up on the full query set; single-query
  throughput is one pass of 1,000 calls; latency is the median of 1,000 single calls after a warm-up.
  Repeated runs of the same configuration differ by up to 15%, so smaller differences are noise.
- **FAISS at its best:** its flat L2 needs `faiss.cvar.distance_compute_blas_threshold = 20` (see
  "FAISS's BLAS threshold"), and its compressed indexes need dimensions in multiples of 8 (see "List
  storage"); every comparison sets both. FAISS's Metal GPU path has a section of its own, its exact
  search run in calls of at most 1,000 queries ("FAISS's Metal path on the GPU").
- **On the GPU,** the strongest alternative is [mlxvs](https://github.com/mikepapadim/mlxvs) 0.16.1,
  which runs NVIDIA cuVS's methods on MLX (CAGRA graphs, IVF variants, exact search; L2, cosine and inner
  product, not L1 or L-infinity), measured on the same machine (see "Against mlxvs on the GPU").

## Which to use

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

What decides it:

- **Batch size.** The GPU pays a fixed cost per call and needs many queries to fill it; a graph index
  on the CPU (HNSW) touches far fewer vectors per query and wins small batches on SIFT-like data. A
  single query is a contest of per-query work, where splitting it over a few threads trades cores for
  latency.
- **The data.** HNSW touches 7-9 times fewer vectors than an IVF on SIFT but only 4-5 times fewer on
  GloVe, where its scattered reads cost it the difference (see "Work per query").
- **Recall.** HNSW's work grows faster with recall than an IVF's, so on SIFT the GPU IVF overtakes it in
  batches of 100 at recall 0.99.
- **The metric,** for exact search. L1 and L-infinity have no matrix-multiply form, so FAISS computes
  them with plain loops, where siann runs kernels laid out like a matrix multiply; L2, cosine and inner
  product are matrix multiplies, which FAISS runs through BLAS too.
- **Not the framework.** A custom Metal kernel runs at the same speed in MLX and PyTorch, and built-in
  operations lose to FAISS's CPU everywhere (see "Framework built-in operations").

## Exact search on the GPU

*Reproduce with `backends.py --dataset sift|glove`, `backends.py --metrics l1,linf`,
`backends.py --dataset glove-raw --metrics ip`, `profile_l2.py`.*

1,000 queries; the speed-up over FAISS's flat index on the CPU:

| Backend | SIFT L2 | SIFT L1 | GloVe L2 (cosine) | GloVe L1 |
| --- | ---: | ---: | ---: | ---: |
| FAISS flat (CPU), time | 0.405 s | 2.07 s | 0.431 s | 1.65 s |
| siann, MLX | 8.7x | 25x | 8.2x | 18.5x |
| siann, PyTorch MPS | 4.0x | 25x | 3.9x | 18.4x |
| siann, CPU (Rust for L1; BLAS for L2) | 1.7x | 4.3x | 1.9x | 3.9x |
| Best framework built-in operations (GPU) | 0.65x | 0.48x | 0.54x | 0.43x |
| Best framework built-in operations (CPU) | 0.35x | 0.57x | | |

- **Gaussian vectors** (1,000,000 of 128 dimensions, 1,000 queries): FAISS L2 0.39 s and L1 1.82 s;
  MLX 0.046 s (8.4x) and 0.082 s (22x); with 10,000 queries, 7.6x and 21x.
- **L-infinity** (200,000 Gaussian vectors; `linf_gaussian.txt`): FAISS 0.972 s, three times its own
  L1 (0.334 s); MLX 49.0x and PyTorch 47.2x, where for L1 they are 17.4x and 16.6x. siann's tiled
  kernel takes the same time for both metrics (0.019-0.021 s).
- **Inner product** (raw GloVe; `ip_exact_glove_raw*.txt`): MLX 17.7x FAISS's `IndexFlatIP` on 100
  queries (0.008 against 0.136 s), 5.9x on 1,000 (0.049 against 0.287 s), 4.1x on 10,000 (0.482 against
  1.985 s).
- **Small collections:** at 200,000 vectors the GPU barely pays for L2 (FAISS 0.079 s against MLX
  0.099 s, without `addmm`) and still does for L1 (0.300 against 0.106 s).
- **PyTorch's L2 gap is its matrix multiply:** 256 x 1M x 128 takes 6.7 ms in MLX and 14.2 ms in
  PyTorch (with `addmm`, 6.9 and 18.1 ms). The custom kernels run at the same speed in both (L1 0.082
  s; the top-k kernel 4.2 against 3.8 ms per tile).

## Exact search on the CPU

*Reproduce with `backends.py --backends faiss-cpu,siann-rust-cpu`, `blas_l2.py`, `rust_cpu_sweep.py`.*

**L1 and L-infinity: siann's Rust kernel.**

| | SIFT L1 | GloVe L1 | Gaussian L1 (1M) | Gaussian L-infinity (200,000) |
| --- | ---: | ---: | ---: | ---: |
| FAISS flat | 2.07 s | 1.65 s | 1.79 s | 0.972 s |
| siann Rust | 4.3x | 3.9x | 3.8x (0.471 s) | 7.3x (0.133 s) |

FAISS's L1 does about 70 G absolute differences per second, siann's about 270 G. Queries per thread task
(8 to 64) make no measurable difference; the default is 32. Building the blocked layout takes 0.10-0.13
s for 1M x 128.

**L2, cosine and inner product: NumPy's BLAS and a Rust top-k** (`blas_l2_*.txt`,
`ip_blas_glove_raw.txt`):

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
  power of two. The sweep is not smooth (128 x 65,536: 0.93x on SIFT, 1.56x on GloVe), so the default
  is a starting point to sweep, not a fitted formula.
- **The multiply is most of the time** (0.14-0.16 s of 0.22-0.24 s for 1,000 queries).
- **FAISS's inner product is cheaper than its L2** (0.29 against 0.46 s on GloVe, no norms to add),
  leaving less to win on 1,000 queries.
- **Batches below 100 queries** are outside the measurements; there a multiply has little to reuse.

### FAISS's BLAS threshold

*Reproduce with `faiss_blas.py`.*

FAISS 1.16 sets `faiss.cvar.distance_compute_blas_threshold` to 128,000, so with fewer queries its flat
L2 below 128 dimensions skips BLAS for a path 4-7x slower:

| Dimensions (500,000 Gaussian vectors, 1,000 queries) | 32 | 64 | 96 | 112 | 127 | 128 | 192 | 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FAISS flat L2, default | 0.184 s | 0.453 | 0.803 | 1.002 | 1.209 | 0.177 | 0.200 | 0.230 |
| Threshold 20 | 0.156 s | 0.185 | 0.159 | 0.166 | 0.172 | 0.170 | 0.205 | 0.232 |

On GloVe-100, 2.90 s by default and 0.419 s at 20, with the same neighbors. L1 and 128 dimensions are
unaffected; HNSW does not use flat search.

## Framework built-in operations

*Reproduce with `backends.py`.*

What a user of each framework would write: L2 as norms minus twice a matrix multiply, L1 as a
broadcast absolute difference (MLX) or `torch.cdist` (PyTorch), then the framework's top-k.
1,000,000 x 128 Gaussian vectors, 1,000 queries:

| Backend | L2 | vs FAISS | L1 | vs FAISS |
| --- | ---: | ---: | ---: | ---: |
| FAISS flat (CPU) | 0.39 s | 1.00x | 1.82 s | 1.00x |
| MLX built-in (GPU) | 0.65 s | 0.60x | 7.20 s | 0.25x |
| MLX compiled (GPU) | 0.63 s | 0.61x | 4.37 s | 0.42x |
| PyTorch built-in (MPS) | 0.70 s | 0.55x | 4.72 s | 0.39x |
| PyTorch built-in (CPU) | 1.10 s | 0.35x | 4.52 s | 0.40x |
| MLX built-in (CPU) | 6.8 s | 0.06x | 32.5 s | 0.06x |

The ratios hold on SIFT1M and GloVe-100 (0.34-0.65x on the GPU). Built-in operations lose to FAISS's CPU
everywhere, on the GPU too: the GPU's advantage comes entirely from custom kernels.

## Approximate search: batches

*Reproduce with `ivf_cpu.py --dataset sift|glove --metric l2|l1`, `fp16_lists.py --faiss`,
`fine_lists.py`.*

All IVFs, siann's and FAISS's, share the same 4,000 k-means lists, so their recall at each nprobe is
identical. The GPU IVF runs the query-major scan below 1,000 queries and the list-major scan above.
Queries per second.

**float32 lists, against FAISS's HNSW (M = 32) on float32 vectors**, at the nearest recall
(`ivf_sift_l2.txt`, `ivf_sift_l1.txt`, `ivf_glove_l2.txt`):

| Dataset | Recall: IVF / HNSW | Single queries: Rust IVF / HNSW | Batches of 100: GPU IVF / HNSW | 10,000: GPU IVF / Rust list-major / HNSW |
| --- | --- | ---: | ---: | ---: |
| SIFT L2 | 0.925 / 0.937 | 7,228 / 9,565 | 57,943 / 93,668 | 291,416 / 70,969 / 120,988 |
| SIFT L2 | 0.973 / 0.978 | 4,453 / 6,464 | 34,630 / 53,646 | 181,901 / 38,391 / 68,227 |
| SIFT L2 | 0.993 / 0.993 | 2,547 / 3,279 | 18,996 / 28,686 | 104,438 / 21,839 / 38,725 |
| SIFT L1 | 0.904 / 0.898 | 6,757 / 8,336 | 45,784 / 80,028 | 200,500 / 72,870 / 101,592 |
| SIFT L1 | 0.960 / 0.957 | 4,084 / 4,457 | 27,081 / 48,911 | 123,855 / 40,794 / 58,059 |
| SIFT L1 | 0.988 / 0.985 | 2,410 / 2,620 | 15,753 / 28,104 | 75,288 / 21,922 / 32,206 |
| GloVe | 0.780 / 0.781 | 8,186 / 5,370 | 61,870 / 47,436 | 303,593 / 77,181 / 56,904 |
| GloVe | 0.839 / 0.845 | 5,057 / 2,989 | 38,744 / 28,088 | 191,532 / 41,521 / 32,114 |
| GloVe | 0.890 / 0.892 | 2,934 / 1,614 | 22,197 / 15,853 | 113,440 / 22,981 / 17,637 |

**Compressed lists, against FAISS's compressed HNSW** (`IndexHNSWSQ`, in the same storage as siann's
lists), the strongest alternative, at the nearest recall (`faiss_sq_sift_l2.txt`,
`faiss_sq_glove_l2.txt`):

| Dataset, recall | Batch | siann | FAISS HNSW, best storage |
| --- | --- | ---: | ---: |
| SIFT, 0.973 / 0.978 | 100 | GPU uint8 93,502; Rust uint8 37,879 | **uint8 114,125** |
| SIFT, 0.973 / 0.978 | 10,000 | **GPU uint8 199,347** | uint8 153,710 |
| SIFT, 0.840 / 0.862 | 10,000 | **GPU uint8 566,410** | uint8 479,703 |
| GloVe, 0.839 / 0.843 | 100 | **GPU float16 60,559**; Rust float16 24,357 | float16 34,348 |
| GloVe, 0.711 / 0.704 | 100 | **GPU float16 118,622** | float16 95,705 |
| GloVe, 0.839 / 0.843 | 10,000 | **GPU float16 199,030** | float16 40,092 |

**Batches of 100 on SIFT, by recall,** against uint8 HNSW in the same run (uint8 lists;
`fine_lists_sift.txt`):

| Recall: siann / HNSW | siann GPU IVF (lists) | FAISS HNSW, uint8 | siann / HNSW |
| --- | ---: | ---: | ---: |
| 0.828 / 0.842 | 167,361 (4,000) | 251,083 | 0.67x |
| 0.925 / 0.925 | 118,810 (4,000) | 175,348 | 0.68x |
| 0.973 / 0.978 | 91,706 (4,000) | 109,788 | 0.84x |
| 0.993 / 0.993 | 63,130 (4,000) | 63,698 | 0.99x |
| 0.997 / 0.998 | 56,499 (8,000) | 34,124 | **1.66x** |

- **Large batches belong to the GPU's list-major scan:** 2.4-2.7x float32 HNSW on SIFT L2, 2.0-2.3x on
  SIFT L1 and 5.3-6.4x on GloVe; against compressed HNSW, 1.2-1.3x on SIFT and 5.0x on GloVe.
- **Batches of about 100 favor the GPU on GloVe** (1.3-1.4x float32 HNSW, 1.2-1.8x compressed). **On
  SIFT they favor HNSW** below recall 0.99: 1.2-1.5x with uint8 for both; level at 0.99; siann ahead
  above it, with 8,000 lists.
- **Compression helps HNSW as much as the IVF:** uint8 makes HNSW 1.6-2.3x as fast on SIFT at every
  batch size (ef 64: 60,424 to 114,125 queries per second in batches of 100).
- **Against FAISS's own IVF on the same lists:** Rust's query-major scan is faster on single queries,
  most at low nprobe, where FAISS's fixed cost per query dominates (SIFT L2 at nprobe 4: 16,705 against
  3,720 per second; at 128: 2,547 against 1,224), and level in batches of 100, where both are
  memory-bound. Rust's list-major scan is 2.2-4.0x FAISS's IVF in batches of 10,000 with float32 lists
  from nprobe 16 (1.9x at nprobe 4), level with uint8 lists (38,567 against 40,490 at nprobe 64) and
  about 1.4x with float16.
- **On the CPU, the query-major scan suits single queries and the list-major scan large batches:**
  list-major is slower for single queries (it pays for grouping), wins in batches of 100 from nprobe 64
  (32 on SIFT L1), and always wins at 10,000.

### The IVF's k-means

*Reproduce with `kmeans_check.py --dataset sift|glove|music100` and `kmeans_check.py --dataset dbpedia
--base 100000 --nlist 1000 --queries 1000`.*

siann trains the IVF's centroids with its own k-means (`siann.kmeans`: 10 iterations of Lloyd's algorithm
on 100 sample vectors per list, nearest centroids found by the exact kernels on the GPU). The measurements
on this page use lists from FAISS's k-means with the same settings; siann's give the same recall within
the spread between seeds, and better on Music-100. Recall@10 by nprobe, GPU IVF, list-major scan, every
query (`kmeans_check_*.txt`):

| Dataset | k-means | nprobe 8 | 16 | 32 | 64 | 128 | Training |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SIFT | FAISS / siann | 0.716 / 0.718 | 0.840 / 0.839 | 0.925 / 0.924 | 0.973 / 0.974 | 0.993 / 0.993 | 12.9 / 3.5 s |
| GloVe | FAISS / siann | 0.631 / 0.631 | 0.712 / 0.712 | 0.780 / 0.780 | 0.839 / 0.839 | 0.890 / 0.889 | 13.1 / 2.9 s |
| Music-100 | FAISS / siann | 0.850 / **0.866** | 0.911 / **0.922** | 0.948 / **0.957** | 0.967 / **0.976** | 0.978 / **0.987** | 12.4 / 2.9 s |
| DBpedia, 100,000 (1,000 lists) | FAISS / siann | 0.847 / 0.850 | 0.902 / 0.904 | 0.943 / 0.945 | 0.969 / 0.972 | 0.987 / 0.988 | 2.0 / 4.5 s |

A second seed moves siann's recall by up to 0.004. Training takes a third to a quarter of FAISS's time on a
million vectors, but more than twice as long at 1,536 dimensions with 1,000 lists, where finding each
vector's one nearest centroid goes through the top-k selection built for many.

### Where a GPU call's time goes

*Reproduce with `gpu_small_batch.py`, `gpu_select_ab.py`.*

A query-major call for 100 queries has three stages: the coarse step (distances to the 4,000
centroids, then each query's nprobe nearest lists), the scan and the merge. SIFT, uint8 lists, nprobe
64, per call, each stage forced to finish before the next, so each includes a synchronization of about
0.175 ms (`gpu_small_batch_sift.txt`):

| Stage | Time per call |
| --- | ---: |
| Coarse distances (one fused multiply) | 0.224 ms |
| Choosing the lists: two-stage ordered selection, 32 threads per query | 1.015 ms |
| Choosing the lists: MLX's `argpartition` (unordered) | **0.301 ms** |
| Scan, 64 threads per query | 1.139 ms (202 GB/s) |
| Scan, 128 / 256 threads per query | 0.880 / 0.810 ms (261 / 284 GB/s) |
| Merge, as separate MLX operations over every thread's candidates | 0.272 ms |

- **Choose the probes by the scan and nprobe.** The query-major scan uses `argpartition` at every
  nprobe. For the list-major scan, the ordered selection is faster up to nprobe 16 (938,134 against
  581,377 queries per second at nprobe 8 on SIFT) and `argpartition` from 32 (317,163 against 199,095)
  (`gpu_select_ab_crossover.txt`).
- **128 threads per query** give the fastest whole call (91,774 queries per second at nprobe 64,
  against 73,913 for 64 threads and 88,992 for 256), and are no slower than 64 at any measured nprobe.
- **Merging inside the scan kernel** (each query's 128 threads in one threadgroup, their sorted best k
  merged pairwise in seven steps) is 0-5% faster than the separate merge and writes only k results per
  query (`gpu_merge_ab.txt`).

### What a new batch size costs

*Reproduce with `compile_cost.py`.*

The GPU kernels read the sizes that change from call to call (how many queries, how many probes, a row's
length) at run time, and compile in only those that size arrays in registers or threadgroup memory (k,
threads per query) or that a loop is unrolled for (the dimension). With the batch size compiled in, every
new batch size compiles the kernels again, costing the first call 26-74 ms (median) on top of a 1-4 ms
search; read at run time, a new batch size costs 0-3 ms. The first call with a new k still compiles once, 38-113 ms for an
IVF. Random vectors, 100,000 x 128, k = 10, nine batch sizes from 7 to 1,001 (`compile_cost_before.txt`,
`compile_cost_after.txt`). Warm searches run at the same speed either way: on SIFT1M, exact search and both
IVF scans, on both GPU backends, are within 3% of each other with the sizes compiled in or read at run time.

### Work per query

*Reproduce with `dist_counts.py`.*

Small batches are limited by memory traffic, so the work is the number of vectors a query touches:
for the IVF, the centroids compared plus the vectors in the probed lists; for HNSW, FAISS's own count
(`hnsw_stats.ndis`). The IVF with 16,000 lists and the two-level coarse step (a quarter of 250 groups
probed), against HNSW (M = 32) at the nearest recall (`dist_counts_sift.txt`, `dist_counts_glove.txt`):

| Dataset, recall (IVF / HNSW) | IVF: distances, list KB | HNSW: distances, KB | HNSW touches fewer |
| --- | ---: | ---: | ---: |
| SIFT, 0.839 / 0.842 | 6,798, 301 | 764, 96 | 8.9x distances, 3.1x bytes |
| SIFT, 0.919 / 0.925 | 9,144, 595 | 1,184, 148 | 7.7x, 4.0x |
| SIFT, 0.967 / 0.974 | 13,736, 1,168 | 1,962, 245 | 7.0x, 4.8x |
| GloVe, 0.719 / 0.711 | 8,567, 1,085 | 1,568, 613 | 5.5x, 1.8x |
| GloVe, 0.832 / 0.856 | 15,948, 3,968 | 4,317, 1,686 | 3.7x, 2.4x |

HNSW touches far fewer vectors on both datasets, but each visit costs it more (scattered reads, each
step depending on the last), where an IVF streams contiguous lists. On GloVe its 4-5x fewer vectors are
cancelled by that cost; on SIFT, 7-9x fewer keep it ahead in small batches below recall 0.99. Finer
partitions cut the IVF's work (on SIFT at recall 0.97, 16,000 lists touch 13,736 vectors and 1,168 KB of
lists, 4,000 lists 21,934 and 2,242 KB), which pays on the GPU at high recall (8,000 lists, above) but
not on the CPU (next).

### Where a small batch's time goes on the CPU

*Reproduce with `batch_profile.py`; the profile with `profile_batch.py` under `samply`, summed by
`samply_top.py`.*

SIFT, uint8 lists with the integer path, the Rust IVF in batches of 100, queries per second
(`batch_profile_sift.txt`, `batch_profile_sift_gprobe.txt`, `batch_profile_samply.txt`):

| | 4,000 lists, nprobe 64 | 16,000 lists, nprobe 128 |
| --- | ---: | ---: |
| One thread | 8,871 | 7,615 |
| 8 threads | 40,373 (4.6x) | 38,475 (5.0x) |
| 12 threads (the performance cores) | 40,626 | 40,683 |
| 16 threads | 38,974 | 37,789 |
| nprobe 1 (the coarse step and one list) | 289,401 | 149,314 |
| k = 1 / 10 / 100 | 38,970 / 38,974 / 37,334 | 37,701 / 37,789 / 36,114 |
| Exact coarse step instead of two-level | 37,818 | 27,315 |

- **The list scan is the cost:** about 84% of the search, the coarse step about 9% and the top-k updates
  about 6%.
- **Throughput stops growing at about 8 threads,** at about 40,000 queries per second; the efficiency
  cores neither help nor hurt.
- **With 4,000 lists the scan is bandwidth-bound:** float32 lists (four times the bytes) run 2.9x slower
  than uint8, and at the plateau the scan streams about 88 GB/s. With 16,000 lists (about 63 vectors
  each) it stops at the same rate with half the bytes, which points to a fixed cost per list (the start
  of each list's stream) rather than bandwidth: 16,000 lists touch 1.6x fewer vectors than 4,000 at
  recall 0.97 and run at the same speed (37,878 against 38,798 queries per second;
  `fine_lists_sift.txt`).
- **The centroids cost little:** varying the groups the two-level step probes (16,000 lists) changes
  the centroid bytes per query 11x (381 to 4,125 KB) and throughput 1.4x (49,159 to 34,966), since the
  centroids stay in cache. A quarter of the groups is more than needed: 31 of 250 give recall 0.963 at
  41,664, against 0.967 at 38,735 for 62.

## Approximate search: single queries

*Reproduce with `single_query.py --splits 1,4 --faiss`, `coarse_step.py`.*

Latency, median microseconds; HNSW on one thread, as it answers a single query
(`faiss_single_*_l2.txt`, `single_*_l2.txt`, `int_*_sift_l2.txt`):

| Dataset, recall (IVF / HNSW) | siann, one thread | siann, best split (storage, threads) | FAISS HNSW, float32 | FAISS HNSW, compressed |
| --- | ---: | ---: | ---: | ---: |
| SIFT, 0.83-0.85 / 0.842 | 42 (uint8, 8,000 lists, 8 groups; 0.851) | 64 (float32, 4; 0.828) | 53 | **29** |
| SIFT, 0.92 / 0.925 | 66 (uint8, 16 of 64 groups) | | 88 | **51** |
| SIFT, 0.972 / 0.974 | 147 (uint8) | **76** (uint8, 4) | 161 | 94 |
| SIFT, 0.994 / 0.992 | 244 (uint8) | **111** (uint8, 4) | 299 | 179 |
| GloVe, 0.712 / 0.716 | **58** (float32, 16 of 64 groups) | 60 (float32, 4) | 102 | 89 |
| GloVe, 0.83-0.84 / 0.852 | 173 (float32, 16 of 64 groups) | **111** (float32, 4) | 338 | 285 |
| GloVe, 0.894 / 0.894 | 338 (float32) | **174** (float32, 4) | 665 | 544 |

- **On GloVe, siann's IVF answers single queries faster than HNSW:** 1.5-1.65x compressed HNSW on one
  thread (with the two-level coarse step), 1.5-3.1x with 4 threads.
- **On SIFT, compressed HNSW is faster on one thread** (1.3-1.45x at recall 0.85-0.92), and siann leads
  from recall 0.97 with 4 threads.
- **Splitting buys latency, not throughput:** a split query occupies 4 cores where HNSW uses one.
  Under many concurrent queries, one query per thread is the fair comparison.
- **Four threads are best:** 1.3-2.0x one thread's latency at every nprobe for float32 (most at high
  nprobe); eight help only at nprobe 128 with compressed lists, and at low nprobe eight are slower than
  one (72 against 66 microseconds at nprobe 4). Split queries make compressed lists pay, since the split
  scan is memory-bound.
- **On rayon's global pool** (all 16 cores), a split query pays about 100 microseconds to wake the
  pool's threads (nprobe 4: 103 against 66 on one thread; with 4 tasks, 130); on a pool of 4 threads of
  its own, 45. FAISS's own split (`parallel_mode = 1`) has the same floor: its IVF answers one query
  faster on one thread at nprobe 16 and below.

### The coarse step

*Reproduce with `coarse_step.py`.*

Comparing a query with every centroid is half of a single query at nprobe 16. Fitting single-query
latency (Rust IVF, float32 lists, one thread, SIFT L2) against nprobe from 1 to 16 gives the coarse
step as the intercept and a list's cost as the slope; the call itself takes about 1 microsecond
(`coarse_sift_l2.txt`):

| Lists | Coarse step | Per list | nprobe 16: recall, latency |
| ---: | ---: | ---: | ---: |
| 4,000 | **50 microseconds** | 2.7 | 0.828, 94 |
| 2,000 | 27 | 4.8 | 0.874, 104 |
| 1,000 | 16 | 8.8 | 0.924, 156 |

Fewer lists do not help: at matched recall, 1,000 lists (nprobe 8: 0.822 in 89 microseconds) are level
with 4,000 (nprobe 16: 0.828 in 94), and 4,000 win from recall 0.92. The two-level coarse step does
help (one thread, median microseconds; `coarse_two_level_*.txt`):

| Dataset, lists (groups), nprobe | Exact coarse step: recall, latency | 16 groups probed | 8 groups probed |
| --- | ---: | ---: | ---: |
| SIFT, 4,000 (64), 16 | 0.828, 96 | 0.828, **60** | 0.821, 54 |
| SIFT, 4,000 (64), 32 | 0.921, 141 | 0.920, **103** | 0.906, 98 |
| SIFT, 8,000 (128), 32 | 0.879, 152 | 0.876, **71** | 0.851, 65 |
| GloVe, 4,000 (64), 16 | 0.716, 89 | 0.712, **58** | 0.698, 52 |
| GloVe, 4,000 (64), 64 | 0.843, 198 | 0.832, **173** | 0.813, 168 |

Probing 16 of 64 groups makes single queries 1.1-2.1x faster, most at low nprobe and with many lists,
and costs at most 0.011 in recall; 16 of 128 groups cost up to 0.022 on GloVe, and 8 groups up to 0.03.

## List storage

*Reproduce with `fp16_lists.py --storages float32,float16,uint8 --faiss`, `single_query.py`,
`faiss_sq8_query.py`.*

Speed over float32 lists on the same path, L2, same run (`faiss_sq_sift_l2.txt`,
`faiss_sq_glove_l2.txt`):

| Path | float16, SIFT and GloVe, nprobe 16 / 64 | uint8, SIFT, nprobe 16 / 64 |
| --- | ---: | ---: |
| Rust, batches of 100 | 1.50-1.52x / 1.68-1.73x | **2.41x / 3.01x** |
| GPU, batches of 100 | 1.33-1.39x / 1.64-1.78x | **1.85x / 2.72x** |
| GPU, 10,000 (list-major) | 1.02-1.04x / 1.03-1.05x | 1.08x / 1.08x |
| Rust, 10,000 (list-major) | 0.89-0.90x / 0.85-0.89x | 0.96x / 0.98x |
| Rust, single queries (1 thread) | 0.65-0.68x / 0.54-0.57x | **1.23x / 1.55x** (the integer path) |

Smaller lists pay where the scan is memory-bound (batches of 100, and split single queries) and cost
where it is not: a single query on one thread, unless byte lists take the integer path, and the
list-major scans, which already read each list once for many queries. float16 moves recall by at most
0.001 (it is exact on SIFT); uint8 is lossless for integer data such as SIFT's.

**The integer path** (byte lists, integer queries), SIFT L2, single queries, median microseconds
(`int_single_sift_l2.txt`, `single_sift_l2_direct.txt`):

| | nprobe 16 | nprobe 64 | nprobe 128 |
| --- | ---: | ---: | ---: |
| uint8, float path, 1 thread | | 263 | 478 |
| **uint8, integer path, 1 thread** | **77** | **147** | **244** |
| uint8, integer path, 4 threads | 52 | 76 | 111 |
| FAISS IVF uint8, 1 thread | 62 | 136 | 249 |
| siann float32 lists, 1 thread | 103 | 237 | 405 |

The integer path is 1.8x the float path on one thread and level with FAISS's IVF; uint8 single queries
reach 6,254 per second at nprobe 64 against float32's 3,915. FAISS's uint8 lists (`QT_8bit_direct`)
round the query: with queries offset by 0.3 from integers, its distances equal the rounded query's
exactly (84,810, where the float query's is 84,802.08). That is exact on SIFT, whose queries are
integers, and approximate otherwise; siann's integer path applies only to integer queries.

**FAISS's compressed indexes need dimensions in multiples of 8:** at 100 (GloVe) its float16 search is
2-6x slower than its own float32 (IVF single queries at nprobe 64: 786 against 4,452 per second; flat
float16 search on Gaussian data 0.118 s at 100 dimensions, 0.016 s at 104, float32 0.024 s at both).
Zero-padding to 104 leaves L2 distances unchanged, and the comparisons pad; padding costs FAISS's float32
IVF about 15% on GloVe, so for float32 the better of padded and unpadded is used.

## Approximate search through siann.Index

*Reproduce with `index_recall.py --dataset sift|glove|music100`, and with `--calibrate-on queries`.*

`siann.Index(xb, metric, recall=0.95)` on the full datasets, its recall measured on the datasets' own
queries (ties counted as found), and its speed at the batch sizes it routes (`index_recall_*.txt`):

| Dataset | Calibrated on | nprobe | Recall on the queries | Build | Queries/s: 1 / 100 / 10,000 |
| --- | --- | ---: | ---: | ---: | ---: |
| SIFT, L2 | its own vectors | 38 | 0.940 | 6.0 s | 11,503 / 103,959 / 199,346 |
| SIFT, L2 | 1,000 of the queries (recall on the other 9,000) | 46 | **0.955** | 6.0 s | 11,103 / 98,239 / 180,252 |
| GloVe, cosine | its own vectors | 364 | 0.947 | 8.6 s | 2,396 / 21,616 / 46,450 |
| Music-100, inner product | its own vectors | 129 | 0.986 | 5.7 s | 9,546 / 60,076 / 170,253 |
| Music-100, inner product | 1,000 of the queries (recall on the other 9,000) | 25 | **0.946** | 4.8 s | 15,782 / 168,543 / 333,396 |

- **Calibrate on queries like the real ones.** A sample of the index's own vectors misses the target on
  real queries, either way: SIFT's queries are harder than its base vectors (0.940 against 0.95), Music-100's
  much easier (0.986, at five times the nprobe it needs). Calibrated on 1,000 of the queries, the other
  9,000 land within 0.005 of the target.
- **The IVF is chosen on every dataset,** 2-11 times faster than exact search on the calibration sample.
- **The build takes 5-9 seconds for a million vectors:** k-means and the lists (3.5-4.0 s), calibration,
  then the routing measurement (0.9-3.6 s).

### Routing approximate search

*Reproduce with `index_routing_ab.py --dataset sift|glove|music100`, with and without `--calibrate-on
queries`, and `ivf_routing.py`.*

Each index times its CPU and GPU scans on 1 to 4,096 queries when it is built and routes each batch to the
fastest. The table compares that routing with fixed thresholds (single queries on the CPU, the GPU's
query-major scan below 1,000 queries and its list-major scan from 1,000) on the same index: queries per
second of the measured route over the fixed one, at the batch sizes where the two differ
(`index_routing_ab_*.txt`; elsewhere they take the same path).

| Dataset, calibrated on | 2-8 queries | 16-64 | 100-512 | 1,000-2,048 |
| --- | ---: | ---: | ---: | ---: |
| SIFT, its own vectors | 2.3-3.8x (CPU) | 1.4x at 16 (CPU) | same | 1.3x at 1,000 (query-major) |
| SIFT, queries | 2.5-3.8x (CPU) | 1.6x at 16 (CPU) | same | 1.2x at 1,000 (query-major) |
| GloVe, its own vectors | 1.5-3.7x | 1.2-1.3x (list-major) | 1.3-1.9x (list-major) | same |
| Music-100, its own vectors | 2.6-6.3x (CPU) | same | **0.77x at 512** (list-major) | same |
| Music-100, queries | 3.2-5.3x (CPU) | 1.2-1.9x (CPU) | same | 1.6x (query-major) |

- **Small batches gain most:** the GPU's fixed cost per call outweighs its speed below 8-64 queries,
  depending on nprobe, and the CPU answers them 1.2-6.3 times faster.
- **The crossovers move with nprobe** (`ivf_routing.txt`, `siann.tune` on each dataset's queries): at nprobe
  46 on SIFT the CPU wins up to 16 queries and the GPU's list-major scan beyond 1,024; at nprobe 370 on
  GloVe the GPU wins from 8 queries and its list-major scan at every size; at nprobe 25 on Music-100 the
  CPU wins up to 64 and the GPU's query-major scan up to 4,096.
- **Time on queries like the real ones.** The one loss comes from timing on the index's own vectors: on
  Music-100 they made the list-major scan look faster at 512 queries, and on real queries it runs at 0.77
  times the query-major scan. Calibrated and timed on 1,000 queries, the index takes no slower route at
  any batch size.
- **A measured route replaces the default only where it was at least 10% faster,** about the spread of the
  timings, and between the batch sizes measured the default holds, so no threshold rests on a difference
  the timings cannot resolve.
- **Time each route back to back, warmed up.** After calls on the CPU, the GPU's next calls run up to 40%
  slower until it is busy again, so routes timed in alternation, or a fresh index's first calls, favor the
  CPU.

## Against mlxvs on the GPU

*Reproduce with `mlxvs_compare.py --dataset sift|glove|music100`.*

mlxvs runs only on Apple GPUs. FAISS's Metal path is level with it in small batches and under half its
speed in large ones, with no graph index ("FAISS's Metal path on the GPU"), so mlxvs is the GPU baseline. Same
data and queries, recall@10 against exact search, queries per second for single queries, batches of 100
and one batch of 10,000 (`mlxvs_sift.txt`, `mlxvs_glove.txt`, `mlxvs_music100.txt`). mlxvs's CAGRA uses
graph degree 32 pruned from 64, as in mlxvs's own benchmarks, in float32 and float16; the faster of the
two is given.

**Exact search:** siann's GPU kernels are 2.6-5.7x mlxvs's `brute_force_knn`.

| Dataset | Single: siann / mlxvs | Batches of 100 | Batch of 10,000 |
| --- | ---: | ---: | ---: |
| SIFT, L2 | 154 / 60 | 10,722 / 3,669 | 23,811 / 6,199 |
| GloVe, cosine | 162 / 54 | 10,139 / 3,153 | 20,396 / 5,341 |
| Music-100, inner product | 491 / 86 | 11,521 / 4,483 | 16,238 / 5,385 |

**IVF against IVF** (4,000 lists each, each library's own k-means; siann's lists uint8 on SIFT and
float16 on GloVe, mlxvs's IVF-Flat float32): recall within 0.006 of each other at the same nprobe, and
siann's GPU IVF 1.2-3.9x mlxvs's on single queries, 1.6-4.2x in batches of 100 and 1.7-2.4x in
batches of 10,000 on SIFT and GloVe. On Music-100 (siann's lists assigned by L2, float16;
`mlxvs_music100_ivf.txt`), mlxvs's IVF reaches a little more recall at the same nprobe, and at the
nearest recall siann's is 1.5-2.2x on single queries, 1.3-1.7x in batches of 100 and 1.2-1.3x in
batches of 10,000.

**mlxvs's CAGRA against siann's GPU IVF,** at the nearest recall (siann / CAGRA):

| Dataset, recall | Batches of 100 | Batch of 10,000 | CAGRA / siann |
| --- | ---: | ---: | ---: |
| SIFT, 0.925 / 0.932 | 123,922 / 182,460 | 311,030 / 805,121 | 1.5x, 2.6x |
| SIFT, 0.973 / 0.979 | 91,600 / 122,811 | 197,625 / 510,580 | 1.3x, 2.6x |
| SIFT, 0.993 / 0.994 | 63,257 / 77,252 | 117,203 / 246,364 | 1.2x, 2.1x |
| GloVe, 0.711 / 0.725 | 121,797 / 124,109 | 549,953 / 498,089 | 1.0x, 0.9x |
| GloVe, 0.890 / 0.882 | 37,456 / 42,228 | 118,631 / 108,056 | 1.1x, 0.9x |
| Music-100, 0.947 / 0.952 | 139,191 / 129,148 | 383,114 / 573,452 | 0.9x, 1.5x |
| Music-100, 0.977 / 0.976 | 69,098 / 78,246 | 191,041 / 277,303 | 1.1x, 1.45x |

- **On SIFT, CAGRA is faster in batches:** 1.2-1.5x siann's GPU IVF in batches of 100 and 2.1-2.6x in
  batches of 10,000. It also edges FAISS's HNSW in batches of 100 (122,811 against uint8 HNSW's 109,788
  at recall 0.98).
- **On Music-100 (inner product), the two are level in batches of 100** (0.9-1.1x), and CAGRA is
  1.45-1.5x in batches of 10,000.
- **On GloVe, the two are level** (0.9-1.1x), and CAGRA reaches at most recall 0.882 (itopk_size 256,
  the largest measured), where siann's IVF goes on to higher recall with more probes.
- **Single queries belong to siann's CPU path:** mlxvs answers 1,380-1,970 single queries per second on
  the GPU (about 0.5-0.7 ms each), siann's Rust IVF on one thread 4,453 on SIFT at recall 0.973 (6,254
  with uint8 lists) and 5,057 on GloVe at 0.839.
- **L1 and L-infinity** are siann's alone: mlxvs supports L2, cosine and inner product.

## FAISS's Metal path on the GPU

*Reproduce with `faiss_metal.py --stage tiny|small|sift-exact|sift-ivf`, `--chunk` setting the queries per
call.*

FAISS 1.16's Mac builds run its flat and IVF indexes on the GPU through Metal (`index_cpu_to_gpu`). On
SIFT1M, with an IVF-Flat of 4,000 lists from FAISS's own k-means (float32 lists), against mlxvs and siann
from the section above, in queries per second at nprobe 16, 32 and 64 (`faiss_metal_*.txt`). In every run
FAISS's GPU returns exactly its CPU index's neighbors, and its recall at each nprobe is within 0.004 of the
other two IVFs' on the same 10,000 queries (0.840, 0.924, 0.973).

| | FAISS Metal | mlxvs | siann's GPU |
| --- | ---: | ---: | ---: |
| Exact search, batches of 100 | 3,963 | 3,669 | 10,722 |
| IVF, single queries | 2,527 / 2,477 / 2,348 | 655 / 646 / 622 | 2,568 / 2,055 / 1,511 |
| IVF, batches of 100 | 34,839 / 38,130 / 24,890 | 42,986 / 36,116 / 29,974 | 180,286 / 123,922 / 91,600 |
| IVF, batches of 10,000 | 121,760 / 70,352 / 37,719 | 240,542 / 184,001 / 107,910 | 573,795 / 311,030 / 197,625 |

- **mlxvs is the stronger GPU baseline.** FAISS's Metal path is level with mlxvs in batches of 100
  (0.8-1.1 times) and at 0.35-0.51 times its speed in batches of 10,000, and it has no graph index
  (FAISS's CAGRA runs only on CUDA).
- **Single queries:** FAISS's GPU IVF is 3.8-3.9 times mlxvs's and level with siann's GPU IVF; siann
  sends single queries to the CPU, which is faster again.
- **Keep FAISS's exact search on the GPU to small calls.** Its GPU memory is counted outside the process:
  one call of 1,000 queries against SIFT1M (4,432 queries per second) lowered the machine's free memory by
  10 GB while the process peaked at 1.6 GB. Calls of 10,000 queries would need more than 40 GB of this
  machine's 64, and such calls froze it. IVF calls of 10,000 queries hold only the probed lists' distances
  (process peak 5.9 GB, free memory steady).

## High-dimensional text embeddings

*Reproduce with `mlxvs_compare.py --dataset dbpedia --base 100000 --queries 1000 --small 200 --nlist
1000` and `ivf_cpu.py --dataset dbpedia --metric l2 --base 100000 --queries 1000 --small 200 --nlist
1000 --no-torch`.*

DBpedia OpenAI embeddings (1,536 dimensions, cosine; [DATASETS](DATASETS.md)), measured on the first
100,000 base vectors, with 1,000 queries and 1,000 lists, so batches go up to 1,000 queries. A graph
index's lead over an IVF usually grows with the collection, so these give the direction for a million
vectors, not the size of the gap (`mlxvs_dbpedia_100k.txt`, `ivf_dbpedia_100k.txt`).

- **Exact search on the GPU:** siann's is 1.1-1.2x mlxvs's (26,744 against 21,912 queries per second in
  a batch of 1,000), against 2.6-5.7x at 100-128 dimensions: at 1,536 dimensions the matrix multiply,
  the same in both, is most of the time. It gives recall 1.0 faster than FAISS's HNSW gives any recall
  in a batch of 1,000 (HNSW: 22,932 at recall 0.915), and within 1.7x of CAGRA at recall 0.99.
- **mlxvs's CAGRA is 1.9-2.8x siann's GPU IVF** at the nearest recall, in batches of 100 and of 1,000:

  | Recall: siann / CAGRA | Batches of 100: siann / CAGRA | Batch of 1,000 |
  | --- | ---: | ---: |
  | 0.943 / 0.952 | 30,450 / 63,912 | 74,872 / 142,080 |
  | 0.969 / 0.976 | 16,831 / 38,855 | 41,151 / 81,354 |
  | 0.986 / 0.990 | 8,829 / 24,999 | 22,889 / 46,195 |

- **Against FAISS's HNSW on the CPU,** siann's GPU IVF (float16 lists) is 1.7-2.1x in batches of 100
  and 4.2-5.0x in batches of 1,000; on single queries HNSW is 2.4-2.7x siann's Rust IVF.
- **Against mlxvs's IVF-Flat,** siann's GPU IVF is 1.1-1.8x up to nprobe 32 (recall 0.94); from nprobe
  64, mlxvs's is 1.2-2.6x on single queries and in batches of 100, and the two are level in a batch of
  1,000.

### A graph built on the GPU, searched on the CPU

*Reproduce with `graph_faiss.py --dataset sift|glove|music100` and `graph_faiss.py --dataset dbpedia
--base 100000 --queries 1000`; the hnswlib comparison with `graph_cpu.py` (same options, plus
`--native`).*

mlxvs builds a CAGRA graph on the GPU (32 links per vector). Loaded into a FAISS HNSW index as its only
layer, entered at the vector nearest the mean, it is searched by FAISS's own code, against FAISS's own
HNSW (M = 32, 64 links per vector at its bottom layer): graph against graph, same search code. Single
queries on one thread, median microseconds at the nearest recall (`graph_faiss_*.txt`):

| Dataset | Build: GPU graph / FAISS HNSW | Single query: GPU graph / FAISS HNSW (recall) | GPU graph faster |
| --- | ---: | ---: | ---: |
| SIFT | 4.4 / 9.5 s | 73 / 86 (0.94); 130 / 152 (0.98); 236 / 285 (0.99) | 1.15-1.2x |
| GloVe | 6.5 / 14.6 s | 281 / about 290 (0.83-0.85); 520 / about 590 (0.88-0.89) | level to 1.2x |
| Music-100 (inner product) | 5.1 / 8.0 s | 53 / 60 (0.96); 94 / 112 (0.98) | 1.15-1.2x |
| DBpedia, 100,000 | 1.8 / 8.0 s | 172 / 185 (0.95); 281 / 315 (0.98); 485 / 542 (0.99) | 1.1x |

- **A graph built on the GPU searches as well as FAISS's HNSW, or a little faster,** on the CPU: level to
  1.2 times at matched recall, with the same gains for batches on every thread. At the same 32 links per
  vector, FAISS's HNSW (M = 16) reaches far less recall.
- **The GPU builds it 1.6-4.4 times faster** than FAISS builds its HNSW on 16 threads.
- **The loaded graph behaves as exported:** searched by hnswlib, the same graph reaches the same recall at
  every ef. **hnswlib's search is slow on Apple silicon** (its hand-written SIMD is for x86): on its own
  HNSW graph it takes 1.0-1.4 times FAISS's time at the same ef at 100-128 dimensions and about 4 times at
  1,536 (`graph_cpu_*.txt`).

## Inner product

*Reproduce with `backends.py --dataset glove-raw|music100 --metrics ip`, `blas_l2.py --metric ip`,
`ip_lists.py`, `ivf_cpu.py --metric ip`, `ip_reduction.py`.*

Two datasets with vectors of uneven length: Music-100, made to be compared by inner product (lengths
0.017 to 2.12, the longest 2.9 times the median), and raw GloVe-100, word vectors with their lengths
kept (0.04 to 18.7, 3.4 times the median) ([DATASETS](DATASETS.md)).

**Exact search** against FAISS's `IndexFlatIP` (`ip_exact_*.txt`, `ip_blas_*.txt`):

| Queries | Music-100: GPU | Music-100: CPU (BLAS) | Raw GloVe: GPU | Raw GloVe: CPU (BLAS) |
| ---: | ---: | ---: | ---: | ---: |
| 100 | **16.9x** (0.007 against 0.115 s) | **2.71x** | **17.7x** (0.008 against 0.136 s) | **2.70x** |
| 1,000 | **6.3x** (0.039 against 0.247 s) | 0.90x | **5.9x** (0.049 against 0.287 s) | 0.94x |
| 10,000 | **4.3x** (0.388 against 1.666 s) | | **4.1x** (0.482 against 1.985 s) | |

**The IVF's lists are assigned by L2 and probed by inner product.** Assigned by inner product (as
FAISS's `METRIC_INNER_PRODUCT` IVF assigns, its quantizer both assigning and probing), centroids
pointing towards long vectors attract much of the data; assigned by L2, the lists balance. GPU IVF,
the same centroids, float16 lists (`ip_lists_*.txt`):

| Dataset, lists assigned by | Largest list | 10 largest hold | Empty | Recall, nprobe 16 / 128 | 10,000 queries/s, nprobe 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Music-100, inner product | 20,645 | 10% | 22% | 0.887 / 0.979 | 77,575 |
| Music-100, **L2** | 1,402 | 1% | none | **0.911** / 0.977 | **652,963** |
| Raw GloVe, inner product | 35,725 | 21% | 39% | 0.373 / 0.641 | 17,117 |
| Raw GloVe, **L2** | 2,066 | 2% | none | 0.210 / 0.443 | **876,751** |

At matched recall the L2-assigned lists are faster on both datasets: on Music-100 1.7-2.6x in batches
of 100 and 4.1-8.4x in batches of 10,000; on raw GloVe, where they need more probes for the same
recall (nprobe 512 for 0.727, against 256 for 0.725), 2.6x and 7.6x at recall 0.73, and 1.8x and 4.6x
at 0.85-0.87 (nprobe 1024; `ip_lists_glove-raw_wide.txt`). Probing by L2 as well fails (recall at most
0.40). When the probed lists hold fewer than k vectors, which happens on raw GloVe at small nprobe, the
missing neighbors are returned as -1.

**On Music-100, the IVF against FAISS's HNSW** (M = 32) with inner product, at the nearest recall; queries
per second, siann's fastest path for each batch (`ip_ivf_music100.txt`):

| Recall: IVF / HNSW | Single, one thread: Rust IVF / HNSW | Batches of 100: siann / HNSW | 10,000: GPU IVF / HNSW |
| --- | ---: | ---: | ---: |
| 0.911 / 0.924 | 22,130 / 26,121 | 173,743 (Rust) / 181,860 | 638,213 / 291,948 |
| 0.948 / 0.956 | 18,368 / 13,057 | 127,096 (GPU) / 121,176 | 383,269 / 170,000 |
| 0.978 / 0.976 | 7,645 / 8,288 | 54,784 (GPU) / 70,719 | 194,597 / 93,038 |

siann's IVF is level with HNSW on single queries and in batches of 100 (0.8-1.4x) and 2.1-2.3x HNSW in
batches of 10,000. FAISS's own IVF, with its inner-product lists, is slower than siann's at every nprobe,
from 2.3x (batches of 100, nprobe 4) to 34x (batches of 10,000, nprobe 128). Exact search on the GPU gives recall 1.0 at 25,773 queries per second in batches of
10,000.

**On raw GloVe no IVF works well:** siann's reaches recall 0.443 by nprobe 128 and FAISS's 0.641
(`ip_ivf_glove-raw.txt`). FAISS's HNSW with inner product reaches 0.888 at 14,126 queries per second in
batches of 100 and 15,294 in batches of 10,000 (efSearch 256); exact search on the GPU gives recall 1.0
at about 12,500 in batches of 100 (0.88x HNSW) and about 20,700 in batches of 10,000 (1.35x).

**Reduced to L2** (each base vector extended by sqrt(M^2 - |x|^2) to the longest length M, each query by
0, so that L2 ranks as inner product), the lists balance but recall falls, for the IVF and FAISS's HNSW
alike: on Music-100 the IVF reaches 0.24-0.81 up to nprobe 128 and HNSW 0.58-0.79 (against 0.88-0.99
with inner product), on raw GloVe 0.03-0.34 and 0.16-0.34. Most vectors are far shorter than the
longest, so the added coordinate dominates the distances.

## PyTorch against MLX

*Reproduce with `torch_select_ab.py --dataset sift|glove`.*

The GPU IVF's Metal code runs unchanged through `torch.mps.compile_shader`, with the same coarse step,
probe selection, threads per query and merge. On the same lists (float32, the same centroids) it returns
the same neighbors as MLX's IVF, index for index. L2, queries per second, PyTorch / MLX
(`torch_select_ab_sift.txt`, `torch_select_ab_glove.txt`):

| Dataset | nprobe | Recall | Single | Batches of 100 | Batch of 10,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SIFT | 8 | 0.716 | 1,655 / 2,572 | 100,110 / 118,950 | 871,159 / 879,927 |
| SIFT | 16 | 0.840 | 1,446 / 2,070 | 75,656 / 86,824 | 521,842 / 525,111 |
| SIFT | 32 | 0.925 | 1,121 / 1,468 | 52,877 / 58,333 | 313,611 / 295,787 |
| SIFT | 64 | 0.973 | 805 / 968 | 32,852 / 34,921 | 194,432 / 182,272 |
| SIFT | 128 | 0.993 | 541 / 614 | 19,121 / 19,803 | 108,639 / 107,322 |
| GloVe | 8 | 0.631 | 1,639 / 2,513 | 99,963 / 113,876 | 899,436 / 889,301 |
| GloVe | 16 | 0.712 | 1,450 / 1,992 | 80,595 / 91,142 | 532,096 / 530,652 |
| GloVe | 32 | 0.780 | 699 / 1,395 | 57,325 / 63,073 | 339,220 / 303,899 |
| GloVe | 64 | 0.839 | 817 / 945 | 36,660 / 39,236 | 206,804 / 194,198 |
| GloVe | 128 | 0.890 | 422 / 602 | 21,199 / 22,571 | 113,667 / 114,019 |

On batches of 10,000 the two run level up to nprobe 16, and PyTorch is up to 1.12x MLX above it. On
batches of 100 PyTorch reaches 0.84-0.97x MLX's speed, and on single queries 0.50-0.88x, from its fixed
cost of about 0.2 ms per call.

## Reproducing

With [Rust](https://rustup.rs) and [uv](https://docs.astral.sh/uv/) installed, `uv sync --group bench`
builds siann, Rust extension included, with the benchmarks' dependencies (FAISS, PyTorch, h5py); the datasets go in `data/` ([DATASETS](DATASETS.md)). Then, from `benchmarks/`:

```sh
uv run --group bench python check_kernels.py                     # every kernel against NumPy
uv run --group bench python backends.py --dataset sift --q 1000  # exact search, every backend
uv run --group bench python ivf_cpu.py --dataset glove --metric l2   # IVF and HNSW
uv run --group bench python fp16_lists.py --dataset sift --storages float32,float16,uint8 --faiss
uv run --group bench python single_query.py --dataset sift --splits 1,4 --faiss
```

uv rebuilds the extension when its Rust sources change. FAISS and PyTorch cannot share a process on
macOS, so the scripts that use both run each in a process of its own. An IVF or compressed-index run
takes three to five minutes, an exact-search run on every backend about two.
