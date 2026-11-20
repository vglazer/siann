# Algorithms

*Part of the [siann documentation](README.md#documentation).*

siann answers k-nearest-neighbor queries: given a set of base vectors and a query, find the k base
vectors closest to the query. It does this exactly, by comparing the query with every base vector,
or approximately, with an IVF index that compares it only with the vectors in a few clusters. This
page describes how each search works, on the GPU and on the CPU, and the methods that do not pay.
[BENCHMARKS](BENCHMARKS.md) has the measurements, on an Apple M3 Max, and the scripts behind them.

## Distances

- **L2** (Euclidean): the square root of the sum of squared differences. Searches compare squared
  distances, which rank the same way and need no square root.
- **L1** (Manhattan): the sum of absolute differences. It is less sensitive to a few large
  differences than L2, and has no matrix-multiply form (see below), which makes it the harder case
  and the one where siann gains most.
- **L-infinity** (Chebyshev): the largest absolute difference in any one coordinate. It shares L1's
  kernels, with a running maximum where L1 keeps a running sum. A maximum involves no rounding, so
  its distances are exact.
- **Inner product:** q·x, largest first, for embedding models trained to be compared by it. siann
  ranks by -q·x, so that smaller is nearer, as for the distances. Exact search is a matrix multiply,
  like L2 without the norms. The blocked layouts pad with infinity, which for inner product would be
  the best possible score, so exact search skips padding slots by index.
- **Cosine:** the angle between vectors. Normalized to unit length, vectors rank by cosine exactly as
  they rank by L2, so cosine search is L2 search on normalized vectors (see [DATASETS](DATASETS.md)).

## Exact search on the GPU

Exact search computes a distance for every pair of query and base vector, then keeps each query's k
smallest. Both halves need care on a GPU: written with a framework's built-in operations, either is
slower than FAISS on the CPU.

**L2 as a matrix multiply.** The squared distance expands to
`|q - x|² = |q|² - 2 q·x + |x|²`. The term `|q|²` is the same for every base vector of a query, so it
does not change the ranking and is dropped. What remains, for a tile of queries Q against all base
vectors X, is `|x|² - 2 Q Xᵀ`: one matrix multiply, which GPUs do at close to their peak speed, plus
the base vectors' squared norms, computed once. siann computes it in one fused call (`addmm`, which
adds a matrix to a scaled product), 2.6 times as fast as the multiply, scaling and subtraction as
separate steps.

**L1 as a tiled kernel.** L1 has no such expansion, so siann computes it with a custom GPU kernel
(a function run by thousands of GPU threads at once) laid out like a matrix multiply. Each group of
16 × 16 threads computes a 64 × 64 block of distances, 4 × 4 per thread, held in registers. The
group loads 16 dimensions of its 64 queries and 64 base vectors at a time into threadgroup memory
(a small, fast memory shared by the threads of a group), so each base vector is read from main memory
once per 64 queries rather than once per query. The same kernel computes L-infinity: the step that
folds a coordinate difference into the running distance changes from a sum to a maximum, and the
speed does not change.

**Top-k selection.** With the distances computed, each query needs its k smallest of a million. The
frameworks' own selection (`argpartition`, `topk`) takes eight times as long as the distances
themselves. siann selects in two stages: 512 threads per query each scan a strided share of the row,
keeping their best k in registers by insertion, then a small selection over the 512 × k survivors gives
the final k. That is 35 times faster than the frameworks' selection.

**Tiles.** Queries go 256 at a time, and each tile is evaluated before the next, so that only one
tile's distances (256 × 1M, 1 GB) are held at once. MLX evaluates lazily (it builds a graph of
operations and runs it when a result is needed), so a search evaluated only at the end keeps every
tile's distance matrix alive and runs some 200 times slower.

## Exact search on the CPU

For L2, the matrix-multiply form works on the CPU too, through BLAS (the tuned linear-algebra
library; on a Mac, Apple's Accelerate, which runs matrix multiplies on the chip's matrix units). FAISS's
flat index does this, and so does siann, without FAISS: NumPy's `@` (which calls Accelerate) computes
the products of a tile of queries with a block of the base, and a Rust top-k ranks them by
|x|² - 2 q·x, keeping each query's best k across blocks. The tile size matters: products of 4-8 million
values (16-32 MB) are fast, smaller ones pay per-call overhead and larger ones fall out of cache. The
default takes up to 512 queries and about 4 million values per product.

L1 has no BLAS form, and FAISS's L1 runs at about 70 G absolute differences per second, well below
what the cores can do, so siann has its own L1 kernel, in Rust.

**Blocked layout.** The base vectors are stored once, at build time, in blocks of 8, dimension by
dimension: the first dimension of 8 vectors, then their second, and so on. The inner loop compares 4
queries with one block, keeping 32 running sums, one per pair. The sums do not depend on each other,
so the compiler turns the loop into NEON instructions (Apple's vector instructions, 4 numbers at a
time), 8 absolute differences and 8 additions per dimension, without reordering any floating-point
additions. Floating-point addition is not associative, so a compiler may not vectorize a single
running sum by splitting it; 32 separate sums sidestep the issue.

**Cache tiles and threads.** Each thread takes 32 queries and scans the base in tiles of 1,024
vectors (512 KB), comparing each tile with all of its queries while the tile is in cache. Each query
keeps its best k in a small sorted list. The last block is padded with vectors of infinity, whose
distances never enter a top-k. The same kernel computes L2 by squaring the differences (BLAS is faster
for L2) and L-infinity with a running maximum, written as a comparison and select.

## IVF: approximate search

An IVF (inverted file) index clusters the base vectors and searches only the clusters nearest the
query. It trades a little recall (the share of the true k nearest neighbors found) for a large cut in
work.

**Building.** k-means (FAISS's, 10 iterations, on a sample of 100 vectors per cluster) finds the
cluster centers, the centroids; siann uses 4,000 for a million vectors. Each base vector then goes to
the list of its nearest centroid in the search metric, found by exact search. k-means itself runs in
L2, even for an L1 index: the centroids only need to spread over the data.

**Searching.** A query's `nprobe` nearest centroids (the coarse step) give the lists to scan; the k
nearest vectors among those lists are the answer. Raising `nprobe` raises recall and cost: on SIFT1M
with L2, nprobe 16 finds 84% of the true neighbors, nprobe 64 97%, nprobe 128 99%.

**Query-major and list-major scans.** There are two ways to scan:

- **Query-major:** each query walks its own lists. On the GPU, 128 threads per query each take a
  strided share and keep their best k in registers, then merge them pairwise in threadgroup memory
  (seven steps), so the kernel writes only k results per query. On the CPU, one thread per query.
  Cheap for a single query, but in a batch every query reads its lists separately, so a popular list
  is read from memory many times.
- **List-major:** queries are first grouped by the lists they probe, and each list is scanned once
  for all of them. On the GPU, one group of 256 threads per list handles 16 queries at a time, with
  the list's vectors staged through threadgroup memory; the queries are grouped on the GPU too (a
  sort of the probe lists and a count per list). On the CPU, each list is scanned with the exact
  kernel, 4 queries at a time, while it is in cache. Each query's results from its lists are then
  merged.

List-major pays off with batches: in a batch of 10,000 queries at nprobe 64, each list is probed by
about 160 of them, and read once instead of 160 times. The GPU uses the query-major scan below 1,000
queries and the list-major scan above.

**Split single queries.** A single query can be split: its coarse step and its lists divided among a
few threads (a pool of their own, 4 by default), each keeping its own best k, merged in probe order so
that the result is the one-thread result, ties included. This cuts a single query's latency at the
cost of occupying the threads.

**The coarse step.** Finding the nearest nprobe centroids is itself a small exact search (4,000
centroids), and for a single query it is half the work at nprobe 16. A two-level coarse step cuts it:
the centroids are clustered into groups (64 for 4,000 lists), and a query is compared with the groups'
centres, then only with the centroids of its nearest groups (16 of 64 in the measurements). On the GPU, the
distances to the centroids are one fused multiply, and the costly part is choosing each query's nprobe
nearest. The scans need the lists in no order, so the query-major scan chooses them with MLX's
`argpartition`, unordered; the list-major scan does too above nprobe 16, and at 16 and below uses the
two-stage selection with 32 threads per query, which is cheaper there for thousands of queries.

**List storage.** The scan reads every vector of every probed list, so in a batch of queries scanned
one by one it is limited by memory traffic, not arithmetic. The lists can therefore be stored in a
smaller type, with distances still computed in float32 from the stored values:

- **float16** (half the bytes): each coordinate rounded to half precision. On SIFT, whose
  coordinates are small integers, this is exact; on GloVe it changes each coordinate by at most
  2.4e-4, and recall by at most 0.001.
- **uint8** (a quarter): one byte per coordinate, for data whose coordinates are integers from 0 to
  255, like SIFT's, so nothing is lost. Other data is refused: scaling it into bytes (int8
  quantization) would lose precision and need re-ranking.

On the GPU, the kernels read the smaller type directly. On the CPU, a query whose coordinates are all
integers from 0 to 255 is compared with byte lists in exact integer arithmetic, over a row-major copy
of the lists (each vector's bytes contiguous, so that 16 of them fill a register and L2's sum is one
dot-product instruction per 16 values). Other queries convert the bytes to float32 in registers inside
the distance loop, so nothing is rounded; with the copy, byte lists take two bytes per coordinate, half
of float32's. Rust has no stable float16 type, so float16 lists hold the half-precision bits as 16-bit
integers, and each block is widened to float32 in one flat loop before the distance code runs on it.
The widening is exact: sign, exponent and mantissa move into float32's positions, read as a number
2^112 times too small, normal or subnormal, which one multiply corrects; `check_kernels.py` compares
it with NumPy's conversion on every finite float16.

Smaller lists pay where the scan is memory-bound (batches of about 100 queries, split single queries)
and cost where it is not: a single query on one thread reads its lists below the memory system's limit,
so widening float16 is pure extra work, while the list-major scans already read each list once for many
queries. Byte lists with integer queries are faster everywhere on the CPU, through the integer path.

## What does not pay

Each of these is measured; the scripts are named in parentheses, and their output is in
[proto/results/](proto/results/).

- **LSH for dense vectors** (`lsh_l1.py`). Locality-sensitive hashing hashes vectors so that near ones
  tend to collide, then compares only the colliding ones. With Cauchy random projections (the standard
  LSH for L1) on SIFT1M, reaching recall 0.72 means re-ranking a tenth of the database, which exact
  search on the GPU already does faster. Real data has low contrast: a neighbor and a point twice as
  far collide with nearly the same probability (0.28 and 0.26).
- **A fused L1 kernel** (distance and selection in one pass): it reads the whole base once per query,
  so it is limited by memory speed, at a third of the tiled kernel's speed.
- **`f32::max` for L-infinity in Rust** (`rust_cpu_sweep.py`): the compiler treats a maximum as a
  reduction it may reorder, and vectorizes the wrong loop, taking twice L1's time (an unsigned-integer
  maximum on the float bits does the same). A comparison and select keeps one running maximum per
  lane: 1.4 times L1's time.
- **IVF for inner product on vectors of very uneven length** (`ip_reduction.py`; raw GloVe, lengths
  0.04 to 18.7). Assigned by inner product, the lists are badly unbalanced (39% empty, 10 lists holding
  a fifth of the data) and recall stays below 0.65; reduced to L2 by extending each vector to the same
  length, the lists balance but recall falls below 0.35 (and FAISS's HNSW on the extended vectors fails
  too). Exact search on the GPU is the method there.
- **BLAS in the CPU IVF's list-major scan** (`blas_ivf.py`): one multiply per list, its queries against
  its vectors, the products ranked in Rust. The multiplies are small (about 160 queries by 250 vectors
  at nprobe 64), so overheads dominate: 0.5-0.8 times the Rust list-major scan on batches of 10,000,
  and 0.14-0.19 times on batches of 100. With the coarse step and grouping free it would at best draw
  level. BLAS pays for exact L2, where the multiplies are large.
- **Splitting a single query on the global thread pool** (`single_query.py`): a query injected into
  rayon's pool (all 16 cores) wakes its sleeping threads, about 100 microseconds per query, slower than
  one thread at low nprobe. A pool of its own (4 threads) avoids it.
- **Widening float16 inside the distance loops in Rust**, or over nested `[u16; 8]` arrays: the
  compiler loads the halves one lane at a time. Widening each block into a buffer first, in a loop over
  flat slices, is 1.3-1.9 times faster.
- **Fewer IVF lists to cheapen the coarse step** (`coarse_step.py`): each list grows as the coarse step
  shrinks, which cancels out at matched recall; the two-level coarse step is what helps.
- **Finer partitions for small batches on the CPU** (`fine_lists.py`, `batch_profile.py`): 16,000 lists
  touch 1.6 times fewer vectors than 4,000 at recall 0.97 and run at the same speed, held back by a
  fixed cost per list. On the GPU they pay at high recall.
- **An integer kernel in the blocked layout:** no faster than the float path (259 against 263
  microseconds at nprobe 64, written with the `wide` crate's vector types); the integer path needs each
  vector's bytes contiguous.
- **The integer path in the CPU's list-major scan** (large batches; `fp16_lists.py`): Rust's list-major
  scan with uint8 lists does 38,567 queries per second on SIFT at nprobe 64, against the GPU IVF's
  199,347 and compressed HNSW's 153,710; even the single-query path's gain (1.6-1.8 times) would leave
  it far behind both.
- **Reading four coordinates per load in the GPU scan** (`uchar4`, `half4`, `float4`, with identical
  results; `gpu_load4_ab.txt`): up to 6% faster on SIFT and 20-45% slower on GloVe.
