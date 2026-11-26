# Implementation

*Part of the [siann documentation](README.md#documentation).*

siann answers k-nearest-neighbor queries, exactly or to a recall target. It is fast for two reasons. It
**chooses**: it measures on your data and machine which method, settings and device meet the target
fastest for each batch size, reports what it chose, and chooses again as the index changes. And **each path
it chooses between is fast**, through custom GPU kernels and CPU code laid out for the hardware. This page
covers how siann chooses, the search methods and the kernels, where other libraries are faster and why, and
the code layout. The measurements are in [BENCHMARKS](BENCHMARKS.md), their summary in the README's
[How it compares](README.md#how-it-compares), and what to do when the defaults do not fit in
[TUNING](TUNING.md).

## Why siann is fast

### It chooses

How fast a search can be depends on the data, the metric, the machine, the batch size and the recall
wanted. siann takes the recall target and measures the rest:

| Choice | Made from |
| --- | --- |
| How many lists a query scans (nprobe), how wide a graph search runs | the least that reaches the target on a sample of queries |
| Exact search, the IVF or the graph, on the CPU or the GPU, and which scan, by batch size | every path timed on 1 to 4,096 queries when the index is built: a table of the fastest at each size |
| How the lists are stored | the data: bytes where lossless, float16 for GPU batches |
| Who builds the graph | mlxvs where it is installed (for L2, cosine and inner product), siann's own builder elsewhere |

The choices interact, which is why they are measured together rather than set from rules of thumb: where
the CPU and the GPU cross depends on how many lists a query scans, and that depends on the data and the
target, and a graph can win in a band of batch sizes and lose above and below it ([TUNING](TUNING.md#approximate-search-where-each-batch-runs) has the crossovers measured).
[Approximate search through siann.Index](BENCHMARKS.md#approximate-search-through-siannindex) measures what
the choices gain.

Other libraries leave these choices to you: [FAISS](https://github.com/facebookresearch/faiss) (Meta's similarity-search library) and [mlxvs](https://github.com/mikepapadim/mlxvs)
(NVIDIA's cuVS methods on MLX) offer a family of indexes and their parameters,
and you pick the index, its settings and, for FAISS, the device. FAISS's auto-tuning (`ParameterSpace`)
explores one index's parameters on one device against ground truth you compute; it does not choose between
exact and approximate search, between scans, or between the CPU and the GPU.

### Each path is fast

The choice is worth making only if every option is fast. Written with a framework's built-in operations,
every GPU backend is slower than FAISS on the CPU ([BACKENDS](BACKENDS.md#summary)), so siann runs its own
code: custom Metal kernels on the GPU (Metal is Apple's GPU language), and on the CPU Rust kernels laid out
for the vector instructions and BLAS (the tuned linear-algebra library) for L2. [The kernels in
detail](#the-kernels-in-detail) explains where each gain comes from.

## How siann chooses

### Which path a search takes

```mermaid
flowchart TB
    build(["siann.Index(xb, metric, recall=?)"]) --> target{"recall<br/>target?"}
    target -- no --> exact["exact search"]
    target -- yes --> ivf["IVF: siann's k-means lists,<br/>nprobe calibrated to the target"]
    ivf --> graph["graph: 32 links per vector,<br/>search widths calibrated to the target"]
    graph --> table["every path timed at 1 to 4,096 queries:<br/>a table of the fastest at each batch size"]
    table --> everywhere{"exact search the fastest<br/>at every size?"}
    everywhere -- yes --> exact
    everywhere -- no --> approx["the table routes each search"]
    exact --> ebatch{"GPU, and a batch<br/>above the profile's threshold?"}
    ebatch -- yes --> ge["GPU kernels<br/>L2, cosine, inner product: matrix multiply + two-stage top-k<br/>L1, L-infinity: tiled kernel + top-k"]
    ebatch -- no --> metric{"metric?"}
    metric -- "L2, cosine, inner product" --> cb["NumPy BLAS + Rust top-k"]
    metric -- "L1, L-infinity" --> cr["Rust blocked kernel"]
    approx --> paths["the table's path for the batch size:<br/>the graph on the CPU (Rust) or the GPU (mlxvs),<br/>the IVF's CPU or GPU scans, or exact search"]
```

### From a recall target to an index

`siann.Index(xb, metric, recall=0.95)` turns a recall target into an IVF, a graph and their settings, in
about 10-20 seconds for a million vectors on an M3 Max and 50-70 on 4 Linux CPUs:

1. **The lists:** about 4 sqrt(n) of them (4,000 for a million vectors), from siann's k-means, stored as
   bytes where every coordinate is an integer from 0 to 255 (lossless), and otherwise as float16 on the GPU
   (recall moves by at most 0.001) and float32 on the CPU.
2. **Calibration:** a sample of 1,000 queries is searched exactly, then with the IVF at a doubling nprobe
   until it reaches the target recall@10, and a bisection narrows nprobe to the smallest that does. The
   sample is the index's own vectors (each one's own match left out) unless representative queries are
   given (`calibration_queries=`). Recall counts a neighbor as found when it is no farther than the true
   10th, so that ties broken differently are not misses.
3. **The graph:** a CAGRA graph over the same vectors, 32 links per vector ([Graph](#graph-approximate-search)),
   and the widths of its searches, on the CPU and on the GPU, calibrated to the target on the same sample
   as nprobe.
4. **Routing:** every path is timed at batch sizes doubling from 1 to 4,096 queries: exact search on the CPU
   and on the GPU, the IVF's CPU scans (one query split over 4 threads; one thread per query, its coarse
   step in Rust or through BLAS; list-major) and GPU scans (query-major, list-major), and the graph on each
   device. Each path is timed by itself on distinct queries and rated by its mean call, the GPU's buffer
   cache emptied before each GPU path and a rate far below its neighbors' timed again. The result is a
   table: at each size measured, the paths fastest first, the default first unless another is 10% faster
   (about the timings' spread). A search takes the entry of the largest size measured at or below its
   batch, and in it the first path it may use: on the device asked for, approximate where `nprobe=` is
   given ([TUNING](TUNING.md#approximate-search-where-each-batch-runs)).
5. **IVF or exact:** where exact search is the fastest at every size measured, the index keeps exact search
   alone; otherwise exact search stays one of the routes, used at the sizes where it wins.

Exact search routes by a *profile*: for each metric, the batch size from which the GPU is faster; siann's
own, or one `siann.tune` measures on your data and machine ([TUNING](TUNING.md#exact-search-gpu-or-cpu)).
Every choice can be overridden (the README's [Python](README.md#python) section lists how).

Recall on new queries lands within a few thousandths of the target, on either side: the calibration sample
estimates recall to about 0.004 for the IVF and 0.006 for the graph, and on Music-100, for a target of
0.95, the IVF reached 0.947 on 9,000 other queries, the graph 0.951 on the GPU and 0.957 on the CPU
([Calibration on new queries](BENCHMARKS.md#calibration-on-new-queries)).

### Threads

An index can be searched from several threads at once, and changed while it is searched. Each search
begins by taking a snapshot of the index, its vectors, lists, buffer and mask as they are at that moment,
and searches that: changes replace the arrays they change rather than writing into them, and take a lock,
so a snapshot never mixes two states. Searches do not hold the lock while they run, so concurrent searches
proceed in parallel: on the CPU, 4 threads answer 1.6-1.9 times as many queries as one (BENCHMARKS,
[Threads](BENCHMARKS.md#threads)). Changes run one at a time, and a search that began before a change
returns results from before it. Each device's copy of the vectors is built once, by the first search that
needs it.

### Memory

On Apple silicon the GPU's memory is the machine's, so a GPU search that outgrows it does not fail: the
system swaps or stalls. siann keeps each GPU call's working memory within a budget, 4 GB, or a quarter of
the GPU's recommended working set (what Metal says it can use without paging) on a machine where that is
smaller:

- **A large batch is searched in equal chunks.** The IVF's working memory grows with the batch: each
  query's distances to every centroid, and in the list-major scan 16 lanes of k candidates per probed list.
  The chunk is the budget over that, 25,600 queries on SIFT1M at nprobe 46. Exact search works on 256
  queries at a time, fewer once a tile's distances to every base vector would exceed the budget (from
  about 2.8 million vectors).
- **Splitting never changes results,** as each query's depends on it alone, so the budget may depend on
  the machine and the neighbors still do not.
- **MLX's cache of freed buffers is capped** at twice the budget (by default it may hold as much as the
  working set), unless the program has set a lower cap itself. PyTorch's allocator is left as it is: it
  held at most 2.5 GB after the same searches.
- **An index whose GPU copy would take more than half the working set stays on the CPU,** at the build, at
  a load, or at the merge that grows it past that; `device="gpu"` is then refused with the reason, and
  `index.plan` says so.

What the index holds grows with it, not with the batch: the vectors, each device's copy, and during a
merge, copies of the vectors as the lists are rebuilt (BENCHMARKS, [Memory](BENCHMARKS.md#memory)).

### Saving and loading

`index.save(path)` writes one NumPy `.npz` file, read back with pickling disabled: the base vectors (as
bytes where every coordinate is an integer from 0 to 255), the IVF's centroids and each vector's list,
the calibration sample, and a JSON header with the metric, the recall target, the calibrated nprobe, the
storage, the routes, the profile and the plan, with the machine the routes were measured on.
`siann.Index.load(path)` rebuilds each device's copy of the lists from them the first time a search needs
it, as when the index was built, so the file assumes no device. What the index decided from the data, its
lists and nprobe, and its choice between the IVF and exact search, are kept, so a loaded index returns the
same neighbors anywhere. What it decided from the machine, its routes and profile, is kept on the same
machine and re-measured on another (`remeasure=` forces either). A file of a later format is refused.

### Changing an index

`index.add(x, ids=)` adds vectors and `index.remove(ids)` removes them. Each vector has an id, which searches
return: its row number unless `ids=` names it. Rebuilding each device's copy of the index for every change
would cost under a second per million vectors, but each time, so changes are gathered first:

```mermaid
flowchart LR
    add(["add(x)"]) --> asg["each vector's list found now<br/>(an IVF: its nearest centroid)"] --> buf["a buffer,<br/>searched exactly"]
    buf -- "10,000 vectors, or 1% of the index" --> merge["merge: the lists rebuilt,<br/>nprobe recalibrated"]
    rm(["remove(ids)"]) --> mask["a mask: searches<br/>drop those rows"]
    mask -- "512 rows" --> merge
    merge -- "largest list doubled against the mean,<br/>or the index doubled" --> recl["re-cluster: k-means again,<br/>routes re-measured"]
```

- **Added vectors join the lists of their nearest centroids,** which stay as they are. That keeps the IVF's
  recall: as a collection grows, even tenfold or into regions its centroids never saw, recall at the
  calibrated nprobe holds or rises. Until the buffer is merged, its vectors are searched exactly and the
  two sets of results merged by score.
- **nprobe is recalibrated at every merge,** on the calibration sample (a fresh sample of the index's own
  vectors, or the given queries), so that the target holds after changes: queries in new regions can fall
  below it otherwise.
- **The cost of inserting without re-clustering is speed, through unbalanced lists,** as new vectors crowd
  into the lists nearest them. So the IVF is re-clustered when its largest list against the mean has
  doubled since its last clustering, or the index has doubled in size (the number of lists grows with it).
  `index.recluster()` does it on request. Re-clustering runs in the `add()` that crosses the threshold.
- **Removed vectors are masked,** and each search fetches 8 more neighbors than asked and drops them; a
  query that lost more than 8 is searched again, fetching as many more as are masked, so results stay
  exact. The index is compacted, and nprobe recalibrated, once 512 are masked.
- **An index can start empty,** from an array of shape (0, d), and grow by `add()`. Until the first merge
  its vectors are in the buffer, searched exactly. With a recall target, the choice between an IVF and
  exact search waits for that merge, as there is nothing to measure before it, and is then made as at a
  build. An index that keeps exact search makes the choice again at a merge once it has doubled in size
  since the last one, so that it turns into an IVF as it grows past where exact search is faster, without
  building an IVF at every merge. Removing every vector returns an index to empty.
- **`index.calibrate(queries)`** gives the calibration queries after the build, for example once real
  queries exist: an IVF recalibrates nprobe and re-measures its routing on them at once, and every later
  merge and choice uses them.
- `index.save()` merges and compacts first, and the file keeps the ids, the options the index was built
  with and the calibration queries. `index.plan` says what changed, when the index last merged and
  re-clustered, and why.

The measurements behind these choices are in BENCHMARKS, [Changing an index](BENCHMARKS.md#changing-an-index).

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
  like L2 without the norms.
- **Cosine:** the angle between vectors. Normalized to unit length, vectors rank by cosine exactly as
  they rank by L2, so cosine search is L2 search on normalized vectors (see [DATASETS](DATASETS.md)).

## The search methods

### Exact search

Exact search computes a distance for every pair of query and base vector, then keeps each query's k
smallest.

**L2 as a matrix multiply.** The squared distance expands to `|q - x|² = |q|² - 2 q·x + |x|²`. The term
`|q|²` is the same for every base vector of a query, so it does not change the ranking and is dropped.
What remains, for a tile of queries Q against all base vectors X, is `|x|² - 2 Q Xᵀ`: one matrix
multiply, which GPUs and the CPU's BLAS do at close to their peak speed, plus the base vectors' squared
norms, computed once. Inner product is the same multiply without the norms.

**L1 and L-infinity** have no such expansion, so every distance is computed coordinate by coordinate,
by kernels laid out like a matrix multiply so that each base vector is read from memory once for many
queries.

**Top-k selection.** With the distances computed, each query needs its k smallest of a million, which is
as costly as the distances unless it is done carefully (see the kernels below).

### IVF: approximate search

An IVF (inverted file) index clusters the base vectors and searches only the clusters nearest the
query. It trades a little recall (the share of the true k nearest neighbors found) for a large cut in
work.

**Building.** k-means (siann's own, on the GPU where there is one: 10 iterations of Lloyd's algorithm on a
sample of 100 vectors per cluster, from randomly chosen starting vectors) finds the cluster centers, the
centroids. Each base vector then goes to the list of its nearest centroid in the search metric, found by
exact search, except for inner product, whose lists are assigned by L2 (by inner product, centroids
pointing towards long vectors attract much of the data and the lists fall badly out of balance). k-means
itself runs in L2, even for an L1 index: the centroids only need to spread over the data. Queries always
probe by the search metric. Where the probed lists hold fewer than k vectors, the missing neighbors are
returned as -1. The GPU and CPU IVFs take their lists from the same build, so they hold the same lists.

**Searching.** A query's `nprobe` nearest centroids (the coarse step) give the lists to scan; the k
nearest vectors among those lists are the answer. Raising `nprobe` raises recall and cost: on SIFT1M
with L2, nprobe 16 finds 84% of the true neighbors, nprobe 64 97%, nprobe 128 99%.

**Query-major and list-major scans.** There are two ways to scan:

```mermaid
flowchart TB
    qs(["queries"]) --> coarse["coarse step: each query's nprobe<br/>nearest centroids"]
    coarse --> mode{"scan"}
    mode -- "query-major" --> qm["each query walks its own lists<br/>128 threads per query on the GPU,<br/>one thread per query on the CPU"]
    mode -- "list-major, large batches" --> grp["group the queries by list"] --> lm["read each list once<br/>for all the queries that probe it"]
    mode -- "split, one query" --> sp["one query's lists over a small<br/>pool of its own, 4 threads"]
    qm --> merge["each query's best k"]
    lm --> merge
    sp --> merge2["merge the threads' best k:<br/>the same result"] --> merge
```

- **Query-major:** each query walks its own lists. Cheap for a single query, but in a batch every query
  reads its lists separately, so a popular list is read from memory many times. FAISS's IVF scans this
  way.
- **List-major:** queries are first grouped by the lists they probe, and each list is scanned once for
  all of them, then each query's results from its lists are merged. In a batch of 10,000 queries at
  nprobe 64, each list is probed by about 160 of them, and read once instead of 160 times.
- **Split single queries:** a single query's coarse step and lists divided among a few threads (a pool of
  their own, 4 by default), each keeping its own best k, merged into the one-thread result, ties included. This cuts a single query's latency, 1.3-2.0 times, at the cost of
  occupying the threads.

Which scan pays depends on the batch and on nprobe, which is why each index measures its routing. Every
scan returns the same neighbors: among vectors at exactly the same distance, the one with the smaller id
comes first, whatever order the lists were scanned in (ties are common on integer data such as SIFT's).

**The coarse step.** Finding the nearest nprobe centroids is itself a small exact search (4,000
centroids), and for a single query on one thread it is about half the work at nprobe 16. Splitting a
single query over 4 threads spreads the coarse step with the lists, so a cheaper coarse step buys
little there ([What does not pay](DEVELOPMENT.md#what-does-not-pay)).

For a batch on the CPU, by L2, cosine or inner product, the coarse step can instead be one BLAS matrix
multiply of the queries against the centroids, whose products Rust ranks as it ranks its own distances,
finding the same lists. The list-major scan always does it this way (1.06-2.0 times faster from 10 queries). For the
query-major scan it depends on the machine: with 4 threads it pays from about 32 queries (up to 1.6 times
faster), with 16 it does not, as each query's own coarse step is already spread over 16 cores. So it is a
route each index measures (BENCHMARKS, [The coarse step](BENCHMARKS.md#the-coarse-step)).

```mermaid
flowchart LR
    b(["a batch, L2 / cosine / ip"]) --> mm["NumPy's BLAS:<br/>queries @ centroids.T"] --> rank["Rust: |c|² - 2 q·c,<br/>each query's nprobe smallest"] --> scan["scan those lists"]
    b2(["L1, L-infinity, or<br/>routed so"]) --> own["Rust: each query's distances<br/>to every centroid"] --> scan
```

**List storage.** The scan reads every vector of every probed list, so in a batch of queries scanned one
by one it is limited by memory traffic, not arithmetic. The lists can therefore be stored in a smaller
type, with distances still computed in float32 from the stored values:

- **float16** (half the bytes): each coordinate rounded to half precision. On SIFT, whose coordinates are
  small integers, this is exact; on GloVe it changes each coordinate by at most 2.4e-4, and recall by at
  most 0.001.
- **uint8** (a quarter): one byte per coordinate, for data whose coordinates are integers from 0 to 255,
  like SIFT's, so nothing is lost. Other data is refused: scaling it into bytes (int8 quantization) would
  lose precision and need re-ranking.

Smaller lists make the query-major scan 1.3-3.0 times as fast in batches of 100, with recall unchanged.
They cost where the scan is not memory-bound: a single query on one thread reads its lists below the
memory system's limit, so widening float16 is pure extra work, while the list-major scans already read
each list once for many queries.

### Graph: approximate search

A graph index links each vector to some of its near neighbors and searches by walking the links from a few
entry vectors, keeping the best candidates found, so a query touches far fewer vectors than an IVF scans.
siann's graphs are CAGRA graphs (the graph index of NVIDIA's cuVS): 32 links per vector, each list chosen
from 64 near neighbors.

**Building.** Where [mlxvs](https://github.com/mikepapadim/mlxvs) is installed (`pip install siann[graph]`,
Apple silicon) and the metric is L2, cosine or inner product, mlxvs builds the graph on the GPU. Everywhere
else siann's own builder does, on the CPU or the GPU, for any of siann's metrics:

1. **Each vector's 64 nearest**, by the index's own IVF searching the base vectors themselves with 16 lists
   probed, on the scan a sample of 4,096 vectors times fastest (on SIFT1M, the CPU's query-major scan on an
   M3 Max, faster there than the GPU, and on 4 Linux CPUs). The result is rough: on SIFT1M it holds 78% of
   each vector's true 64 nearest.
2. **CAGRA's optimization**, in Rust: each candidate link u→v gets a count of its detours, the other
   candidates w of u ranked before v whose own lists rank v before v's rank in u's
   (max(rank(u, w), rank(w, v)) < rank(u, v): v is reachable through w). The 32 links with the fewest
   detours are kept, ties by rank; then the second half of each list is given over to reverse links, the
   vectors that kept this one, those that ranked it highest first, so that every vector can be reached.
   Ranks decide, not distances, so the same code builds a graph for any metric.

Pruning recovers most of what the rough first step misses: on SIFT1M a first step holding 78% of the true
neighbors gives a graph searched as fast at recall 0.95 as one from 96% (64 lists probed; a width of 40
against 35), built in under half the time, and siann's graphs need the same search width as mlxvs's and a
narrower one than cuVS's for the same recall
([Building graphs](BENCHMARKS.md#building-graphs)).

**Searching on the CPU.** A best-first search (siann's Rust, any machine) keeps the `ef` nearest found so
far, as HNSW searches its bottom layer, starting from 256 entry vectors spread through the collection (the
one nearest the mean, and the rest evenly through the rows), since nothing guarantees that a CAGRA graph's
clusters are linked to each other. It reads each candidate's row, one value per cache line, before computing
the distances of a vector's neighbors, four rows at a time, a safe stand-in for prefetching; a query's visited
vectors are marked in a per-thread array of epochs, so nothing is cleared between queries. A single query
runs on the calling thread; small batches on a thread pool sized to the batch. `ef` is calibrated to the
target like nprobe.

**Searching on the GPU** goes through mlxvs's CAGRA search, whichever builder made the graph, at an
`itopk_size` calibrated to the target; mlxvs has L2, cosine and inner product, so a graph for L1 or
L-infinity is searched on the CPU.

**Where a graph fails.** Data made of tight clusters far apart from each other leaves each cluster linked
only to itself, and a search reaches only the clusters its entry vectors fall in: on the quickstart
notebook's 1,000 such clusters, the widest search reaches recall 0.29. The build measures this, and an index
whose graph cannot reach the target declines it (the plan's "no graph index" line) and searches with its IVF,
whose coarse step goes to the right clusters whatever their distance; `kind="graph"` raises instead.

The graph and the IVF find different neighbors, each calibrated to the target on its own, so an index's
results can depend on the batch size it routes; `index.plan` says so. A saved index keeps its graph, and any
machine searches it on the CPU. Adding vectors extends the graph through mlxvs where it is installed;
elsewhere, and after removals or re-clustering, the graph is rebuilt.

## The kernels in detail

### On the GPU

siann's GPU kernels are Metal source, written once (`siann._metal`) and compiled by either GPU backend: MLX
(`mx.fast.metal_kernel`, the faster on Apple silicon) or PyTorch's MPS backend (`torch.mps.compile_shader`),
chosen by `siann.Index(..., gpu="mlx" | "torch")`.

The two-stage top-k, for one query's row of distances to a million base vectors:

```mermaid
flowchart LR
    row["one query's row:<br/>1,000,000 distances"]
    row --> t0["thread 0 reads distances<br/>0, 512, 1,024, …<br/>and keeps its best k<br/>in registers"]
    row --> t1["thread 1 reads<br/>1, 513, 1,025, …"]
    row --> tm["…"]
    row --> t511["thread 511 reads<br/>511, 1,023, 1,535, …"]
    t0 --> s2["stage 2: the best k<br/>among 512 × k survivors"]
    t1 --> s2
    tm --> s2
    t511 --> s2
    s2 --> out["the query's<br/>k nearest"]
```

Every thread reads the row once, in a pattern where neighboring threads read neighboring memory, and keeps
only k values, so nothing but the survivors is written back; a full selection over the row would sort or
partition all of it.

- **A two-stage top-k, the largest single gain.** The frameworks' own selection (`argpartition`, `topk`)
  takes 146-152 ms per tile of 256 queries against a million vectors, eight times the distances. siann
  selects in two stages: 512 threads per query each scan a strided share of the row, keeping their best k
  in registers by insertion, then a small selection over the 512 × k survivors gives the final k: 4.1 ms,
  35 times faster.
- **L1 and L-infinity as a tiled kernel.** Each group of 16 × 16 threads computes a 64 × 64 block of
  distances, 4 × 4 per thread, held in registers. The group loads 16 dimensions of its 64 queries and 64
  base vectors at a time into threadgroup memory (a small, fast memory shared by the threads of a group),
  so each base vector is read from main memory once per 64 queries rather than once per query. The step
  that folds a coordinate difference into the running distance is a sum for L1 and a maximum for
  L-infinity, at the same speed, where FAISS's L-infinity takes three times its own L1 time.
- **L2 as one fused multiply** (`addmm`, which adds a matrix to a scaled product): 6.9 against 17.7 ms per
  tile with the multiply, scaling and subtraction as separate steps.
- **Each tile evaluated before the next.** Queries go 256 at a time, so that only one tile's distances
  (256 × 1M, 1 GB) are held at once. MLX evaluates lazily (it builds a graph of operations and runs it
  when a result is needed), so a search evaluated only at the end keeps every tile's distance matrix
  alive: 196 s instead of 0.9 s for 10,000 queries.
- **The IVF's query-major scan:** 128 threads per query each take a strided share of its lists and keep
  their best k in registers, then merge them pairwise in threadgroup memory (seven steps), so the kernel
  writes only k results per query. Choosing each query's nprobe lists in order costs as much as the scan
  (at nprobe 64 a running top-64 does not fit in registers), and the scan needs them in no order, so the
  GPU backend's own partial selection (MLX's `argpartition`, PyTorch's `topk`) chooses them, at a third of
  the cost.
- **The IVF's list-major scan:** one group of 256 threads per list handles 16 queries at a time, with the
  list's vectors staged through threadgroup memory; the queries are grouped on the GPU too (a sort of the
  probe lists and a count per list). Above nprobe 16 its probe selection is unordered as well; at 16 and
  below the two-stage selection with 32 threads per query is cheaper for thousands of queries.
- **Smaller lists read directly:** the kernels read float16 and uint8 lists without converting them first.

### On the CPU

On the CPU siann runs a small Rust extension, exposed to Python through PyO3 (the bridge that lets Python
call Rust), plus NumPy's matrix multiply, which on a Mac is Apple's Accelerate BLAS (it runs matrix
multiplies on the chip's matrix units).

- **L2 through BLAS, in cache-sized tiles.** NumPy's `@` computes the products of a tile of queries with a
  block of the base, and a Rust top-k ranks them by |x|² - 2 q·x, reading NumPy's arrays in place (through
  rust-numpy, without copying) and keeping each query's best k across blocks. Products of 4-8 million
  values (16-32 MB) are fast; smaller ones pay per-call overhead and larger ones fall out of cache. The
  default takes up to 512 queries and about 4 million values per product. FAISS's exact L2 is the same
  Accelerate multiply, and its own blocks (4,096 queries by 1,024 vectors) fall in the same window, so
  siann's lead over it is not simply its block size.
- **L1 and L-infinity: a blocked layout.** FAISS's flat L1 runs at about 70 G absolute differences per
  second, well below what the cores can do. siann stores the base vectors once, at build time, in blocks
  of 8, dimension by dimension: the first dimension of 8 vectors, then their second, and so on. The inner
  loop compares 4 queries with one block, keeping 32 running sums, one per pair. The sums do not depend
  on each other, so the compiler turns the loop into NEON instructions (Apple's vector instructions, 4
  numbers at a time), 8 absolute differences and 8 additions per dimension, without reordering any
  floating-point additions. (Floating-point addition is not associative, so a compiler may not vectorize
  a single running sum by splitting it; 32 separate sums sidestep the issue.) About 270 G per second.
- **Cache tiles and threads.** Each thread takes 32 queries and scans the base in tiles of 1,024 vectors
  (512 KB), comparing each tile with all of its queries while the tile is in cache. A batch too small to
  occupy every thread this way (below 512 queries on 16 threads) also divides the base among threads, by
  ranges of tiles, and merges each query's best k from the ranges in order of distance, then index, the
  order one thread keeps, so the neighbors are the same: 1.6-10.5 times faster on small batches. Each query keeps its
  best k in a small sorted list. The last block is padded with vectors of infinity, whose distances never
  enter a top-k (for inner product, where infinity would be the best score, padding slots are skipped by
  index). The same kernel computes L2 by squaring the differences and L-infinity with a running maximum,
  written as a comparison and select.
- **The IVF on the CPU:** the query-major scan runs one thread per query; the list-major scan runs each
  list with the exact kernel, 4 queries at a time, while it is in cache. A split single query uses a thread pool of its own: a shared pool costs about 100
  microseconds per query to wake its threads, as FAISS's own split (`parallel_mode = 1`) does.
- **Integer arithmetic for byte lists.** A query whose coordinates are all integers from 0 to 255 is
  compared with byte lists in exact integer arithmetic, over a row-major copy of the lists (each vector's
  bytes contiguous, so that 16 of them fill a register and L2's sum is one dot-product instruction per 16
  values): 1.8 times the float path on one thread, level with FAISS's IVF. Other queries convert the bytes
  to float32 in registers inside the distance loop, so nothing is rounded; with the copy, byte lists take
  two bytes per coordinate, half of float32's.
- **float16 lists.** Rust has no stable float16 type, so float16 lists hold the half-precision bits as
  16-bit integers, and each block is widened to float32 in one flat loop before the distance code runs on
  it. The widening is exact: sign, exponent and mantissa move into float32's positions, read as a number
  2^112 times too small, normal or subnormal, which one multiply corrects; `check_kernels.py` compares it
  with NumPy's conversion on every finite float16.

## Where other libraries are faster, and why

The baseline is always the strongest option for the case: FAISS's on the CPU, its compressed indexes
included, and mlxvs's on the GPU. How much each leads by is in the README's
[How it compares](README.md#how-it-compares); the reasons are these.

- **FAISS's HNSW for single queries and small batches,** by a little: siann's own graph on the CPU runs
  within 4-10% of it on SIFT and Music-100 (single queries and batches of 100), Music-100's single queries,
  answered by the IVF, at 0.72 times, and on 4 Linux CPUs siann leads at single SIFT queries and trails at
  some sizes on both ([The graph through siann.Index](BENCHMARKS.md#the-graph-through-siannindex)). On GloVe-like data a graph must walk much further
  for the same recall, and siann's IVF leads at every batch size.
- **High-dimensional single queries.** At 1,536 dimensions FAISS's HNSW answers single queries 2.8 times as
  fast as siann, which sends them to the graph on the GPU, the path its build measured fastest there: a GPU
  call's fixed cost is most of a single query's time. From 100 queries siann's GPU graph leads 3.9-9.2 times.
- **FAISS's HNSW on vectors of very uneven length.** On raw GloVe (lengths 0.04 to 18.7) no IVF reaches high
  recall, and FAISS's HNSW with inner product is the approximate method; siann's exact search on the GPU,
  4-18 times FAISS's exact search, is the choice where recall must be 1.0
  ([Inner product](BENCHMARKS.md#inner-product)).
- **mlxvs's CAGRA searching alone, in GPU batches,** by 5-14%: siann searches the same graphs through
  mlxvs and then ranks the results itself. At batches of 100 the build's timing has sent SIFT and Music-100
  to the CPU graph where CAGRA on the GPU was 1.17-1.28 times faster. mlxvs has no L1 or L-infinity, and on
  single queries siann's CPU graph is 10-14 times faster than mlxvs's GPU search.

## Code layout

```mermaid
flowchart TB
    user(["Python: NumPy vectors and queries"])

    subgraph api["siann.Index"]
        idx["_index.py: the API, the plan, exact routing"]
        ivfp["_ivf.py: the IVF, calibration, measured routing"]
        graphp["_graph.py, _graph_build.py: the graph"]
        prof["_profile.py: profiles, siann.tune"]
    end

    subgraph gpu["GPU backend: MLX (or PyTorch, the same kernels)"]
        kern["_metal.py, knn_kernels.py<br/>custom Metal kernels:<br/>tiled L1 and L-infinity, two-stage top-k"]
        ivfg["ivf_gpu.py<br/>IVF: query-major and list-major scans"]
    end

    subgraph cpu["CPU"]
        rust["siann._cpu, Rust through PyO3<br/>FlatIndex: exact L1, L2, L-infinity<br/>IvfIndex: float32, float16, uint8 lists; split queries<br/>GraphIndex: graph search; optimize_graph: graph build<br/>L2TopK: ranks BLAS products"]
        blas["NumPy matrix multiply<br/>Accelerate BLAS: exact L2"]
    end

    user --> idx
    idx --> ivfp
    ivfp --> graphp
    graphp --> rust
    graphp -- "GPU search" --> mlxvs(["mlxvs (optional)"])
    idx --> prof
    idx --> kern
    idx --> rust
    idx --> blas
    ivfp --> ivfg
    ivfp --> rust
    ivfg --> kern
    blas -- "products, read in place" --> rust
```

| File | What it holds |
| --- | --- |
| `python/siann/_index.py` | `siann.Index`: building, searching, the plan, exact routing |
| `python/siann/_ivf.py` | The IVF behind `Index(recall=)`: calibration, the timed choice against exact search, measured routing |
| `python/siann/_graph.py`, `_graph_build.py` | The graph index: built by mlxvs or by siann's own builder, searched on the CPU or through mlxvs on the GPU |
| `python/siann/_profile.py` | `siann.Profile`, `siann.tune`, the default profile |
| `python/siann/_memory.py` | The GPU memory budget, MLX's cache cap and whether an index fits on the GPU |
| `python/siann/_build.py` | Checks run before the Rust extension loads: the processor's features, OpenBLAS's threads |
| `python/siann/kmeans.py`, `ivf_build.py` | siann's k-means and the IVF's lists, built once for every backend |
| `python/siann/_metal.py` | The Metal kernels' text, shared by both GPU backends: the two-stage top-k, the tiled L1 and L-infinity kernel, the IVF scans |
| `python/siann/knn_kernels.py`, `exact.py` | Exact search on MLX: the kernels compiled from `_metal`, fused L2, the top-k, a tile of queries at a time |
| `python/siann/ivf_gpu.py` | The GPU IVF: query-major and list-major scans, float16 and uint8 lists |
| `python/siann/blas_l2.py` | Exact L2 and inner product on the CPU through NumPy's BLAS and `L2TopK`, with the default tile sizes |
| `python/siann/torch_kernels.py`, `ivf_torch.py` | The same kernels on PyTorch MPS |
| `crates/siann/` | The engine, a Rust crate with no Python types: the metrics (`metric.rs`), list storage (`elem.rs`), the blocked layout, distance loops and top-k (`kernels.rs`), exact search (`flat.rs`), the IVF (`ivf.rs`), the graph's search (`graph.rs`) and build (`graph_build.rs`), ranking BLAS products (`products.rs`), and scoring, ranking, merging and bookkeeping (`ops.rs`); tested by `cargo test` (`tests.rs`) |
| `crates/siann-py/` | The Python bindings, `siann._cpu`: `FlatIndex`, `IvfIndex`, `GraphIndex`, `L2TopK` and the functions, over the engine |
| `benchmarks/check_kernels.py` | Every kernel and index against NumPy's exact search |
| `benchmarks/` | The benchmarks against FAISS and mlxvs, their raw output in `results/` |

There is no Rust core between Python and the GPU: a GPU backend goes from Python to Metal directly. The
Python package lives in `python/siann/`, and maturin (the tool that builds Rust extensions for Python)
compiles the Rust crate into it as `siann._cpu`; `pyproject.toml` at the top of the repository ties the
two together. FAISS appears only in the benchmarks: siann does not call it. [DEVELOPMENT](DEVELOPMENT.md)
covers building, testing and the rules for changing the code.
