# Benchmark scripts

These scripts produce every measurement in [BENCHMARKS.md](../BENCHMARKS.md), and the ones behind
[DEVELOPMENT.md](../DEVELOPMENT.md)'s "What does not pay", [TUNING.md](../TUNING.md)'s defaults and
[DATASETS.md](../DATASETS.md)'s facts. Their raw output is in [results/](results/). Run them from this folder
after `uv sync --group bench` (from the top of the repository), as `uv run --group bench python <script>`;
`--help` lists each one's options. The datasets go in `../data` ([DATASETS.md](../DATASETS.md)).

Times are for an Apple M3 Max, where every number was measured. Runs on the full datasets take from a few
minutes to about an hour; the options to run on fewer vectors (`--base`) or queries (`--queries`) are in
each script's `--help`.

## Comparisons behind the recommendations

The README's "Which to use" table and BENCHMARKS.md's main sections.

| Script | What it measures | BENCHMARKS.md section |
| --- | --- | --- |
| `backends.py` | Exact search on every backend (FAISS, siann's GPU kernels and Rust code, each framework's built-in operations), one process per backend | Exact search on the GPU; on the CPU; Framework built-in operations; Inner product |
| `blas_l2.py` | Exact L2 and inner product on the CPU through NumPy's BLAS and siann's Rust top-k, against FAISS | Exact search on the CPU; Inner product |
| `ivf_cpu.py` | siann's CPU and GPU IVFs against FAISS's IVF and HNSW, at single queries, batches of 100 and one large batch | Approximate search: batches; High-dimensional text embeddings; Inner product |
| `ivf_torch.py` | The PyTorch GPU IVF, run by `ivf_cpu.py` in a process of its own | Approximate search: batches |
| `single_query.py` | Single-query latency: siann's IVF on one thread and split, against FAISS's HNSW and IVF | Approximate search: single queries; List storage |
| `fp16_lists.py` | float16 and uint8 lists against float32, and against FAISS's compressed IVF and HNSW | Approximate search: batches; List storage |
| `faiss_metal.py` | FAISS's Metal GPU path (exact search, IVF-Flat), in stages from tiny to SIFT1M; exact-search calls kept to 1,000 queries | FAISS's Metal path on the GPU |
| `x86_cpu.py` | siann's CPU path against FAISS on synthetic vectors, on any machine; run on GitHub's Linux runners (x86-64 and ARM) by the `x86_bench.yml` workflow, with a baseline x86-64 build alongside the published AVX2 one | On Linux |
| `mlxvs_compare.py` | siann's GPU paths against mlxvs (exact search, IVF-Flat, CAGRA) | Against mlxvs on the GPU; High-dimensional text embeddings |
| `index_recall.py` | `siann.Index(recall=)` on the full datasets: recall on the real queries, what it chose, speed by batch size; `--gpu torch` on PyTorch | Approximate search through siann.Index |
| `graph_faiss.py` | A CAGRA graph built on the GPU, searched by FAISS's HNSW code, against FAISS's own HNSW | A graph built on the GPU, searched on the CPU |
| `builtin_ivf.py` | The IVF with each framework's built-in operations only, against the same IVF on siann's custom kernels, on MLX and PyTorch | The IVF with built-in operations |
| `torch_select_ab.py` | The PyTorch GPU IVF against MLX's on the same lists | PyTorch against MLX |

## Measurements behind siann's defaults

| Script | What it decides | Where |
| --- | --- | --- |
| `exact_routing.py` | Exact search's default profile: where the CPU beats the GPU, by metric and batch size | [TUNING.md](../TUNING.md), Exact search |
| `ivf_routing.py` | Approximate search's routes on each dataset, and how they move with nprobe | Routing approximate search |
| `index_routing_ab.py` | Routing measured per index against fixed thresholds, on the same index | Routing approximate search |
| `drift.py` | An IVF trained on part of a collection with the rest inserted into its lists, against one re-clustered on the same contents: recall, list balance, speed, and the cost of inserting | Changing an index |
| `add_remove.py` | What adding and removing vectors costs: searches with a buffer and with masked rows, merges and compaction | Changing an index |
| `kmeans_check.py` | siann's k-means against FAISS's, as the IVF's centroids | The IVF's k-means |
| `ip_lists.py` | Inner-product lists assigned by L2 rather than by inner product | Inner product |
| `coarse_blas.py` | A batch's coarse step on the CPU through BLAS against the Rust one: same neighbors (`--check`), speed by path and threads, and siann.Index against FAISS's IVF-Flat on the same lists (`--ab`) | The coarse step |
| `coarse_step.py` | The coarse step's share of a single query, and the two-level coarse step | The coarse step |
| `cpu_small_batch.py` | The Rust exact kernel on small batches, dividing the base among threads or not, with outputs checked bit for bit | Exact search on the CPU |
| `compile_cost.py` | What a new batch size costs the GPU kernels (sizes passed at run time) | What a new batch size costs |
| `gpu_select_ab.py` | The GPU IVF's probe selection and threads per query, before and after | Where a GPU call's time goes |

## Ideas that do not pay

Each backs an entry in [DEVELOPMENT.md](../DEVELOPMENT.md#what-does-not-pay)'s "What does not pay".

| Script | What it tried |
| --- | --- |
| `lsh_l1.py` | LSH for L1 (Cauchy random projections) on SIFT1M |
| `blas_ivf.py` | BLAS in the CPU IVF's list-major scan |
| `faer_gemm/`, `numpy_gemm.py` | A matrix multiply in safe Rust (faer) instead of NumPy's BLAS, on exact L2 search's shapes |
| `ip_reduction.py` | Reducing inner product to L2 for an IVF |
| `fine_lists.py` | Finer partitions (16,000 lists) for small batches on the CPU |
| `rust_cpu_sweep.py` | The Rust exact kernel's chunk size and vector instructions, including `f32::max` for L-infinity |
| `kernel_variants.py` | A fused L1 kernel (distance and selection in one pass), checked by `check_kernels.py` |
| `graph_cpu.py` | Searching GPU-built graphs through hnswlib, whose search is slow on Apple silicon (superseded by `graph_faiss.py`) |

## Profiling and analysis

| Script | What it shows | BENCHMARKS.md section |
| --- | --- | --- |
| `profile_l2.py` | Where exact L2 spends its time per tile, in MLX and PyTorch | Exact search on the GPU |
| `gpu_small_batch.py` | Where a GPU IVF call's time goes in small batches | Where a GPU call's time goes |
| `batch_profile.py` | Where a small batch's time goes in the Rust IVF | Where a small batch's time goes on the CPU |
| `profile_batch.py`, `samply_top.py` | A driver for the samply profiler, and a summary of its profile | Where a small batch's time goes on the CPU |
| `dist_counts.py` | Distance computations per query at matched recall, IVF against HNSW | Work per query |
| `faiss_blas.py` | FAISS's BLAS threshold, which slows its L2 below 128 dimensions | FAISS's BLAS threshold |
| `faiss_sq8_query.py` | Whether FAISS rounds the query for its byte lists | List storage |

## Data, checks and shared code

| Script | What it does |
| --- | --- |
| `check_kernels.py` | Every kernel and index against NumPy's exact search; run after any kernel change |
| `ivf_gpu.py` | `--check`: the GPU IVF against NumPy (run by `check_kernels.py`); without it, recall and speed by nprobe |
| `prepare_dbpedia.py` | Builds the DBpedia OpenAI dataset as ann-benchmarks does |
| `dataset_stats.py` | The facts DATASETS.md states about each dataset |
| `datasets.py` | Loads the datasets (shared) |
| `exact.py` | FAISS's exact search and the MLX exact path, shared by the comparisons |
| `timing.py` | Throughput measurement shared by the IVF comparisons |
