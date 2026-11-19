# Benchmarks

*Part of the siann documentation: [README](README.md), [ALGORITHMS](ALGORITHMS.md),
[ARCHITECTURE](ARCHITECTURE.md), [BACKENDS](BACKENDS.md), [DATASETS](DATASETS.md), [BENCHMARKS](BENCHMARKS.md); every measurement,
with its script, in [RESULTS](RESULTS.md).*

For each kind of similarity search on a Mac, which path is fastest, and by how much? This page gives
the answer and the main measurements behind it. Every comparison is against FAISS's strongest option
for the case, its compressed indexes included, with exact search returning exactly FAISS's neighbors
and approximate search compared at matched recall. Every number comes from a script in `proto/`
([RESULTS](RESULTS.md) has them all, with the setup).

**Setup.** Apple M3 Max (12 performance and 4 efficiency cores, 64 GB), 16 CPU threads. FAISS 1.16
(CPU, with its BLAS threshold set to 20 and, for its compressed indexes, data padded to a multiple of
8 dimensions; see [BACKENDS](BACKENDS.md)), MLX 0.32.4, PyTorch 2.15.0, JAX 0.11.2 with jax-mps 0.11.1.
k = 10. Datasets: SIFT1M, GloVe-100 and Gaussian vectors ([DATASETS](DATASETS.md)).

## Recommendation

| Workload | Use | Against FAISS's strongest option |
| --- | --- | --- |
| Exact search, batches, GPU available | **siann's GPU kernels** (MLX) | L2 and cosine 8 times FAISS; L1 18-25 times; L-infinity 47-49 times (random vectors); inner product 4-18 times |
| Exact search on the CPU, L1 or L-infinity | **siann's Rust kernel** | 3.8-4.3 times (L1), 7.3 times (L-infinity) |
| Exact search on the CPU, L2, cosine or inner product | **siann's BLAS path** (NumPy's multiply, Accelerate, then siann's Rust top-k) | L2 and cosine 1.7-1.9 times on 1,000 queries, 3.0-3.6 times on 100; inner product level to 2.7 times |
| Approximate, large batches (thousands) | **siann's GPU IVF, list-major** | GloVe: 5.0 times HNSW; SIFT: 1.2-1.3 times |
| Approximate, batches of about 100 | GloVe-like data: **siann's GPU IVF**; SIFT-like data: **FAISS HNSW** below recall 0.99, **siann's GPU IVF** from 0.99 | GloVe: 1.2-1.8 times HNSW; SIFT: HNSW 1.2-1.5 times faster below 0.99, level at 0.99, siann 1.7 times at 0.997 |
| Approximate, single queries, one core each (throughput) | GloVe-like data: **siann's Rust IVF**, two-level coarse step; SIFT-like data: **FAISS HNSW** | GloVe: 1.5-1.65 times HNSW; SIFT: HNSW 1.3-1.45 times faster |
| Approximate, single query, lowest latency with cores to spare | **siann's Rust IVF, split over 4 threads**, except SIFT-like data below recall 0.97 | GloVe: 1.5-3.1 times HNSW; SIFT: 1.2-1.6 times from recall 0.97 |
| Inner product on vectors of uneven length | **siann's exact search on the GPU** (an IVF fails there) | recall 1.0 at 0.88-1.36 times the speed of FAISS's HNSW at recall 0.89 |

For approximate search, store the lists as bytes when every coordinate is an integer from 0 to 255
(lossless), and otherwise as float16 for batches on the GPU (recall moves by at most 0.001) and float32
for single queries. On SIFT-like data, HNSW and the GPU cross over between 100 and 10,000 queries per
batch below recall 0.99, and at about 100 from 0.99; on GloVe-like data, below 100.

**How much faster, exact search** (the speed-up over FAISS's flat index; 1,000 queries; L-infinity on
200,000 Gaussian vectors, inner product on raw GloVe-100; `proto/backends.py`, `proto/blas_l2.py`):

| Metric | siann's GPU | siann's CPU |
| --- | ---: | ---: |
| L-infinity | **47-49x** (random vectors only) | **7.3x** |
| L1 | **18-25x** | **3.8-4.3x** |
| L2 and cosine | **8x** | 1.7-3.6x |
| Inner product | **4-18x** | level to 2.7x |

L1 and L-infinity have no matrix-multiply form, so FAISS computes them with plain loops (its L-infinity
takes three times as long as its L1), where siann runs kernels laid out like a matrix multiply. L2,
cosine and inner product are matrix multiplies, which FAISS runs through BLAS too, so on the CPU siann
gains less there.

**How much faster, approximate search** (siann's GPU IVF against FAISS's HNSW, queries per second, at
the nearest recall; `proto/ivf_cpu.py`, `proto/fp16_lists.py --faiss`, `proto/fine_lists.py`):

| Dataset, batch | Against float32 HNSW | Against compressed HNSW |
| --- | ---: | ---: |
| GloVe, 10,000 queries (recall 0.84) | **6.0x** (191,532 against 32,114) | **5.0x** (199,030 against 40,092) |
| SIFT L2, 10,000 (recall 0.97) | **2.7x** (181,901 against 68,227) | **1.3x** (199,347 against 153,710) |
| SIFT L1, 10,000 (recall 0.96) | **2.1x** (123,855 against 58,059) | (FAISS has no compressed L1) |
| GloVe, batches of 100 (recall 0.84) | **1.4x** (38,744 against 28,088) | **1.8x** (60,559 against 34,348) |
| SIFT L2, batches of 100 | 0.65x at recall 0.97 | 0.84x at 0.97, 0.99x at 0.99, **1.66x** at 0.997 |
| SIFT L1, batches of 100 (recall 0.96) | 0.55x | (FAISS has no compressed L1) |

**What decides it:**

- **Batch size.** The GPU pays a fixed cost per call and needs many queries to fill it; a graph index
  on the CPU (HNSW) touches far fewer vectors per query and wins small batches on SIFT-like data. A
  single query is a contest of per-query work, where splitting it over a few threads trades cores for
  latency.
- **The data.** HNSW touches 7-9 times fewer vectors than an IVF on SIFT but only 4-5 times fewer on
  GloVe, where its scattered reads cost it the difference: siann leads at every batch size on GloVe,
  or draws. On inner product with very uneven vector lengths, an IVF fails outright.
- **Recall.** HNSW's work grows faster with recall than an IVF's, so on SIFT the GPU IVF overtakes it in
  batches of 100 at recall 0.99.
- **The metric,** for exact search: see above.
- **Not the framework.** A custom Metal kernel runs at the same speed in MLX and PyTorch; MLX's edge is
  its faster matrix multiply (exact L2) and smaller cost per call. JAX on a Mac cannot run custom
  kernels at all, and built-in operations lose to FAISS's CPU everywhere.

## Exact search

1,000 queries; the speed-up over FAISS's flat index on the CPU (`proto/backends.py --dataset`,
`proto/blas_l2.py`):

| Backend | SIFT L2 | SIFT L1 | GloVe L2 (cosine) | GloVe L1 |
| --- | ---: | ---: | ---: | ---: |
| FAISS flat (CPU), time | 0.405 s | 2.07 s | 0.431 s | 1.65 s |
| siann, MLX custom kernels (GPU) | 8.7x | 25x | 8.2x | 18.5x |
| siann, PyTorch MPS custom kernels | 4.0x | 25x | 3.9x | 18.4x |
| siann, CPU (Rust for L1; BLAS for L2) | 1.7x | 4.3x | 1.9x | 3.9x |
| Best framework built-in operations (GPU) | 0.65x | 0.48x | 0.54x | 0.43x |
| Best framework built-in operations (CPU) | 0.35x | 0.57x | | |

L-infinity (200,000 Gaussian vectors): siann's GPU kernels 47-49 times FAISS and its Rust kernel 7.3
times. Inner product (raw GloVe-100): siann's GPU 17.7 times FAISS's `IndexFlatIP` on 100 queries, 5.9
times on 1,000 and 4.1 times on 10,000; its CPU BLAS path 2.7 times on 100 and level on 1,000.

## Approximate search

All IVFs, siann's and FAISS's, share the same 4,000 k-means lists, so their recall at each nprobe is
identical. Queries per second; single queries run one per call, on one thread, as HNSW's do.

**float32 lists, against FAISS's HNSW (M = 32) on float32 vectors**, at the nearest recall
(`proto/ivf_cpu.py`):

| Dataset | Recall: IVF / HNSW | Single: Rust IVF / HNSW | Batches of 100: GPU IVF / HNSW | 10,000: GPU IVF / Rust list-major / HNSW |
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

The GPU IVF column is MLX's. PyTorch's port of the same kernels, with the ordered probe selection, is
level with it on large batches at nprobe 16 and below and up to 2.3 times slower above
([RESULTS](RESULTS.md), "IVF on PyTorch MPS"). FAISS's own IVF with the same lists is level with siann's
Rust query-major scan on batches and slower on single queries (2.1 times on SIFT L2 at nprobe 128, 4.5
times at nprobe 4).

**Compressed lists, against FAISS's compressed indexes** (`IndexHNSWSQ` and `IndexIVFScalarQuantizer`,
in the same storages as siann's lists; L2), with siann's two-level coarse step, integer path for byte
lists and split queries for single queries (`proto/fp16_lists.py --faiss`, `proto/fine_lists.py`,
`proto/single_query.py --faiss`, `proto/coarse_step.py`). siann's best against FAISS's best at the
nearest recall (above 1: siann faster):

| Workload | SIFT (uint8 for both) | GloVe (float16, or float32 where faster) |
| --- | --- | --- |
| Batches of 10,000 | **1.3x** at recall 0.97, **1.2x** at 0.84 (GPU) | **5.0x** at 0.84 (GPU) |
| Batches of 100 | 0.67x at 0.83, 0.84x at 0.97, 0.99x at 0.99, **1.66x** at 0.997 (GPU) | **1.24x** at 0.71, **1.76x** at 0.84 (GPU) |
| Single queries, one thread | 0.69x at 0.85 (42 against 29 microseconds), 0.77x at 0.92 (66 against 51) | **1.5x** at 0.71 (58 against 89), **1.65x** at 0.83-0.85 (173 against 285) |
| Single query, split over 4 threads | 0.45x at 0.83, **1.2x** at 0.97 (76 against 94), **1.6x** at 0.99 (111 against 179) | **1.5x** at 0.71, **2.6x** at 0.84, **3.1x** at 0.89 |

FAISS's uint8 IVF rounds the query to integers, which is exact for SIFT's integer queries and
approximate for others; siann's integer path applies only to integer queries, so it is exact
throughout, and on one thread it is level with FAISS's IVF (147 against 136 microseconds at nprobe
64).

**Inner product** on raw GloVe-100 (vector lengths 0.04 to 18.7; `proto/ip_reduction.py`): an IVF
fails, for FAISS's as for siann's (recall below 0.65, the lists badly unbalanced), and FAISS's HNSW with
inner product reaches recall 0.888 at 14,224 queries per second in batches of 100 and 15,275 in
batches of 10,000. siann's exact search on the GPU, at recall 1.0, does about 12,500 (0.88 times) and
20,700 (1.36 times).

## Reproducing

From `proto/`, with the Rust extension built (`cd rust_cpu && VIRTUAL_ENV=../../.venv uvx maturin
develop --release --uv`) and the datasets in `data/`:

```sh
uv run --no-sync --group proto python check_kernels.py                     # every kernel against NumPy
uv run --no-sync --group proto python backends.py --dataset sift --q 1000  # exact search, every backend
uv run --no-sync --group proto python blas_l2.py --dataset sift            # exact L2 on the CPU, BLAS
uv run --no-sync --group proto python backends.py --metrics l1,linf --backends faiss-cpu,siann-rust-cpu,mlx-custom-gpu,torch-custom-mps
uv run --no-sync --group proto python ivf_cpu.py --dataset glove --metric l2      # IVF and HNSW (with PyTorch)
uv run --no-sync --group proto python fp16_lists.py --dataset sift --storages float32,float16,uint8 --faiss
uv run --no-sync --group proto python fine_lists.py --dataset sift              # batches of 100 by recall
uv run --no-sync --group proto python single_query.py --dataset sift --splits 1,4 --faiss
uv run --no-sync --group proto python coarse_step.py --dataset sift              # the coarse step, two-level
uv run --no-sync --group proto python gpu_select_ab.py --dataset sift --lists uint8 --defaults
uv run --no-sync --group proto python backends.py --dataset glove-raw --metrics ip --backends faiss-cpu,mlx-custom-gpu
uv run --no-sync --group proto python ip_reduction.py --dataset glove-raw        # IVF and inner product
```

`--no-sync` keeps uv from removing the locally built Rust extension. An IVF or compressed-index run
takes three to five minutes, an exact-search run on every backend about two.
