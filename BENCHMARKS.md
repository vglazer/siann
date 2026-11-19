# Benchmarks

*Part of the siann documentation: [README](README.md), [ALGORITHMS](ALGORITHMS.md),
[ARCHITECTURE](ARCHITECTURE.md), [BACKENDS](BACKENDS.md), [DATASETS](DATASETS.md), [BENCHMARKS](BENCHMARKS.md); every measurement,
with its script, in [RESULTS](RESULTS.md).*

The question: for each kind of similarity search on a Mac, which backend is fastest, and by how
much? This page gives the answer and the main measurements behind it. Every number comes from a
script in `proto/`, and every result was checked against exact search.

**Setup.** Apple M3 Max (12 performance and 4 efficiency cores, 64 GB), 16 CPU threads. FAISS 1.16
(CPU, with its BLAS threshold set to 20; see [BACKENDS](BACKENDS.md)), MLX 0.32.4, PyTorch 2.15.0,
JAX 0.11.2 with jax-mps 0.11.1. k = 10. Datasets: SIFT1M and GloVe-100 ([DATASETS](DATASETS.md)).
Times are medians of three runs after a warm-up, except single queries in the IVF tables (one pass of
1,000 queries).

## Recommendation

| Workload | Use | Measured against the best alternative |
| --- | --- | --- |
| Exact search, batches | **GPU, siann's custom kernels** (MLX; PyTorch is as fast for L1) | 8 times FAISS for L2, 18-25 times for L1 |
| Exact search on the CPU, L1 | **siann's Rust kernel** | 3.8-4.3 times FAISS |
| Exact search on the CPU, L2 (and cosine) | **siann's BLAS path** (NumPy's multiply, which is Accelerate, then siann's Rust top-k) | 1.7-1.9 times FAISS flat on 1,000 queries, 3.0-3.6 times on 100 (`blas_l2.py`) |
| Approximate, one query at a time | **siann's Rust IVF** on GloVe-like data; **FAISS HNSW** on SIFT-like data | GloVe: 1.7-1.9 times HNSW; SIFT: HNSW faster, 1.1 times (L1) and 1.3-1.4 times (L2) |
| Approximate, batches of about 100, or many concurrent queries | **FAISS HNSW** | 1.0-3.0 times the best IVF |
| Approximate, large batches (10,000) | **GPU IVF, list-major** (MLX or PyTorch) | SIFT: 1.1-2.0 times HNSW; GloVe: 2.6-4.6 times |

Where HNSW and the GPU cross over depends on the data: between 100 and 10,000 queries per batch, and
nearer 100 on GloVe, where they are already level at 100 for recall 0.84.

**What decides it:**

- **Batch size.** The GPU pays a fixed cost per call and needs thousands of queries to fill it; a
  graph index on the CPU (HNSW) reads little memory per query and wins in between. Single queries
  are a contest of per-query work on one CPU thread.
- **The data.** On GloVe, HNSW must walk much further through its graph for the same recall, and
  IVF's dense scans gain: siann wins at every batch size there. On SIFT, HNSW is strong up to large
  batches.
- **The metric,** for exact search: L2 is a matrix multiply, where BLAS does the heavy lifting on the
  CPU (for siann as for FAISS, which use the same library, Accelerate); L1 is not, and siann's kernels
  gain far more.
- **Not the framework.** A custom Metal kernel runs at the same speed in MLX and PyTorch. MLX's edge
  is its faster matrix multiply (exact L2) and smaller cost per call. JAX on a Mac cannot run custom
  kernels at all, and built-in operations lose to FAISS's CPU everywhere.

## Exact search

1,000 queries; the speed-up over FAISS's flat index on the CPU. From `proto/backends.py --dataset`;
see [RESULTS](RESULTS.md), "Real data".

| Backend | SIFT L2 | SIFT L1 | GloVe L2 (cosine) | GloVe L1 |
| --- | ---: | ---: | ---: | ---: |
| FAISS flat (CPU), time | 0.405 s | 2.07 s | 0.431 s | 1.65 s |
| siann, MLX custom kernels (GPU) | 8.7x | 25x | 8.2x | 18.5x |
| siann, PyTorch MPS custom kernels | 4.0x | 25x | 3.9x | 18.4x |
| siann, Rust (CPU) | | 4.3x | | 3.9x |
| Best framework built-in operations (GPU) | 0.65x | 0.48x | 0.54x | 0.43x |
| Best framework built-in operations (CPU) | 0.35x | 0.57x | | |

## Approximate search: IVF against HNSW

Each IVF row against FAISS's HNSW (M = 32) at the nearest recall. All IVFs share the same 4,000
k-means lists, so their recall at each nprobe is identical. Queries per second; single queries run on
one thread, as HNSW's do. From `proto/ivf_cpu.py` (raw output in `proto/results/`).

| Dataset | Recall: IVF / HNSW | Single: Rust IVF / HNSW | Batches of 100: GPU IVF / HNSW | 10,000: GPU IVF / Rust list-major / HNSW |
| --- | --- | ---: | ---: | ---: |
| SIFT L2 | 0.925 / 0.937 | 7,862 / 11,330 | 33,313 / 95,579 | 199,112 / 71,867 / 119,917 |
| SIFT L2 | 0.973 / 0.978 | 4,682 / 6,389 | 25,193 / 58,470 | 141,078 / 40,433 / 69,749 |
| SIFT L2 | 0.993 / 0.993 | 2,614 / 3,287 | 12,783 / 32,131 | 44,501 / 22,089 / 38,921 |
| SIFT L1 | 0.904 / 0.898 | 7,528 / 8,383 | 26,722 / 80,580 | 150,092 / 73,073 / 101,050 |
| SIFT L1 | 0.960 / 0.957 | 4,418 / 4,806 | 19,609 / 48,898 | 102,057 / 41,344 / 58,297 |
| SIFT L1 | 0.988 / 0.985 | 2,369 / 2,647 | 10,601 / 28,248 | 38,694 / 22,536 / 32,515 |
| GloVe | 0.780 / 0.781 | 8,959 / 5,402 | 34,337 / 47,519 | 206,607 / 76,521 / 56,891 |
| GloVe | 0.839 / 0.845 | 5,314 / 3,005 | 27,062 / 27,960 | 146,998 / 42,512 / 32,304 |
| GloVe | 0.890 / 0.892 | 3,074 / 1,623 | 13,800 / 16,004 | 46,409 / 23,159 / 17,721 |

The GPU IVF column is MLX's; PyTorch's is within 3% of it on batches of 10,000 and up to a third
slower on smaller calls. FAISS's own IVF with the same lists is level with siann's Rust query-major
scan on batches and 1.3-6.6 times slower on single queries (see [RESULTS](RESULTS.md)).

## L-infinity, compressed lists, single queries and inner product

These come from shorter runs than the tables above (fewer settings per dataset; L-infinity on random
vectors only), and the recommendation table above does not include them, except for exact L2 on the
CPU. Details in [RESULTS](RESULTS.md).

**L-infinity, exact search** (200,000 Gaussian vectors, 1,000 queries): siann's GPU kernels 47-49
times FAISS, and its Rust kernel 7.3 times. FAISS's L-infinity takes three times its L1 time; siann's
GPU kernels take the same time for both. From `proto/backends.py --metrics l1,linf`.

**IVF lists in float16 and uint8** (L2, the same 4,000 lists; speed over float32 lists on the same
path). From `proto/fp16_lists.py`.

| Path | float16, SIFT and GloVe | uint8, SIFT only |
| --- | ---: | ---: |
| Rust, batches of 100 | 1.6-1.8x | 2.2-2.9x |
| GPU, batches of 100 | 1.3-1.6x | 1.7-2.0x |
| GPU, batch of 10,000 | 1.0-1.1x | 1.1x |
| Rust, batch of 10,000 (list-major) | 0.9-1.0x | 1.0-1.1x |
| Rust, single queries | 0.5-0.6x | 0.7-0.8x |

Recall does not change (SIFT is exact in both; on GloVe, float16 moves it by at most 0.001).
Compressed lists help most in batches of about 100. They are slower for single queries on one thread,
so the best storage depends on the batch size. But FAISS compresses too, and its HNSW gains as
much: see "Against FAISS's compressed indexes" below.

**Single queries split over threads** (L2, latency per query, median; HNSW on one thread, in the same
run; from `proto/single_query.py`). Splitting one query's coarse step and lists over a pool of 4
threads cuts the Rust IVF's latency 1.3-2.0 times:

| Dataset, recall (IVF / HNSW) | HNSW | Rust IVF, 1 thread | Rust IVF, 4 threads (best storage) |
| --- | ---: | ---: | ---: |
| SIFT, 0.83 / 0.84 | **53 µs** | 97 µs | 62 µs |
| SIFT, 0.97 | 161 µs | 224 µs | **116 µs** (uint8) |
| SIFT, 0.99 | 299 µs | 391 µs | **155 µs** (uint8, 8 threads) |
| GloVe, 0.71 | 103 µs | 88 µs | **60 µs** |
| GloVe, 0.84 / 0.86 | 346 µs | 203 µs | **111 µs** |
| GloVe, 0.89 / 0.90 | 649 µs | 338 µs | **176 µs** |

Against uncompressed HNSW, as here, the split IVF wins everywhere but SIFT at low recall. Against
FAISS's compressed HNSW, below, it keeps its lead on GloVe and loses it on SIFT. It uses 4-8 cores per
query against HNSW's one.

**Against FAISS's compressed indexes.** The strongest alternatives are FAISS's own compressed indexes
in the same storages: IVF with `IndexIVFScalarQuantizer` (`QT_fp16`, `QT_8bit_direct`, not by residual:
the same bytes as siann's lists) and `IndexHNSWSQ`, given data padded to a multiple of 8 dimensions,
without which FAISS's float16 is 2-6 times slower on GloVe's 100. L2 only. From `proto/fp16_lists.py
--faiss` and `proto/single_query.py --faiss`. siann's best against FAISS's best, at the nearest recall
(above 1: siann faster):

| Workload | SIFT (uint8 for both) | GloVe (float16 for both, or float32 where faster) |
| --- | --- | --- |
| Single query, latency (siann split over 4 threads; HNSW on 1) | 0.45x at recall 0.84, 0.79x at 0.97, 1.0x at 0.99 | **1.5x at 0.71, 2.6x at 0.84, 3.1x at 0.89** |
| Single queries, throughput (one thread each) | (not compared at matched recall) | **1.4x** at 0.84 (Rust float32) |
| Batches of 100 | 0.47x (GPU) at 0.97 | 1.0-1.07x (GPU) |
| Batches of 10,000 | 0.96x at 0.97, **1.25x** at 0.84 (GPU) | **3.8x** at 0.84 (GPU) |

Since then, a two-level coarse step (`proto/coarse_step.py`) has made siann's single queries 1.1-2.1 times
faster on one thread (most at low nprobe): on GloVe it then beats compressed HNSW on one thread (1.5-1.65 times), and on SIFT
it draws level with uncompressed HNSW, while compressed HNSW still leads by about 2 times.

So on SIFT-like data, compressed HNSW leads or draws everywhere but large batches at low recall, and
on GloVe-like data siann leads everywhere but batches of 100, where it draws. FAISS's uint8 IVF was also twice as fast as siann's Rust on one
thread, because it rounds the query to integers and computes in integer arithmetic (exact for SIFT's
integer queries, approximate for others). siann now has an exact integer path for integer queries and
is level with it (147 against 136 microseconds on SIFT at nprobe 64); with the two-level coarse step,
compressed HNSW's single-query lead on SIFT is 1.3-1.45 times on one thread, and siann leads with 4
threads from recall 0.97 ([RESULTS](RESULTS.md), "An integer path for byte lists").

**Inner product** (raw GloVe-100, unnormalized, lengths 0.04 to 18.7; [RESULTS](RESULTS.md), "Inner
product"). Exact search on the GPU is 4.1-17.7 times FAISS's IndexFlatIP (5.9 times on 1,000 queries);
the CPU BLAS path is level with it on 1,000 queries and 2.7 times on 100. IVF fails on this data (recall
below 0.65, for FAISS's IVF too), and FAISS's HNSW with inner product reaches recall 0.888 at about
14,000-15,000 queries per second; siann's exact GPU search is 0.88 times that in batches of 100 and
1.36 times faster in batches of 10,000, at recall 1.0. So for inner product on vectors of uneven
length, the recommendation is exact search on the GPU.

## Reproducing

From `proto/`, with the Rust extension built (`cd rust_cpu && VIRTUAL_ENV=../../.venv uvx maturin
develop --release --uv`) and the datasets in `data/`:

```sh
uv run --no-sync --group proto python check_kernels.py                     # every kernel against NumPy
uv run --no-sync --group proto python backends.py --dataset sift --q 1000  # exact search, every backend
uv run --no-sync --group proto python ivf_cpu.py --dataset glove --metric l2   # IVF and HNSW (with PyTorch)
uv run --no-sync --group proto python backends.py --metrics l1,linf --backends faiss-cpu,siann-rust-cpu,mlx-custom-gpu,torch-custom-mps
uv run --no-sync --group proto python fp16_lists.py --dataset sift --metric l2 --storages float32,float16,uint8
uv run --no-sync --group proto python single_query.py --dataset sift --splits 1,4,8   # about 1.5 minutes
uv run --no-sync --group proto python blas_l2.py --dataset sift                        # exact L2 on the CPU, BLAS
uv run --no-sync --group proto python coarse_step.py --dataset sift                    # the coarse step, two-level
uv run --no-sync --group proto python backends.py --dataset glove-raw --metrics ip --backends faiss-cpu,mlx-custom-gpu
uv run --no-sync --group proto python ip_reduction.py --dataset glove-raw              # IVF and inner product
uv run --no-sync --group proto python fp16_lists.py --dataset sift --storages float32,float16,uint8 --faiss   # about 4 minutes
uv run --no-sync --group proto python single_query.py --dataset sift --splits 1,4 --faiss                     # about 3 minutes
```

`--no-sync` keeps uv from removing the locally built Rust extension. Each IVF run takes about three
minutes, an exact-search run on every backend about two.
