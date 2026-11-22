# Tuning

*Part of the [siann documentation](README.md#documentation).*

siann makes its choices from measurements, some taken on your data when an index is built and some on one
machine and a few datasets. This page covers what each choice depends on, what it is measured on by
default, and what to do when the defaults do not fit your data or your machine.

Two words name the two kinds of measurement. **Calibration** makes an approximate index meet its recall
target: it finds the fewest lists a query must scan (nprobe) on a sample of queries. **Tuning** measures
where searches run fastest, on the CPU or the GPU and with which scan; it changes only speed, since every
route returns the same neighbors, checked against exact search. The two meet in approximate search, where
the same sample of queries calibrates the index and times its routes.

## What siann decides, and what you can do

| Decision | Measured on, by default | To improve it |
| --- | --- | --- |
| nprobe, for a recall target | 1,000 of the index's own vectors, when it is built | pass `calibration_queries=`, a sample of your queries, or later `index.calibrate(queries)` |
| Exact search or an IVF | the same sample, both timed | pass `calibration_queries=`; `kind="exact"` or `"ivf"` to choose |
| Approximate search: CPU or GPU, which scan | the same sample, at 1 to 4,096 queries per batch | pass `calibration_queries=`; a profile's `ivf` thresholds to fix them |
| Exact search: CPU or GPU | siann's default profile, from an Apple M3 Max and about a million vectors | `siann.tune(xb)` on your data and machine |
| When an index that changes re-clusters | its lists' balance and size at each merge of added vectors | `index.recluster()`, when you choose |

Everything here is optional: `siann.Index(xb, metric)` and `siann.Index(xb, metric, recall=0.95)` need
nothing more. `index.plan` reports each choice, and what it was measured on.

## Calibration: meeting the recall target on your queries

`siann.Index(xb, metric, recall=0.95)` finds the nprobe that reaches the target on a sample of 1,000
queries. Without `calibration_queries=`, the sample is the index's own vectors (each one's own match left
out), and the target holds for queries like them. Real queries often differ, in either direction, and then
calibrating on them pays (a million vectors, target 0.95, recall measured on 9,000 other queries;
[BENCHMARKS](BENCHMARKS.md#approximate-search-through-siannindex)):

| Dataset | Calibrated on its own vectors | Calibrated on 1,000 queries |
| --- | --- | --- |
| SIFT1M, L2 | nprobe 38, recall 0.940: below the target | nprobe 46, recall 0.955 |
| Music-100, inner product | nprobe 129, recall 0.986: five times the lists needed | nprobe 25, recall 0.946, 2.0-2.8 times faster |

Pass queries when you have them, drawn like the ones the index will serve; 1,000 are used. An index built
before you had them, or grown from empty, takes them later: `index.calibrate(queries)` recalibrates nprobe
on them at once and keeps them for every later recalibration. To trade recall
for speed by hand, `search(..., nprobe=)` overrides the calibrated value for one search, and `kind=` picks
exact search or the IVF regardless of which was faster.

## Approximate search: where each batch runs

Where an IVF search runs fastest depends on how many lists its queries scan, so no fixed threshold fits.
On SIFT1M at nprobe 46 the CPU is faster up to 16 queries per batch and the GPU's query-major scan up to
1,024; on GloVe-100 at nprobe 370 the GPU wins from 8 queries and its list-major scan at every size; on
Music-100 at nprobe 25 the CPU wins up to 64 and the GPU's query-major scan at every size measured, up to
4,096 (`benchmarks/ivf_routing.py`).

So each index times its scans when it is built, on 1 to 4,096 of its calibration queries, in about one to
four seconds for a million vectors, and the plan reports the thresholds. A measured route replaces the
default (single queries on the CPU, the GPU's query-major scan below 1,000 queries, its list-major scan
from 1,000) only where it was at least 10% faster. Like calibration, the timing is only as representative
as the queries: on its own vectors, Music-100's index sends batches of 512 to the list-major scan, which
runs at 0.77 times the speed of the default on real queries. Passing `calibration_queries=` fixes both.
An index loaded on another machine re-measures its routes on the same saved sample
([IMPLEMENTATION](IMPLEMENTATION.md#saving-and-loading)).

To fix the thresholds instead, give a profile thresholds of its own: `siann.tune(xb, xq, recall=0.95)`
measures them on your data, or set them by hand:

```python
p = siann.Profile(siann.DEFAULT_PROFILE.cpu_below, "my thresholds",
                  ivf={"cpu_below": 32, "gpu_list_from": 2000, "cpu_list_from": 10 ** 9})
index = siann.Index(xb, "l2", recall=0.95, profile=p)
```

`cpu_below` is the batch size from which searches go to the GPU (a single query on the CPU is split over 4
threads), and `gpu_list_from` and `cpu_list_from` the sizes from which each device reads each list once
for all the queries that probe it. `cpu_blas_from` (L2, cosine and inner product) is the size from which the
CPU's other scan finds each query's lists with one BLAS matrix multiply for the batch; it pays on machines
with few cores, so a profile tuned on one machine may not suit another.

## Exact search: GPU or CPU

Where a GPU is available, `siann.Index` sends each exact search to the GPU or the CPU by its batch size: the
CPU below the profile's threshold for the metric, the GPU from it. A GPU call has a fixed cost that a large
batch spreads thin; the CPU's cost grows with the number of vectors searched. So where the two cross
depends on the metric, the hardware and the size of the collection.

**The default** comes from `siann.tune` on an Apple M3 Max (12 performance and 4 efficiency cores, 40-core
GPU) with collections of about a million vectors: SIFT1M for L2, L1 and L-infinity, GloVe-100 for cosine and
inner product, and the first 100,000 DBpedia OpenAI vectors (1,536 dimensions) for cosine and L2. For each
metric it takes the smallest threshold across those datasets, so the CPU gets only batch sizes where it
is clearly faster on every one (`benchmarks/exact_routing.py`, `benchmarks/results/exact_routing.txt`):

| Metric | Default | Queries per second at one query: CPU / GPU |
| --- | --- | --- |
| L2 | CPU for single queries, GPU from 2 | SIFT 153 / 143, DBpedia 177 / 140 |
| L1, L-infinity | GPU always | SIFT 202 / 204 (L1), 184 / 201 (L-infinity) |
| Cosine | GPU always | GloVe 96 / 361, DBpedia 176 / 354 |
| Inner product | GPU always | GloVe 96 / 355 |

**Smaller collections favor the CPU for small batches.** On 200,000 vectors of 128 dimensions (the
quickstart notebook's), the CPU is faster for L2 up to 16 queries per batch, where the default sends every
batch from 2 to the GPU. If your collection is far from a million vectors, or your Mac is not an M3 Max,
tune.

**L1 and L-infinity stay on the GPU.** On SIFT1M the two devices are level from 1 to 4 queries (L1 at one
query: 202 per second on the CPU, 204 on the GPU), within the spread of repeated runs, and the GPU is
faster from 8 (1,606 against 1,276). A threshold moves only where the other device is at least 10%
faster, so the default keeps both metrics on the GPU; on smaller collections the CPU wins more batch
sizes by a clear margin, and tuning finds them.

## NumPy's BLAS on Linux and Windows

On Linux and Windows, NumPy's BLAS is OpenBLAS, whose threads keep spinning for a while after each call.
siann's CPU searches by L2, cosine and inner product run a BLAS multiply and then Rust code on the same
cores, so spinning threads slow the Rust code down: exact search and an IVF batch whose coarse step goes
through BLAS ([On Linux](BENCHMARKS.md#on-linux)). OpenBLAS's `OPENBLAS_THREAD_TIMEOUT` puts its threads
to sleep sooner, but it is read only once, when NumPy loads. siann sets it to 4 (the shortest wait) unless
it is already set, which takes effect when siann is imported before NumPy:

```python
import siann            # first: OpenBLAS loads with the short timeout
import numpy as np
```

Where NumPy came first, `index.plan` says so. Setting the variable before Python starts works whatever
the order (`export OPENBLAS_THREAD_TIMEOUT=4`). Other code in the same process that calls OpenBLAS many
times in a row may run slightly slower with it, as its threads wake for each call. On a Mac, NumPy uses
Apple's Accelerate, which does not spin, and none of this applies.

## How to tune

```python
import siann

profile = siann.tune(xb)                                # your base vectors; about a minute
print(profile)                                          # thresholds, and the measured rates
index = siann.Index(xb, "l2", profile=profile)         # one index
siann.use_profile(profile)                              # or every index in this process
profile.save("profile.json")                            # and siann.Profile.load("profile.json") later
```

- **Data:** `siann.tune(xb)` times both devices on all of your base vectors, since the crossover depends
  on the collection's size, with a sample of them as queries. Pass `xq=` to time with real queries.
- **What it measures:** `metrics=` (all five by default) at `batches=` (1 to 1,024 queries, doubling), with
  `k=10`. Ask only for the metrics you use: each takes its share of the budget. With `recall=`, it also
  builds an IVF calibrated to that recall (on `xq` if given) and measures its routes at 1 to 4,096
  queries.
- **Time:** `budget=60` seconds in all. Each batch size is timed on each device as the median of three
  calls after a warm-up call, which also compiles the GPU kernels for that shape; a tight budget cuts the
  repeats, not the batch sizes.
- **The threshold** for a metric moves from siann's default only where the other device was at least 10%
  faster: the GPU takes every batch from the smallest measured size from which it wins at every larger
  one, the CPU winning a size only by that margin, and between the sizes measured the default holds.
  Repeated runs differ by up to 15%, so a smaller difference is not acted on.
- **A latency budget:** `print(profile)` lists the measured queries per second of every path at each batch
  size, so you can see which batch sizes fit.
- **Applying it:** a profile takes effect only where you pass it, with `profile=` or
  `siann.use_profile`, so results never depend on a file you forgot about. `index.plan` names the profile
  and the threshold in use, and `index.route(batch)` says where a batch of that size goes.
- **Pinning a device:** `device="cpu"` or `"gpu"`, for an index or for one search, overrides any profile.
- **The GPU backend:** siann's default profile was measured with MLX. With `gpu="torch"`, whose fixed cost
  per call is higher, tune with `siann.tune(xb, gpu="torch")`.

To rerun the measurements behind siann's defaults, on your own copy of the datasets:

```sh
cd benchmarks
uv run --group bench python exact_routing.py    # exact search's default profile
uv run --group bench python ivf_routing.py      # approximate search's routes, by dataset
```
