# Tuning

*Part of the [siann documentation](README.md#documentation).*

siann decides where each search runs from thresholds measured on one machine and a few datasets. This
page covers what each threshold decides, its default and where that came from, and how to measure it on
your own data and hardware with `siann.tune`. Tuning changes only speed: every route returns the same
neighbors, checked against exact search.

Here *tuning* means measuring siann's routing on your data and machine. An index's own *calibration*,
finding the settings that reach a recall target, is a different thing and needs no tuning.

## Exact search: GPU or CPU

Where a GPU is available, `siann.Index` sends each exact search to the GPU or the CPU by its batch size: the
CPU below the profile's threshold for the metric, the GPU from it. A GPU call has a fixed cost that a large
batch spreads thin; the CPU's cost grows with the number of vectors searched. So where the two cross
depends on the metric, the hardware and the size of the collection.

**The default** comes from `siann.tune` on an Apple M3 Max (12 performance and 4 efficiency cores, 40-core
GPU) with collections of about a million vectors: SIFT1M for L2, L1 and L-infinity, GloVe-100 for cosine and
inner product, and the first 100,000 DBpedia OpenAI vectors (1,536 dimensions) for cosine and L2. For each
metric it takes the smallest crossover across those datasets, so the CPU gets only batch sizes where it
is faster on every one (`benchmarks/exact_routing.py`, `benchmarks/results/exact_routing.txt`):

| Metric | Default | Queries per second at one query: CPU / GPU |
| --- | --- | --- |
| L2 | CPU for single queries, GPU from 2 | SIFT 155 / 142, DBpedia 175 / 142 |
| L1, L-infinity | GPU always | SIFT 41 / 288 (L1), 30 / 295 (L-infinity) |
| Cosine | GPU always | GloVe 96 / 367, DBpedia 179 / 345 |
| Inner product | GPU always | GloVe 95 / 429 |

**Smaller collections favor the CPU for small batches.** On 200,000 vectors of 128 dimensions (the
quickstart notebook's), the CPU is faster for L2 up to 16 queries per batch, where the default sends every
batch from 2 to the GPU. If your collection is far from a million vectors, or your Mac is not an M3 Max,
tune.

**L1 and L-infinity on the CPU** run a batch of up to 32 queries on one thread (siann's Rust kernel
divides its work among threads 32 queries at a time), so their CPU throughput stays flat from 4 to 32
queries (169 and 121 per second on SIFT1M), and the GPU wins every batch size.

## Approximate search: the recall target and routing

**Calibrate on representative queries (optional).** `siann.Index(xb, metric, recall=0.95)` finds the nprobe
that reaches the target on a sample of 1,000 queries, with nothing more to set. Without
`calibration_queries=`, the sample is the index's own vectors, and the target holds for queries like them; real queries often differ. On SIFT1M its
own vectors gave 0.940 on the real queries, on Music-100 0.986 at five times the nprobe needed; calibrated
on 1,000 real queries, both came within 0.005 of 0.95 on the rest, and Music-100's index ran 2.0-2.8 times
faster (BENCHMARKS.md, "Approximate search through siann.Index"). `search(..., nprobe=)` overrides the calibrated value for one search.

**Routing is measured by each index.** Where an IVF search should run depends on how many lists its
queries probe, so no fixed threshold fits: on SIFT1M at nprobe 46 the CPU is faster up to 16 queries per
batch and the GPU's query-major scan up to 1,024; on GloVe-100 at nprobe 370 the GPU wins from 8
queries and its list-major scan at every size; on Music-100 at nprobe 25 the CPU wins up to 64 and the
GPU's query-major scan at every size measured, up to 4,096 (`benchmarks/ivf_routing.py`). So each index
times its scans on 1 to 4,096 of its calibration queries when it is built, in about one to four seconds
for a million vectors, and the plan reports the thresholds. A measured route replaces the default (single
queries on the CPU, the GPU's query-major scan below 1,000 queries, its list-major scan from 1,000) only
where it was at least 10% faster. Like calibration, the timing is only as representative as the queries:
without `calibration_queries=` it runs on the index's own vectors, which on Music-100 sent batches of 512 to
the list-major scan at 0.77 times the speed of the default on real queries.

To set the thresholds yourself, give a profile thresholds of its own: `siann.tune(xb, recall=0.95)`
measures them on your data, or by hand:

```python
p = siann.Profile(siann.DEFAULT_PROFILE.cpu_below, "my thresholds",
                  ivf={"cpu_below": 32, "gpu_list_from": 2000, "cpu_list_from": 10 ** 9})
index = siann.Index(xb, "l2", recall=0.95, profile=p)
```

`cpu_below` is the batch size from which searches go to the GPU (a single query on the CPU is split over 4
threads), and `gpu_list_from` and `cpu_list_from` the sizes from which each device reads each list once
for all the queries that probe it.

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
  `k=10`. Ask only for the metrics you use: each takes its share of the budget.
- **Time:** `budget=60` seconds in all. Each batch size is timed on each device as the median of three
  calls after a warm-up call, which also compiles the GPU kernels for that shape; a tight budget cuts the
  repeats, not the batch sizes.
- **The threshold** for a metric is the smallest measured batch size from which the GPU is at least as fast
  as the CPU at every larger measured size. Repeated runs differ by up to 15%, so near a crossover the
  threshold can move by a step; a larger budget steadies it.
- **Applying it:** a profile takes effect only where you pass it, with `profile=` or
  `siann.use_profile`, so results never depend on a file you forgot about. `index.plan` names the profile
  and the threshold in use, and `index.route(batch)` says where a batch of that size goes.
- **Pinning a device:** `device="cpu"` or `"gpu"`, for an index or for one search, overrides any profile.

To rerun the measurement behind siann's default, on your own copy of the datasets:

```sh
cd benchmarks && uv run --group bench python exact_routing.py
```
