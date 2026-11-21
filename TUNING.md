# Tuning

*Part of the [siann documentation](README.md#documentation).*

siann decides where each search runs from thresholds measured on one machine and a few datasets. This
page covers what each threshold decides, its default and where that came from, and how to measure it on
your own data and hardware with `siann.tune`. Tuning changes only speed: every route returns the same
neighbors, checked against exact search.

Here *tuning* means measuring siann's routing on your data and machine. An index's own *calibration*,
finding the settings that reach a recall target, is a different thing and needs no tuning.

## Exact search: GPU or CPU

Where MLX finds a GPU, `siann.Index` sends each exact search to the GPU or the CPU by its batch size: the
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
