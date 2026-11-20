# Datasets

*Part of the [siann documentation](README.md#documentation).*

siann is measured on two standard real datasets from
[ann-benchmarks](https://github.com/erikbern/ann-benchmarks), the usual benchmark suite for
approximate nearest-neighbor search, on Music-100, the standard benchmark for maximum inner product
search, on DBpedia OpenAI embeddings, modern 1,536-dimensional text embeddings, and on random Gaussian
vectors. `benchmarks/datasets.py` loads the real ones from `data/` (not
committed): the ann-benchmarks HDF5 files (`sift-128-euclidean.hdf5`, `glove-100-angular.hdf5`) as
published on the ann-benchmarks site, and Music-100's raw files in `data/music-100/`. The figures below
come from `benchmarks/dataset_stats.py`.

| Dataset | Base vectors | Queries | Dimensions | Metric | Values |
| --- | ---: | ---: | ---: | --- | --- |
| SIFT1M | 1,000,000 | 10,000 | 128 | L2 (and L1) | integers, 0 to 218 |
| GloVe-100 | 1,183,514 | 10,000 | 100 | cosine | real numbers, -6.5 to 6.0 |
| Music-100 | 1,000,000 | 10,000 | 100 | inner product | real numbers, -0.70 to 0.75 |
| DBpedia OpenAI | 990,000 | 10,000 | 1,536 | cosine | real numbers, -0.69 to 0.29, unit length |
| Gaussian | 1,000,000 by default | 1,000 by default | 128 by default | L2, L1 | standard normal |

The HDF5 files (and Music-100's) also come with each query's true nearest neighbors in the dataset's
own metric. siann computes its own ground truth by exact search instead, because it also searches in
L1, for which the files have none.

## SIFT1M

SIFT descriptors of image patches: each vector describes the gradients around a point of interest in
an image. It is the classic benchmark for L2 search, and siann also uses it for L1.

- **The coordinates are integers** (0 to 218), so **L1 distances are integers too, and tie often.**
  Two correct searches may then return different neighbors at the k-th place. siann's recall counts
  a returned neighbor as correct when it is no farther than the true k-th nearest, so that a tie
  broken differently is not a miss (`agreement` in `benchmarks/backends.py`).
- **The vectors are nearly normalized:** every length is between 506 and 512. L2 search on SIFT
  therefore ranks almost as cosine search would.
- **128 dimensions,** a multiple of the kernels' 16-dimension tiles, and the size at which FAISS's
  L2 is fast by default (see below).
- **The coordinates fit in a byte,** so IVF lists can be stored as uint8 (a quarter of float32's
  memory) or float16 with nothing lost (see [ALGORITHMS](ALGORITHMS.md), "List storage").
- **The queries are integers too,** so FAISS's uint8 index, which rounds queries to integers, is
  exact on SIFT (see [BACKENDS](BACKENDS.md)).
- **L-infinity distances are integers from 0 to 218 too,** so they tie even more often than L1's; the
  same tie-aware recall applies.

## GloVe-100

Word vectors: GloVe embeddings of 1.2 million words and phrases, trained on Twitter text, 100
dimensions each. Similar words point in similar directions, so the metric is cosine (ann-benchmarks
calls it "angular").

- **Normalized on loading:** `benchmarks/datasets.py` scales every base vector and query to unit length,
  so that L2 ranks as cosine and every L2 method searches cosine unchanged. The raw lengths vary
  widely (0.04 to 18.7); none is zero.
- **A harder dataset for graph indexes:** HNSW walks much further through its graph on GloVe than on
  SIFT for the same recall, and reaches lower recall at each setting. IVF's dense scans suit it
  better, so this is where siann's IVF gains most over HNSW.
- **FAISS's compressed indexes are slow at 100 dimensions:** its scalar quantizer's fast path needs a
  multiple of 8, so siann's comparisons pad GloVe to 104 for them (see [BACKENDS](BACKENDS.md)).
- **float16 lists round every coordinate,** by at most 2.4e-4 after normalizing, which moves IVF
  recall by at most 0.001. uint8 lists do not apply: the coordinates are not integers.
- **100 dimensions,** not a multiple of the kernels' 16-dimension tiles; the kernels pad the last
  tile, and the checks in `benchmarks/check_kernels.py` cover such sizes.
- **FAISS's exact L2 is slow here by default:** below 128 dimensions, FAISS 1.16 skips BLAS unless
  `faiss.cvar.distance_compute_blas_threshold` is lowered, and takes 2.90 s instead of 0.42 s for
  1,000 queries. siann's comparisons set it to 20 (see [BACKENDS](BACKENDS.md)).

## GloVe-100, unnormalized (`glove-raw`)

The same GloVe vectors without the normalization, for inner product. Their lengths run from 0.04 to
18.7, so ranking by inner product differs from ranking by cosine: long vectors dominate every query's
results. That makes it a hard case for maximum inner product search, where an IVF's lists, assigned
by inner product, become badly unbalanced (see [BENCHMARKS](BENCHMARKS.md#inner-product)).

## Music-100 (`music100`)

A million 100-dimensional vectors and 10,000 queries made to be compared by inner product, introduced
with Morozov and Babenko's ip-nsw ("Non-metric Similarity Graphs for Maximum Inner Product Search",
NeurIPS 2018) and loaded by FAISS's own benchmarks (`DatasetMusic100`). The files are on Yandex Disk,
linked from the [ip-nsw repository](https://github.com/stanis-morozov/ip-nsw#dataset)
(`database_music100.bin` and `query_music100.bin`, raw float32, 400 MB and 4 MB); the repository
states no license.

- **The lengths are uneven,** as inner-product data's often are: base vectors run from 0.017 to 2.12
  long (median 0.73, 99th percentile 1.43), the longest 2.9 times the median, close to raw GloVe's 3.4.
  Queries run from 0.0001 to 5.0 (median 0.69). A query's length scales every inner product it takes
  by the same amount, so it does not change its ranking.
- **An IVF's lists, assigned by inner product, are unbalanced** (22% empty), though less than raw
  GloVe's; assigned by L2 they balance, and siann's IVF reaches recall 0.98 (see
  [BENCHMARKS](BENCHMARKS.md#inner-product)).
- **Not integers,** so uint8 lists do not apply; no zero-length vectors.
- **100 dimensions,** as GloVe: not a multiple of the kernels' 16-dimension tiles or of the 8 FAISS's
  compressed indexes need.

## DBpedia OpenAI (`dbpedia`)

OpenAI's text-embedding-ada-002 embeddings of a million DBpedia entities, 1,536 dimensions, compared by
cosine, typical of the vectors retrieval systems search. The source is the Hugging Face dataset
[KShivendu/dbpedia-entities-openai-1M](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M)
(MIT license, 26 Parquet files, 9.5 GB), from which ann-benchmarks builds `dbpedia-openai-1000k-angular`.
`benchmarks/prepare_dbpedia.py` builds it the same way: 10,000 queries split off by scikit-learn's
`train_test_split` with seed 42, leaving 990,000 base vectors, written to
`data/dbpedia-openai-1000k-angular.hdf5`.

- **Already unit length** (0.9999998 to 1.0000001 as published), so cosine and L2 rank alike; the
  loader normalizes anyway, as for GloVe.
- **1,536 dimensions,** twelve times SIFT's: the base takes 6.1 GB as float32, and exact search is
  dominated by its matrix multiply.
- **Measured on a 100,000-vector subset** in these docs (see
  [BENCHMARKS](BENCHMARKS.md#high-dimensional-text-embeddings)).

## Gaussian vectors

Independent standard normal coordinates, from NumPy's generator with seed 0, base vectors first and
then queries (`data` in `benchmarks/backends.py`). Their sizes are free (`--n`, `--d`, `--q`), which
makes them the dataset for timing at sizes and dimensions the real ones lack.

Random vectors are the hardest case for approximate search: in high dimensions every point is
nearly equally far from every other, so there is no cluster structure for IVF or a graph to
exploit. They are used for exact search only, where structure does not matter and the timings match
the real datasets' (see [BENCHMARKS](BENCHMARKS.md#exact-search-on-the-gpu)).
