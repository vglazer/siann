"""A fair retest of LSH on dense vectors (lsh_l1.py tried only Cauchy LSH for L1, without multi-probe).

Two of the strongest LSH families for angular distance, both with multi-probe (probing, beyond each
table's own bucket, the neighboring buckets most likely to hold neighbors, in one order across all
tables), against FAISS's IVF and HNSW:

- **Cross-polytope LSH** (as in FALCONN): rotate the vector by a random rotation and hash it to the
  coordinate of largest magnitude and its sign (2d buckets); k such hashes form a table's key.
  Multi-probe alternatives: the next-largest coordinates, scored by the drop in squared magnitude.
- **SimHash** (hyperplane LSH): k random hyperplanes, one bit each. Multi-probe: flip the bits whose
  projections are smallest, scored by the sum of their magnitudes.

The cost measure is distance computations per query, which does not depend on how carefully each
method is coded: for LSH, the distinct candidates re-ranked by exact distance, plus the hashing
(reported separately: L * k * d dot products for cross-polytope with dense rotations, which FALCONN's
fast pseudo-random rotations make much cheaper; L * k for SimHash); for IVF, the vectors scanned plus
the 4,000 centroids; for HNSW, FAISS's own count. Recall counts ties as correct.

Hashing uses unit-length vectors; re-ranking uses the dataset's metric (on SIFT, L2; its vectors are
nearly unit length up to scale, so angular LSH suits it too).

    uv run --group bench python lsh_retest.py --dataset glove
"""
import argparse
import itertools
import time

import faiss
import mlx.core as mx
import numpy as np
from siann._cpu import FlatIndex

from backends import agreement
from datasets import load

K = 10


class Tables:
    """L hash tables over the base vectors: for each, the sorted keys and the members in that order."""

    def __init__(self, keys: np.ndarray):   # keys: (L, n) int64
        self.order = np.argsort(keys, axis=1, kind="stable")
        self.sorted = np.take_along_axis(keys, self.order, axis=1)

    def candidates(self, probes: list[tuple[int, int]]) -> np.ndarray:
        """The distinct base vectors in the given (table, key) buckets."""
        parts = []
        for t, key in probes:
            lo, hi = np.searchsorted(self.sorted[t], [key, key + 1])
            if hi > lo:
                parts.append(self.order[t, lo:hi])
        return np.unique(np.concatenate(parts)) if parts else np.empty(0, np.int64)


def mlx_project(x: np.ndarray, mats: np.ndarray, chunk: int = 100_000) -> np.ndarray:
    """x @ mats[i] for each i, on the GPU, in chunks: (len(mats), n, cols)."""
    m = mx.array(mats)
    out = []
    for s in range(0, len(x), chunk):
        y = mx.einsum("nd,hdc->hnc", mx.array(x[s:s + chunk]), m)
        mx.eval(y)
        out.append(np.array(y))
    return np.concatenate(out, axis=1)


class CrossPolytope:
    def __init__(self, xb: np.ndarray, k: int, n_tables: int, seed: int = 0, alts: int = 3):
        d = xb.shape[1]
        self.k, self.L, self.d, self.alts = k, n_tables, d, alts
        rng = np.random.default_rng(seed)
        rots = [np.linalg.qr(rng.standard_normal((d, d)))[0] for _ in range(n_tables * k)]
        self.rots = np.stack(rots).astype(np.float32)   # (L * k, d, d)
        keys = np.zeros((n_tables, len(xb)), np.int64)
        for t in range(n_tables):   # one table at a time, to bound memory
            y = mlx_project(xb, self.rots[t * k:(t + 1) * k])   # (k, n, d)
            h = self._hash(y)
            keys[t] = self._key(h)
        self.tables = Tables(keys)

    def _hash(self, y: np.ndarray) -> np.ndarray:
        i = np.abs(y).argmax(axis=-1)
        sign = np.take_along_axis(y, i[..., None], axis=-1)[..., 0] < 0
        return i + self.d * sign

    def _key(self, h: np.ndarray) -> np.ndarray:   # h: (k, ...) in [0, 2d)
        return sum(h[i].astype(np.int64) * (2 * self.d) ** i for i in range(self.k))

    def probes(self, q: np.ndarray, count: int) -> list[tuple[int, int]]:
        """The `count` most promising (table, key) buckets for one query, across all tables."""
        # q @ R, as the base vectors were hashed (x @ R in mlx_project), not R @ q
        y = np.einsum("d,hdc->hc", q, self.rots).reshape(self.L, self.k, self.d)   # (L, k, d)
        mag = np.abs(y)
        top = np.argsort(-mag, axis=-1)[..., :self.alts]   # (L, k, alts)
        sq = np.take_along_axis(mag, top, axis=-1) ** 2
        cost = sq[..., :1] - sq   # 0 for the hash itself
        sign = np.take_along_axis(y, top, axis=-1) < 0
        h = top + self.d * sign   # (L, k, alts) alternative hash values
        scored = []
        for t in range(self.L):
            for combo in itertools.product(range(self.alts), repeat=self.k):
                c = sum(cost[t, i, a] for i, a in enumerate(combo))
                key = sum(int(h[t, i, a]) * (2 * self.d) ** i for i, a in enumerate(combo))
                scored.append((c, t, key))
        scored.sort()
        return [(t, key) for _, t, key in scored[:count]]

    def hash_cost(self) -> int:
        return self.L * self.k * self.d


class SimHash:
    def __init__(self, xb: np.ndarray, k: int, n_tables: int, seed: int = 1, flips: int = 8):
        d = xb.shape[1]
        self.k, self.L, self.flips = k, n_tables, flips
        rng = np.random.default_rng(seed)
        self.planes = rng.standard_normal((n_tables, d, k)).astype(np.float32)
        y = mlx_project(xb, self.planes)   # (L, n, k)
        self.tables = Tables(self._key(y > 0))
        self.subsets = [s for r in range(4) for s in itertools.combinations(range(flips), r)]

    def _key(self, bits: np.ndarray) -> np.ndarray:
        return (bits.astype(np.int64) << np.arange(self.k)).sum(axis=-1)

    def probes(self, q: np.ndarray, count: int) -> list[tuple[int, int]]:
        y = np.einsum("d,ldk->lk", q, self.planes)   # (L, k)
        base = self._key(y > 0)
        low = np.argsort(np.abs(y), axis=1)[:, :self.flips]   # the least certain bits
        scored = []
        for t in range(self.L):
            mags = np.abs(y[t, low[t]])
            for s in self.subsets:
                key = int(base[t])
                for j in s:
                    key ^= 1 << int(low[t, j])
                scored.append((float(mags[list(s)].sum()) if s else 0.0, t, key))
        scored.sort()
        return [(t, key) for _, t, key in scored[:count]]

    def hash_cost(self) -> int:
        return self.L * self.k


def lsh_rows(name: str, lsh, xb_unit: np.ndarray, xq_unit: np.ndarray, xb, xq, truth, metric, probe_counts):
    """Recall and candidates per query for each probe count."""
    d_exact = xb.astype(np.float32)
    for count in probe_counts:
        cand_sizes, found = [], []
        for qi in range(len(xq)):
            c = lsh.tables.candidates(lsh.probes(xq_unit[qi], count))
            cand_sizes.append(len(c))
            if len(c) == 0:
                found.append(np.zeros(K, np.int64))
                continue
            diff = d_exact[c] - xq[qi]
            dist = np.abs(diff).sum(1) if metric == "l1" else (diff * diff).sum(1)
            best = c[np.argsort(dist)[:K]]
            found.append(np.pad(best, (0, K - len(best)), constant_values=best[0]))
        r = agreement(xb, xq, np.array(found), truth, metric)
        n = np.mean(cand_sizes)
        print(f"  {name:32s} probes {count:5d}: recall {r:.3f}, candidates {n:9,.0f} "
              f"({100 * n / len(xb):5.2f}%), hashing {lsh.hash_cost():6,d}", flush=True)


def check() -> bool:
    """Small sanity checks on random unit vectors: every base vector lands in its own bucket (its
    first probe in every table holds it), and more probes give more candidates."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20_000, 32)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    ok = True
    for name, lsh in (("cross-polytope k=2 L=4", CrossPolytope(x, 2, 4)), ("SimHash k=12 L=4", SimHash(x, 12, 4))):
        own = np.mean([i in lsh.tables.candidates(lsh.probes(x[i], 1)) for i in range(200)])
        every = np.mean([all(i in lsh.tables.candidates([pr]) for pr in lsh.probes(x[i], lsh.L))
                         for i in range(200)])
        sizes = [len(lsh.tables.candidates(lsh.probes(x[0], m))) for m in (lsh.L, 4 * lsh.L, 9 * lsh.L)]
        good = own == 1.0 and every == 1.0 and sizes == sorted(sizes) and sizes[0] < sizes[-1]
        ok &= good
        print(f"{name}: own bucket in first probe {own:.2f}, in every table's first probe {every:.2f}, "
              f"candidates by probes {sizes}: {'ok' if good else 'FAILED'}")
    return ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="glove", choices=["glove", "sift"])
    p.add_argument("--queries", type=int, default=500)
    p.add_argument("--nlist", type=int, default=4000)
    p.add_argument("--check", action="store_true", help="run only the sanity checks (seconds)")
    a = p.parse_args()
    if a.check:
        raise SystemExit(0 if check() else 1)
    metric = "l2"
    xb, xq = load(a.dataset, a.queries)
    n, d = xb.shape
    truth = np.array(FlatIndex(xb, d, metric).search(xq, K), dtype=np.int64).reshape(-1, K)
    xb_unit = (xb / (np.linalg.norm(xb, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
    xq_unit = (xq / (np.linalg.norm(xq, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
    print(f"{a.dataset}: {n:,} vectors, {d} dimensions, {len(xq)} queries, recall@{K}; "
          f"cost = distance computations per query")

    # FAISS IVF and HNSW, with FAISS's own distance counts.
    faiss.cvar.distance_compute_blas_threshold = 20
    sample = xb[np.random.default_rng(0).choice(n, min(n, 100 * a.nlist), replace=False)]
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, a.nlist)
    ivf.train(sample)
    ivf.add(xb)
    for nprobe in (8, 16, 32, 64, 128, 256):
        ivf.nprobe = nprobe
        faiss.cvar.indexIVF_stats.reset()
        found = ivf.search(xq, K)[1]
        ndis = faiss.cvar.indexIVF_stats.ndis / len(xq) + a.nlist
        print(f"  {'IVF (4,000 lists)':32s} nprobe {nprobe:4d}: recall {agreement(xb, xq, found, truth, metric):.3f}, "
              f"distances {ndis:9,.0f} ({100 * ndis / n:5.2f}%)", flush=True)
    hnsw = faiss.IndexHNSWFlat(d, 32)
    hnsw.hnsw.efConstruction = 40
    hnsw.add(xb)
    for ef in (16, 32, 64, 128, 256, 512):
        hnsw.hnsw.efSearch = ef
        faiss.cvar.hnsw_stats.reset()
        found = hnsw.search(xq, K)[1]
        ndis = faiss.cvar.hnsw_stats.ndis / len(xq)
        print(f"  {'HNSW (M=32)':32s} efSearch {ef:3d}: recall {agreement(xb, xq, found, truth, metric):.3f}, "
              f"distances {ndis:9,.0f} ({100 * ndis / n:5.2f}%)", flush=True)

    for k, n_tables in ((2, 16), (2, 32), (3, 32)):
        start = time.perf_counter()
        cp = CrossPolytope(xb_unit, k, n_tables)
        print(f"cross-polytope k={k}, L={n_tables}: built in {time.perf_counter() - start:.0f} s", flush=True)
        lsh_rows(f"cross-polytope k={k} L={n_tables}", cp, xb_unit, xq_unit, xb, xq, truth, metric,
                 [n_tables, 4 * n_tables, 16 * n_tables, 64 * n_tables])
        del cp
    for k, n_tables in ((18, 16), (18, 32), (22, 32)):
        start = time.perf_counter()
        sh = SimHash(xb_unit, k, n_tables)
        print(f"SimHash k={k}, L={n_tables}: built in {time.perf_counter() - start:.0f} s", flush=True)
        lsh_rows(f"SimHash k={k} L={n_tables}", sh, xb_unit, xq_unit, xb, xq, truth, metric,
                 [n_tables, 4 * n_tables, 16 * n_tables, 64 * n_tables])
        del sh


if __name__ == "__main__":
    main()
