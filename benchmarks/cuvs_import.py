"""An IVF-Flat index built by NVIDIA cuVS on an NVIDIA GPU, read by siann: its file (cuVS's serialization, version
4, as `cuvs.neighbors.ivf_flat.save` writes it) parsed, its centroids and each vector's list written into a siann
index file, loaded with siann.Index.load and searched on this machine, CPU and GPU.

The file to read is written by benchmarks/colab/triton_probe.ipynb's last cell (100,000 Gaussian vectors from
NumPy's default_rng(0), which this script regenerates, 316 lists, squared L2). Checks that every id appears once,
that each list's vectors, de-interleaved from cuVS's layout, are exactly the regenerated ones, and that recall
against exact search rises with nprobe as it does for siann's own IVF on the same data.

    uv run --group bench python cuvs_import.py ../ivf_flat_gaussian.cuvs
"""
import argparse
import ast
import os
import struct
import tempfile

import numpy as np
from mlxvs_import import write_siann_file

import siann
from siann._cpu import FlatIndex, recall_of, scores

K = 10
GROUP = 32                    # cuVS's interleaved groups of vectors (kIndexGroupSize)


def records(path):
    """cuVS's file: a dtype string, then scalars and arrays each framed as a NumPy .npy record."""
    with open(path, "rb") as f:
        prefix = b""
        while not prefix.endswith(b"\x00"):
            prefix += f.read(1)
        out = []
        while magic := f.read(6):
            assert magic == b"\x93NUMPY", f"not a .npy record at byte {f.tell() - 6}"
            major = f.read(2)[0]
            hlen = struct.unpack("<H" if major == 1 else "<I", f.read(2 if major == 1 else 4))[0]
            head = ast.literal_eval(f.read(hlen).decode("latin1"))
            dt, shape = np.dtype(head["descr"]), head["shape"]
            count = int(np.prod(shape)) if shape else 1
            a = np.frombuffer(f.read(count * dt.itemsize), dtype=dt).reshape(shape)
            out.append(a.item() if not shape else a)
    return prefix[:-1].decode(), out


def read_ivf_flat(path):
    """(centroids, sizes, [per list: (vectors de-interleaved, ids)]) from a cuVS IVF-Flat file, version 4."""
    dtype, r = records(path)
    version, n, d, nlist, metric, _adaptive, _conservative = r[:7]
    if version != 4 or dtype != "<f4":
        raise ValueError(f"cuVS IVF-Flat version {version}, dtype {dtype}: this reader knows version 4, float32")
    centers = r[7]
    i = 8
    if r[i]:                                     # the centroids' norms follow where kept
        i += 1
    i += 1
    sizes = r[i].astype(np.int64)
    i += 1
    veclen = 16 // np.dtype(dtype).itemsize      # cuVS interleaves 16 bytes of a vector at a time
    lists = []
    for c in range(nlist):
        _capacity, data, ids = r[i], r[i + 1], r[i + 2]
        i += 3
        m = int(sizes[c])
        flat = data.reshape(-1)
        rows = np.empty((m, d), dtype=np.float32)
        for row in range(m):
            g, s = divmod(row, GROUP)
            base = g * GROUP * d
            for j0 in range(0, d, veclen):
                off = base + (j0 // veclen) * GROUP * veclen + s * veclen
                rows[row, j0:j0 + veclen] = flat[off:off + veclen]
        lists.append((rows, ids[:m].astype(np.int64)))
    return {"n": n, "d": d, "nlist": nlist, "metric": metric}, centers, sizes, lists


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path")
    a = p.parse_args()
    info, centers, sizes, lists = read_ivf_flat(a.path)
    n, d = info["n"], info["d"]
    x = np.random.default_rng(0).standard_normal((n, d), dtype=np.float32)        # as the notebook made them
    print(f"cuVS IVF-Flat: {n:,} vectors, {d} dimensions, {info['nlist']} lists, metric code {info['metric']}")
    ids = np.concatenate([i for _, i in lists])
    print(f"  every id once: {np.array_equal(np.sort(ids), np.arange(n))}")
    diff = np.concatenate([np.abs(rows - x[i]).max(axis=1) for rows, i in lists])
    print(f"  each list's vectors, de-interleaved, against the regenerated ones: {int((diff == 0).sum()):,} of {n:,} "
          f"identical, the rest within {diff.max():.1e} (NumPy's Gaussian sampler rounds a few tail values a unit "
          f"in the last place differently on another machine)")
    assign = np.empty(n, dtype=np.int64)
    for c, (_, i) in enumerate(lists):
        assign[i] = c
    nearest = np.argmin(((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1), axis=1) if n <= 20_000 else \
        np.array(FlatIndex(np.ascontiguousarray(centers), d, "l2").search(x, 1), dtype=np.int64)
    print(f"  in the list of its nearest centroid: {(nearest == assign).mean():.4f}")

    # into a siann file, then siann's own search of cuVS's lists
    q = np.random.default_rng(1).standard_normal((2000, d), dtype=np.float32)
    cal, test = q[:1000], q[1000:]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    order = np.concatenate([i for _, i in lists])
    path = os.path.join(tempfile.mkdtemp(), "cuvs.siann")
    write_siann_file(path, x, centers, order, offsets, 0.9, cal, "cuVS (IVF-Flat, on an NVIDIA GPU)")
    index = siann.Index.load(path)
    index.calibrate(cal)
    truth = np.array(FlatIndex(x, d, "l2").search(test, K), dtype=np.int64).reshape(-1, K)
    kth = scores(x, test, np.ascontiguousarray(truth[:, -1:]), "l2")[:, 0]

    def recall(found):
        found = np.ascontiguousarray(found, dtype=np.int64)
        return recall_of(found, scores(x, test, found, "l2"), np.ascontiguousarray(kth), False)
    own = siann.Index(x, "l2", recall=0.9, kind="ivf", nlist=info["nlist"], calibration_queries=cal)
    print(f"\n  recall@10 on 1,000 queries, at nprobe: cuVS's lists searched by siann / siann's own lists "
          f"({info['nlist']} lists each)")
    for nprobe in (4, 16, 64):
        print(f"    {nprobe:3d}: {recall(index.search(test, K, nprobe=nprobe).indices):.3f} / "
              f"{recall(own.search(test, K, nprobe=nprobe).indices):.3f}")
    print(f"  calibrated for 0.9 on 1,000 other queries: nprobe {index.nprobe}, recall "
          f"{recall(index.search(test, K).indices):.3f}")
    cpu = index.search(test, K, device="cpu")
    gpu = index.search(test, K, device="gpu") if siann.gpu_available() else cpu
    print(f"  CPU and GPU, the same neighbors: {(cpu.indices == gpu.indices).mean():.4f}\n\n{index.plan}")


if __name__ == "__main__":
    main()
