"""Does FAISS round the query for its uint8 lists (QT_8bit_direct)? Searches an IVF with such lists,
not by residual, with queries that are not integers, and compares FAISS's distances with the exact
distances from the float queries and from the queries rounded to integers. If they match the rounded
ones, FAISS computes in integer arithmetic on a rounded query: exact for integer queries (SIFT's), an
approximation otherwise.

    uv run --group bench python faiss_sq8_query.py
"""
import faiss
import numpy as np

rng = np.random.default_rng(0)
x = rng.integers(0, 200, (5000, 32)).astype(np.float32)
q = (rng.integers(0, 200, (3, 32)) + 0.3).astype(np.float32)   # not integers
quantizer = faiss.IndexFlatL2(32)
quantizer.add(x[:16])
ix = faiss.IndexIVFScalarQuantizer(quantizer, 32, 16, faiss.ScalarQuantizer.QT_8bit_direct, faiss.METRIC_L2)
ix.by_residual = False
ix.train(x[:1000])
ix.add(x)
ix.nprobe = 16
D, I = ix.search(q, 3)
exact = ((q[:, None] - x[I]) ** 2).sum(-1)
rounded = ((np.rint(q)[:, None] - x[I]) ** 2).sum(-1)
print(f"faiss {faiss.__version__}, IndexIVFScalarQuantizer QT_8bit_direct, queries with fraction 0.3")
print("FAISS distances:          ", D[0])
print("exact, float query:       ", exact[0])
print("exact, query rounded:     ", rounded[0])
print("FAISS matches the rounded query:", bool(np.allclose(D, rounded)), "; the float query:", bool(np.allclose(D, exact)))
