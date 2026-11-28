"""gpu="torch": siann.Index on PyTorch's MPS backend, the same results as on MLX. Skipped where PyTorch is not
installed or finds no Apple GPU (as in CI); the GPU tests run locally before each release."""
import subprocess
import sys

import siann
from siann import _index

import numpy as np
import pytest

torch_only = pytest.mark.skipif(not _index._torch_gpu(), reason="PyTorch with an Apple GPU (MPS) not available")
both = pytest.mark.skipif(not (_index._torch_gpu() and _index._mlx_gpu()), reason="needs MLX and PyTorch on MPS")


def clustered(n, d, seed):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((max(20, n // 400), d), dtype=np.float32)
    return c[rng.integers(0, len(c), n)] + 0.3 * rng.standard_normal((n, d), dtype=np.float32)


def same_up_to_ties(a, b, sa, sb):
    """The same neighbors in every row, or neighbors whose scores tie."""
    np.testing.assert_allclose(sa, sb, rtol=1e-5, atol=1e-5)
    assert (a == b).mean() > 0.99


def test_refuses_an_unknown_backend():
    with pytest.raises(ValueError, match="gpu must be one of"):
        siann.Index(clustered(100, 4, 0), "l2", gpu="cuda")


@both
@pytest.mark.parametrize("metric", siann.METRICS)
def test_exact_search_matches_mlx(metric):
    x, q = clustered(20_000, 16, 1), clustered(200, 16, 2)
    for b in (1, 200):
        t = siann.Index(x, metric, gpu="torch").search(q[:b], 10, device="gpu")
        m = siann.Index(x, metric, gpu="mlx").search(q[:b], 10, device="gpu")
        same_up_to_ties(t.indices, m.indices, t.scores, m.scores)


@both
@pytest.mark.parametrize("metric", ["l2", "l1", "linf", "cosine", "ip"])
def test_the_ivf_on_the_same_lists_matches_mlx(tmp_path, metric):
    x = clustered(30_000, 24, 3)
    q = x[:300] + 0.05
    built = siann.Index(x, metric, recall=0.9, kind="ivf", gpu="mlx")
    built.save(tmp_path / "i.siann")
    on_torch = siann.Index.load(tmp_path / "i.siann", gpu="torch")
    for b in (1, 300):
        t, m = on_torch.search(q[:b], 10, device="gpu"), built.search(q[:b], 10, device="gpu")
        np.testing.assert_allclose(t.scores, m.scores, rtol=1e-3, atol=1e-3)       # MLX ranks in float16
    big = on_torch._ivf.search(q, 10, on_torch.nprobe, "gpu-list")
    assert (big == on_torch._ivf.search(q, 10, on_torch.nprobe, "gpu-query")).mean() > 0.99


@torch_only
def test_builds_searches_and_changes_without_mlx():
    code = """
import sys
import numpy as np
import siann
x = np.random.default_rng(0).standard_normal((20000, 16), dtype=np.float32)
index = siann.Index(x, "l2", recall=0.9, kind="ivf", gpu="torch")
index.search(x[:100], 10, device="gpu")
index.add(x[:50] + 0.1)
index.remove([3])
assert "GPU backend: PyTorch" in str(index.plan)
assert "mlx" not in sys.modules and "mlx.core" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@torch_only
def test_recall_on_torch(tmp_path):
    x = clustered(30_000, 24, 4)
    q = x[:300] + 0.05
    truth = siann.Index(x, "l2").search(q, 10, device="cpu").indices
    index = siann.Index(x, "l2", recall=0.9, kind="ivf", gpu="torch")
    found = index.search(q, 10, device="gpu").indices
    assert np.mean([len(set(a) & set(b)) / 10 for a, b in zip(found, truth)]) >= 0.85
    index.save(tmp_path / "t.siann")
    again = siann.Index.load(tmp_path / "t.siann", gpu="torch").search(q, 10, device="gpu").indices
    np.testing.assert_array_equal(again, found)
