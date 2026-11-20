"""siann.Index against NumPy's exact search, on every metric and every available device."""
import numpy as np
import pytest

import siann
from siann._index import _gpu_available

DEVICES = ["cpu"] + (["gpu"] if _gpu_available() else [])
gpu_only = pytest.mark.skipif(not _gpu_available(), reason="MLX finds no GPU")


def natural(xb, q, metric):
    """Every query's natural score against every base vector (float64): distances, or similarities."""
    xb, q = xb.astype(np.float64), q.astype(np.float64)
    if metric == "cosine":
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
    if metric in ("cosine", "ip"):
        return q @ xb.T
    diff = q[:, None, :] - xb[None, :, :]
    if metric == "l2":
        return np.sqrt((diff ** 2).sum(axis=2))
    if metric == "l1":
        return np.abs(diff).sum(axis=2)
    return np.abs(diff).max(axis=2)


def check(xb, q, k, metric, res):
    """The returned neighbors are a true top k (ties may break either way), best first, and their
    scores are their exact natural scores."""
    s = natural(xb, q, metric)
    sign = -1 if metric in ("cosine", "ip") else 1
    best = np.sort(sign * s, axis=1)[:, :k] * sign
    got = np.take_along_axis(s, res.indices, axis=1)
    np.testing.assert_allclose(got, best, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(res.scores, got, rtol=1e-4, atol=1e-4)
    assert res.indices.dtype == np.int64 and res.scores.dtype == np.float32
    assert all(len(set(r)) == k for r in res.indices)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("metric", siann.METRICS)
@pytest.mark.parametrize("n,d,m,k", [(2000, 37, 25, 10), (513, 128, 7, 1), (300, 5, 40, 32)])
def test_exact_matches_numpy(device, metric, n, d, m, k):
    rng = np.random.default_rng(n + d)
    xb = rng.standard_normal((n, d), dtype=np.float32)
    q = rng.standard_normal((m, d), dtype=np.float32)
    index = siann.Index(xb, metric, device=device)
    check(xb, q, k, metric, index.search(q, k))


@pytest.mark.parametrize("metric", siann.METRICS)
def test_devices_agree(metric):
    if len(DEVICES) < 2:
        pytest.skip("MLX finds no GPU")
    rng = np.random.default_rng(7)
    xb = rng.standard_normal((3000, 64), dtype=np.float32)
    q = rng.standard_normal((50, 64), dtype=np.float32)
    index = siann.Index(xb, metric)
    a, b = index.search(q, 10, device="cpu"), index.search(q, 10, device="gpu")
    np.testing.assert_allclose(a.scores, b.scores, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("metric", siann.METRICS)
def test_k_beyond_n(device, metric):
    xb = np.random.default_rng(1).standard_normal((5, 8), dtype=np.float32)
    res = siann.Index(xb, metric, device=device).search(xb[:2], 8)
    assert res.indices.shape == (2, 8)
    assert (res.indices[:, 5:] == -1).all() and sorted(res.indices[0, :5]) == list(range(5))
    worst = np.inf if metric in ("l2", "l1", "linf") else -np.inf
    assert (res.scores[:, 5:] == worst).all()


def test_one_query_and_types():
    xb = np.arange(40, dtype=np.int32).reshape(10, 4)      # integers are accepted
    res = siann.Index(xb, "l2", device="cpu").search(xb[3], 1)
    assert res.indices.shape == (1, 1) and res.indices[0, 0] == 3 and res.scores[0, 0] == 0


def test_errors():
    xb = np.zeros((4, 3), dtype=np.float32) + 1
    with pytest.raises(ValueError, match="metric"):
        siann.Index(xb, "hamming")
    with pytest.raises(NotImplementedError, match="recall"):
        siann.Index(xb, "l2", recall=0.9)
    with pytest.raises(ValueError, match="dimensions"):
        siann.Index(xb, "l2", device="cpu").search(np.ones((1, 5)), 1)
    with pytest.raises(ValueError, match="zero vector"):
        siann.Index(np.zeros((2, 3)), "cosine")
    with pytest.raises(ValueError, match="NaN"):
        siann.Index(np.array([[np.nan, 1.0]]), "l2")


def test_sparse_input_is_refused():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    with pytest.raises(NotImplementedError, match="dense"):
        siann.Index(scipy_sparse.random(10, 5, density=0.3, format="csr"), "ip")


def test_plan_explains_the_choice():
    xb = np.ones((6, 4), dtype=np.float32)
    text = str(siann.Index(xb, "cosine", device="cpu").plan)
    assert "6 vectors" in text and "exact" in text and "device: cpu" in text and "unit length" in text


@gpu_only
def test_gpu_plan_by_default():
    assert siann.Index(np.ones((6, 4), dtype=np.float32), "l2").plan.device == "gpu"
