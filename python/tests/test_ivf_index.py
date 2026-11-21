"""siann.Index with a recall target: the IVF, its calibration, its routes, and its overrides."""
import numpy as np
import pytest

import siann

gpu_only = pytest.mark.skipif(not siann.gpu_available(), reason="no GPU available")


def clustered(n, d, seed, integer=False):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((max(20, n // 400), d), dtype=np.float32)
    x = centers[rng.integers(0, len(centers), n)] + 0.3 * rng.standard_normal((n, d), dtype=np.float32)
    if integer:
        x = np.clip(np.rint((x - x.min()) / (x.max() - x.min()) * 255), 0, 255).astype(np.float32)
    return x


def recall(found, truth):
    return np.mean([len(set(a) & set(b)) / len(b) for a, b in zip(found, truth)])


@pytest.mark.parametrize("metric", siann.METRICS)
def test_reaches_its_target_on_new_queries(metric):
    x = clustered(60_000, 32, 1)
    q = clustered(300, 32, 1)[:300] + 0.05            # near the data, not in it
    index = siann.Index(x, metric, recall=0.9, kind="ivf")
    assert index.kind == "ivf" and index.nprobe >= 1
    truth = siann.Index(x, metric).search(q, 10).indices
    assert recall(index.search(q, 10).indices, truth) >= 0.85    # the target, less sampling noise
    full = index.search(q, 10, nprobe=10_000).indices             # every list: exact, up to ties
    assert recall(full, truth) >= 0.99


def test_every_path_gives_the_same_neighbors():
    x = clustered(30_000, 24, 2, integer=True)                    # bytes: lossless on every device
    q = x[:200] + 1
    index = siann.Index(x, "l2", recall=0.9, kind="ivf")
    assert "bytes" in str(index.plan)
    paths = ["cpu-query", "cpu-list", "cpu-split"] + (["gpu-query", "gpu-list"] if siann.gpu_available() else [])
    results = {}
    for path in paths:
        qq = q[:1] if path == "cpu-split" else q
        ids = index._ivf.search(qq, 10, index.nprobe, path)
        results[path] = index._scores(qq, ids)
    for path in paths:
        n = len(results[path])
        np.testing.assert_allclose(results[path], results["cpu-query"][:n])


@gpu_only
def test_routes_by_batch_size():
    index = siann.Index(clustered(30_000, 16, 3), "l2", recall=0.9, kind="ivf")
    assert index._ivf_path(1, None) == "cpu-split" and index.route(1) == "cpu"
    assert index._ivf_path(100, None) == "gpu-query" and index._ivf_path(5000, None) == "gpu-list"
    assert index._ivf_path(5000, "cpu") == "cpu-list" and index._ivf_path(10, "cpu") == "cpu-query"
    p = siann.Profile({"l2": 1}, "test", ivf={"cpu_below": 64, "list_major_from": 2000})
    tuned = siann.Index(clustered(30_000, 16, 3), "l2", recall=0.9, kind="ivf", profile=p)
    assert tuned._ivf_path(50, None) == "cpu-query" and tuned._ivf_path(1000, None) == "gpu-query"


def test_auto_keeps_exact_search_when_it_is_faster():
    index = siann.Index(clustered(800, 8, 4), "l2", recall=0.99)      # tiny: exact search wins
    assert index.kind in ("exact", "ivf")
    text = str(index.plan)
    assert ("kept exact search" in text) == (index.kind == "exact") and "calibrated" in text
    q = clustered(800, 8, 4)[:20]
    np.testing.assert_array_equal(index.search(q, 5).indices.shape, (20, 5))


def test_fewer_than_k_and_overrides():
    x = clustered(5000, 8, 5)
    index = siann.Index(x, "l1", recall=0.8, kind="ivf", nlist=50, storage="float32")
    assert index._ivf.nlist == 50 and index.plan.kind == "ivf"
    r = index.search(x[:3], 10, nprobe=1)
    assert ((r.indices == -1) == np.isinf(r.scores)).all()


def test_errors():
    x = clustered(3000, 8, 6)
    with pytest.raises(ValueError, match="recall"):
        siann.Index(x, "l2", recall=0)
    with pytest.raises(ValueError, match="needs a recall"):
        siann.Index(x, "l2", kind="ivf")
    with pytest.raises(ValueError, match="uint8"):
        siann.Index(x, "l2", recall=0.9, storage="uint8")
    with pytest.raises(ValueError, match="nprobe"):
        siann.Index(x, "l2", recall=0.9, kind="ivf").search(x[:2], 5, nprobe=0)


def test_calibration_queries():
    x = clustered(40_000, 24, 7)
    q = clustered(2000, 24, 8) * 1.1                          # queries unlike the base vectors
    index = siann.Index(x, "l2", recall=0.9, kind="ivf", calibration_queries=q[:1000])
    assert "given calibration queries" in str(index.plan)
    truth = siann.Index(x, "l2").search(q[1000:], 10).indices
    assert recall(index.search(q[1000:], 10).indices, truth) >= 0.86
    with pytest.raises(ValueError, match="calibration_queries"):
        siann.Index(x, "l2", recall=0.9, calibration_queries=np.ones((5, 3)))
