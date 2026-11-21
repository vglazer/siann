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
    paths = ["cpu-query", "cpu-query-blas", "cpu-list", "cpu-split"] + (
        ["gpu-query", "gpu-list"] if siann.gpu_available() else [])
    results = {}
    for path in paths:
        qq = q[:1] if path == "cpu-split" else q
        ids = index._ivf.search(qq, 10, index.nprobe, path)
        results[path] = index._scores(qq, ids)
    for path in paths:
        n = len(results[path])
        np.testing.assert_allclose(results[path], results["cpu-query"][:n])


def test_routes_are_measured_at_build():
    index = siann.Index(clustered(30_000, 16, 3), "l2", recall=0.9, kind="ivf")
    assert "routing measured" in str(index.plan)
    r = index._routes
    assert {"cpu_below", "gpu_list_from", "cpu_list_from", "cpu_blas_from"} <= set(r)
    assert index._ivf_path(1, None) in ("cpu-split", "gpu-query", "gpu-list")
    for b in (1, 16, 300, 5000):                                   # every batch has a path, on its device
        path = index._ivf_path(b, None)
        assert path.split("-")[0] == ("cpu" if b < r["cpu_below"] else "gpu") or not siann.gpu_available()
    assert index._ivf_path(10, "cpu") in ("cpu-query", "cpu-query-blas", "cpu-list")


@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
def test_the_coarse_step_through_blas_finds_the_same_lists(metric):
    x = clustered(30_000, 24, 4)
    q = clustered(300, 24, 5) + 0.05
    if metric == "cosine":
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
    index = siann.Index(x, metric, recall=0.9, kind="ivf", device="cpu")
    ivf = index._ivf
    for path in ("cpu-query", "cpu-list"):
        rust = ivf.search(q, 10, index.nprobe, path, coarse="rust")
        np.testing.assert_array_equal(ivf.search(q, 10, index.nprobe, path, coarse="blas"), rust)
    np.testing.assert_array_equal(ivf.search(q, 10, index.nprobe, "cpu-query-blas"),
                                  ivf.search(q, 10, index.nprobe, "cpu-query"))


def test_ties_go_to_the_smaller_id_on_every_cpu_path():
    x = np.repeat(clustered(2_000, 8, 6, integer=True), 3, axis=0)    # every vector three times, in a row
    q = x[::50] + 1
    index = siann.Index(x, "l1", recall=0.9, kind="ivf", device="cpu")
    ivf, nprobe = index._ivf, index.nprobe
    full = ivf.search(q, 9, nprobe, "cpu-query")
    np.testing.assert_array_equal(ivf.search(q, 9, nprobe, "cpu-list"), full)
    split = np.concatenate([ivf.search(q[i:i + 1], 9, nprobe, "cpu-split") for i in range(len(q))])
    np.testing.assert_array_equal(split, full)
    dist = np.abs(x[full] - q[:, None, :]).sum(axis=2)
    for drow, row in zip(dist, full):                              # equal distances: ids increasing
        for a in range(8):
            if drow[a] == drow[a + 1]:
                assert row[a] < row[a + 1]



@gpu_only
def test_a_profile_overrides_the_measured_routes():
    p = siann.Profile({"l2": 1}, "test", ivf={"cpu_below": 64, "gpu_list_from": 2000, "cpu_list_from": 10 ** 9})
    tuned = siann.Index(clustered(30_000, 16, 3), "l2", recall=0.9, kind="ivf", profile=p)
    assert "routing from the profile" in str(tuned.plan)
    assert tuned._ivf_path(1, None) == "cpu-split" and tuned._ivf_path(50, None) == "cpu-query"
    assert tuned._ivf_path(1000, None) == "gpu-query" and tuned._ivf_path(5000, None) == "gpu-list"


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


def test_routes_leave_the_default_only_for_a_clear_win():
    from siann._ivf import NEVER, crossover_with_margin
    a = {2: 100.0, 4: 100.0, 8: 100.0}
    # b's side from 4 by default: a 5% lead for a does not move it, a 20% lead does
    assert crossover_with_margin(a, {2: 50.0, 4: 96.0, 8: 120.0}, 4) == 4
    # a wins at 4: b from 5, the unmeasured batches above 4 taking the default
    assert crossover_with_margin(a, {2: 50.0, 4: 80.0, 8: 120.0}, 4) == 5
    # a's side everywhere by default: b must be 10% faster to take a batch
    assert crossover_with_margin(a, {2: 105.0, 4: 105.0, 8: 105.0}, NEVER) == NEVER
    assert crossover_with_margin(a, {2: 105.0, 4: 115.0, 8: 130.0}, NEVER) == 4
    # b wins everywhere: from the smallest batch; a wins at the largest: the default beyond it
    assert crossover_with_margin(a, {2: 200.0, 4: 200.0, 8: 200.0}, 4) == 2
    assert crossover_with_margin(a, {2: 200.0, 4: 200.0, 8: 50.0}, 4) == 9
    assert crossover_with_margin(a, {2: 200.0, 4: 200.0, 8: 50.0}, 1000) == 1000
