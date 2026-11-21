"""Profiles: routing by batch size, saving and loading, siann.tune and siann.use_profile."""
import numpy as np
import pytest

import siann
from siann._profile import crossover

gpu_only = pytest.mark.skipif(not siann.gpu_available(), reason="no GPU available")
XB = np.random.default_rng(0).standard_normal((2000, 16), dtype=np.float32)


def test_default_profile():
    p = siann.DEFAULT_PROFILE
    assert p.threshold("l2") == 2 and all(p.threshold(m) == 1 for m in ("l1", "linf", "cosine", "ip"))
    assert "M3 Max" in p.source


@gpu_only
def test_routing_follows_the_profile():
    index = siann.Index(XB, "l2", profile=siann.Profile({"l2": 10}, "test"))
    assert [index.route(b) for b in (1, 9, 10, 500)] == ["cpu", "cpu", "gpu", "gpu"]
    assert "below 10 queries" in str(index.plan) and "test" in str(index.plan)
    always_cpu = siann.Index(XB, "cosine", profile=siann.Profile({"cosine": None}, "test"))
    assert always_cpu.route(10_000) == "cpu"


def test_device_overrides_the_profile():
    index = siann.Index(XB, "l1", device="cpu", profile=siann.Profile({"l1": 1}, "test"))
    assert index.route(10_000) == "cpu"


def test_routes_give_the_same_neighbors():
    q = XB[:40] + 0.01
    routes = [siann.Profile({"l2": t}, "test") for t in (None, 1, 20)]
    results = [siann.Index(XB, "l2", profile=p).search(q, 5) for p in routes]
    for r in results[1:]:
        np.testing.assert_allclose(r.scores, results[0].scores, rtol=1e-4, atol=1e-4)


def test_save_and_load(tmp_path):
    p = siann.Profile({"l2": 8, "cosine": None}, "test", (("l2", "cpu", 1, 10.0), ("l2", "gpu", 1, 5.0)), {"n": 3})
    p.save(tmp_path / "p.json")
    assert siann.Profile.load(tmp_path / "p.json") == p


def test_use_profile():
    p = siann.Profile({"l2": 50}, "process-wide")
    siann.use_profile(p)
    try:
        assert siann.Index(XB, "l2").profile is p
    finally:
        siann.use_profile(None)
    assert siann.Index(XB, "l2").profile is siann.DEFAULT_PROFILE
    with pytest.raises(TypeError):
        siann.use_profile({"l2": 3})


def test_crossover():
    assert crossover({1: 10, 8: 10, 64: 10}, {1: 5, 8: 12, 64: 20}) == 8
    assert crossover({1: 10, 8: 10}, {1: 11, 8: 12}) == 1
    assert crossover({1: 10, 8: 10}, {1: 5, 8: 9}) is None
    assert crossover({1: 10, 8: 10, 64: 10}, {1: 12, 8: 9, 64: 20}) == 64   # must hold for every larger batch


def test_tune():
    p = siann.tune(XB, metrics=("l2", "l1"), batches=(1, 8, 64), budget=5)
    assert set(p.cpu_below) == {"l2", "l1"} and "2,000 x 16" in p.source or "no GPU" in p.source
    for m in ("l2", "l1"):
        assert p.threshold(m) in (None, 1, 8, 64)
    if siann.gpu_available():
        assert len(p.rates) == 2 * 3 * 2
    assert "queries per second" in str(p) or not p.rates
    with pytest.raises(ValueError, match="unknown metric"):
        siann.tune(XB, metrics=("hamming",))
