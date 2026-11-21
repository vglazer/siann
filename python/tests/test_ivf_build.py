"""siann's k-means and the IVF's shared lists: the same lists for every backend."""
import numpy as np
import pytest

import siann
from siann._cpu import IvfIndex
from siann.ivf_build import assign_metric, build_lists, nearest
from siann.kmeans import kmeans

RNG = np.random.default_rng(5)
X = RNG.standard_normal((4000, 24), dtype=np.float32)
Q = RNG.standard_normal((30, 24), dtype=np.float32)
DEVICES = ["cpu"] + (["gpu"] if siann.gpu_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_kmeans(device):
    c = kmeans(X, 40, seed=3, device=device)
    assert c.shape == (40, 24) and c.dtype == np.float32 and np.isfinite(c).all()
    np.testing.assert_array_equal(c, kmeans(X, 40, seed=3, device=device))       # reproducible
    assert len(np.unique(nearest(X, c, "l2", device))) == 40                       # no empty cluster
    with pytest.raises(ValueError):
        kmeans(X[:5], 10)


@pytest.mark.parametrize("metric", ["l1", "l2", "linf", "ip"])
def test_nearest_agrees_across_devices(metric):
    if len(DEVICES) < 2:
        pytest.skip("no GPU available")
    c = X[:50]
    np.testing.assert_array_equal(nearest(X, c, metric, "cpu"), nearest(X, c, metric, "gpu"))


@pytest.mark.parametrize("metric", ["l1", "l2", "linf", "ip"])
def test_lists(metric):
    built = build_lists(X, 32, metric, device="cpu")
    assert built.nlist == 32 and built.sizes().sum() == len(X)
    assert assign_metric(metric) == ("l2" if metric == "ip" else metric)
    # stored contiguously by list, each vector once
    np.testing.assert_array_equal(np.sort(built.order), np.arange(len(X)))
    np.testing.assert_array_equal(built.assign[built.order], np.repeat(np.arange(32), built.sizes()))


@pytest.mark.parametrize("metric", ["l1", "l2", "linf", "ip"])
def test_rust_ivf_takes_the_shared_lists(metric):
    built = build_lists(X, 32, metric, device="cpu")
    shared = IvfIndex(X, built.centroids, 24, metric, lists=built.assign)
    assert list(shared.list_sizes()) == list(built.sizes())
    own = IvfIndex(X, built.centroids, 24, metric)      # its own assignment, the same metric
    assert list(own.list_sizes()) == list(built.sizes())
    assert np.array_equal(shared.search(Q, 10, 4), own.search(Q, 10, 4))
    with pytest.raises(ValueError, match="lists"):
        IvfIndex(X, built.centroids, 24, metric, lists=np.full(len(X), 99, dtype=np.int64))
