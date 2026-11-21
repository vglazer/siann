"""index.save and siann.Index.load: the same results after a round trip, what is kept and what is
re-measured on another machine, and refusals."""
import json

import numpy as np
import pytest

import siann
from siann import _index

gpu_only = pytest.mark.skipif(not siann.gpu_available(), reason="no GPU available")


def clustered(n, d, seed, integer=False):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((max(20, n // 400), d), dtype=np.float32)
    x = centers[rng.integers(0, len(centers), n)] + 0.3 * rng.standard_normal((n, d), dtype=np.float32)
    if integer:
        x = np.clip(np.rint((x - x.min()) / (x.max() - x.min()) * 255), 0, 255).astype(np.float32)
    return x


def same(a, b):
    np.testing.assert_array_equal(a.indices, b.indices)
    np.testing.assert_array_equal(a.scores, b.scores)


@pytest.mark.parametrize("metric", siann.METRICS)
def test_exact_round_trip(tmp_path, metric):
    x = clustered(5_000, 16, 1)
    q = clustered(50, 16, 2)
    index = siann.Index(x, metric)
    index.save(tmp_path / "exact.siann")
    loaded = siann.Index.load(tmp_path / "exact.siann")
    assert loaded.kind == "exact" and loaded.metric == metric and loaded.n == index.n
    for device in ["cpu"] + (["gpu"] if siann.gpu_available() else []):
        same(loaded.search(q, 10, device=device), index.search(q, 10, device=device))


@pytest.mark.parametrize("integer", [False, True])
def test_ivf_round_trip_on_every_path(tmp_path, integer):
    x = clustered(30_000, 24, 3, integer=integer)
    q = x[:100] + 1
    index = siann.Index(x, "l2", recall=0.9, kind="ivf")
    path = tmp_path / "ivf.siann"
    index.save(path)
    loaded = siann.Index.load(path)
    assert loaded.kind == "ivf" and loaded.nprobe == index.nprobe and loaded.recall == 0.9
    assert ("bytes" in str(loaded.plan)) == integer
    paths = ["cpu-query", "cpu-list", "cpu-split"] + (["gpu-query", "gpu-list"] if siann.gpu_available() else [])
    for p in paths:
        np.testing.assert_array_equal(loaded._ivf.search(q, 10, loaded.nprobe, p),
                                      index._ivf.search(q, 10, index.nprobe, p))
    same(loaded.search(q, 10), index.search(q, 10))


def test_bytes_stored_as_bytes(tmp_path):
    x = clustered(20_000, 32, 4, integer=True)
    floats, uint8 = tmp_path / "f.siann", tmp_path / "u.siann"
    siann.Index(x, "l2").save(floats)
    siann.Index(x, "l2", recall=0.9, kind="ivf").save(uint8)
    assert uint8.stat().st_size < 0.5 * floats.stat().st_size


def test_cosine_keeps_unit_vectors(tmp_path):
    x = clustered(5_000, 16, 5) * 7
    q = clustered(20, 16, 6)
    index = siann.Index(x, "cosine")
    index.save(tmp_path / "c.siann")
    same(siann.Index.load(tmp_path / "c.siann").search(q, 5), index.search(q, 5))


def test_calibration_queries_kept(tmp_path):
    x = clustered(30_000, 24, 7)
    cq = clustered(500, 24, 8) + 0.05
    index = siann.Index(x, "l2", recall=0.9, kind="ivf", calibration_queries=cq)
    index.save(tmp_path / "q.siann")
    loaded = siann.Index.load(tmp_path / "q.siann", remeasure=True)
    np.testing.assert_array_equal(loaded._cal_q, index._cal_q)
    assert "re-measured" in str(loaded.plan)


def test_same_machine_keeps_routes_and_profile(tmp_path):
    x = clustered(30_000, 24, 9)
    profile = siann.Profile({"l2": 7}, "my profile")
    index = siann.Index(x, "l2", recall=0.9, kind="ivf", profile=profile)
    index.save(tmp_path / "r.siann")
    loaded = siann.Index.load(tmp_path / "r.siann")
    assert loaded._routes == index._routes
    assert loaded.profile.source == "my profile"
    assert "routing as saved" in str(loaded.plan) and "loaded from r.siann" in str(loaded.plan)


def test_other_machine_remeasures(tmp_path, monkeypatch):
    x = clustered(30_000, 24, 10)
    index = siann.Index(x, "l2", recall=0.9, kind="ivf", profile=siann.Profile({"l2": 7}, "my profile"))
    index.save(tmp_path / "m.siann")
    monkeypatch.setattr(_index, "_machine", lambda: "Another processor (arm64, Darwin)")
    loaded = siann.Index.load(tmp_path / "m.siann")
    plan = str(loaded.plan)
    assert "re-measured on this machine" in plan and "this process's" in plan
    assert loaded.profile.source != "my profile" and loaded.nprobe == index.nprobe
    kept = siann.Index.load(tmp_path / "m.siann", remeasure=False)
    assert kept._routes == index._routes and kept.profile.source == "my profile"


def test_written_at_exactly_the_path(tmp_path):
    siann.Index(clustered(1_000, 8, 11), "l1").save(tmp_path / "no_suffix")
    assert (tmp_path / "no_suffix").exists() and not (tmp_path / "no_suffix.npz").exists()


def test_refuses_other_files(tmp_path):
    np.savez(tmp_path / "other.npz", a=np.zeros(3))
    with pytest.raises(ValueError, match="not a saved siann index"):
        siann.Index.load(tmp_path / "other.npz")
    (tmp_path / "text").write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="not a saved siann index"):
        siann.Index.load(tmp_path / "text")


def test_refuses_a_later_format(tmp_path):
    index = siann.Index(clustered(1_000, 8, 12), "l2")
    index.save(tmp_path / "a.siann")
    data = dict(np.load(tmp_path / "a.siann", allow_pickle=False))
    meta = json.loads(bytes(data["meta"]).decode("utf-8"))
    meta["format_version"] = _index.FORMAT_VERSION + 1
    data["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    with open(tmp_path / "b.siann", "wb") as f:
        np.savez(f, **data)
    with pytest.raises(ValueError, match="upgrade siann"):
        siann.Index.load(tmp_path / "b.siann")


@gpu_only
def test_loads_on_the_cpu_only(tmp_path):
    x = clustered(30_000, 24, 13)
    q = x[:50] + 0.1
    index = siann.Index(x, "l2", recall=0.9, kind="ivf")
    index.save(tmp_path / "g.siann")
    loaded = siann.Index.load(tmp_path / "g.siann", device="cpu")
    assert loaded.route(1000) == "cpu"
    same(loaded.search(q, 10), index.search(q, 10, device="cpu"))
