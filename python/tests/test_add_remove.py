"""index.add, index.remove and index.recluster: the same neighbors as an index built from scratch on the same
vectors, ids kept, removed vectors never returned, an IVF's recall kept through merges and re-clustering, and
the changes saved and loaded."""
import json

import numpy as np
import pytest

import siann
from siann import _index
from siann._ivf import default_nlist

gpu_only = pytest.mark.skipif(not siann.gpu_available(), reason="no GPU available")
DEVICES = ["cpu"] + (["gpu"] if siann.gpu_available() else [])


def clustered(n, d, seed, integer=False, centers=None):
    rng = np.random.default_rng(seed)
    c = centers if centers is not None else rng.standard_normal((max(20, n // 400), d), dtype=np.float32)
    x = c[rng.integers(0, len(c), n)] + 0.3 * rng.standard_normal((n, d), dtype=np.float32)
    if integer:
        x = np.clip(np.rint((x - x.min()) / (x.max() - x.min()) * 255), 0, 255).astype(np.float32)
    return x


def recall(found, truth):
    return np.mean([len(set(a) & set(b)) / len(b) for a, b in zip(found, truth)])


@pytest.fixture
def small_limits(monkeypatch):
    """Merges after 500 added vectors and compaction after 8 removed, so that tests reach both."""
    monkeypatch.setattr(_index, "BUFFER_MIN", 500)
    monkeypatch.setattr(_index, "BUFFER_SHARE", 0.0)
    monkeypatch.setattr(_index, "DEAD_MAX", 8)
    monkeypatch.setattr(_index, "FETCH_EXTRA", 2)


@pytest.mark.parametrize("metric", siann.METRICS)
@pytest.mark.parametrize("merged", [False, True])
def test_exact_add_matches_a_fresh_index(metric, merged, small_limits):
    x = clustered(6_000, 16, 1)
    q = clustered(40, 16, 2)
    index = siann.Index(x[:5_000], metric)
    added = index.add(x[5_000:5_100] if not merged else x[5_000:])
    assert list(added[:3]) == [5_000, 5_001, 5_002]
    n = 5_100 if not merged else 6_000
    assert index.n == n and (index._buf is None) == merged
    fresh = siann.Index(x[:n], metric)
    for device in DEVICES:
        ours, ref = index.search(q, 10, device=device), fresh.search(q, 10, device=device)
        np.testing.assert_allclose(ours.scores, ref.scores, rtol=1e-5, atol=1e-5)
        assert (ours.indices == ref.indices).mean() > 0.99            # the rest: near-ties in another order


def test_ids_given_and_refused():
    x = clustered(3_000, 8, 3)
    index = siann.Index(x[:2_000], "l2")
    ids = np.arange(100_000, 101_000)
    assert np.array_equal(index.add(x[2_000:], ids=ids), ids)
    r = index.search(x[2_500], 1)
    assert r.indices[0, 0] == 100_500
    with pytest.raises(ValueError, match="already in the index"):
        index.add(x[:1], ids=[100_000])
    with pytest.raises(ValueError, match="distinct"):
        index.add(x[:2], ids=[7, 7])
    with pytest.raises(ValueError, match="dimensions"):
        index.add(np.zeros((1, 9), dtype=np.float32))
    assert index.add(x[:1])[0] == 101_000                             # continuing from the largest


@pytest.mark.parametrize("compacted", [False, True])
def test_removed_vectors_never_return(compacted, small_limits):
    x = clustered(4_000, 12, 4)
    q = x[:50] + 0.01
    index = siann.Index(x, "l2")
    gone = np.arange(0, 50, 10) if not compacted else np.arange(0, 400, 10)   # 5 masked, or 40: compacted
    assert index.remove(gone) == len(gone)
    assert (index._dead is not None) != compacted
    keep = np.setdiff1d(np.arange(4_000), gone)
    fresh = siann.Index(x[keep], "l2")
    for device in DEVICES:
        ours, ref = index.search(q, 10, device=device), fresh.search(q, 10, device=device)
        assert not np.isin(ours.indices, gone).any()
        np.testing.assert_array_equal(ours.indices, keep[ref.indices])
        np.testing.assert_allclose(ours.scores, ref.scores, rtol=1e-5, atol=1e-5)


def test_queries_near_many_removed_vectors_are_searched_again(small_limits, monkeypatch):
    monkeypatch.setattr(_index, "DEAD_MAX", 1_000)
    x = clustered(5_000, 8, 17)
    q = x[:20] + 0.001
    index = siann.Index(x, "l2")
    near = index.search(q, 30).indices                                 # remove each query's 20 nearest
    gone = np.unique(near[:, :20])
    index.remove(gone)
    assert index._dead is not None                                    # masked, not compacted
    keep = np.setdiff1d(np.arange(5_000), gone)
    ref = siann.Index(x[keep], "l2").search(q, 10)
    for device in DEVICES:
        ours = index.search(q, 10, device=device)
        np.testing.assert_array_equal(ours.indices, keep[ref.indices])


def test_remove_refuses_unknown_ids_and_emptying():
    index = siann.Index(clustered(100, 4, 5), "l1")
    with pytest.raises(KeyError, match="not in the index: 500"):
        index.remove([3, 500])
    assert index.n == 100                                             # nothing removed
    with pytest.raises(ValueError, match="empty"):
        index.remove(np.arange(100))
    index.remove([3])
    with pytest.raises(KeyError):
        index.remove([3])                                             # already gone
    assert index.add(np.ones(4, dtype=np.float32), ids=[3])[0] == 3   # an id removed can be reused


def test_add_and_remove_in_the_buffer():
    x = clustered(2_000, 8, 6)
    index = siann.Index(x[:1_500], "l2")
    index.add(x[1_500:])
    index.remove([1_600, 1_700, 10])
    r = index.search(x[[1_600, 1_701, 10]], 1)
    assert not np.isin(r.indices, [1_600, 1_700, 10]).any() and r.indices[1, 0] == 1_701
    assert "changed since built: 500 added, 3 removed" in str(index.plan)


def test_cosine_added_vectors_normalized():
    x = clustered(2_000, 8, 7)
    index = siann.Index(x[:1_000], "cosine")
    index.add(x[1_000:] * 7)
    fresh = siann.Index(x, "cosine")
    q = clustered(20, 8, 8)
    np.testing.assert_allclose(index.search(q, 5).scores, fresh.search(q, 5).scores, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("metric", ["l2", "ip", "l1"])
def test_ivf_keeps_recall_through_merges(metric, small_limits):
    x = clustered(40_000, 24, 9)
    q = x[:200] + 0.05                       # queries like the vectors, which the IVF is calibrated on
    index = siann.Index(x[:20_000], metric, recall=0.9, kind="ivf")
    for s in range(20_000, 40_000, 5_000):
        index.add(x[s:s + 5_000])
    assert index._buf is None and index._merges >= 4
    assert "nprobe recalibrated" in str(index.plan)
    truth = siann.Index(x, metric).search(q, 10).indices
    assert recall(index.search(q, 10, device="cpu").indices, truth) >= 0.85


def test_drift_reclusters_when_the_lists_fall_out_of_balance(small_limits):
    rng = np.random.default_rng(11)
    x = clustered(20_000, 16, 11)
    far = 6 + 0.05 * rng.standard_normal((20_000, 16), dtype=np.float32)    # one region the centroids never saw
    index = siann.Index(x, "l2", recall=0.9, kind="ivf")
    nlist = index._ivf.nlist
    index.add(far[:5_000])
    assert index._reclusters == 1 and "largest list reached" in str(index.plan)
    assert index._ivf.nlist == default_nlist(25_000) > nlist          # the default for the new size
    both, q = np.concatenate([x, far[:5_000]]), far[5_000:5_200]
    truth = siann.Index(both, "l2").search(q, 10).indices
    fresh = siann.Index(both, "l2", recall=0.9, kind="ivf")          # a tight blob is hard for any IVF
    assert recall(index.search(q, 10).indices, truth) >= recall(fresh.search(q, 10).indices, truth) - 0.02


def test_growth_reclusters_with_more_lists(small_limits):
    x = clustered(30_000, 16, 12)
    index = siann.Index(x[:10_000], "l2", recall=0.9, kind="ivf")
    nlist = index._ivf.nlist
    index.add(x[10_000:])
    assert index._reclusters >= 1 and index._ivf.nlist > nlist


def test_recluster_on_request(small_limits):
    x = clustered(20_000, 16, 13)
    index = siann.Index(x, "l2", recall=0.9, kind="ivf")
    index.remove(np.arange(5))
    index.recluster()
    assert index._dead is None and index._reclusters == 1 and "index.recluster() was called" in str(index.plan)


def test_bytes_become_float_when_a_float_vector_is_added(small_limits):
    x = clustered(10_000, 16, 14, integer=True)
    index = siann.Index(x, "l2", recall=0.9, kind="ivf")
    assert index._storage == "uint8"
    index.add(np.full((600, 16), 0.5, dtype=np.float32))
    assert index._storage == "float32"


@pytest.mark.parametrize("kind", ["exact", "ivf"])
def test_saved_after_changes(tmp_path, kind, small_limits):
    x = clustered(20_000, 16, 15)
    q = x[:30] + 0.02
    index = siann.Index(x[:15_000], "l2", **({} if kind == "exact" else {"recall": 0.9, "kind": "ivf"}))
    index.add(x[15_000:15_100], ids=np.arange(900_000, 900_100))
    index.remove([1, 2, 900_005])
    before = index.search(q, 10)
    index.save(tmp_path / "c.siann")
    assert index._buf is None and index._dead is None                 # merged and compacted to save
    after = index.search(q, 10)
    loaded = siann.Index.load(tmp_path / "c.siann")
    again = loaded.search(q, 10)
    np.testing.assert_array_equal(again.indices, after.indices)
    np.testing.assert_array_equal(again.scores, after.scores)
    if kind == "exact":                                               # exact search: merging changes nothing
        np.testing.assert_allclose(after.scores, before.scores, rtol=1e-5, atol=1e-6)
    assert loaded.n == 15_097 and loaded.add(x[:1])[0] == 900_100


def test_reads_format_1(tmp_path):
    x = clustered(2_000, 8, 16)
    siann.Index(x, "l2").save(tmp_path / "a.siann")
    data = dict(np.load(tmp_path / "a.siann", allow_pickle=False))
    meta = json.loads(bytes(data["meta"]).decode("utf-8"))
    meta["format_version"] = 1
    del meta["clustered"], meta["next_id"]
    data["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    with open(tmp_path / "b.siann", "wb") as f:
        np.savez(f, **data)
    loaded = siann.Index.load(tmp_path / "b.siann")
    assert loaded.add(x[:1])[0] == 2_000
