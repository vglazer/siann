"""The processor check that runs before the Rust extension loads (siann._build)."""
import pytest

from siann import _build

V3 = ("AVX2", "FMA3")


def test_source_builds_require_nothing():
    _build.check((), "x86_64", {})


def test_other_architectures_pass():
    _build.check(V3, "arm64", {})
    _build.check(V3, "aarch64", {})


def test_x86_with_the_features_passes():
    _build.check(V3, "x86_64", {"AVX2": True, "FMA3": True})
    _build.check(V3, "AMD64", {"AVX2": True, "FMA3": True})


def test_x86_without_them_says_why():
    with pytest.raises(ImportError, match="lacks AVX2.*Build siann from source"):
        _build.check(V3, "AMD64", {"AVX2": False, "FMA3": True})


def test_numpy_reports_these_names():
    from numpy._core._multiarray_umath import __cpu_features__
    assert all(f in __cpu_features__ for f in V3)
