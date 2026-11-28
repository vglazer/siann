"""The processor check that runs before the Rust extension loads (siann._build)."""
from siann import _build

import pytest

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


def test_openblas_timeout_set_unless_given():
    import os
    assert os.environ.get("OPENBLAS_THREAD_TIMEOUT")


def test_plan_says_when_openblas_spins(monkeypatch):
    import siann

    import numpy as np
    x = np.random.default_rng(0).standard_normal((2_000, 8), dtype=np.float32)
    monkeypatch.setattr(_build, "openblas_spins", lambda: True)
    assert "OPENBLAS_THREAD_TIMEOUT" in str(siann.Index(x, "l2", device="cpu").plan)
    assert "OPENBLAS_THREAD_TIMEOUT" not in str(siann.Index(x, "l1", device="cpu").plan)
    monkeypatch.setattr(_build, "openblas_spins", lambda: False)
    assert "OPENBLAS_THREAD_TIMEOUT" not in str(siann.Index(x, "l2", device="cpu").plan)



def test_importing_numpy_first_warns_only_where_openblas_spins():
    import subprocess
    import sys
    snippet = ("import warnings; warnings.simplefilter('always'); import numpy; import siann; from siann import _build;"
               " print('spins' if _build.openblas_spins() else 'quiet')")
    env = {k: v for k, v in __import__("os").environ.items() if k != "OPENBLAS_THREAD_TIMEOUT"}
    r = subprocess.run([sys.executable, "-c", snippet], check=False, capture_output=True, text=True,
                       encoding="utf-8", env=env)
    assert r.returncode == 0, r.stderr
    assert ("OPENBLAS_THREAD_TIMEOUT=4" in r.stderr) == (r.stdout.strip() == "spins")
    env["OPENBLAS_THREAD_TIMEOUT"] = "4"                              # the user's own setting: no warning
    r = subprocess.run([sys.executable, "-c", snippet], check=False, capture_output=True, text=True,
                       encoding="utf-8", env=env)
    assert "OPENBLAS_THREAD_TIMEOUT=4" not in r.stderr and r.stdout.strip() == "quiet"
