"""What siann needs of the processor and of NumPy's BLAS, settled before either loads.

OpenBLAS (NumPy's BLAS on Linux and Windows) keeps its threads spinning after each call, which takes the
cores from siann's Rust threads, which run right after it in exact L2 search and in a CPU IVF batch whose
coarse step goes through BLAS (BENCHMARKS, "On Linux"; TUNING, "NumPy's BLAS on Linux and Windows"). OpenBLAS reads
OPENBLAS_THREAD_TIMEOUT once, when NumPy loads, so siann sets it, unless it is set already, before importing
NumPy; if NumPy was imported first, it is too late, and ``index.plan`` says so.


The x86-64 wheels on PyPI are compiled for x86-64-v3 (AVX2 and FMA: Intel from Haswell, 2013, and AMD from
Excavator, 2015, and every Zen), which makes L1 and L-infinity search 1.6-2.4x faster than a baseline
x86-64 build. On a processor without them the extension would crash Python with an illegal instruction, so
siann checks first and raises an ImportError that says why. The release workflow sets X86_FEATURES when it
builds those wheels; a build from source targets the machine that builds it and requires nothing.
"""
import functools
import os
import platform
import sys

OPENBLAS_TIMEOUT_GIVEN = "OPENBLAS_THREAD_TIMEOUT" in os.environ
NUMPY_LOADED_FIRST = "numpy" in sys.modules
os.environ.setdefault("OPENBLAS_THREAD_TIMEOUT", "4")     # its threads sleep after 2^4 cycles, not 2^28

X86_FEATURES: tuple[str, ...] = ()


def check(features: tuple[str, ...] = X86_FEATURES, machine: str | None = None,
          cpu: dict[str, bool] | None = None) -> None:
    """Raise ImportError if this is an x86-64 processor without one of ``features`` (NumPy's names)."""
    machine = (machine or platform.machine()).lower()
    if not features or machine not in ("x86_64", "amd64"):
        return
    if cpu is None:
        try:
            from numpy._core._multiarray_umath import __cpu_features__ as cpu
        except ImportError:
            return                    # NumPy no longer says; let the extension load
    missing = [f for f in features if not cpu.get(f, False)]
    if missing:
        raise ImportError(
            f"this processor lacks {', '.join(missing)}, which siann's x86-64 wheels require (Intel from Haswell, "
            "2013, or AMD from Excavator, 2015). Build siann from source instead, which needs a Rust toolchain: "
            "pip install --no-binary siann siann")



@functools.cache
def openblas_spins() -> bool:
    """Whether NumPy's BLAS is OpenBLAS, loaded before siann without OPENBLAS_THREAD_TIMEOUT: its threads then
    spin after each call."""
    if OPENBLAS_TIMEOUT_GIVEN or not NUMPY_LOADED_FIRST:
        return False
    import numpy as np
    try:
        name = np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    except (KeyError, TypeError, AttributeError):
        return False
    return "openblas" in str(name).lower()


check()
