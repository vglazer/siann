"""What the Rust extension (siann._cpu) was compiled to require of the processor, checked before it loads.

The x86-64 wheels on PyPI are compiled for x86-64-v3 (AVX2 and FMA: Intel from Haswell, 2013, and AMD from
Excavator, 2015, and every Zen), which makes L1 and L-infinity search 1.9-2.4x faster than a baseline
x86-64 build. On a processor without them the extension would crash Python with an illegal instruction, so
siann checks first and raises an ImportError that says why. The release workflow sets X86_FEATURES when it
builds those wheels; a build from source targets the machine that builds it and requires nothing.
"""
import platform

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


check()
