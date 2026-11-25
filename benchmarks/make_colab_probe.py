"""Writes benchmarks/colab/triton_probe.ipynb, a notebook to run on Colab or Kaggle with an NVIDIA GPU: it installs
FAISS's and cuVS's GPU packages, writes out triton_probe.py (embedded here, so the two cannot drift apart), checks
the Triton kernel against exact search on the GPU, benchmarks it against torch.cdist, FAISS and cuVS, and saves a
cuVS IVF-Flat index of seeded Gaussian vectors (which siann can regenerate) for download, to check that siann
reads cuVS's file format.

    python make_colab_probe.py
"""
import json
from pathlib import Path

from colab_cells import notebook, numbered

HERE = Path(__file__).parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


def main() -> None:
    probe = (HERE / "triton_probe.py").read_text(encoding="utf-8")
    cells = [
        md("""
# siann's Triton probe

Run on Colab with a T4 GPU (Runtime, Change runtime type, T4 GPU): Colab comes with PyTorch, Triton and cuVS, so
the default steps install almost nothing. **Steps 1-4, about 5 minutes in all:** siann's exact L1 kernel, written
in Triton, timed against cuVS's brute force on 200,000 Gaussian vectors, with the memory each takes; the run
stops itself at its time budget. Steps 5 and later are optional: FAISS (its install takes a few minutes), the
kernels' check against exact search, and the long tuning. Copy the output of each step you run back.
"""),
        code("!nvidia-smi --query-gpu=name,memory.total --format=csv"),
        code('''
# small packages for the memory figures; cuVS only where missing (Colab comes with it), for this CUDA
import subprocess
import sys

import torch

cuda = torch.version.cuda.split(".")[0]
print("PyTorch", torch.__version__, "for CUDA", torch.version.cuda)


def works(stmt):
    return subprocess.run([sys.executable, "-c", stmt], check=False, capture_output=True).returncode == 0


def install(pkg):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg, "--extra-index-url=https://pypi.nvidia.com"],
                       check=False, capture_output=True, text=True)
    print(pkg, "installed" if r.returncode == 0 else "not installed: " + (r.stderr.strip().splitlines() or ["?"])[-1])
    return r.returncode == 0


for pkg in ("nvidia-ml-py", "psutil"):
    install(pkg)
if not works("from cuvs.neighbors import brute_force"):
    print("cuVS missing: installing it takes several minutes (Colab comes with it)")
    install(f"libcuvs-cu{cuda}")
    install(f"cuvs-cu{cuda}")
print("cuVS", "works" if works("from cuvs.neighbors import brute_force") else "does not work: the benchmark skips it")
'''),
        code("%%writefile triton_probe.py\n" + probe),
        code("!python -u triton_probe.py --bench --budget 180"),
        code('''
# FAISS's GPU package for this CUDA (it publishes no CUDA 13 build: its CUDA 12 one brings its own runtime)
import subprocess  # noqa: F811 (again here, so that this step runs on its own too)
import sys  # noqa: F811

import torch

cuda = torch.version.cuda.split(".")[0]
for pkg in (f"faiss-gpu-cu{cuda}", "faiss-gpu-cu12"):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False, capture_output=True, text=True)
    print(pkg, "installed" if r.returncode == 0 else "not installed: " + (r.stderr.strip().splitlines() or ["?"])[-1])
    if r.returncode == 0:
        break
'''),
        code("!python -u triton_probe.py --bench --budget 180"),
        code("!python -u triton_probe.py --check"),
        code("!python -u triton_probe.py --bench --tune --budget 3600"),
    ]
    nb = notebook(numbered(cells, [("Which GPU", 0), ("Small installs", 1), ("Write triton_probe.py", 0),
                                   ("Benchmark against cuVS", 3),
                                   ("Install FAISS", 4, "optional"), ("Benchmark with FAISS too", 3, "optional"),
                                   ("Check the kernels against exact search", 15, "optional"),
                                   ("Benchmark after tuning all 24 configurations", 45, "optional")]), "GPU")
    out = HERE / "colab" / "triton_probe.ipynb"
    out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
