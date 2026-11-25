"""Writes two notebooks for Colab with a T4 GPU, each finishing within 5 minutes when run with Run All, and each
writing out triton_probe.py (embedded here, so the notebooks and the script cannot drift apart):

benchmarks/colab/triton_probe.ipynb: siann's exact L1 kernels in Triton (the in-kernel merge, and the 2D-tile
kernel) against cuVS's brute force, with the memory each takes; Colab comes with cuVS, so almost nothing is
installed. benchmarks/colab/triton_faiss.ipynb: FAISS's GPU package installed, then the merge kernel against FAISS
and cuVS, and the check that siann's neighbors are FAISS's.

Each probe's budget is what is left of the notebook's 5 minutes when it starts. The kernels' check against exact
search and the tuning of all 24 configurations stay in the script (--check, --tune), for runs of their own.

    python make_colab_probe.py
"""
import json
from pathlib import Path

from colab_cells import BUDGET, Steps, notebook

HERE = Path(__file__).parent

SMALL_INSTALLS = '''
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
'''

FAISS_INSTALL = '''
# FAISS's GPU package for this CUDA (it publishes no CUDA 13 build: its CUDA 12 one brings its own runtime)
for pkg in (f"faiss-gpu-cu{cuda}", "faiss-gpu-cu12"):
    if install(pkg):
        break
'''


def write(name: str, intro: str, faiss: bool) -> None:
    s = Steps()
    s.md(intro)
    s.shell("Which GPU", 0, "nvidia-smi --query-gpu=name,memory.total --format=csv")
    s.code("Install FAISS and small packages" if faiss else "Small installs", 2 if faiss else 1,
           SMALL_INSTALLS + (FAISS_INSTALL if faiss else ""))
    s.write("triton_probe.py", (HERE / "triton_probe.py").read_text(encoding="utf-8"))
    s.shell("Benchmark against FAISS and cuVS" if faiss else "Benchmark against cuVS", 3,
            f"python -u triton_probe.py --bench {'--lean ' if faiss else ''}--budget {BUDGET}")
    out = HERE / "colab" / name
    out.write_text(json.dumps(notebook(s.cells(), "GPU"), indent=1) + "\n", encoding="utf-8")
    print(out)


def main() -> None:
    write("triton_probe.ipynb", """
# siann's Triton probe, against cuVS

Run on Colab with a T4 GPU (Runtime, Change runtime type, T4 GPU), then Runtime, Run all: it finishes within 5
minutes. Colab comes with PyTorch, Triton and cuVS, so almost nothing is installed. siann's exact L1 kernels in
Triton (the selection merged inside the kernel, and the earlier 2D-tile kernel) are timed against cuVS's brute
force on 200,000 Gaussian vectors, with the memory each takes; the benchmark stops itself at what is left of the
5 minutes. Copy the output of the last step back.
""", faiss=False)
    write("triton_faiss.ipynb", """
# siann's Triton probe, against FAISS

Run on Colab with a T4 GPU (Runtime, Change runtime type, T4 GPU), then Runtime, Run all: it finishes within 5
minutes. It installs FAISS's GPU package, then times siann's L1 kernel with the selection merged inside it against
FAISS's and cuVS's brute force on 200,000 Gaussian vectors, and checks that its neighbors are FAISS's; the
benchmark stops itself at what is left of the 5 minutes, so a slow install shortens it. Copy the output of the
last step back.
""", faiss=True)


if __name__ == "__main__":
    main()
