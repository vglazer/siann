"""Writes two notebooks for Colab (or Kaggle) with a TPU, each finishing within 5 minutes when run with Run All,
and each writing out its probe (embedded here, so the notebooks and the scripts cannot drift apart):

benchmarks/colab/tpu_probe.ipynb: JAX (jax_probe.py), which TPU runtimes come with: exact search through JAX's
built-in operations, and L1 through siann's kernel in Pallas, checked first against exact search.
benchmarks/colab/torchtpu_probe.ipynb: PyTorch (tpu_probe.py): TorchTPU installed (or PyTorch/XLA where TorchTPU
cannot run), then exact search through PyTorch's built-in operations.

Each probe's budget is what is left of the notebook's 5 minutes when it starts.

    python make_colab_tpu.py
"""
import json
from pathlib import Path

from colab_cells import BUDGET, Steps, notebook

HERE = Path(__file__).parent

WHICH_TPU = '''
# which TPU this runtime has, and Python's version (TorchTPU needs 3.11 or later)
import os
import sys

print(sys.version)
for var in ("TPU_ACCELERATOR_TYPE", "TPU_NAME", "COLAB_TPU_ADDR"):
    print(var, os.environ.get(var))
print("accelerator devices:", sorted(f for f in os.listdir("/dev") if f.startswith(("accel", "vfio"))))
'''
JAX_INSTALL = '''
# JAX for TPUs (Colab's TPU runtimes come with it; reinstalled only if it cannot see the TPU)
import subprocess
import sys

r = subprocess.run([sys.executable, "-c", "import jax; print(jax.__version__, jax.devices())"], check=False,
                   capture_output=True, text=True)
if "Tpu" not in r.stdout and "TPU" not in r.stdout:
    print("JAX cannot see a TPU:", (r.stdout + r.stderr).strip().splitlines()[-1:] or "?")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "jax[tpu]"], check=False)
    r = subprocess.run([sys.executable, "-c", "import jax; print(jax.__version__, jax.devices())"], check=False,
                       capture_output=True, text=True)
print((r.stdout + r.stderr).strip())
'''
TORCH_INSTALL = '''
# TorchTPU (validated on TPU v5e, v5p, v6e and v7); where it cannot run here, PyTorch/XLA if the runtime has it
import subprocess
import sys


def works(stmt):
    r = subprocess.run([sys.executable, "-c", stmt], check=False, capture_output=True, text=True)
    return r.returncode == 0, (r.stderr.strip().splitlines() or [""])[-1]


TPU = "import torch\\ntry:\\n    import torch_tpu\\nexcept ImportError:\\n    pass\\ntorch.ones(1, device='tpu')"
ok, why = works(TPU)
if not ok:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "psutil", "torch_tpu",
                        "--extra-index-url", "https://download.pytorch.org/whl/cpu"],
                       check=False, capture_output=True, text=True)
    print("torch_tpu", "installed" if r.returncode == 0 else "not installed: " + (r.stderr.strip().splitlines() or ["?"])[-1])
    ok, why = works(TPU)
print("TorchTPU", "works" if ok else f"does not work here: {why}")
if not ok:
    ok, why = works("import torch_xla; print(torch_xla.device())")
    print("PyTorch/XLA", "works: the probe uses it" if ok else f"does not work either: {why}")
'''


def save(name, steps):
    out = HERE / "colab" / name
    out.write_text(json.dumps(notebook(steps.cells(), "TPU"), indent=1) + "\n", encoding="utf-8")
    print(out)


def main() -> None:
    s = Steps()
    s.md("""
# siann's TPU probe: JAX and Pallas

Run on a TPU (Colab: Runtime, Change runtime type, TPU; Kaggle: Accelerator, TPU), then Run all: it finishes within
5 minutes. TPU runtimes come with JAX, so nothing is installed. siann's L1 kernel in Pallas is checked against
exact search, then exact search through JAX's built-in operations (L2 and inner product by a matrix multiply and
`lax.top_k` or the TPU's `lax.approx_min_k`; L1 by tiles) is timed against the Pallas kernel on 200,000 Gaussian
vectors, every result checked against NumPy; the benchmark stops itself at what is left of the 5 minutes. Copy
the output of the last two steps back.
""")
    s.code("Which TPU", 0, WHICH_TPU)
    s.code("JAX for the TPU", 1, JAX_INSTALL)
    s.write("jax_probe.py", (HERE / "jax_probe.py").read_text(encoding="utf-8"))
    s.shell("Check the Pallas kernel against exact search", 1, "python -u jax_probe.py --check")
    s.shell("JAX benchmark", 3, f"python -u jax_probe.py --budget {BUDGET}")
    save("tpu_probe.ipynb", s)

    s = Steps()
    s.md("""
# siann's TPU probe: PyTorch

Run on a TPU (Colab: Runtime, Change runtime type, TPU), then Run all: it finishes within 5 minutes. It installs
TorchTPU (validated on TPU v5e and newer; where it cannot run, it tries PyTorch/XLA), then times exact search
through PyTorch's built-in operations on 200,000 Gaussian vectors, eagerly and compiled, every result checked
against NumPy; the benchmark stops itself at what is left of the 5 minutes, so a slow install shortens it. Copy
the output of the last step back.
""")
    s.code("Which TPU", 0, WHICH_TPU)
    s.code("TorchTPU or PyTorch/XLA", 3, TORCH_INSTALL)
    s.write("tpu_probe.py", (HERE / "tpu_probe.py").read_text(encoding="utf-8"))
    s.shell("PyTorch benchmark", 2, f"python -u tpu_probe.py --budget {BUDGET}")
    save("torchtpu_probe.ipynb", s)


if __name__ == "__main__":
    main()
