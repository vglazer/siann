"""Writes benchmarks/colab/tpu_probe.ipynb, a notebook to run on Colab with a TPU. First JAX (jax_probe.py):
exact search through JAX's built-in operations, and L1 through a custom Pallas kernel, checked against NumPy.
Then PyTorch (tpu_probe.py): it installs TorchTPU (or falls back to PyTorch/XLA where TorchTPU cannot run) and
times exact search through PyTorch's built-in operations. Both scripts are embedded here, so the notebook and
the scripts cannot drift apart.

    python make_colab_tpu.py
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
    probe = (HERE / "tpu_probe.py").read_text(encoding="utf-8")
    jax_probe = (HERE / "jax_probe.py").read_text(encoding="utf-8")
    cells = [
        md("""
# siann's TPU probe

Run on a TPU (Colab: Runtime, Change runtime type, TPU). Two parts, each in its own process:

1. **JAX:** exact search through JAX's built-in operations (L2 and inner product by a matrix multiply and
   `lax.top_k` or the TPU's `lax.approx_min_k`, at the default matrix-multiply precision and at full float32;
   L1 by tiles), and L1 through siann's custom kernel in Pallas, checked first against exact search.
2. **PyTorch:** the same searches through PyTorch's built-in operations on TorchTPU (or PyTorch/XLA), eagerly
   and compiled.

Every result is checked against NumPy, and each number prints as it is measured. Copy the outputs of the check
and benchmark cells back.
"""),
        code('''
# which TPU this runtime has, and Python's version (TorchTPU needs 3.11 or later)
import os
import sys

print(sys.version)
for var in ("TPU_ACCELERATOR_TYPE", "TPU_NAME", "COLAB_TPU_ADDR"):
    print(var, os.environ.get(var))
print("accelerator devices:", sorted(f for f in os.listdir("/dev") if f.startswith(("accel", "vfio"))))
'''),
        md("## 1. JAX"),
        code('''
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
'''),
        code("%%writefile jax_probe.py\n" + jax_probe),
        code("!python -u jax_probe.py --check"),
        code("!python -u jax_probe.py"),
        md("## 2. PyTorch"),
        code('''
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
'''),
        code("%%writefile tpu_probe.py\n" + probe),
        code("!python -u tpu_probe.py"),
    ]
    nb = notebook(numbered(cells, [("Which TPU", 0), ("JAX for the TPU", 3), ("Write jax_probe.py", 0),
                           ("Check the Pallas kernel against exact search", 3), ("JAX benchmark", 15),
                           ("TorchTPU or PyTorch/XLA", 5), ("Write tpu_probe.py", 0), ("PyTorch benchmark", 20)]), "TPU")
    out = HERE / "colab" / "tpu_probe.ipynb"
    out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
