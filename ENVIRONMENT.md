# Environments

The interface uses Python ≥3.11 (`tomllib`). The numerical backend is explicit;
there is no automatic CPU/GPU fallback.

The local assembly tests ran on macOS/arm64 with Python 3.13.7, NumPy 2.5.1,
SciPy 1.18.0, pytest 9.1.1, and Matplotlib 3.11.0, using an existing environment.
These versions were queried directly. They are local verification facts, not a
claim about the historical GPU environment. Machine-readable details are in
`docs/verified_environment.json`.

For a new CPU environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
```

Requirements are intentionally unpinned: only the environment above was verified
locally, and no universal environment lock has been validated. Matplotlib is
needed only for rendering. SciPy is used for kinetic Cholesky/eigendecomposition,
training geometry and KS MAT ingestion. NumPy/SciPy BLAS implementations and
threading can affect numerical results; run records retain NumPy's backend
configuration when available. A saved source hash does not imply bitwise parity
across platforms.

L63/KS training and recursive prediction additionally require a working NVIDIA
CUDA device and an appropriate CuPy distribution for that CUDA installation.
Choose the CuPy wheel for the actual target system rather than installing several
CUDA variants together. This assembly did not install CuPy, query an NVIDIA GPU,
or run CUDA tests. An exact historical CuPy/CUDA/driver lock is not established.
The dispatcher fails clearly if the required backend is absent. CPU listing,
configuration resolution, data preparation and saved-result rendering remain
available. There is no complete CPU KRR port for L63 or KS, and no kinetic GPU
backend.

Kinetic reference generation, full-rank PCA, dense 8208-center kernels, and the
L63/KS search grids can require substantial memory and time. No production-sized
fit, search, protected-data evaluation or data-generation run was performed during
assembly. Follow DATA.md before invoking numerical workflows.
