# Learning Closure of Dynamical Systems with Kernel Ridge Regression

Code and experiment configurations for the manuscript of the same name. This
candidate contains **38 KRR paper cases**: 16 Lorenz-63, 16 Kuramoto–Sivashinsky,
and six kinetic BGK closure conditions. Case membership identifies an experiment
specification; it does not claim that an execution reproduced the published numbers.

```bash
python run_case.py --list-paper-cases
python run_case.py l63 --paper-case l63-rbf-rk4-cubic-n512 --resolve-only
python run_case.py ks --paper-case ks-diffusion-maps-m0-n16384 --resolve-only
python run_case.py kinetic --paper-case kinetic-global-type-i-plus-type-ii --resolve-only
```

Listing and configuration resolution require only Python 3.11 or newer. Numerical
execution requires the dependencies and external inputs described in
[ENVIRONMENT.md](ENVIRONMENT.md) and [DATA.md](DATA.md). No datasets or model
coefficients are downloaded automatically.

| Problem | Implementation | Required execution backend |
|---|---|---|
| Lorenz-63 | Raw `(x,y,z)` input predicts the unresolved update; Heun and quadratic through quintic RK4 stage treatments; three validation rollouts | NumPy reference utilities; CuPy/CUDA KRR and rollout |
| KS | Raw resolved state/closure histories; five validation rollouts per candidate; reduced forced ETDRK4 | CPU preprocessing; CuPy/CUDA KRR and rollout |
| Kinetic | Global PCA-KRR, zeroth-order local, first-order local residual; Type I and mixed Type I + II training | NumPy/SciPy CPU |

KS reference dynamics are truncated to `1 <= |k| <= 48`. The supplied archive
preprocessor has a distinct 48-row input convention; its producer binding and the
missing full-order generator are explained in DATA.md.

The kinetic reference solver retains the conservative discrete Maxwellian of
Mieussens for initialization and BGK relaxation. Learned macroscopic prediction
uses the conservative corrected-flux formulation:

```text
G_hat_eff = G_macro_eff(U_hat) + R_eff_hat(U_hat)
U_hat_next = U_hat - dt/dx * (G_hat_eff - roll(G_hat_eff, 1))
```

Targets are differences of stage-weighted transport fluxes. Global PCA is full
rank, 768/768, without whitening. The direct local model uses ten cells × three
conserved variables; the residual uses nine adjacent differences × three variables,
scaled by `1/dx`. Both residual conditions share the fixed Type-I zeroth-order
baseline. The kinetic Gaussian is `exp(-||x-x'||²/(4 epsilon))`, with literal
ridge system `(K + lambda_reg I) A = Y` and no factor of sample count.

`paper_cases.json` and `configs/paper/` identify all paper configurations.
[docs/result_commands.csv](docs/result_commands.csv) supplies one concrete command
per case; [REPRODUCIBILITY.md](REPRODUCIBILITY.md) explains each workflow and the
figure commands. The final kinetic figure trajectory is `validation_II_VIIB_06`.
Table 4 uncertainties use population SD (`ddof=0`) over the 24 protected
trajectories of each family. The surviving L63 Table 1 quadratic RK4/RBF summary
rounds to **5.9242 ± 0.0980**.

Custom settings are explicit, for example:

```bash
python run_case.py ks --config configs/custom_examples/ks.toml --resolve-only
```

A scientific change to a preset yields `experiment_status = custom`; unchanged
presets yield `paper_case`. Output paths and CUDA device ordinals are operational
settings. Unsupported numerical backends and unimplemented scientific overrides
fail rather than silently selecting another algorithm.

The package has a thin dispatcher, separate problem pipelines, and no common
numerical-solver superclass:

```text
run_case.py             CLI and lazy dispatch
interface/              configuration, identities, input and run records
problems/l63/           reference, CUDA KRR, protocol and pipeline
problems/ks/            aligned data processing, CUDA KRR, ETDRK4 and pipeline
problems/kinetic/       reference, features, KRR, conservative rollout and metrics
configs/paper/          38 presets and kinetic membership CSVs
expected/              small historical L63/KS summary values
scripts/               explicit kinetic data construction
```

The LSTM comparison uses an implementation/closure framework developed in prior
work [14]; neural-network implementation code is not duplicated here. Reference
[14]: Harlim, Jiang, Liang and Yang, “Machine learning for prediction with missing
dynamics,” *Journal of Computational Physics* 428 (2021), 109922.

Run the local CPU checks with `python -m pytest -q`. Release-time
verification details and known capability limits are summarized in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) and [ENVIRONMENT.md](ENVIRONMENT.md).
Source-provenance and attribution notes are provided in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No explicit software license is
currently included with this repository.
