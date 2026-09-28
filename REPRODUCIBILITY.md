# Reproducibility workflows

Start with `python run_case.py --list-paper-cases --json`. The registry reports
backend and external-input requirements. `--resolve-only` prints a fully expanded
public specification and its digest without importing numerical modules or
executing science. Historical artifact identities that are unavailable are not
guessed. The presets are release specifications; source derivation and local
verification do not certify an immutable historical producing revision.

## Paper results and commands

`docs/result_commands.csv` expands this mapping to all **38 concrete cases**.
Replace the data-root path with a provisioned directory matching DATA.md.

| Result | Cases and workflow |
|---|---|
| L63 Table 1 | Both kernels, N4096, Heun/quadratic/cubic/quartic/quintic; `reproduce` searches the supplied literal candidate schedules then tests selected fits |
| L63 Table 2 | Both kernels, cubic RK4, N512/1024/2048/4096; same workflow |
| KS Table 5 | Both kernels, N2048/4096/8192/16384, m0/m1; `reproduce` uses the 13×13 grid and five validation windows |
| KS Figure 1 | N16384/m0/diffusion maps; saved validation surface, representative model index 3 |
| KS Figure 2 | Same case; saved test trajectory, model/segment metadata and training-pool normalization |
| KS Figure 3 | Eight m0 table outputs supply scaling points; full manuscript styling and external neural comparison overlays are not reproduced by this candidate |
| Kinetic Table 4 | Six frozen conditions; `reproduce` fits at selected parameters on supplied training support/targets, then evaluates a complete supplied 24+24 protected bundle |
| Kinetic Tables 3 and 7 | Bundled IC memberships, center allocations and data/feature construction definitions |
| Kinetic Figures 4–8 | Three mixed conditions, final trajectory `validation_II_VIIB_06`; `replay` followed by `render` produces public profile/temperature comparisons |
| Neural comparison Table 6 | External prior-work implementation; no neural workflow is included |

Examples:

```bash
python run_case.py l63 --paper-case l63-rbf-rk4-quadratic-n4096 --workflow reproduce --data-root /path/to/external
python run_case.py ks --paper-case ks-diffusion-maps-m0-n16384 --workflow reproduce --data-root /path/to/external
python run_case.py kinetic --paper-case kinetic-first-order-local-baseline-type-i-residual-type-i-plus-type-ii --workflow replay --data-root /path/to/external
python run_case.py ks --paper-case ks-diffusion-maps-m0-n16384 --workflow render --render-kind validation_heatmap --data-root /path/to/external
python run_case.py ks --paper-case ks-diffusion-maps-m0-n16384 --workflow render --render-kind rollout --data-root /path/to/external
python run_case.py kinetic --paper-case kinetic-global-type-i-plus-type-ii --workflow render --data-root /path/to/external
```

Figure renderers use saved arrays only and produce comparison layouts, not
byte-identical manuscript PDFs. Each kinetic model is run separately; all three
mixed cases are identified in the registry. Run metadata distinguishes loaded
numerical results from newly computed figures. The explicit rendering layout,
input checksum and selected trajectory appear in `figure_manifest.json`.

## Meaning of each workflow

L63/KS **reproduce** runs validation and refits the winning candidate on its
original training pairs, followed by test rollouts. Selection is maximum mean
validation VPT with the first candidate index breaking exact ties. An all-failed
model stops execution. L63 preserves full raw xyz inputs and N−1 supervised pairs;
KS uses N pairs from N+m+1 raw samples. Kernels retain each problem's audited
density conventions and floating arithmetic. No new heuristic is substituted for
L63's missing literal historical schedules.

L63/KS **replay** loads supplied validation scores to reproduce the selected
hyperparameters, then refits and evaluates. It is a saved-selection replay, not a
claim to have loaded historical coefficient bytes. Selected fitted arrays,
normalization and diffusion densities are exported to NPZ. GPU execution and
cross-platform numerical parity remain unverified in this assembly.

Kinetic **reproduce** performs a new EVD fit at an already selected epsilon/ridge;
**replay** loads supplied coefficients. Both use the supplied exact first-interval
correction, then reevaluate learned corrections from the recursively predicted
conserved state. The fixed Type-I baseline is bound separately for both residual
conditions. Fit health rejects relative solve residuals above 1e-6; there is no
hidden jitter or ridge fallback. The baseline construction uses canonical
Cholesky, while selected comparison candidates use the retained EVD convention.

The complete coupled kinetic hyperparameter search is **not an automated
dispatcher workflow in this candidate**. The scientific components are provided:
`selection.initial_grids/candidates/assess/expand`, `fit.fit_ridges`,
`protocol.selection_gate`, reference generation, preprocessing, center/target
construction, rollout and metrics. The policy uses support-specific epsilon_50,
quarter-decade grids, I-only/mixed union of near-optimal boundary triggers, at most
two expansion rounds, deterministic IDs/ties and the final-versus-transient score
consistency gate. A future orchestration validation is needed before claiming
an end-to-end rerun of the entire selection campaign. No searches were performed
or protected outcomes consulted during assembly.

Saved-score **render** for L63/KS writes table statistics on CPU. Use
`--render-kind validation_heatmap` or `rollout` for the supported KS figures.
Kinetic render writes profiles and temperature panels for the final validation
case. Small `expected/` summaries are historical references, not fresh results.
Automatic tolerance-based agreement with the paper is not asserted. Appropriate
cross-platform acceptance tolerances still need scientific review.

## Outputs and failure behavior

Default output is a unique, non-overwriting directory:

```text
runs/<problem>/<timestamp_digest_unique>/
  resolved_config.json
  data_manifest.json
  run_manifest.json
  validation_results.csv        # when validation is run or loaded
  selected_model.json           # when a model is selected/loaded
  test_metrics.json             # when evaluation or score reduction occurs
  models/
  figures/
  figure_manifest.json          # only when rendering occurs
```

The configuration is written before numerical work. Inputs are checksum-verified;
stages are marked computed or loaded. Failure preserves a run record with the
reason and does not fabricate downstream files. Kinetic failed rollouts retain
explicit health/completed-step information; incomplete protected results do not
produce a successful family aggregate. Output directories are never reused.

Table uncertainty is sample SD (`ddof=1`) across ten L63 model means, population
SD (`ddof=0`) across ten KS model means, and population SD across the 24 kinetic
per-trajectory values in each family. Kinetic F is the mean of four final spatial
relative L2 errors (rho,u,theta,E); M is the mean of their separate time maxima.
`J_final = .5*(mean_F_I + mean_F_II)` and the manuscript family sum is
`2*J_final`. The transient score adds `.125*(mean_M_I + mean_M_II)`.

## Scope of verification

Run `python -m pytest -q` and `python -m compileall -q .`. CPU tests cover registry
membership, paper/custom resolution, import safety, conservative macroscopic
steps, the absence of transitive equilibrium dependencies, reference equilibrium
moments, stencil differences, literal ridge, PCA conventions and metrics.
Release-time tests use small synthetic
arrays; they do not replace full scientific reproduction. CUDA workflows, large
external-input paths and production-scale construction remain unexecuted locally.
