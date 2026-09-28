# Public repository assembly report — 2026-09-28

Created `public_repo/` as a new local release candidate. The pre-existing workspace
assembly report and private provenance pool were not rewritten. **209 fingerprinted
private source/planning files remain byte-identical.** No Git repository was
initialized, nothing was published, and no software license was added.

## Contents assembled

- Thin `run_case.py`, strict configuration resolver, authenticated input loading,
  unique run directories, software/backend records and truthful stage status.
- `paper_cases.json` and 38 presets: **L63 16, KS 16, kinetic 6**. Scientific
  overrides become custom; unsupported implementations are rejected.
- L63 reference/protocol utilities, retained CUDA RBF/diffusion kernels, coupled
  stage-reconstruction models, explicit validation/selected-fit/test pipeline.
- KS aligned CPU preprocessing, retained CUDA kernels, grouped-eigen validation,
  reduced ETDRK4 rollout, and saved-surface/trajectory renderers.
- Kinetic reference equilibrium/IMEX/finite-volume components; explicit IC and
  transport-target generation; normalization/PCA, center and residual-target
  construction; KRR fitting/selection components; corrected-flux macroscopic
  rollout; four-field metrics and saved-result plotting.
- README, DATA, reproducibility/environment/rights documentation; concrete
  result-to-command CSV; expected summaries for 16 L63 and 16 KS cases.

[docs/file_inventory.tsv](docs/file_inventory.tsv) lists the assembled files and
release SHA-256 values. [docs/source_derivation.json](docs/source_derivation.json)
records copied/extracted scientific sources and adaptations. All 39 first-source
hashes checked against the planning map matched. Numerical definitions in 16
retained kernel/solver modules were AST-identical to their source definitions;
this is structural evidence, not a claim of production numerical parity.

Final overrides are represented directly: six selected kinetic IDs, fixed Type-I
residual baseline, stage-weighted transport targets, full unwhitened 768/768 PCA,
U30/DU27 stencils, literal ridge, final `validation_II_VIIB_06` figure case, and
Table 4 population SD. KS uses five validation rollouts and the manuscript
`1 <= |k| <= 48` reference truncation. L63 uses three validation rollouts and full
raw xyz input. Its quadratic/RBF historical summary rounds to 5.9242 ± 0.0980.
The macroscopic deployment import closure excludes reference equilibrium machinery.
No neural implementation was copied.

## Verification actually performed

| Check | Result |
|---|---|
| `python -m pytest -q` | **29 passed**; final run 13.06 s |
| `python -m compileall -q public_repo` | Passed, all Python files |
| CPU-safe package/module imports | 33 imported successfully |
| CLI JSON list | 38 cases, correct 16/16/6 counts |
| Configuration resolution | All 38 presets; custom/invalid/backend cases tested |
| Corrected-flux step and periodic conservation | Passed |
| Transitive macroscopic dependency scan and blocked-reference-import execution | Passed |
| Exact first interval / subsequent predicted-state queries | Passed |
| Reference equilibrium and tiny two-step transport-target test | Passed |
| Literal ridge, local differences, PCA, family SD, selection rules | Passed |
| Input checksum/path and missing-input run records | Passed |
| CPU saved-score dispatch and synthetic kinetic PDF rendering | Passed |
| Private-path/stale-case public-tree scan | No findings |
| Private-file preservation | 209/209 unchanged |

Verified local environment: Python 3.13.7, NumPy 2.5.1, SciPy 1.18.0,
Matplotlib 3.11.0, pytest 9.1.1, macOS arm64. CuPy/CUDA was unavailable.
GPU execution was **not tested** or simulated. Tests used small synthetic arrays;
no scientific hyperparameter searches, production fits, protected-data tuning,
large data generation, or expensive experiments were run.

## Data, capability limits and publication blockers

Bundled data are small scientific memberships/time indices and historical summary
values. The 15 large-input catalog entries are explicitly external, including the
approximately 223 MB KS MAT file, L63 protocol archive, and kinetic preprocessing
arrays. No model coefficients or raw trajectory banks are bundled. No private
manuscript PDFs, audit dumps, diagnostic logs or internal handoffs were copied.

1. **Rights and license:** confirm shared diffusion-kernel/L63/KS coauthor lineage,
   KSmodel-derived numerical utilities, legacy MATLAB/SBGK adaptations and data
   permissions; a human must select the license. See
   [THIRD_PARTY_NOTICES_PENDING.md](THIRD_PARTY_NOTICES_PENDING.md).
2. **External inputs:** provision owner-reviewed checksummed arrays and safe
   historical exports, including L63 literal schedules and authenticated KS
   processed data, kinetic coefficients/baseline/evaluation banks. Supplied
   checksums alone do not authenticate an original producer. Establish public
   distribution locations and original-to-export bindings.
3. **End-to-end verification:** validate CUDA pipelines on supported hardware and
   production-sized kinetic paths when separately authorized. Historical L63
   producer identity, exact GPU environments, and numerical tolerances remain
   unverified. There is no full CPU L63/KS port or kinetic GPU backend.
4. **Reference/design provenance:** the KS full-order producer and the mapping
   from its 48-row archive to the manuscript truncation remain unresolved. The
   Type-II design producer/seed is absent; exact frozen IC rows are supplied.
5. **Workflow coverage:** the kinetic dispatcher supports selected-parameter
   fitting and coefficient replay; automatic orchestration of the complete
   coupled search campaign is not implemented. Scientific search components are
   supplied. Public figure layouts are not exact manuscript artwork; Figure 3's
   complete scaling/ external-comparison renderer is not supplied. Original
   protected per-case Table 4 artifacts are not bundled, so its numerical values
   were not independently verified here.

The candidate is assembled and locally checked within these stated limits. It is
not certified as a complete historical replay or publication-ready distribution.
Stop point respected: no Git/GitHub initialization, push or publication.
