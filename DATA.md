# Data and external inputs

Bundled: source, 38 presets, small kinetic IC/membership CSVs and nested time
indices, historical L63/KS summary values, and small checksum/derivation manifests.
No raw trajectory banks, fitted coefficients, PCA/normalizer arrays, manuscript
PDFs, or historical execution logs are bundled. `data/external_assets.json`
records known SHA-256 values and explicitly marks every listed large input as
**not bundled**. There is no public download endpoint supplied in this candidate.
Files must be obtained from their owners or generated as described below.

The approximately 223 MB KS `ks0p01.mat` is external. Its owner-supplied SHA-256 is
`80d6bd17fe8717fff5c392ed02581ec09e4590ae4b461d811ec864d257463d84`;
this assembly did not read or recompute that file. The historical processed
`ks6_closure_phys14.npz` checksum is not established. The L63 frozen protocol
archive checksum is
`7c5ed13c95c57cd17d38277c41fc5cad98c2a48ae904d6053413a5f84f452f9f`.
These are archive identities, not checksums for an arbitrary extracted/exported file.

## Input manifest

Each dispatcher run reads `<data-root>/<paper-case-id>.json`. A custom run derived
from a case uses that parent ID as its manifest filename, but requires the custom
resolved scientific digest. A fully specified independent custom run uses
`<problem>.json`. Paths inside a manifest are relative to the data root. Obtain the
scientific digest with `--resolve-only`.

```json
{
  "scientific_sha256": "<digest from resolved configuration>",
  "selected_model_id": "U30_mixed_58ce5aaa903185fd271d32cf",
  "files": {
    "features": {"path": "kinetic/preprocessing/U30_mixed_features.npy", "sha256": "<actual SHA-256>"},
    "coefficients": {"path": "kinetic/models/U30_mixed_coefficients.npy", "sha256": "<actual SHA-256>"},
    "normalizers": {"path": "kinetic/preprocessing/normalizers.npz", "sha256": "<actual SHA-256>"},
    "evaluation": {"path": "kinetic/evaluation/validation_II_VIIB_06.npz", "sha256": "<actual SHA-256>"}
  }
}
```

All required file checksums are verified before computation. Kinetic paper runs
also enforce the known frozen feature, normalizer, PCA and baseline-feature
checksums, and require `selected_model_id`. Newly generated or changed preprocessing
arrays must use a custom `data_identity`; they are not silently labeled frozen
artifacts. This establishes the
identity supplied by the operator; it is **not** authentication of an original
historical producer or certification of a manuscript result. Historical replay
claims still require reviewed original-to-export bindings. NPZ/NPY loaders use
`allow_pickle=False`; executable pickle archives are not accepted by the public
interface. An owner must export trusted historical pickle arrays into the schemas
below and preserve original/export checksums. No such large export is bundled.

| Workflow | Required input roles and schema |
|---|---|
| L63 reproduce | `protocol`: NPZ `train_blocks (10,N,3)`, `validation_pool (10,3000,3)`; `test`: NPY `(500,2500,3)`; `candidates`: NPZ `epsilon`, `lambda_reg`, each `(10,4096)` |
| L63 replay | Same, plus `validation_scores`: NPZ `vpt (10,4096)`; selects from saved scores and refits original training blocks |
| KS reproduce | `processed`: NPZ `v6_series (T,6)` and `theta_rhs6 (T,6)` complex128, `x14_phys (T,14)` and `theta14_phys (T,14)` float64; at least 500000 samples |
| KS replay | Same, plus `validation_scores`: NPZ `vpt (10,169)` in epsilon-major, ridge-minor order |
| L63/KS render summary | `scores`: NPZ `vpts_all (models,test_segments)`; reduction of loaded scores only |
| Kinetic replay | `features`: NPY `(centers,feature_dimension)`; `coefficients`: NPY `(centers,target_dimension)`; `normalizers`; `evaluation`; `pca` for global; both baseline roles for residual |
| Kinetic reproduce | Same, but `targets` replaces `coefficients`; performs EVD fitting at the frozen selected epsilon/ridge; does not rerun selection |
| Kinetic render | `rollout`: one successful saved NPZ from `runs/.../models/validation_II_VIIB_06.npz` |

Kinetic normalizers NPZ keys: `U30_mean`, `U30_scale`, `DU27_mean`, `DU27_scale`,
each shape `(3,)`. PCA keys: `U_mean`, `R_mean` shape `(768,)`, and
`U_components`, `R_components` shape `(768,768)`; full orthonormal bases without
whitening. Feature dimensions are global 768, direct local 30, residual 27;
target dimensions are global 768, local 3. Center counts are 8208, 4032 and 8064.

Kinetic `evaluation` NPZ contains `truth_U (cases,456,3,256)`,
`initial_correction (cases,3,256)`, `times (456,)`, and Unicode `trajectory_ids`.
The initial correction is `R_eff[0]` from stage-weighted transport fluxes.
Only declared validation/protected members are accepted. Protected evaluation
requires the complete 24+24 membership and already frozen selected models.
Protected outcomes are never inputs to the fitting or selection routines.

Residual runs additionally require `baseline_features` and
`baseline_coefficients` NPY arrays, shapes `(4032,30)` and `(4032,3)`, plus:

```json
"baseline": {
  "support": "U30_I_only", "N": 4032,
  "kernel_epsilon": 2953.231481793101, "lambda_reg": 0.000001
}
```

For a residual selected-fit run, `residual_targets_baseline_sha256` must equal
the SHA-256 of the supplied baseline coefficients. This baseline is a separate
fixed fit; the independently selected direct local comparison cannot replace it.

KS Figure 1 rendering requires `validation_surface`: NPZ `vpt`, `epsilon` and
`lambda_reg`, all `(10,169)`, with model index 3 displayed. Figure 2 rendering
requires `rollout`: NPZ `truth_real12`, `predicted_real12` both `(time,12)`,
`times (time,)`, `state_scale_real12 (12,)`, and a Unicode `metadata_json` object
binding `paper_case_id`, `model_index` and `test_segment` from the saved selection.
The metric scale is the training-pool componentwise population standard deviation.

## Generation and preprocessing

L63 `problems.l63.reference.gen_data()` exposes the surviving NumPy RK4 generator
with seeds 22 and 43; it runs only when explicitly called. A new trajectory is not
an authenticated substitute for the frozen historical protocol bundle or literal
candidate schedules. `problems.l63.protocol.blocks()` constructs explicit raw
state training/validation windows. Smaller-N historical bandwidth/cache details
are preserved by loading the literal schedules, rather than recomputing heuristics.

The source-backed aligned KS processor is available as:

```bash
python -m problems.ks.data /path/to/ks0p01.mat /path/to/ks6_closure_phys14.npz
```

It retains division by 48, valid archive modes ±1…±23 with zero/Nyquist
placeholders excluded, six positive resolved modes, full-minus-resolved convolution
and the matching physical-14 transform. The manuscript reference truncation is
`1 <= |k| <= 48`. A 48-row stored archive does not establish that full-order
truncation. The original MATLAB producer, initial conditions, burn-in, output
stride and binding to the historical processed arrays remain unavailable. This
processor is an explicit available archive-format route, not a claimed recovered
from-scratch generator. SciPy-readable MAT format is supported; MATLAB v7.3/HDF5
adaptation is not implemented or verified.

Kinetic generation preserves the frozen IC rows and Mieussens equilibrium:

```bash
python -m scripts.kinetic_data generate --bank /path/to/bank --trajectory-id training_I_A1.5_sigma0.04_phase+0.0000
python -m scripts.kinetic_data preprocess --bank /path/to/bank --output /path/to/new-preprocessing
python -m scripts.kinetic_data centers --bank /path/to/bank --output /path/to/new-supports
```

Use an exact ID from the bundled membership CSV. Generation accepts one
training/validation member at a time, refuses overwrites, and writes output
checksums. Bank loading requires the matching `.json` member/checksum sidecar. Center construction can be expensive: it uses the complete training
bank, shared Type-I normalization/PCA, deterministic local traversal, exact-byte
deduplication and source quotas, then fits the fixed baseline to construct residual
targets. None of these production-sized stages was run during assembly. The
original Type-II space-filling design producer/seed is unavailable; supplied IC
rows reproduce membership, not the unpublished design optimization.

Supporting supplementary material can accurately be described as code,
experiment configurations, reproducibility instructions, and the small supporting
data/manifests supplied here. Large inputs and historical fitted models require
separate distribution and permissions.
