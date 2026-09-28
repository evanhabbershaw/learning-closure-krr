# Attribution and redistribution review

No root software license has been chosen. This local candidate is not a grant of
redistribution rights. Existing numerical-source lineage comments are retained;
`docs/source_derivation.json` records the minimal source names, checksums and
extraction/adaptation boundaries without private research paths or audit records.

| Included material | Human decision required before publication |
|---|---|
| L63/KS `src/dm_main.py`, `src/krr_model.py`, extracted experiment utilities | Confirm copyright holders/coauthors, upstream revisions and redistribution terms for shared diffusion-kernel and historical research code |
| KS `src/ks_forced_etdrk4_stepper.py` | Resolve the retained `KSmodel/test.py` implementation reference and contributor rights; renamed classes are numerical steppers, with no neural network included |
| Kinetic `kinetic_closure/sbgk_*`, finite-volume and equilibrium utilities | Confirm legacy MATLAB SBGK authors/upstream and rights for translated or adapted routines; retain Mieussens conservative-equilibrium method attribution |
| IC memberships, expected-result summaries and future external arrays | Confirm data producers, permissions and attribution for the specific distributed files |
| New public orchestration and derived routines | Confirm coauthor ownership and choose an appropriate project license only after review |

The Mieussens equilibrium, IMEX, Rusanov finite-volume and ETDRK4 method names
identify numerical methods; method attribution does not establish implementation
licensing. Dependencies are not vendored. NumPy, SciPy, CuPy/CUDA, Matplotlib and
pytest retain their respective upstream terms.

External comparison attribution: Harlim, Jiang, Liang and Yang, “Machine learning
for prediction with missing dynamics,” *Journal of Computational Physics* 428
(2021), 109922, manuscript reference [14]. The LSTM comparison uses an
implementation/closure framework developed in that prior work; neural-network
implementation code is not duplicated here.
