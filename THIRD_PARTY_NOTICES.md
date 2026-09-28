# Attribution and source provenance

This repository accompanies the manuscript *Learning Closure of Dynamical Systems
with Kernel Ridge Regression*. It contains new reproducibility/orchestration code
together with selected numerical and KRR routines adapted from the research code
used in the development of this work and related earlier work.

For the Lorenz-63 and Kuramoto--Sivashinsky experiments, portions of the included
KRR and experiment utilities were adapted from the shared research-code lineage
associated with the earlier KRR work. The file
`docs/source_derivation.json` records source names, checksums, and the
extraction/adaptation boundaries used to construct this release.

The kinetic SBGK implementation and its Python adaptation were developed by Evan
Habbershaw. Numerical methods used by the kinetic solver, including conservative
discrete-equilibrium, IMEX, and finite-volume components, are identified and
attributed in the manuscript and source comments where appropriate.

The LSTM comparison is an external comparison associated with:

Harlim, Jiang, Liang and Yang, “Machine learning for prediction with missing
dynamics,” *Journal of Computational Physics* 428 (2021), 109922.

Neural-network implementation code from that work is not included in this
repository.

Dependencies such as NumPy, SciPy, CuPy/CUDA, Matplotlib, and pytest are not
vendored and retain their respective upstream terms.

No explicit software license is currently included with this repository.
Additional attribution or licensing information may be added following
coauthor/source-provenance review.
