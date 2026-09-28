"""Canonical manuscript KRR for the separately versioned learning stage.

Pure array operations; no truth access, preprocessing, heuristic estimation,
grid freezing, or scientific execution on import. Ridge is the matrix shift.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.spatial.distance import cdist

VERSION = "public_krr_v1"
TRUTH_PROTOCOL = "public_kinetic_v1"
TRUTH_PROTOCOL_SHA256 = None
KERNEL_FORMULA = "exp(-D2 / (4 * kernel_epsilon))"
RIDGE_FORMULA = "(K + lambda_reg * I) A = Y"


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _matrix(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or 0 in value.shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    return value


def _count(N):
    if isinstance(N, (bool, np.bool_)) or not isinstance(N, (int, np.integer)) or N <= 0:
        raise ValueError("N must be a positive integer")
    return int(N)


def legacy_parameters(*, kernel_epsilon, lambda_reg, N):
    """The sole canonical -> historical unweighted kinetic compatibility layer."""
    kernel_epsilon = _positive(kernel_epsilon, "kernel_epsilon")
    lambda_reg = _positive(lambda_reg, "lambda_reg")
    N = _count(N)
    return {"ell": _positive(np.sqrt(2.0) * np.sqrt(kernel_epsilon), "ell"),
            "lambda_bar": _positive(lambda_reg / N, "lambda_bar")}


def canonical_parameters(*, ell, lambda_bar, N):
    """Inverse compatibility conversion; not a fitting API."""
    ell = _positive(ell, "ell")
    lambda_bar = _positive(lambda_bar, "lambda_bar")
    return {"kernel_epsilon": _positive((ell / np.sqrt(2.0)) ** 2, "kernel_epsilon"),
            "lambda_reg": _positive(_count(N) * lambda_bar, "lambda_reg")}


def kernel_from_squared_distances(D2, *, kernel_epsilon):
    kernel_epsilon = _positive(kernel_epsilon, "kernel_epsilon")
    D2 = _matrix(D2, "D2")
    if np.any(D2 < 0):
        raise ValueError("D2 must be nonnegative")
    # Dividing sequentially avoids overflow of 4 * kernel_epsilon.
    with np.errstate(over="ignore"):
        return np.exp(-(D2 / kernel_epsilon) / 4.0)


def kernel(X, Z=None, *, kernel_epsilon):
    X = _matrix(X, "X")
    Z = X if Z is None else _matrix(Z, "Z")
    if X.shape[1] != Z.shape[1]:
        raise ValueError("Feature dimensions differ")
    return kernel_from_squared_distances(cdist(X, Z, metric="sqeuclidean"),
                                         kernel_epsilon=kernel_epsilon)


def regularized_matrix(K, *, lambda_reg):
    lambda_reg = _positive(lambda_reg, "lambda_reg")
    K = _matrix(K, "K")
    if K.shape[0] != K.shape[1] or not np.array_equal(K, K.T):
        raise ValueError("K must be square and symmetric")
    result = K.copy()
    result.flat[::len(K) + 1] += lambda_reg
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite regularized matrix")
    return result


def solve(K, Y, *, lambda_reg):
    """Float64 Cholesky, no N scaling, clipping, fallback, or hidden jitter."""
    Y = _matrix(Y, "Y")
    A = regularized_matrix(K, lambda_reg=lambda_reg)
    if len(Y) != len(A):
        raise ValueError("Target row count differs")
    result = cho_solve(cho_factor(A, lower=True, overwrite_a=True), Y)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite coefficients")
    return result


@dataclass(frozen=True)
class SpectralCache:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    projected_targets: np.ndarray

    def coefficients(self, *, lambda_reg):
        shifted = self.eigenvalues + _positive(lambda_reg, "lambda_reg")
        if not np.isfinite(shifted).all() or np.any(shifted <= 0):
            raise ValueError("Nonpositive/nonfinite shifted spectrum; no repair allowed")
        result = self.eigenvectors @ (self.projected_targets / shifted[:, None])
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite coefficients")
        return result


def prepare_spectral(K, Y):
    K, Y = _matrix(K, "K"), _matrix(Y, "Y")
    if K.shape != (len(Y), len(Y)) or not np.array_equal(K, K.T):
        raise ValueError("Kernel/target shape or symmetry mismatch")
    values, vectors = eigh(K, driver="evd")
    return SpectralCache(values, vectors, vectors.T @ Y)


def candidate_record(*, kernel_epsilon, lambda_reg):
    kernel_epsilon = _positive(kernel_epsilon, "kernel_epsilon")
    lambda_reg = _positive(lambda_reg, "lambda_reg")
    identity = json.dumps([VERSION, kernel_epsilon.hex(), lambda_reg.hex()], separators=(",", ":"))
    return {"candidate_id": "ke_lr_" + hashlib.sha256(identity.encode()).hexdigest(),
            "kernel_epsilon": kernel_epsilon, "lambda_reg": lambda_reg,
            "log10_kernel_epsilon": float(np.log10(kernel_epsilon)),
            "log10_lambda_reg": float(np.log10(lambda_reg))}


def proposed_grid(*, kernel_epsilon_reference, lambda_reg_reference):
    """In-memory proposal only. Preserve L63 log midpoints; no disk/freeze API.

    References must later come from final training geometry. Synthetic callers
    are allowed for engineering tests. L63 ridge midpoint is 10*reference.
    """
    e = _positive(kernel_epsilon_reference, "kernel_epsilon_reference")
    r = _positive(lambda_reg_reference, "lambda_reg_reference")
    e_axis = e * 10.0 ** (-2.0 + np.arange(17) / 4.0)
    r_axis = r * 10.0 ** (-1.0 + np.arange(17) / 4.0)
    return {"schema": VERSION + "_grid_proposal", "frozen": False,
            "candidates": [candidate_record(kernel_epsilon=a, lambda_reg=b)
                           for a in e_axis for b in r_axis]}


@dataclass(frozen=True)
class Model:
    X_train: np.ndarray
    coefficients: np.ndarray
    kernel_epsilon: float
    lambda_reg: float

    def predict(self, X):
        return kernel(X, self.X_train, kernel_epsilon=self.kernel_epsilon) @ self.coefficients

    def metadata(self):
        return {"schema": VERSION, "truth_protocol": TRUTH_PROTOCOL,
                "truth_protocol_sha256": TRUTH_PROTOCOL_SHA256,
                **candidate_record(kernel_epsilon=self.kernel_epsilon, lambda_reg=self.lambda_reg),
                "kernel_formula": KERNEL_FORMULA, "ridge_formula": RIDGE_FORMULA,
                "N": len(self.X_train)}

    def save(self, path):
        # Exclusive creation prevents accidental replacement of an artifact.
        with Path(path).open("xb") as f:
            np.savez(f, X_train=self.X_train, coefficients=self.coefficients,
                     metadata_json=np.asarray(json.dumps(self.metadata(), sort_keys=True, allow_nan=False)))


def fit(X, Y, *, kernel_epsilon, lambda_reg):
    X = _matrix(X, "X").copy()
    kernel_epsilon = _positive(kernel_epsilon, "kernel_epsilon")
    lambda_reg = _positive(lambda_reg, "lambda_reg")
    coefficients = solve(kernel(X, kernel_epsilon=kernel_epsilon), Y, lambda_reg=lambda_reg)
    return Model(X, coefficients, kernel_epsilon, lambda_reg)


def load_model(path):
    with np.load(path, allow_pickle=False) as z:
        X = _matrix(z["X_train"], "X_train").copy()
        coefficients = _matrix(z["coefficients"], "coefficients").copy()
        meta = json.loads(str(z["metadata_json"]))
    model = Model(X, coefficients, _positive(meta["kernel_epsilon"], "kernel_epsilon"),
                  _positive(meta["lambda_reg"], "lambda_reg"))
    if len(X) != len(coefficients) or meta != model.metadata():
        raise ValueError("Model contract/metadata mismatch")
    return model
