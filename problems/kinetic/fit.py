"""Canonical fit health; residual targets bind a separately fitted baseline."""
import numpy as np
from scipy.spatial.distance import pdist
from .kinetic_closure import final_learning_krr_20260920_v1 as krr
from .selection import BASELINE


def epsilon_reference(features):
    return float(np.median(pdist(features, metric='sqeuclidean'))/(4*np.log(2)))


def fit_ridges(features, targets, epsilon, ridges):
    gram = krr.kernel(features, kernel_epsilon=epsilon)
    cache = krr.prepare_spectral(gram, targets)
    for ridge in ridges:
        try:
            coeff = cache.coefficients(lambda_reg=ridge)
            residual = float(np.linalg.norm(gram@coeff+ridge*coeff-targets)/max(np.linalg.norm(targets), np.finfo(float).tiny))
            if not np.isfinite(residual) or residual > 1e-6:
                raise ValueError('Canonical solve residual exceeds 1e-6; no repair')
            yield krr.Model(features, coeff, epsilon, ridge), {'health':'PASS', 'relative_solve_residual':residual}
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            yield None, {'health':'FAIL', 'failure':str(exc)}


def fit_baseline(features, targets):
    if features.shape != (4032,30) or targets.shape != (4032,3):
        raise ValueError('Fixed Type-I baseline requires 4032 U30 centers')
    model = krr.fit(features, targets, kernel_epsilon=BASELINE['kernel_epsilon'], lambda_reg=BASELINE['lambda_reg'])
    gram = krr.kernel(features, kernel_epsilon=model.kernel_epsilon)
    residual = float(np.linalg.norm(gram@model.coefficients+model.lambda_reg*model.coefficients-targets)/max(np.linalg.norm(targets), np.finfo(float).tiny))
    if not np.isfinite(residual) or residual > 1e-6:
        raise ValueError('Baseline Cholesky solve residual exceeds 1e-6; no fallback')
    return model
