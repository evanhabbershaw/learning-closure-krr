"""Moment calculations for the legacy SBGK Sod reference case."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_moments(f: Any, xC: Any, vC: Any, t: float) -> np.ndarray:
    """Match legacy_matlab/SBGK_sod/SBGK_moments.m.

    Shape conventions follow the MATLAB exports:
    f has shape (Nv, Nx), and the returned moments array has shape (3, Nx).
    """
    del t

    f_array = np.asarray(f, dtype=float)
    x_centers = np.asarray(xC, dtype=float).reshape(-1)
    v_centers = np.asarray(vC, dtype=float).reshape(-1)

    if f_array.shape != (v_centers.size, x_centers.size):
        raise ValueError(
            "Expected f shape (Nv, Nx) matching vC and xC; "
            f"got f={f_array.shape}, Nv={v_centers.size}, Nx={x_centers.size}."
        )

    h_v = v_centers[1] - v_centers[0]

    n_dens = h_v * np.sum(f_array, axis=0)
    momentum_1d = h_v * np.sum(v_centers[:, np.newaxis] * f_array, axis=0)
    u_out = momentum_1d / n_dens

    v_minus_u_sq = (v_centers[:, np.newaxis] - u_out[np.newaxis, :]) ** 2
    fv_diff_sq_integral = h_v * np.sum(v_minus_u_sq * f_array, axis=0)
    t_out = fv_diff_sq_integral / n_dens

    return np.vstack([n_dens, u_out, t_out])
