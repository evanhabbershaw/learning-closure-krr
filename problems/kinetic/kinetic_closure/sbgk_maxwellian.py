"""Maxwellian construction for the legacy SBGK Sod reference case."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_maxwellian(moments: Any, vC: Any) -> np.ndarray:
    """Match legacy_matlab/SBGK_sod/SBGK_getMax.m.

    Shape conventions follow the MATLAB exports:
    moments has shape (3, Nx), and the returned Maxwellian has shape (Nv, Nx).
    """
    moments_array = np.asarray(moments, dtype=float)
    v_centers = np.asarray(vC, dtype=float).reshape(-1)

    if moments_array.ndim != 2 or moments_array.shape[0] != 3:
        raise ValueError(
            "Expected moments shape (3, Nx); "
            f"got moments={moments_array.shape}."
        )

    n = moments_array[0, :]
    u = moments_array[1, :]
    theta = moments_array[2, :]

    v_matrix = v_centers[:, np.newaxis]
    exponent = -np.abs(v_matrix - u[np.newaxis, :]) ** 2 / (
        2.0 * theta[np.newaxis, :]
    )

    return n[np.newaxis, :] / np.sqrt(2.0 * np.pi * theta[np.newaxis, :]) * np.exp(
        exponent
    )
