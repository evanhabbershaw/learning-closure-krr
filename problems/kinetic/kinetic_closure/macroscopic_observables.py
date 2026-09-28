"""Macroscopic observables from a one-dimensional velocity distribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MacroscopicObservables:
    n: np.ndarray
    rho: np.ndarray
    u: np.ndarray
    theta: np.ndarray
    p: np.ndarray
    E: np.ndarray
    q: np.ndarray
    U: np.ndarray


def compute_macroscopic_observables(
    f: Any,
    vC: Any,
    mass: float = 1.0,
) -> MacroscopicObservables:
    """Match legacy_matlab/SBGK_sod/moments_plus_q.m.

    The input distribution uses shape (Nv, Nx). Scalar outputs use shape (Nx,),
    and the conservative state U uses shape (3, Nx) with rows (rho, rho*u, E).
    """
    f_array = np.asarray(f, dtype=float)
    v_centers = np.asarray(vC, dtype=float).reshape(-1)

    if f_array.ndim != 2:
        raise ValueError(f"Expected f shape (Nv, Nx); got {f_array.shape}.")
    if f_array.shape[0] != v_centers.size:
        raise ValueError(
            "Expected f velocity dimension to match vC; "
            f"got f Nv={f_array.shape[0]}, vC size={v_centers.size}."
        )

    h_v = v_centers[1] - v_centers[0]

    n = h_v * np.sum(f_array, axis=0)
    momentum = h_v * (v_centers @ f_array)
    n_safe = np.maximum(n, np.finfo(float).eps)
    u = momentum / n_safe

    v_minus_u = v_centers[:, np.newaxis] - u[np.newaxis, :]
    variance_integral = h_v * np.sum((v_minus_u**2) * f_array, axis=0)
    theta = variance_integral / n_safe

    rho = mass * n
    p = rho * theta
    energy = 0.5 * rho * (u**2) + 0.5 * rho * theta
    q = (mass / 2.0) * h_v * np.sum((v_minus_u**2) * v_minus_u * f_array, axis=0)
    U = np.vstack([rho, rho * u, energy])

    return MacroscopicObservables(
        n=n,
        rho=rho,
        u=u,
        theta=theta,
        p=p,
        E=energy,
        q=q,
        U=U,
    )
