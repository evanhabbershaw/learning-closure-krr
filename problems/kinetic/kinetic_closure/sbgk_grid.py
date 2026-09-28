"""Grid construction for the tiny SBGK Sod reference case."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SBGKGrid:
    Nx: int
    Nv: int
    xMin: float
    xMax: float
    vMin: float
    vMax: float
    tInit: float
    thetaX: float
    thetaV: float
    hX: float
    hV: float
    xC: np.ndarray
    vC: np.ndarray
    tFinal: float


def make_sbgk_grid(
    Nx: int,
    Nv: int,
    x_min: float,
    x_max: float,
    v_min: float,
    v_max: float,
    *,
    t_final: float,
    t_init: float = 0.0,
    theta_x: float = 2.0,
    theta_v: float = 2.0,
) -> SBGKGrid:
    """Construct an SBGK grid using the legacy MATLAB midpoint-cell formula."""
    x = np.linspace(x_min, x_max, Nx + 1)
    h_x = x[1] - x[0]
    x_c = x[:-1] + 0.5 * (x[1:] - x[:-1])

    v = np.linspace(v_min, v_max, Nv + 1)
    h_v = v[1] - v[0]
    v_c = v[:-1] + 0.5 * (v[1:] - v[:-1])

    return SBGKGrid(
        Nx=Nx,
        Nv=Nv,
        xMin=x_min,
        xMax=x_max,
        vMin=v_min,
        vMax=v_max,
        tInit=t_init,
        thetaX=theta_x,
        thetaV=theta_v,
        hX=h_x,
        hV=h_v,
        xC=x_c,
        vC=v_c,
        tFinal=t_final,
    )


def make_tiny_sod_grid() -> SBGKGrid:
    """Reproduce the grid setup in reference_exports/matlab/sbgk_sod_export_reference.m."""
    return make_sbgk_grid(
        8,
        16,
        -1.0,
        1.0,
        -8.0,
        8.0,
        t_final=0.9 * 0.25 / (2.0 * 8.0),
    )
