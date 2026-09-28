"""Advection right-hand side for the legacy SBGK Sod reference case."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np


KineticBoundary = Literal["reflective", "periodic"]


def compute_advection_rhs(
    f: Any,
    grid: Any,
    *,
    boundary: KineticBoundary = "reflective",
) -> np.ndarray:
    """Match legacy_matlab/SBGK_sod/SBGK_advFunc.m.

    Shape conventions follow the MATLAB exports: f and the returned RHS both
    have shape (Nv, Nx). Reflective boundary mode is the legacy MATLAB behavior.
    """
    h_x = float(_grid_value(grid, "hX"))
    flux_right, flux_left = _compute_advection_flux_pair(f, grid, boundary=boundary)

    return -(flux_right - flux_left) / h_x


def compute_advection_fluxes(
    f: Any,
    grid: Any,
    *,
    boundary: KineticBoundary = "reflective",
) -> np.ndarray:
    """Return kinetic numerical fluxes Fhat_f at faces i+1/2.

    The returned array has shape (Nv, Nx), where column i is the flux through
    the right face of physical cell i. Its divergence reproduces
    compute_advection_rhs via -(Fhat_i+1/2 - Fhat_i-1/2)/hX. The construction is
    intentionally the same PLM/minmod/upwind operation used by the RHS.
    """
    flux_right, _ = _compute_advection_flux_pair(f, grid, boundary=boundary)
    return flux_right


def _compute_advection_flux_pair(
    f: Any,
    grid: Any,
    *,
    boundary: KineticBoundary,
) -> tuple[np.ndarray, np.ndarray]:
    if boundary not in ("reflective", "periodic"):
        raise ValueError(
            "boundary must be either 'reflective' or 'periodic'; "
            f"got {boundary!r}."
        )

    f_array = np.asarray(f, dtype=float)
    nx = int(_grid_value(grid, "Nx"))
    nv = int(_grid_value(grid, "Nv"))
    h_x = float(_grid_value(grid, "hX"))
    theta_x = float(_grid_value(grid, "thetaX"))
    v_centers = np.asarray(_grid_value(grid, "vC"), dtype=float).reshape(-1)

    if f_array.shape != (nv, nx):
        raise ValueError(
            "Expected f shape (Nv, Nx) matching grid; "
            f"got f={f_array.shape}, Nv={nv}, Nx={nx}."
        )
    if v_centers.size != nv:
        raise ValueError(f"Expected Nv velocity centers; got {v_centers.size}.")

    del_x = 1
    left = del_x
    right = left + nx

    y = np.zeros((nv, nx + 2 * del_x))
    y[:, left:right] = f_array
    _fill_spatial_ghost_cells(y, left=left, right=right, boundary=boundary)

    s_x = _slopes_x(y, h_x=h_x, theta_x=theta_x, left=left, right=right)
    _fill_spatial_ghost_slopes(
        s_x,
        y,
        v_centers,
        h_x=h_x,
        theta_x=theta_x,
        left=left,
        right=right,
        boundary=boundary,
    )

    v_pos = (v_centers * (v_centers > 0))[:, np.newaxis]
    v_neg = (v_centers * (v_centers <= 0))[:, np.newaxis]

    f_p = np.zeros_like(y)
    f_m = np.zeros_like(y)

    interior = slice(left, right)
    plus = slice(left + 1, right + 1)
    minus = slice(left - 1, right - 1)

    f_p[:, interior] = (y[:, interior] + 0.5 * h_x * s_x[:, interior]) * v_pos + (
        y[:, plus] - 0.5 * h_x * s_x[:, plus]
    ) * v_neg
    f_m[:, interior] = (y[:, minus] + 0.5 * h_x * s_x[:, minus]) * v_pos + (
        y[:, interior] - 0.5 * h_x * s_x[:, interior]
    ) * v_neg

    return f_p[:, interior], f_m[:, interior]


def _fill_spatial_ghost_cells(
    y: np.ndarray,
    *,
    left: int,
    right: int,
    boundary: KineticBoundary,
) -> None:
    if boundary == "reflective":
        # MATLAB loop: y(iv,left ghost) = y(Nv+1-iv, first interior), and
        # likewise at the right edge. This is reflective in velocity index.
        y[:, left - 1] = y[::-1, left]
        y[:, right] = y[::-1, right - 1]
        return

    y[:, left - 1] = y[:, right - 1]
    y[:, right] = y[:, left]


def _fill_spatial_ghost_slopes(
    s_x: np.ndarray,
    y: np.ndarray,
    v_centers: np.ndarray,
    *,
    h_x: float,
    theta_x: float,
    left: int,
    right: int,
    boundary: KineticBoundary,
) -> None:
    if boundary == "reflective":
        pos_mask = v_centers > 0
        neg_mask = v_centers < 0
        s_x[pos_mask, right] = theta_x * (y[pos_mask, -1] - y[pos_mask, -2]) / h_x
        s_x[neg_mask, left - 1] = theta_x * (
            y[neg_mask, left + 1] - y[neg_mask, left]
        ) / h_x
        return

    # Periodic ghosts represent the opposite physical edge, so their slopes
    # must use the corresponding physical-cell slopes.
    s_x[:, left - 1] = s_x[:, right - 1]
    s_x[:, right] = s_x[:, left]


def _slopes_x(y: np.ndarray, *, h_x: float, theta_x: float, left: int, right: int) -> np.ndarray:
    """Match slopesX.m for interior x cells."""
    s_x = np.zeros_like(y)

    sx = np.empty((y.shape[0], right - left, 3))
    sx[:, :, 0] = (y[:, left + 1 : right + 1] - y[:, left - 1 : right - 1]) / 2.0
    sx[:, :, 1] = theta_x * (y[:, left:right] - y[:, left - 1 : right - 1])
    sx[:, :, 2] = theta_x * (y[:, left + 1 : right + 1] - y[:, left:right])

    s_x[:, left:right] = _minmod(sx) / h_x
    return s_x


def _minmod(x: np.ndarray) -> np.ndarray:
    """Match minmod.m over the final axis of a 3-slope array."""
    return (
        0.25
        * np.abs(np.sign(x[:, :, 0]) + np.sign(x[:, :, 1]))
        * (np.sign(x[:, :, 0]) + np.sign(x[:, :, 2]))
        * np.min(np.abs(x), axis=2)
    )


def _grid_value(grid: Any, field: str) -> Any:
    if isinstance(grid, dict):
        return grid[field]
    return getattr(grid, field)
