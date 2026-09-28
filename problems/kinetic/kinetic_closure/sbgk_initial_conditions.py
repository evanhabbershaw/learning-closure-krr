"""Initial conditions for SBGK Sod benchmark problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .equilibrium import equilibrium_from_primitives


@dataclass(frozen=True)
class PrimitiveFields:
    rho: np.ndarray
    u: np.ndarray
    theta: np.ndarray


def make_local_maxwellian_initial_condition(
    grid: Any,
    rho: Any,
    u: Any,
    theta: Any,
    *,
    equilibrium_mode="continuous",
    discrete_equilibrium_options=None,
) -> np.ndarray:
    """Construct f(v, x) from spatial primitive fields using the selected equilibrium policy."""
    rho_array = _validate_spatial_field(rho, grid, "rho")
    u_array = _validate_spatial_field(u, grid, "u")
    theta_array = _validate_spatial_field(theta, grid, "theta")

    if np.any(rho_array <= 0.0):
        raise ValueError("rho must be strictly positive.")
    if np.any(theta_array <= 0.0):
        raise ValueError("theta must be strictly positive.")

    moments = np.vstack([rho_array, u_array, theta_array])
    return equilibrium_from_primitives(moments, _grid_value(grid, "vC"),
        equilibrium_mode=equilibrium_mode, discrete_equilibrium_options=discrete_equilibrium_options)


def make_sod_initial_condition(
    grid: Any,
    *,
    left_state: tuple[float, float, float] = (1.0, 0.0, 1.0),
    right_state: tuple[float, float, float] = (0.125, 0.0, 0.8),
    discontinuity: float = 0.0,
    equilibrium_mode="continuous",
    discrete_equilibrium_options=None,
) -> np.ndarray:
    """Match legacy_matlab/SBGK_sod/SBGK_getIC.m for testNumber=1.

    The states are (n, u, theta). Cells with xC <= discontinuity use the left
    Maxwellian; cells to the right use the right Maxwellian.
    """
    x_centers = np.asarray(_grid_value(grid, "xC"), dtype=float).reshape(-1)
    v_centers = np.asarray(_grid_value(grid, "vC"), dtype=float).reshape(-1)
    nx = int(_grid_value(grid, "Nx"))
    nv = int(_grid_value(grid, "Nv"))

    if x_centers.size != nx:
        raise ValueError(f"Expected Nx x-centers; got {x_centers.size}.")
    if v_centers.size != nv:
        raise ValueError(f"Expected Nv velocity centers; got {v_centers.size}.")

    left = np.asarray(left_state, dtype=float)
    right = np.asarray(right_state, dtype=float)
    if left.shape != (3,) or right.shape != (3,):
        raise ValueError("left_state and right_state must be (n, u, theta) triples.")

    left_mask = x_centers <= discontinuity
    rho = np.empty(nx)
    u = np.empty(nx)
    theta = np.empty(nx)
    rho[left_mask] = left[0]
    rho[~left_mask] = right[0]
    u[left_mask] = left[1]
    u[~left_mask] = right[1]
    theta[left_mask] = left[2]
    theta[~left_mask] = right[2]

    return make_local_maxwellian_initial_condition(grid, rho, u, theta,
        equilibrium_mode=equilibrium_mode, discrete_equilibrium_options=discrete_equilibrium_options)


def thermal_gaussian_primitive_fields(
    grid: Any,
    *,
    rho0: float = 1.0,
    u0: float = 0.0,
    theta0: float = 1.0,
    amplitude: float = 0.2,
    sigma: float = 0.2,
    center: float = 0.0,
    periodic: bool = True,
) -> PrimitiveFields:
    """Return primitive fields for a thermal Gaussian pulse."""
    if rho0 <= 0.0:
        raise ValueError("rho0 must be strictly positive.")
    if theta0 <= 0.0:
        raise ValueError("theta0 must be strictly positive.")
    if sigma <= 0.0:
        raise ValueError("sigma must be strictly positive.")

    x_centers = np.asarray(_grid_value(grid, "xC"), dtype=float).reshape(-1)
    if periodic:
        distance = periodic_distance(
            x_centers,
            center,
            _grid_value(grid, "xMin"),
            _grid_value(grid, "xMax"),
        )
    else:
        distance = x_centers - center

    rho = np.full(x_centers.shape, rho0, dtype=float)
    u = np.full(x_centers.shape, u0, dtype=float)
    theta = theta0 + amplitude * np.exp(-((distance / sigma) ** 2))

    return PrimitiveFields(rho=rho, u=u, theta=theta)


def make_thermal_gaussian_initial_condition(
    grid: Any,
    *,
    rho0: float = 1.0,
    u0: float = 0.0,
    theta0: float = 1.0,
    amplitude: float = 0.2,
    sigma: float = 0.2,
    center: float = 0.0,
    periodic: bool = True,
    equilibrium_mode="continuous",
    discrete_equilibrium_options=None,
) -> np.ndarray:
    """Construct a local Maxwellian thermal Gaussian pulse."""
    fields = thermal_gaussian_primitive_fields(
        grid,
        rho0=rho0,
        u0=u0,
        theta0=theta0,
        amplitude=amplitude,
        sigma=sigma,
        center=center,
        periodic=periodic,
    )
    return make_local_maxwellian_initial_condition(
        grid,
        fields.rho,
        fields.u,
        fields.theta,
        equilibrium_mode=equilibrium_mode, discrete_equilibrium_options=discrete_equilibrium_options,
    )


def periodic_distance(x: Any, center: float, x_min: float, x_max: float) -> np.ndarray:
    """Signed shortest periodic distance from center on [x_min, x_max)."""
    x_array = np.asarray(x, dtype=float)
    length = x_max - x_min
    if length <= 0.0:
        raise ValueError("x_max must be greater than x_min.")
    return ((x_array - center + 0.5 * length) % length) - 0.5 * length


def _validate_spatial_field(value: Any, grid: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    nx = int(_grid_value(grid, "Nx"))
    if array.shape != (nx,):
        raise ValueError(f"{name} must have shape ({nx},); got {array.shape}.")
    return array.copy()


def _grid_value(grid: Any, field: str) -> Any:
    if isinstance(grid, dict):
        return grid[field]
    return getattr(grid, field)
