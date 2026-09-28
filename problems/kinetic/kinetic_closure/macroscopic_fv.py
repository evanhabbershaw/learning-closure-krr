"""Periodic finite-volume macroscopic closure stepper utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .sbgk_butcher import ButcherTable


MacroBoundary = Literal["periodic"]


@dataclass(frozen=True)
class MacroPrimitives:
    rho: np.ndarray
    u: np.ndarray
    p: np.ndarray
    theta: np.ndarray


def conservative_to_primitives(U: Any) -> MacroPrimitives:
    """Recover primitive fields from U=(rho, rho*u, E), all shape (Nx,)."""
    U_array = _validate_U(U)
    rho = U_array[0].copy()
    momentum = U_array[1]
    energy = U_array[2]

    if np.any(rho <= 0.0):
        raise ValueError("rho must be strictly positive.")

    u = momentum / rho
    p = 2.0 * energy - momentum**2 / rho
    if np.any(p <= 0.0):
        raise ValueError("pressure p must be strictly positive.")

    theta = p / rho
    if np.any(theta <= 0.0):
        raise ValueError("theta must be strictly positive.")

    return MacroPrimitives(rho=rho, u=u, p=p, theta=theta)


def physical_euler_flux(U: Any) -> np.ndarray:
    """Return the 1D Euler flux without the prescribed heat flux q."""
    U_array = _validate_U(U)
    primitives = conservative_to_primitives(U_array)
    rho = primitives.rho
    u = primitives.u
    p = primitives.p
    energy = U_array[2]

    return np.vstack(
        [
            rho * u,
            rho * u**2 + p,
            (energy + p) * u,
        ]
    )


def max_macroscopic_wave_speed(U: Any) -> float:
    """Return max_x |u| + sqrt(3*theta), consistent with gamma=3."""
    primitives = conservative_to_primitives(U)
    return float(np.max(np.abs(primitives.u) + np.sqrt(3.0 * primitives.theta)))


def macroscopic_cfl_dt(grid: Any, U: Any, cfl: float = 0.9) -> float:
    """Return cfl*hX/max_wave_speed for the macroscopic system."""
    if cfl <= 0.0:
        raise ValueError(f"cfl must be positive; got {cfl}.")
    return float(cfl) * float(_grid_value(grid, "hX")) / max_macroscopic_wave_speed(U)


def compute_macroscopic_rhs(
    U: Any,
    q: Any,
    grid: Any,
    *,
    boundary: MacroBoundary = "periodic",
) -> np.ndarray:
    """Return -(Fhat[i+1/2]-Fhat[i-1/2])/hX for periodic macro FV."""
    if boundary != "periodic":
        raise ValueError(f"Only periodic macro boundary is supported; got {boundary!r}.")

    h_x = float(_grid_value(grid, "hX"))
    flux_hat = compute_macroscopic_flux(U, q, grid, boundary=boundary)
    return -(flux_hat - np.roll(flux_hat, shift=1, axis=1)) / h_x


def compute_macroscopic_flux(
    U: Any,
    q: Any,
    grid: Any,
    *,
    boundary: MacroBoundary = "periodic",
) -> np.ndarray:
    """Return the existing Rusanov flux at faces i+1/2 (column i)."""
    if boundary != "periodic":
        raise ValueError(f"Only periodic macro boundary is supported; got {boundary!r}.")

    U_array = _validate_U(U)
    q_array = _validate_q(q, U_array.shape[1])
    theta_x = float(_grid_value(grid, "thetaX"))
    U_left, U_right = _periodic_face_values(U_array, theta_x)
    q_left, q_right = _periodic_face_values(q_array[np.newaxis, :], theta_x)
    flux_left = physical_euler_flux(U_left)
    flux_right = physical_euler_flux(U_right)
    primitives_left = conservative_to_primitives(U_left)
    primitives_right = conservative_to_primitives(U_right)
    wave_speed = np.maximum(
        np.abs(primitives_left.u) + np.sqrt(3.0 * primitives_left.theta),
        np.abs(primitives_right.u) + np.sqrt(3.0 * primitives_right.theta),
    )
    flux_hat = 0.5 * (flux_left + flux_right) - 0.5 * wave_speed[np.newaxis, :] * (
        U_right - U_left
    )
    flux_hat[2] += 0.5 * (q_left[0] + q_right[0])
    return flux_hat


def macroscopic_explicit_rk_stage_fluxes(
    U: Any,
    q_stages: Any,
    butcher_table: ButcherTable,
    grid: Any,
    dt: float,
    *,
    boundary: MacroBoundary = "periodic",
) -> np.ndarray:
    """Return the interface flux from every explicit RK stage."""
    if dt <= 0.0:
        raise ValueError(f"dt must be positive; got {dt}.")
    U_array = _validate_U(U).copy()
    q_array = _validate_q_stages(
        q_stages, butcher_table.num_stages, U_array.shape[1]
    )
    rhs_stages = np.zeros((butcher_table.num_stages, *U_array.shape))
    flux_stages = np.zeros_like(rhs_stages)
    h_x = float(_grid_value(grid, "hX"))
    for stage in range(butcher_table.num_stages):
        U_stage = U_array.copy()
        for previous in range(stage):
            U_stage += float(dt) * butcher_table.Ae[stage, previous] * rhs_stages[previous]
        flux_stages[stage] = compute_macroscopic_flux(
            U_stage, q_array[stage], grid, boundary=boundary
        )
        rhs_stages[stage] = -(
            flux_stages[stage] - np.roll(flux_stages[stage], shift=1, axis=1)
        ) / h_x
    return flux_stages


def advance_macroscopic_explicit_rk(
    U: Any,
    q_stages: Any,
    butcher_table: ButcherTable,
    grid: Any,
    dt: float,
    *,
    boundary: MacroBoundary = "periodic",
) -> np.ndarray:
    """Advance U by one explicit RK step using butcher_table.Ae/be and q_stages."""
    if dt <= 0.0:
        raise ValueError(f"dt must be positive; got {dt}.")

    U_array = _validate_U(U).copy()
    num_stages = butcher_table.num_stages
    q_array = _validate_q_stages(q_stages, num_stages, U_array.shape[1])
    rhs_stages = np.zeros((num_stages, *U_array.shape))

    for stage in range(num_stages):
        U_stage = U_array.copy()
        for previous in range(stage):
            U_stage += float(dt) * butcher_table.Ae[stage, previous] * rhs_stages[previous]
        rhs_stages[stage] = compute_macroscopic_rhs(
            U_stage,
            q_array[stage],
            grid,
            boundary=boundary,
        )

    U_next = U_array.copy()
    for stage in range(num_stages):
        U_next += float(dt) * butcher_table.be[stage] * rhs_stages[stage]
    return U_next


def _periodic_face_values(values: np.ndarray, theta_x: float) -> tuple[np.ndarray, np.ndarray]:
    """Return left/right PLM states at faces i+1/2 for periodic cell data."""
    slopes = _periodic_limited_slopes(values, theta_x)
    left = values + 0.5 * slopes
    right = np.roll(values, shift=-1, axis=1) - 0.5 * np.roll(slopes, shift=-1, axis=1)
    return left, right


def _periodic_limited_slopes(values: np.ndarray, theta_x: float) -> np.ndarray:
    backward = values - np.roll(values, shift=1, axis=1)
    forward = np.roll(values, shift=-1, axis=1) - values
    centered = 0.5 * (np.roll(values, shift=-1, axis=1) - np.roll(values, shift=1, axis=1))
    candidates = np.stack([centered, theta_x * backward, theta_x * forward], axis=2)
    return _minmod(candidates)


def _minmod(candidates: np.ndarray) -> np.ndarray:
    return (
        0.25
        * np.abs(np.sign(candidates[:, :, 0]) + np.sign(candidates[:, :, 1]))
        * (np.sign(candidates[:, :, 0]) + np.sign(candidates[:, :, 2]))
        * np.min(np.abs(candidates), axis=2)
    )


def _validate_U(U: Any) -> np.ndarray:
    U_array = np.asarray(U, dtype=float)
    if U_array.ndim != 2 or U_array.shape[0] != 3:
        raise ValueError(f"U must have shape (3, Nx); got {U_array.shape}.")
    if U_array.shape[1] == 0:
        raise ValueError("U must contain at least one spatial cell.")
    return U_array


def _validate_q(q: Any, nx: int) -> np.ndarray:
    q_array = np.asarray(q, dtype=float)
    if q_array.shape != (nx,):
        raise ValueError(f"q must have shape ({nx},); got {q_array.shape}.")
    return q_array


def _validate_q_stages(q_stages: Any, num_stages: int, nx: int) -> np.ndarray:
    q_array = np.asarray(q_stages, dtype=float)
    expected = (num_stages, nx)
    if q_array.shape != expected:
        raise ValueError(f"q_stages must have shape {expected}; got {q_array.shape}.")
    return q_array.copy()


def _grid_value(grid: Any, field: str) -> Any:
    if isinstance(grid, dict):
        return grid[field]
    return getattr(grid, field)
