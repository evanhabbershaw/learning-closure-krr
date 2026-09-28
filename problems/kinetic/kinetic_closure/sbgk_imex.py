"""Minimal kinetic IMEX update for the legacy SBGK Sod reference case."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np

from .macroscopic_observables import compute_macroscopic_observables
from .discrete_maxwellian import (
    DiscreteMaxwellianDiagnostics,
    DiscreteMaxwellianOptions,
)
from .sbgk_advection import KineticBoundary, compute_advection_rhs
from .sbgk_butcher import ButcherTable
from .equilibrium import (EquilibriumMode, equilibrium_from_distribution,
    validate_equilibrium_policy, equilibrium_metadata)


StageAdvectionMode = Literal["original", "corrected"]


@dataclass(frozen=True)
class KineticTrajectory:
    times: np.ndarray
    f_history: np.ndarray
    U_history: np.ndarray
    q_history: np.ndarray
    dt: float
    equilibrium_metadata: dict | None = None


def solve_one_step(
    f0: Any,
    butcher_table: ButcherTable,
    grid: Any,
    epsi: float,
    *,
    stage_advection_mode: StageAdvectionMode = "corrected",
    boundary: KineticBoundary = "reflective",
    equilibrium_mode: EquilibriumMode = "continuous",
    discrete_equilibrium_options: DiscreteMaxwellianOptions | None = None,
    equilibrium_diagnostic_callback: Callable[
        [DiscreteMaxwellianDiagnostics], None
    ] | None = None,
) -> np.ndarray:
    """Return the final kinetic distribution from SBGK_imexSolver.m.

    This intentionally omits the MATLAB diagnostic/file-writing path because it
    does not feed back into the returned Zf. Use stage_advection_mode="original"
    to preserve the legacy bug where later stages advect fStage(:,:,1).
    """
    if stage_advection_mode not in ("original", "corrected"):
        raise ValueError(
            "stage_advection_mode must be either 'original' or 'corrected'; "
            f"got {stage_advection_mode!r}."
        )

    validate_equilibrium_policy(equilibrium_mode, discrete_equilibrium_options)
    f = np.array(f0, dtype=float, copy=True, order="K")
    _validate_distribution_shape(f, grid)
    t_current = float(_grid_value(grid, "tInit"))
    t_final = float(_grid_value(grid, "tFinal"))

    c_max_1 = kinetic_cfl_dt(grid) / 0.9
    plot_times = np.asarray([0.025, 0.1, t_final], dtype=float)
    stamp_number = 0
    time_stamp = plot_times[stamp_number]
    redo = 0

    while t_current < t_final:
        possible_time_steps = np.asarray(
            [0.9 * c_max_1, t_final - t_current, time_stamp - t_current],
            dtype=float,
        )
        dt = float(np.min(possible_time_steps) / (1.0 + float(redo > 0)))
        redo = 0

        f = advance_kinetic_fixed_dt(
            f,
            butcher_table,
            grid,
            epsi,
            dt,
            stage_advection_mode=stage_advection_mode,
            boundary=boundary,
            equilibrium_mode=equilibrium_mode,
            discrete_equilibrium_options=discrete_equilibrium_options,
            equilibrium_diagnostic_callback=equilibrium_diagnostic_callback,
        )
        t_current = t_current + dt

        if t_current == time_stamp and stamp_number < plot_times.size - 1:
            stamp_number += 1
            time_stamp = plot_times[stamp_number]

    return f


def advance_kinetic_fixed_dt(
    f: Any,
    butcher_table: ButcherTable,
    grid: Any,
    epsi: float,
    dt: float,
    *,
    stage_advection_mode: StageAdvectionMode = "corrected",
    boundary: KineticBoundary = "reflective",
    equilibrium_mode: EquilibriumMode = "continuous",
    discrete_equilibrium_options: DiscreteMaxwellianOptions | None = None,
    equilibrium_diagnostic_callback: Callable[
        [DiscreteMaxwellianDiagnostics], None
    ] | None = None,
) -> np.ndarray:
    """Advance the kinetic distribution by exactly one fixed physical dt."""
    if stage_advection_mode not in ("original", "corrected"):
        raise ValueError(
            "stage_advection_mode must be either 'original' or 'corrected'; "
            f"got {stage_advection_mode!r}."
        )
    if dt <= 0.0:
        raise ValueError(f"dt must be positive; got {dt}.")
    validate_equilibrium_policy(equilibrium_mode, discrete_equilibrium_options)

    f_array = np.array(f, dtype=float, copy=True, order="K")
    _validate_distribution_shape(f_array, grid)

    return _advance_kinetic_stage_update(
        f_array,
        butcher_table,
        grid,
        float(epsi),
        float(dt),
        stage_advection_mode=stage_advection_mode,
        boundary=boundary,
        equilibrium_mode=equilibrium_mode,
        discrete_equilibrium_options=discrete_equilibrium_options,
        equilibrium_diagnostic_callback=equilibrium_diagnostic_callback,
    )


def generate_kinetic_trajectory(
    f0: Any,
    butcher_table: ButcherTable,
    grid: Any,
    epsi: float,
    n_steps: int,
    dt: float,
    *,
    stage_advection_mode: StageAdvectionMode = "corrected",
    boundary: KineticBoundary = "reflective",
    equilibrium_mode: EquilibriumMode = "continuous",
    discrete_equilibrium_options: DiscreteMaxwellianOptions | None = None,
    equilibrium_diagnostic_callback: Callable[
        [DiscreteMaxwellianDiagnostics], None
    ] | None = None,
) -> KineticTrajectory:
    """Generate a fixed-dt kinetic trajectory and save every numerical step."""
    if n_steps < 0:
        raise ValueError(f"n_steps must be nonnegative; got {n_steps}.")

    f = np.array(f0, dtype=float, copy=True, order="K")
    _validate_distribution_shape(f, grid)
    validate_equilibrium_policy(equilibrium_mode, discrete_equilibrium_options)

    nx = int(_grid_value(grid, "Nx"))
    nv = int(_grid_value(grid, "Nv"))
    t_init = float(_grid_value(grid, "tInit"))
    v_centers = np.asarray(_grid_value(grid, "vC"), dtype=float).reshape(-1)

    times = t_init + float(dt) * np.arange(n_steps + 1, dtype=float)
    f_history = np.zeros((n_steps + 1, nv, nx))
    U_history = np.zeros((n_steps + 1, 3, nx))
    q_history = np.zeros((n_steps + 1, nx))

    _store_trajectory_state(f, v_centers, 0, f_history, U_history, q_history)

    for step in range(1, n_steps + 1):
        f = advance_kinetic_fixed_dt(
            f,
            butcher_table,
            grid,
            epsi,
            dt,
            stage_advection_mode=stage_advection_mode,
            boundary=boundary,
            equilibrium_mode=equilibrium_mode,
            discrete_equilibrium_options=discrete_equilibrium_options,
            equilibrium_diagnostic_callback=equilibrium_diagnostic_callback,
        )
        _store_trajectory_state(f, v_centers, step, f_history, U_history, q_history)

    return KineticTrajectory(
        times=times,
        f_history=f_history,
        U_history=U_history,
        q_history=q_history,
        dt=float(dt),
        equilibrium_metadata=equilibrium_metadata(equilibrium_mode, discrete_equilibrium_options),
    )


def kinetic_cfl_dt(grid: Any) -> float:
    """Return the kinetic CFL dt: 0.9*hX/(2*max(abs(vMin), abs(vMax)))."""
    h_x = float(_grid_value(grid, "hX"))
    v_min = float(_grid_value(grid, "vMin"))
    v_max = float(_grid_value(grid, "vMax"))
    return 0.9 * h_x / (2.0 * max(abs(v_min), abs(v_max)))


def _advance_kinetic_stage_update(
    f: np.ndarray,
    butcher_table: ButcherTable,
    grid: Any,
    epsi: float,
    dt: float,
    *,
    stage_advection_mode: StageAdvectionMode,
    boundary: KineticBoundary,
    equilibrium_mode: EquilibriumMode,
    discrete_equilibrium_options: DiscreteMaxwellianOptions | None,
    equilibrium_diagnostic_callback: Callable[
        [DiscreteMaxwellianDiagnostics], None
    ] | None,
) -> np.ndarray:
    nx = int(_grid_value(grid, "Nx"))
    nv = int(_grid_value(grid, "Nv"))
    v_centers = np.asarray(_grid_value(grid, "vC"), dtype=float).reshape(-1)
    num_stages = butcher_table.num_stages

    f_star = np.zeros((num_stages, nv, nx))
    f_stage = np.zeros((num_stages, nv, nx))
    f_explicit = np.zeros((num_stages, nv, nx))
    f_implicit = np.zeros((num_stages, nv, nx))
    maxwellian = np.zeros((num_stages, nv, nx))

    f_star[0] = f
    maxwellian[0] = _compute_equilibrium(
        f,
        v_centers,
        equilibrium_mode,
        discrete_equilibrium_options,
        equilibrium_diagnostic_callback,
    )
    f_stage[0] = (
        epsi * f_star[0] + dt * butcher_table.Ai[0, 0] * maxwellian[0]
    ) / (epsi + dt * butcher_table.Ai[0, 0])

    f_explicit[0] = compute_advection_rhs(f_stage[0], grid, boundary=boundary)
    f_implicit[0] = maxwellian[0] - f_stage[0]

    for stage in range(1, num_stages):
        explicit_sum = _stage_sum(butcher_table.Ae[stage], f_explicit)
        implicit_sum = _stage_sum(butcher_table.Ai[stage], f_implicit)

        f_star[stage] = f + dt * explicit_sum + (dt / epsi) * implicit_sum
        maxwellian[stage] = _compute_equilibrium(
            f_star[stage],
            v_centers,
            equilibrium_mode,
            discrete_equilibrium_options,
            equilibrium_diagnostic_callback,
        )
        f_stage[stage] = (
            epsi * f_star[stage]
            + dt * butcher_table.Ai[stage, stage] * maxwellian[stage]
        ) / (epsi + dt * butcher_table.Ai[stage, stage])

        advection_stage = 0 if stage_advection_mode == "original" else stage
        f_explicit[stage] = compute_advection_rhs(
            f_stage[advection_stage],
            grid,
            boundary=boundary,
        )
        f_implicit[stage] = maxwellian[stage] - f_stage[stage]

    return f + dt * _stage_sum(butcher_table.be, f_explicit) + (
        dt / epsi
    ) * _stage_sum(butcher_table.bi, f_implicit)


def _store_trajectory_state(
    f: np.ndarray,
    v_centers: np.ndarray,
    index: int,
    f_history: np.ndarray,
    U_history: np.ndarray,
    q_history: np.ndarray,
) -> None:
    observables = compute_macroscopic_observables(f, v_centers)
    f_history[index] = f
    U_history[index] = observables.U
    q_history[index] = observables.q


def _validate_distribution_shape(f: np.ndarray, grid: Any) -> None:
    nx = int(_grid_value(grid, "Nx"))
    nv = int(_grid_value(grid, "Nv"))
    if f.shape != (nv, nx):
        raise ValueError(
            "Expected distribution shape (Nv, Nx) matching grid; "
            f"got f={f.shape}, Nv={nv}, Nx={nx}."
        )


def _stage_sum(coefficients: np.ndarray, values: np.ndarray) -> np.ndarray:
    total = np.zeros_like(values[0])
    for coefficient, value in zip(coefficients, values):
        total = total + coefficient * value
    return total


def _grid_value(grid: Any, field: str) -> Any:
    if isinstance(grid, dict):
        return grid[field]
    return getattr(grid, field)


def _compute_equilibrium(f, v_centers, equilibrium_mode, options, diagnostic_callback):
    return equilibrium_from_distribution(
        f, v_centers, equilibrium_mode=equilibrium_mode,
        discrete_equilibrium_options=options,
        equilibrium_diagnostic_callback=diagnostic_callback)
