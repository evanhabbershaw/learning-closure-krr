"""Kinetic reference IMEX transport stages."""
import numpy as np
from .kinetic_closure.equilibrium import equilibrium_from_distribution
from .kinetic_closure.sbgk_advection import compute_advection_fluxes, compute_advection_rhs

def compute_step_stage_data(f, grid, butcher, epsi: float, dt: float, *,
                            equilibrium_mode="continuous", discrete_equilibrium_options=None) -> dict[str, np.ndarray]:
    nx = grid.Nx
    nv = grid.Nv
    num_stages = butcher.num_stages
    f_star = np.zeros((num_stages, nv, nx))
    f_stage = np.zeros((num_stages, nv, nx))
    f_explicit = np.zeros((num_stages, nv, nx))
    f_implicit = np.zeros((num_stages, nv, nx))
    maxwellian = np.zeros((num_stages, nv, nx))
    fluxes = np.zeros((num_stages, nv, nx))

    f_star[0] = f
    maxwellian[0] = equilibrium_from_distribution(f_star[0], grid.vC, equilibrium_mode=equilibrium_mode,
        discrete_equilibrium_options=discrete_equilibrium_options)
    f_stage[0] = (epsi * f_star[0] + dt * butcher.Ai[0, 0] * maxwellian[0]) / (
        epsi + dt * butcher.Ai[0, 0]
    )
    fluxes[0] = compute_advection_fluxes(f_stage[0], grid, boundary="periodic")
    f_explicit[0] = compute_advection_rhs(f_stage[0], grid, boundary="periodic")
    f_implicit[0] = maxwellian[0] - f_stage[0]

    for i_stage in range(1, num_stages):
        explicit_sum = stage_sum_distributions(butcher.Ae[i_stage], f_explicit)
        implicit_sum = stage_sum_distributions(butcher.Ai[i_stage], f_implicit)
        f_star[i_stage] = f + dt * explicit_sum + (dt / epsi) * implicit_sum
        maxwellian[i_stage] = equilibrium_from_distribution(f_star[i_stage], grid.vC, equilibrium_mode=equilibrium_mode,
            discrete_equilibrium_options=discrete_equilibrium_options)
        f_stage[i_stage] = (
            epsi * f_star[i_stage] + dt * butcher.Ai[i_stage, i_stage] * maxwellian[i_stage]
        ) / (epsi + dt * butcher.Ai[i_stage, i_stage])
        fluxes[i_stage] = compute_advection_fluxes(f_stage[i_stage], grid, boundary="periodic")
        f_explicit[i_stage] = compute_advection_rhs(f_stage[i_stage], grid, boundary="periodic")
        f_implicit[i_stage] = maxwellian[i_stage] - f_stage[i_stage]

    return {
        "f_star": f_star,
        "f_stage": f_stage,
        "f_explicit": f_explicit,
        "f_implicit": f_implicit,
        "maxwellian": maxwellian,
        "fluxes": fluxes,
    }


def distribution_moments(values: np.ndarray, v_centers: np.ndarray) -> np.ndarray:
    h_v = float(v_centers[1] - v_centers[0])
    return np.vstack(
        [
            h_v * np.sum(values, axis=0),
            h_v * np.sum(v_centers[:, np.newaxis] * values, axis=0),
            0.5 * h_v * np.sum((v_centers[:, np.newaxis] ** 2) * values, axis=0),
        ]
    )


def stage_sum_distributions(coefficients: np.ndarray, values: np.ndarray) -> np.ndarray:
    total = np.zeros_like(values[0])
    for coefficient, value in zip(coefficients, values):
        total = total + coefficient * value
    return total
