"""Conservative discrete Maxwellians for the finite SBGK velocity grid.

The repository uses midpoint velocity cells with the uniform rectangle rule
and collision invariants ``(1, v, 0.5*v**2)``.  This module constructs the
exponential discrete equilibrium whose moments match a supplied conservative
state in exactly that convention.

For numerical stability the nonlinear solve eliminates ``alpha_0`` using the
mass constraint.  The remaining two variables minimize the strictly convex
log-partition dual with damped Newton steps.  The returned distribution is
still exactly ``exp(alpha dot m)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DiscreteMaxwellianOptions:
    """Controls for the two-dimensional damped Newton solve."""

    absolute_tolerance: float = 2.0e-13
    relative_tolerance: float = 2.0e-12
    maximum_iterations: int = 50
    maximum_backtracks: int = 30
    armijo_parameter: float = 1.0e-4
    minimum_step: float = 2.0**-30
    condition_limit: float = 1.0e16


@dataclass(frozen=True)
class DiscreteMaxwellianDiagnostics:
    """Convergence information for one spatial state."""

    converged: bool
    iterations: int
    residual_norm: float
    relative_residual_norm: float
    minimum_value: float
    final_condition_number: float
    maximum_condition_number: float
    backtracks: int
    alpha: np.ndarray
    message: str


@dataclass(frozen=True)
class BatchedDiscreteMaxwellianDiagnostics:
    """Array-valued convergence information for a batch of spatial states."""

    converged: np.ndarray
    iterations: np.ndarray
    residual_norm: np.ndarray
    relative_residual_norm: np.ndarray
    minimum_value: np.ndarray
    final_condition_number: np.ndarray
    maximum_condition_number: np.ndarray
    backtracks: np.ndarray
    alpha: np.ndarray

    def cell_diagnostics(self) -> tuple[DiscreteMaxwellianDiagnostics, ...]:
        """Materialize legacy per-cell records only when explicitly requested."""
        return tuple(
            DiscreteMaxwellianDiagnostics(
                converged=bool(self.converged[index]),
                iterations=int(self.iterations[index]),
                residual_norm=float(self.residual_norm[index]),
                relative_residual_norm=float(
                    self.relative_residual_norm[index]
                ),
                minimum_value=float(self.minimum_value[index]),
                final_condition_number=float(
                    self.final_condition_number[index]
                ),
                maximum_condition_number=float(
                    self.maximum_condition_number[index]
                ),
                backtracks=int(self.backtracks[index]),
                alpha=self.alpha[:, index].copy(),
                message="Converged to the requested discrete moment tolerance.",
            )
            for index in range(self.iterations.size)
        )


@dataclass(frozen=True)
class _VelocityGridData:
    velocity: np.ndarray
    quadrature: np.ndarray
    log_weights: np.ndarray
    basis: np.ndarray
    features: np.ndarray
    feature_products: np.ndarray


class DiscreteMaxwellianError(RuntimeError):
    """Raised when a target is invalid, unrealizable, or Newton fails."""

    def __init__(
        self,
        message: str,
        diagnostics: DiscreteMaxwellianDiagnostics | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def velocity_quadrature_weights(v_centers: Any) -> np.ndarray:
    """Return the repository's uniform midpoint-cell velocity weights."""
    velocity = _validate_velocity_grid(v_centers)
    spacing = np.diff(velocity)
    h_v = float(spacing[0])
    if not np.allclose(spacing, h_v, rtol=1.0e-13, atol=1.0e-14):
        raise ValueError("The SBGK moment convention requires a uniform velocity grid.")
    return np.full(velocity.shape, h_v)


def conserved_moment_basis(v_centers: Any) -> np.ndarray:
    """Return rows ``(1, v, 0.5*v**2)`` on the discrete velocity grid."""
    velocity = _validate_velocity_grid(v_centers)
    return np.vstack((np.ones_like(velocity), velocity, 0.5 * velocity**2))


def compute_discrete_conserved_moments(
    values: Any,
    v_centers: Any,
    weights: Any | None = None,
) -> np.ndarray:
    """Compute finite-grid conservative moments for one or many columns."""
    velocity = _validate_velocity_grid(v_centers)
    quadrature = _validate_weights(weights, velocity)
    array = np.asarray(values, dtype=float)
    if array.ndim not in (1, 2) or array.shape[0] != velocity.size:
        raise ValueError(
            "values must have shape (Nv,) or (Nv, Nx) matching v_centers; "
            f"got {array.shape}."
        )
    return conserved_moment_basis(velocity) @ (quadrature[:, None] * array) \
        if array.ndim == 2 else conserved_moment_basis(velocity) @ (quadrature * array)


def discrete_moment_residual_and_jacobian(
    alpha: Any,
    target: Any,
    v_centers: Any,
    weights: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the full three-moment residual and its analytic Jacobian.

    This direct evaluator is intended for verification and diagnostics.  The
    production nonlinear solve below uses an equivalent normalized
    log-partition formulation to avoid exponential overflow.
    """
    velocity = _validate_velocity_grid(v_centers)
    quadrature = _validate_weights(weights, velocity)
    alpha_array = np.asarray(alpha, dtype=float).reshape(-1)
    target_array = np.asarray(target, dtype=float).reshape(-1)
    if alpha_array.shape != (3,) or target_array.shape != (3,):
        raise ValueError("alpha and target must both have shape (3,).")
    if not np.all(np.isfinite(alpha_array)) or not np.all(np.isfinite(target_array)):
        raise ValueError("alpha and target must contain only finite values.")

    basis = conserved_moment_basis(velocity)
    exponent = alpha_array @ basis
    overflow_limit = np.log(np.finfo(float).max) - 2.0
    if float(np.max(exponent)) > overflow_limit:
        raise FloatingPointError("alpha produces exponential overflow.")
    equilibrium = np.exp(exponent)
    weighted = quadrature * equilibrium
    residual = basis @ weighted - target_array
    jacobian = (basis * weighted[None, :]) @ basis.T
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(jacobian)):
        raise FloatingPointError("Non-finite discrete moment residual or Jacobian.")
    return residual, jacobian


def solve_discrete_maxwellian(
    target: Any,
    v_centers: Any,
    weights: Any | None = None,
    *,
    initial_alpha: Any | None = None,
    options: DiscreteMaxwellianOptions | None = None,
) -> tuple[np.ndarray, DiscreteMaxwellianDiagnostics]:
    """Reference single-cell solve of the conservative moment equations.

    ``alpha_0`` is eliminated analytically.  If ``z_j=(v_j, 0.5*v_j**2)``,
    Newton solves ``E_beta[z] = (momentum/rho, E/rho)`` with Jacobian
    ``Cov_beta(z)``.  Log-sum-exp evaluation prevents overflow and an Armijo
    line search globalizes each Newton step.  Failure is always explicit.
    """
    opts = options or DiscreteMaxwellianOptions()
    _validate_options(opts)
    velocity = _validate_velocity_grid(v_centers)
    quadrature = _validate_weights(weights, velocity)
    target_array = _validate_target(target)
    _check_strict_realizability(target_array, velocity)

    rho, momentum, energy = target_array
    target_normalized = np.array((momentum / rho, energy / rho))
    features = np.vstack((velocity, 0.5 * velocity**2))
    log_weights = np.log(quadrature)

    if initial_alpha is None:
        mean_velocity = momentum / rho
        theta = 2.0 * energy / rho - mean_velocity**2
        beta = np.array((mean_velocity / theta, -1.0 / theta))
    else:
        alpha_guess = np.asarray(initial_alpha, dtype=float).reshape(-1)
        if alpha_guess.shape != (3,) or not np.all(np.isfinite(alpha_guess)):
            raise ValueError("initial_alpha must be a finite array with shape (3,).")
        beta = alpha_guess[1:].copy()

    maximum_condition = 0.0
    total_backtracks = 0
    final_condition = np.inf
    last_alpha = np.full(3, np.nan)
    last_equilibrium = np.full(velocity.shape, np.nan)
    last_residual_norm = np.inf

    for iteration in range(opts.maximum_iterations + 1):
        log_partition, mean_features, covariance, dual = _dual_state(
            beta, features, log_weights, target_normalized
        )
        alpha = np.array((np.log(rho) - log_partition, beta[0], beta[1]))
        exponent = alpha @ conserved_moment_basis(velocity)
        equilibrium = np.exp(exponent)
        moment_residual = (
            compute_discrete_conserved_moments(equilibrium, velocity, quadrature)
            - target_array
        )
        residual_norm = float(np.linalg.norm(moment_residual, ord=np.inf))
        target_scale = max(float(np.linalg.norm(target_array, ord=np.inf)), 1.0)
        relative_residual = residual_norm / target_scale
        tolerance = opts.absolute_tolerance + opts.relative_tolerance * target_scale
        final_condition = float(np.linalg.cond(covariance))
        maximum_condition = max(maximum_condition, final_condition)
        last_alpha = alpha
        last_equilibrium = equilibrium
        last_residual_norm = residual_norm

        if not (
            np.all(np.isfinite(equilibrium))
            and np.all(np.isfinite(moment_residual))
            and np.isfinite(final_condition)
        ):
            _raise_failure(
                "Non-finite value encountered during the discrete Maxwellian solve.",
                iteration,
                residual_norm,
                relative_residual,
                equilibrium,
                final_condition,
                maximum_condition,
                total_backtracks,
                alpha,
            )
        if np.any(equilibrium <= 0.0):
            _raise_failure(
                "Discrete equilibrium underflowed to a nonpositive floating-point value.",
                iteration,
                residual_norm,
                relative_residual,
                equilibrium,
                final_condition,
                maximum_condition,
                total_backtracks,
                alpha,
            )
        if residual_norm <= tolerance:
            diagnostics = DiscreteMaxwellianDiagnostics(
                converged=True,
                iterations=iteration,
                residual_norm=residual_norm,
                relative_residual_norm=relative_residual,
                minimum_value=float(np.min(equilibrium)),
                final_condition_number=final_condition,
                maximum_condition_number=maximum_condition,
                backtracks=total_backtracks,
                alpha=alpha.copy(),
                message="Converged to the requested discrete moment tolerance.",
            )
            return equilibrium, diagnostics
        if iteration == opts.maximum_iterations:
            break
        if final_condition > opts.condition_limit:
            _raise_failure(
                "Discrete dual Hessian is too ill-conditioned for a reliable Newton step.",
                iteration,
                residual_norm,
                relative_residual,
                equilibrium,
                final_condition,
                maximum_condition,
                total_backtracks,
                alpha,
            )

        reduced_residual = mean_features - target_normalized
        reduced_residual_norm = float(np.linalg.norm(reduced_residual, ord=2))
        try:
            newton_step = np.linalg.solve(covariance, -reduced_residual)
        except np.linalg.LinAlgError as exc:
            _raise_failure(
                f"Discrete dual Hessian solve failed: {exc}",
                iteration,
                residual_norm,
                relative_residual,
                equilibrium,
                final_condition,
                maximum_condition,
                total_backtracks,
                alpha,
            )
        directional_derivative = float(reduced_residual @ newton_step)
        if not np.isfinite(directional_derivative) or directional_derivative >= 0.0:
            _raise_failure(
                "Newton direction is not a finite descent direction.",
                iteration,
                residual_norm,
                relative_residual,
                equilibrium,
                final_condition,
                maximum_condition,
                total_backtracks,
                alpha,
            )

        step_length = 1.0
        accepted = False
        for backtrack in range(opts.maximum_backtracks + 1):
            candidate = beta + step_length * newton_step
            candidate_state = _dual_state(
                candidate, features, log_weights, target_normalized
            )
            candidate_dual = candidate_state[3]
            candidate_residual_norm = float(
                np.linalg.norm(
                    candidate_state[1] - target_normalized,
                    ord=2,
                )
            )
            armijo_decrease = candidate_dual <= (
                dual
                + opts.armijo_parameter
                * step_length
                * directional_derivative
            )
            # Very near the minimizer, the predicted dual decrease is below
            # floating-point resolution even though one final Newton step can
            # materially reduce the moment residual.  A residual-contraction
            # acceptance test avoids false line-search stagnation there.
            residual_contraction = candidate_residual_norm <= (
                1.0 - opts.armijo_parameter * step_length
            ) * reduced_residual_norm
            if np.isfinite(candidate_dual) and (
                armijo_decrease or residual_contraction
            ):
                beta = candidate
                total_backtracks += backtrack
                accepted = True
                break
            step_length *= 0.5
            if step_length < opts.minimum_step:
                break
        if not accepted:
            _raise_failure(
                "Damped Newton line search failed to decrease the convex dual.",
                iteration,
                residual_norm,
                relative_residual,
                equilibrium,
                final_condition,
                maximum_condition,
                total_backtracks,
                alpha,
            )

    scale = max(float(np.linalg.norm(target_array, ord=np.inf)), 1.0)
    _raise_failure(
        f"Discrete Maxwellian did not converge in {opts.maximum_iterations} iterations.",
        opts.maximum_iterations,
        last_residual_norm,
        last_residual_norm / scale,
        last_equilibrium,
        final_condition,
        maximum_condition,
        total_backtracks,
        last_alpha,
    )
    raise AssertionError("unreachable")


def solve_discrete_maxwellian_batch(
    targets: Any,
    v_centers: Any,
    weights: Any | None = None,
    *,
    initial_alpha: Any | None = None,
    options: DiscreteMaxwellianOptions | None = None,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, BatchedDiscreteMaxwellianDiagnostics]:
    """Solve the same discrete equilibrium problem for all spatial cells.

    The reduced convex-dual Newton method is identical to the reference
    single-cell formulation, but active cells are advanced together and the
    symmetric 2-by-2 covariance systems are solved explicitly in arrays.
    """
    opts = options or DiscreteMaxwellianOptions()
    _validate_options(opts)
    grid_data = _prepare_velocity_grid_data(v_centers, weights)
    target_array = _validate_targets(targets)
    _check_strict_realizability_batch(target_array, grid_data.velocity)

    rho = target_array[0]
    target_normalized = target_array[1:] / rho[None, :]
    cell_count = target_array.shape[1]
    if initial_alpha is None:
        mean_velocity = target_normalized[0]
        theta = 2.0 * target_normalized[1] - mean_velocity**2
        beta = np.vstack((mean_velocity / theta, -1.0 / theta))
    else:
        alpha_guess = np.asarray(initial_alpha, dtype=float)
        if alpha_guess.shape != (3, cell_count) or not np.all(
            np.isfinite(alpha_guess)
        ):
            raise ValueError(
                "initial_alpha must be finite with shape (3, number_of_cells)."
            )
        beta = alpha_guess[1:].copy()

    equilibrium = np.empty((grid_data.velocity.size, cell_count))
    alpha = np.full((3, cell_count), np.nan)
    iterations = np.full(cell_count, -1, dtype=int)
    residual_norm = np.full(cell_count, np.inf)
    relative_residual = np.full(cell_count, np.inf)
    minimum_value = np.full(cell_count, np.nan)
    final_condition = np.full(cell_count, np.inf)
    maximum_condition = np.zeros(cell_count)
    backtracks = np.zeros(cell_count, dtype=int)
    target_scale = np.maximum(np.max(np.abs(target_array), axis=0), 1.0)
    tolerance = opts.absolute_tolerance + opts.relative_tolerance * target_scale
    active = np.ones(cell_count, dtype=bool)

    for iteration in range(opts.maximum_iterations + 1):
        active_indices = np.flatnonzero(active)
        state = _dual_state_batch(
            beta[:, active_indices],
            grid_data,
            target_normalized[:, active_indices],
        )
        log_partition, mean_features, covariance, _ = state
        alpha_active = np.vstack(
            (
                np.log(rho[active_indices]) - log_partition,
                beta[:, active_indices],
            )
        )
        exponent = alpha_active.T @ grid_data.basis
        equilibrium_active = np.exp(exponent).T
        moment_residual = (
            grid_data.basis
            @ (grid_data.quadrature[:, None] * equilibrium_active)
            - target_array[:, active_indices]
        )
        residual_active = np.max(np.abs(moment_residual), axis=0)
        relative_active = residual_active / target_scale[active_indices]
        condition_active = _symmetric_2x2_condition_numbers(covariance)
        minimum_active = np.min(equilibrium_active, axis=0)

        equilibrium[:, active_indices] = equilibrium_active
        alpha[:, active_indices] = alpha_active
        residual_norm[active_indices] = residual_active
        relative_residual[active_indices] = relative_active
        minimum_value[active_indices] = minimum_active
        final_condition[active_indices] = condition_active
        maximum_condition[active_indices] = np.maximum(
            maximum_condition[active_indices], condition_active
        )

        invalid = ~(
            np.all(np.isfinite(equilibrium_active), axis=0)
            & np.all(np.isfinite(moment_residual), axis=0)
            & np.isfinite(condition_active)
        )
        if np.any(invalid):
            local = int(np.flatnonzero(invalid)[0])
            _raise_batch_failure(
                "Non-finite value encountered during the discrete Maxwellian solve.",
                int(active_indices[local]),
                iteration,
                residual_norm,
                relative_residual,
                equilibrium,
                final_condition,
                maximum_condition,
                backtracks,
                alpha,
            )
        underflow = minimum_active <= 0.0
        if np.any(underflow):
            local = int(np.flatnonzero(underflow)[0])
            _raise_batch_failure(
                "Discrete equilibrium underflowed to a nonpositive floating-point value.",
                int(active_indices[local]),
                iteration,
                residual_norm,
                relative_residual,
                equilibrium,
                final_condition,
                maximum_condition,
                backtracks,
                alpha,
            )

        converged_local = residual_active <= tolerance[active_indices]
        converged_indices = active_indices[converged_local]
        iterations[converged_indices] = iteration
        active[converged_indices] = False
        if not np.any(active):
            diagnostics = BatchedDiscreteMaxwellianDiagnostics(
                converged=np.ones(cell_count, dtype=bool),
                iterations=iterations,
                residual_norm=residual_norm,
                relative_residual_norm=relative_residual,
                minimum_value=minimum_value,
                final_condition_number=final_condition,
                maximum_condition_number=maximum_condition,
                backtracks=backtracks,
                alpha=alpha,
            )
            return (equilibrium, diagnostics) if return_diagnostics else equilibrium
        if iteration == opts.maximum_iterations:
            break

        remaining_local = ~converged_local
        remaining_indices = active_indices[remaining_local]
        covariance_remaining = covariance[:, remaining_local]
        condition_remaining = condition_active[remaining_local]
        ill_conditioned = condition_remaining > opts.condition_limit
        if np.any(ill_conditioned):
            local = int(np.flatnonzero(ill_conditioned)[0])
            _raise_batch_failure(
                "Discrete dual Hessian is too ill-conditioned for a reliable Newton step.",
                int(remaining_indices[local]),
                iteration,
                residual_norm,
                relative_residual,
                equilibrium,
                final_condition,
                maximum_condition,
                backtracks,
                alpha,
            )

        reduced_residual = (
            mean_features[:, remaining_local]
            - target_normalized[:, remaining_indices]
        )
        newton_step = _solve_symmetric_2x2_batch(
            covariance_remaining, -reduced_residual
        )
        directional_derivative = np.sum(reduced_residual * newton_step, axis=0)
        bad_direction = ~np.isfinite(directional_derivative) | (
            directional_derivative >= 0.0
        )
        if np.any(bad_direction):
            local = int(np.flatnonzero(bad_direction)[0])
            _raise_batch_failure(
                "Newton direction is not a finite descent direction.",
                int(remaining_indices[local]),
                iteration,
                residual_norm,
                relative_residual,
                equilibrium,
                final_condition,
                maximum_condition,
                backtracks,
                alpha,
            )

        current_dual = state[3][remaining_local]
        reduced_norm = np.linalg.norm(reduced_residual, axis=0)
        pending = np.ones(remaining_indices.size, dtype=bool)
        step_length = np.ones(remaining_indices.size)
        for backtrack in range(opts.maximum_backtracks + 1):
            pending_local = np.flatnonzero(pending)
            candidate = (
                beta[:, remaining_indices[pending_local]]
                + step_length[pending_local][None, :]
                * newton_step[:, pending_local]
            )
            candidate_state = _dual_state_batch(
                candidate,
                grid_data,
                target_normalized[:, remaining_indices[pending_local]],
            )
            candidate_dual = candidate_state[3]
            candidate_norm = np.linalg.norm(
                candidate_state[1]
                - target_normalized[:, remaining_indices[pending_local]],
                axis=0,
            )
            lengths = step_length[pending_local]
            accepted = np.isfinite(candidate_dual) & (
                (
                    candidate_dual
                    <= current_dual[pending_local]
                    + opts.armijo_parameter
                    * lengths
                    * directional_derivative[pending_local]
                )
                | (
                    candidate_norm
                    <= (1.0 - opts.armijo_parameter * lengths)
                    * reduced_norm[pending_local]
                )
            )
            accepted_local = pending_local[accepted]
            beta[:, remaining_indices[accepted_local]] = candidate[:, accepted]
            backtracks[remaining_indices[accepted_local]] += backtrack
            pending[accepted_local] = False
            if not np.any(pending):
                break
            step_length[pending] *= 0.5
            if np.any(pending & (step_length < opts.minimum_step)):
                break
        if np.any(pending):
            local = int(np.flatnonzero(pending)[0])
            _raise_batch_failure(
                "Damped Newton line search failed to decrease the convex dual.",
                int(remaining_indices[local]),
                iteration,
                residual_norm,
                relative_residual,
                equilibrium,
                final_condition,
                maximum_condition,
                backtracks,
                alpha,
            )

    failed_index = int(np.flatnonzero(active)[0])
    _raise_batch_failure(
        f"Discrete Maxwellian did not converge in {opts.maximum_iterations} iterations.",
        failed_index,
        opts.maximum_iterations,
        residual_norm,
        relative_residual,
        equilibrium,
        final_condition,
        maximum_condition,
        backtracks,
        alpha,
    )
    raise AssertionError("unreachable")


def compute_discrete_maxwellian(
    moments: Any,
    v_centers: Any,
    *,
    weights: Any | None = None,
    options: DiscreteMaxwellianOptions | None = None,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, tuple[DiscreteMaxwellianDiagnostics, ...]]:
    """Construct columnwise equilibria from ``(rho, u, theta)`` primitives."""
    primitive = np.asarray(moments, dtype=float)
    velocity = _validate_velocity_grid(v_centers)
    if primitive.ndim != 2 or primitive.shape[0] != 3:
        raise ValueError(f"Expected moments shape (3, Nx); got {primitive.shape}.")
    if not np.all(np.isfinite(primitive)):
        raise ValueError("moments must contain only finite values.")
    rho, mean_velocity, theta = primitive
    if np.any(rho <= 0.0) or np.any(theta <= 0.0):
        raise DiscreteMaxwellianError("rho and theta must be strictly positive.")

    targets = np.vstack(
        (rho, rho * mean_velocity, 0.5 * rho * (mean_velocity**2 + theta))
    )
    solved = solve_discrete_maxwellian_batch(
        targets,
        velocity,
        weights,
        options=options,
        return_diagnostics=return_diagnostics,
    )
    if return_diagnostics:
        equilibrium, diagnostics = solved
        return equilibrium, diagnostics.cell_diagnostics()
    return solved


def compute_discrete_maxwellian_reference(
    moments: Any,
    v_centers: Any,
    *,
    weights: Any | None = None,
    options: DiscreteMaxwellianOptions | None = None,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, tuple[DiscreteMaxwellianDiagnostics, ...]]:
    """Original scalar spatial loop retained for verification and timing."""
    primitive = np.asarray(moments, dtype=float)
    velocity = _validate_velocity_grid(v_centers)
    if primitive.ndim != 2 or primitive.shape[0] != 3:
        raise ValueError(f"Expected moments shape (3, Nx); got {primitive.shape}.")
    if not np.all(np.isfinite(primitive)):
        raise ValueError("moments must contain only finite values.")
    rho, mean_velocity, theta = primitive
    if np.any(rho <= 0.0) or np.any(theta <= 0.0):
        raise DiscreteMaxwellianError("rho and theta must be strictly positive.")
    targets = np.vstack(
        (rho, rho * mean_velocity, 0.5 * rho * (mean_velocity**2 + theta))
    )
    equilibrium = np.empty((velocity.size, primitive.shape[1]))
    diagnostics: list[DiscreteMaxwellianDiagnostics] = []
    for column in range(primitive.shape[1]):
        equilibrium[:, column], diagnostic = solve_discrete_maxwellian(
            targets[:, column], velocity, weights, options=options
        )
        if return_diagnostics:
            diagnostics.append(diagnostic)
    if return_diagnostics:
        return equilibrium, tuple(diagnostics)
    return equilibrium


def _prepare_velocity_grid_data(
    v_centers: Any, weights: Any | None
) -> _VelocityGridData:
    velocity = _validate_velocity_grid(v_centers)
    quadrature = _validate_weights(weights, velocity)
    return _cached_velocity_grid_data(
        tuple(velocity.tolist()), tuple(quadrature.tolist())
    )


@lru_cache(maxsize=16)
def _cached_velocity_grid_data(
    velocity_key: tuple[float, ...], quadrature_key: tuple[float, ...]
) -> _VelocityGridData:
    velocity = np.asarray(velocity_key, dtype=float)
    quadrature = np.asarray(quadrature_key, dtype=float)
    basis = np.vstack((np.ones_like(velocity), velocity, 0.5 * velocity**2))
    features = basis[1:]
    feature_products = np.vstack(
        (features[0] ** 2, features[0] * features[1], features[1] ** 2)
    )
    for array in (velocity, quadrature, basis, features, feature_products):
        array.setflags(write=False)
    log_weights = np.log(quadrature)
    log_weights.setflags(write=False)
    return _VelocityGridData(
        velocity=velocity,
        quadrature=quadrature,
        log_weights=log_weights,
        basis=basis,
        features=features,
        feature_products=feature_products,
    )


def _dual_state_batch(
    beta: np.ndarray,
    grid_data: _VelocityGridData,
    target_normalized: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    log_terms = beta.T @ grid_data.features + grid_data.log_weights[None, :]
    shift = np.max(log_terms, axis=1)
    scaled = np.exp(log_terms - shift[:, None])
    scaled_sum = np.sum(scaled, axis=1)
    probabilities = scaled / scaled_sum[:, None]
    log_partition = shift + np.log(scaled_sum)
    mean_features = (probabilities @ grid_data.features.T).T
    raw_products = (probabilities @ grid_data.feature_products.T).T
    covariance = np.empty((3, beta.shape[1]))
    covariance[0] = raw_products[0] - mean_features[0] ** 2
    covariance[1] = raw_products[1] - mean_features[0] * mean_features[1]
    covariance[2] = raw_products[2] - mean_features[1] ** 2
    dual = log_partition - np.sum(beta * target_normalized, axis=0)
    return log_partition, mean_features, covariance, dual


def _symmetric_2x2_condition_numbers(covariance: np.ndarray) -> np.ndarray:
    a, b, c = covariance
    discriminant = np.sqrt(np.maximum((a - c) ** 2 + 4.0 * b**2, 0.0))
    largest = 0.5 * (a + c + discriminant)
    smallest = 0.5 * (a + c - discriminant)
    return np.divide(
        largest,
        smallest,
        out=np.full_like(largest, np.inf),
        where=smallest > 0.0,
    )


def _solve_symmetric_2x2_batch(
    covariance: np.ndarray, right_hand_side: np.ndarray
) -> np.ndarray:
    a, b, c = covariance
    determinant = a * c - b**2
    solution = np.empty_like(right_hand_side)
    solution[0] = (
        c * right_hand_side[0] - b * right_hand_side[1]
    ) / determinant
    solution[1] = (
        a * right_hand_side[1] - b * right_hand_side[0]
    ) / determinant
    return solution


def _dual_state(
    beta: np.ndarray,
    features: np.ndarray,
    log_weights: np.ndarray,
    target_normalized: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    log_terms = log_weights + beta @ features
    shift = float(np.max(log_terms))
    scaled = np.exp(log_terms - shift)
    scaled_sum = float(np.sum(scaled))
    probabilities = scaled / scaled_sum
    log_partition = shift + np.log(scaled_sum)
    mean_features = features @ probabilities
    centered = features - mean_features[:, None]
    covariance = (centered * probabilities[None, :]) @ centered.T
    dual = float(log_partition - beta @ target_normalized)
    return log_partition, mean_features, covariance, dual


def _check_strict_realizability(target: np.ndarray, velocity: np.ndarray) -> None:
    rho, momentum, energy = target
    mean_velocity = momentum / rho
    mean_energy = energy / rho
    if not velocity[0] < mean_velocity < velocity[-1]:
        raise DiscreteMaxwellianError(
            "Target mean velocity is outside the strict finite-grid realizability interval."
        )
    upper_index = int(np.searchsorted(velocity, mean_velocity, side="right"))
    lower_index = upper_index - 1
    left, right = velocity[lower_index], velocity[upper_index]
    fraction = (mean_velocity - left) / (right - left)
    minimum_energy = 0.5 * ((1.0 - fraction) * left**2 + fraction * right**2)
    endpoint_fraction = (mean_velocity - velocity[0]) / (velocity[-1] - velocity[0])
    maximum_energy = 0.5 * (
        (1.0 - endpoint_fraction) * velocity[0] ** 2
        + endpoint_fraction * velocity[-1] ** 2
    )
    margin = 64.0 * np.finfo(float).eps * max(1.0, abs(maximum_energy))
    if not minimum_energy + margin < mean_energy < maximum_energy - margin:
        raise DiscreteMaxwellianError(
            "Target lies on or outside the strict discrete mass/momentum/energy "
            "realizability region."
        )


def _check_strict_realizability_batch(
    targets: np.ndarray, velocity: np.ndarray
) -> None:
    rho, momentum, energy = targets
    mean_velocity = momentum / rho
    mean_energy = energy / rho
    velocity_valid = (velocity[0] < mean_velocity) & (
        mean_velocity < velocity[-1]
    )
    if not np.all(velocity_valid):
        index = int(np.flatnonzero(~velocity_valid)[0])
        raise DiscreteMaxwellianError(
            "Target mean velocity is outside the strict finite-grid realizability "
            f"interval (cell {index})."
        )
    upper_index = np.searchsorted(velocity, mean_velocity, side="right")
    lower_index = upper_index - 1
    left = velocity[lower_index]
    right = velocity[upper_index]
    fraction = (mean_velocity - left) / (right - left)
    minimum_energy = 0.5 * (
        (1.0 - fraction) * left**2 + fraction * right**2
    )
    endpoint_fraction = (mean_velocity - velocity[0]) / (
        velocity[-1] - velocity[0]
    )
    maximum_energy = 0.5 * (
        (1.0 - endpoint_fraction) * velocity[0] ** 2
        + endpoint_fraction * velocity[-1] ** 2
    )
    margin = 64.0 * np.finfo(float).eps * np.maximum(
        1.0, np.abs(maximum_energy)
    )
    realizable = (minimum_energy + margin < mean_energy) & (
        mean_energy < maximum_energy - margin
    )
    if not np.all(realizable):
        index = int(np.flatnonzero(~realizable)[0])
        raise DiscreteMaxwellianError(
            "Target lies on or outside the strict discrete mass/momentum/energy "
            f"realizability region (cell {index})."
        )


def _validate_target(target: Any) -> np.ndarray:
    array = np.asarray(target, dtype=float).reshape(-1)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise DiscreteMaxwellianError("target must be a finite conservative vector of shape (3,).")
    if array[0] <= 0.0:
        raise DiscreteMaxwellianError("Target mass must be strictly positive.")
    return array


def _validate_targets(targets: Any) -> np.ndarray:
    array = np.asarray(targets, dtype=float)
    if (
        array.ndim != 2
        or array.shape[0] != 3
        or array.shape[1] == 0
        or not np.all(np.isfinite(array))
    ):
        raise DiscreteMaxwellianError(
            "targets must be a finite conservative array of shape (3, number_of_cells)."
        )
    if np.any(array[0] <= 0.0):
        raise DiscreteMaxwellianError("Target mass must be strictly positive.")
    return array


def _validate_velocity_grid(v_centers: Any) -> np.ndarray:
    velocity = np.asarray(v_centers, dtype=float).reshape(-1)
    if velocity.size < 3 or not np.all(np.isfinite(velocity)):
        raise ValueError("v_centers must contain at least three finite values.")
    if np.any(np.diff(velocity) <= 0.0):
        raise ValueError("v_centers must be strictly increasing.")
    return velocity


def _validate_weights(weights: Any | None, velocity: np.ndarray) -> np.ndarray:
    quadrature = (
        velocity_quadrature_weights(velocity)
        if weights is None
        else np.asarray(weights, dtype=float).reshape(-1)
    )
    if quadrature.shape != velocity.shape:
        raise ValueError(f"weights must have shape {velocity.shape}; got {quadrature.shape}.")
    if not np.all(np.isfinite(quadrature)) or np.any(quadrature <= 0.0):
        raise ValueError("weights must be finite and strictly positive.")
    return quadrature


def _validate_options(options: DiscreteMaxwellianOptions) -> None:
    # Zero tolerances and zero update/backtrack budgets deliberately remain valid
    # for exact-match requests and explicit failure tests.
    for name in ("absolute_tolerance", "relative_tolerance", "armijo_parameter",
                 "minimum_step", "condition_limit"):
        value = getattr(options, name)
        if not np.isscalar(value) or not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if options.absolute_tolerance < 0.0 or options.relative_tolerance < 0.0:
        raise ValueError("Moment tolerances must be nonnegative.")
    for name in ("maximum_iterations", "maximum_backtracks"):
        value = getattr(options, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
    if not 0.0 < options.armijo_parameter < 1.0:
        raise ValueError("armijo_parameter must lie strictly between zero and one.")
    if not 0.0 < options.minimum_step <= 1.0 or options.condition_limit <= 1.0:
        raise ValueError("minimum_step must lie in (0, 1] and condition_limit must exceed one.")


def _raise_failure(
    message: str,
    iterations: int,
    residual_norm: float,
    relative_residual: float,
    equilibrium: np.ndarray,
    final_condition: float,
    maximum_condition: float,
    backtracks: int,
    alpha: np.ndarray,
) -> None:
    finite_equilibrium = equilibrium[np.isfinite(equilibrium)]
    diagnostic = DiscreteMaxwellianDiagnostics(
        converged=False,
        iterations=iterations,
        residual_norm=float(residual_norm),
        relative_residual_norm=float(relative_residual),
        minimum_value=(
            float(np.min(finite_equilibrium))
            if finite_equilibrium.size
            else np.nan
        ),
        final_condition_number=float(final_condition),
        maximum_condition_number=float(maximum_condition),
        backtracks=backtracks,
        alpha=np.asarray(alpha, dtype=float).copy(),
        message=message,
    )
    raise DiscreteMaxwellianError(message, diagnostic)


def _raise_batch_failure(
    message: str,
    cell: int,
    iterations: int,
    residual_norm: np.ndarray,
    relative_residual: np.ndarray,
    equilibrium: np.ndarray,
    final_condition: np.ndarray,
    maximum_condition: np.ndarray,
    backtracks: np.ndarray,
    alpha: np.ndarray,
) -> None:
    message_with_cell = f"{message} (cell {cell})."
    _raise_failure(
        message_with_cell,
        iterations,
        float(residual_norm[cell]),
        float(relative_residual[cell]),
        equilibrium[:, cell],
        float(final_condition[cell]),
        float(maximum_condition[cell]),
        int(backtracks[cell]),
        alpha[:, cell],
    )
