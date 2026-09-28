"""Shared finite-grid equilibrium policy. Conservative failures never fall back.

Inputs have shape (3, Nx). ``continuous`` explicitly reproduces the sampled
Gaussian; ``discrete_mieussens`` matches all three discrete moments.
The caller must select the mode explicitly.
"""
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Literal

import numpy as np

from .discrete_maxwellian import (
    DiscreteMaxwellianOptions, _validate_options,
    compute_discrete_maxwellian, solve_discrete_maxwellian_batch,
    compute_discrete_conserved_moments,
)
from .sbgk_maxwellian import compute_maxwellian
from .sbgk_moments import compute_moments

EquilibriumMode = Literal["continuous", "discrete_mieussens"]


def validate_equilibrium_policy(equilibrium_mode, options=None):
    if equilibrium_mode not in ("continuous", "discrete_mieussens"):
        raise ValueError(f"Unknown equilibrium_mode: {equilibrium_mode!r}")
    if options is not None:
        _validate_options(options)


def equilibrium_from_primitives(primitives, velocity, *,
                                equilibrium_mode,
                                discrete_equilibrium_options=None,
                                equilibrium_diagnostic_callback=None):
    validate_equilibrium_policy(equilibrium_mode, discrete_equilibrium_options)
    if equilibrium_mode == "continuous":
        return compute_maxwellian(primitives, velocity)
    result = compute_discrete_maxwellian(
        primitives, velocity, options=discrete_equilibrium_options,
        return_diagnostics=equilibrium_diagnostic_callback is not None)
    if equilibrium_diagnostic_callback is None:
        return result
    values, diagnostics = result
    for diagnostic in diagnostics:
        equilibrium_diagnostic_callback(diagnostic)
    return values


def equilibrium_from_conserved(targets, velocity, *,
                               equilibrium_mode,
                               discrete_equilibrium_options=None,
                               equilibrium_diagnostic_callback=None):
    validate_equilibrium_policy(equilibrium_mode, discrete_equilibrium_options)
    targets = np.asarray(targets, dtype=float)
    if targets.ndim != 2 or targets.shape[0] != 3:
        raise ValueError("targets must have shape (3, Nx)")
    if equilibrium_mode == "continuous":
        rho, momentum, energy = targets
        u = momentum / rho
        return compute_maxwellian(np.vstack((rho, u, 2 * energy / rho - u**2)), velocity)
    result = solve_discrete_maxwellian_batch(
        targets, velocity, options=discrete_equilibrium_options,
        return_diagnostics=equilibrium_diagnostic_callback is not None)
    if equilibrium_diagnostic_callback is None:
        return result
    values, diagnostics = result
    for diagnostic in diagnostics.cell_diagnostics():
        equilibrium_diagnostic_callback(diagnostic)
    return values


def equilibrium_from_distribution(f, velocity, *, equilibrium_mode, **policy):
    """Target actual quadrature moments; preserve legacy centered variance arithmetic."""
    policy = {"equilibrium_mode": equilibrium_mode, **policy}
    if equilibrium_mode == "continuous":
        primitive = compute_moments(f, np.arange(f.shape[1]), velocity, 0.0)
        return equilibrium_from_primitives(primitive, velocity, **policy)
    return equilibrium_from_conserved(
        compute_discrete_conserved_moments(f, velocity), velocity, **policy)


def equilibrium_metadata(equilibrium_mode, discrete_equilibrium_options=None):
    """JSON-compatible policy plus hashes of the actual equilibrium/kinetic sources."""
    validate_equilibrium_policy(equilibrium_mode, discrete_equilibrium_options)
    directory = Path(__file__).parent
    names = ("equilibrium.py", "discrete_maxwellian.py", "sbgk_maxwellian.py",
             "sbgk_moments.py", "sbgk_initial_conditions.py", "sbgk_imex.py",
             "sbgk_advection.py", "sbgk_butcher.py", "sbgk_grid.py",
             "macroscopic_observables.py")
    return {"equilibrium_mode": equilibrium_mode,
            "discrete_equilibrium_options": asdict(discrete_equilibrium_options or DiscreteMaxwellianOptions()),
            "equilibrium_source_hashes": {name: sha256((directory / name).read_bytes()).hexdigest() for name in names}}
