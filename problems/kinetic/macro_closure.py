"""Conservative effective-flux macroscopic deployment."""
from typing import Any
import numpy as np
from .kinetic_closure.macroscopic_fv import macroscopic_explicit_rk_stage_fluxes

def macro_effective_flux_state(U: np.ndarray, butcher: Any, grid: Any, dt: float) -> np.ndarray:
    q_zero = np.zeros((butcher.num_stages, U.shape[-1]))
    stages = macroscopic_explicit_rk_stage_fluxes(U, q_zero, butcher, grid, dt)
    return np.einsum("s,scx->cx", butcher.be, stages)


def conservative_update(U, G_eff, dt, dx):
    """Periodic common-interface update of (rho, rho*u, E)."""
    U = np.asarray(U, dtype=np.float64)
    G_eff = np.asarray(G_eff, dtype=np.float64)
    if U.ndim != 2 or U.shape[0] != 3 or G_eff.shape != U.shape:
        raise ValueError('U and effective flux must have shape (3, Nx)')
    if not np.isfinite(U).all() or not np.isfinite(G_eff).all():
        raise ValueError('Nonfinite state or flux')
    if not np.isfinite([dt, dx]).all() or dt <= 0 or dx <= 0:
        raise ValueError('dt and dx must be finite and positive')
    return U - (dt / dx) * (G_eff - np.roll(G_eff, 1, axis=-1))


def corrected_flux_step(U, R_eff, butcher, grid, dt):
    """G_hat_eff = G_macro_eff(U_hat) + R_eff_hat(U_hat)."""
    dx = grid['hX'] if isinstance(grid, dict) else grid.hX
    flux = macro_effective_flux_state(U, butcher, grid, dt) + R_eff
    return conservative_update(U, flux, dt, dx)
