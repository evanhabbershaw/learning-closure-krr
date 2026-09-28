"""Recursive learned macroscopic state evolution on CPU."""
import numpy as np
from .macro_closure import corrected_flux_step
from .protocol import local_features


def primitive_fields(u):
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        rho = u[:, 0]
        speed = u[:, 1] / rho
        theta = 2*u[:, 2]/rho - speed**2
    return np.stack((rho, speed, theta, u[:, 2]), axis=1)


def predict_closure(model, representation, state, norms, pca=None, baseline=None, *, dx):
    if representation == 'global':
        features = (state.reshape(1, -1)-pca['U_mean']) @ pca['U_components'].T
        return (model.predict(features) @ pca['R_components'] + pca['R_mean']).reshape(state.shape)
    if representation == 'U30':
        return model.predict(local_features(state, norms['U454'], 'U30', dx=dx)).T
    if representation != 'DU27' or baseline is None:
        raise ValueError('DU27 requires its fixed Type-I zeroth-order baseline')
    base = baseline.predict(local_features(state, norms['U454'], 'U30', dx=dx))
    correction = model.predict(local_features(state, norms['DU454'], 'DU27', dx=dx))
    return (base + correction).T


def rollout(initial, initial_correction, predict, butcher, grid, dt, steps):
    """Use the supplied exact correction for interval zero, then predicted states.

    Failure preserves the completed prefix and returns its explicit reason.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError('steps must be a positive integer')
    history = [np.asarray(initial, dtype=np.float64).copy()]
    for n in range(steps):
        try:
            correction = initial_correction if n == 0 else predict(history[-1])
            next_u = corrected_flux_step(history[-1], correction, butcher, grid, dt)
            fields = primitive_fields(next_u[None])
            if not np.isfinite(fields).all() or np.any(fields[:, 0] <= 0) or np.any(fields[:, 2] <= 0):
                raise ValueError('nonfinite/nonpositive state')
            history.append(next_u)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            return np.asarray(history), {'health': 'FAIL', 'completed_steps': n, 'failure': str(exc)}
    return np.asarray(history), {'health': 'PASS', 'completed_steps': steps, 'failure': None}
