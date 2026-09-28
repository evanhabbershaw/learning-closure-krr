"""Explicit kinetic reference generation and stage-weighted transport targets."""
import json
from pathlib import Path
import numpy as np
from .protocol import GRID, OPTIONS, MODE, read_csv


def members(role):
    names = {'training': 'training_manifest.csv', 'validation': 'validation_manifest.csv',
             'protected_test': 'protected_membership_manifest.csv'}
    root = Path(__file__).resolve().parents[2] / 'configs/paper/kinetic'
    return read_csv(root / names[role])


def generate_reference(row, *, grid_config=None):
    """Generate one declared IC; callers must keep test data out of selection."""
    from .kinetic_closure.sbgk_grid import make_sbgk_grid
    from .kinetic_closure.sbgk_butcher import get_butcher
    from .kinetic_closure.sbgk_initial_conditions import make_local_maxwellian_initial_condition, periodic_distance
    from .kinetic_closure.sbgk_imex import generate_kinetic_trajectory
    from .kinetic_closure.discrete_maxwellian import DiscreteMaxwellianOptions
    from .stage_fluxes import compute_step_stage_data, distribution_moments
    from .macro_closure import macro_effective_flux_state
    c = dict(GRID if grid_config is None else grid_config)
    grid = make_sbgk_grid(c['Nx'], c['Nv'], *c['x_domain'], *c['v_domain'],
                          t_final=c['dt']*c['intervals'], theta_x=c['theta_x'], theta_v=c['theta_v'])
    butcher = get_butcher(c['butcher_table'])
    p = {k: float(v) for k, v in json.loads(row['ic_parameters']).items()}
    theta = np.full(c['Nx'], p['theta0'])
    pulses = [(p['A'], p['sigma'], p['mu'])] if row['family'] == 'I' else [
        (p[f'A{j}'], p[f'sigma{j}'], p[f'mu{j}']) for j in (1, 2)]
    for a, s, mu in pulses:
        theta += a*np.exp(-(periodic_distance(grid.xC, mu, *c['x_domain'])/s)**2)
    policy = dict(equilibrium_mode=MODE, discrete_equilibrium_options=DiscreteMaxwellianOptions(**OPTIONS))
    initial = make_local_maxwellian_initial_condition(grid, np.full(c['Nx'], p['rho0']),
                                                      np.full(c['Nx'], p['u0']), theta, **policy)
    trajectory = generate_kinetic_trajectory(initial, butcher, grid, c['epsi'], c['intervals'], c['dt'],
                                            boundary='periodic', stage_advection_mode='corrected', **policy)
    kinetic, macro = [], []
    for f, u in zip(trajectory.f_history[:-1], trajectory.U_history[:-1]):
        stage = compute_step_stage_data(f, grid, butcher, c['epsi'], c['dt'], **policy)
        transport = np.asarray([distribution_moments(s, grid.vC) for s in stage['fluxes']])
        kinetic.append(np.einsum('s,scx->cx', butcher.be, transport))
        macro.append(macro_effective_flux_state(u, butcher, grid, c['dt']))
    kinetic, macro = np.asarray(kinetic), np.asarray(macro)
    return dict(times=trajectory.times, U_node=trajectory.U_history, G_kin_eff=kinetic,
                G_macro_eff=macro, R_eff=kinetic-macro)


def load_bank(bank, row):
    path = Path(bank) / row['role'] / (row['trajectory_id']+'.npz')
    from interface.outputs import sha256
    binding = json.loads(path.with_suffix('.json').read_text())
    if binding.get('member') != row or binding.get('sha256') != sha256(path):
        raise ValueError('Bank member/checksum binding mismatch')
    with np.load(path, allow_pickle=False) as z:
        arrays = {k:z[k] for k in ('times', 'U_node', 'R_eff')}
    if arrays['U_node'].shape != (456, 3, 256) or arrays['R_eff'].shape != (455, 3, 256):
        raise ValueError('Bank shape mismatch')
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Nonfinite bank')
    if not np.array_equal(arrays['times'], np.arange(456)*GRID['dt']):
        raise ValueError('Bank time grid mismatch')
    return arrays
