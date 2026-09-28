"""Scientific membership, sampling and selection contracts."""
import csv
import json
import hashlib
from pathlib import Path
import numpy as np
MODE = 'discrete_mieussens'
OPTIONS = dict(absolute_tolerance=2e-13, relative_tolerance=2e-12,
               maximum_iterations=50, maximum_backtracks=30,
               armijo_parameter=1e-4, minimum_step=2.0**-30, condition_limit=1e16)
GRID = dict(Nx=256, Nv=128, x_domain=[-1, 1], v_domain=[-8, 8],
            dx=2/256, dt=0.000439453125, intervals=455, nodes=456,
            final_time=0.199951171875, requested_final_time=0.2,
            theta_x=2, theta_v=2, boundary='periodic', butcher_table=2,
            stage_advection_mode='corrected', epsi=0.001)
COUNTS = {'training': {'I': 72, 'II': 72}, 'validation': {'I': 6, 'II': 6},
          'protected_test': {'I': 24, 'II': 24}}
BUDGETS = {'global': {'I_only': (72, 114), 'mixed': (144, 57), 'N': 8208},
           'U30': {'I_only': (72, 56), 'mixed': (144, 28), 'N': 4032},
           'DU27': {'I_only': (72, 112), 'mixed': (144, 56), 'N': 8064}}

class ProtocolError(ValueError):
    pass


def require(value, message):
    if not value:
        raise ProtocolError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def global_times():
    large = np.unique(np.rint(np.linspace(1, 454, 114)).astype(np.int64))
    small = large[::2]
    require(len(large) == 114 and len(small) == 57 and set(small) <= set(large),
            'STOP: approved nested temporal support failed')
    require(small[0] == 1 and small[-1] == 450, 'STOP: unexpected T57 endpoints')
    return large, small


def normalization_contract():
    return dict(reference='regenerated conservative Family-I training only', sources=72,
                times=list(range(1, 455)), cells=256, observations=72*454*256, ddof=0,
                U='conserved [rho, momentum, E]', DU='(roll(U,-1)-U)/dx',
                normalizers=['U454', 'DU454'], shared_across_conditions=True,
                zero_scale_policy='abort', state_stencil=list(range(-4, 6)))


def pca_contract():
    return dict(d_U=768, d_R=768, whiten=False, full_rank=True,
                reference='regenerated conservative Family-I training only', sources=72,
                times=list(range(1, 455)), snapshots=72*454,
                state='U_node[n]', target='R_eff[n]', flatten_order='C: component then cell',
                covariance_ddof=1, shared_object=True, secondary_reduced_experiment=False)


def local_order():
    rev = lambda v, b: int(f'{v:0{b}b}'[::-1], 2)
    return np.array([rev(t, 4)*64 + rev((cycle+5*t) % 64, 6)
                     for cycle in range(64) for t in range(16)], dtype=np.int64)


def local_candidates(states):
    require(states.shape == (456, 3, 256), 'Candidate state shape mismatch')
    require(np.isfinite(states).all(), 'Nonfinite candidate states')
    result = []
    for n in np.unique(np.rint(np.linspace(1, 454, 16)).astype(int)):
        u = states[n]
        activity = np.linalg.norm((.5*(u+np.roll(u, -1, axis=1)) -
                                   np.array([1., 0., .5])[:, None]) /
                                  np.array([1., 1., .5])[:, None], axis=0)
        faces = np.argsort(activity, kind='stable')[np.arange(2, 256, 4)]
        result.extend((int(n), int(face)) for face in faces)
    require(len(result) == 1024, 'Candidate count mismatch')
    return [result[i] for i in local_order()]


def local_features(u, normalizer, representation, *, dx=None):
    require(representation in ('U30', 'DU27'), 'Unknown representation')
    require(u.ndim == 2 and u.shape[0] == 3 and u.shape[1] >= 10, 'State shape mismatch')
    nx = u.shape[1]
    dx = GRID['dx'] if dx is None else dx
    require(np.isfinite(dx) and dx > 0, 'Invalid dx')
    index = (np.arange(nx)[:, None] + np.arange(-4, 6)) % nx
    raw = u[:, index].transpose(1, 2, 0)
    if representation == 'DU27':
        raw = np.diff(raw, axis=1) / dx
    mean, scale = np.asarray(normalizer['mean']), np.asarray(normalizer['scale'])
    require(mean.shape == scale.shape == (3,) and np.all(scale > 0), 'Invalid normalizer')
    return ((raw-mean)/scale).reshape(nx, -1)


def feature_key(row):
    a = np.array(row, dtype='<f8', order='C', copy=True)
    require(a.ndim == 1 and np.isfinite(a).all(), 'Invalid feature row')
    a[a == 0] = 0.0
    return a.tobytes()


def select_unique(features, quotas, representation):
    """Independent invocation/seen set for each model and representation.

    Feature rows are already in candidate traversal order. Acceptance uses the
    complete normalized bytes, never a rounded value or a hash collision proxy.
    """
    require(representation in ('U30', 'DU27'), 'Unknown representation')
    dim = 30 if representation == 'U30' else 27
    require(set(features) == set(quotas) and all(q > 0 for q in quotas.values()), 'Quota keys mismatch')
    for x in features.values():
        require(x.ndim == 2 and x.shape[1] == dim and np.isfinite(x).all(), 'Feature shape/value mismatch')
    cursor = dict.fromkeys(features, 0)
    accepted = dict.fromkeys(features, 0)
    seen, selected = set(), []
    while any(accepted[k] < quotas[k] for k in features):
        for source in sorted(features):
            if accepted[source] == quotas[source]:
                continue
            x = features[source]
            while cursor[source] < len(x):
                i = cursor[source]
                cursor[source] += 1
                key = feature_key(x[i])
                if key not in seen:
                    seen.add(key)
                    accepted[source] += 1
                    selected.append((source, i))
                    break
            else:
                raise ProtocolError(f'STOP: {representation} source quota exhausted: {source}')
    return selected


def center_plan(protocol, representation, condition):
    require(representation in BUDGETS and condition in ('I_only', 'mixed'), 'Unknown center plan')
    rows = protocol.members('training', 'I' if condition == 'I_only' else None)
    count, quota = BUDGETS[representation][condition]
    require(len(rows) == count and count*quota == BUDGETS[representation]['N'], 'Center arithmetic mismatch')
    return {r['trajectory_id']: quota for r in rows}


def selection_gate(scores):
    require(bool(scores), 'No eligible candidates')
    require(all(np.isfinite([s['J_final'], s['J_tr']]).all() and
                0 <= s['J_final'] <= s['J_tr'] for s in scores.values()), 'Invalid selection score')
    # Exact ties use stable candidate ID; J_tr never breaks a J_final tie.
    f = min(scores, key=lambda k: (scores[k]['J_final'], k))
    t = min(scores, key=lambda k: (scores[k]['J_tr'], k))
    def relative(a, b):
        return a/b-1 if b else (0. if a == 0 else float('inf'))
    df = relative(scores[t]['J_final'], scores[f]['J_final'])
    dt = relative(scores[f]['J_tr'], scores[t]['J_tr'])
    # Compare the equivalent ratio threshold without cancellation in ratio-1.
    # E.g. float64 (1.02/1-1) is slightly above .02 at the exact input boundary.
    stop = f != t and (scores[t]['J_final'] > 1.02*scores[f]['J_final'] or
                       scores[f]['J_tr'] > 1.02*scores[t]['J_tr'])
    return dict(theta_F=f, theta_T=t, Delta_F=df if np.isfinite(df) else None,
                Delta_T=dt if np.isfinite(dt) else None, infinite_delta=not np.isfinite([df, dt]).all(),
                status='STOP_HUMAN_REVIEW' if stop else 'PASS', selected=None if stop else f,
                protected_testing_allowed=False)  # selection must be frozen before test evaluation
