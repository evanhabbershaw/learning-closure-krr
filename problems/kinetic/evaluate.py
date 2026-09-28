"""Four-field spatial relative errors and family population statistics."""
import numpy as np
from .rollout import primitive_fields


def errors(prediction, truth):
    if prediction.shape != truth.shape:
        raise ValueError('Incomplete or mismatched trajectory')
    p, y = primitive_fields(prediction), primitive_fields(truth)
    err = np.linalg.norm(p-y, axis=2) / np.maximum(np.linalg.norm(y, axis=2), np.finfo(float).tiny)
    if not np.isfinite(err).all():
        raise ValueError('Nonfinite error history')
    return {'F': float(err[-1].mean()), 'M': float(err.max(axis=0).mean()),
            'fields': ['rho', 'u', 'theta', 'E'], 'final_errors': err[-1].tolist(),
            'max_errors': err.max(axis=0).tolist()}, err


def aggregate(rows, *, count_per_family=24):
    """Table 4: ddof=0 over per-trajectory F in each protected family."""
    if len({r['trajectory_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate trajectory')
    result = {}
    for family in ('I', 'II'):
        selected = [r for r in rows if r['family'] == family]
        if len(selected) != count_per_family or any(r['health'] != 'PASS' for r in selected):
            raise ValueError('Missing or failed family trajectories')
        result[family] = {}
        for metric in ('F', 'M'):
            values = np.array([r[metric] for r in selected])
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError('Invalid error')
            result[family][metric] = {'mean': float(values.mean()), 'std': float(values.std(ddof=0)),
                                      'ddof': 0, 'count': count_per_family}
    result['J_final'] = .5*(result['I']['F']['mean'] + result['II']['F']['mean'])
    result['family_sum'] = 2*result['J_final']
    result['J_tr'] = result['J_final'] + .125*(result['I']['M']['mean'] + result['II']['M']['mean'])
    return result
