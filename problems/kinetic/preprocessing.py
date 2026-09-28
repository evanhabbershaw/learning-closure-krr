"""Training-only normalization and full unwhitened PCA."""
import numpy as np
from .protocol import require

class Moments:
    """Stable per-source centered population moment merge, deterministic order."""
    def __init__(self):
        self.n, self.mean, self.m2 = 0, None, None

    def add(self, x):
        x = np.asarray(x, dtype=np.float64)
        require(x.ndim == 2 and len(x) > 0 and np.isfinite(x).all(), 'Invalid moment observations')
        n, mean = len(x), x.mean(axis=0)
        m2 = np.sum((x-mean)**2, axis=0)
        if self.n == 0:
            self.n, self.mean, self.m2 = n, mean, m2
        else:
            delta = mean-self.mean
            self.m2 += m2+delta**2*self.n*n/(self.n+n)
            self.mean += delta*n/(self.n+n)
            self.n += n

    def result(self):
        scale = np.sqrt(self.m2/self.n)
        require(np.all(scale > 0) and np.isfinite(scale).all(), 'Zero/nonfinite normalization scale: STOP')
        return dict(mean=self.mean.tolist(), scale=scale.tolist(), observations=self.n, ddof=0)


def complete_pca(x):
    """Full eigenbasis, deterministic pivot signs, no scaling or whitening."""
    x = np.asarray(x, dtype=np.float64)
    require(x.ndim == 2 and len(x) > 1 and np.isfinite(x).all(), 'Invalid PCA input')
    mean = x.mean(axis=0)
    centered = x-mean
    cov = centered.T @ centered/(len(x)-1)
    values, vectors = np.linalg.eigh((cov+cov.T)*.5)
    order = np.argsort(values)[::-1]
    components = vectors[:, order].T
    pivots = np.argmax(abs(components), axis=1)
    signs = np.sign(components[np.arange(len(components)), pivots])
    signs[signs == 0] = 1
    components *= signs[:, None]
    return dict(mean=mean, components=components, eigenvalues=np.maximum(values[order], 0))


def fit_preprocessing(rows, load):
    """Fit shared Type-I statistics using nodes/intervals 1 through 454."""
    if len(rows) != 72 or any(r['role'] != 'training' or r['family'] != 'I' for r in rows):
        raise ValueError('Paper preprocessing requires the 72 Type-I training members')
    from .data import members
    expected = [r for r in members('training') if r['family'] == 'I']
    if sorted(rows, key=lambda r:r['trajectory_id']) != sorted(expected, key=lambda r:r['trajectory_id']):
        raise ValueError('Paper preprocessing membership differs')
    u_mom, du_mom, states, targets = Moments(), Moments(), [], []
    for row in sorted(rows, key=lambda r: r['trajectory_id']):
        a = load(row)
        u = a['U_node'][1:455]
        du = (np.roll(u, -1, axis=2)-u)/(2/256)
        u_mom.add(u.transpose(0, 2, 1).reshape(-1, 3))
        du_mom.add(du.transpose(0, 2, 1).reshape(-1, 3))
        states.append(u.reshape(454, 768))
        targets.append(a['R_eff'][1:455].reshape(454, 768))
    pca = {f'{prefix}_{k}':v for prefix, arrays in [('U', states), ('R', targets)]
           for k, v in complete_pca(np.concatenate(arrays)).items()}
    return {'U454':u_mom.result(), 'DU454':du_mom.result()}, pca


def prepare_centers(rows, load, representation, condition, norms, pca, baseline=None):
    """Frozen quotas, traversal and exact normalized-byte deduplication."""
    from .protocol import BUDGETS, global_times, local_candidates, local_features, select_unique
    rows = sorted(rows, key=lambda r:r['trajectory_id'])
    if any(r['role'] != 'training' for r in rows):
        raise ValueError('Centers must come from training only')
    if condition == 'I_only':
        rows = [r for r in rows if r['family'] == 'I']
    from .data import members
    expected = [r for r in members('training') if condition != 'I_only' or r['family'] == 'I']
    if rows != sorted(expected, key=lambda r:r['trajectory_id']):
        raise ValueError('Paper center membership differs')
    count, quota = BUDGETS[representation][condition]
    if len(rows) != count:
        raise ValueError('Training membership count mismatch')
    features, targets, locations = {}, {}, {}
    if representation == 'global':
        times = global_times()[condition == 'mixed']
        for row in rows:
            a = load(row); sid = row['trajectory_id']
            features[sid] = (a['U_node'][times].reshape(len(times), -1)-pca['U_mean']) @ pca['U_components'].T
            targets[sid] = (a['R_eff'][times].reshape(len(times), -1)-pca['R_mean']) @ pca['R_components'].T
            locations[sid] = [(int(n), -1) for n in times]
        selected = [(r['trajectory_id'], i) for r in rows for i in range(quota)]
    else:
        for row in rows:
            a = load(row); sid = row['trajectory_id']; candidates = local_candidates(a['U_node'])
            cache = {}
            for n, _ in candidates:
                if n not in cache:
                    cache[n] = local_features(a['U_node'][n], norms['U454' if representation == 'U30' else 'DU454'], representation)
            features[sid] = np.asarray([cache[n][i] for n, i in candidates])
            y = np.asarray([a['R_eff'][n, :, i] for n, i in candidates])
            if representation == 'DU27':
                if baseline is None:
                    raise ValueError('Residual targets require the fixed Type-I baseline')
                base_cache = {n: baseline.predict(local_features(a['U_node'][n], norms['U454'], 'U30')) for n in cache}
                y -= np.asarray([base_cache[n][i] for n, i in candidates])
            targets[sid], locations[sid] = y, candidates
        selected = select_unique(features, {r['trajectory_id']:quota for r in rows}, representation)
    identities = [{'trajectory_id':sid, 'n':locations[sid][i][0], 'interface':locations[sid][i][1]} for sid,i in selected]
    return np.asarray([features[s][i] for s,i in selected]), np.asarray([targets[s][i] for s,i in selected]), identities
