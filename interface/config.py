"""Standard-library case registry, strict resolution and scientific identity."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
OPERATIONAL = {'data_root', 'output', 'workflow', 'device', 'render_kind'}
SUPPORTED_CHANGES = {
    'l63': {'training_size', 'kernel', 'stage_reconstruction', 'candidate_count', 'candidate_chunk',
            'models', 'training_starts', 'validation_length', 'validation_offsets', 'validation_horizon',
            'test_trials', 'test_horizon', 'dt', 'error_threshold', 'lyapunov_exponent', 'summary_ddof', 'data_identity'},
    'ks': {'training_size', 'kernel', 'memory_order', 'candidate_chunk', 'epsilon_count', 'lambda_count',
           'epsilon_log10_bounds', 'lambda_log10_bounds', 'models', 'training_starts', 'validation_length',
           'validation_offsets', 'validation_horizon', 'test_trials', 'test_horizon', 'split_index',
           'dt', 'alpha', 'substeps', 'contour_points', 'error_threshold', 'lyapunov_exponent', 'summary_ddof', 'data_identity'},
    'kinetic': {'kernel_epsilon', 'lambda_reg', 'centers', 'data_identity'},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def registry(problem=None):
    cases = json.loads((ROOT/'paper_cases.json').read_text())['cases']
    return [c for c in cases if problem is None or c['problem'] == problem]


def merge(base, changes, prefix=''):
    result = deepcopy(base)
    for key, value in changes.items():
        if key not in base:
            raise ValueError(f'Unknown scientific key: {prefix}{key}')
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f'Expected a table: {prefix}{key}')
            result[key] = merge(base[key], value, prefix+key+'.')
        else:
            if isinstance(base[key], bool) and not isinstance(value, bool):
                raise ValueError(f'Expected boolean: {prefix}{key}')
            result[key] = value
    return result


def validate(problem, s):
    import math
    def finite_numbers(value):
        if isinstance(value, dict):
            return all(finite_numbers(v) for v in value.values())
        if isinstance(value, list):
            return all(finite_numbers(v) for v in value)
        return not isinstance(value, float) or math.isfinite(value)
    if not finite_numbers(s):
        raise ValueError('Scientific numeric values must be finite')
    expected = 'numpy_scipy_cpu' if problem == 'kinetic' else 'cupy_cuda'
    if s['backend'] != expected:
        raise ValueError(f'{problem} requires {expected}; no alternative backend implemented')
    if s['kernel'] not in ({'rbf'} if problem == 'kinetic' else {'rbf', 'diffusion_maps'}):
        raise ValueError('Unsupported kernel')
    for k in ('training_size', 'models', 'candidate_count', 'candidate_chunk', 'epsilon_count', 'lambda_count', 'substeps', 'contour_points', 'test_trials', 'test_horizon', 'validation_horizon', 'centers', 'Nx', 'Nv', 'intervals'):
        if k in s and (isinstance(s[k], bool) or not isinstance(s[k], int) or s[k] < 1):
            raise ValueError(f'{k} must be a positive integer')
    if not math.isfinite(s['dt']) or s['dt'] <= 0:
        raise ValueError('dt must be positive and finite')
    if problem == 'l63' and s['stage_reconstruction'] not in {'heun','rk4_quadratic','rk4_cubic','rk4_quartic','rk4_quintic'}:
        raise ValueError('Unsupported stage reconstruction')
    if problem == 'ks' and (isinstance(s['memory_order'], bool) or not isinstance(s['memory_order'], int) or s['memory_order'] < 0):
        raise ValueError('memory_order must be a nonnegative integer')
    if problem != 'kinetic':
        if s['error_threshold'] <= 0 or s['lyapunov_exponent'] <= 0:
            raise ValueError('Metric threshold and Lyapunov exponent must be positive')
        if isinstance(s['summary_ddof'], bool) or not isinstance(s['summary_ddof'], int) or not 0 <= s['summary_ddof'] < s['models']:
            raise ValueError('Invalid summary ddof')
        if len(s['training_starts']) != s['models'] or len(set(s['training_starts'])) != s['models']:
            raise ValueError('Distinct training starts required for each model')
        if any(i < 0 or i+s['validation_horizon'] > s['validation_length'] for i in s['validation_offsets']):
            raise ValueError('Validation windows exceed pool')
        length = s['training_size'] + (s.get('memory_order', -1)+1) + s['validation_length']
        starts = sorted(s['training_starts'])
        if any(a+length > b for a,b in zip(starts, starts[1:])):
            raise ValueError('Overlapping model training/validation blocks')
        if problem == 'ks' and starts[-1]+length > s['split_index']:
            raise ValueError('Training/validation overlaps test split')
    else:
        if s['representation'] != {'global':'global','zeroth_order_local':'U30','first_order_local':'DU27'}.get(s['model']):
            raise ValueError('Model and representation disagree')
        for k in ['kernel_epsilon','lambda_reg','dx']:
            if not math.isfinite(s[k]) or s[k] <= 0:
                raise ValueError(f'{k} must be positive and finite')
        if 'baseline' in s and s['baseline']['training_data'] != 'type_i':
            raise ValueError('This release implements the fixed Type-I residual baseline only')


def resolve(problem, *, paper_case=None, config=None, overrides=None, operational=None):
    file = {} if config is None else tomllib.loads(Path(config).read_text())
    unknown = set(file)-{'extends','problem','scientific',*OPERATIONAL}
    if unknown:
        raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    if file.get('problem', problem) != problem:
        raise ValueError('Config problem differs from dispatcher')
    if paper_case and file.get('extends', paper_case) != paper_case:
        raise ValueError('Conflicting paper case and extends')
    selected = paper_case or file.get('extends')
    cases = {c['id']:c for c in registry(problem)}
    if selected is not None and selected not in cases:
        raise ValueError(f'Unknown {problem} paper case: {selected}')
    if selected is None:
        if 'scientific' not in file:
            raise ValueError('Choose --paper-case or supply a complete custom config')
        template = next((c['scientific'] for c in cases.values() if problem != 'kinetic' or c['scientific']['model'] == file['scientific'].get('model')), None)
        if template is None or set(file['scientific']) != set(template):
            raise ValueError('Custom config must supply a complete scientific table or use extends')
        base = template; original = None
    else:
        original = cases[selected]; base = original['scientific']
    scientific = merge(base, file.get('scientific', {}))
    scientific = merge(scientific, overrides or {})
    validate(problem, scientific)
    ops = {'workflow':'replay' if problem=='kinetic' else 'reproduce', 'data_root':'external', 'output':None, 'device':0, 'render_kind':'summary'}
    ops.update({k:file[k] for k in OPERATIONAL if k in file})
    ops.update({k:v for k,v in (operational or {}).items() if v is not None})
    if ops['workflow'] not in {'reproduce','replay','render'}:
        raise ValueError('Unsupported workflow')
    allowed_render = {'l63':{'summary'}, 'ks':{'summary','validation_heatmap','rollout'}, 'kinetic':{'summary','rollout'}}
    if ops['render_kind'] not in allowed_render[problem] or (ops['workflow'] != 'render' and ops['render_kind'] != 'summary'):
        raise ValueError('Render kind is unsupported for this problem/workflow')
    if not isinstance(ops['device'], int) or ops['device'] < 0:
        raise ValueError('Device ordinal must be nonnegative')
    paper = selected is not None and scientific == base
    changed = {k:v for k,v in scientific.items() if v != base.get(k)}
    unsupported = set(changed) - SUPPORTED_CHANGES[problem]
    if unsupported:
        raise ValueError(f'No dispatcher implementation for overrides: {sorted(unsupported)}; use supported settings or the numerical array API')
    if problem == 'kinetic' and not paper:
        scientific['selected_model_id'] = None
    return dict(schema_version=1, problem=problem, scientific=scientific, operational=ops,
                experiment_status='paper_case' if paper else 'custom', paper_case_id=selected if paper else None,
                derived_from_paper_case=selected if not paper else None, overrides=changed,
                scientific_sha256=digest(scientific), original_spec_sha256=digest(base) if original else None,
                publication_mapping=original['publication_mapping'] if paper else [],
                resolution_status='resolved_specification', input_status='external_inputs_required',
                implementation_verification='cpu_unit_tests_only; CUDA execution unverified',
                historical_identity_status='public_derivation; historical_byte_replay_not_certified')
