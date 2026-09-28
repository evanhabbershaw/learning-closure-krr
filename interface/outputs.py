"""Run directories, authenticated external inputs, and truthful stage records."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import uuid
from .config import ROOT


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def source_identity():
    h = hashlib.sha256()
    paths = [ROOT/'run_case.py', ROOT/'paper_cases.json']
    for directory in ('interface', 'problems', 'configs', 'scripts', 'data', 'expected'):
        paths.extend((ROOT/directory).rglob('*'))
    for p in sorted(paths):
        rel = p.relative_to(ROOT)
        if p.is_file() and p.suffix in {'.py','.json','.toml','.csv'} and not any(x in rel.parts for x in ('runs','external','__pycache__','.pytest_cache')):
            h.update(str(rel).encode()); h.update(p.read_bytes())
    return h.hexdigest()


class Run:
    def __init__(self, config):
        ops = config['operational']
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        run_id = stamp+'_'+config['scientific_sha256'][:8]+'_'+uuid.uuid4().hex[:8]
        self.path = Path(ops['output']) if ops['output'] else ROOT/'runs'/config['problem']/run_id
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path/'models').mkdir(); (self.path/'figures').mkdir()
        write_json(self.path/'resolved_config.json', config)
        versions = {}
        for name in ['numpy','scipy','cupy','cupy-cuda12x','cupy-cuda13x','matplotlib']:
            try: versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: pass
        self.manifest = dict(schema_version=1, run_status='started', stages=[],
                             source_sha256=source_identity(), python=platform.python_version(),
                             platform=platform.platform(), machine=platform.machine(), packages=versions,
                             expected_result_check='not_performed', workflow=ops['workflow'])
        try:
            import numpy as np
            self.manifest['numpy_backend'] = np.show_config(mode='dicts')
        except (ImportError, TypeError):
            self.manifest['numpy_backend'] = 'not available through show_config(mode=dicts)'
        self.inputs=[]
        self.flush()

    def flush(self):
        write_json(self.path/'run_manifest.json', self.manifest)
        write_json(self.path/'data_manifest.json', {'schema_version':1,'inputs':self.inputs})

    def stage(self, name, status, **details):
        self.manifest['stages'].append(dict(name=name,status=status,**details)); self.flush()

    def finish(self, status, reason=None):
        self.manifest.update(run_status=status, reason=reason); self.flush()


def inputs(config, run, required):
    """Every supplied file is checksum-bound; this is not producer authentication."""
    root = Path(config['operational']['data_root']).resolve()
    case = config['paper_case_id'] or config['derived_from_paper_case'] or config['problem']
    manifest = root/(case+'.json')
    if not manifest.is_file():
        raise FileNotFoundError(f'Missing {manifest.name} under data root. See DATA.md for input manifest schema and roles: {required}')
    bundle=json.loads(manifest.read_text())
    if bundle.get('scientific_sha256') != config['scientific_sha256']:
        raise ValueError('Input bundle scientific_sha256 does not match resolved config')
    run.inputs.append(dict(role='input_manifest',name=manifest.name,sha256=sha256(manifest)))
    result={}
    expected_roles = {}
    if config['problem'] == 'kinetic' and config['experiment_status'] == 'paper_case':
        support = config['scientific']['selected_model_id'].rsplit('_', 1)[0]
        catalog = json.loads((ROOT/'data/external_assets.json').read_text())['assets']
        names = {'features':support+'_features.npy', 'normalizers':'normalizers.npz',
                 'pca':'global_pca.npz', 'baseline_features':'U30_I_only_features.npy'}
        expected_roles = {role: next(a['sha256'] for a in catalog if Path(a['path']).name == name)
                          for role, name in names.items()}
    for role in required:
        if role not in bundle.get('files', {}):
            raise ValueError(f'Missing input role {role}')
        spec=bundle['files'][role]; p=(root/spec['path']).resolve()
        if not p.is_relative_to(root):
            raise ValueError('Input path escapes data root')
        if not p.is_file(): raise FileNotFoundError(f'Missing input {role}: {spec["path"]}')
        actual=sha256(p)
        if actual != spec.get('sha256'): raise ValueError(f'Input checksum mismatch: {role}')
        if role in expected_roles and actual != expected_roles[role]:
            raise ValueError(f'Frozen paper asset mismatch: {role}; generated/changed arrays require a custom data_identity')
        result[role]=p
        run.inputs.append(dict(role=role,name=spec['path'],sha256=actual,bytes=p.stat().st_size))
    run.flush()
    return result,bundle


def require_cuda(device=0):
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() <= device:
            raise RuntimeError('Requested CUDA device is unavailable')
        cp.cuda.Device(device).use()
        return cp
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError('This workflow requires CuPy with an NVIDIA CUDA device; no CPU KRR fallback is implemented.') from exc


def record_cuda(run, cp, device):
    props=cp.cuda.runtime.getDeviceProperties(device)
    name=props.get('name','unknown')
    if isinstance(name,bytes): name=name.decode(errors='replace')
    run.manifest['cuda']=dict(device=device,name=name,runtime_version=cp.cuda.runtime.runtimeGetVersion(),driver_version=cp.cuda.runtime.driverGetVersion())
    run.flush()


def save_gpu_model(path, model, cp):
    """Export numerical fitted arrays without executable pickle payloads."""
    import numpy as np
    arrays={}
    for key in ['data_old','beta','q1','q2','epsilon','lambda_reg']:
        value=getattr(model.dm,key,None)
        if value is not None: arrays[key]=cp.asnumpy(cp.asarray(value))
    if hasattr(model,'std_nmse'): arrays['std_nmse']=cp.asnumpy(cp.asarray(model.std_nmse))
    with Path(path).open('xb') as f: np.savez(f,**arrays)
    return sha256(path)
