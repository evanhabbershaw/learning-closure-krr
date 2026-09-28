import ast
import importlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from interface.config import ROOT, registry, resolve


def test_registry_exact_combinations_and_resolution():
    cases=registry()
    assert len(cases)==len({c['id'] for c in cases})==38
    assert {p:sum(c['problem']==p for c in cases) for p in ['l63','ks','kinetic']}=={'l63':16,'ks':16,'kinetic':6}
    for c in cases:
        r=resolve(c['problem'],config=ROOT/'configs/paper'/c['problem']/(c['id']+'.toml'))
        assert r['experiment_status']=='paper_case'
        assert r['scientific']==c['scientific']
    assert all(c['scientific']['stage_reconstruction']=='rk4_cubic' or c['scientific']['training_size']==4096 for c in registry('l63'))


def test_custom_classification_and_operational_changes():
    base='l63-rbf-rk4-cubic-n512'
    assert resolve('l63',paper_case=base,operational={'output':'example'})['experiment_status']=='paper_case'
    r=resolve('l63',paper_case=base,overrides={'training_size':1536})
    assert r['experiment_status']=='custom' and r['paper_case_id'] is None and r['derived_from_paper_case']==base
    assert resolve('l63',paper_case=base,overrides={'training_size':512})['experiment_status']=='paper_case'
    for problem in ['l63','ks','kinetic']:
        assert resolve(problem,config=ROOT/'configs/custom_examples'/f'{problem}.toml')['experiment_status']=='custom'


@pytest.mark.parametrize('overrides',[{'backend':'numpy_cpu'},{'training_size':0},{'training_size':True},{'typo':1},{'dt':float('nan')}])
def test_reject_invalid_or_unsupported_configuration(overrides):
    with pytest.raises(ValueError): resolve('l63',paper_case='l63-rbf-rk4-cubic-n512',overrides=overrides)


def test_cli_listing_and_resolve_without_numeric_imports():
    code="import run_case,sys; assert 'numpy' not in sys.modules; assert 'cupy' not in sys.modules; run_case.main(['--list-paper-cases','--json'])"
    r=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,check=True)
    assert len(json.loads(r.stdout))==38
    r=subprocess.run([sys.executable,'run_case.py','ks','--paper-case','ks-diffusion-maps-m0-n16384','--resolve-only'],cwd=ROOT,capture_output=True,text=True,check=True)
    assert len(json.loads(r.stdout)['scientific']['validation_offsets'])==5


def test_final_protocol_overrides():
    for c in registry('l63'): assert len(c['scientific']['validation_offsets'])==3 and c['scientific']['input']=='raw_xyz'
    for c in registry('ks'): assert len(c['scientific']['validation_offsets'])==5 and c['scientific']['reference_truncation']=='1 <= |k| <= 48'
    for c in registry('kinetic'):
        s=c['scientific'];assert s['figure_case']=='validation_II_VIIB_06' and s['summary_ddof']==0
        assert s['pca_rank']==768 and s['pca_whiten'] is False
        if s['representation']=='DU27':
            assert s['baseline']['training_data']=='type_i' and s['baseline']['kernel_epsilon']==2953.231481793101


def test_memberships_disjoint_and_complete():
    from problems.kinetic.data import members
    seen=set()
    for role,count in [('training',72),('validation',6),('protected_test',24)]:
        rows=members(role);ids={r['trajectory_id'] for r in rows}
        assert len(ids)==len(rows)==2*count and not seen.intersection(ids)
        assert all(sum(r['family']==f for r in rows)==count for f in ['I','II']);seen|=ids
    assert 'validation_II_VIIB_06' in seen


def test_kinetic_corrected_flux_step_and_conservation():
    from problems.kinetic.macro_closure import corrected_flux_step,macro_effective_flux_state
    from problems.kinetic.kinetic_closure.sbgk_butcher import get_butcher
    from problems.kinetic.kinetic_closure.sbgk_grid import make_sbgk_grid
    grid=make_sbgk_grid(16,16,-1,1,-8,8,t_final=.1,theta_x=2,theta_v=2);butcher=get_butcher(2)
    x=np.arange(16)*2*np.pi/16
    U=np.vstack((1+.05*np.sin(x),.01*np.cos(x),1+.03*np.sin(x)))
    R=np.random.default_rng(41).normal(size=U.shape)*.001;dt=.0001
    flux=macro_effective_flux_state(U,butcher,grid,dt)+R
    expected=U-dt/grid.hX*(flux-np.roll(flux,1,axis=-1))
    actual=corrected_flux_step(U,R,butcher,grid,dt)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_allclose(actual.sum(axis=1),U.sum(axis=1),rtol=0,atol=1e-14)


def test_macro_transitive_dependency_boundary():
    """Inspect the whole public import closure, not just one source string."""
    forbidden={'discrete_maxwellian','equilibrium','sbgk_imex','sbgk_initial_conditions','stage_fluxes','data'}
    todo=['problems.kinetic.rollout','problems.kinetic.macro_closure'];seen=set()
    while todo:
        module=todo.pop()
        if module in seen: continue
        seen.add(module)
        path=ROOT.joinpath(*module.split('.')).with_suffix('.py')
        if not path.exists(): continue
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):
                if node.level:
                    package=module.rsplit('.',1)[0]
                    target=importlib.util.resolve_name('.'*node.level+(node.module or ''),package)
                else: target=node.module or ''
                assert not (set(target.split('.'))&forbidden),target
                if target.startswith('problems.kinetic'):todo.append(target)
    for module in ['rollout','macro_closure']:
        s=(ROOT/'problems/kinetic'/f'{module}.py').read_text()
        for token in ['moment_maxwellian','collision','gamma','delta','C_eff']:
            assert token not in s


def test_macro_import_and_run_with_reference_imports_blocked():
    code="""
import builtins
old=builtins.__import__
def guarded(name,*args,**kw):
    if any(x in name for x in ['discrete_maxwellian','equilibrium','sbgk_imex']):
        raise AssertionError(name)
    return old(name,*args,**kw)
builtins.__import__=guarded
from problems.kinetic.rollout import rollout
from problems.kinetic.kinetic_closure.sbgk_grid import make_sbgk_grid
from problems.kinetic.kinetic_closure.sbgk_butcher import get_butcher
import numpy as np
grid=make_sbgk_grid(16,16,-1,1,-8,8,t_final=.1,theta_x=2,theta_v=2)
u=np.tile(np.array([[1.],[0.],[.5]]),(1,16))
h,status=rollout(u,np.zeros_like(u),lambda u:np.zeros_like(u),get_butcher(2),grid,.0001,2)
assert status['health']=='PASS'
"""
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)


def test_local_features_stencil_and_spatial_difference():
    from problems.kinetic.protocol import local_features
    u=np.arange(3*16,dtype=float).reshape(3,16)
    norms={'mean':[0,0,0],'scale':[1,1,1]}
    direct=local_features(u,norms,'U30',dx=.125)
    differences=local_features(u,norms,'DU27',dx=.125)
    assert direct.shape==(16,30) and differences.shape==(16,27)
    np.testing.assert_array_equal(differences,np.diff(direct.reshape(16,10,3),axis=1).reshape(16,27)/.125)
    np.testing.assert_array_equal(direct[0].reshape(10,3),u[:,np.arange(-4,6)%16].T)


def test_krr_literal_ridge_and_kernel():
    from problems.kinetic.kinetic_closure.final_learning_krr_20260920_v1 import kernel,solve
    X=np.array([[0.],[2.]])
    K=kernel(X,kernel_epsilon=2.)
    np.testing.assert_allclose(K,[[1,np.exp(-.5)],[np.exp(-.5),1]])
    Y=np.array([[1.],[3.]])
    np.testing.assert_allclose(solve(K,Y,lambda_reg=.1),np.linalg.solve(K+.1*np.eye(2),Y),rtol=1e-14)


def test_pca_no_whitening_round_trip():
    from problems.kinetic.preprocessing import complete_pca
    X=np.random.default_rng(3).normal(size=(12,4))*[1,2,3,4]
    p=complete_pca(X);Z=(X-p['mean'])@p['components'].T
    np.testing.assert_allclose(Z@p['components']+p['mean'],X,atol=1e-14)
    assert p['components'].shape==(4,4)


def test_table4_population_sd():
    from problems.kinetic.evaluate import aggregate
    rows=[dict(trajectory_id=f'{f}{i}',family=f,health='PASS',F=float(i),M=float(i+1)) for f in ['I','II'] for i in range(24)]
    a=aggregate(rows)
    assert a['I']['F']['std']==np.std(np.arange(24),ddof=0)
    assert a['J_final']==11.5


def test_exact_byte_dedup_and_quota_failure():
    from problems.kinetic.protocol import select_unique
    x=np.zeros((3,30));x[1,0]=1;x[2,0]=2
    selected=select_unique({'a':x,'b':x+3},{'a':2,'b':2},'U30')
    assert selected==[('a',0),('b',0),('a',1),('b',1)]
    with pytest.raises(ValueError):select_unique({'a':np.zeros((2,30))},{'a':2},'U30')


def test_ks_fourier_transform_and_l63_reference():
    from problems.ks.data import pos6_to_phys14
    from problems.l63.reference import L63,rk4_step
    v=np.random.default_rng(2).normal(size=(5,6))+1j
    np.testing.assert_allclose(np.fft.fft(pos6_to_phys14(v),axis=1)[:,1:7]/14,v,atol=1e-14)
    np.testing.assert_array_equal(rk4_step(L63,np.zeros(3),.01),np.zeros(3))


def test_reference_retains_conservative_equilibrium():
    from problems.kinetic.kinetic_closure.equilibrium import equilibrium_from_primitives
    from problems.kinetic.kinetic_closure.discrete_maxwellian import compute_discrete_conserved_moments,DiscreteMaxwellianOptions
    v=np.linspace(-7.875,7.875,64)
    f=equilibrium_from_primitives(np.array([[1.],[.1],[1.2]]),v,equilibrium_mode='discrete_mieussens',discrete_equilibrium_options=DiscreteMaxwellianOptions())
    moments=compute_discrete_conserved_moments(f[:,0],v)
    np.testing.assert_allclose(moments,[1.,.1,.605],rtol=1e-10,atol=1e-12)


def test_cpu_modules_import():
    modules=['interface.config','interface.outputs','problems.l63.reference','problems.l63.protocol','problems.l63.reporting','problems.l63.pipeline',
             'problems.ks.data','problems.ks.pipeline','problems.kinetic.pipeline','problems.kinetic.data','problems.kinetic.fit','problems.kinetic.selection','problems.kinetic.preprocessing']
    modules += ['problems.kinetic.kinetic_closure.'+p.stem for p in (ROOT/'problems/kinetic/kinetic_closure').glob('*.py') if p.stem!='__init__']
    for name in modules: importlib.import_module(name)


def test_first_interval_exact_then_query_predicted_state():
    from problems.kinetic.rollout import rollout
    from problems.kinetic.kinetic_closure.sbgk_grid import make_sbgk_grid
    from problems.kinetic.kinetic_closure.sbgk_butcher import get_butcher
    grid=make_sbgk_grid(16,16,-1,1,-8,8,t_final=.1,theta_x=2,theta_v=2)
    u=np.tile(np.array([[1.],[0.],[.5]]),(1,16))
    correction=np.zeros_like(u);correction[0]=.001*np.sin(np.arange(16))
    queried=[]
    def predict(state):
        queried.append(state.copy());return np.zeros_like(state)
    history,status=rollout(u,correction,predict,get_butcher(2),grid,.0001,3)
    assert status['health']=='PASS' and len(queried)==2
    np.testing.assert_array_equal(queried[0],history[1])
    np.testing.assert_array_equal(queried[1],history[2])
    assert not np.array_equal(history[1],u)


def test_tiny_reference_transport_target_identity():
    from problems.kinetic.data import generate_reference
    from problems.kinetic.protocol import GRID
    c=dict(GRID,Nx=16,Nv=32,dx=.125,dt=.0001,intervals=2)
    row=dict(family='I',ic_parameters=json.dumps(dict(rho0=1,u0=0,theta0=1,A=0,sigma=.1,mu=0)))
    a=generate_reference(row,grid_config=c)
    assert a['U_node'].shape==(3,3,16)
    np.testing.assert_array_equal(a['R_eff'],a['G_kin_eff']-a['G_macro_eff'])
    np.testing.assert_allclose(a['U_node'],np.broadcast_to(np.array([1,0,.5])[None,:,None],(3,3,16)),atol=1e-11)


def test_selection_ties_and_coupled_boundary_policy():
    from problems.kinetic.protocol import selection_gate
    from problems.kinetic.selection import assess, expand
    result=selection_gate({'b':dict(J_final=1,J_tr=1.1),'a':dict(J_final=1,J_tr=1.1)})
    assert result['selected']=='a'
    result=selection_gate({'a':dict(J_final=1,J_tr=2),'b':dict(J_final=1.1,J_tr=1.1)})
    assert result['selected'] is None
    grids={'U30':dict(aq=[0,1,2],bq=[0,1,2],round=0)}
    rows=[dict(support='U30_'+c,aq=a,bq=b,candidate_id=f'{c}-{a}-{b}',eligible=True,J_final=1 if (a,b)==((0,1) if c=='mixed' else (1,1)) else 5) for c in ['I_only','mixed'] for a in range(3) for b in range(3)]
    checked=assess(rows,grids)
    assert checked['U30']['sides']==['lower-a']
    assert expand(grids,checked)['U30']['aq']==list(range(-4,3))


def test_missing_input_writes_failure_record_without_downstream_outputs(tmp_path):
    dest=tmp_path/'run'
    result=subprocess.run([sys.executable,'run_case.py','kinetic','--paper-case','kinetic-global-type-i',
                           '--data-root',str(tmp_path/'missing'),'--output',str(dest)],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==2,result.stderr
    manifest=json.loads((dest/'run_manifest.json').read_text())
    assert manifest['run_status']=='blocked_or_failed' and 'Missing' in manifest['reason']
    assert (dest/'resolved_config.json').is_file() and (dest/'data_manifest.json').is_file()
    assert not (dest/'test_metrics.json').exists() and not (dest/'validation_results.csv').exists()


def test_input_hash_and_root_containment(tmp_path):
    from interface.outputs import Run,inputs
    c=resolve('l63',paper_case='l63-rbf-rk4-cubic-n512',operational={'output':str(tmp_path/'run'),'data_root':str(tmp_path),'workflow':'render'})
    r=Run(c);p=tmp_path/(c['paper_case_id']+'.json')
    (tmp_path/'sample.npz').write_bytes(b'not numerical data')
    bundle=dict(scientific_sha256=c['scientific_sha256'],files=dict(scores=dict(path='sample.npz',sha256='0'*64)))
    p.write_text(json.dumps(bundle))
    with pytest.raises(ValueError,match='checksum mismatch'):inputs(c,r,['scores'])
    bundle['files']['scores']['path']='../escape.npz';p.write_text(json.dumps(bundle))
    with pytest.raises(ValueError,match='escapes'):inputs(c,r,['scores'])


def test_unsupported_science_override_rejected():
    with pytest.raises(ValueError,match='No dispatcher implementation'):
        resolve('ks',paper_case='ks-rbf-m0-n4096',overrides={'pca':True})
    with pytest.raises(ValueError,match='No dispatcher implementation'):
        resolve('kinetic',paper_case='kinetic-global-type-i',overrides={'pca_whiten':True})


def test_cpu_render_dispatch_records_loaded_scores(tmp_path):
    from interface.outputs import sha256
    case='l63-rbf-rk4-cubic-n512'
    config=resolve('l63',paper_case=case,operational={'workflow':'render'})
    np.savez(tmp_path/'scores.npz',vpts_all=np.ones((10,500)))
    bundle=dict(scientific_sha256=config['scientific_sha256'],files=dict(scores=dict(path='scores.npz',sha256=sha256(tmp_path/'scores.npz'))))
    (tmp_path/(case+'.json')).write_text(json.dumps(bundle))
    dest=tmp_path/'render'
    result=subprocess.run([sys.executable,'run_case.py','l63','--paper-case',case,'--workflow','render',
                           '--data-root',str(tmp_path),'--output',str(dest)],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr+result.stdout
    manifest=json.loads((dest/'run_manifest.json').read_text())
    assert manifest['run_status']=='complete'
    assert manifest['stages'][0]['status']=='loaded'
    metrics=json.loads((dest/'test_metrics.json').read_text())
    assert metrics['mean']==1 and metrics['std']==0 and metrics['ddof']==1
    assert not (dest/'selected_model.json').exists()


def test_saved_kinetic_figure_rendering_on_cpu(tmp_path):
    from interface.outputs import Run
    from problems.kinetic.plotting import render
    c=resolve('kinetic',paper_case='kinetic-global-type-i-plus-type-ii',operational={'output':str(tmp_path/'run'),'workflow':'render'})
    run=Run(c)
    u=np.broadcast_to(np.array([1.,0.,.5])[None,:,None],(2,3,16)).copy()
    archive=tmp_path/'rollout.npz'
    np.savez(archive,predicted_U=u,truth_U=u,times=np.array([0.,.1]),trajectory_id=np.array('validation_II_VIIB_06'))
    render(archive,run,'validation_II_VIIB_06')
    assert (run.path/'figures/profiles.pdf').stat().st_size>0
    assert (run.path/'figures/temperature.pdf').stat().st_size>0
    manifest=json.loads((run.path/'figure_manifest.json').read_text())
    assert manifest['trajectory_id']=='validation_II_VIIB_06'
