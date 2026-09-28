"""Frozen selected-model CPU fit/replay and evaluation; no protected tuning."""
import json
from interface.outputs import inputs, write_json, sha256


def run(config, run):
    import numpy as np
    from .kinetic_closure.final_learning_krr_20260920_v1 import Model
    from .kinetic_closure.sbgk_grid import make_sbgk_grid
    from .kinetic_closure.sbgk_butcher import get_butcher
    from .rollout import predict_closure, rollout
    from .evaluate import errors, aggregate
    from .data import members
    from .selection import BASELINE
    s=config['scientific'];workflow=config['operational']['workflow'];rep=s['representation']
    if workflow=='render':
        from .plotting import render
        files,_=inputs(config,run,['rollout'])
        return render(files['rollout'],run,s['figure_case'])
    roles=['features','normalizers','evaluation', 'targets' if workflow=='reproduce' else 'coefficients']
    if rep=='global': roles.append('pca')
    if rep=='DU27': roles.extend(['baseline_features','baseline_coefficients'])
    files,bundle=inputs(config,run,roles)
    if config['experiment_status']=='paper_case' and bundle.get('selected_model_id')!=s['selected_model_id']:
        raise ValueError('Input manifest must identify the frozen selected_model_id')
    if (s['Nx'],s['Nv'],s['intervals'],s['dt'],s['dx']) != (256,128,455,.000439453125,2/256):
        raise ValueError('Dispatcher supports the paper grid; use the explicit array API for custom meshes')
    X=np.load(files['features'],allow_pickle=False)
    dim={'global':768,'U30':30,'DU27':27}[rep];outdim=768 if rep=='global' else 3
    if X.shape!=(s['centers'],dim): raise ValueError('Support feature dimensions differ from config')
    with np.load(files['normalizers'],allow_pickle=False) as z:
        norms={'U454':dict(mean=z['U30_mean'],scale=z['U30_scale']), 'DU454':dict(mean=z['DU27_mean'],scale=z['DU27_scale'])}
    for norm in norms.values():
        if norm['mean'].shape!=(3,) or norm['scale'].shape!=(3,) or not np.isfinite(norm['mean']).all() or not np.isfinite(norm['scale']).all() or np.any(norm['scale']<=0):
            raise ValueError('Invalid normalizer')
    pca=None
    if rep=='global':
        with np.load(files['pca'],allow_pickle=False) as z:
            pca={k:z[k] for k in ['U_mean','U_components','R_mean','R_components']}
        if any(pca[k].shape!=(768,768) for k in ['U_components','R_components']):
            raise ValueError('Full 768/768 unwhitened PCA required')
        for key in ['U_components','R_components']:
            if not np.allclose(pca[key]@pca[key].T,np.eye(768),rtol=0,atol=1e-11):
                raise ValueError('PCA must be orthonormal and unwhitened')
    baseline=None
    if rep=='DU27':
        binding=bundle.get('baseline',{})
        for k in ['support','N','kernel_epsilon','lambda_reg']:
            if binding.get(k)!=BASELINE[k]: raise ValueError('Residual requires the fixed Family-I U30 baseline binding')
        bx=np.load(files['baseline_features'],allow_pickle=False);bc=np.load(files['baseline_coefficients'],allow_pickle=False)
        if bx.shape!=(4032,30) or bc.shape!=(4032,3): raise ValueError('Fixed baseline shape mismatch')
        baseline=Model(bx,bc,BASELINE['kernel_epsilon'],BASELINE['lambda_reg'])
        if workflow=='reproduce' and bundle.get('residual_targets_baseline_sha256')!=sha256(files['baseline_coefficients']):
            raise ValueError('Residual targets must bind exactly these baseline coefficients')
    if workflow=='reproduce':
        from .fit import fit_ridges
        Y=np.load(files['targets'],allow_pickle=False)
        if Y.shape!=(len(X),outdim): raise ValueError('Target dimensions differ')
        model,health=next(fit_ridges(X,Y,s['kernel_epsilon'],[s['lambda_reg']]))
        if model is None: raise ValueError('Selected fit failed: '+str(health))
        np.save(run.path/'models/coefficients.npy',model.coefficients)
        run.stage('selected_fit','computed',health=health,selection='loaded frozen hyperparameters; no validation search')
    else:
        coeff=np.load(files['coefficients'],allow_pickle=False)
        if coeff.shape!=(len(X),outdim) or not np.isfinite(coeff).all(): raise ValueError('Coefficient shape/value mismatch')
        model=Model(X,coeff,s['kernel_epsilon'],s['lambda_reg']);run.stage('selected_model','loaded')
    write_json(run.path/'selected_model.json',dict(selected_model_id=s['selected_model_id'],kernel_epsilon=s['kernel_epsilon'],lambda_reg=s['lambda_reg'],
              support_sha256=sha256(files['features']),baseline=BASELINE if rep=='DU27' else None,
              baseline_coefficients_sha256=sha256(files['baseline_coefficients']) if rep=='DU27' else None,
              selection='frozen before evaluation',historical_coefficients_authenticated=False))
    with np.load(files['evaluation'],allow_pickle=False) as z:
        truth=z['truth_U'];R0=z['initial_correction'];times=z['times'];ids=z['trajectory_ids'].tolist()
    if truth.shape!=(len(ids),456,3,256) or R0.shape!=(len(ids),3,256) or not np.array_equal(times,np.arange(456)*s['dt']):
        raise ValueError('Evaluation bundle shape/time mismatch')
    if not np.isfinite(truth).all() or not np.isfinite(R0).all():
        raise ValueError('Nonfinite evaluation data')
    known={r['trajectory_id']:r for role in ['validation','protected_test'] for r in members(role)}
    if len(set(ids))!=len(ids) or any(i not in known for i in ids): raise ValueError('Unknown or duplicate evaluation member')
    protected=[r['trajectory_id'] for r in members('protected_test')]
    if any(known[i]['role']=='protected_test' for i in ids) and set(ids)!=set(protected):
        raise ValueError('Protected comparison requires all 48 frozen members')
    grid=make_sbgk_grid(256,128,-1,1,-8,8,t_final=.2,theta_x=2,theta_v=2);butcher=get_butcher(2)
    results=[]
    for i,sid in enumerate(ids):
        query=lambda state:predict_closure(model,rep,state,norms,pca,baseline,dx=s['dx'])
        predicted,health=rollout(truth[i,0],R0[i],query,butcher,grid,s['dt'],s['intervals'])
        row=dict(trajectory_id=sid,family=known[sid]['family'],**health)
        if health['health']=='PASS':
            metric,history=errors(predicted,truth[i]);row.update(metric)
            np.savez_compressed(run.path/'models'/(sid+'.npz'),predicted_U=predicted,truth_U=truth[i],times=times,error=history,trajectory_id=np.array(sid))
        results.append(row)
    metrics=dict(per_trajectory=results,failures=sum(r['health']!='PASS' for r in results),summary_ddof=0)
    if set(ids)==set(protected) and metrics['failures']==0: metrics['protected_families']=aggregate(results)
    write_json(run.path/'test_metrics.json',metrics)
    run.stage('evaluation','computed',members=ids)
    if metrics['failures']: raise ValueError('One or more macroscopic rollouts failed; see test_metrics.json')
