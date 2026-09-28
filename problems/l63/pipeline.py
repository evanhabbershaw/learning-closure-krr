"""L63 CUDA fitting with explicit literal candidate schedules."""
from interface.outputs import inputs, require_cuda, write_json, record_cuda, save_gpu_model


def run(config, run):
    if config['operational']['workflow']=='render':
        from .reporting import render_scores
        return render_scores(config,run)
    import numpy as np
    from .protocol import load
    from .reporting import summarize, save_validation
    s=config['scientific']; workflow=config['operational']['workflow']
    files,_=inputs(config,run,['protocol','test','candidates']+(['validation_scores'] if workflow=='replay' else []))
    train,valid=load(files['protocol'],s)
    test=np.load(files['test'],allow_pickle=False)
    if test.shape != (s['test_trials'],s['test_horizon'],3): raise ValueError('L63 test shape mismatch; use time-major segments')
    with np.load(files['candidates'],allow_pickle=False) as z: eps=z['epsilon']; ridge=z['lambda_reg']
    if eps.shape != (s['models'],s['candidate_count']) or ridge.shape != eps.shape:
        raise ValueError('Literal L63 per-model candidate schedule shape mismatch')
    if not np.isfinite(eps).all() or not np.isfinite(ridge).all() or min(eps.min(),ridge.min())<=0:
        raise ValueError('Invalid candidate values')
    cp=require_cuda(config['operational']['device'])
    record_cuda(run,cp,config['operational']['device'])
    from . import experiment as e
    mode='diffusion' if s['kernel']=='diffusion_maps' else 'rbf'
    opts=dict(map_type='skip-connection',pipeline='batch',target='z',score_target='full',
              xy_integrator={'rk4_quadratic':'rk4_quad'}.get(s['stage_reconstruction'],s['stage_reconstruction']),
              sigma=s['sigma'],rho=s['rho'],verbose_perf=0)
    e.opts=opts;e.dt=s['dt'];e.Lyapunov_exp=s['lyapunov_exponent'];e.error_threshold=s['error_threshold']
    e.validation_horizon=s['validation_horizon'];e.validation_repeats=len(s['validation_offsets'])
    e.validation_offsets=s['validation_offsets'];e.cv_trials_per_device=s['candidate_count']
    device=config['operational']['device'];e.n_chunk_lst={device:s['candidate_chunk']}
    scores=np.empty_like(eps)
    if workflow=='replay':
        with np.load(files['validation_scores'],allow_pickle=False) as z: scores=z['vpt']
        if scores.shape != eps.shape: raise ValueError('Validation score shape mismatch')
    else:
        for i in range(s['models']):
            _,scores[i]=e.batched_compute_inner_cv(train[i],valid[i],eps[i],ridge[i],mode,device)
    if np.any(np.all(~np.isfinite(scores),axis=1)): raise ValueError('Model has no eligible validation candidate')
    save_validation(run.path/'validation_results.csv',eps,ridge,scores)
    run.stage('validation','loaded' if workflow=='replay' else 'computed',rollouts_per_candidate=len(s['validation_offsets']))
    best=np.argmax(np.where(np.isfinite(scores),scores,-np.inf),axis=1); selected=[]; vpts=[]
    for i,j in enumerate(best):
        model=e.Modeler(data=train[i],**opts)
        X=cp.asarray(model.inp,dtype=cp.float64);n=cp.sum(X*X,axis=1)
        distance=X@X.T;distance*=-2.;distance+=n[:,None];distance+=n[None,:]
        cp.maximum(distance,0.,out=distance);cp.sqrt(distance,out=distance);cp.fill_diagonal(distance,0.)
        distance=.5*(distance+distance.T)
        model.fit_model(eps[i,j],ridge[i,j],mode,distance_matrix=distance)
        values=[]
        for start in range(0,len(test),s['candidate_chunk']):
            v,_=model.get_performance(test[start:start+s['candidate_chunk']],dt=s['dt'],
                                     Lyapunov_time=1/s['lyapunov_exponent'],error_threshold=s['error_threshold'])
            values.extend(np.asarray(v).reshape(-1).tolist())
        vpts.append(values); selected.append(dict(model=i,candidate=int(j),epsilon=float(eps[i,j]),lambda_reg=float(ridge[i,j])))
        selected[-1]['model_arrays_sha256']=save_gpu_model(run.path/'models'/f'model_{i:02d}.npz',model,cp)
        e.free_gpu_memory()
    write_json(run.path/'selected_model.json',dict(models=selected,selection='maximum validation VPT; first candidate index on tie',fit_source='original training blocks'))
    write_json(run.path/'test_metrics.json',summarize(vpts,s['summary_ddof']))
    np.savez(run.path/'models/selected_scores.npz',vpts_all=np.asarray(vpts),validation_vpt=scores)
    run.stage('selected_fit_and_test','computed',coefficient_serialization='NPZ fitted arrays, densities and metric normalization')
