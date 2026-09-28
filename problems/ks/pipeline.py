"""Explicit KS CUDA validation, selected fitting and reduced ETDRK4 test."""
from interface.outputs import inputs, require_cuda, write_json, record_cuda, save_gpu_model


def run(config, run):
    if config['operational']['workflow']=='render' and config['operational']['render_kind']!='summary':
        from .plotting import render
        return render(config,run)
    if config['operational']['workflow']=='render':
        from problems.l63.reporting import render_scores
        return render_scores(config,run)
    import numpy as np
    from .data import load
    from problems.l63.reporting import save_validation, summarize
    s=config['scientific'];workflow=config['operational']['workflow']
    files,_=inputs(config,run,['processed']+(['validation_scores'] if workflow=='replay' else []))
    arrays,v6,scale=load(files['processed'],s['split_index'])
    N,m=s['training_size'],s['memory_order'];length=N+m+1;split=s['split_index']
    if len(v6)<split+s['test_trials']*s['test_horizon']: raise ValueError('KS test pool is too short')
    cp=require_cuda(config['operational']['device'])
    record_cuda(run,cp,config['operational']['device'])
    from . import experiment as e
    settings=dict(ROLLOUT_REP='spec12',dt=s['dt'],alpha=s['alpha'],substeps=s['substeps'],Mquad=s['contour_points'],
                  TARGET_VAR='thetaHat',INPUT_SOURCES=('v','thetaHat'),INPUT_SCALING_MODE='identity',TARGET_SCALING_MODE='identity',TARGET_MODE='direct',
                  MEMORY=m,INPUT_MODE='v_thetaHat',INTERP_DEGREE=s['closure_interpolation'],USE_PCA_INPUT=False,KRR_SOLVER_MODE=s['solver'],
                  ROLLOUT_STATE_SOURCE='v',ROLLOUT_THETA_SOURCE='thetaHat',state_scale_real12=scale,
                  VALIDATION_OFFSETS=np.array(s['validation_offsets']),validation_repeats=len(s['validation_offsets']),validation_horizon=s['validation_horizon'],
                  Lyapunov_exp=s['lyapunov_exponent'],error_threshold=s['error_threshold'],RUN_EIG_SOLVER_CHECK=False,SOURCE_DIM={'v':12,'thetaHat':12})
    e.__dict__.update(settings)
    device=config['operational']['device'];e.n_chunk_lst={device:s['candidate_chunk']}
    a,b=np.meshgrid(10**np.linspace(*s['epsilon_log10_bounds'],s['epsilon_count']),10**np.linspace(*s['lambda_log10_bounds'],s['lambda_count']),indexing='ij')
    eps=np.tile(a.ravel(),(s['models'],1));ridge=np.tile(b.ravel(),(s['models'],1));scores=np.empty_like(eps)
    mode='diffusion' if s['kernel']=='diffusion_maps' else 'rbf'
    if workflow=='replay':
        with np.load(files['validation_scores'],allow_pickle=False) as z: scores=z['vpt']
        if scores.shape!=eps.shape: raise ValueError('KS validation score shape mismatch')
    else:
        for i,start in enumerate(s['training_starts']):
            train={k:v[start:start+length] for k,v in arrays.items()}
            valid={k:v[start+length:start+length+s['validation_length']] for k,v in arrays.items()}
            _,scores[i],_=e.batched_compute_inner_cv(train,valid,v6[start+length:start+length+s['validation_length']],eps[i],ridge[i],mode,device)
    if np.any(np.all(~np.isfinite(scores),axis=1)): raise ValueError('No eligible candidate for one model')
    save_validation(run.path/'validation_results.csv',eps,ridge,scores)
    run.stage('validation','loaded' if workflow=='replay' else 'computed',rollouts_per_candidate=len(s['validation_offsets']))
    best=np.argmax(np.where(np.isfinite(scores),scores,-np.inf),axis=1);selected=[];vpts=[]
    trials,steps=s['test_trials'],s['test_horizon']
    test={k:v[split:split+trials*steps].reshape(trials,steps,-1) for k,v in arrays.items()}
    test_v6=v6[split:split+trials*steps].reshape(trials,steps,-1)
    stepper=e.ReducedKSForcedETDRK4Stepper(rep='spec12',dt=s['dt'],alpha=s['alpha'],substeps=s['substeps'],M=s['contour_points'])
    for i,(start,j) in enumerate(zip(s['training_starts'],best)):
        train={k:v[start:start+length] for k,v in arrays.items()}
        X,Y,scalers,target_scaler=e.build_train_pairs_general(train,train['thetaHat'],input_sources=('v','thetaHat'),input_scaling_mode='identity',target_scaling_mode='identity',target_mode='direct',memory=m)
        model=e.Modeler(map_type='direct',pipeline='single',inp=X,out=Y,std_nmse_ref=Y,solver=s['selected_solver'])
        distance=e.compute_distance_matrix(cp.asarray(model.inp,dtype=cp.float64))
        model.fit_model(eps[i,j],ridge[i,j],mode,distance_matrix=distance)
        values=[]
        for begin in range(0,trials,s['candidate_chunk']):
            sl=slice(begin,begin+s['candidate_chunk'])
            v,_=e.get_vpt_learned_multipoint_single(model,test['v'][sl],test['thetaHat'][sl],test_v6[sl],stepper,scalers,target_scaler,scale,
                    pca_input=None,target_mode='direct',input_sources=('v','thetaHat'),rollout_state_source='v',rollout_theta_source='thetaHat',memory=m,
                    interp_degree=s['closure_interpolation'],dt=s['dt'],Lyapunov_exp=s['lyapunov_exponent'],error_threshold=s['error_threshold'])
            values.extend(np.asarray(v).reshape(-1).tolist())
        vpts.append(values);selected.append(dict(model=i,candidate=int(j),epsilon=float(eps[i,j]),lambda_reg=float(ridge[i,j])))
        selected[-1]['model_arrays_sha256']=save_gpu_model(run.path/'models'/f'model_{i:02d}.npz',model,cp)
        e.free_gpu_memory()
    write_json(run.path/'selected_model.json',dict(models=selected,selection='maximum validation VPT; first index on tie',fit_source='original training pairs'))
    write_json(run.path/'test_metrics.json',summarize(vpts,s['summary_ddof']))
    np.savez(run.path/'models/selected_scores.npz',vpts_all=np.asarray(vpts),validation_vpt=scores)
    run.stage('selected_fit_and_test','computed',coefficient_serialization='NPZ fitted arrays, densities and metric normalization')
