"""Saved-score reductions; no model fitting on import."""
import csv
import numpy as np
from interface.outputs import write_json


def summarize(scores, ddof):
    scores=np.asarray(scores,dtype=float)
    if scores.ndim != 2 or len(scores)<=ddof or not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError('Expected finite [model, test_segment] VPT scores')
    means=scores.mean(axis=1)
    return dict(mean=float(means.mean()),std=float(means.std(ddof=ddof)),ddof=ddof,
                model_means=means.tolist(),per_segment=scores.tolist(),metric='VPT')


def save_validation(path, eps, ridge, scores):
    with open(path,'w',newline='') as f:
        w=csv.writer(f);w.writerow(['model','candidate','epsilon','lambda_reg','score','metric','eligible','failure'])
        for i in range(len(scores)):
            for j in range(scores.shape[1]):
                finite=bool(np.isfinite(scores[i,j]))
                w.writerow([i,j,eps[i,j],ridge[i,j],scores[i,j] if finite else '', 'mean_validation_VPT',finite,'' if finite else 'nonfinite_score'])


def render_scores(config, run):
    from interface.outputs import inputs
    files,_=inputs(config,run,['scores'])
    with np.load(files['scores'],allow_pickle=False) as z: scores=z['vpts_all']
    s=config['scientific']
    if scores.shape!=(s['models'],s['test_trials']):
        raise ValueError('Saved score dimensions differ from the resolved experiment')
    summary=summarize(scores,config['scientific']['summary_ddof'])
    write_json(run.path/'test_metrics.json',summary)
    run.stage('test_metrics','loaded',source='scores',aggregation='computed from saved per-segment scores')
