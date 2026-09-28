"""CPU renderers for saved KS validation surfaces and test trajectories."""
import json
import numpy as np
from interface.outputs import inputs, write_json, sha256


def render(config,run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    kind=config['operational']['render_kind']
    role='validation_surface' if kind=='validation_heatmap' else 'rollout'
    files,_=inputs(config,run,[role])
    with np.load(files[role],allow_pickle=False) as z: a={k:z[k] for k in z.files}
    if kind=='validation_heatmap':
        scores=a['vpt'];eps=a['epsilon'];ridge=a['lambda_reg']
        if scores.shape!=(10,169) or eps.shape!=scores.shape or ridge.shape!=scores.shape:
            raise ValueError('Paper Figure 1 expects ten 13 x 13 validation surfaces')
        index=3
        e=np.unique(eps[index]);r=np.unique(ridge[index])
        if len(e)!=13 or len(r)!=13: raise ValueError('Surface is not a 13 x 13 grid')
        grid=np.full((len(e),len(r)),np.nan)
        for e0,r0,value in zip(eps[index],ridge[index],scores[index]):
            grid[np.searchsorted(e,e0),np.searchsorted(r,r0)]=value
        if not np.isfinite(grid).all(): raise ValueError('Nonfinite or incomplete heatmap')
        fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True)
        im=ax.pcolormesh(np.log10(r),np.log10(e),grid,shading='auto')
        ax.set(xlabel='log10 lambda_reg',ylabel='log10 epsilon',title='Mean validation VPT, model index 3')
        fig.colorbar(im,ax=ax);name='validation_heatmap.pdf';details=dict(model_index=index,validation_rollouts=5)
    elif kind=='rollout':
        from .data import pos6_to_phys14
        truth=a['truth_real12'];pred=a['predicted_real12'];times=a['times'];scale=a['state_scale_real12']
        if truth.shape!=pred.shape or truth.ndim!=2 or truth.shape[1]!=12 or len(times)!=len(truth) or scale.shape!=(12,) or np.any(scale<=0):
            raise ValueError('Invalid saved spec12 rollout/normalization')
        metadata=json.loads(str(a['metadata_json']))
        if metadata.get('paper_case_id')!=config['paper_case_id'] or 'model_index' not in metadata or 'test_segment' not in metadata:
            raise ValueError('Saved model/segment selection must identify this paper case')
        true_field=pos6_to_phys14(truth[:,0::2]+1j*truth[:,1::2]);pred_field=pos6_to_phys14(pred[:,0::2]+1j*pred[:,1::2])
        nmse=np.mean(((pred-truth)/scale)**2,axis=1)
        fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
        for ax,field,title in zip(axes[:2],[true_field,pred_field],['reference','KRR']):
            im=ax.pcolormesh(np.arange(14),times,field,shading='auto');ax.set(title=title,xlabel='spatial grid',ylabel='t');fig.colorbar(im,ax=ax)
        axes[2].plot(times,nmse);axes[2].axhline(.25,color='k',linestyle='--');axes[2].set(xlabel='t',ylabel='normalized MSE')
        name='typical_rollout.pdf';details=metadata
    else: raise ValueError('Unsupported KS render kind')
    fig.savefig(run.path/'figures'/name);plt.close(fig)
    write_json(run.path/'figure_manifest.json',dict(input_sha256=sha256(files[role]),files=['figures/'+name],selection=details,
                 layout='public comparison rendering; manuscript byte identity not claimed'))
    run.stage('figures','computed',numerical_inputs='loaded')
