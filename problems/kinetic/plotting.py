"""Profiles and space-time errors from a saved final validation rollout."""
import numpy as np
from interface.outputs import write_json, sha256
from .rollout import primitive_fields


def render(path, run, expected_case):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(path,allow_pickle=False) as z:
        p=z['predicted_U'];y=z['truth_U'];times=z['times'];case=str(z['trajectory_id'])
    if case!=expected_case: raise ValueError('Saved rollout is not the final validation figure case')
    pf,yf=primitive_fields(p),primitive_fields(y)
    x=-1+(np.arange(p.shape[-1])+.5)*2/p.shape[-1]
    fig,axes=plt.subplots(2,2,figsize=(9,6),constrained_layout=True)
    for i,(ax,label) in enumerate(zip(axes.flat,['density','velocity','temperature','energy'])):
        ax.plot(x,yf[-1,i],label='reference');ax.plot(x,pf[-1,i],'--',label='KRR');ax.set(xlabel='x',ylabel=label)
    axes.flat[0].legend();fig.savefig(run.path/'figures/profiles.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
    for ax,values,title in zip(axes,[yf[:,2],pf[:,2],abs(pf[:,2]-yf[:,2])],['reference temperature','KRR temperature','absolute error']):
        im=ax.pcolormesh(x,times,values,shading='auto');ax.set(title=title,xlabel='x',ylabel='t');fig.colorbar(im,ax=ax)
    fig.savefig(run.path/'figures/temperature.pdf');plt.close(fig)
    write_json(run.path/'figure_manifest.json',dict(trajectory_id=case,input_sha256=sha256(path),files=['figures/profiles.pdf','figures/temperature.pdf'],
               rendering='public comparison layout; not a byte-identical manuscript PDF'))
    run.stage('figures','computed',numerical_rollout='loaded')
