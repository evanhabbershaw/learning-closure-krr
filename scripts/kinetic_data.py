"""Generate or preprocess declared training/validation data (potentially costly).

Run from the repository root with: python -m scripts.kinetic_data --help
"""
import argparse
import json
from pathlib import Path
import numpy as np
from interface.outputs import sha256, write_json
from problems.kinetic.data import members, generate_reference, load_bank
from problems.kinetic.preprocessing import fit_preprocessing, prepare_centers
from problems.kinetic.fit import fit_baseline


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['generate','preprocess','centers']);p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--trajectory-id')
    a=p.parse_args()
    training=members('training');load=lambda row:load_bank(a.bank,row)
    if a.stage=='generate':
        rows=[r for role in ['training','validation'] for r in members(role) if r['trajectory_id']==a.trajectory_id]
        if len(rows)!=1: p.error('Supply exactly one known training/validation --trajectory-id')
        row=rows[0];dest=a.bank/row['role']/(row['trajectory_id']+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists(): p.error('Output exists; refusing replacement')
        arrays=generate_reference(row)
        with dest.open('xb') as f: np.savez_compressed(f,**arrays)
        write_json(dest.with_suffix('.json'),dict(member=row,sha256=sha256(dest),stage='computed'))
        return
    if a.output is None: p.error('--output required for preprocess/centers')
    a.output.mkdir(parents=True,exist_ok=False)
    norms,pca=fit_preprocessing([r for r in training if r['family']=='I'],load)
    np.savez(a.output/'normalizers.npz',U30_mean=norms['U454']['mean'],U30_scale=norms['U454']['scale'],DU27_mean=norms['DU454']['mean'],DU27_scale=norms['DU454']['scale'])
    np.savez(a.output/'global_pca.npz',**pca)
    if a.stage=='centers':
        bx,by,_=prepare_centers(training,load,'U30','I_only',norms,pca)
        baseline=fit_baseline(bx,by);np.save(a.output/'baseline_coefficients.npy',baseline.coefficients)
        for rep in ['global','U30','DU27']:
            for cond in ['I_only','mixed']:
                X,Y,identities=prepare_centers(training,load,rep,cond,norms,pca,baseline)
                prefix=rep+'_'+cond;np.save(a.output/(prefix+'_features.npy'),X);np.save(a.output/(prefix+'_targets.npy'),Y)
                write_json(a.output/(prefix+'_centers.json'),identities)
    write_json(a.output/'data_manifest.json',dict(inputs=[dict(member=r,sha256=sha256(a.bank/r['role']/(r['trajectory_id']+'.npz'))) for r in training],
               outputs={p.name:sha256(p) for p in a.output.iterdir() if p.is_file()},stage='computed'))

if __name__=='__main__': main()
