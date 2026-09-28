"""Coupled I/mixed validation grids and deterministic boundary policy."""
import math
from .protocol import require, digest
VERSION = "kinetic_final_validation_20260922_v1"
SUPPORTS = tuple(r+"_"+c for r in ("global", "U30", "DU27") for c in ("I_only", "mixed"))
BOUNDS = {"global": (-6,10,-24,-8), "U30": (-4,12,-28,-12), "DU27": (-6,10,-18,2)}
BASELINE = dict(support="U30_I_only", N=4032, kernel_epsilon=2953.231481793101, lambda_reg=1e-6)

def initial_grids():
    return {rep: dict(aq=list(range(a0,a1+1)), bq=list(range(b0,b1+1)), round=0)
            for rep,(a0,a1,b0,b1) in BOUNDS.items()}


def candidates(grids, references):
    require(set(references)==set(SUPPORTS), 'Six support-specific epsilon_50 values required')
    result=[]
    for support in SUPPORTS:
        rep, condition = support.split('_',1)
        e50=float(references[support]); require(math.isfinite(e50) and e50>0, 'Invalid epsilon_50')
        for aq in grids[rep]['aq']:
            for bq in grids[rep]['bq']:
                identity=dict(version=VERSION,support=support,aq=aq,bq=bq,epsilon_50=e50)
                result.append(dict(**identity,candidate_id=support+'_'+digest(identity)[:24],
                                   representation=rep,condition=condition,epsilon_offset=aq/4,
                                   kernel_epsilon=e50*10.**(aq/4),log10_lambda_reg=bq/4,lambda_reg=10.**(bq/4)))
    return result


def epsilon_tasks(rows):
    keys=sorted({(r['support'],r['aq']) for r in rows})
    result=[]
    for support,aq in keys:
        group=sorted((r for r in rows if (r['support'],r['aq'])==(support,aq)),key=lambda r:r['bq'])
        identity=dict(support=support,aq=aq,candidate_ids=[r['candidate_id'] for r in group])
        result.append(dict(**identity,task_id=support+'_'+digest(identity)[:24],
                           epsilon_offset=aq/4,kernel_epsilon=group[0]['kernel_epsilon'],candidates=group))
    return result


def condition_boundary(rows, grid):
    eligible=[r for r in rows if r['eligible']]
    if not eligible: return dict(status='STOP_HUMAN_REVIEW',reason='no eligible candidate',sides=[],component=[])
    best=min(eligible,key=lambda r:(r['J_final'],r['candidate_id']))
    near={(r['aq'],r['bq']):r for r in eligible if r['J_final']<=1.10*best['J_final']}
    seen={(best['aq'],best['bq'])}; todo=list(seen)
    while todo:
        a,b=todo.pop()
        for p in ((a-1,b),(a+1,b),(a,b-1),(a,b+1)):
            if p in near and p not in seen: seen.add(p); todo.append(p)
    sides=[]
    for side,axis,value in [('lower-a',0,min(grid['aq'])),('upper-a',0,max(grid['aq'])),
                            ('lower-b',1,min(grid['bq'])),('upper-b',1,max(grid['bq']))]:
        if any(p[axis]==value for p in seen): sides.append(side)
    boundary=[near[p]['candidate_id'] for p in sorted(seen) if p[0] in (min(grid['aq']),max(grid['aq'])) or p[1] in (min(grid['bq']),max(grid['bq']))]
    return dict(status='ASSESSED',minimizer=best['candidate_id'],sides=sides,
                component=[near[p]['candidate_id'] for p in sorted(seen)],boundary_candidates=boundary)


def assess(rows, grids):
    result={}
    for rep,g in grids.items():
        conditions={c:condition_boundary([r for r in rows if r['support']==rep+'_'+c],g) for c in ('I_only','mixed')}
        sides=sorted({s for v in conditions.values() for s in v['sides']})
        stop=any(v['status']=='STOP_HUMAN_REVIEW' for v in conditions.values()) or (bool(sides) and g['round']>=2)
        result[rep]=dict(conditions=conditions,sides=sides,round=g['round'],
                         status='STOP_HUMAN_REVIEW' if stop else ('EXPAND' if sides else 'BRACKETED'))
    return result


def expand(grids, assessment):
    require(not any(a['status']=='STOP_HUMAN_REVIEW' for a in assessment.values()), 'STOP_HUMAN_REVIEW')
    result={r:dict(aq=list(g['aq']),bq=list(g['bq']),round=g['round']) for r,g in grids.items()}
    require(any(a['status']=='EXPAND' for a in assessment.values()), 'All representations already bracketed')
    for rep,a in assessment.items():
        if a['status']!='EXPAND': continue
        g=result[rep]; require(g['round']<2,'Maximum two expansion rounds')
        for side in a['sides']:
            axis=side[-1]+'q'; values=g[axis]
            g[axis]=(list(range(min(values)-4,min(values)))+values if side.startswith('lower')
                     else values+list(range(max(values)+1,max(values)+5)))
        g['round']+=1
    return result
