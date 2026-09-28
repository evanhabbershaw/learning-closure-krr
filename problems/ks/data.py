"""Aligned archive preprocessing on CPU; this is not a truth generator.

The input format has 48 FFT rows. Its relation to the manuscript reference
truncation 1 <= |k| <= 48 requires the original producer; see DATA.md.
"""
import numpy as np


def mode_to_index(k, N):
    return k if k>=0 else N+k


def pos6_to_phys14(vpos):
    spec=np.zeros((len(vpos),14),dtype=np.complex128)
    spec[:,1:7]=14*vpos;spec[:,8:]=14*np.conj(vpos[:,::-1])
    return np.real(np.fft.ifft(spec,axis=1)).astype(np.float64)


def preprocess(mat_path, output):
    from scipy.io import loadmat
    vv=np.asarray(loadmat(mat_path)['vv'],dtype=np.complex128)
    if vv.ndim!=2 or vv.shape[0]!=48 or not np.isfinite(vv).all():
        raise ValueError('Expected finite vv with 48 archive rows')
    a=(vv.T/48).astype(np.complex128);v6=a[:,1:7].copy()
    valid_modes=list(range(-23,0))+list(range(1,24))
    rhs=np.zeros((len(a),6),dtype=np.complex128);closure=np.zeros_like(rhs)
    for kk in range(1,7):
        full=np.zeros(len(a),dtype=np.complex128);resolved=np.zeros_like(full)
        for p in valid_modes:
            q=kk-p
            if q not in valid_modes: continue
            ap=a[:,mode_to_index(p,48)];aq=a[:,mode_to_index(q,48)]
            full+=ap*aq
            if abs(p)<=6 and abs(q)<=6: resolved+=ap*aq
        closure[:,kk-1]=full-resolved
        rhs[:,kk-1]=(-.5j*kk*np.sqrt(.085))*(full-resolved)
    with open(output,'xb') as f:
        np.savez_compressed(f,x14_phys=pos6_to_phys14(v6),theta14_phys=pos6_to_phys14(rhs),
                            v6_series=v6,theta_rhs6=rhs,f_closure6=closure,Nfull=48,Nphys=14,Kres=6,alpha=.085)


def load(path, split_index=400000):
    with np.load(path,allow_pickle=False) as z:
        v=z['v6_series'];theta=z['theta_rhs6'];x=z['x14_phys'];t=z['theta14_phys']
    if v.ndim!=2 or v.shape[1]!=6 or theta.shape!=v.shape or x.shape!=(len(v),14) or t.shape!=x.shape:
        raise ValueError('Aligned KS input shape mismatch')
    if not all(np.isfinite(a).all() for a in [v,theta,x,t]): raise ValueError('Nonfinite KS data')
    if not np.allclose(np.fft.fft(x[:1000],axis=1)[:,1:7]/14,v[:1000],rtol=1e-12,atol=1e-12):
        raise ValueError('KS state Fourier convention mismatch')
    if not np.allclose(np.fft.fft(t[:1000],axis=1)[:,1:7]/14,theta[:1000],rtol=1e-12,atol=1e-12):
        raise ValueError('KS closure Fourier convention mismatch')
    def real12(z):
        a=np.empty((len(z),12));a[:,0::2]=z.real;a[:,1::2]=z.imag;return a
    arrays={'v':real12(v),'thetaHat':real12(theta)}
    scale=np.std(arrays['v'][:split_index],axis=0);scale[scale==0]=1.
    return arrays,v,scale

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mat');p.add_argument('output')
    a=p.parse_args();preprocess(a.mat,a.output)
