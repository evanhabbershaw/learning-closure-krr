"""L63 raw-state block construction and frozen-bundle schema."""
import numpy as np


def blocks(train, training_size, starts, offsets=(353,314,81), horizon=1500, validation_length=3000):
    train=np.asarray(train)
    if train.ndim != 2 or train.shape[1] != 3 or not np.isfinite(train).all():
        raise ValueError('Expected finite raw (time, xyz) trajectory')
    if min(starts)<0 or max(starts)+training_size+validation_length>len(train):
        raise ValueError('Insufficient reference trajectory')
    full=np.stack([train[i:i+training_size+validation_length] for i in starts])
    pool=full[:,training_size:]
    valid=np.stack([pool[:,i:i+horizon] for i in offsets],axis=1)
    return dict(train_blocks=full[:,:training_size],validation_pool=pool,validation_rollouts=valid)


def load(path, s):
    with np.load(path,allow_pickle=False) as z:
        train=z['train_blocks'];valid=z['validation_pool']
    if train.shape != (s['models'],s['training_size'],3) or valid.shape != (s['models'],s['validation_length'],3):
        raise ValueError('L63 protocol bundle dimensions differ from config')
    if not np.isfinite(train).all() or not np.isfinite(valid).all(): raise ValueError('Nonfinite protocol')
    return train,valid
