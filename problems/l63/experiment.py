"""Audited CUDA validation arithmetic; configured by pipeline."""
import numpy as np
import cupy as cp
from .src.krr_model import Modeler

def free_gpu_memory():
    import gc
    from multiprocessing import active_children

    active = active_children()
    for child in active:
        child.terminate()
    for child in active:
        child.join()

    for obj in gc.get_objects():
        if isinstance(obj, cp.ndarray):
            del obj

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    gc.collect()
    cp.cuda.runtime.deviceSynchronize()


def batched_compute_inner_cv(
    k_train,
    k_valid,
    epsilon_array,
    lambda_array,
    mode,
    device
    ):
    with cp.cuda.Device(device):

        local_opts = dict(opts)
        local_opts["data"] = k_train
        model = Modeler(**local_opts)

        X = cp.asarray(model.inp, dtype=cp.float64)
        n = cp.sum(X * X, axis=1)

        distance_matrix = X @ X.T
        distance_matrix *= -2.0
        distance_matrix += n[:, None]
        distance_matrix += n[None, :]
        cp.maximum(distance_matrix, 0.0, out=distance_matrix)
        cp.sqrt(distance_matrix, out=distance_matrix)
        cp.fill_diagonal(distance_matrix, 0.0)
        distance_matrix = 0.5 * (distance_matrix + distance_matrix.T)

        tau_f_array = np.zeros(cv_trials_per_device, dtype=float)
        vpt_array   = np.zeros(cv_trials_per_device, dtype=float)

        validation_indices = validation_offsets
        for j in range(validation_repeats):
            j_valid = k_valid[
                validation_indices[j] : validation_indices[j] + validation_horizon
            ]

            chunk_size = n_chunk_lst[device]

            for start in range(0, cv_trials_per_device, chunk_size):
                end = min(cv_trials_per_device, start + chunk_size)

                eps_chunk = epsilon_array[start:end]
                lam_chunk = lambda_array[start:end]

                model.fit_model(eps_chunk, lam_chunk, mode, distance_matrix=distance_matrix)
                _vpt_chunk, _tau_f_chunk = model.get_performance(
                    j_valid,
                    dt=dt,
                    Lyapunov_time=1 / Lyapunov_exp,
                    error_threshold=error_threshold,
                )

                _vpt_chunk   = cp.asnumpy(_vpt_chunk)
                _tau_f_chunk = cp.asnumpy(_tau_f_chunk)

                vpt_array[start:end]   += _vpt_chunk
                tau_f_array[start:end] += _tau_f_chunk

            free_gpu_memory()

        vpt_array   /= validation_repeats
        tau_f_array /= validation_repeats

        return tau_f_array, vpt_array
