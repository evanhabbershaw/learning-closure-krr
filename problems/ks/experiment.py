"""CUDA KRR and reduced ETDRK4 numerical workflow."""
import gc
import time
import numpy as np
import cupy as cp
from .src.krr_model import Modeler
from .src.ks_forced_etdrk4_stepper import ReducedKSForcedETDRK4Stepper
TIMING_SYNC_GPU = False

class TimingRecorder:
    def __init__(self):
        self.sections = {}

    def record(self, name, elapsed):
        print(f"TIMING {name} = {elapsed:.6f} seconds", flush=True)
        stats = self.sections.setdefault(
            name,
            {"total_seconds": 0.0, "calls": 0},
        )
        stats["total_seconds"] += float(elapsed)
        stats["calls"] += 1

    def as_dict(self):
        return {
            name: {
                "total_seconds": float(stats["total_seconds"]),
                "calls": int(stats["calls"]),
            }
            for name, stats in self.sections.items()
        }


def record_timing(timer, name, t0):
    if timer is not None:
        timer.record(name, time.perf_counter() - t0)


def maybe_sync_gpu():
    if TIMING_SYNC_GPU:
        cp.cuda.runtime.deviceSynchronize()


def free_gpu_memory():
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp.cuda.runtime.deviceSynchronize()


def compute_distance_matrix(X_cp: cp.ndarray) -> cp.ndarray:
    """
    Euclidean distance matrix, not squared.
    """
    n = cp.sum(X_cp * X_cp, axis=1)
    D = X_cp @ X_cp.T
    D *= -2.0
    D += n[:, None]
    D += n[None, :]
    cp.maximum(D, 0.0, out=D)
    cp.sqrt(D, out=D)
    cp.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)
    return D


def real12_from_pos_modes(vpos: cp.ndarray) -> cp.ndarray:
    """
    vpos: (..., 6) complex
    returns (..., 12) real:
        [Re v1, Im v1, ..., Re v6, Im v6]
    """
    out = cp.empty(vpos.shape[:-1] + (12,), dtype=cp.float64)
    out[..., 0::2] = cp.real(vpos)
    out[..., 1::2] = cp.imag(vpos)
    return out


def real12_from_complex6_np(z6: np.ndarray) -> np.ndarray:
    """
    z6: (..., 6) complex
    returns (..., 12) real:
        [Re z1, Im z1, ..., Re z6, Im z6]
    """
    z6 = np.asarray(z6, dtype=np.complex128)
    out = np.empty(z6.shape[:-1] + (12,), dtype=np.float64)
    out[..., 0::2] = np.real(z6)
    out[..., 1::2] = np.imag(z6)
    return out


def real12_from_complex6_cp(z6: cp.ndarray) -> cp.ndarray:
    """
    z6: (..., 6) complex CuPy array
    returns (..., 12) real:
        [Re z1, Im z1, ..., Re z6, Im z6]
    """
    z6 = cp.asarray(z6, dtype=cp.complex128)
    out = cp.empty(z6.shape[:-1] + (12,), dtype=cp.float64)
    out[..., 0::2] = cp.real(z6)
    out[..., 1::2] = cp.imag(z6)
    return out


def complex7_from_real12_cp(z12: cp.ndarray) -> cp.ndarray:
    """
    z12: (..., 12) real [Re z1, Im z1, ..., Re z6, Im z6]
    returns (..., 7) complex [0, z1, ..., z6]
    """
    z12 = cp.asarray(z12, dtype=cp.float64)

    if z12.shape[-1] != 12:
        raise ValueError(f"Expected real12 Fourier data, got last dim {z12.shape[-1]}.")

    z7 = cp.zeros(z12.shape[:-1] + (7,), dtype=cp.complex128)
    z7[..., 1:7] = z12[..., 0::2] + 1j * z12[..., 1::2]
    return z7


def real12_from_state_cp(x_state: cp.ndarray, stepper) -> cp.ndarray:
    """
    Convert current state in stepper.rep representation to real12 positive modes.
    For rep='phys14', this performs an FFT-based conversion.
    For rep='spec12', this is just real12 -> modes -> real12.
    """
# def real12_from_state_cp(x_phys14: cp.ndarray, stepper) -> cp.ndarray:
    # """
    # Convert current phys14 state to real12 Fourier modes.
    # """
    v_pos = stepper.state_to_modes(x_state)[..., 1:7]
    return real12_from_pos_modes(v_pos)


class MinMaxPM1Scaler:
    """
    Columnwise min-max scaler mapping each feature to [-1,1].

    Fit on training data only.
    """

    def __init__(self, tiny: float = 1e-14):
        self.tiny = float(tiny)
        self.xmin = None
        self.xmax = None
        self.xrange = None
        self.constant = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)

        self.xmin = np.min(X, axis=0)
        self.xmax = np.max(X, axis=0)
        self.xrange = self.xmax - self.xmin

        self.constant = np.abs(self.xrange) < self.tiny
        self.xrange[self.constant] = 1.0

        return self

    def transform_np(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xs = 2.0 * (X - self.xmin) / self.xrange - 1.0
        Xs[..., self.constant] = 0.0
        return Xs

    def inverse_np(self, Xs):
        Xs = np.asarray(Xs, dtype=np.float64)
        X = 0.5 * (Xs + 1.0) * self.xrange + self.xmin
        X[..., self.constant] = self.xmin[self.constant]
        return X

    def transform_cp(self, X):
        X = cp.asarray(X, dtype=cp.float64)
        xmin = cp.asarray(self.xmin, dtype=cp.float64)
        xrange = cp.asarray(self.xrange, dtype=cp.float64)
        constant = cp.asarray(self.constant)

        Xs = 2.0 * (X - xmin) / xrange - 1.0
        Xs[..., constant] = 0.0
        return Xs

    def inverse_cp(self, Xs):
        Xs = cp.asarray(Xs, dtype=cp.float64)
        xmin = cp.asarray(self.xmin, dtype=cp.float64)
        xrange = cp.asarray(self.xrange, dtype=cp.float64)
        constant = cp.asarray(self.constant)

        X = 0.5 * (Xs + 1.0) * xrange + xmin
        X[..., constant] = xmin[constant]
        return X


class IdentityScaler:
    """
    No-op scaler.

    This has the same interface as MinMaxPM1Scaler, but it does not
    transform the data. Use this when we want PCA/KRR to see the raw
    phys14 coordinates directly.
    """

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.shape_ = X.shape[1:]
        return self

    def transform_np(self, X):
        return np.asarray(X, dtype=np.float64)

    def inverse_np(self, X):
        return np.asarray(X, dtype=np.float64)

    def transform_cp(self, X):
        return cp.asarray(X, dtype=cp.float64)

    def inverse_cp(self, X):
        return cp.asarray(X, dtype=cp.float64)


class ZScoreScaler:
    """
    Columnwise z-score scaler.

    For each feature j,
        X_scaled[:, j] = (X[:, j] - mean_j) / std_j.

    Fit on training data only.

    (This is for z-score normalization before PCA)
    """

    def __init__(self, tiny: float = 1e-14, ddof: int = 0):
        self.tiny = float(tiny)
        self.ddof = int(ddof)
        self.mean = None
        self.std = None
        self.constant = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0, ddof=self.ddof)

        self.constant = np.abs(self.std) < self.tiny
        self.std[self.constant] = 1.0

        return self

    def transform_np(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xs = (X - self.mean) / self.std
        Xs[..., self.constant] = 0.0
        return Xs

    def inverse_np(self, Xs):
        Xs = np.asarray(Xs, dtype=np.float64)
        X = Xs * self.std + self.mean
        X[..., self.constant] = self.mean[self.constant]
        return X

    def transform_cp(self, X):
        X = cp.asarray(X, dtype=cp.float64)
        mean = cp.asarray(self.mean, dtype=cp.float64)
        std = cp.asarray(self.std, dtype=cp.float64)
        constant = cp.asarray(self.constant)

        Xs = (X - mean) / std
        Xs[..., constant] = 0.0
        return Xs

    def inverse_cp(self, Xs):
        Xs = cp.asarray(Xs, dtype=cp.float64)
        mean = cp.asarray(self.mean, dtype=cp.float64)
        std = cp.asarray(self.std, dtype=cp.float64)
        constant = cp.asarray(self.constant)

        X = Xs * std + mean
        X[..., constant] = mean[constant]
        return X


def make_scaler_from_mode(scaling_mode):
    if scaling_mode == "identity":
        return IdentityScaler()
    elif scaling_mode == "minmax":
        return MinMaxPM1Scaler()
    elif scaling_mode == "zscore":
        return ZScoreScaler(ddof=0)
    else:
        raise ValueError(
            "Scaling mode must be one of 'identity', 'minmax', or 'zscore'. "
            f"Got {scaling_mode!r}."
        )


def build_raw_memory_input(
    arrays_by_source,
    *,
    input_sources,
    memory: int,
):
    """
    Build raw KRR input matrix from arbitrary source arrays.

    arrays_by_source[src] has shape (T, d_src).

    Output layout:
        [src1_n, src1_{n-1}, ..., src1_{n-m},
         src2_n, src2_{n-1}, ..., src2_{n-m},
         ...]
    """
    T = None
    blocks = []

    for src in input_sources:
        A = np.asarray(arrays_by_source[src], dtype=np.float64)

        if T is None:
            T = A.shape[0]
        elif A.shape[0] != T:
            raise ValueError("All source arrays must have the same time length.")

        for lag in range(memory + 1):
            blocks.append(A[memory - lag : T - 1 - lag])

    if len(blocks) == 0:
        raise ValueError("input_sources cannot be empty.")

    return np.concatenate(blocks, axis=1).astype(np.float64)


def build_scaled_memory_input(
    source_arrays,
    input_sources,
    input_scalers,
    *,
    memory,
):
    """
    Build the scaled memory input matrix for KRR.

    source_arrays:
        dict mapping source names to arrays, e.g.
            source_arrays["x"]        shape (T, 14)
            source_arrays["theta"]    shape (T, 14)
            source_arrays["v"]        shape (T, 12)
            source_arrays["thetaHat"] shape (T, 12)

    input_sources:
        tuple such as ("x",), ("v",), ("x","theta"), ("v","thetaHat")

    input_scalers:
        dict mapping source names to fitted scalers.

    Output layout:
        [src1_n, src1_{n-1}, ..., src1_{n-m},
         src2_n, src2_{n-1}, ..., src2_{n-m}, ...]
    """
    if len(input_sources) == 0:
        raise ValueError("input_sources must contain at least one source.")

    T = None
    blocks = []

    for src in input_sources:
        if src not in source_arrays:
            raise KeyError(f"Missing source_arrays[{src!r}].")

        if src not in input_scalers:
            raise KeyError(f"Missing input_scalers[{src!r}].")

        arr = np.asarray(source_arrays[src], dtype=np.float64)

        if T is None:
            T = arr.shape[0]
        elif arr.shape[0] != T:
            raise ValueError(
                f"All source arrays must have same time length. "
                f"Got {src} with length {arr.shape[0]}, expected {T}."
            )

        arr_s = input_scalers[src].transform_np(arr)

        for lag in range(memory + 1):
            blocks.append(arr_s[memory - lag : T - 1 - lag])

    X = np.concatenate(blocks, axis=1)
    return X.astype(np.float64)


def build_train_pairs_general(
    arrays_by_source,
    target_array,
    *,
    input_sources,
    input_scaling_mode,
    target_scaling_mode,
    target_mode: str,
    memory: int,
    timer=None,
    timing_prefix="train_pairs",
):
    """
    General supervised KRR pair builder.

    X is built from INPUT_MODE sources.
    Y is built from TARGET_VAR.

    Scaling convention:
        input scaler is fit on X_train_raw
        target scaler is fit on target_array
    """
    target_array = np.asarray(target_array, dtype=np.float64)

    t0 = time.perf_counter()
    input_scalers = {
        src: make_scaler_from_mode(input_scaling_mode).fit(arrays_by_source[src])
        for src in input_sources
    }
    record_timing(timer, f"{timing_prefix}.input_scaler_fit", t0)

    t0 = time.perf_counter()
    target_scaler = make_scaler_from_mode(target_scaling_mode).fit(target_array)
    record_timing(timer, f"{timing_prefix}.target_scaler_fit", t0)

    t0 = time.perf_counter()
    X = build_scaled_memory_input(
        arrays_by_source,
        input_sources=input_sources,
        input_scalers=input_scalers,
        memory=memory,
    )
    record_timing(timer, f"{timing_prefix}.memory_input_build", t0)

    t0 = time.perf_counter()
    target_s = target_scaler.transform_np(target_array)
    record_timing(timer, f"{timing_prefix}.target_scaling", t0)

    # X_raw = build_raw_memory_input(
    #     arrays_by_source,
    #     input_sources=input_sources,
    #     memory=memory,
    # )

    # input_scaler = make_scaler_from_mode(input_scaling_mode).fit(X_raw)
    # target_scaler = make_scaler_from_mode(target_scaling_mode).fit(target_array)

    # X = input_scaler.transform_np(X_raw)

    # target_s = target_scaler.transform_np(target_array)

    T = target_s.shape[0]
    target_n = target_s[memory : T - 1]
    target_np1 = target_s[memory + 1 : T]

    if target_mode == "direct":
        Y = target_np1
    elif target_mode == "delta":
        Y = target_np1 - target_n
    else:
        raise ValueError("target_mode must be 'direct' or 'delta'.")

    t0 = time.perf_counter()
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    record_timing(timer, f"{timing_prefix}.astype", t0)

    return X, Y, input_scalers, target_scaler


def make_memory_input_from_histories(
    state_hist,
    theta_hist,
    *,
    input_sources,
    input_scalers,
    target_scaler,
    rollout_state_source,
    rollout_theta_source,
):
    """
    state_hist:
        [state_n, state_{n-1}, ...]
        If ROLLOUT_REP='spec12', this is v history.
        If ROLLOUT_REP='phys14', this is x history.

    theta_hist:
        [theta_n, theta_{n-1}, ...]
        If ROLLOUT_REP='spec12', this is thetaHat history.
        If ROLLOUT_REP='phys14', this is theta history.
    """
    hist_by_source = {
        rollout_state_source: state_hist,
        rollout_theta_source: theta_hist,
    }

    blocks = []

    for src in input_sources:
        if src not in hist_by_source:
            raise ValueError(
                f"Cannot build input source {src!r} from rollout histories. "
                f"Available sources are {tuple(hist_by_source)}."
            )

        if src not in input_scalers:
            raise KeyError(f"Missing input_scalers[{src!r}].")

        scaler = input_scalers[src]

        for arr in hist_by_source[src]:
            d_src = arr.shape[-1]
            arr2 = arr.reshape((-1, d_src))
            arr2_s = scaler.transform_cp(arr2)
            blocks.append(arr2_s.reshape(arr.shape))
    
    if len(blocks) == 0:
        raise ValueError("No input blocks were constructed.")

    X_cur = cp.concatenate(blocks, axis=-1)

    d_theta = theta_hist[0].shape[-1]
    theta_cur_scaled = target_scaler.transform_cp(
        theta_hist[0].reshape((-1, d_theta))
    ).reshape(theta_hist[0].shape)

    return X_cur, theta_cur_scaled


def get_vpt_learned_batch(
    model,
    test_state_np,
    test_theta_np,
    test_v6_np,
    stepper,
    input_scalers,
    target_scaler,
    state_scale_real12,
    *,
    pca_input=None,
    target_mode: str,
    input_sources,
    rollout_state_source,
    rollout_theta_source,
    memory: int,
    interp_degree: int,
    dt,
    Lyapunov_exp,
    error_threshold=0.5**2,
):
    """
    Validation helper for multiple hyperparameters at once.

    Representation-agnostic:
        if ROLLOUT_REP='phys14':
            test_state_np = x14_phys
            test_theta_np = theta14_phys

        if ROLLOUT_REP='spec12':
            test_state_np = v_real12
            test_theta_np = thetaHat_real12
    """
    test_state = cp.asarray(test_state_np, dtype=cp.float64)
    test_theta = cp.asarray(test_theta_np, dtype=cp.float64)
    test_v6 = cp.asarray(test_v6_np, dtype=cp.complex128)

    T, d_state = test_state.shape
    d_theta = test_theta.shape[-1]

    P = int(np.asarray(model.epsilon).size)

    if T <= memory + 1:
        raise ValueError("Validation segment is too short for requested memory.")

    x_hist = [
        cp.repeat(test_state[memory - lag][None, None, :], P, axis=0)
        for lag in range(memory + 1)
    ]

    theta_hist = [
        cp.repeat(test_theta[memory - lag][None, None, :], P, axis=0)
        for lag in range(memory + 1)
    ]

    theta_interp_hist = theta_hist[:min(len(theta_hist), interp_degree)]

    nmse_flag = cp.zeros(P, dtype=cp.bool_)
    se_flag = cp.zeros(P, dtype=cp.bool_)

    tau_nmse = cp.zeros(P, dtype=cp.float64)
    tau_se = cp.zeros(P, dtype=cp.float64)

    state_scale_cp = cp.asarray(state_scale_real12, dtype=cp.float64)[None, :]

    for i in range(memory + 1, T):
        X_cur, theta_cur_scaled = make_memory_input_from_histories(
            x_hist,
            theta_hist,
            input_sources=input_sources,
            input_scalers=input_scalers,
            target_scaler=target_scaler,
            rollout_state_source=rollout_state_source,
            rollout_theta_source=rollout_theta_source,
        )

        if pca_input is not None:
            X_cur = pca_input.transform_cp(X_cur)

        pred_scaled = model.mapping(X_cur).reshape(P, 1, d_theta)

        if target_mode == "direct":
            theta_next_scaled = pred_scaled
        elif target_mode == "delta":
            theta_next_scaled = theta_cur_scaled + pred_scaled
        else:
            raise ValueError("Bad target_mode.")

        theta_next = target_scaler.inverse_cp(
            theta_next_scaled.reshape(P, d_theta)
        ).reshape(P, 1, d_theta)

        x_next = stepper.step_with_theta_history(
            x_hist[0].reshape(P, d_state),
            [th.reshape(P, d_theta) for th in theta_interp_hist],
            theta_next.reshape(P, d_theta),
            interp_degree=interp_degree,
        ).reshape(P, 1, d_state)

        nan_mask = cp.isnan(x_next.reshape(P, d_state)).any(axis=1)

        if nan_mask.any():
            success_steps = i - memory - 1

            upd_nmse = cp.logical_and(~nmse_flag, nan_mask)
            upd_se = cp.logical_and(~se_flag, nan_mask)

            tau_nmse = cp.where(upd_nmse, success_steps, tau_nmse)
            tau_se = cp.where(upd_se, success_steps, tau_se)

            nmse_flag = cp.logical_or(nmse_flag, nan_mask)
            se_flag = cp.logical_or(se_flag, nan_mask)

        # NMSE is always measured in real12 Fourier mode coordinates.
        if stepper.rep == "spec12":
            # x_next is already v_real12.
            v_true_real12 = real12_from_complex6_cp(test_v6[i][None, :])
            diff_real12 = x_next[:, 0, :] - v_true_real12
        else:
            # x_next is physical; convert predicted state to modes.
            v_pred_pos = stepper.state_to_modes(x_next.reshape(P, d_state))[:, 1:7]
            v_true_pos = test_v6[i][None, :]
            diff_real12 = real12_from_pos_modes(v_pred_pos - v_true_pos)

        nmse = cp.mean((diff_real12 / state_scale_cp) ** 2, axis=1)

        # Secondary state-space error. In spec12, this is Fourier-state error.
        diff_state = test_state[i][None, :] - x_next[:, 0, :]
        denom = cp.linalg.norm(test_state[i]) ** 2
        if float(denom.get()) == 0.0:
            se = cp.zeros(P, dtype=cp.float64)
        else:
            se = cp.linalg.norm(diff_state, axis=1) ** 2 / denom

        success_steps = i - memory - 1

        cond_nmse = cp.logical_and(~nmse_flag, nmse > error_threshold)
        tau_nmse = cp.where(cond_nmse, success_steps, tau_nmse)
        nmse_flag = cp.logical_or(nmse_flag, cond_nmse)

        cond_se = cp.logical_and(~se_flag, se > error_threshold)
        tau_se = cp.where(cond_se, success_steps, tau_se)
        se_flag = cp.logical_or(se_flag, cond_se)

        if cp.logical_and(nmse_flag, se_flag).all():
            break

        x_hist = [x_next] + x_hist[:-1]
        theta_hist = [theta_next] + theta_hist[:-1]
        theta_interp_hist = [theta_next] + theta_interp_hist[:max(0, interp_degree - 1)]

    last_step = T - memory

    tau_nmse = cp.where(~nmse_flag, last_step, tau_nmse)
    tau_se = cp.where(~se_flag, last_step, tau_se)

    factor = dt * Lyapunov_exp
    tau_nmse *= factor
    tau_se *= factor

    return cp.asnumpy(tau_nmse), cp.asnumpy(tau_se)


def get_vpt_learned_multipoint_single(
    model,
    test_state_np,
    test_theta_np,
    test_v6_np,
    stepper,
    input_scalers,
    target_scaler,
    state_scale_real12,
    *,
    pca_input=None,
    target_mode: str,
    input_sources,
    rollout_state_source,
    rollout_theta_source,
    # include_x: bool,
    # include_theta: bool,
    memory: int,
    interp_degree: int,
    dt,
    Lyapunov_exp,
    error_threshold=0.5**2,
):
    """
    Test helper for one fitted single-pipeline model over multiple trajectories.

    Uses memory input:
        [x_n, ..., x_{n-memory}, theta_n, ..., theta_{n-memory}].
    """
    test_state = cp.asarray(test_state_np, dtype=cp.float64)
    test_theta = cp.asarray(test_theta_np, dtype=cp.float64)
    test_v6 = cp.asarray(test_v6_np, dtype=cp.complex128)

    # M, T, d = test_state.shape
    M, T, d_state = test_state.shape
    d_theta = test_theta.shape[-1]

    if T <= memory + 1:
        raise ValueError("Test segment is too short for requested memory.")

    # Histories ordered as [current, previous, ..., previous-memory].
    x_hist = [
        test_state[:, memory - lag, :]
        for lag in range(memory + 1)
    ]

    theta_hist = [
        test_theta[:, memory - lag, :]
        for lag in range(memory + 1)
    ]

    theta_interp_hist = theta_hist[:min(len(theta_hist), interp_degree)]

    nmse_flag = cp.zeros(M, dtype=cp.bool_)
    se_flag = cp.zeros(M, dtype=cp.bool_)

    tau_nmse = cp.zeros(M, dtype=cp.float64)
    tau_se = cp.zeros(M, dtype=cp.float64)

    state_scale_cp = cp.asarray(state_scale_real12, dtype=cp.float64)[None, :]

    for i in range(memory + 1, T):
        X_cur, theta_cur_scaled = make_memory_input_from_histories(
            x_hist,
            theta_hist,
            input_sources=input_sources,
            input_scalers=input_scalers,
            target_scaler=target_scaler,
            rollout_state_source=rollout_state_source,
            rollout_theta_source=rollout_theta_source,
        )

        if pca_input is not None:
            X_cur = pca_input.transform_cp(X_cur)

        # pred_scaled = cp.asarray(model.mapping(X_cur), dtype=cp.float64).reshape(M, d)
        pred_scaled = cp.asarray(model.mapping(X_cur), dtype=cp.float64).reshape(M, d_theta)

        if target_mode == "direct":
            theta_next_scaled = pred_scaled
        elif target_mode == "delta":
            theta_next_scaled = theta_cur_scaled + pred_scaled
        else:
            raise ValueError("Bad target_mode.")

        theta_next = target_scaler.inverse_cp(theta_next_scaled)

        x_next = stepper.step_with_theta_history(
            x_hist[0],
            theta_interp_hist,
            theta_next,
            interp_degree=interp_degree,
        )

        nan_mask = cp.isnan(x_next).any(axis=1)

        if nan_mask.any():
            success_steps = i - memory - 1

            upd_nmse = cp.logical_and(~nmse_flag, nan_mask)
            upd_se = cp.logical_and(~se_flag, nan_mask)

            tau_nmse[upd_nmse] = success_steps
            tau_se[upd_se] = success_steps

            nmse_flag = cp.logical_or(nmse_flag, nan_mask)
            se_flag = cp.logical_or(se_flag, nan_mask)

        v_pred_pos = stepper.state_to_modes(x_next)[:, 1:7]
        v_true_pos = test_v6[:, i, :]

        diff_real12 = real12_from_pos_modes(v_pred_pos - v_true_pos)
        nmse = cp.mean((diff_real12 / state_scale_cp) ** 2, axis=1)

        diff_phys = test_state[:, i, :] - x_next
        se = cp.linalg.norm(diff_phys, axis=1) ** 2 / cp.linalg.norm(test_state[:, i, :], axis=1) ** 2

        success_steps = i - memory - 1

        cond_nmse = cp.logical_and(~nmse_flag, nmse > error_threshold)
        tau_nmse[cond_nmse] = success_steps
        nmse_flag = cp.logical_or(nmse_flag, cond_nmse)

        cond_se = cp.logical_and(~se_flag, se > error_threshold)
        tau_se[cond_se] = success_steps
        se_flag = cp.logical_or(se_flag, cond_se)

        if cp.logical_and(nmse_flag, se_flag).all():
            break

        # Update histories:
        # [x_{n+1}, x_n, x_{n-1}, ...]
        x_hist = [x_next] + x_hist[:-1]
        theta_hist = [theta_next] + theta_hist[:-1]

        theta_interp_hist = [theta_next] + theta_interp_hist[:max(0, interp_degree - 1)]

    last_step = T - memory

    tau_nmse[~nmse_flag] = last_step
    tau_se[~se_flag] = last_step

    factor = dt * Lyapunov_exp
    tau_nmse *= factor
    tau_se *= factor

    return cp.asnumpy(tau_nmse), cp.asnumpy(tau_se)


def batched_compute_inner_cv(
    k_train_arrays,
    k_valid_arrays,
    # k_state_train,
    # k_theta_train,
    # k_state_valid,
    # k_theta_valid,
    k_v6_valid,
    epsilon_array,
    lambda_array,
    mode,
    device,
):
    timer = TimingRecorder()

    with cp.cuda.Device(device):
        t0 = time.perf_counter()
        stepper = ReducedKSForcedETDRK4Stepper(
            rep=ROLLOUT_REP,
            dt=dt,
            alpha=alpha,
            substeps=substeps,
            M=Mquad,
        )
        timer.record(f"validation.device{device}.stepper_init", time.perf_counter() - t0)

        # input_scalers, target_scaler = make_state_target_scalers(
        #     SCALING_MODE,
        #     k_state_train,
        #     k_theta_train,
        # )

        # X_train, Y_train = build_theta_pairs_scaled(
        #     k_state_train,
        #     k_theta_train,
        #     input_scalers=input_scalers,
        #     target_scaler=target_scaler,
        #     target_mode=TARGET_MODE,
        #     include_x=INCLUDE_X_INPUT,
        #     include_theta=INCLUDE_THETA_INPUT,
        #     memory=MEMORY,
        # )
        target_train = k_train_arrays[TARGET_VAR]

        t0 = time.perf_counter()
        X_train, Y_train, input_scalers, target_scaler = build_train_pairs_general(
            k_train_arrays,
            target_train,
            input_sources=INPUT_SOURCES,
            input_scaling_mode=INPUT_SCALING_MODE,
            target_scaling_mode=TARGET_SCALING_MODE,
            target_mode=TARGET_MODE,
            memory=MEMORY,
            timer=timer,
            timing_prefix=f"validation.device{device}.train_pairs",
        )
        timer.record(f"validation.device{device}.train_pairs_total", time.perf_counter() - t0)

        if device == 0:
            audit_train_pair_alignment(
                X_train=X_train,
                Y_train=Y_train,
                k_train_arrays=k_train_arrays,
                input_sources=INPUT_SOURCES,
                input_scalers=input_scalers,
                target_scaler=target_scaler,
                target_var=TARGET_VAR,
                source_dim=SOURCE_DIM,
                target_mode=TARGET_MODE,
                memory=MEMORY,
                label=f"{INPUT_MODE}->{TARGET_VAR}, {INPUT_SCALING_MODE}/{TARGET_SCALING_MODE}",
            )

        # if device == 0 and INPUT_MODE == "v" and TARGET_VAR == "thetaHat" and MEMORY == 0:
        #     X_expected = k_train_arrays["v"][:-1]
        #     Y_expected = k_train_arrays["thetaHat"][1:]
        #     dx = np.max(np.abs(X_train - X_expected))
        #     dy = np.max(np.abs(Y_train - Y_expected))

        #     print(f"[align check] max |X - v_n| = {dx:.17e}")
        #     print(f"[align check] max |Y - thetaHat_np1| = {dy:.17e}")
        #     print("[align check] X exact array_equal:", np.array_equal(X_train, X_expected))
        #     print("[align check] Y exact array_equal:", np.array_equal(Y_train, Y_expected))
            
        #     # if INPUT_MODE == "v" and TARGET_VAR == "thetaHat" and MEMORY == 0:

        #     X_back = input_scalers["v"].inverse_np(X_train)
        #     Y_back = target_scaler.inverse_np(Y_train)

        #     print(f"[align/inverse check] max |X_back - v_n| = {np.max(np.abs(X_back - X_expected)):.17e}")
        #     print(f"[align/inverse check] max |Y_back - thetaHat_np1| = {np.max(np.abs(Y_back - Y_expected)):.17e}")
        #     # print("[align check] max |X - v_n| =",
        #     #     np.max(np.abs(X_train - X_expected)))
        #     # print("[align check] max |Y - thetaHat_np1| =",
        #     #     np.max(np.abs(Y_train - Y_expected)))

        if device == 0:
            print(
                f"[memory={MEMORY}, interp={INTERP_DEGREE}, "
                f"target_mode={TARGET_MODE}, target_var={TARGET_VAR}] "
                f"X_train shape = {X_train.shape}, Y_train shape = {Y_train.shape}"
            )
        if device == 0:
            print(
                f"[scaling] input={INPUT_SCALING_MODE}, target={TARGET_SCALING_MODE}, "
                f"X_train min/max = {np.min(X_train):.3e}/{np.max(X_train):.3e}, "
                f"Y_train min/max = {np.min(Y_train):.3e}/{np.max(Y_train):.3e}"
            )
        
        pca_input = None

        X_train_model = X_train

        local_opts = {
            "map_type": "direct",
            "pipeline": "batch",
            "inp": X_train_model,
            "out": Y_train,
            "std_nmse_ref": Y_train,
            "solver": "eig" if KRR_SOLVER_MODE == "eig_grouped" else KRR_SOLVER_MODE,
        }

        t0 = time.perf_counter()
        model = Modeler(**local_opts)
        timer.record(f"validation.device{device}.model_init", time.perf_counter() - t0)

        maybe_sync_gpu()
        t0 = time.perf_counter()
        X_cp = cp.asarray(model.inp, dtype=cp.float64)
        distance_matrix = compute_distance_matrix(X_cp)
        maybe_sync_gpu()
        timer.record(f"validation.device{device}.distance_matrix", time.perf_counter() - t0)

        tau_f_array = np.zeros(len(epsilon_array), dtype=float)
        vpt_array = np.zeros(len(epsilon_array), dtype=float)

        validation_indices = VALIDATION_OFFSETS
        use_grouped_eig = KRR_SOLVER_MODE == "eig_grouped"

        if use_grouped_eig and device == 0:
            print(
                "[validation] using grouped fixed-epsilon eig path "
                f"for mode={mode}, candidates={len(epsilon_array)}"
            )

        chunk_size = max(1, n_chunk_lst[device])

        for start in range(0, len(epsilon_array), chunk_size):
            end = min(len(epsilon_array), start + chunk_size)

            eps_chunk = epsilon_array[start:end]
            lam_chunk = lambda_array[start:end]



            if use_grouped_eig:
                unique_eps = np.unique(eps_chunk)

                if device == 0:
                    print(
                        "[validation/grouped eig] "
                        f"chunk {start}:{end}, unique epsilons={len(unique_eps)}"
                    )

                maybe_sync_gpu()
                t0 = time.perf_counter()
                model.fit_model_grouped_epsilon_eig(
                    eps_chunk,
                    lam_chunk,
                    mode,
                    distance_matrix,
                )
                maybe_sync_gpu()
                timer.record(
                    f"validation.device{device}.{mode}.grouped_eig_fit",
                    time.perf_counter() - t0,
                )
            else:
                maybe_sync_gpu()
                t0 = time.perf_counter()
                model.fit_model(eps_chunk, lam_chunk, mode, distance_matrix)
                maybe_sync_gpu()
                timer.record(
                    f"validation.device{device}.{mode}.fit_model",
                    time.perf_counter() - t0,
                )

            for j in range(validation_repeats):
                start_j = validation_indices[j]
                end_j = start_j + validation_horizon

                j_valid_state = k_valid_arrays[ROLLOUT_STATE_SOURCE][start_j:end_j]
                j_valid_theta = k_valid_arrays[ROLLOUT_THETA_SOURCE][start_j:end_j]
                j_valid_v6 = k_v6_valid[start_j:end_j]

                maybe_sync_gpu()
                t0 = time.perf_counter()
                vpt_chunk, tau_f_chunk = get_vpt_learned_batch(
                    model,
                    j_valid_state,
                    j_valid_theta,
                    j_valid_v6,
                    stepper,
                    input_scalers,
                    target_scaler,
                    state_scale_real12,
                    pca_input=pca_input,
                    target_mode=TARGET_MODE,
                    input_sources=INPUT_SOURCES,
                    rollout_state_source=ROLLOUT_STATE_SOURCE,
                    rollout_theta_source=ROLLOUT_THETA_SOURCE,
                    memory=MEMORY,
                    interp_degree=INTERP_DEGREE,
                    dt=dt,
                    Lyapunov_exp=Lyapunov_exp,
                    error_threshold=error_threshold,
                )
                maybe_sync_gpu()
                timer.record(
                    f"validation.device{device}.{mode}.rollout",
                    time.perf_counter() - t0,
                )

                vpt_array[start:end] += np.asarray(vpt_chunk)
                tau_f_array[start:end] += np.asarray(tau_f_chunk)

            free_gpu_memory()

        vpt_array /= validation_repeats
        tau_f_array /= validation_repeats

        return tau_f_array, vpt_array, timer.as_dict()


def audit_train_pair_alignment(
    *,
    X_train,
    Y_train,
    k_train_arrays,
    input_sources,
    input_scalers,
    target_scaler,
    target_var,
    source_dim,
    target_mode,
    memory,
    label="",
):
    """
    Audit that X_train and Y_train are correctly aligned with the raw data.

    This checks both:
        1. scaled-space agreement,
        2. inverse-scaled raw-space agreement.

    Works for identity/minmax/zscore and memory m >= 0.
    """
    T = k_train_arrays[target_var].shape[0]
    n_rows = T - memory - 1

    if X_train.shape[0] != n_rows:
        raise ValueError(
            f"{label} X_train has {X_train.shape[0]} rows, "
            f"expected {n_rows}."
        )

    if Y_train.shape[0] != n_rows:
        raise ValueError(
            f"{label} Y_train has {Y_train.shape[0]} rows, "
            f"expected {n_rows}."
        )

    print(f"[audit {label}] X_train shape = {X_train.shape}")
    print(f"[audit {label}] Y_train shape = {Y_train.shape}")

    col0 = 0

    for src in input_sources:
        d_src = source_dim[src]
        scaler = input_scalers[src]
        raw_src = k_train_arrays[src]

        for lag in range(memory + 1):
            col1 = col0 + d_src

            X_block = X_train[:, col0:col1]

            raw_expected = raw_src[memory - lag : T - 1 - lag]
            scaled_expected = scaler.transform_np(raw_expected)
            raw_back = scaler.inverse_np(X_block)

            scaled_err = np.max(np.abs(X_block - scaled_expected))
            inv_err = np.max(np.abs(raw_back - raw_expected))

            print(
                f"[audit {label}] src={src:8s}, lag={lag}: "
                f"scaled err = {scaled_err:.17e}, "
                f"inverse raw err = {inv_err:.17e}"
            )

            col0 = col1

    if col0 != X_train.shape[1]:
        raise ValueError(
            f"{label} X_train column parse ended at {col0}, "
            f"but X_train has {X_train.shape[1]} columns."
        )

    raw_target = k_train_arrays[target_var]
    target_s = target_scaler.transform_np(raw_target)

    target_n = target_s[memory : T - 1]
    target_np1 = target_s[memory + 1 : T]

    if target_mode == "direct":
        Y_expected = target_np1
        Y_back = target_scaler.inverse_np(Y_train)
        raw_expected = raw_target[memory + 1 : T]

        scaled_err = np.max(np.abs(Y_train - Y_expected))
        inv_err = np.max(np.abs(Y_back - raw_expected))

        print(
            f"[audit {label}] target={target_var}, direct: "
            f"scaled err = {scaled_err:.17e}, "
            f"inverse raw err = {inv_err:.17e}"
        )

    elif target_mode == "delta":
        Y_expected = target_np1 - target_n

        # For delta mode, do NOT inverse-transform Y_train directly.
        # Reconstruct target_{n+1} in scaled coordinates first.
        target_np1_reconstructed_s = target_n + Y_train
        target_np1_back = target_scaler.inverse_np(target_np1_reconstructed_s)
        raw_expected = raw_target[memory + 1 : T]

        scaled_err = np.max(np.abs(Y_train - Y_expected))
        inv_err = np.max(np.abs(target_np1_back - raw_expected))

        print(
            f"[audit {label}] target={target_var}, delta: "
            f"scaled delta err = {scaled_err:.17e}, "
            f"reconstructed inverse raw err = {inv_err:.17e}"
        )

    else:
        raise ValueError(f"Unknown target_mode={target_mode!r}.")
