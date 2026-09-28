from __future__ import annotations


import numpy as np
import cupy as cp
from typing import Any, Literal, Optional
from .dm_main import DMClass

Pipeline = Literal["single", "batch"]

class Modeler:
    """
    Wrapper that constructs either single modeler (single pipeline)
    or batch_modeler (batch pipeline) depending on `pipeline` variable.
    """

    def __init__(
        self,
        pipeline: Pipeline = "single",
        *,
        use_batch: Optional[bool] = None,
        **kwargs: Any,
    ):
        # Allow either pipeline="batch"/"single" OR use_batch=True/False.
        if use_batch is not None:
            pipeline = "batch" if use_batch else "single"

        if pipeline not in ("single", "batch"):
            raise ValueError("pipeline must be 'single' or 'batch'")

        self.pipeline: Pipeline = pipeline

        if self.pipeline == "batch":
            self._impl = BatchModeler(**kwargs)
        else:
            self._impl = SingleModeler(**kwargs)

    def __getattr__(self, name: str) -> Any:
        """
        Forward attribute access to the underlying implementation.
        Called only if normal lookup on self fails.
        """
        return getattr(self._impl, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        If _impl exists, set attributes on _impl unless it's a wrapper field.
        """
        wrapper_fields = {"_impl", "pipeline"}
        if name in wrapper_fields or "_impl" not in self.__dict__:
            object.__setattr__(self, name, value)
        else:
            setattr(self._impl, name, value)

    def fit_model(self, *args: Any, **kwargs: Any) -> Any:
        return self._impl.fit_model(*args, **kwargs)

    def forecast(self, *args: Any, **kwargs: Any) -> Any:
        return self._impl.forecast(*args, **kwargs)

    def get_performance(self, *args: Any, **kwargs: Any) -> Any:
        return self._impl.get_performance(*args, **kwargs)
    

class SingleModeler:
    def __init__(self, **kwargs):
        self.dm = None
        self.def_opt = {
            'map_type' : 'direct',   # 'direct' or 'skip-connection'
            'inp'      : None,       # X (raw)
            'out'      : None,       # Y (raw)
            'data'     : None,       # timeseries (X=data[:-1], Y=data[1:])
            'norm'     : False,      # bool only
            'target'     : 'full',
            'sigma' : 10.0,
            'rho'   : 28.0,
            'verbose_perf': 0,
            'xy_integrator': 'rk4_cubic',   # 'heun' or 'rk4_quad' or 'rk4_cubic'
            'score_target': 'full',         # 'z' or 'full'
        }
        for k, v in self.def_opt.items(): 
            setattr(self, k, v)
        for k, v in kwargs.items():
            if k in self.def_opt: 
                setattr(self, k, v)

        if self.map_type not in {"direct", "skip-connection"}:
            raise ValueError("map_type must be 'direct' or 'skip-connection'")
        if not isinstance(self.norm, bool):
            raise ValueError("norm must be a boolean: True or False")

        # Build raw X, Y
        if self.inp is not None and self.out is not None:
            X_raw = np.asarray(self.inp); Y_raw = np.asarray(self.out)
        elif self.data is not None:
            data_raw = np.asarray(self.data)
            X_raw = data_raw[:-1]; Y_raw = data_raw[1:]
        else:
            raise ValueError("Provide either (inp, out) or data.")

        # Keep a full 3D copy for scoring (needed when score_target="full")
        Y_raw_full = np.asarray(Y_raw)  # shape (N,3)

        # --- Optionally restrict output to z only (still keep 3D input) ---
        if getattr(self, "target", "full") == "z":
            Y_eff = Y_raw_full[:, 2:3]     # shape (N,1)
        else:
            Y_eff = Y_raw_full             # shape (N,3)

        # --- Build labels ---
        if self.map_type == "direct":
            self.label = Y_eff
        else:
            if getattr(self, "target", "full") == "z":
                self.label = Y_eff - X_raw[:, 2:3]   # (N,1)
            else:
                self.label = Y_eff - X_raw           # (N,3)

        # --- Store inputs ---
        self.inp = X_raw

        if not hasattr(self, "mean_state"): self.mean_state = None
        if not hasattr(self, "std_state"):  self.std_state  = None

        # ALWAYS store a 3-vector std for scoring (even when target="z")
        self.std_nmse = Y_raw_full.std(axis=0)                 # shape (3,)
        self.std_nmse = np.where(self.std_nmse == 0.0, 1.0, self.std_nmse)

    def fit_model(self, epsilon, lambda_reg, mode, distance_matrix=None):
        self.epsilon    = epsilon
        self.lambda_reg = lambda_reg
        self.mode       = mode

        self.dm = DMClass(
            cp.array(self.inp),
            self.label,
            epsilon,
            lambda_reg,
            mode=mode,
            distance_matrix=distance_matrix,
            pipeline="single"
        )

    def mapping(self, x0):
        return self.dm.predict(x0)

    def forecast(self, x0):
        xp = cp.get_array_module(x0)
        x0 = xp.array(x0)
        if self.dm is None:
            raise ValueError("Model not trained yet")

        t = self.mapping(x0)
        return x0 + t if self.map_type == 'skip-connection' else t

    def get_performance(self, test, dt, Lyapunov_time, error_threshold=0.3**2, return_pred=False):
        vp = int(getattr(self, "verbose_perf", 0))
        if vp:
            tgt = getattr(self, "target", "full")
            xyi = getattr(self, "xy_integrator", "heun")
            sc  = getattr(self, "score_target", "z")
            print(
                f"[PERF DISPATCH][single] target={tgt} | score_target={sc} | xy_integrator={xyi} | "
                f"test_ndim={np.asarray(test).ndim}",
                flush=True
            )

        assert self.epsilon is not None
        assert self.lambda_reg is not None

        # ---- z-target special handling (A3 only) ----
        if getattr(self, "target", "full") == "z":
            if vp:
                print("[PERF BRANCH][single] A3: z-only model, xy coupled rollout", flush=True)
            return self.get_vpt_z_coupled_xy(
                test, dt, Lyapunov_time,
                error_threshold=error_threshold,
                return_pred=return_pred
            )

        # ---- default full-state path ----
        return self.get_vpt(
            test, dt, Lyapunov_time,
            error_threshold=error_threshold,
            return_pred=return_pred
        )

    def get_vpt_z_coupled_xy(self, test, dt, Lyapunov_time, error_threshold=0.3**2, return_pred=False):
        """
        A3 (single hyperparameter):
        - z_{n+1} predicted by learned z-model
        - (x,y) advanced by known ODEs using chosen xy_integrator
        - VPT scored by score_target: 'z' or 'full'
        """
        test = cp.asarray(test)
        dt = float(dt)
        T = int(test.shape[0])

        sigma = float(getattr(self, "sigma", 10.0))
        rho   = float(getattr(self, "rho",   28.0))

        xy_integrator = getattr(self, "xy_integrator", "heun")
        score_target  = getattr(self, "score_target", "z")

        # stds for scoring (always length-3 now)
        stds = cp.asarray(self.std_nmse, dtype=test.dtype).reshape(-1)
        sx, sy, sz = stds[0], stds[1], stds[2]

        # predicted trajectory storage (always store x,y,z since we roll them out)
        pred = None
        if return_pred:
            pred = cp.zeros((T, 3), dtype=test.dtype)

        # initial condition (truth)
        x_hat = test[0, 0]
        y_hat = test[0, 1]
        z_hat = test[0, 2]
        if return_pred:
            pred[0, 0] = x_hat
            pred[0, 1] = y_hat
            pred[0, 2] = z_hat

        # history for RK midpoint interpolation
        z_prev1 = z_hat
        z_prev2 = z_hat

        nmse_flag = cp.array(False)
        se_flag   = cp.array(False)
        tau_nmse  = cp.array(0, dtype=cp.int32)
        tau_se    = cp.array(0, dtype=cp.int32)

        for i in range(1, T):
            # ---- z-model prediction (always coupled: use predicted x_hat,y_hat,z_hat) ----
            inp = cp.asarray([[x_hat, y_hat, z_hat]], dtype=test.dtype)   # (1,3)
            t   = self.mapping(inp).reshape(-1)[0]                         # scalar (cupy)
            z_next = (z_hat + t) if (self.map_type == "skip-connection") else t
            z_np1_xy = z_next

            # ---- advance (x,y) using chosen integrator ----
            if xy_integrator == "heun":
                dx1 = sigma * (y_hat - x_hat)
                dy1 = x_hat * (rho - z_hat) - y_hat
                x_star = x_hat + dt * dx1
                y_star = y_hat + dt * dy1

                dx2 = sigma * (y_star - x_star)
                dy2 = x_star * (rho - z_np1_xy) - y_star

                x_next = x_hat + 0.5 * dt * (dx1 + dx2)
                y_next = y_hat + 0.5 * dt * (dy1 + dy2)

            elif xy_integrator == "rk4_quad":
                if i == 1:
                    z_half = 0.5 * (z_hat + z_np1_xy)
                else:
                    z_half = (-0.125) * z_prev1 + 0.75 * z_hat + 0.375 * z_np1_xy

                k1x = sigma * (y_hat - x_hat)
                k1y = x_hat * (rho - z_hat) - y_hat

                x2 = x_hat + 0.5 * dt * k1x
                y2 = y_hat + 0.5 * dt * k1y
                k2x = sigma * (y2 - x2)
                k2y = x2 * (rho - z_half) - y2

                x3 = x_hat + 0.5 * dt * k2x
                y3 = y_hat + 0.5 * dt * k2y
                k3x = sigma * (y3 - x3)
                k3y = x3 * (rho - z_half) - y3

                x4 = x_hat + dt * k3x
                y4 = y_hat + dt * k3y
                k4x = sigma * (y4 - x4)
                k4y = x4 * (rho - z_np1_xy) - y4

                x_next = x_hat + (dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
                y_next = y_hat + (dt / 6.0) * (k1y + 2*k2y + 2*k3y + k4y)

            elif xy_integrator == "rk4_cubic":
                if i == 1:
                    z_half = 0.5 * (z_hat + z_np1_xy)
                elif i == 2:
                    z_half = (-0.125) * z_prev1 + 0.75 * z_hat + 0.375 * z_np1_xy
                else:
                    z_half = (1/16) * z_prev2 + (-5/16) * z_prev1 + (15/16) * z_hat + (5/16) * z_np1_xy

                k1x = sigma * (y_hat - x_hat)
                k1y = x_hat * (rho - z_hat) - y_hat

                x2 = x_hat + 0.5 * dt * k1x
                y2 = y_hat + 0.5 * dt * k1y
                k2x = sigma * (y2 - x2)
                k2y = x2 * (rho - z_half) - y2

                x3 = x_hat + 0.5 * dt * k2x
                y3 = y_hat + 0.5 * dt * k2y
                k3x = sigma * (y3 - x3)
                k3y = x3 * (rho - z_half) - y3

                x4 = x_hat + dt * k3x
                y4 = y_hat + dt * k3y
                k4x = sigma * (y4 - x4)
                k4y = x4 * (rho - z_np1_xy) - y4

                x_next = x_hat + (dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
                y_next = y_hat + (dt / 6.0) * (k1y + 2*k2y + 2*k3y + k4y)

            else:
                raise ValueError(f"Unknown xy_integrator={xy_integrator}")

            # ---- advance state + z-history ----
            z_old = z_hat
            x_hat, y_hat, z_hat = x_next, y_next, z_next
            z_prev2 = z_prev1
            z_prev1 = z_old

            if return_pred:
                pred[i, 0] = x_hat
                pred[i, 1] = y_hat
                pred[i, 2] = z_hat

            # ---- NaN handling ----
            nan_mask = cp.isnan(x_hat) | cp.isnan(y_hat) | cp.isnan(z_hat)
            if bool(nan_mask.item()):
                last_idx = i - 1
                if not bool(nmse_flag.item()):
                    tau_nmse = cp.array(last_idx, dtype=cp.int32)
                    nmse_flag = cp.array(True)
                if not bool(se_flag.item()):
                    tau_se = cp.array(last_idx, dtype=cp.int32)
                    se_flag = cp.array(True)

            # ---- scoring ----
            if score_target == "z":
                diff = test[i, 2] - z_hat
                nmse = (diff / sz) ** 2
                denom = test[i, 2] ** 2
                denom = cp.where(denom == 0, 1.0, denom)
                se = (diff ** 2) / denom

            elif score_target == "full":
                dx = test[i, 0] - x_hat
                dy = test[i, 1] - y_hat
                dz = test[i, 2] - z_hat
                nmse = ((dx / sx) ** 2 + (dy / sy) ** 2 + (dz / sz) ** 2) / 3.0

                denomx = test[i, 0] ** 2
                denomy = test[i, 1] ** 2
                denomz = test[i, 2] ** 2
                denomx = cp.where(denomx == 0, 1.0, denomx)
                denomy = cp.where(denomy == 0, 1.0, denomy)
                denomz = cp.where(denomz == 0, 1.0, denomz)
                se = ((dx ** 2) / denomx + (dy ** 2) / denomy + (dz ** 2) / denomz) / 3.0

            else:
                raise ValueError(f"Unknown score_target={score_target}")

            cond_nmse = cp.logical_and(~nmse_flag, nmse > error_threshold)
            if bool(cond_nmse.item()):
                tau_nmse = cp.array(i - 1, dtype=cp.int32)
                nmse_flag = cp.array(True)

            cond_se = cp.logical_and(~se_flag, se > error_threshold)
            if bool(cond_se.item()):
                tau_se = cp.array(i - 1, dtype=cp.int32)
                se_flag = cp.array(True)

            if bool((nmse_flag & se_flag).item()) and not return_pred:
                break

        last_step = T
        tau_nmse = cp.where(~nmse_flag, last_step, tau_nmse)
        tau_se   = cp.where(~se_flag,   last_step, tau_se)

        factor = dt / Lyapunov_time
        tau_nmse = tau_nmse * factor
        tau_se   = tau_se   * factor

        tau_nmse_np = float(cp.asnumpy(tau_nmse))
        tau_se_np   = float(cp.asnumpy(tau_se))

        if return_pred:
            return tau_nmse_np, tau_se_np, pred
        return tau_nmse_np, tau_se_np

    def get_vpt(self, test, dt, Lyapunov_time, error_threshold=0.3**2,
            return_pred=False):
        """
        Forecast times (Lyapunov units) using NMSE and SE.

        Assumes epsilon is scalar (single model). Uses 0-D flags/tau on GPU.
        """
        test = cp.asarray(test)
        test_points = test.shape[0]
        d = test.shape[-1]

        pred = cp.zeros((test_points, d), dtype=test.dtype)

        u_hat = test[0].reshape(1, -1)
        pred[0] = u_hat[0]

        nmse_flag  = cp.array(False)
        se_flag    = cp.array(False)
        tau_f_nmse = cp.array(0, dtype=cp.int32)
        tau_f_se   = cp.array(0, dtype=cp.int32)

        _scl = cp.asarray(self.std_nmse).reshape(1, -1)  # (1, d)

        for i in range(1, test_points):
            u_hat = self.forecast(u_hat)

            u_hat = u_hat.reshape(1, -1)
            pred[i] = u_hat

            nan_mask = cp.isnan(u_hat).any()
            if nan_mask:
                last_idx = i - 1

                upd_nmse = cp.logical_and(~nmse_flag, nan_mask)
                upd_se   = cp.logical_and(~se_flag,   nan_mask)

                tau_f_nmse = cp.where(upd_nmse, last_idx, tau_f_nmse)
                tau_f_se   = cp.where(upd_se,   last_idx, tau_f_se)

                nmse_flag = cp.logical_or(nmse_flag, nan_mask)
                se_flag   = cp.logical_or(se_flag,   nan_mask)

            diff  = test[i] - u_hat
            mdiff = diff / _scl[0]

            nmse_ = cp.mean(mdiff**2)
            se_   = cp.linalg.norm(diff)**2/ (cp.linalg.norm(test[i])**2)

            cond_nmse = cp.logical_and(~nmse_flag, nmse_ > error_threshold)
            tau_f_nmse = cp.where(cond_nmse, i - 1, tau_f_nmse)
            nmse_flag  = cp.logical_or(nmse_flag, cond_nmse)

            cond_se = cp.logical_and(~se_flag, se_ > error_threshold)
            tau_f_se = cp.where(cond_se, i - 1, tau_f_se)
            se_flag  = cp.logical_or(se_flag, cond_se)

            if cp.logical_and(nmse_flag, se_flag) and not return_pred:
                break

        last_step = test_points
        tau_f_nmse = cp.where(~nmse_flag, last_step, tau_f_nmse)
        tau_f_se   = cp.where(~se_flag,   last_step, tau_f_se)

        factor = dt / Lyapunov_time
        tau_f_nmse = tau_f_nmse * factor
        tau_f_se   = tau_f_se * factor

        tau_f_nmse_np = float(cp.asnumpy(tau_f_nmse))
        tau_f_se_np   = float(cp.asnumpy(tau_f_se))

        if return_pred:
            return tau_f_nmse_np, tau_f_se_np, pred
        return tau_f_nmse_np, tau_f_se_np



class BatchModeler:
    def __init__(self, **kwargs):
        self.dm = None
        self.def_opt = {
            'map_type' : 'direct',   # 'direct' or 'skip-connection'
            'inp'      : None,       # X (raw)
            'out'      : None,       # Y (raw)
            'data'     : None,       # timeseries (X=data[:-1], Y=data[1:])
            'norm'     : False,      # bool only
            'target'     : 'full',
            'sigma' : 10.0,
            'rho'   : 28.0,
            'verbose_perf': 0,
            'xy_integrator': 'rk4_cubic',   # 'heun' or 'rk4_quad' or 'rk4_cubic'
            'score_target': 'full',           # 'z' or 'full'
        }
        for k, v in self.def_opt.items(): 
            setattr(self, k, v)
        for k, v in kwargs.items():
            if k in self.def_opt: 
                setattr(self, k, v)

        if self.map_type not in {"direct", "skip-connection"}:
            raise ValueError("map_type must be 'direct' or 'skip-connection'")
        if not isinstance(self.norm, bool):
            raise ValueError("norm must be a boolean: True or False")

        # Build raw X, Y
        if self.inp is not None and self.out is not None:
            X_raw = np.asarray(self.inp); Y_raw = np.asarray(self.out)
        elif self.data is not None:
            data_raw = np.asarray(self.data)
            X_raw = data_raw[:-1]; Y_raw = data_raw[1:]
        else:
            raise ValueError("Provide either (inp, out) or data.")
        
        # Keep a full 3D copy for scoring (needed when score_target="full")
        Y_raw_full = np.asarray(Y_raw)  # shape (N,3)

        # --- Optionally restrict output to z only (still keep 3D input) ---
        if getattr(self, "target", "full") == "z":
            Y_eff = Y_raw_full[:, 2:3]      # shape (N,1)
        else:
            Y_eff = Y_raw_full              # shape (N,3)

        # --- Build labels ---
        if self.map_type == "direct":
            self.label = Y_eff
        else:
            if getattr(self, "target", "full") == "z":
                self.label = Y_eff - X_raw[:, 2:3]   # (N,1) - (N,1)
            else:
                self.label = Y_eff - X_raw           # (N,3) - (N,3)

        # --- Store inputs ---
        self.inp = X_raw

        # std_nmse used for scoring:
        # ALWAYS store a 3-vector so full-state scoring is possible even when target="z".
        self.std_nmse = Y_raw_full.std(axis=0)           # shape (3,)
        self.std_nmse = np.where(self.std_nmse == 0.0, 1.0, self.std_nmse)

    def fit_model(self, epsilon, lambda_reg, mode, distance_matrix=None):
        # make sure epsilon and lambda_reg are 1D arrays
        epsilon    = np.atleast_1d(np.asarray(epsilon))
        lambda_reg = np.atleast_1d(np.asarray(lambda_reg))

        self.epsilon    = epsilon
        self.lambda_reg = lambda_reg
        self.mode       = mode

        self.dm = DMClass(
            cp.array(self.inp),
            self.label,
            epsilon,
            lambda_reg,
            mode=mode,
            distance_matrix=distance_matrix,
            pipeline="batch"
        )


    def mapping(self, x0):
        return self.dm.predict(x0)

    def forecast(self, x0):
        """Accepts RAW x0 (NumPy or CuPy), returns prediction in RAW units."""
        xp = cp.get_array_module(x0)
        x0 = xp.array(x0)
        if self.dm is None:
            raise ValueError("Model not trained yet")

        t = self.mapping(x0)
        return x0 + t if self.map_type == 'skip-connection' else t

    def get_performance(self, test, dt, Lyapunov_time, error_threshold=0.3**2, return_pred=False):
        vp = int(getattr(self, "verbose_perf", 0))
        if vp:
            tgt = getattr(self, "target", "full")
            xyi = getattr(self, "xy_integrator", "heun")
            print(
                f"[PERF DISPATCH][batch] target={tgt} | xy_integrator={xyi} | "
                f"test_ndim={np.asarray(test).ndim}",
                flush=True
            )

        # Always enforce that fit_model() happened before performance evaluation.
        assert self.epsilon is not None
        assert self.lambda_reg is not None

        # ---- z-target special handling (A3 only) ----
        if getattr(self, "target", "full") == "z":
            if vp:
                shape_tag = "(T,3)" if np.asarray(test).ndim == 2 else "(M,T,3)"
                print(f"[PERF BRANCH][batch] A3: z-only, xy coupled | test_shape={shape_tag}", flush=True)

            if np.asarray(test).ndim == 2:
                return self.get_vpt_z_coupled_xy(
                    test, dt, Lyapunov_time,
                    error_threshold=error_threshold, return_pred=return_pred
                )
            else:
                return self.get_vpt_multipoint_z_coupled_xy(
                    test, dt, Lyapunov_time,
                    error_threshold=error_threshold
                )

        # ---- default full-state paths ----
        assert np.asarray(test).ndim in (2, 3)

        if np.asarray(test).ndim == 2:
            return self.get_vpt(
                test, dt, Lyapunov_time,
                error_threshold=error_threshold,
                return_pred=return_pred
            )
        else:
            return self.get_vpt_multipoint(
                test, dt, Lyapunov_time,
                error_threshold=error_threshold
            )

    def get_vpt_z_coupled_xy(self, test, dt, Lyapunov_time, error_threshold=0.3**2, return_pred=False):
        """
        Batch hyperparams:
        - test is (T,3)
        - evaluate P hyperparams in parallel
        - coupled rollout: z predicted; (x,y) advanced via Heun endpoints with z_n and z_{n+1}
        - VPT scored on (xyz) or z-only (depending on score_target)
        """
        test = cp.asarray(test)
        dt = float(dt)

        P = int(np.asarray(self.epsilon).size)
        T = test.shape[0]
        assert P >= 1

        sigma = float(getattr(self, "sigma", 10.0))
        rho   = float(getattr(self, "rho",   28.0))

        # predicted state per hyperparam: (P,1,1)
        x0 = test[0, 0]
        y0 = test[0, 1]
        z0 = test[0, 2]

        x_hat = cp.full((P, 1, 1), x0, dtype=test.dtype)
        y_hat = cp.full((P, 1, 1), y0, dtype=test.dtype)
        z_hat = cp.full((P, 1, 1), z0, dtype=test.dtype)

        xy_integrator = getattr(self, "xy_integrator", "heun")
        score_target  = getattr(self, "score_target", "z")

        stds = cp.asarray(self.std_nmse, dtype=test.dtype).reshape(-1)
        sx, sy, sz = stds[0], stds[1], stds[2]

        # history buffers: z_prevK = z_{n-K}
        z_prev1 = z_hat.copy()   # z_{n-1} (starts as z_0)
        z_prev2 = z_hat.copy()   # z_{n-2} (also starts as z_0; used only once i>=3)
        z_prev3 = z_hat.copy()   # z_{n-3} (also starts as z_0; used only once i>=4)
        z_prev4 = z_hat.copy()   # z_{n-4} (also starts as z_0; used only once i>=5)        

        # optional prediction storage
        pred = None
        if return_pred:
            pred = cp.zeros((T, P, 3), dtype=test.dtype)
            pred[0, :, 0] = x_hat.reshape(P)
            pred[0, :, 1] = y_hat.reshape(P)
            pred[0, :, 2] = z_hat.reshape(P)

        nmse_flag = cp.zeros(P, dtype=cp.bool_)
        se_flag   = cp.zeros(P, dtype=cp.bool_)
        tau_nmse  = cp.zeros(P, dtype=cp.float64)
        tau_se    = cp.zeros(P, dtype=cp.float64)

        for i in range(1, T):
            inp = cp.concatenate([x_hat, y_hat, z_hat], axis=2)  # (P,1,3)
            t = self.mapping(inp)                               # (P,1,1)

            z_next = (z_hat + t) if (self.map_type == "skip-connection") else t
            z_np1_xy = z_next

            # ---- advance (x,y) using chosen integrator ----
            if xy_integrator == "heun":
                # ---- existing Heun endpoints step for (x,y) ----
                dx1 = sigma * (y_hat - x_hat)
                dy1 = x_hat * (rho - z_hat) - y_hat

                # predictor using z_n
                x_star = x_hat + dt * dx1
                y_star = y_hat + dt * dy1

                dx2 = sigma * (y_star - x_star)
                dy2 = x_star * (rho - z_np1_xy) - y_star

                # corrector using z_{n+1}
                x_next = x_hat + 0.5 * dt * (dx1 + dx2)
                y_next = y_hat + 0.5 * dt * (dy1 + dy2)

            elif xy_integrator == "rk4_quad":
                # ---- RK4 for (x,y) with midpoint z_half (quadratic in time) ----
                if i == 1:
                    z_half = 0.5 * (z_hat + z_np1_xy)    # linear midpoint for first step
                else:
                    z_half = (-0.125) * z_prev1 + 0.75 * z_hat + 0.375 * z_np1_xy

                # k1 at z_n
                k1x = sigma * (y_hat - x_hat)
                k1y = x_hat * (rho - z_hat) - y_hat

                # k2 at z_half
                x2 = x_hat + 0.5 * dt * k1x
                y2 = y_hat + 0.5 * dt * k1y
                k2x = sigma * (y2 - x2)
                k2y = x2 * (rho - z_half) - y2

                # k3 at z_half
                x3 = x_hat + 0.5 * dt * k2x
                y3 = y_hat + 0.5 * dt * k2y
                k3x = sigma * (y3 - x3)
                k3y = x3 * (rho - z_half) - y3

                # k4 at z_{n+1}
                x4 = x_hat + dt * k3x
                y4 = y_hat + dt * k3y
                k4x = sigma * (y4 - x4)
                k4y = x4 * (rho - z_np1_xy) - y4

                x_next = x_hat + (dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
                y_next = y_hat + (dt / 6.0) * (k1y + 2*k2y + 2*k3y + k4y)

            elif xy_integrator == "rk4_cubic":
                # ---- RK4 for (x,y) with midpoint z_half (quadratic in time) ----
                if i == 1:
                    z_half = 0.5 * (z_hat + z_np1_xy)    # linear midpoint for first step
                elif i == 2:
                    z_half = (-0.125) * z_prev1 + 0.75 * z_hat + 0.375 * z_np1_xy    # quadratic interp for second step
                else:
                    z_half = (1/16) * z_prev2 + (-5/16) * z_prev1 + (15/16) * z_hat + (5/16) * z_np1_xy

                # k1 at z_n
                k1x = sigma * (y_hat - x_hat)
                k1y = x_hat * (rho - z_hat) - y_hat

                # k2 at z_half
                x2 = x_hat + 0.5 * dt * k1x
                y2 = y_hat + 0.5 * dt * k1y
                k2x = sigma * (y2 - x2)
                k2y = x2 * (rho - z_half) - y2

                # k3 at z_half
                x3 = x_hat + 0.5 * dt * k2x
                y3 = y_hat + 0.5 * dt * k2y
                k3x = sigma * (y3 - x3)
                k3y = x3 * (rho - z_half) - y3

                # k4 at z_{n+1}
                x4 = x_hat + dt * k3x
                y4 = y_hat + dt * k3y
                k4x = sigma * (y4 - x4)
                k4y = x4 * (rho - z_np1_xy) - y4

                x_next = x_hat + (dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
                y_next = y_hat + (dt / 6.0) * (k1y + 2*k2y + 2*k3y + k4y)
            
            elif xy_integrator == "rk4_quartic":
                # fallbacks for startup steps:
                if i == 1:
                    z_half = 0.5 * (z_hat + z_np1_xy)
                elif i == 2:
                    z_half = (-0.125) * z_prev1 + 0.75 * z_hat + 0.375 * z_np1_xy
                elif i == 3:
                    z_half = (1/16) * z_prev2 + (-5/16) * z_prev1 + (15/16) * z_hat + (5/16) * z_np1_xy
                else:
                    # quartic: uses z_{n-3}, z_{n-2}, z_{n-1}, z_n, z_{n+1}
                    z_half = (-5/128) * z_prev3 + (7/32) * z_prev2 + (-35/64) * z_prev1 \
                            + (35/32) * z_hat + (35/128) * z_np1_xy

                # RK4 for (x,y) with stage z-values: z_n, z_half, z_half, z_{n+1}
                k1x = sigma * (y_hat - x_hat)
                k1y = x_hat * (rho - z_hat) - y_hat

                x2 = x_hat + 0.5 * dt * k1x
                y2 = y_hat + 0.5 * dt * k1y
                k2x = sigma * (y2 - x2)
                k2y = x2 * (rho - z_half) - y2

                x3 = x_hat + 0.5 * dt * k2x
                y3 = y_hat + 0.5 * dt * k2y
                k3x = sigma * (y3 - x3)
                k3y = x3 * (rho - z_half) - y3

                x4 = x_hat + dt * k3x
                y4 = y_hat + dt * k3y
                k4x = sigma * (y4 - x4)
                k4y = x4 * (rho - z_np1_xy) - y4

                x_next = x_hat + (dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
                y_next = y_hat + (dt / 6.0) * (k1y + 2*k2y + 2*k3y + k4y)

            elif xy_integrator == "rk4_quintic":
                # startup fallbacks:
                if i == 1:
                    z_half = 0.5 * (z_hat + z_np1_xy)
                elif i == 2:
                    z_half = (-0.125) * z_prev1 + 0.75 * z_hat + 0.375 * z_np1_xy
                elif i == 3:
                    z_half = (1/16) * z_prev2 + (-5/16) * z_prev1 + (15/16) * z_hat + (5/16) * z_np1_xy
                elif i == 4:
                    z_half = (-5/128) * z_prev3 + (7/32) * z_prev2 + (-35/64) * z_prev1 \
                            + (35/32) * z_hat + (35/128) * z_np1_xy
                else:
                    # quintic: z_{n-4}, z_{n-3}, z_{n-2}, z_{n-1}, z_n, z_{n+1}
                    z_half = (7/256) * z_prev4 + (-45/256) * z_prev3 + (63/128) * z_prev2 \
                            + (-105/128) * z_prev1 + (315/256) * z_hat + (63/256) * z_np1_xy

                # RK4 for (x,y) with stage z-values: z_n, z_half, z_half, z_{n+1}
                k1x = sigma * (y_hat - x_hat)
                k1y = x_hat * (rho - z_hat) - y_hat

                x2 = x_hat + 0.5 * dt * k1x
                y2 = y_hat + 0.5 * dt * k1y
                k2x = sigma * (y2 - x2)
                k2y = x2 * (rho - z_half) - y2

                x3 = x_hat + 0.5 * dt * k2x
                y3 = y_hat + 0.5 * dt * k2y
                k3x = sigma * (y3 - x3)
                k3y = x3 * (rho - z_half) - y3

                x4 = x_hat + dt * k3x
                y4 = y_hat + dt * k3y
                k4x = sigma * (y4 - x4)
                k4y = x4 * (rho - z_np1_xy) - y4

                x_next = x_hat + (dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
                y_next = y_hat + (dt / 6.0) * (k1y + 2*k2y + 2*k3y + k4y)
            
            else:
                raise ValueError(f"Unknown xy_integrator={xy_integrator}")

            # ---- advance state + z-history ----
            z_old = z_hat                      # store z_n before overwriting
            x_hat, y_hat, z_hat = x_next, y_next, z_next
            # shift history (only matters for rk4_cubic, but cheap to do always)
            z_prev4 = z_prev3
            z_prev3 = z_prev2
            z_prev2 = z_prev1
            z_prev1 = z_old

            x_s = x_hat.reshape(P)
            y_s = y_hat.reshape(P)
            z_s = z_hat.reshape(P)

            if return_pred:
                pred[i, :, 0] = x_s
                pred[i, :, 1] = y_s
                pred[i, :, 2] = z_s

            # ---- NaN handling (per hyperparam) ----
            nan_mask = cp.isnan(x_s) | cp.isnan(y_s) | cp.isnan(z_s)
            if nan_mask.any():
                last_idx = i - 1
                upd_nmse = cp.logical_and(~nmse_flag, nan_mask)
                upd_se   = cp.logical_and(~se_flag,   nan_mask)
                tau_nmse[upd_nmse] = last_idx
                tau_se[upd_se]     = last_idx
                nmse_flag = cp.logical_or(nmse_flag, nan_mask)
                se_flag   = cp.logical_or(se_flag,   nan_mask)

            if score_target == "z":
                diff = test[i, 2] - z_s
                nmse = (diff / sz) ** 2

                denom = test[i, 2] ** 2
                denom = cp.where(denom == 0, 1.0, denom)
                se = (diff ** 2) / denom

            elif score_target == "full":
                dx = test[i, 0] - x_s
                dy = test[i, 1] - y_s
                dz = test[i, 2] - z_s

                nmse = ((dx / sx) ** 2 + (dy / sy) ** 2 + (dz / sz) ** 2) / 3.0

                denomx = test[i, 0] ** 2
                denomy = test[i, 1] ** 2
                denomz = test[i, 2] ** 2
                denomx = cp.where(denomx == 0, 1.0, denomx)
                denomy = cp.where(denomy == 0, 1.0, denomy)
                denomz = cp.where(denomz == 0, 1.0, denomz)

                se = ((dx ** 2) / denomx + (dy ** 2) / denomy + (dz ** 2) / denomz) / 3.0

            else:
                raise ValueError(f"Unknown score_target={score_target}")

            cond_nmse = cp.logical_and(~nmse_flag, nmse > error_threshold)
            tau_nmse[cond_nmse] = i - 1
            nmse_flag = cp.logical_or(nmse_flag, cond_nmse)

            cond_se = cp.logical_and(~se_flag, se > error_threshold)
            tau_se[cond_se] = i - 1
            se_flag = cp.logical_or(se_flag, cond_se)

            if cp.logical_and(nmse_flag, se_flag).all() and not return_pred:
                break

        last_step = T
        tau_nmse[~nmse_flag] = last_step
        tau_se[~se_flag]     = last_step

        tau_nmse = tau_nmse * dt / Lyapunov_time
        tau_se   = tau_se   * dt / Lyapunov_time

        tau_nmse_np = cp.asnumpy(tau_nmse)
        tau_se_np   = cp.asnumpy(tau_se)

        if return_pred:
            return tau_nmse_np, tau_se_np, pred
        return tau_nmse_np, tau_se_np

    def get_vpt_multipoint_z_coupled_xy(self, test, dt, Lyapunov_time, error_threshold=0.3**2):
        """
        Multi-trajectory, typically P == 1 in test:
        - test is (M,T,3)
        - coupled rollout: z predicted; (x,y) advanced via Heun endpoints with z_n and z_{n+1}
        - VPT scored on (xyz) or z-only (depending on score_target)
        """
        test = cp.asarray(test)
        dt = float(dt)
        M, T, d = test.shape
        assert d == 3

        P = int(np.asarray(self.epsilon).size)
        assert P == 1, "This multipoint path assumes a single (best) hyperparameter."

        sigma = float(getattr(self, "sigma", 10.0))
        rho   = float(getattr(self, "rho",   28.0))

        # (M,1,1)
        x_hat = test[:, 0, 0].reshape(M, 1, 1)
        y_hat = test[:, 0, 1].reshape(M, 1, 1)
        z_hat = test[:, 0, 2].reshape(M, 1, 1)

        xy_integrator = getattr(self, "xy_integrator", "heun")
        score_target = getattr(self, "score_target", "z")

        stds = cp.asarray(self.std_nmse, dtype=test.dtype).reshape(-1)
        sx, sy, sz = stds[0], stds[1], stds[2]

        # history buffers: z_prevK = z_{n-K}
        z_prev1 = z_hat.copy()
        z_prev2 = z_hat.copy()
        z_prev3 = z_hat.copy()
        z_prev4 = z_hat.copy()

        nmse_flag = cp.zeros(M, dtype=cp.bool_)
        se_flag   = cp.zeros(M, dtype=cp.bool_)
        tau_nmse  = cp.zeros(M, dtype=cp.float64)
        tau_se    = cp.zeros(M, dtype=cp.float64)

        for i in range(1, T):
            inp = cp.concatenate([x_hat, y_hat, z_hat], axis=2)  # (M,1,3)
            t = self.mapping(inp)                               # (M,1,1)

            z_next = (z_hat + t) if (self.map_type == "skip-connection") else t
            z_np1_xy = z_next

            # ---- advance (x,y) using chosen integrator ----
            if xy_integrator == "heun":
                # ---- existing Heun endpoints step for (x,y) ----
                dx1 = sigma * (y_hat - x_hat)
                dy1 = x_hat * (rho - z_hat) - y_hat

                # predictor using z_n
                x_star = x_hat + dt * dx1
                y_star = y_hat + dt * dy1

                dx2 = sigma * (y_star - x_star)
                dy2 = x_star * (rho - z_np1_xy) - y_star

                # corrector using z_{n+1}
                x_next = x_hat + 0.5 * dt * (dx1 + dx2)
                y_next = y_hat + 0.5 * dt * (dy1 + dy2)

            elif xy_integrator == "rk4_quad":
                # ---- RK4 for (x,y) with midpoint z_half (quadratic in time) ----
                if i == 1:
                    z_half = 0.5 * (z_hat + z_np1_xy)    # linear midpoint for first step
                else:
                    z_half = (-0.125) * z_prev1 + 0.75 * z_hat + 0.375 * z_np1_xy

                # k1 at z_n
                k1x = sigma * (y_hat - x_hat)
                k1y = x_hat * (rho - z_hat) - y_hat

                # k2 at z_half
                x2 = x_hat + 0.5 * dt * k1x
                y2 = y_hat + 0.5 * dt * k1y
                k2x = sigma * (y2 - x2)
                k2y = x2 * (rho - z_half) - y2

                # k3 at z_half
                x3 = x_hat + 0.5 * dt * k2x
                y3 = y_hat + 0.5 * dt * k2y
                k3x = sigma * (y3 - x3)
                k3y = x3 * (rho - z_half) - y3

                # k4 at z_{n+1}
                x4 = x_hat + dt * k3x
                y4 = y_hat + dt * k3y
                k4x = sigma * (y4 - x4)
                k4y = x4 * (rho - z_np1_xy) - y4

                x_next = x_hat + (dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
                y_next = y_hat + (dt / 6.0) * (k1y + 2*k2y + 2*k3y + k4y)

            elif xy_integrator == "rk4_cubic":
                # ---- RK4 for (x,y) with midpoint z_half (quadratic in time) ----
                if i == 1:
                    z_half = 0.5 * (z_hat + z_np1_xy)    # linear midpoint for first step
                elif i == 2:
                    z_half = (-0.125) * z_prev1 + 0.75 * z_hat + 0.375 * z_np1_xy   # quadratic interp for second step
                else:
                    z_half = (1/16) * z_prev2 + (-5/16) * z_prev1 + (15/16) * z_hat + (5/16) * z_np1_xy

                # k1 at z_n
                k1x = sigma * (y_hat - x_hat)
                k1y = x_hat * (rho - z_hat) - y_hat

                # k2 at z_half
                x2 = x_hat + 0.5 * dt * k1x
                y2 = y_hat + 0.5 * dt * k1y
                k2x = sigma * (y2 - x2)
                k2y = x2 * (rho - z_half) - y2

                # k3 at z_half
                x3 = x_hat + 0.5 * dt * k2x
                y3 = y_hat + 0.5 * dt * k2y
                k3x = sigma * (y3 - x3)
                k3y = x3 * (rho - z_half) - y3

                # k4 at z_{n+1}
                x4 = x_hat + dt * k3x
                y4 = y_hat + dt * k3y
                k4x = sigma * (y4 - x4)
                k4y = x4 * (rho - z_np1_xy) - y4

                x_next = x_hat + (dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
                y_next = y_hat + (dt / 6.0) * (k1y + 2*k2y + 2*k3y + k4y)

            elif xy_integrator == "rk4_quartic":
                # fallbacks for startup steps:
                if i == 1:
                    z_half = 0.5 * (z_hat + z_np1_xy)
                elif i == 2:
                    z_half = (-0.125) * z_prev1 + 0.75 * z_hat + 0.375 * z_np1_xy
                elif i == 3:
                    z_half = (1/16) * z_prev2 + (-5/16) * z_prev1 + (15/16) * z_hat + (5/16) * z_np1_xy
                else:
                    # quartic: uses z_{n-3}, z_{n-2}, z_{n-1}, z_n, z_{n+1}
                    z_half = (-5/128) * z_prev3 + (7/32) * z_prev2 + (-35/64) * z_prev1 \
                            + (35/32) * z_hat + (35/128) * z_np1_xy

                # RK4 for (x,y) with stage z-values: z_n, z_half, z_half, z_{n+1}
                k1x = sigma * (y_hat - x_hat)
                k1y = x_hat * (rho - z_hat) - y_hat

                x2 = x_hat + 0.5 * dt * k1x
                y2 = y_hat + 0.5 * dt * k1y
                k2x = sigma * (y2 - x2)
                k2y = x2 * (rho - z_half) - y2

                x3 = x_hat + 0.5 * dt * k2x
                y3 = y_hat + 0.5 * dt * k2y
                k3x = sigma * (y3 - x3)
                k3y = x3 * (rho - z_half) - y3

                x4 = x_hat + dt * k3x
                y4 = y_hat + dt * k3y
                k4x = sigma * (y4 - x4)
                k4y = x4 * (rho - z_np1_xy) - y4

                x_next = x_hat + (dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
                y_next = y_hat + (dt / 6.0) * (k1y + 2*k2y + 2*k3y + k4y)

            elif xy_integrator == "rk4_quintic":
                # startup fallbacks:
                if i == 1:
                    z_half = 0.5 * (z_hat + z_np1_xy)
                elif i == 2:
                    z_half = (-0.125) * z_prev1 + 0.75 * z_hat + 0.375 * z_np1_xy
                elif i == 3:
                    z_half = (1/16) * z_prev2 + (-5/16) * z_prev1 + (15/16) * z_hat + (5/16) * z_np1_xy
                elif i == 4:
                    z_half = (-5/128) * z_prev3 + (7/32) * z_prev2 + (-35/64) * z_prev1 \
                            + (35/32) * z_hat + (35/128) * z_np1_xy
                else:
                    # quintic: z_{n-4}, z_{n-3}, z_{n-2}, z_{n-1}, z_n, z_{n+1}
                    z_half = (7/256) * z_prev4 + (-45/256) * z_prev3 + (63/128) * z_prev2 \
                            + (-105/128) * z_prev1 + (315/256) * z_hat + (63/256) * z_np1_xy

                # RK4 for (x,y) with stage z-values: z_n, z_half, z_half, z_{n+1}
                k1x = sigma * (y_hat - x_hat)
                k1y = x_hat * (rho - z_hat) - y_hat

                x2 = x_hat + 0.5 * dt * k1x
                y2 = y_hat + 0.5 * dt * k1y
                k2x = sigma * (y2 - x2)
                k2y = x2 * (rho - z_half) - y2

                x3 = x_hat + 0.5 * dt * k2x
                y3 = y_hat + 0.5 * dt * k2y
                k3x = sigma * (y3 - x3)
                k3y = x3 * (rho - z_half) - y3

                x4 = x_hat + dt * k3x
                y4 = y_hat + dt * k3y
                k4x = sigma * (y4 - x4)
                k4y = x4 * (rho - z_np1_xy) - y4

                x_next = x_hat + (dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
                y_next = y_hat + (dt / 6.0) * (k1y + 2*k2y + 2*k3y + k4y)

            else:
                raise ValueError(f"Unknown xy_integrator={xy_integrator}")
            
            # ---- advance state + z-history ----
            z_old = z_hat                      # store z_n before overwriting
            x_hat, y_hat, z_hat = x_next, y_next, z_next
            # shift history (only matters for rk4_cubic, but cheap to do always)
            z_prev4 = z_prev3
            z_prev3 = z_prev2
            z_prev2 = z_prev1
            z_prev1 = z_old

            z_s = z_hat.reshape(M)
            x_s = x_hat.reshape(M)
            y_s = y_hat.reshape(M)

            nan_mask = cp.isnan(x_s) | cp.isnan(y_s) | cp.isnan(z_s)
            if nan_mask.any():
                last_idx = i - 1
                upd_nmse = cp.logical_and(~nmse_flag, nan_mask)
                upd_se   = cp.logical_and(~se_flag,   nan_mask)
                tau_nmse[upd_nmse] = last_idx
                tau_se[upd_se]     = last_idx
                nmse_flag = cp.logical_or(nmse_flag, nan_mask)
                se_flag   = cp.logical_or(se_flag,   nan_mask)

            if score_target == "z":
                diff = test[:, i, 2] - z_s
                nmse = (diff / sz) ** 2

                denom = test[:, i, 2] ** 2
                denom = cp.where(denom == 0, 1.0, denom)
                se = (diff ** 2) / denom

            elif score_target == "full":
                dx = test[:, i, 0] - x_s
                dy = test[:, i, 1] - y_s
                dz = test[:, i, 2] - z_s

                nmse = ((dx / sx) ** 2 + (dy / sy) ** 2 + (dz / sz) ** 2) / 3.0

                denomx = test[:, i, 0] ** 2
                denomy = test[:, i, 1] ** 2
                denomz = test[:, i, 2] ** 2
                denomx = cp.where(denomx == 0, 1.0, denomx)
                denomy = cp.where(denomy == 0, 1.0, denomy)
                denomz = cp.where(denomz == 0, 1.0, denomz)

                se = ((dx ** 2) / denomx + (dy ** 2) / denomy + (dz ** 2) / denomz) / 3.0

            else:
                raise ValueError(f"Unknown score_target={score_target}")

            cond_nmse = cp.logical_and(~nmse_flag, nmse > error_threshold)
            tau_nmse[cond_nmse] = i - 1
            nmse_flag = cp.logical_or(nmse_flag, cond_nmse)

            cond_se = cp.logical_and(~se_flag, se > error_threshold)
            tau_se[cond_se] = i - 1
            se_flag = cp.logical_or(se_flag, cond_se)

            if cp.logical_and(nmse_flag, se_flag).all():
                break

        last_step = T
        tau_nmse[~nmse_flag] = last_step
        tau_se[~se_flag]     = last_step

        tau_nmse = tau_nmse * dt / Lyapunov_time
        tau_se   = tau_se   * dt / Lyapunov_time

        return cp.asnumpy(tau_nmse), cp.asnumpy(tau_se)
        

    def get_vpt_multipoint(self, test, dt, Lyapunov_time, error_threshold=0.3**2):
        """
        Multi-trajectory VPT / tau_f computation on GPU.

        Parameters
        ----------
        test : array-like, shape (n_traj, T, d)
            Multiple test trajectories in RAW units.
        dt : float
        Lyapunov_time : float
        error_threshold : float, optional
            Threshold for both NMSE and SE.
        return_pred : bool, optional
            If True, also returns predictions with shape (n_traj, T, d) as CuPy array.

        Returns
        -------
        tau_f_nmse : np.ndarray, shape (n_traj,)
            Tau_f based on NMSE per trajectory (in Lyapunov units)
        tau_f_se   : np.ndarray, shape (n_traj,)
            Tau_f based on SE per trajectory (in Lyapunov units)
        pred       : cp.ndarray, shape (n_traj, T, d) (only if return_pred=True)
            Forecast trajectories in RAW units.
        """
        test = cp.asarray(test)
        M, T, d = test.shape

        u_hat = test[:, 0, :]              # (M, d)
        u_hat = u_hat[:, None, :]

        nmse_flag  = cp.zeros(M, dtype=cp.bool_)
        se_flag    = cp.zeros(M, dtype=cp.bool_)
        tau_f_nmse = cp.zeros(M, dtype=cp.float64)
        tau_f_se   = cp.zeros(M, dtype=cp.float64)

        _scl = cp.asarray(self.std_nmse)[None, None, :]

        for i in range(1, T):
            u_hat = self.forecast(u_hat)

            # NaN handling: per trajectory
            nan_mask = cp.isnan(u_hat).any(axis=(1,2))   # (M,)
            if nan_mask.any():
                last_idx = i - 1

                upd_nmse = cp.logical_and(~nmse_flag, nan_mask)
                upd_se   = cp.logical_and(~se_flag, nan_mask)
                tau_f_nmse[upd_nmse] = last_idx
                tau_f_se[upd_se]     = last_idx
                nmse_flag = cp.logical_or(nmse_flag, nan_mask)
                se_flag   = cp.logical_or(se_flag, nan_mask)

            diff  = test[:, [i], :] - u_hat
            mdiff = diff / _scl             # (M, 1, d)

            nmse_ = cp.mean(mdiff**2, axis=(1,2))    # (M,)
            se_ = cp.linalg.norm(diff, axis=(1,2))**2 / cp.linalg.norm(test[:, i, :], axis=-1)**2                                           # (M,)

            cond_nmse = cp.logical_and(~nmse_flag, nmse_ > error_threshold)
            tau_f_nmse[cond_nmse] = i - 1
            nmse_flag = cp.logical_or(nmse_flag, cond_nmse)

            cond_se = cp.logical_and(~se_flag, se_ > error_threshold)
            tau_f_se[cond_se] = i - 1
            se_flag = cp.logical_or(se_flag, cond_se)

            if cp.logical_and(nmse_flag, se_flag).all():
                break

        last_step = T
        tau_f_nmse[~nmse_flag] = last_step
        tau_f_se[~se_flag]     = last_step

        tau_f_nmse = tau_f_nmse * dt / Lyapunov_time
        tau_f_se   = tau_f_se * dt / Lyapunov_time


        return cp.asnumpy(tau_f_nmse), cp.asnumpy(tau_f_se)

        
    def get_vpt(self, test, dt, Lyapunov_time, error_threshold=0.3**2, return_pred=False):
        """
        Forecast times (Lyapunov units) using NMSE and SE.
        If epsilon is scalar, it's treated as a length-1 vector (P = 1).
        """
        test = cp.asarray(test)
        P = int(np.asarray(self.epsilon).size)
        assert P >= 1

        test_points = test.shape[0]
        d = test.shape[-1]

        # predictions: (T, P, d)
        pred = cp.zeros((test_points, P, d), dtype=test.dtype)

        # initial state (1, d)
        u0 = test[0].reshape(1, -1)
        pred[0] = u0  # broadcast into (P, d) later

        # initial states for each hyperparameter: (P, 1, d)
        u_hat = cp.ones((P, 1, 1)) * u0[None, :, :]  # (P, 1, d)

        nmse_flag  = cp.zeros(P, dtype=cp.bool_)
        se_flag    = cp.zeros(P, dtype=cp.bool_)
        tau_f_nmse = cp.zeros(P, dtype=cp.float64)
        tau_f_se   = cp.zeros(P, dtype=cp.float64)

        _scl = cp.asarray(self.std_nmse).reshape(1, -1)

        for i in range(1, test_points):
            # forecast for all hyperparams at once
            u_hat = self.forecast(u_hat)      # expected shape (P, 1, d) or (P, d)

            # ensure shape (P, d) no matter what forecast returns
            u_hat_squeezed = u_hat.reshape(P, -1)   # (P, d)
            pred[i] = u_hat_squeezed

            # NaN handling: per hyperparam
            nan_mask = cp.isnan(u_hat_squeezed).any(axis=1)  # (P,)
            if nan_mask.any():
                last_idx = i - 1
                # set tau_f for those that have not yet hit threshold
                upd_nmse = cp.logical_and(~nmse_flag, nan_mask)
                upd_se   = cp.logical_and(~se_flag,   nan_mask)
                tau_f_nmse[upd_nmse] = last_idx
                tau_f_se[upd_se]     = last_idx
                nmse_flag = cp.logical_or(nmse_flag, nan_mask)
                se_flag   = cp.logical_or(se_flag,   nan_mask)

            # error metrics
            # test[i] is (d,), broadcast to (P, d)
            diff  = (test[i] - u_hat_squeezed)       # (P, d)
            mdiff = diff / _scl                      # (P, d)

            nmse_ = cp.mean(mdiff**2, axis=-1)       # (P,)
            se_   = (cp.linalg.norm(diff, axis=1)**2
                    / (cp.linalg.norm(test[i])**2)) # (P,)

            # update nmse-based tau_f
            cond_nmse = cp.logical_and(~nmse_flag, nmse_ > error_threshold)
            tau_f_nmse[cond_nmse] = i - 1
            nmse_flag = cp.logical_or(nmse_flag, cond_nmse)

            # update se-based tau_f
            cond_se = cp.logical_and(~se_flag, se_ > error_threshold)
            tau_f_se[cond_se] = i - 1
            se_flag = cp.logical_or(se_flag, cond_se)

            # if all hyperparams are done and we don't need predictions, break
            if cp.logical_and(nmse_flag, se_flag).all() and not return_pred:
                break

        # any hyperparams that never crossed thresholds get full length
        last_step = test_points
        tau_f_nmse[~nmse_flag] = last_step
        tau_f_se[~se_flag]     = last_step

        factor = dt / Lyapunov_time
        tau_f_nmse = tau_f_nmse * factor
        tau_f_se   = tau_f_se * factor

        # ---- Output: if P == 1, squeeze back to scalar-like API ----
        tau_f_nmse_np = cp.asnumpy(tau_f_nmse)
        tau_f_se_np   = cp.asnumpy(tau_f_se)

        if P == 1:
            if return_pred:
                # pred: (T, P, d) -> (T, d)
                return float(tau_f_nmse_np[0]), float(tau_f_se_np[0]), pred[:, 0, :]
            else:
                return float(tau_f_nmse_np[0]), float(tau_f_se_np[0])
        else:
            if return_pred:
                return tau_f_nmse_np, tau_f_se_np, pred
            else:
                return tau_f_nmse_np, tau_f_se_np



