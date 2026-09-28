from __future__ import annotations
import numpy as np
import cupy as cp


class ReducedKSClosureStepper:
    """
    Reduced KS closure stepper matching the midpoint-style update in KSmodel/test.py.

    Supported state/closure representations:
      - rep="spec12"   : 12-real representation [Re v1, Im v1, ..., Re v6, Im v6]
      - rep="coarse13" : 13-real coarse physical representation corresponding to
                         modes k = -6,...,6 with zero mode included as the 0th slot
      - rep="coarse14" : 14-real coarse physical representation corresponding to
                         modes k = -6,...,6 with zero/Nyquist mode included as the 0th/8th slot

    The one-step update uses both theta_n and theta_{n+1}:
        x_{n+1} = Phi(x_n, theta_n, theta_{n+1})
    """

    def __init__(self, *, rep: str, dt: float = 0.05, alpha: float = 0.085):
        self.rep = rep
        self.dt = float(dt)

        # q_k = k * sqrt(alpha), k=0,...,6
        q = cp.arange(7, dtype=cp.float64) * cp.sqrt(alpha) # q_k = k * sqrt(0.085)
        self.ql = q           # store q_k
        self.dl = q**2 - q**4 # store d_k = q_k^2 - q_k^4
        self._init_reduced_convolution_indices()

    # ------------------------------------------------------------------
    # Representation transforms
    # ------------------------------------------------------------------

    def _spec12_to_modes(self, x: cp.ndarray) -> cp.ndarray:
        """
        (..., 12) real -> (..., 7) complex with slot 0 = 0, slots 1:7 = v1..v6
        This converts x=(Re(v_1),Im(v_1),...,Re(v_6),Im(v_6)) in R^{12} to v=(v_0,v_1,...,v_6) in C^7 [with v_0=0]
        """
        x = cp.asarray(x, dtype=cp.float64)
        if x.shape[-1] != 12:
            raise ValueError(f"Expected spec12 state with last dim 12, got {x.shape[-1]}")
        v = cp.zeros(x.shape[:-1] + (7,), dtype=cp.complex128) 
        v[..., 1:] = x[..., 0::2] + 1j * x[..., 1::2] # takes 0,2,4,6,8,10 elts of x to real part, 1,3,5,7,9,11 elts of x to complex part
        return v # vector of (v_0,v_1,...,v_6) in C^7 ; v_0 is left as zero

    def _modes_to_spec12(self, v: cp.ndarray) -> cp.ndarray:
        """
        (..., 7) complex -> (..., 12) real from slots 1:7
        """
        v = cp.asarray(v, dtype=cp.complex128)
        out = cp.empty(v.shape[:-1] + (12,), dtype=cp.float64)
        out[..., 0::2] = cp.real(v[..., 1:]) # store Re(v1),Re(v2),...,Re(v6) into elt (0,2,...,10) # ignore zero elt
        out[..., 1::2] = cp.imag(v[..., 1:]) # store Im(v1),Im(v2),...,Im(v6) into elt (0,2,...,10) # ignore zero elt
        return out

    def _coarse13_to_modes(self, x: cp.ndarray) -> cp.ndarray:
        """
        (..., 13) real coarse physical -> (..., 7) complex modes [v0,...,v6]
        """
        x = cp.asarray(x, dtype=cp.float64)
        if x.shape[-1] != 13:
            raise ValueError(f"Expected coarse13 state with last dim 13, got {x.shape[-1]}")
        spec = cp.fft.fft(x, axis=-1)
        return spec[..., :7].astype(cp.complex128)

    def _modes_to_coarse13(self, v: cp.ndarray) -> cp.ndarray:
        """
        (..., 7) complex modes [v0,...,v6] -> (..., 13) real coarse physical
        """
        v = cp.asarray(v, dtype=cp.complex128)
        spec = cp.zeros(v.shape[:-1] + (13,), dtype=cp.complex128)
        spec[..., :7] = v
        spec[..., 7:] = cp.conj(v[..., 1:][..., ::-1])
        x = cp.fft.ifft(spec, axis=-1)
        return cp.real(x).astype(cp.float64)

    def _coarse14_to_modes(self, x: cp.ndarray) -> cp.ndarray:
        """
        (..., 14) real coarse physical -> (..., 7) complex modes [v0,...,v6].

        FFT ordering for length 14 is:
            0, 1, ..., 6, 7, -6, ..., -1

        The Nyquist slot k=7 is ignored and assumed to be zero.
        """
        x = cp.asarray(x, dtype=cp.float64)
        if x.shape[-1] != 14:
            raise ValueError(f"Expected coarse14 state with last dim 14, got {x.shape[-1]}")
        spec = cp.fft.fft(x, axis=-1) # perform fft to get (v_0=0 , v_1,...,v_6, v_7=0 ,v_{-6},...,v_{-1})
        return spec[..., :7].astype(cp.complex128) # extract (v_0=0 , v_1,...,v_6)

    def _modes_to_coarse14(self, v: cp.ndarray) -> cp.ndarray:
        """
        (..., 7) complex modes [v0,...,v6] -> (..., 14) real coarse physical.

        The extra Nyquist slot is set to zero.
        """
        v = cp.asarray(v, dtype=cp.complex128)
        spec = cp.zeros(v.shape[:-1] + (14,), dtype=cp.complex128) # form this vector to then take ifft

        # modes k=0,1,...,6
        spec[..., :7] = v

        # mode k=7, the Nyquist slot, stays zero
        spec[..., 7] = 0.0

        # modes k=-6,...,-1
        spec[..., 8:] = cp.conj(v[..., 1:][..., ::-1]) # v_{-k} = overline{v_k}, stored in reverse order

        x = cp.fft.ifft(spec, axis=-1) # ifft on constructed vector from C^7
        return cp.real(x).astype(cp.float64)

    def _phys14_to_modes(self, x: cp.ndarray) -> cp.ndarray:
        """
        (..., 14) real physical-space coordinates -> (..., 7) complex modes.

        Convention:
            x = 14 * ifft(spec14).real

        Therefore:
            spec14 = fft(x) / 14.
        """
        x = cp.asarray(x, dtype=cp.float64)
        if x.shape[-1] != 14:
            raise ValueError(f"Expected phys14 state with last dim 14, got {x.shape[-1]}")

        spec = cp.fft.fft(x, axis=-1) / 14.0
        return spec[..., :7].astype(cp.complex128)


    def _modes_to_phys14(self, v: cp.ndarray) -> cp.ndarray:
        """
        (..., 7) complex modes [v0,...,v6] -> (..., 14) real physical coordinates.

        Convention:
            x = 14 * ifft(spec14).real
        """
        v = cp.asarray(v, dtype=cp.complex128)

        spec = cp.zeros(v.shape[:-1] + (14,), dtype=cp.complex128)

        # modes k = 0,1,...,6
        spec[..., :7] = v

        # Nyquist slot k=7 stays zero
        spec[..., 7] = 0.0

        # modes k=-6,...,-1
        spec[..., 8:] = cp.conj(v[..., 1:][..., ::-1])

        x = 14.0 * cp.fft.ifft(spec, axis=-1)
        return cp.real(x).astype(cp.float64)

    def state_to_modes(self, x: cp.ndarray) -> cp.ndarray:
        """
        this converts current state representation into the common internal mode representation (v_0,v_1,...,v_6)
        in modes, the time update doesn't care whether input is from spec12, coarse13, coarse14, or phys14 , since it's all just modes
        """
        if self.rep == "spec12":
            return self._spec12_to_modes(x)
        elif self.rep == "coarse13":
            return self._coarse13_to_modes(x)
        elif self.rep == "coarse14":
            return self._coarse14_to_modes(x)
        elif self.rep == "phys14":
            return self._phys14_to_modes(x)
        else:
            raise ValueError(f"Unknown rep: {self.rep}")

    def modes_to_state(self, v: cp.ndarray) -> cp.ndarray:
        """
        this converts modes (v_0,v_1,...,v_6) into desired state spec12, coarse13, or coarse14
        """
        if self.rep == "spec12":
            return self._modes_to_spec12(v)
        elif self.rep == "coarse13":
            return self._modes_to_coarse13(v)
        elif self.rep == "coarse14":
            return self._modes_to_coarse14(v)
        elif self.rep == "phys14":
            return self._modes_to_phys14(v)
        else:
            raise ValueError(f"Unknown rep: {self.rep}")

    # ------------------------------------------------------------------
    # Reduced midpoint update
    # ------------------------------------------------------------------

    # def _build_full_modes(self, v: cp.ndarray) -> cp.ndarray:
    #     """
    #     Given positive spectral modes v[..., 0:7] = (v_0, v_1, ..., v_6),
    #     build a bookkeeping array containing modes p=-6,...,6.

    #     This length-13 array is NOT the coarse13 physical representation.
    #     It is only used internally to compute convolution sums over
    #     positive and negative Fourier modes.

    #     Index convention:
    #         full[..., p + 6] = v_p,  p=-6,...,6.

    #     The negative modes are reconstructed using real-valued symmetry:
    #         v_{-k} = conjugate(v_k).
        
    #     Build full modes indexed by p=-6,...,6 in a length-13 array.
    #     Stored as:
    #         full[..., 0]   = mode -6
    #         ...
    #         full[..., 6]   = mode 0
    #         ...
    #         full[..., 12]  = mode +6
    #     """
    #     full = cp.zeros(v.shape[:-1] + (13,), dtype=cp.complex128)
    #     full[..., 6] = v[..., 0]  # zero mode
    #     for k in range(1, 7):
    #         full[..., 6 + k] = v[..., k]          # mode +k : store complex modes into full vector 
    #         full[..., 6 - k] = cp.conj(v[..., k]) # mode -k : negative modes: v_{-k} = overline{v_k}
    #     return full
    def _build_full_modes(self, v: cp.ndarray) -> cp.ndarray:
        """
        This version is vectorized !!!
        Build full modes indexed by p=-6,...,6.

        full[..., p+6] = v_p.
        """
        v = cp.asarray(v, dtype=cp.complex128)

        full = cp.zeros(v.shape[:-1] + (13,), dtype=cp.complex128)

        # mode 0
        full[..., 6] = v[..., 0]

        # positive modes +1,...,+6
        full[..., 7:13] = v[..., 1:7]

        # negative modes -6,...,-1
        full[..., 0:6] = cp.conj(v[..., 6:0:-1])

        return full

    # def _nonlinear_sum(self, v: cp.ndarray) -> cp.ndarray:
    #     """
    #     Compute the resolved convolution coefficients
    #         NL_k(v) = sum_p v_p v_{k-p},     k = 1,...,6,
    #     where only modes p = -6,...,6 are retained (use the conjugate reconstruction for negative modes.)

    #     This is an exact reduced convolution sum. It does NOT compute u^2
    #     on a length-14 physical grid, so it avoids aliasing from modes such
    #     as k=12 wrapping back into low modes.
    #     """
    #     full = self._build_full_modes(v)
    #     out = cp.zeros_like(v)

    #     for k in range(1, 7):
    #         s = 0.0 + 0.0j
    #         for p in range(-6, 7):
    #             q = k - p
    #             if -6 <= q <= 6:
    #                 s = s + full[..., p + 6] * full[..., q + 6]
    #         out[..., k] = s

    #     return out
    def _nonlinear_sum(self, v: cp.ndarray) -> cp.ndarray:
        """
        Vectorized exact reduced convolution.

        Computes

            NL_k(v) = sum_p v_p v_{k-p},     k=1,...,6.

        Returns an array shaped like v, with slot 0 set to zero.
        """
        v = cp.asarray(v, dtype=cp.complex128)

        full = self._build_full_modes(v)

        terms = full[..., self._conv_p_idx] * full[..., self._conv_q_idx]
        sums = cp.sum(terms * self._conv_mask, axis=-1)

        out = cp.zeros_like(v)
        out[..., 1:7] = sums

        return out

    def step(self, x_prev: cp.ndarray, theta_prev: cp.ndarray, theta_next: cp.ndarray) -> cp.ndarray:
        """
        One midpoint-style reduced step:
            x_{n+1} = Phi(x_n, theta_n, theta_{n+1})

        Inputs and output are all in the same chosen representation.
        Supports batch shape (..., d).
        """
        x_prev = cp.asarray(x_prev, dtype=cp.float64)
        theta_prev = cp.asarray(theta_prev, dtype=cp.float64)
        theta_next = cp.asarray(theta_next, dtype=cp.float64)

        v = self.state_to_modes(x_prev)              # (..., 7), complex # this is x_n (?)
        f = self.state_to_modes(theta_prev)          # (..., 7), complex # this is theta_n
        ff = self.state_to_modes(theta_next)         # (..., 7), complex # this is theta_{n+1}

        # Stage 1 # this is FE-like slope over one step
        F1 = cp.zeros_like(v)
        NL1 = self._nonlinear_sum(v)
        F1[..., 1:] = ((-1j) / 2.0) * self.ql[1:] * (NL1[..., 1:] + f[..., 1:]) * self.dt \
                      + v[..., 1:] * self.dl[1:] * self.dt

        # Midpoint stage
        v_mid = v + 0.5 * F1 # the midpoint slope approx ... the forcing is averaged over current and next time levels
        F2 = cp.zeros_like(v)
        NL2 = self._nonlinear_sum(v_mid)
        F2[..., 1:] = ((-1j) / 2.0) * self.ql[1:] * (NL2[..., 1:] + 0.5 * (f[..., 1:] + ff[..., 1:])) * self.dt \
                      + v_mid[..., 1:] * self.dl[1:] * self.dt

        v_next = v + F2                    # the next step update
        return self.modes_to_state(v_next) # finally, convert back to original state
    
    def _init_reduced_convolution_indices(self):
        """
        Precompute index arrays for the exact reduced convolution

            sum_p v_p v_{k-p},     k = 1,...,6,

        with retained modes p=-6,...,6.
        """
        Kres = 6
        max_pairs = 2 * Kres + 1

        p_idx = np.zeros((Kres, max_pairs), dtype=np.int64)
        q_idx = np.zeros((Kres, max_pairs), dtype=np.int64)
        mask = np.zeros((Kres, max_pairs), dtype=np.float64)

        for row, k in enumerate(range(1, Kres + 1)):
            pairs = []

            for p in range(-Kres, Kres + 1):
                q = k - p
                if -Kres <= q <= Kres:
                    pairs.append((p + Kres, q + Kres))

            for j, (ip, iq) in enumerate(pairs):
                p_idx[row, j] = ip
                q_idx[row, j] = iq
                mask[row, j] = 1.0

        self._conv_p_idx = cp.asarray(p_idx)
        self._conv_q_idx = cp.asarray(q_idx)
        self._conv_mask = cp.asarray(mask)


class ReducedKSExactConvETDRK4Stepper(ReducedKSClosureStepper):
    """
    Reduced six-mode KS stepper using ETDRK4 with exact convolution sums.

    This class reuses the representation maps from ReducedKSClosureStepper:

        spec12   <-> modes v_0,...,v_6
        coarse13 <-> modes v_0,...,v_6
        coarse14 <-> modes v_0,...,v_6

    The evolution is performed directly on the six retained complex modes,
    using the exact reduced convolution

        sum_p v_p v_{k-p},

    rather than squaring on a coarse physical grid. Therefore this avoids
    length-13 or length-14 aliasing.

    This class is intended for the A1 residual workflow:

        x_{n+1} = G_ETDRK4(x_n) + R_hat(x_n).

    It does not include theta forcing inside the time step. It is just the
    known reduced model G_ETDRK4.
    """

    def __init__(
        self,
        *,
        rep: str,
        dt: float = 0.05,
        alpha: float = 0.085,
        substeps: int = 10,
        M: int = 16,
    ):
        super().__init__(rep=rep, dt=dt, alpha=alpha)

        self.obs_dt = float(dt)
        self.substeps = int(substeps)
        self.h = self.obs_dt / self.substeps
        self.M = int(M)

        # Linear symbol d_k = q_k^2 - q_k^4, k=0,...,6.
        D = self.dl.astype(cp.complex128)

        self.E = cp.exp(self.h * D)
        self.E2 = cp.exp(0.5 * self.h * D)

        # Kassam--Trefethen contour averages.
        r = cp.exp(
            1j * cp.pi * (cp.arange(1, self.M + 1, dtype=cp.float64) - 0.5) / self.M
        )
        LR = self.h * D[:, None] + r[None, :]

        av = lambda X: cp.real(cp.mean(X, axis=1))

        self.Q = self.h * av((cp.exp(LR / 2.0) - 1.0) / LR)

        self.f1 = self.h * av(
            (-4.0 - LR + cp.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3
        )

        self.f2 = self.h * av(
            (2.0 + LR + cp.exp(LR) * (-2.0 + LR)) / LR**3
        )

        self.f3 = self.h * av(
            (-4.0 - 3.0 * LR - LR**2 + cp.exp(LR) * (4.0 - LR)) / LR**3
        )

    def _nonlinear_rhs(self, v: cp.ndarray) -> cp.ndarray:
        """
        Compute the nonlinear part N(v), excluding the linear term.

        For k=1,...,6,

            N_k(v) = -(i q_k / 2) sum_p v_p v_{k-p}.

        The convolution sum is computed exactly over p=-6,...,6
        using _nonlinear_sum.
        """
        v = cp.asarray(v, dtype=cp.complex128)

        out = cp.zeros_like(v)
        NL = self._nonlinear_sum(v)

        out[..., 1:] = (-0.5j) * self.ql[1:] * NL[..., 1:]

        # Mode zero remains zero.
        out[..., 0] = 0.0

        return out

    def _etdrk4_step_modes(self, v: cp.ndarray) -> cp.ndarray:
        """
        One internal ETDRK4 step on modes v_0,...,v_6.

        The reduced ODE is

            v_t = D v + N(v),

        where D is diagonal with entries d_k = q_k^2 - q_k^4,
        and N(v) is computed by exact convolution.
        """
        v = cp.asarray(v, dtype=cp.complex128)

        Nv = self._nonlinear_rhs(v)

        a = self.E2 * v + self.Q * Nv
        Na = self._nonlinear_rhs(a)

        b = self.E2 * v + self.Q * Na
        Nb = self._nonlinear_rhs(b)

        c = self.E2 * a + self.Q * (2.0 * Nb - Nv)
        Nc = self._nonlinear_rhs(c)

        v_next = (
            self.E * v
            + self.f1 * Nv
            + 2.0 * self.f2 * (Na + Nb)
            + self.f3 * Nc
        )

        # Keep zero mode fixed at zero.
        v_next[..., 0] = 0.0

        return v_next

    def step(self, x_prev: cp.ndarray) -> cp.ndarray:
        """
        Advance one observed time step Delta t = self.obs_dt.

        Internally uses self.substeps ETDRK4 steps of size

            h = Delta t / substeps.

        Input and output are in the chosen representation.
        """
        x_prev = cp.asarray(x_prev, dtype=cp.float64)

        v = self.state_to_modes(x_prev)

        for _ in range(self.substeps):
            v = self._etdrk4_step_modes(v)

        return self.modes_to_state(v)

class ReducedKSForcedETDRK4Stepper(ReducedKSExactConvETDRK4Stepper):
    """
    Reduced six-mode KS ETDRK4 stepper with additive RHS closure forcing.

    This advances

        v_t = D v + N_res(v) + theta(t),

    where theta is already the RHS closure term, i.e.

        theta_k = -(i q_k / 2) * unresolved_convolution_k.

    State and theta are both provided in the same representation, e.g. rep="phys14".
    """

    def _etdrk4_step_modes_forced(
        self,
        v: cp.ndarray,
        theta_left: cp.ndarray,
        theta_mid: cp.ndarray,
        theta_right: cp.ndarray,
    ) -> cp.ndarray:
        """
        One internal ETDRK4 step with prescribed theta at the left,
        midpoint, and right endpoint of the internal substep.

        All arrays are in mode representation [v_0, v_1, ..., v_6].
        """
        v = cp.asarray(v, dtype=cp.complex128)

        Nv = self._nonlinear_rhs(v) + theta_left

        a = self.E2 * v + self.Q * Nv
        Na = self._nonlinear_rhs(a) + theta_mid

        b = self.E2 * v + self.Q * Na
        Nb = self._nonlinear_rhs(b) + theta_mid

        c = self.E2 * a + self.Q * (2.0 * Nb - Nv)
        Nc = self._nonlinear_rhs(c) + theta_right

        v_next = (
            self.E * v
            + self.f1 * Nv
            + 2.0 * self.f2 * (Na + Nb)
            + self.f3 * Nc
        )

        v_next[..., 0] = 0.0
        return v_next

    def step_with_theta(
        self,
        x_prev: cp.ndarray,
        theta_prev: cp.ndarray,
        theta_next: cp.ndarray,
    ) -> cp.ndarray:
        """
        Advance from x_n to x_{n+1} using theta_n and theta_{n+1}.

        Uses linear interpolation of theta over the observed interval.
        Internally, the observed step dt is split into self.substeps
        ETDRK4 steps.
        """
        x_prev = cp.asarray(x_prev, dtype=cp.float64)
        theta_prev = cp.asarray(theta_prev, dtype=cp.float64)
        theta_next = cp.asarray(theta_next, dtype=cp.float64)

        v = self.state_to_modes(x_prev)

        th0 = self.state_to_modes(theta_prev)
        th1 = self.state_to_modes(theta_next)

        v_next = self.step_modes_with_theta_modes(v, th0, th1)

        # th0[..., 0] = 0.0
        # th1[..., 0] = 0.0

        # for s in range(1, self.substeps + 1):
        #     frac_left = (s - 1) / self.substeps
        #     frac_mid = (s - 0.5) / self.substeps
        #     frac_right = s / self.substeps

        #     theta_left = (1.0 - frac_left) * th0 + frac_left * th1
        #     theta_mid = (1.0 - frac_mid) * th0 + frac_mid * th1
        #     theta_right = (1.0 - frac_right) * th0 + frac_right * th1

        #     v = self._etdrk4_step_modes_forced(
        #         v,
        #         theta_left,
        #         theta_mid,
        #         theta_right,
        #     )

        return self.modes_to_state(v_next)
    
    def step_modes_with_theta_modes(
        self,
        v_prev: cp.ndarray,
        theta_prev: cp.ndarray,
        theta_next: cp.ndarray,
    ) -> cp.ndarray:
        """
        Advance one observed step directly in mode space.

        Inputs:
            v_prev     : (..., 7) complex modes [v0,...,v6]
            theta_prev : (..., 7) complex RHS closure modes
            theta_next : (..., 7) complex RHS closure modes

        Returns:
            v_next     : (..., 7) complex modes
        """
        v = cp.asarray(v_prev, dtype=cp.complex128)
        th0 = cp.asarray(theta_prev, dtype=cp.complex128)
        th1 = cp.asarray(theta_next, dtype=cp.complex128)

        th0[..., 0] = 0.0
        th1[..., 0] = 0.0

        for s in range(1, self.substeps + 1):
            frac_left = (s - 1) / self.substeps
            frac_mid = (s - 0.5) / self.substeps
            frac_right = s / self.substeps

            theta_left = (1.0 - frac_left) * th0 + frac_left * th1
            theta_mid = (1.0 - frac_mid) * th0 + frac_mid * th1
            theta_right = (1.0 - frac_right) * th0 + frac_right * th1

            v = self._etdrk4_step_modes_forced(
                v,
                theta_left,
                theta_mid,
                theta_right,
            )

        v[..., 0] = 0.0
        return v
    
    def _interp_theta_modes_from_history(
        self,
        theta_hist_modes,
        theta_next_modes,
        tau: float,
        interp_degree: int = 3,
    ) -> cp.ndarray:
        """
        Interpolate theta at time t_n + tau * Delta t.

        theta_hist_modes is ordered as
            [theta_n, theta_{n-1}, theta_{n-2}, ...]

        theta_next_modes is theta_{n+1}.

        The interpolation nodes, relative to t_n, are
            0, -1, -2, ..., 1.

        For interp_degree=3, this uses at most
            theta_n, theta_{n-1}, theta_{n-2}, theta_{n+1}.

        If not enough history is available, it automatically falls back:
            2 nodes -> linear,
            3 nodes -> quadratic,
            4 nodes -> cubic.
        """
        if interp_degree < 1:
            raise ValueError("interp_degree must be at least 1.")

        # Need at most interp_degree history values plus theta_{n+1}.
        # For cubic, use [theta_n, theta_{n-1}, theta_{n-2}] plus theta_{n+1}.
        hist_len = min(len(theta_hist_modes), interp_degree)

        nodes = list(theta_hist_modes[:hist_len]) + [theta_next_modes]
        offsets = [float(-j) for j in range(hist_len)] + [1.0]

        out = cp.zeros_like(nodes[0])

        for j, xj in enumerate(offsets):
            w = 1.0
            for m, xm in enumerate(offsets):
                if m != j:
                    w *= (tau - xm) / (xj - xm)

            out = out + w * nodes[j]

        out[..., 0] = 0.0
        return out

    def step_with_theta_history(
        self,
        x_prev: cp.ndarray,
        theta_history,
        theta_next: cp.ndarray,
        *,
        interp_degree: int = 3,
    ) -> cp.ndarray:
        """
        Advance from x_n to x_{n+1} using a polynomial interpolation of theta.

        Parameters
        ----------
        x_prev:
            State x_n in the chosen representation, e.g. phys14.

        theta_history:
            List ordered as
                [theta_n, theta_{n-1}, theta_{n-2}, ...]
            in the chosen representation.

        theta_next:
            Predicted or true theta_{n+1} in the chosen representation.

        interp_degree:
            Maximum interpolation degree. Use 1 for linear, 2 for quadratic,
            3 for cubic.
        """
        x_prev = cp.asarray(x_prev, dtype=cp.float64)
        theta_next = cp.asarray(theta_next, dtype=cp.float64)

        if not isinstance(theta_history, (list, tuple)):
            raise TypeError("theta_history must be a list or tuple.")

        if len(theta_history) == 0:
            raise ValueError("theta_history must contain at least theta_n.")

        v = self.state_to_modes(x_prev)

        theta_hist_modes = [
            self.state_to_modes(cp.asarray(th, dtype=cp.float64))
            for th in theta_history
        ]
        theta_next_modes = self.state_to_modes(theta_next)

        for th in theta_hist_modes:
            th[..., 0] = 0.0
        theta_next_modes[..., 0] = 0.0

        for s in range(1, self.substeps + 1):
            frac_left = (s - 1) / self.substeps
            frac_mid = (s - 0.5) / self.substeps
            frac_right = s / self.substeps

            theta_left = self._interp_theta_modes_from_history(
                theta_hist_modes,
                theta_next_modes,
                frac_left,
                interp_degree=interp_degree,
            )

            theta_mid = self._interp_theta_modes_from_history(
                theta_hist_modes,
                theta_next_modes,
                frac_mid,
                interp_degree=interp_degree,
            )

            theta_right = self._interp_theta_modes_from_history(
                theta_hist_modes,
                theta_next_modes,
                frac_right,
                interp_degree=interp_degree,
            )

            v = self._etdrk4_step_modes_forced(
                v,
                theta_left,
                theta_mid,
                theta_right,
            )

        return self.modes_to_state(v)

