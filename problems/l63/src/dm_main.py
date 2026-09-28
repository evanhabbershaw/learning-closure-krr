from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np
import cupy as cp
import cupyx.scipy.linalg

ArrayLike = Union[np.ndarray, cp.ndarray]


class DMClass:
    """
    - This class supports epsilon/lambda_reg as scalar or (P,) vector.
    - Training kernel:
        * P==1: uses cp.linalg.norm distances
        * P>1: uses a Gram-trick based distance->kernel path
    - Test kernel:
        * x.ndim==2 (requires P==1): uses cp.linalg.norm
        * x.ndim==3: uses Gram trick 
    """

    def __init__(
        self,
        data: ArrayLike,                  # (N, d)
        f: ArrayLike,                     # (N,) or (N, d)
        epsilon: Union[float, ArrayLike], # scalar or (P,)
        lambda_reg: Union[float, ArrayLike],  # scalar or (P,)
        mode: str = "rbf",
        distance_matrix: Optional[ArrayLike] = None,  # (N,N) Euclidean distances (not squared)
        pipeline="single",
    ):
        if mode not in {"rbf", "diffusion"}:
            raise ValueError("mode must be 'rbf' or 'diffusion'")

        self.mode = mode

        self.data_old = cp.asarray(data)
        if self.data_old.ndim != 2:
            raise ValueError("data must be (N, d)")
        self.N, self.d = self.data_old.shape

        self.f = f
        if self.f.ndim == 1:
            self.f = self.f[:, None]
        if self.f.ndim != 2 or self.f.shape[0] != self.N:
            raise ValueError("f must be (N,) or (N, d)")

        epsilon = cp.asarray(epsilon)
        lambda_reg = cp.asarray(lambda_reg)
        if epsilon.ndim == 0:
            epsilon = epsilon[None]
        if lambda_reg.ndim == 0:
            lambda_reg = lambda_reg[None]
        if epsilon.ndim != 1 or lambda_reg.ndim != 1:
            raise ValueError("epsilon and lambda_reg must be scalar or 1D")
        if epsilon.size != lambda_reg.size:
            raise ValueError("epsilon and lambda_reg must have the same length (P)")

        self.epsilon = epsilon
        self.lambda_reg = lambda_reg
        self.P = int(epsilon.size)
        self.pipeline = pipeline
        self.kernel_matrix, self.q1, self.q2, self.distance_matrix = self._compute_kernel_matrix_and_densities(
            self.data_old,
            self.epsilon,
            mode=self.mode,
            distance_matrix=distance_matrix,
            pipeline=pipeline
        )
        self.q1 = cp.asarray(self.q1) if self.q1 is not None else None
        self.q2 = cp.asarray(self.q2) if self.q2 is not None else None

        self._fit(self.f, self.lambda_reg)

    def predict(self, x: ArrayLike) -> cp.ndarray:
        if self.pipeline == "single":
            if self.P != 1:
                raise ValueError("For P>1, call predict with x shaped (m, d)")
            kernel_matrix = self._kernel_eval(x)
            out = cp.dot(kernel_matrix.T, self.beta)
            return out

        elif x.ndim == 2 and self.pipeline == "batch":
            kernel_matrix = self._kernel_eval(x)
            out = cp.einsum("nm,knd->kmd", kernel_matrix, self.beta)
            return out

        elif x.ndim == 3 and self.pipeline == "batch":
            kernel_matrix = self._kernel_eval(x)
            out = cp.einsum("knm,knd->kmd", kernel_matrix, self.beta)
            return out
        
        else:
            raise ValueError(f"x with shape {x.shape} is incompatible with pipeline {self.pipeline}")

    def _fit(self, f: cp.ndarray, lambda_reg: cp.ndarray) -> None:
        if self.pipeline == "single":
            lam0 = lambda_reg[0]
            A = self.kernel_matrix + lam0 * cp.eye(self.N, dtype=self.kernel_matrix.dtype)
            self.beta = self._solve_2d(A, f)
        else:
            lam = cp.asarray(lambda_reg)
            self.beta = self._solve_batched(cp.array(self.kernel_matrix) \
                         + lam[:, None, None] * cp.eye(self.N)[None, :, :], cp.array(f))

    @staticmethod
    def _solve_2d(A: cp.ndarray, B: cp.ndarray) -> cp.ndarray:
        A = cp.asarray(A)
        B = cp.asarray(B)
        if B.ndim == 1:
            B = B[:, None]
        L = cp.linalg.cholesky(A)
        Z0 = cupyx.scipy.linalg.solve_triangular(L, B, lower=True)
        Z1 = cupyx.scipy.linalg.solve_triangular(L, Z0, lower=True, trans=1)
        return Z1

    @staticmethod
    def _solve_batched(A: cp.ndarray, B: cp.ndarray) -> cp.ndarray:
        """
        A: (P, N, N)
        B: (N, d) or (P, N, d)
        returns: (P, N, d)
        """
        A = cp.asarray(A)
        B = cp.asarray(B)

        P, N, N2 = A.shape
        if N != N2:
            raise ValueError("A must be (P,N,N)")

        if B.ndim == 1:
            B = B[:, None]         # (N,1)
            B = B[None, :, :]      # (1,N,1)
            B = cp.broadcast_to(B, (P, N, 1))
        elif B.ndim == 2:
            B = B[None, :, :]      # (1,N,d)
            B = cp.broadcast_to(B, (P, N, B.shape[-1]))
        elif B.ndim == 3:
            if B.shape[0] != P or B.shape[1] != N:
                raise ValueError("For batched solve, B must be (P, N, d)")
        else:
            raise ValueError("Unsupported B.ndim")

        L = cp.linalg.cholesky(A)  # (P,N,N)
        Z1 = cp.empty_like(B)
        for p in range(P):
            Z0 = cupyx.scipy.linalg.solve_triangular(L[p], B[p], lower=True)
            Z1[p] = cupyx.scipy.linalg.solve_triangular(L[p], Z0, trans=1, lower=True)
        return Z1

    def _kernel_eval(self, x: cp.ndarray) -> cp.ndarray:

        X = self.data_old
        N, d = self.data_old.shape

        if x.ndim == 1:
            x = x[None]

        if self.pipeline == "single":
            if self.P != 1:
                raise ValueError("pipeline for single is only supported for P==1")
            if x.shape[1] != d:
                raise ValueError(f"x has d={x.shape[1]} but training data has d={d}")

            dist = cp.linalg.norm(X[:, None] - x[None], axis=-1)
            kernel_matrix = cp.exp(-(dist**2) / (4 * self.epsilon))

            if self.mode == "diffusion":
                q1 = self.q1
                q2 = self.q2
                assert q1 is not None and q2 is not None

                q = cp.power(cp.mean(kernel_matrix, axis=0), -1)
                kernel_matrix = cp.einsum("ij,j,i->ij", kernel_matrix, q, q1)
                q = cp.power(cp.mean(kernel_matrix, axis=0), -1 / 2)
                kernel_matrix = cp.einsum("ij,j,i->ij", kernel_matrix, q, q2)

            return kernel_matrix

        else:
            eps = cp.asarray(self.epsilon)
            P, m, d2 = x.shape
            if d2 != d:
                raise ValueError(f"test data has d={d2} but training data has d={d}")

            Y = x.reshape(P * m, d)

            XX = cp.sum(X * X, axis=1)[:, None]
            YY = cp.sum(Y * Y, axis=1)[None, :]
            dist2 = XX + YY - 2.0 * (X @ Y.T)
            cp.maximum(dist2, 0.0, out=dist2)

            dist2 = dist2.reshape(N, P, m)
            dist2 = dist2.transpose(1, 0, 2)

            eps_b   = eps[:, None, None]
            kernel_matrix = cp.exp(-dist2 / (4.0 * eps_b))

            if self.mode == "diffusion":
                assert self.q1 is not None and self.q2 is not None

                q = cp.power(cp.mean(kernel_matrix, axis=1), -1)
                kernel_matrix = cp.einsum("pnm,pm,pn->pnm", kernel_matrix, q, self.q1)
                q = cp.power(cp.mean(kernel_matrix, axis=1), -1 / 2)
                kernel_matrix = cp.einsum("pnm,pm,pn->pnm", kernel_matrix, q, self.q2)

            return kernel_matrix

    @staticmethod
    def _compute_kernel_matrix_and_densities(
        data: cp.ndarray,
        epsilon: cp.ndarray,
        mode: str,
        distance_matrix: Optional[ArrayLike] = None,
        pipeline : bool = "single"
    ) -> Tuple[cp.ndarray, Optional[cp.ndarray], Optional[cp.ndarray], cp.ndarray]:
        """
        Training kernel builder.

        Compatibility rule:
          - P==1: match Code 2 => use cp.linalg.norm if distance_matrix not provided.
          - P>1: match Code 1 batch style => use Gram trick (dist2 = xx+yy-2xy).
        """
        X = cp.asarray(data, dtype=cp.float64)
        N, d = X.shape

        eps = cp.asarray(epsilon)
        if eps.ndim == 0:
            eps = eps[None]
        P = int(eps.size)

        if distance_matrix is not None:
            base_dist = cp.asarray(distance_matrix)
        else:
            if pipeline=="single":
                distance_matrix = cp.linalg.norm(X[:, None] - X[None, :], axis=-1)  # (N,N)
                base_dist = distance_matrix
            else:
                n = cp.sum(X * X, axis=1) 
                distance_matrix = X @ X.T
                distance_matrix *= -2.0
                distance_matrix += n[:, None]
                distance_matrix += n[None, :]
                cp.maximum(distance_matrix, 0.0, out=distance_matrix)
                cp.sqrt(distance_matrix, out=distance_matrix)
                cp.fill_diagonal(distance_matrix, 0.0)
                distance_matrix = 0.5 * (distance_matrix + distance_matrix.T)
                base_dist = distance_matrix

        if pipeline == "single":
            kernel_matrix = cp.exp(-(base_dist**2) / (4.0 * eps[0]))  # (N,N)

            if mode == "rbf":
                return kernel_matrix, None, None, base_dist

            q1 = cp.power(cp.mean(kernel_matrix, axis=-1), -1)        # (N,)
            kernel_matrix = cp.einsum("ij,i,j->ij", kernel_matrix, q1, q1)
            q2 = cp.power(cp.mean(kernel_matrix, axis=-1), -1 / 2)    # (N,)
            kernel_matrix = cp.einsum("ij,i,j->ij", kernel_matrix, q2, q2)

            return kernel_matrix, q1, q2, base_dist

        else:
            base_dist = base_dist[None, :, :]
            epsilon = epsilon[:, None, None]
            kernel_matrix = cp.exp(-(base_dist**2) / (4 * epsilon))
            if mode == "rbf":
                return kernel_matrix, None, None, base_dist

            q1 = cp.power(cp.mean(kernel_matrix, axis=2), -1)
            kernel_matrix = cp.einsum("kij,ki,kj->kij", kernel_matrix, q1, q1)
            q2 = cp.power(cp.mean(kernel_matrix, axis=2), -1/2)
            kernel_matrix = cp.einsum("kij,ki,kj->kij", kernel_matrix, q2, q2)

            return kernel_matrix, q1, q2, base_dist
