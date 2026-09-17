"""evaluate V_delta(beta) using the paper's scalar reduction or gamma formulation

RobustRisk.normalized(X, y, delta, p) takes the Wasserstein radius delta
RobustRisk(X, y, delta**p, p) bounds the average cost by mean(t**p) <= delta**p
at p=infinity, both take delta directly and each perturbation length satisfies t_i <= delta

RobustRisk maps beta to |r| and B=||beta||_*, then calls _ScalarRisk.solve
_ScalarRisk.solve selects the explicit cases or solves for lambda, maximizing over t inside
primal and dual return mean risk V_delta by default, or n V_delta with per_sample=False
see docs/robust_risk.md for the equations, call structure, and examples
"""

import cvxpy as cp
import numpy as np
from scipy.optimize import brentq


class RobustRisk:
    """evaluate risk for nonempty data, delta >= 0, 2 <= p <= infinity, and norm >= 1"""

    def __init__(self, X, y, delta, p, norm=np.inf, *, normalized=False):
        """accept the average-cost bound as delta, or the radius when normalized=True

        for finite p, pass delta**p to bound mean(t**p), or use normalized(X, y, delta, p)
        at p=infinity, delta always denotes the radius
        self.delta stores the radius in either case
        """
        self.X = X
        self.y = y
        self.n = len(y)
        self.delta = delta if normalized or np.isinf(p) else delta ** (1 / p)
        self.p = p
        self.q = _conjugate(p)
        self.norm_dual = _conjugate(norm)

    @classmethod
    def normalized(cls, X, y, delta, p, norm=np.inf):
        """accept the Wasserstein radius delta and construct V_delta"""
        return cls(X, y, delta, p, norm, normalized=True)

    def primal(self, beta, per_sample=True):
        """evaluate mean risk V_delta through the scalar reduction in eqs (14)/(35)

        pass per_sample=False to return summed loss n V_delta
        """
        residual_abs = np.abs(self.X @ beta - self.y)
        B = np.linalg.norm(beta, self.norm_dual)
        n_value = _ScalarRisk(self.delta, self.p).solve(residual_abs, B)
        return n_value / self.n if per_sample else n_value

    def dual(self, beta, per_sample=True):
        """evaluate mean risk V_delta through the concave gamma maximization in eq (20)

        pass per_sample=False to return summed loss n V_delta
        """
        residual_abs = np.abs(self.X @ beta - self.y)
        B = np.linalg.norm(beta, self.norm_dual)
        gamma = cp.Variable(self.n, nonneg=True)
        objective = (
            self.n ** (1 / self.p)
            * self.delta
            * B
            * cp.power(cp.sum(gamma), 1 / self.q)
            + residual_abs @ cp.power(gamma, 1 / self.q)
            - cp.sum(cp.power(gamma, 2 / self.q)) / 4
        )
        problem = cp.Problem(cp.Maximize(objective))
        problem.solve()
        return problem.value / self.n if per_sample else problem.value


class _ScalarRisk:
    """solve inf_lambda [n delta**p lambda + sum_i sup_t f_i(t)] in eqs (14)/(35)

    f_i(t) = (|r_i| + B*t)**2 - lambda*t**p for finite p
    solve handles the explicit cases before calling _solve_finite_p
    _solve_finite_p calls _solve_lambda, which calls _maximize_t for each candidate lambda
    """

    def __init__(self, delta, p):
        self.delta = delta
        self.p = p

    def solve(self, residual_abs, B):
        """return the summed risk for residual magnitudes |r| and B=||beta||_*"""
        if self.delta == 0:
            # no perturbations are allowed
            return float(np.sum(residual_abs**2))
        if np.isinf(self.p):
            # each observation can use t_i = delta in adversarial training
            return float(np.sum((residual_abs + B * self.delta) ** 2))
        if B == 0:
            # moving covariates cannot change a zero-coefficient prediction
            return float(np.sum(residual_abs**2))
        if self.p == 2:
            return self._sqrt_lasso(residual_abs, B)
        return self._solve_finite_p(residual_abs, B)

    def _sqrt_lasso(self, residual_abs, B):
        """evaluate n*V_delta(beta) = (||r||_2 + sqrt(n)*delta*B)**2, proposition 2"""
        residual_norm = float(np.linalg.norm(residual_abs))
        return (residual_norm + np.sqrt(len(residual_abs)) * self.delta * B) ** 2

    def _solve_finite_p(self, residual_abs, B):
        """solve for lambda and t at 2 < p < infinity, delta > 0, B > 0

        use t = delta*u, a = |r|/scale, b = delta*B/scale so mean(u**p) = 1
        lambda below is the scaled penalty lambda_original * delta**p / scale**2
        scaling avoids forming a potentially overflowing original lambda for a finite risk
        """
        scale = max(float(np.max(residual_abs)), self.delta * B)
        a, b = residual_abs / scale, self.delta * B / scale
        lam, u = self._solve_lambda(a, b)
        n_value = self._checked_value(a, b, lam, u)
        return n_value * scale**2

    def _solve_lambda(self, a, b):
        """find lambda with mean(u(lambda)**p) = 1, the scaled outer first-order condition"""
        # f_i'(1) = 0 gives endpoints where all maximizing u_i are >= 1 or <= 1
        lam_lo = 2 * b * (float(np.min(a)) + b) / self.p
        lam_hi = 2 * b * (float(np.max(a)) + b) / self.p
        if not np.isfinite(lam_hi) or lam_lo <= 0:
            raise FloatingPointError(
                "Scalar risk is outside the supported floating-point scale"
            )
        if lam_lo == lam_hi:
            return lam_hi, np.ones_like(a)

        def cost_excess(relative_lam):
            u = self._maximize_t(a, b, relative_lam * lam_hi)
            return np.mean(u**self.p) ** (1 / self.p) - 1

        # search in lambda/lam_hi so tolerances do not depend on the size of lambda
        relative_lam, result = brentq(
            cost_excess,
            lam_lo / lam_hi,
            1.0,
            xtol=1e-14,
            rtol=1e-13,
            maxiter=1000,
            full_output=True,
        )
        if not result.converged:
            raise RuntimeError(f"Scalar risk root solve failed: {result.flag}")
        lam = relative_lam * lam_hi
        return lam, self._maximize_t(a, b, lam)

    def _maximize_t(self, residual_abs, B, lam):
        """argmax_{t>=0} f(t) per datapoint, using unimodality after eq (35)

        f(t) = (|r| + B*t)**2 - lambda*t**p, so bisect f' for the sign change
        the scaled solver passes a, b, and scaled lambda to this same problem and obtains u
        """

        def f_prime(t):
            return 2 * B * (residual_abs + t * B) - lam * self.p * t ** (self.p - 1)

        lo = np.zeros_like(residual_abs)
        hi = np.ones_like(residual_abs)
        for _ in range(100):  # grow hi until f' <= 0, at or past the maximizer
            derivative = f_prime(hi)
            rising = derivative > 0
            if not rising.any():
                break
            hi = np.where(rising, 2 * hi, hi)  # double only endpoints where f' is still positive
        derivative = f_prime(hi)
        if np.any(derivative > 0) or not np.isfinite(derivative).all():
            raise FloatingPointError(
                "Could not bracket the inner perturbation maximizer"
            )

        for _ in range(100):
            mid = (lo + hi) / 2
            rising = f_prime(mid) > 0
            lo = np.where(rising, mid, lo)
            hi = np.where(rising, hi, mid)
        return hi

    def _checked_value(self, a, b, lam, u):
        """evaluate the scaled scalar objective and compare it with a feasible perturbation loss

        shrink u only if mean(u**p) > 1, giving a feasible lower bound
        the scalar expression is an upper bound with exact inner maximizers
        their numerical agreement checks the solve, without giving a rigorous error certificate
        """
        n = len(a)
        upper = n * lam + np.sum((a + b * u) ** 2 - lam * u**self.p)
        feasible_u = u / max(1.0, float(np.mean(u**self.p)) ** (1 / self.p))
        lower = float(np.sum((a + b * feasible_u) ** 2))
        if not np.isfinite(upper) or abs(upper - lower) > 1e-8 * max(upper, lower):
            raise RuntimeError(
                f"Scalar risk bounds disagree: lower={lower}, upper={upper}"
            )
        return float(upper)


def _conjugate(exponent):
    """return the conjugate exponent, with 1 and infinity as endpoints"""
    if exponent == 1:
        return np.inf
    if np.isinf(exponent):
        return 1.0
    return exponent / (exponent - 1.0)
