"""
evaluate V_delta(beta) using the paper's scalar reduction or gamma formulation

RobustRisk(X, y, delta, p) takes the Wasserstein radius delta

for finite p, perturbation lengths satisfy mean(t**p) <= delta**p,
at p=infinity, each perturbation length satisfies t_i <= delta

the scalar evaluator solves for lambda, maximizing over perturbation lengths inside
primal and dual return mean risk V_delta by default, or n V_delta with per_sample=False

see docs/robust_risk.md for the equations, call structure, and examples
"""

import cvxpy as cp
import numpy as np
from scipy.optimize import elementwise, minimize_scalar


class RobustRisk:
    """
    evaluate risk

    assume nonempty data, delta >= 0, 2 <= p <= infinity, and norm >= 1
    """

    def __init__(self, X, y, delta, p, norm=np.inf):
        """construct V_delta with Wasserstein radius delta"""
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.n = len(y)
        self.delta = delta
        self.p = p
        self.q = _conjugate(p)
        self.norm_dual = _conjugate(norm)

    def primal(self, beta, per_sample=True, *, return_t=False):
        """
        evaluate mean risk V_delta through the scalar reduction in eq. (16)

        pass per_sample=False to return summed loss n V_delta
        pass return_t=True to return (value, maximizing perturbation lengths)
        """

        # form residual and B
        residual_abs = np.abs(self.X @ beta - self.y)
        B = np.linalg.norm(beta, self.norm_dual)

        # compute n V_delta with the _ScalarRisk class
        result = _ScalarRisk(self.delta, self.p).solve(residual_abs, B, return_t=return_t)
        if return_t:
            n_value, t = result
            return (n_value / self.n if per_sample else n_value), t
        return result / self.n if per_sample else result

    def dual(self, beta, per_sample=True):
        """
        evaluate mean risk V_delta through the concave gamma maximization in eq. (22)

        pass per_sample=False to return summed loss n V_delta
        """

        # form residual, B, gamma as cp var
        residual_abs = np.abs(self.X @ beta - self.y)
        B = np.linalg.norm(beta, self.norm_dual)
        gamma = cp.Variable(self.n, nonneg=True)

        # form objective
        objective = (
            self.n ** (1 / self.p)
            * self.delta
            * B
            * cp.power(cp.sum(gamma), 1 / self.q, max_denom=65536)
            + residual_abs @ cp.power(gamma, 1 / self.q, max_denom=65536)
            - cp.sum(cp.power(gamma, 2 / self.q, max_denom=65536)) / 4
        )

        # solve
        problem = cp.Problem(cp.Maximize(objective))
        problem.solve()

        return problem.value / self.n if per_sample else problem.value


class _ScalarRisk:
    """
    scalar reduction from eq. (16) inf_lambda n*delta**p*lambda + sum_i sup_t f_i(t)

    here, f_i(t) = (|r_i| + B*t)**2 - lambda*t**p

    the outer optimum spends the transport budget, i.e., mean(t**p) = delta**p
    """

    def __init__(self, delta, p):
        self.delta = delta
        self.p = p

    def solve(self, residual_abs, B, *, return_t=False):
        """
        return summed risk

        pass return_t=True to also return the maximizing lengths
        p=2 and p=infinity have explicit formulas
        """

        residual_abs = np.asarray(residual_abs, dtype=float)
        n = len(residual_abs)

        # divide into cases (see docs)
        if self.delta == 0 or B == 0:  # OLS
            value = float(np.sum(residual_abs**2))
            if return_t:
                # at B=0 every feasible choice maximizes the loss
                return value, np.full(n, self.delta)
            return value
        if np.isinf(self.p):  # adversarial training
            value = float(np.sum((residual_abs + B * self.delta) ** 2))
            return (value, np.full(n, self.delta)) if return_t else value
        if self.p == 2:  # square-root lasso
            residual_norm = np.linalg.norm(residual_abs)
            value = (residual_norm + np.sqrt(n) * self.delta * B) ** 2
            if return_t:
                t = (np.sqrt(n) * self.delta * residual_abs / residual_norm
                     if residual_norm > 0 else np.full(n, self.delta))
                return value, t
            return value
        # only remaining 2 < p < infty case where delta, B > 0
        return self._solve_finite_p(residual_abs, B, return_t=return_t)

    def _solve_finite_p(self, residual_abs, B, *, return_t=False):
        """return summed risk and optionally maximizing lengths for 2 < p < infinity"""

        # rescale to keep problem numerically well-behaved (see docs)
        scale = max(float(np.max(residual_abs)), self.delta * B)
        a = residual_abs / scale
        b = self.delta * B / scale

        # analytical bounds for search (see docs)
        lambda_lo = 2 * b * (float(np.min(a)) + b) / self.p
        lambda_hi = 2 * b * (float(np.max(a)) + b) / self.p
        if lambda_lo == lambda_hi:
            value = float(np.sum((a + b) ** 2) * scale**2)
            return (value, np.full(len(a), self.delta)) if return_t else value

        def objective(relative_lambda):
            lam = relative_lambda * lambda_hi
            return len(a) * lam + np.sum(self._maximize_t(a, b, lam))

        # use minimize_scalar() from scipy.optimize to find opt lambda
        result = minimize_scalar(
            objective,
            bounds=(lambda_lo / lambda_hi, 1),
            method="bounded",
            options={"xatol": 1e-10},
        )

        # check for invalid result
        if not result.success:
            raise RuntimeError(f"scalar risk minimization failed: {result.message}")

        # scale back up and optionally recover lengths at the final lambda
        value = float(result.fun * scale**2)
        if return_t:
            _, u = self._maximize_t(a, b, result.x * lambda_hi, return_t=True)
            # correct the numerical budget discrepancy so the lengths are feasible
            u = u / np.mean(u**self.p) ** (1 / self.p)
            return value, self.delta * u
        return value

    def _maximize_t(self, a, b, lam, *, return_t=False):
        """
        maximize each scaled f_i over u_i = t_i / delta

        return each inner maximum by minimizing the negative objective
        with return_t=True, also return the maximizing scaled lengths u
        """

        def negative_objective(u, a):
            # negative inner objective in scaled coordinates
            # we min -obj rather than max obj
            return lam * u**self.p - (a + b * u) ** 2

        # create bracket which we knows contains min but not where
        bracket = elementwise.bracket_minimum(
            negative_objective, 1.0, xmin=0.0, args=(a,)
        )

        # check for invalid bracket
        if not np.all(bracket.success):
            raise RuntimeError("could not bracket the perturbation maximum")

        # now find exactly where the min is within the provided bracket
        result = elementwise.find_minimum(
            negative_objective, bracket.bracket, args=(a,)
        )

        # check convergence
        if not np.all(result.success):
            raise RuntimeError("perturbation maximization did not converge")

        return (-result.f_x, result.x) if return_t else -result.f_x


def _conjugate(exponent):
    """return the conjugate exponent, with 1 and infinity as endpoints"""
    if exponent == 1:
        return np.inf
    if np.isinf(exponent):
        return 1.0
    return exponent / (exponent - 1.0)
