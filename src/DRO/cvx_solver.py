"""minimize V_delta(beta) with CVXPY and check the fitted risk independently

CvxOptimizer takes a RobustRisk object and handles response scaling and returned units
_CvxProblem builds and solves the convex model, then checks its numerical solution
minimize_cvx constructs RobustRisk from data and calls CvxOptimizer(risk).minimize(...)
radius is delta; returned values are mean robust squared losses
see docs/cvx_solver.md for the equations, call structure, and examples
"""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from DRO.robust_risk import RobustRisk, _conjugate


@dataclass
class CvxResult:
    """coefficients, mean robust squared loss, and numerical diagnostics"""

    beta: np.ndarray
    value: float                 # mean robust squared loss V_delta(beta)
    diagnostics: dict


class CvxOptimizer:
    """minimize a RobustRisk object using its data, radius, p, and coefficient norm

    assumes the input conventions of RobustRisk and ground norm 1, 2, or infinity
    builds a convex formulation and checks its value with the scalar evaluator
    response scaling uses a separate risk object, leaving the supplied risk unchanged
    """

    def __init__(self, risk):
        self.risk = risk

    def minimize(self, *, fixed_beta=None, solver="CLARABEL", tolerance=1e-9,
                 check_tolerance=2e-6):
        """return CvxResult after checking the convex solve and scalar risk

        fixed_beta holds coefficients fixed and optimizes only auxiliary variables
        each call builds a fresh model; no previous fit is retained or warm-started
        inaccurate status or failed numerical checks raise RuntimeError
        """
        if fixed_beta is not None:
            fixed_beta = np.asarray(fixed_beta, dtype=float)
        options = _solver_options(solver, tolerance)
        scale = self._response_scale(fixed_beta)
        scaled_beta = None if fixed_beta is None else fixed_beta / scale
        risk = self.risk
        scaled_risk = RobustRisk.normalized(
            risk.X, risk.y / scale, risk.delta, risk.p, norm=_conjugate(risk.norm_dual))
        model = _CvxProblem(scaled_risk, scaled_beta)
        coefficient, diagnostics = model.solve(solver, options, tolerance, check_tolerance)
        return self._restore_units(coefficient, diagnostics, scale)

    def _response_scale(self, fixed_beta):
        """choose common response/coefficient units, preserving X and delta"""
        risk = self.risk
        X, y, delta, norm_dual = risk.X, risk.y, risk.delta, risk.norm_dual
        y_max = float(np.max(np.abs(y)))
        scale = y_max * float(np.sqrt(np.mean((y / y_max)**2))) if y_max else 0.0
        if fixed_beta is not None:
            beta_max = float(np.max(np.abs(fixed_beta)))
            # normalize before taking the norm to avoid squaring large coefficients
            movement_scale = (
                (delta * beta_max) * float(np.linalg.norm(fixed_beta / beta_max, norm_dual))
                if beta_max else 0.0
            )
            scale = max(scale, float(np.max(np.abs(X @ fixed_beta))),
                        movement_scale)
        if not np.isfinite(scale):
            raise FloatingPointError("Response scale is outside floating-point range")
        return scale if scale else 1.0

    def _restore_units(self, coefficient, diagnostics, scale):
        """restore squared-loss units without forming a possibly overflowing scale**2"""
        beta = coefficient * scale
        for key in ("model_value", "scalar_value"):
            diagnostics[key] = (diagnostics[key] * scale) * scale
        if not np.isfinite(beta).all() or not np.isfinite(
            [diagnostics["model_value"], diagnostics["scalar_value"]]
        ).all():
            raise FloatingPointError("Restored solution is outside floating-point range")
        diagnostics["response_scale"] = scale
        return CvxResult(beta, diagnostics["scalar_value"], diagnostics)


class _CvxProblem:
    """build and solve the convex model in response-scaled units

    r = X beta - y and B >= ||beta||_* link the data to the risk objective
    the finite-p branch minimizes tangent upper bounds derived from theorem 1, eq (5)
    endpoints use proposition 2; every branch is checked against RobustRisk.primal
    """

    def __init__(self, risk, fixed_beta):
        self.risk = risk
        self.norm = _conjugate(risk.norm_dual)
        X, y = risk.X, risk.y
        self.fixed_beta = fixed_beta
        self.beta = cp.Variable(X.shape[1]) if fixed_beta is None else cp.Constant(fixed_beta)
        self.r = cp.Variable(len(y))
        self.B = cp.Variable(nonneg=True)
        self.residual_constraint = self.r == X @ self.beta - y
        self.norm_constraint = cp.norm(self.beta, self.risk.norm_dual) <= self.B
        self.constraints = [self.residual_constraint, self.norm_constraint]
        objective = self._build_objective()
        self.problem = cp.Problem(cp.Minimize(objective), self.constraints)

    def _build_objective(self):
        """select squared residual loss, an endpoint, or the finite-p model"""
        n, delta, p = self.risk.n, self.risk.delta, self.risk.p
        r, B = self.r, self.B
        if delta == 0:
            return cp.sum_squares(r)
        if p == 2:
            # minimizing a nonnegative function or its square gives the same beta
            return cp.norm(r, 2) + np.sqrt(n) * delta * B
        if np.isinf(p):
            return cp.sum_squares(cp.abs(r) + delta * B)
        return self._finite_p_objective()

    def _finite_p_objective(self):
        """Minimize worst-case bounds u_i + w_i*t_i**2, with w_scaled=delta**2*w."""
        r, B = self.r, self.B
        n, delta, p = self.risk.n, self.risk.delta, self.risk.p
        u, w_scaled = (cp.Variable(n, name=name) for name in ("u", "w_scaled"))
        k = p / (p - 2)
        # Preserve the supplied exponent using CVXPY's native cone representation.
        w_norm = cp.pnorm(w_scaled, k, approx=False)
        self.constraints += _quadratic_upper_bound(r, delta * B, u, w_scaled)
        return cp.sum(u) + n**(2 / p) * w_norm

    def solve(self, solver, options, tolerance, check_tolerance):
        """solve the model and reject inaccurate or inconsistent solutions"""
        self.problem.solve(solver=solver, **options)
        if self.problem.status != cp.OPTIMAL or self.beta.value is None:
            raise RuntimeError(
                f"{solver} did not establish the requested accuracy: {self.problem.status}"
            )
        coefficient = np.asarray(self.beta.value).reshape(self.risk.X.shape[1])
        diagnostics = self._check_solution(coefficient, solver, tolerance, check_tolerance)
        return coefficient, diagnostics

    def _check_solution(self, coefficient, solver, tolerance, check_tolerance):
        """compare values, check feasibility, and assess coefficient optimality"""
        problem, risk = self.problem, self.risk
        root_objective = risk.delta > 0 and risk.p == 2
        divisor = np.sqrt(risk.n) if root_objective else risk.n
        model_value = float(problem.value / divisor)
        if root_objective:
            model_value = max(0.0, model_value)**2
        # evaluate the original scalar reduction only after the convex solve
        scalar_value = float(risk.primal(coefficient))
        value_error = abs(scalar_value - model_value) / max(1.0, abs(scalar_value))
        feasibility = max(float(np.max(c.violation())) for c in problem.constraints)
        diagnostics = dict(solver=solver, status=problem.status, tolerance=tolerance,
                           check_tolerance=check_tolerance, iterations=problem.solver_stats.num_iters,
                           model_value=model_value,
                           scalar_value=scalar_value,
                           scaled_value_error=value_error,
                           scaled_constraint_violation=feasibility)

        if self.fixed_beta is None:
            stationarity, complementarity = self._coefficient_optimality(coefficient, divisor)
            diagnostics.update(scaled_stationarity_error=stationarity,
                               scaled_complementarity_error=float(complementarity))
        else:
            stationarity = complementarity = 0.0
        errors = [value_error, feasibility, stationarity, complementarity]
        if not np.isfinite(errors).all() or max(errors) > check_tolerance:
            raise RuntimeError(f"Fit failed numerical checks: {diagnostics}")
        return diagnostics

    def _coefficient_optimality(self, beta, divisor):
        """check norm-subgradient conditions across all coefficient coordinates"""
        X, norm, objective = self.risk.X, self.norm, float(self.problem.value)
        residual_constraint, norm_constraint = self.residual_constraint, self.norm_constraint
        g = -np.asarray(residual_constraint.dual_value) / divisor
        mu = float(norm_constraint.dual_value) / divisor
        gradient = X.T @ g
        dual_norm = float(np.linalg.norm(gradient, norm))
        stationarity = max(0.0, dual_norm - mu) / max(1.0, dual_norm, abs(mu))
        complementarity = abs(gradient @ beta + mu * np.linalg.norm(beta, _conjugate(norm)))
        complementarity /= max(1.0, abs(objective / divisor))
        return stationarity, float(complementarity)



def _quadratic_upper_bound(r, movement_scale, u, w):
    """Encode r_i**2/u_i + movement_scale**2/w_i <= 1, including zero boundaries.

    v bounds the movement quotient; 1-v bounds the residual quotient.
    The cones also enforce u,w >= 0 and 0 <= v <= 1.
    """
    v = cp.Variable(r.size)
    return [
        _square_over_linear(r, 1 - v, u),
        _square_over_linear(movement_scale * np.ones(r.size), v, w),
    ]


def _square_over_linear(x, denominator, upper):
    """encode x**2 <= upper*denominator, including zero boundaries, per sample"""
    return cp.SOC(upper + denominator,
                  cp.vstack([2 * x, upper - denominator]), axis=0)


def _solver_options(solver, tolerance):
    if solver == "CLARABEL":
        return dict(tol_gap_abs=tolerance, tol_gap_rel=tolerance,
                    tol_feas=tolerance, max_iter=500, max_threads=1)
    elif solver == "SCS":
        return dict(eps=tolerance, max_iters=200000)
    raise ValueError("Use CLARABEL or SCS")


def minimize_cvx(X, y, radius, p, *, norm=np.inf, fixed_beta=None,
                 solver="CLARABEL", tolerance=1e-9, check_tolerance=2e-6):
    """construct RobustRisk from data and minimize it; radius is delta, value is mean risk"""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    risk = RobustRisk.normalized(X, y, radius, p, norm=norm)
    return CvxOptimizer(risk).minimize(
        fixed_beta=fixed_beta, solver=solver, tolerance=tolerance,
        check_tolerance=check_tolerance)
