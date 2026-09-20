"""
minimize V_delta with the paper's gamma saddle formula and DSP

the explicit p=2 and p=infinity formulas use CVXPY directly with mean loss
DSP uses summed loss to avoid small 1/n coefficients

return mean risk, evaluated independently by RobustRisk.primal

see docs/cvx_solver.md for the equations, call structure, and examples
"""

from dataclasses import dataclass

import cvxpy as cp
import dsp
import numpy as np

from DRO.robust_risk import RobustRisk


@dataclass
class CvxResult:
    """
    object for storing the result

    fitted coefficients, mean robust squared loss, and solver information
    """

    beta: np.ndarray
    value: float
    diagnostics: dict


class CvxOptimizer:
    """minimize a RobustRisk object w.r.t. beta"""

    def __init__(self, risk):
        self.risk = risk

    def minimize(self, *, solver="CLARABEL", **solver_options):
        """
        solve using DSP/CVXPY

        depending on delta and p (delta=0, p=2, or p=infinity), the solver
        selects different branches; see docs
        """

        # convert risk into separate variables
        risk = self.risk
        X, n = risk.X, risk.n

        # define cp vars and rescale to keep problem numerically well-behaved (see docs)
        scale = float(np.sqrt(np.mean(np.asarray(risk.y) ** 2))) or 1.0
        beta = cp.Variable(X.shape[1], name="beta")
        r = X @ beta - risk.y / scale
        B = cp.norm(beta, risk.norm_dual)

        # form and solve problem
        problem, formulation = self._form_problem(beta, r, B)
        if formulation == "dsp-gamma":
            # convert DSP mean-loss agreement tolerance to summed-loss units
            solver_options["eps"] = n * solver_options.get(
                "eps", 1e-3
            )
        problem.solve(solver=solver, **solver_options)

        # check for invalid result
        if beta.value is None or problem.status not in {
            cp.OPTIMAL,
            cp.OPTIMAL_INACCURATE,
        }:
            raise RuntimeError(f"{solver} did not return a solution: {problem.status}")

        # scale back and evaluate risk at beta_hat. save diagnostics
        beta_hat = beta.value * scale
        value = float(risk.primal(beta_hat))
        diagnostics = dict(
            formulation=formulation,
            solver=solver,
            status=problem.status,
            model_value=float(problem.value)
            * scale**2
            / (n if formulation == "dsp-gamma" else 1),
            scalar_value=value,
            response_scale=scale,
        )

        # return result class
        return CvxResult(beta_hat, value, diagnostics)

    def _form_problem(self, beta, r, B):
        """form the objective and its CVXPY/DSP problem for the scaled variables"""
        risk = self.risk
        n, p, q, delta = risk.n, risk.p, risk.q, risk.delta

        # see docs for special cases
        if delta == 0:  # OLS
            objective = cp.sum_squares(r) / n
            formulation = "least-squares"
        elif p == 2:  # square-root lasso
            objective = cp.square(cp.norm(r, 2) / np.sqrt(n) + delta * B)
            formulation = "p=2"
        elif np.isinf(p):  # adversarial training
            objective = cp.sum_squares(cp.abs(r) + delta * B) / n
            formulation = "p=infinity"
        else:  # 2 < p < infty case -- solve saddle formulation using DSP
            gamma = cp.Variable(n, nonneg=True, name="gamma")
            objective = (
                n ** (1 / p)
                * delta
                * dsp.saddle_inner(
                    B, cp.power(cp.sum(gamma), 1 / q, max_denom=65536)
                )  # product between ||beta||_* = B and ||gamma||_1^{1/q}
                + dsp.saddle_inner(
                    cp.abs(r), cp.power(gamma, 1 / q, max_denom=65536)
                )  # product between gamma^{1/q} and |r|
                - cp.sum(cp.power(gamma, 2 / q, max_denom=65536)) / 4
            )
            problem = dsp.SaddlePointProblem(
                dsp.MinimizeMaximize(objective),
                minimization_vars=[beta],
                maximization_vars=[gamma],
            )
            return problem, "dsp-gamma"

        # here inner sup is solved -- no need for dsp.SaddlePointProblem
        return cp.Problem(cp.Minimize(objective)), formulation


def minimize_cvx(X, y, radius, p, *, norm=np.inf, **solve_options):
    """construct RobustRisk from a radius delta and return its minimizing fit"""
    risk = RobustRisk(
        np.asarray(X, dtype=float), np.asarray(y, dtype=float), radius, p, norm=norm
    )
    return CvxOptimizer(risk).minimize(**solve_options)
