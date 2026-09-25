"""
minimize V_delta with the eta trick and weighted ridge regression

use infinity ground norm (l1 beta norm), with 2 <= p <= infinity
keep the smoothing of residual and beta magnitudes fixed

return mean risk, evaluated independently by RobustRisk.primal
model_value is the smoothed mean risk, scalar_value is the original mean risk
"""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge

from DRO.robust_risk import RobustRisk, _ScalarRisk


@dataclass
class EtaResult:
    """
    object for storing the result

    fitted betas, mean robust squared loss, and solver information
    """

    beta: np.ndarray
    value: float
    diagnostics: dict


class EtaOptimizer:
    """minimize a RobustRisk object w.r.t. beta, for infinity ground norm"""

    def __init__(self, risk):
        self.risk = risk

    def minimize(
        self,
        *,
        epsilon=1e-6,
        tol=1e-6,
        maxiter=1000,
        beta0=None,
        solver="cholesky",
        **solver_options,
    ):
        """
        repeat perturbation lengths -> weights -> weighted ridge with fixed smoothing

        stop on relative beta change <= tol, not an optimality certificate
        solver and solver_options are passed to Ridge
        """

        # convert risk object into separate variables
        risk = self.risk
        X, p, delta = risk.X, risk.p, risk.delta
        if risk.norm_dual != 1:
            raise ValueError("eta requires infinity ground norm (norm_dual=1)")

        # rescale the response, as in CvxOptimizer
        scale = float(np.sqrt(np.mean(risk.y**2))) or 1.0
        y = risk.y / scale

        iterations, relative_step = 0, 0.0

        if delta == 0:  # OLS -- no transport or eta updates are needed
            beta = (
                Ridge(alpha=0.0, fit_intercept=False, solver=solver, **solver_options)
                .fit(X, y)
                .coef_
            )
            status = "least-squares"
            model_value = float(np.mean((X @ beta - y) ** 2))
        else:
            # init beta at a ridge estimate
            if beta0 is None:
                # uniform eta gives mean(r**2) + delta**2 * ||beta||_2**2
                # sklearn uses summed loss, so alpha = n * delta**2
                beta = (
                    Ridge(
                        alpha=len(y) * delta**2,
                        fit_intercept=False,
                        solver=solver,
                        **solver_options,
                    )
                    .fit(X, y)
                    .coef_
                )
            else:
                beta = np.asarray(beta0, dtype=float) / scale

            scalar_risk = _ScalarRisk(delta, p)
            status = "iteration_limit"
            for iterations in range(1, maxiter + 1):
                # smooth magnitudes using |x| = sqrt(|x|^2 + eps^2)
                r_smooth = np.hypot(X @ beta - y, epsilon)
                beta_smooth = np.hypot(beta, epsilon)
                beta_norm = float(beta_smooth.sum())

                # compute t_star
                _, t_star = scalar_risk.solve(r_smooth, beta_norm, return_t=True)

                # compute weights w and gamma
                w = (r_smooth + t_star * beta_norm) / r_smooth
                gamma = (
                    float(np.mean(t_star * (r_smooth + t_star * beta_norm)))
                    / beta_smooth
                )

                # solve weighted ridge problem
                beta_next = _weighted_ridge(
                    X, y, w, gamma, solver=solver, **solver_options
                )

                # compute change in beta to decide when to stop iterating
                relative_step = float(
                    np.linalg.norm(beta_next - beta) / (1 + np.linalg.norm(beta))
                )
                beta = beta_next
                if relative_step <= tol:
                    status = "step_tolerance"
                    break

            # evaluate the smoothed objective at the final betas
            r_smooth = np.hypot(X @ beta - y, epsilon)
            beta_norm = float(np.hypot(beta, epsilon).sum())
            _, t_star = scalar_risk.solve(r_smooth, beta_norm, return_t=True)
            model_value = float(np.mean((r_smooth + t_star * beta_norm) ** 2))

        # scale back and evaluate the original risk. save diagnostics
        beta_hat = beta * scale
        value = float(risk.primal(beta_hat))
        diagnostics = dict(
            formulation="least-squares" if delta == 0 else "eta",
            solver=solver,
            status=status,
            iterations=iterations,
            relative_step=relative_step,
            smoothing=epsilon * scale if delta > 0 else 0.0,
            model_value=model_value * scale**2,
            scalar_value=value,
            response_scale=scale,
        )
        return EtaResult(beta_hat, value, diagnostics)


def minimize_eta(X, y, radius, p, *, norm=np.inf, **solve_options):
    """construct RobustRisk from a radius delta and return its minimizing fit"""
    risk = RobustRisk(
        np.asarray(X, dtype=float), np.asarray(y, dtype=float), radius, p, norm=norm
    )
    return EtaOptimizer(risk).minimize(**solve_options)


def _weighted_ridge(X, y, w, gamma, *, solver="cholesky", **solver_options):
    """
    minimize mean(w * (X beta-y)**2) + sum(gamma * beta**2)

    normal equations (X.T W X + n Gamma) beta = X.T W y

    sklearn Ridge uses one penalty strength alpha for all fitted coefficients,
    whereas gamma gives each beta_j its own penalty

    to account for the gamma weights, first multiply the mean-loss objective by n
    to match sklearn's summed-loss convention

    for gamma_j > 0, write beta = D theta with D_jj = 1/sqrt(n * gamma_j).
    then X beta = (X D) theta and
        n * sum_j gamma_j * beta_j**2 = sum_j theta_j**2

    thus fitting theta with design X D and alpha=1 solves our original
    problem

    X * column_scale scales each column of X by its D_jj. multiplying the fitted
    theta by column_scale recovers beta = D theta
    """
    column_scale = 1 / np.sqrt(len(y) * gamma)
    ridge = Ridge(alpha=1.0, fit_intercept=False, solver=solver, **solver_options)
    fit = ridge.fit(X * column_scale, y, sample_weight=w)
    return column_scale * fit.coef_
