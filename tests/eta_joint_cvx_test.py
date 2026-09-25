"""
tests for the joint eta formulation and alternating CVX updates

compare the solver with joint minimization, one eta update, and alternating
eta minimization with ridge; all reference solves use the same smoothing

use small data with mean(y**2)=1 to match the solver's response scaling
moderate smoothing keeps the CVX problems away from the simplex boundary;
eta_solver_test.py also checks the default small smoothing
"""

import cvxpy as cp
import numpy as np
import pytest

from DRO.eta_solver import _weighted_ridge, minimize_eta
from DRO.robust_risk import _ScalarRisk


PS = [2.0, 3.0, 6.0, np.inf]
DELTA, EPSILON = 0.21, 0.08


def make_data(n=6, d=2, seed=29):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = X[:, 0] + 0.3 * rng.standard_normal(n)
    return X, y / np.sqrt(np.mean(y**2))


def conjugate_half(p):
    return np.inf if p == 2 else (1.0 if np.isinf(p) else p / (p - 2))


def make_cvx_problem(X, y, p, *, fixed_beta=False):
    """build G_epsilon; beta is a variable for the joint solve, else a parameter"""
    n, d = X.shape
    beta = cp.Parameter(d) if fixed_beta else cp.Variable(d)
    eta = cp.Variable((n, d + 1), nonneg=True)
    residual = X @ beta - y
    loss = sum(
        cp.quad_over_lin(cp.hstack([residual[i], EPSILON]), eta[i, 0])
        for i in range(n)
    ) / n
    u = cp.hstack([
        sum(cp.quad_over_lin(cp.hstack([beta[j], EPSILON]), eta[i, j + 1])
            for j in range(d))
        for i in range(n)
    ])
    objective = loss + n**(2 / p - 1) * DELTA**2 * cp.norm(u, conjugate_half(p))
    problem = cp.Problem(cp.Minimize(objective), [cp.sum(eta, axis=1) == 1])
    return problem, beta, eta


def solve_cvx(problem):
    problem.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10,
                  tol_feas=1e-10)
    assert problem.status == cp.OPTIMAL
    assert np.isfinite(problem.value)


def check_simplex(eta):
    assert np.all(np.isfinite(eta)) and np.all(eta > 0)
    assert eta.sum(axis=1) == pytest.approx(np.ones(len(eta)), abs=1e-8)


def weights_from_eta(eta, p):
    """
    recover ridge weights from CVX eta, without calling scalar transport

    at optimal eta the columns of 1/eta[:, 1:] are proportional, so the
    norm of their positive sum is the sum of their norms

    for arbitrary eta this gives only an upper bound; check proportional
    columns here and the ridge representation at the CVX optimum below
    """
    inverse_eta = 1 / eta[:, 1:]
    norms = np.linalg.norm(inverse_eta, ord=conjugate_half(p), axis=0)
    profiles = inverse_eta / norms
    assert profiles == pytest.approx(
        np.broadcast_to(profiles[:, :1], profiles.shape), rel=1e-3, abs=1e-6
    )
    gamma = len(eta)**(2 / p - 1) * DELTA**2 * norms
    return 1 / eta[:, 0], gamma


def evaluate_g(X, y, beta, eta, p):
    """evaluate the full norm expression, retaining the smoothing constants"""
    loss = np.mean(((X @ beta - y)**2 + EPSILON**2) / eta[:, 0])
    u = np.sum((beta**2 + EPSILON**2) / eta[:, 1:], axis=1)
    return loss + len(y)**(2 / p - 1) * DELTA**2 * np.linalg.norm(
        u, ord=conjugate_half(p)
    )


def check_ridge_representation(X, y, beta, eta, p):
    w, gamma = weights_from_eta(eta, p)
    quadratic = np.mean(w * ((X @ beta - y)**2 + EPSILON**2))
    quadratic += np.sum(gamma * (beta**2 + EPSILON**2))
    # eta accuracy can be lower than objective accuracy, especially with
    # the max norm at p=2; use a separate tolerance for the representation
    assert quadratic == pytest.approx(evaluate_g(X, y, beta, eta, p), rel=3e-5)


@pytest.mark.parametrize("p", PS)
@pytest.mark.parametrize("n,d,seed", [(6, 2, 29), (3, 5, 11)])
def test_smoothed_fit_against_joint_eta_cvx(p, n, d, seed):
    """solve min_{beta,eta} G jointly, including an underdetermined design"""
    X, y = make_data(n, d, seed)
    problem, beta, eta = make_cvx_problem(X, y, p)
    solve_cvx(problem)
    check_simplex(eta.value)
    fit = minimize_eta(X, y, DELTA, p, epsilon=EPSILON, tol=1e-9)
    assert fit.diagnostics["status"] == "step_tolerance"
    assert fit.diagnostics["model_value"] == pytest.approx(problem.value, rel=2e-7)
    assert fit.beta == pytest.approx(beta.value, abs=3e-5)


@pytest.mark.parametrize("p", PS)
@pytest.mark.parametrize("initial", [[0.4, -0.2, 0.1], [0.4, 0.0, -0.1], [0., 0., 0.]])
def test_cvx_eta_update_matches_transport_and_ridge(p, initial):
    """compare eta, objective, weights, and one beta step at fixed beta"""
    X, y = make_data(n=5, d=3, seed=82)
    initial = np.array(initial)
    y[0] = X[0] @ initial  # an exactly zero residual in every case
    scale = np.sqrt(np.mean(y**2))
    y, initial = y / scale, initial / scale
    problem, beta, eta_variable = make_cvx_problem(X, y, p, fixed_beta=True)
    beta.value = initial
    solve_cvx(problem)
    eta = eta_variable.value
    check_simplex(eta)

    r, b = np.hypot(X @ initial - y, EPSILON), np.hypot(initial, EPSILON)
    t = _ScalarRisk(DELTA, p).transport(r, b.sum())
    A = r + t * b.sum()
    expected_eta = np.column_stack((r, t[:, None] * b)) / A[:, None]
    assert eta == pytest.approx(expected_eta, rel=3e-4, abs=3e-6)
    assert problem.value == pytest.approx(np.mean(A**2), rel=2e-7)

    w, gamma = weights_from_eta(eta, p)
    assert w == pytest.approx(A / r, rel=3e-4)
    assert gamma == pytest.approx(np.mean(t * A) / b, rel=3e-4)
    cvx_step = _weighted_ridge(X, y, w, gamma)
    fit = minimize_eta(X, y, DELTA, p, beta0=initial, epsilon=EPSILON,
                       maxiter=1, tol=1e-12)
    assert fit.beta == pytest.approx(cvx_step, abs=3e-5)
    for candidate in (initial, cvx_step, np.array([-0.3, 0.8, 1.2])):
        check_ridge_representation(X, y, candidate, eta, p)


@pytest.mark.parametrize("p", PS)
def test_alternating_cvx_eta_and_ridge_matches_joint_and_transport(p):
    """the reference loop gets all its eta updates from CVX, never transport"""
    X, y = make_data()
    n, d = X.shape
    initial = np.linalg.solve(X.T @ X + n * DELTA**2 * np.eye(d), X.T @ y)
    beta = initial.copy()
    problem, fixed_beta, eta_variable = make_cvx_problem(X, y, p, fixed_beta=True)
    previous_value = np.inf
    for _ in range(100):
        fixed_beta.value = beta
        solve_cvx(problem)
        eta = eta_variable.value
        check_simplex(eta)
        assert problem.value <= previous_value + 2e-7
        previous_value = problem.value

        w, gamma = weights_from_eta(eta, p)
        next_beta = _weighted_ridge(X, y, w, gamma)
        check_ridge_representation(X, y, next_beta, eta, p)
        assert evaluate_g(X, y, next_beta, eta, p) <= problem.value + 2e-7
        change = np.linalg.norm(next_beta - beta) / (1 + np.linalg.norm(beta))
        beta = next_beta
        if change <= 2e-6:
            break
    else:
        pytest.fail("CVX eta / ridge reference did not converge within 100 iterations")

    # re-optimize eta at the final beta, so all compared values are F_epsilon(beta)
    fixed_beta.value = beta
    solve_cvx(problem)
    reference_value = problem.value
    joint_problem, joint_beta, _ = make_cvx_problem(X, y, p)
    solve_cvx(joint_problem)
    fit = minimize_eta(X, y, DELTA, p, beta0=initial, epsilon=EPSILON, tol=1e-9)
    assert fit.diagnostics["status"] == "step_tolerance"
    assert reference_value == pytest.approx(joint_problem.value, rel=2e-7)
    assert reference_value == pytest.approx(fit.diagnostics["model_value"], rel=2e-7)
    assert beta == pytest.approx(joint_beta.value, abs=3e-5)
    assert beta == pytest.approx(fit.beta, abs=3e-5)
