"""
tests for EtaOptimizer

check known minima, CVX fits, the gamma risk evaluator, and the ridge updates
joint eta and alternating CVX reference checks are in eta_joint_cvx_test.py
"""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from DRO.cvx_solver import minimize_cvx
from DRO.eta_solver import EtaOptimizer, minimize_eta
from DRO.robust_risk import RobustRisk

PS = [2.0, 3.0, 6.0, np.inf]


def make_data(n=25, d=5, seed=82):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = X[:, 0] + 0.3 * rng.standard_normal(n)
    return X, y


@pytest.mark.parametrize("p", PS)
def test_one_observation_minimum(p):
    """for x=y=1, the minimum is min(delta, 1)**2 for every p"""
    X, y = np.array([[1.0]]), np.array([1.0])
    for delta in (0.2, 2.0):
        fit = minimize_eta(X, y, delta, p)
        assert fit.value == pytest.approx(min(delta, 1.0) ** 2, rel=2e-5)
        assert fit.beta[0] == pytest.approx(1.0 if delta < 1 else 0.0, abs=2e-5)


def test_one_dimensional_minimum_against_scipy():
    """check an interior p against direct scalar minimization of the risk"""
    X, y = make_data(n=20, d=1, seed=104)
    risk = RobustRisk(X, y, 0.13, 2.5)
    expected = minimize_scalar(
        lambda b: risk.primal(np.array([b])),
        bounds=(-3.0, 3.0),
        method="bounded",
        options={"xatol": 1e-12},
    )
    fit = EtaOptimizer(risk).minimize()
    assert expected.success
    assert fit.value == pytest.approx(expected.fun, rel=2e-5, abs=1e-9)
    assert fit.beta[0] == pytest.approx(expected.x, abs=2e-4)


@pytest.mark.parametrize("p", PS)
def test_fit_agrees_with_cvx_and_gamma_risk(p):
    """test optimization separately from the final scalar risk evaluation"""
    X, y = make_data()
    risk = RobustRisk(X, y, 0.12, p)
    fit = EtaOptimizer(risk).minimize()
    cvx = minimize_cvx(X, y, 0.12, p)
    gamma_value = risk.dual(fit.beta)
    assert fit.value == pytest.approx(cvx.value, rel=2e-5, abs=2e-7)
    assert fit.value == pytest.approx(gamma_value, rel=3e-6, abs=1e-8)
    # smoothing makes model_value slightly larger than the original risk
    assert fit.diagnostics["model_value"] == pytest.approx(gamma_value, rel=2e-4)


@pytest.mark.parametrize("p,scale", [
    (2.0, 0.01),
    (3.0, 0.01),
    (6.0, 100.0),
    (np.inf, 100.0),
])
def test_response_scaling(p, scale):
    """response normalization also keeps the fixed smoothing scale equivariant"""
    X, y = make_data(n=24, d=4, seed=3)
    baseline = minimize_eta(X, y, 0.2, p)
    fit = minimize_eta(X, scale * y, 0.2, p)
    assert fit.beta / scale == pytest.approx(baseline.beta, abs=2e-5)
    assert fit.value / scale**2 == pytest.approx(baseline.value, rel=2e-6)


def test_zero_radius_is_least_squares():
    X, y = make_data()
    fit = minimize_eta(X, y, 0.0, 3.0)
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    assert fit.beta == pytest.approx(beta_ols, abs=2e-6)
    assert fit.value == pytest.approx(np.mean((X @ beta_ols - y) ** 2), rel=2e-6)


def test_zero_response():
    """the response normalization must also handle an identically zero response"""
    X, y = make_data()
    fit = minimize_eta(X, np.zeros_like(y), 0.2, 3.0)
    assert fit.beta == pytest.approx(np.zeros(X.shape[1]), abs=1e-10)
    assert fit.value < 1e-10


def test_underdetermined_fit_agrees_with_cvx():
    """exercise Ridge's other linear system when there are more columns than rows"""
    X, y = make_data(n=10, d=16, seed=0)
    fit = minimize_eta(X, y, 0.15, 3.0)
    cvx = minimize_cvx(X, y, 0.15, 3.0)
    assert fit.value == pytest.approx(cvx.value, rel=2e-5, abs=2e-7)


@pytest.mark.parametrize("p", [2.0, 2.001, 3.0, 6.0, np.inf])
@pytest.mark.parametrize("beta0", [np.array([0.4, -0.2, 0.1]), np.zeros(3)])
def test_paper_ridge_update_and_exact_surrogate(p, beta0):
    """the full eta norm expression equals the ridge quadratic for every theta"""
    from DRO.robust_risk import _ScalarRisk

    X, y = make_data(n=7, d=3)
    # include an exactly zero residual, alongside zero coefficient coordinates
    y[0] = X[0] @ beta0
    scale = np.sqrt(np.mean(y**2))
    y, beta0 = y / scale, beta0 / scale
    n, delta, epsilon = len(y), 0.23, 0.07
    a = np.hypot(X @ beta0 - y, epsilon)
    b = np.hypot(beta0, epsilon)
    scalar_risk = _ScalarRisk(delta, p)
    _, t = scalar_risk.solve(a, b.sum(), return_t=True)
    A = a + t * b.sum()
    eta = np.column_stack((a, t[:, None] * b)) / A[:, None]
    assert eta.sum(axis=1) == pytest.approx(np.ones(n))
    k = np.inf if p == 2 else (1 if np.isinf(p) else p / (p - 2))
    def stable_norm(x):
        return np.max(x) * np.linalg.norm(x / np.max(x), ord=k)

    C = n**(2 / p) * delta**2 * stable_norm(A / t)
    assert np.sum(t * A) == pytest.approx(C, rel=1e-9)

    w, gamma = A / a, np.mean(t * A) / b
    expected = np.linalg.solve(X.T @ (w[:, None] * X) + n * np.diag(gamma),
                               X.T @ (w * y))
    fit = minimize_eta(X, scale * y, delta, p, beta0=scale * beta0,
                       epsilon=epsilon, maxiter=1, tol=10.0)
    assert fit.beta / scale == pytest.approx(expected, abs=1e-10)

    def surrogate(theta):
        # constants epsilon**2 must remain when comparing objective values
        loss = np.mean(((X @ theta - y)**2 + epsilon**2) / eta[:, 0])
        u = np.sum((theta**2 + epsilon**2) / eta[:, 1:], axis=1)
        G = loss + n**(2 / p - 1) * delta**2 * stable_norm(u)
        phi = loss + np.mean(t**2 * u)
        ridge_value = np.mean(w * ((X @ theta - y)**2 + epsilon**2))
        ridge_value += np.sum(gamma * (theta**2 + epsilon**2))
        assert G == pytest.approx(phi, rel=1e-9)
        assert G == pytest.approx(ridge_value, rel=1e-9)
        return G

    current_value = np.mean(A**2)
    assert surrogate(beta0) == pytest.approx(current_value, rel=1e-9)
    surrogate(np.array([-0.3, 0.8, 1.2]))  # away from both iterates
    next_value = fit.diagnostics["model_value"] / scale**2
    assert next_value <= surrogate(expected) + 1e-10
    assert surrogate(expected) <= current_value + 1e-10
    assert next_value >= fit.value / scale**2 - 1e-10


@pytest.mark.parametrize("p", PS)
def test_ridge_start_and_sample_replication(p):
    """the default ridge start matches beta0 and respects mean-loss scaling"""
    X, y = make_data(n=9, d=3)
    delta = 0.31
    beta0 = np.linalg.solve(X.T @ X + len(y) * delta**2 * np.eye(3), X.T @ y)
    options = dict(epsilon=0.02, maxiter=1, tol=10.0)
    default = minimize_eta(X, y, delta, p, **options)
    explicit = minimize_eta(X, y, delta, p, beta0=beta0, **options)
    repeated = minimize_eta(np.tile(X, (3, 1)), np.tile(y, 3), delta, p, **options)
    # value-based minimization recovers lengths to roughly sqrt(machine epsilon)
    assert default.beta == pytest.approx(explicit.beta, abs=1e-8)
    assert default.beta == pytest.approx(repeated.beta, abs=1e-8)
    assert default.diagnostics["model_value"] == pytest.approx(
        repeated.diagnostics["model_value"], rel=1e-10)

