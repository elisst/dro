"""
tests for CvxOptimizer

check fitted minima against known solutions and independent optimizers,
then check risk values, response scaling, and one experiment-size regression
"""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from DRO.cvx_solver import CvxOptimizer, minimize_cvx
from DRO.robust_risk import RobustRisk

PS = [2.0, 3.0, 6.0, np.inf]


def make_data(n=25, d=5, seed=82):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = X[:, 0] + 0.3 * rng.standard_normal(n)
    return X, y


@pytest.mark.parametrize("p", PS)
def test_one_observation_minimum(p):
    """for x=y=1, the minimum is min(delta, 1)**2"""
    X, y = np.array([[1.0]]), np.array([1.0])
    for delta in (0.2, 2.0):
        fit = minimize_cvx(X, y, delta, p)
        expected = min(delta, 1.0) ** 2
        assert fit.value == pytest.approx(expected, rel=2e-6)
        assert fit.diagnostics["model_value"] == pytest.approx(expected, rel=2e-6)
        assert fit.beta[0] == pytest.approx(1.0 if delta < 1 else 0.0, abs=2e-5)


@pytest.mark.parametrize("p", [2.0, 2.5, 3.0, 6.0, np.inf])
def test_one_dimensional_minimum_against_scipy(p):
    """compare the fitted beta with direct minimization of the scalar risk"""
    X, y = make_data(n=20, d=1, seed=104)
    risk = RobustRisk(X, y, 0.13, p)
    expected = minimize_scalar(
        lambda b: risk.primal(np.array([b])),
        bounds=(-3.0, 3.0),
        method="bounded",
        options={"xatol": 1e-12},
    )
    fit = minimize_cvx(X, y, 0.13, p)
    assert expected.success
    assert fit.value == pytest.approx(expected.fun, rel=2e-6, abs=1e-9)
    assert fit.beta[0] == pytest.approx(expected.x, abs=2e-4)


@pytest.mark.parametrize("p,norm", [
    (2.0, np.inf),
    (3.0, 1),
    (3.0, 2),
    (3.0, np.inf),
    (6.0, np.inf),
    (np.inf, np.inf),
])
def test_fit_agrees_with_second_solver_and_gamma_risk(p, norm):
    """check optimal values with SCS and the gamma evaluator"""
    # use several features since all ground norms coincide in one dimension
    X, y = make_data()
    risk = RobustRisk(X, y, 0.12, p, norm=norm)
    fit = CvxOptimizer(risk).minimize()
    second = minimize_cvx(
        X, y, 0.12, p, norm=norm, solver="SCS", eps_abs=1e-8, eps_rel=1e-8
    )
    expected = risk.dual(fit.beta)

    # model_value comes from the solver; fit.value comes from the scalar evaluator
    assert fit.diagnostics["model_value"] == pytest.approx(expected, rel=3e-6, abs=1e-8)
    assert fit.value == pytest.approx(expected, rel=3e-6, abs=1e-8)
    assert fit.value == pytest.approx(second.value, rel=2e-5, abs=2e-7)


@pytest.mark.parametrize("p,scale", [
    (2.0, 0.01),
    (3.0, 0.01),
    (6.0, 100.0),
    (np.inf, 100.0),
])
def test_response_scaling(p, scale):
    """scaling y by c scales beta_hat by c and the minimum risk by c**2"""
    X, y = make_data(n=24, d=4, seed=3)
    baseline = minimize_cvx(X, y, 0.2, p)
    fit = minimize_cvx(X, scale * y, 0.2, p)
    assert fit.beta / scale == pytest.approx(baseline.beta, abs=2e-5)
    assert fit.value / scale**2 == pytest.approx(baseline.value, rel=2e-6)
    assert fit.diagnostics["model_value"] == pytest.approx(fit.value, rel=2e-6)


def test_zero_radius_is_least_squares():
    """without transport, the fit agrees with ordinary least squares"""
    X, y = make_data()
    fit = minimize_cvx(X, y, 0.0, 3.0)
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    assert fit.beta == pytest.approx(beta_ols, abs=2e-6)
    assert fit.value == pytest.approx(np.mean((X @ beta_ols - y) ** 2), rel=2e-6)


def test_zero_response():
    """y=0 admits beta=0 with zero risk, including after response scaling"""
    X, y = make_data()
    fit = minimize_cvx(X, np.zeros_like(y), 0.2, 3.0)
    assert fit.value < 1e-10


def test_underdetermined_fit_agrees_between_solvers():
    """with more features than observations, compare values rather than coefficients"""
    X, y = make_data(n=10, d=16, seed=0)
    fit = minimize_cvx(X, y, 0.15, 3.0)
    second = minimize_cvx(X, y, 0.15, 3.0, solver="SCS", eps_abs=1e-8, eps_rel=1e-8)
    assert fit.value == pytest.approx(second.value, rel=2e-5, abs=2e-7)


def test_infinity_endpoint_at_experiment_size():
    """keep a large case where solving with summed loss previously failed"""
    n, d = 40960, 10
    rng = np.random.default_rng(497582027)
    X = rng.uniform(-1, 1, (n, d))
    beta_star = np.r_[np.full(5, 3.0), np.zeros(5)]
    y = X @ beta_star + 0.5 * rng.standard_normal(n)
    delta = 3.6 * np.sqrt(np.log(d / 0.01) / n)
    fit = minimize_cvx(X, y, delta, np.inf)
    expected = np.mean(
        (np.abs(X @ fit.beta - y) + delta * np.linalg.norm(fit.beta, 1)) ** 2
    )
    assert fit.value == pytest.approx(expected, rel=1e-5)
    assert fit.diagnostics["model_value"] == pytest.approx(expected, rel=1e-5)
