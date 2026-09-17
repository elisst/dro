"""Checks of CVXPY fits, risk values, scaling, and numerical failure handling."""

import cvxpy as cp
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from DRO.robust_risk import RobustRisk
from DRO.cvx_solver import CvxOptimizer, minimize_cvx


@pytest.mark.parametrize('p', [2.0, 3.0, 6.0, np.inf])
@pytest.mark.parametrize('norm', [1, 2, np.inf])
def test_one_observation_value_and_minimum_are_known(p, norm):
    X, y, beta = np.array([[1.0]]), np.array([1.0]), np.array([0.4])
    for radius in [0.2, 2.0]:
        result = minimize_cvx(X, y, radius, p, norm=norm, fixed_beta=beta)
        expected_value = (0.6 + radius * 0.4) ** 2
        assert result.value == pytest.approx(expected_value, rel=2e-6)
        assert result.diagnostics["model_value"] == pytest.approx(expected_value, rel=2e-6)
        fitted = minimize_cvx(X, y, radius, p, norm=norm)
        expected_minimum = min(radius, 1.0) ** 2
        assert fitted.value == pytest.approx(expected_minimum, rel=2e-6)
        assert fitted.diagnostics["model_value"] == pytest.approx(expected_minimum, rel=2e-6)
        assert fitted.beta[0] == pytest.approx(1.0 if radius < 1 else 0.0, abs=2e-5)


@pytest.mark.parametrize('p', [2.0, 2.01, 2.1, 2.5, np.e, 3.0, np.pi, 4.0, 6.0, 7.0, 10.0, np.inf])
def test_one_dimensional_minimum_against_scipy(p):
    rng = np.random.default_rng(104)
    X = rng.normal(size=(20, 1))
    y = 0.8 * X[:, 0] + rng.normal(size=20) * 0.2
    risk = RobustRisk.normalized(X, y, 0.13, p)
    expected = minimize_scalar(
        lambda b: risk.primal(np.array([b])),
        bounds=(-3.0, 3.0),
        method='bounded',
        options={'xatol': 1e-12},
    )
    result = minimize_cvx(X, y, 0.13, p)
    assert expected.success
    assert result.value == pytest.approx(expected.fun, rel=2e-6, abs=1e-9)
    assert result.beta[0] == pytest.approx(expected.x, abs=2e-4)


@pytest.mark.parametrize('p', [2.0, 3.0, 6.0, np.inf])
def test_two_packages_and_original_gamma_risk_agree(p):
    rng = np.random.default_rng(82)
    X = rng.normal(size=(25, 5))
    y = X[:, 0] + 0.3 * rng.normal(size=25)
    a = minimize_cvx(X, y, 0.12, p)
    b = minimize_cvx(X, y, 0.12, p, solver='SCS', tolerance=1e-8)
    assert a.value == pytest.approx(b.value, rel=2e-5, abs=2e-7)
    risk = RobustRisk.normalized(X, y, 0.12, p)
    assert a.value == pytest.approx(risk.dual(a.beta), rel=3e-6, abs=1e-8)


@pytest.mark.parametrize('p', [2.0, 3.0, 6.0, np.inf])
@pytest.mark.parametrize('response_scale', [1e-20, 1.0, 1e20])
def test_changing_response_units_preserves_minimum(p, response_scale):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(24, 4))
    y = X[:, 0] + 0.2 * rng.normal(size=24)
    baseline = minimize_cvx(X, y, 0.2, p)
    result = minimize_cvx(X, response_scale * y, 0.2, p)
    assert result.value / response_scale**2 == pytest.approx(baseline.value, rel=2e-6)
    assert result.beta / response_scale == pytest.approx(baseline.beta, abs=2e-5)


@pytest.mark.parametrize('p', [2.0, 3.0, 6.0, np.inf])
def test_underdetermined_fit_agrees_between_solvers(p):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 16))
    beta_star = np.r_[np.ones(4), np.zeros(12)]
    y = X @ beta_star + 0.3 * rng.normal(size=10)
    result = minimize_cvx(X, y, 0.15, p)
    second = minimize_cvx(X, y, 0.15, p, solver="SCS", tolerance=1e-8)
    assert result.value == pytest.approx(second.value, rel=2e-5, abs=2e-7)


def test_zero_radius_and_zero_response():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(30, 4))
    y = rng.normal(size=30)
    result = minimize_cvx(X, y, 0, 3)
    assert result.beta == pytest.approx(np.linalg.lstsq(X, y, rcond=None)[0], abs=2e-6)
    assert minimize_cvx(X, np.zeros(30), 0.2, 3).value < 1e-10


@pytest.mark.parametrize('norm', [1, 2, np.inf])
def test_large_response_scale_with_representable_risk(norm):
    # exact interpolation leaves only (radius * ||beta||_*)**2 = 1e280
    result = minimize_cvx([[1.]], [1e160], 1e-20, 2, norm=norm, fixed_beta=[1e160])
    assert result.beta == pytest.approx([1e160])
    assert result.value == pytest.approx(1e280, rel=1e-12)
    assert result.diagnostics['scalar_value'] == result.value
    assert result.diagnostics['response_scale'] == 1e160
    assert np.isfinite(result.diagnostics['model_value'])


def test_unrepresentable_restored_risk_is_rejected():
    with pytest.raises(FloatingPointError, match='Restored solution'):
        minimize_cvx([[0.]], [1e160], 0, 2, fixed_beta=[0.])


def test_inaccurate_status_is_rejected(monkeypatch):
    def inaccurate(self, *args, **kwargs):
        self._status = cp.OPTIMAL_INACCURATE

    monkeypatch.setattr(cp.Problem, 'solve', inaccurate)
    with pytest.raises(RuntimeError, match='requested accuracy'):
        minimize_cvx(np.ones((2, 1)), np.ones(2), 0.1, 3)


def test_disagreement_with_scalar_evaluator_is_rejected(monkeypatch):
    monkeypatch.setattr(RobustRisk, 'primal', lambda *args, **kwargs: 123.0)
    with pytest.raises(RuntimeError, match='failed numerical checks'):
        minimize_cvx(np.ones((2, 1)), np.ones(2), 0.1, 3)


@pytest.mark.parametrize("p", [2.01, 2.1, 2.5, np.e, np.pi, 4.0, 7.0, 10.0])
@pytest.mark.parametrize("solver", ["CLARABEL", "SCS"])
def test_general_exponent_value_matches_scalar_reduction(p, solver):
    """Unequal residuals exercise the length budget for the requested real p."""
    rng = np.random.default_rng(19)
    X = rng.normal(size=(12, 3))
    y = rng.normal(size=12)
    beta = np.array([0.4, -0.2, 0.1])
    risk = RobustRisk.normalized(X, y, 0.2, p)
    fit = minimize_cvx(X, y, 0.2, p, fixed_beta=beta, solver=solver,
                    tolerance=1e-8 if solver == "SCS" else 1e-9)
    expected = risk.primal(beta)
    # Compare the independent conic model, not just the returned scalar value.
    assert fit.diagnostics["model_value"] == pytest.approx(expected, rel=2e-6, abs=2e-7)


@pytest.mark.parametrize("p", [2.0, 3.0, np.inf])
def test_class_can_fit_evaluate_then_fit_with_another_solver(p):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(20, 3))
    y = X @ np.array([1.0, -0.5, 0.0]) + 0.2 * rng.normal(size=20)
    risk = RobustRisk.normalized(X, y, 0.15, p)
    model = CvxOptimizer(risk)
    fitted = model.minimize()
    original_beta = fitted.beta.copy()
    original_diagnostics = fitted.diagnostics.copy()
    beta = np.array([0.8, -0.4, 0.1])
    fixed = model.minimize(fixed_beta=beta)
    second = model.minimize(solver="SCS", tolerance=1e-8)
    wrapper = minimize_cvx(X, y, radius=0.15, p=p)

    assert fixed.beta == pytest.approx(beta, abs=1e-14)
    assert fixed.diagnostics["model_value"] == pytest.approx(
        risk.primal(beta), rel=2e-6)
    assert "scaled_stationarity_error" not in fixed.diagnostics
    assert "scaled_complementarity_error" not in fixed.diagnostics
    assert fitted.diagnostics["scaled_stationarity_error"] <= 2e-6
    assert fitted.diagnostics["scaled_complementarity_error"] <= 2e-6
    assert fitted.value == pytest.approx(second.value, rel=2e-5, abs=2e-7)
    assert fitted.value == pytest.approx(wrapper.value, rel=2e-6)
    assert fitted.beta == pytest.approx(wrapper.beta, abs=2e-5)
    # Later calls do not mutate a previously returned result.
    np.testing.assert_array_equal(fitted.beta, original_beta)
    assert fitted.diagnostics == original_diagnostics


@pytest.mark.parametrize("p", [2.0, 3.0, np.inf])
@pytest.mark.parametrize("norm", [1, 2, np.inf])
def test_multidimensional_ground_norms_against_gamma_risk_and_second_solver(p, norm):
    # In one dimension all supported norms coincide, so use several features.
    rng = np.random.default_rng(31)
    X = rng.normal(size=(18, 3))
    y = X @ np.array([0.7, -0.4, 0.2]) + 0.15 * rng.normal(size=18)
    risk = RobustRisk.normalized(X, y, 0.12, p, norm=norm)
    model = CvxOptimizer(risk)
    result = model.minimize()
    second = model.minimize(solver="SCS", tolerance=1e-8)
    expected = risk.dual(result.beta)
    assert result.diagnostics["model_value"] == pytest.approx(expected, rel=3e-6, abs=1e-8)
    assert result.value == pytest.approx(second.value, rel=2e-5, abs=2e-7)


@pytest.mark.parametrize("p", [2.5, 3.0, 6.0])
@pytest.mark.parametrize("case", ["zero_beta", "interpolation", "some_zero_residuals"])
def test_finite_p_perspectives_include_zero_boundaries(p, case):
    X = np.eye(3)
    beta = np.array([0.5, -0.25, 0.75])
    if case == "zero_beta":
        beta = np.zeros(3)
        y = np.array([1.0, -0.5, 0.0])
        expected = np.mean(y**2)
    elif case == "interpolation":
        y = X @ beta
        expected = (0.2 * np.linalg.norm(beta, 1))**2
    else:
        y = X @ beta + np.array([0.0, 0.3, -0.2])
        expected = RobustRisk.normalized(X, y, 0.2, p).dual(beta)
    risk = RobustRisk.normalized(X, y, 0.2, p)
    result = CvxOptimizer(risk).minimize(fixed_beta=beta)
    assert result.diagnostics["model_value"] == pytest.approx(expected, rel=3e-6, abs=1e-8)
    assert result.value == pytest.approx(expected, rel=3e-6, abs=1e-8)


def test_unsupported_solver_is_rejected():
    risk = RobustRisk.normalized(np.ones((1, 1)), np.ones(1), 0.1, 3)
    model = CvxOptimizer(risk)
    with pytest.raises(ValueError, match="Use CLARABEL or SCS"):
        model.minimize(solver="unsupported")


@pytest.mark.parametrize("p", [2.0, 3.0, np.inf])
def test_optimizer_uses_stored_radius_and_preserves_supplied_risk(p):
    rng = np.random.default_rng(47)
    X = rng.normal(size=(16, 3))
    y = 1e8 * (X @ np.array([0.7, -0.4, 0.2]) + 0.2 * rng.normal(size=16))
    beta = 1e8 * np.array([0.6, -0.3, 0.1])
    radius = 0.15
    cost = radius if np.isinf(p) else radius**p
    risk = RobustRisk(X, y, cost, p, norm=2)
    original_attributes = vars(risk).copy()
    original_X, original_y = X.copy(), y.copy()
    original_value = risk.primal(beta)
    X.setflags(write=False)
    y.setflags(write=False)

    optimizer = CvxOptimizer(risk)
    fitted = optimizer.minimize()
    fixed = optimizer.minimize(fixed_beta=beta)
    wrapper = minimize_cvx(X, y, radius, p, norm=2)

    assert fitted.value == pytest.approx(wrapper.value, rel=2e-6)
    assert fitted.beta / 1e8 == pytest.approx(wrapper.beta / 1e8, abs=2e-5)
    assert fitted.value == pytest.approx(risk.primal(fitted.beta), rel=2e-6)
    assert fixed.value == pytest.approx(original_value, rel=2e-6)
    assert fixed.diagnostics["response_scale"] > 1e6
    assert optimizer.risk is risk
    assert vars(risk).keys() == original_attributes.keys()
    for name, value in original_attributes.items():
        assert getattr(risk, name) is value
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)


@pytest.mark.parametrize("p", [2.5, 3.0, 4.0, 6.0])
def test_quadratic_bound_model_against_direct_two_observation_budget(p):
    """Compare with the original movement maximization, without either dual formula."""
    radius, B = 0.4, 0.7
    X, y, beta = np.zeros((2, 1)), np.array([1.0, 0.0]), np.array([B])
    budget = 2 * radius**p
    maximum_length = budget**(1 / p)

    def mean_loss(first_length):
        second_length = max(0.0, budget - first_length**p)**(1 / p)
        return ((1 + B * first_length)**2 + (B * second_length)**2) / 2

    maximum = minimize_scalar(
        lambda t: -mean_loss(t), bounds=(0.0, maximum_length), method="bounded",
        options={"xatol": 1e-12},
    )
    assert maximum.success
    expected = max(-maximum.fun, mean_loss(0.0), mean_loss(maximum_length))
    result = minimize_cvx(X, y, radius, p, fixed_beta=beta)
    assert result.diagnostics["model_value"] == pytest.approx(expected, rel=2e-6, abs=1e-9)
    assert result.value == pytest.approx(expected, rel=2e-6, abs=1e-9)
