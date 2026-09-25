"""
tests for RobustRisk

compare the scalar reduction with the gamma formulation, then check explicit
cases and the main properties of V_delta
"""

import numpy as np
import pytest

from DRO.robust_risk import RobustRisk

PS = [2.0, 3.0, 6.0, np.inf]
NORMS = [(1, np.inf), (2, 2), (np.inf, 1)]


def make_data(n=30, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    beta_star = rng.standard_normal(d)
    y = X @ beta_star + 0.5 * rng.standard_normal(n)
    beta = beta_star + 0.3 * rng.standard_normal(d)
    return X, y, beta


@pytest.mark.parametrize("p,norm,delta", [
    (2.01, np.inf, 0.2),
    (2.5, np.inf, 0.2),
    (3.0, 1, 0.2),
    (3.0, 2, 0.2),
    (3.0, np.inf, 0.2),
    (6.0, np.inf, 0.01),
    (6.0, np.inf, 1.0),
    (100.0, np.inf, 0.2),
])
def test_primal_equals_dual(p, norm, delta):
    """compare the two formulations at representative exponents, norms, and radii"""
    X, y, beta = make_data()
    risk = RobustRisk(X, y, delta, p, norm=norm)
    assert risk.primal(beta) == pytest.approx(risk.dual(beta), rel=1e-6)


@pytest.mark.parametrize("p", [2.01, 6.0])
def test_primal_equals_dual_with_mixed_zero_residuals(p):
    """check inner maxima near zero alongside those using substantial transport"""
    risk = RobustRisk(np.ones((4, 1)), [1.0, 1.0, 0.8, 0.0], 0.1, p)
    beta = np.array([1.0])
    assert risk.primal(beta) == pytest.approx(risk.dual(beta), rel=1e-6)


@pytest.mark.parametrize("norm,norm_dual", NORMS)
def test_p2_is_sqrt_lasso(norm, norm_dual):
    """sqrt(V_delta) = sqrt(mean(r**2)) + delta*||beta||_*"""
    X, y, beta = make_data()
    delta = 0.2
    risk = RobustRisk(X, y, delta, 2.0, norm=norm)
    expected = (
        np.sqrt(np.mean((X @ beta - y) ** 2))
        + delta * np.linalg.norm(beta, norm_dual)
    ) ** 2
    assert risk.primal(beta) == pytest.approx(expected, rel=1e-6)
    assert risk.dual(beta) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("norm,norm_dual", NORMS)
def test_p_inf_is_adversarial(norm, norm_dual):
    """at p=infinity each observation has the same perturbation budget delta"""
    X, y, beta = make_data()
    delta = 0.2
    risk = RobustRisk(X, y, delta, np.inf, norm=norm)
    expected = np.mean(
        (np.abs(X @ beta - y) + delta * np.linalg.norm(beta, norm_dual)) ** 2
    )
    assert risk.primal(beta) == pytest.approx(expected, rel=1e-6)
    assert risk.dual(beta) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("p", PS)
def test_zero_radius_is_least_squares(p):
    """without transport, the robust risk is the empirical squared loss"""
    X, y, beta = make_data()
    risk = RobustRisk(X, y, 0.0, p)
    expected = np.mean((X @ beta - y) ** 2)
    assert risk.primal(beta) == pytest.approx(expected, rel=1e-6)
    assert risk.dual(beta) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("p", PS)
def test_zero_beta_is_least_squares(p):
    """when beta=0, moving the covariates cannot change the loss"""
    X, y, _ = make_data()
    beta = np.zeros(X.shape[1])
    risk = RobustRisk(X, y, 0.2, p)
    expected = np.mean(y**2)
    assert risk.primal(beta) == pytest.approx(expected, rel=1e-6)
    assert risk.dual(beta) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("p", PS)
def test_perfect_fit(p):
    """at interpolation, equal perturbation lengths give (delta*||beta||_*)**2"""
    X, _, beta = make_data()
    delta = 0.3
    risk = RobustRisk(X, X @ beta, delta, p)
    expected = (delta * np.linalg.norm(beta, 1)) ** 2
    assert risk.primal(beta) == pytest.approx(expected, rel=1e-6)
    assert risk.dual(beta) == pytest.approx(expected, rel=1e-6)


def test_one_observation():
    """with one observation, the risk is (|r| + delta*||beta||_*)**2"""
    risk = RobustRisk(np.array([[1]]), np.array([0]), 0.25, 3.0)
    beta = np.array([1])
    expected = 1.25**2
    assert risk.primal(beta) == pytest.approx(expected, rel=1e-6)
    assert risk.dual(beta) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("p", [3.0, 6.0])
def test_increasing_in_delta(p):
    """a larger ambiguity set cannot decrease the risk"""
    X, y, beta = make_data()
    values = [
        RobustRisk(X, y, delta, p).primal(beta) for delta in (0.0, 0.05, 0.2, 1.0)
    ]
    assert all(a < b for a, b in zip(values, values[1:]))


@pytest.mark.parametrize("p", [3.0, 6.0])
def test_convex_in_beta(p):
    """V_delta lies below the chord between two coefficient vectors"""
    X, y, _ = make_data()
    rng = np.random.default_rng(11)
    b1, b2 = rng.standard_normal((2, X.shape[1]))
    risk = RobustRisk(X, y, 0.2, p)
    v1, v2 = risk.primal(b1), risk.primal(b2)
    for w in (0.25, 0.75):
        chord = w * v1 + (1 - w) * v2
        assert risk.primal(w * b1 + (1 - w) * b2) <= chord * (1 + 1e-9)


def test_large_p_approaches_adversarial():
    """at fixed radius, the risk decreases towards the p=infinity endpoint"""
    X, y, beta = make_data()
    delta = 0.2
    endpoint = RobustRisk(X, y, delta, np.inf).primal(beta)
    values = [
        RobustRisk(X, y, delta, p).primal(beta) for p in (5.0, 20.0, 100.0)
    ]
    assert all(a > b for a, b in zip(values, values[1:]))
    assert all(value > endpoint for value in values)
    assert values[-1] == pytest.approx(endpoint, rel=1e-3)


@pytest.mark.parametrize("p", [2.0, 3.0, np.inf])
def test_mean_and_summed_risk(p):
    """repeating the data preserves mean risk and doubles summed risk"""
    X, y, beta = make_data(n=7, d=3)
    risk = RobustRisk(X, y, 0.2, p)
    repeated = RobustRisk(np.tile(X, (2, 1)), np.tile(y, 2), 0.2, p)

    for method in ("primal", "dual"):
        evaluate = getattr(risk, method)
        evaluate_repeated = getattr(repeated, method)
        mean = evaluate(beta)
        assert evaluate(beta, per_sample=False) == pytest.approx(len(y) * mean, rel=1e-6)
        assert evaluate_repeated(beta) == pytest.approx(mean, rel=1e-6)
        assert evaluate_repeated(beta, per_sample=False) == pytest.approx(
            2 * len(y) * mean, rel=1e-6
        )


@pytest.mark.parametrize("p", [3.0, 6.0])
@pytest.mark.parametrize("scale", [0.01, 100.0])
def test_response_scaling(p, scale):
    """scaling y and beta by c multiplies the risk by c**2"""
    X, y, beta = make_data()
    baseline = RobustRisk(X, y, 0.2, p).primal(beta)
    changed = RobustRisk(X, scale * y, 0.2, p).primal(scale * beta)
    assert changed / scale**2 == pytest.approx(baseline, rel=1e-8)


@pytest.mark.parametrize("p", [2.0, 2.01, 3.0, 6.0, 100.0, np.inf])
@pytest.mark.parametrize("a,B,delta", [
    ([0.2, 1.0, 3.0, 0.01], 1.7, 0.23),
    ([0.0, 1.0, 0.0, 3.0], 1.7, 0.23),
    ([0.0, 0.0, 0.0, 0.0], 1.7, 0.23),
    ([1.0, 1.0, 1.0, 1.0], 1.7, 0.23),
    ([0.0, 1.0, 0.0, 3.0], 0.0, 0.23),
    ([0.0, 1.0, 0.0, 3.0], 1.7, 0.0),
])
def test_returned_lengths_constraints_and_scalar_value(p, a, B, delta):
    """returned lengths are feasible and attain the scalar risk to numerical accuracy"""
    from DRO.robust_risk import _ScalarRisk

    a = np.asarray(a)
    scalar = _ScalarRisk(delta, p)
    value, t = scalar.solve(a, B, return_t=True)
    assert value == pytest.approx(scalar.solve(a, B), rel=1e-12)
    assert np.all(np.isfinite(t)) and np.all(t >= 0)
    assert np.linalg.norm(t, ord=p) == pytest.approx(len(a)**(1 / p) * delta,
                                                   rel=1e-9, abs=1e-12)
    assert np.sum((a + B * t)**2) == pytest.approx(value, rel=1e-7)


@pytest.mark.parametrize("p", PS)
@pytest.mark.parametrize("per_sample", [True, False])
def test_primal_can_return_maximizing_lengths(p, per_sample):
    """mean/sum normalization affects the value only; lengths attain that value"""
    X, y, beta = make_data(n=7, d=3)
    risk = RobustRisk(X, y, 0.2, p)
    value, t = risk.primal(beta, per_sample=per_sample, return_t=True)
    assert value == pytest.approx(risk.primal(beta, per_sample=per_sample), rel=1e-12)
    _, other_t = risk.primal(beta, per_sample=not per_sample, return_t=True)
    assert t == pytest.approx(other_t, abs=1e-12)
    assert np.linalg.norm(t, ord=p) == pytest.approx(len(y)**(1 / p) * 0.2, rel=1e-9)
    loss = (np.abs(X @ beta - y) + np.linalg.norm(beta, 1) * t)**2
    assert value == pytest.approx(np.mean(loss) if per_sample else np.sum(loss), rel=1e-7)
