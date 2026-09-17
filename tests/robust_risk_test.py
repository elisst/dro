"""
tests for RobustRisk

the core check compares the scalar reduction in eqs (14)/(35) with the gamma formulation in eq (20)
"""

import numpy as np
import pytest
from scipy.optimize import minimize

from DRO.robust_risk import RobustRisk

PS = [2.0, 2.5, 3.0, 6.0]
NORMS = [1.0, 2.0, np.inf]
DELTAS = [0.01, 0.1, 1.0]


def make_data(n=30, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    beta_star = rng.standard_normal(d)
    y = X @ beta_star + 0.5 * rng.standard_normal(n)
    beta = beta_star + 0.3 * rng.standard_normal(
        d
    )  # some non-optimal beta to evaluate at
    return X, y, beta


@pytest.mark.parametrize("delta", DELTAS)
@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("p", PS)
def test_primal_equals_dual(p, norm, delta):
    X, y, beta = make_data()
    rr = RobustRisk(X, y, delta, p, norm=norm)
    assert rr.primal(beta) == pytest.approx(rr.dual(beta), rel=1e-6)


@pytest.mark.parametrize("p", [2.01, 2.1, np.e, np.pi, 7.0, 100.0])
def test_primal_equals_dual_general_exponents(p):
    """Cross-check non-integer orders and orders near the endpoints."""
    X, y, beta = make_data()
    rr = RobustRisk.normalized(X, y, 0.2, p)
    assert rr.primal(beta) == pytest.approx(rr.dual(beta), rel=1e-6)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_primal_equals_dual_other_data(seed):
    X, y, beta = make_data(n=12, d=8, seed=seed)
    rr = RobustRisk(X, y, 0.2, 3.0, norm=2.0)
    assert rr.primal(beta) == pytest.approx(rr.dual(beta), rel=1e-6)


@pytest.mark.parametrize("p", [2.0, 6.0])
def test_primal_equals_dual_large_n(p):
    """Compare the two risk representations on a larger sample."""
    X, y, beta = make_data(n=5000, d=10)
    rr = RobustRisk(X, y, 0.1, p)
    assert rr.primal(beta) == pytest.approx(rr.dual(beta), rel=1e-6)


@pytest.mark.parametrize("p", [2.0, 6.0])
def test_primal_equals_dual_large_d(p):
    """d never enters the optimization, it only reaches the problem through ||beta||_*.
    beta is scaled by 1/sqrt(d) so that ||beta||_* stays O(sqrt(d)) rather than O(d).
    This keeps the CVXPY objective at a manageable scale"""
    rng = np.random.default_rng(0)
    n, d = 200, 200
    X = rng.standard_normal((n, d))
    beta_star = rng.standard_normal(d) / np.sqrt(d)
    y = X @ beta_star + 0.5 * rng.standard_normal(n)
    beta = beta_star + 0.3 * rng.standard_normal(d) / np.sqrt(d)
    rr = RobustRisk(X, y, 0.1, p)
    assert rr.primal(beta) == pytest.approx(rr.dual(beta), rel=1e-6)


@pytest.mark.parametrize("p", [2.0, 3.0])
def test_primal_equals_dual_ill_conditioned(p):
    """Compare the two risk representations with nearly collinear columns."""
    rng = np.random.default_rng(7)
    n, d = 200, 60
    factor = rng.standard_normal((n, 1))
    X = 0.95 * factor + 0.05 * rng.standard_normal((n, d))
    beta_star = rng.standard_normal(d)
    y = X @ beta_star + 0.5 * rng.standard_normal(n)
    beta = beta_star + 0.3 * rng.standard_normal(d)
    rr = RobustRisk(X, y, 0.1, p)
    assert rr.primal(beta) == pytest.approx(rr.dual(beta), rel=1e-6)


@pytest.mark.parametrize("delta", [0.05, 0.3])
@pytest.mark.parametrize("p", PS)
def test_primal_equals_brute_force(p, delta):
    """Check against direct optimization over covariate perturbations.
    The budget is sum_i ||u_i||^p <= n delta, where delta is the average cost.
    SLSQP uses multiple starting points because this problem is nonconvex.
    It does not guarantee a global maximum.
    """
    X, y, beta = make_data(n=6, d=3)
    n, d = X.shape

    def neg_objective(u_flat):
        U = u_flat.reshape(n, d)
        return -np.sum(((X + U) @ beta - y) ** 2)

    def budget(u_flat):
        U = u_flat.reshape(n, d)
        return n * delta - np.sum(np.linalg.norm(U, axis=1) ** p)

    rng = np.random.default_rng(0)
    best = -np.inf
    statuses = []
    for trial in range(5):  # try several starts because the objective is nonconvex in u
        u0 = 0.1 * (trial + 1) * rng.standard_normal(n * d)
        res = minimize(
            neg_objective,
            u0,
            constraints=[{"type": "ineq", "fun": budget}],
            method="SLSQP",
            options={"maxiter": 2000, "ftol": 1e-10},
        )
        statuses.append(str(res.message))
        if res.success:
            U = res.x.reshape(n, d)
            spent = np.sum(np.linalg.norm(U, axis=1) ** p)
            if spent > n * delta:
                U = U * ((n * delta) / spent) ** (1 / p)
            assert budget(U.ravel()) >= -1e-12 * (1 + n * delta)
            best = max(best, -neg_objective(U.ravel()))

    rr = RobustRisk(X, y, delta, p, norm=2.0)
    assert np.isfinite(best), f"No successful feasible SLSQP reference: {statuses}"
    assert rr.primal(beta) == pytest.approx(best / n, rel=1e-6)


@pytest.mark.parametrize("delta", DELTAS)
@pytest.mark.parametrize("norm", NORMS)
def test_p2_is_sqrt_lasso(norm, delta):
    """p = 2 is the sqrt-Lasso endpoint. the normalized robust risk is the squared
    sqrt-Lasso objective, which has the explicit form

        sqrt(V_delta) = sqrt(mean_i r_i^2) + delta ||beta||_*

    with the default ground norm inf, ||beta||_* = ||beta||_1 is the usual Lasso penalty
    and delta is the sqrt-Lasso regularization strength for the mean-loss convention"""
    X, y, beta = make_data()
    rr = RobustRisk.normalized(X, y, delta, 2.0, norm=norm)
    r = X @ beta - y
    beta_dual_norm = np.linalg.norm(beta, rr.norm_dual)
    sqrt_lasso = np.sqrt(np.mean(r**2)) + delta * beta_dual_norm
    assert rr.primal(beta) == pytest.approx(sqrt_lasso**2, rel=1e-6)
    assert rr.dual(beta) == pytest.approx(sqrt_lasso**2, rel=1e-6)


@pytest.mark.parametrize("p", [*PS, np.inf])
def test_zero_delta_is_least_squares(p):
    X, y, beta = make_data()
    rr = RobustRisk(X, y, 0.0, p)
    expected = np.mean((X @ beta - y) ** 2)
    assert rr.primal(beta) == pytest.approx(expected, rel=1e-6)
    assert rr.dual(beta) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("p", [*PS, np.inf])
def test_zero_beta_is_least_squares(p):
    """||beta||_* = 0, so nothing is gained by moving x whatever delta is"""
    X, y, _ = make_data()
    rr = RobustRisk(X, y, 0.5, p)
    beta = np.zeros(X.shape[1])
    expected = np.mean(y**2)
    assert rr.primal(beta) == pytest.approx(expected, rel=1e-6)
    assert rr.dual(beta) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("p", [*PS, np.inf])
def test_perfect_fit(p):
    """At interpolation, uniform lengths attain mean risk (delta*||beta||_*)**2."""
    X, _, beta = make_data()
    radius = 0.3
    rr = RobustRisk.normalized(X, X @ beta, radius, p)
    expected = (radius * np.linalg.norm(beta, 1)) ** 2
    assert rr.primal(beta) == pytest.approx(expected, rel=1e-6)
    assert rr.dual(beta) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("p", PS)
def test_increasing_in_delta(p):
    """a bigger ambiguity set can only raise the sup"""
    X, y, beta = make_data()
    values = [
        RobustRisk(X, y, delta, p).primal(beta) for delta in (0.0, 0.05, 0.2, 1.0)
    ]
    assert all(a < b for a, b in zip(values, values[1:]))


@pytest.mark.parametrize("p", PS)
def test_convex_in_beta(p):
    """theorem 1 states K, and hence V_delta, is convex in beta"""
    X, y, _ = make_data()
    rng = np.random.default_rng(11)
    d = X.shape[1]
    b1, b2 = rng.standard_normal(d), rng.standard_normal(d)
    rr = RobustRisk(X, y, 0.2, p)
    v1, v2 = rr.primal(b1), rr.primal(b2)
    for w in (0.1, 0.25, 0.5, 0.75, 0.9):
        chord = w * v1 + (1 - w) * v2
        assert rr.primal(w * b1 + (1 - w) * b2) <= chord * (1 + 1e-9)


@pytest.mark.parametrize("delta", [0.1, 0.3, 0.9])
@pytest.mark.parametrize("norm", NORMS)
def test_p_inf_is_adversarial(norm, delta):
    """at p = inf the cost is the hard constraint ||u|| <= delta, so the risk is the
    adversarial one, with the dual norm written out per ground norm.
    Check each evaluation method against this explicit formula"""
    X, y, beta = make_data()
    r = np.abs(X @ beta - y)
    dual_norm = {1.0: np.inf, 2.0: 2.0, np.inf: 1.0}[norm]
    adversarial = np.mean((r + delta * np.linalg.norm(beta, dual_norm)) ** 2)
    rr = RobustRisk(X, y, delta, np.inf, norm=norm)
    assert rr.primal(beta) == pytest.approx(adversarial, rel=1e-6)
    assert rr.dual(beta) == pytest.approx(adversarial, rel=1e-6)
    # the W_inf radius already bounds per-point movement, so normalizing is a no-op here
    assert RobustRisk.normalized(X, y, delta, np.inf, norm=norm).primal(beta) == pytest.approx(adversarial)


@pytest.mark.parametrize("delta", [0.1, 0.3, 0.9])
def test_large_p_approaches_adversarial(delta):
    """The normalized risk decreases towards the p = inf endpoint as p grows."""
    X, y, beta = make_data()
    endpoint = RobustRisk(X, y, delta, np.inf).primal(beta)
    values = [
        RobustRisk.normalized(X, y, delta, p).primal(beta) for p in (5.0, 10.0, 20.0, 100.0)
    ]
    assert all(a > b for a, b in zip(values, values[1:]))  # decreasing in p
    assert all(v > endpoint for v in values)  # approached from above
    assert values[-1] == pytest.approx(endpoint, rel=1e-3)


@pytest.mark.parametrize("p", PS)
def test_normalized_is_delta_to_the_p(p):
    """V_delta = bar V_(delta^p)"""
    X, y, beta = make_data()
    delta = 0.4
    normalized = RobustRisk.normalized(X, y, delta, p)
    plain = RobustRisk(X, y, delta**p, p)
    assert normalized.delta == delta
    assert plain.delta == pytest.approx(delta)
    assert normalized.primal(beta) == pytest.approx(plain.primal(beta), rel=1e-12)
    assert normalized.dual(beta) == pytest.approx(plain.dual(beta), rel=1e-6)


@pytest.mark.parametrize("p", [2.0, 3.0, np.inf])
@pytest.mark.parametrize("method", ["primal", "dual"])
def test_default_is_mean_and_summed_loss_is_explicit(p, method):
    X, y, beta = make_data(n=7, d=3)
    risk = RobustRisk.normalized(X, y, 0.2, p)
    evaluate = getattr(risk, method)
    mean = evaluate(beta)
    assert mean == pytest.approx(evaluate(beta, per_sample=True), rel=1e-6)
    assert evaluate(beta, per_sample=False) == pytest.approx(len(y) * mean, rel=1e-6)

    # Repeating the dataset leaves the empirical distribution and mean risk unchanged.
    repeated = RobustRisk.normalized(np.tile(X, (2, 1)), np.tile(y, 2), 0.2, p)
    repeated_evaluate = getattr(repeated, method)
    assert repeated_evaluate(beta) == pytest.approx(mean, rel=1e-6)
    assert repeated_evaluate(beta, per_sample=False) == pytest.approx(2 * len(y) * mean, rel=1e-6)


@pytest.mark.parametrize("p", [3., 6.])
@pytest.mark.parametrize("scale", [1e-20, 1e-10, 1., 1e20])
@pytest.mark.parametrize("residual", [0., .3])
def test_scalar_search_respects_response_units_at_one_observation(p, scale, residual):
    # A single observation has the exact risk (|r| + delta*||beta||_*)**2.
    beta = np.array([scale])
    risk = RobustRisk.normalized(np.ones((1, 1)), np.array([(1 - residual) * scale]), .1, p)
    assert risk.primal(beta) / scale**2 == pytest.approx((residual + .1)**2, rel=1e-8)


@pytest.mark.parametrize("p", [3., 6.])
@pytest.mark.parametrize("scale", [1e-20, 1e20])
def test_scalar_search_respects_response_units_for_unequal_residuals(p, scale):
    X, y, beta = make_data(n=20, d=4)
    baseline = RobustRisk.normalized(X, y, .2, p).primal(beta)
    changed = RobustRisk.normalized(X, y * scale, .2, p).primal(beta * scale)
    assert changed / scale**2 == pytest.approx(baseline, rel=1e-8)


def test_scalar_optimizer_failure_is_not_silently_accepted(monkeypatch):
    from types import SimpleNamespace
    import DRO.robust_risk as module
    monkeypatch.setattr(module, "brentq", lambda *args, **kwargs:
                        (1., SimpleNamespace(converged=False, flag="test failure")))
    X, y, beta = make_data()
    with pytest.raises(RuntimeError, match="Scalar risk root solve failed"):
        RobustRisk.normalized(X, y, .2, 3).primal(beta)


def test_scalar_bad_success_flag_is_caught_by_risk_bounds(monkeypatch):
    from types import SimpleNamespace
    import DRO.robust_risk as module
    monkeypatch.setattr(module, "brentq", lambda *args, **kwargs:
                        (1., SimpleNamespace(converged=True)))
    X, y, beta = make_data()
    with pytest.raises(RuntimeError, match="Scalar risk bounds disagree"):
        RobustRisk.normalized(X, y, .2, 3).primal(beta)


@pytest.mark.parametrize('delta,coefficient', [(1e-100, 1e100), (1e100, 1e-100)])
@pytest.mark.parametrize('p', [2., 3., 6., np.inf])
def test_radius_need_not_have_representable_pth_power(delta, coefficient, p):
    """at interpolation with one observation the risk is (delta*B)**2 = 1"""
    X = np.ones((1, 1))
    beta = np.array([coefficient])
    risk = RobustRisk.normalized(X, X @ beta, delta, p)
    assert risk.delta == delta
    assert risk.primal(beta) == pytest.approx(1.)


@pytest.mark.parametrize('p', [3., 6.])
def test_risk_stays_finite_when_original_lambda_overflows(p):
    """one observation has t=delta even when the equivalent lambda exceeds float range"""
    delta = 1e-100 if p == 3 else 1e-50
    coefficient = 1e100
    X = np.ones((1, 1))
    y = np.array([.7 * coefficient])
    beta = np.array([coefficient])
    risk = RobustRisk.normalized(X, y, delta, p)
    value = risk.primal(beta)
    adversarial_residual = abs(coefficient - y[0]) + delta * coefficient
    assert value == pytest.approx(adversarial_residual**2)


@pytest.mark.parametrize('p', [2., 3., 6., np.inf])
def test_integer_inputs_preserve_fractional_transport(p):
    X = np.array([[1]])
    y = np.array([0])
    beta = np.array([1])
    risk = RobustRisk.normalized(X, y, .25, p)
    assert risk.primal(beta) == pytest.approx(1.25**2)
