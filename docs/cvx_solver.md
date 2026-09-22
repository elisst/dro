# Minimize the robust risk

[`CvxOptimizer`](../src/DRO/cvx_solver.py) solves $\min_\beta V_\delta(\beta)$. It uses the explicit endpoint formulas with `CVXPY` and the gamma saddle formulation with [`DSP`](https://github.com/cvxgrp/dsp) for $2<p<\infty$. The [robust risk doc](robust_risk.md) derives these formulas and explains the scalar method used to evaluate the fitted risk independently.

## Fit and inspect the result

```python
import numpy as np
from DRO.robust_risk import RobustRisk
from DRO.cvx_solver import CvxOptimizer, minimize_cvx

# draw some data and form risk object
rng = np.random.default_rng(7)
X = rng.normal(size=(20, 3))
y = X @ np.array([1.0, -0.5, 0.0]) + 0.2 * rng.normal(size=20)
risk = RobustRisk(X, y, delta=0.15, p=3.0, norm=np.inf)

# minimize
fit = CvxOptimizer(risk).minimize()
print(fit.beta)
print(fit.value, fit.diagnostics["model_value"])

# equivalent convenience function
fit = minimize_cvx(X, y, radius=0.15, p=3.0, norm=np.inf)
```

`fit.beta` contains the fitted coefficients and `fit.value` is the mean robust squared loss, $V_\delta(\beta)$.

## Response scaling

We rescale the problem to enhance numerical stability, since the magnitude of the numbers in the problem can affect the accuracy of the solver.

Before forming the model, `minimize` chooses $s=\sqrt{\operatorname{mean}(y^2)}$, or $s=1$ when $y=0$. It solves using $\widetilde y=y/s$ and $\widetilde\beta$ representing $\beta/s$. This reduces sensitivity to the units in which the response $y$ is recorded.

For every feasible perturbation length $t_i$,

$$
|x_i^\top\widetilde\beta-\widetilde y_i|+t_i\|\widetilde\beta\|_*
=\frac1s\bigl(|x_i^\top\beta-y_i|+t_i\|\beta\|_*\bigr).
$$

Squaring and maximizing over the same length budget proves

$$
V_{\delta,y/s}(\beta/s)=\frac{V_{\delta,y}(\beta)}{s^2},
\qquad \widehat\beta=s\widehat{\widetilde\beta}.
$$

Neither $X$ nor $\delta$ changes, meaning transport still measures movement in the original covariate space.

```python
scale = float(np.sqrt(np.mean(np.asarray(risk.y) ** 2))) or 1.0
beta = cp.Variable(X.shape[1], name="beta")
r = X @ beta - risk.y / scale
B = cp.norm(beta, risk.norm_dual)
```

The code's `beta`, `r`, and `B` are thus in scaled response units. This differs from the [robust risks scaling](robust_risk.md#scale-the-lengths-and-loss), which also uses the known residuals and coefficient norm. Both transformations are exact changes of variables so neither changes the mathematical problem.

## The problem passed to the solver

`_form_problem` uses $r=X\widetilde\beta-\widetilde{y}$ and $B=\|\widetilde\beta\|_*$ from the preceding block.

**Zero radius.** No covariate movement is allowed, so the objective is ordinary mean squared loss:

```python
if delta == 0:  # OLS
    objective = cp.sum_squares(r) / n
    formulation = "least-squares"
```

**Quadratic exponent.** The triangle-inequality argument in [the explicit cases](robust_risk.md#explicit-cases) gives $V=(\|r\|_2/\sqrt n+\delta B)^2$. This nonnegative convex expression can be squared within `CVXPY`. With ground norm $\ell_\infty$, this is the squared square-root-Lasso objective:

```python
elif p == 2:  # square-root lasso
    objective = cp.square(cp.norm(r, 2) / np.sqrt(n) + delta * B)
    formulation = "p=2"
```

**Infinite exponent.** Every observation can use length $\delta$, giving $V=\frac1n\sum_i(|r_i|+\delta B)^2$:

```python
elif np.isinf(p):  # adversarial training
    objective = cp.sum_squares(cp.abs(r) + delta * B) / n
    formulation = "p=infinity"
```

These formulas are Corollary 1 of the paper. Each becomes `cp.Problem(cp.Minimize(objective))`.

**Intermediate exponents.** Theorem 6, Eq. (22), writes $nV_\delta(\beta)=\max_{\gamma\geq0}K(\beta,\gamma)$, with $K$ given in the [gamma formulation](robust_risk.md#gamma-formulation):

$$
K(\beta,\gamma)=n^{1/p}\delta B\left(\sum_i\gamma_i\right)^{1/q}
+\sum_i |r_i|\gamma_i^{1/q}-\frac14\sum_i\gamma_i^{2/q}.
$$

Fitting coefficients therefore solves

$$
\min_\beta\max_{\gamma\geq0}K(\beta,\gamma).
$$

For fixed $\gamma$, the terms involving $\beta$ are nonnegative multiples of a norm and absolute residuals, hence convex. For fixed $\beta$, the power exponents make $K$ concave in $\gamma$. We use [`DSP`](https://github.com/cvxgrp/dsp) to solve this saddle problem.

```python
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
```

## Restore units and interpret diagnostics

After solving, the coefficients are multiplied by $s$, and the risk is evaluated with the original `RobustRisk` object. If $J$ is the solver objective, its mean-risk value in original units is $s^2J$ for the closed form special cases and $s^2J/n$ for `DSP`:

```python
beta_hat = beta.value * scale
value = float(risk.primal(beta_hat))
model_value = (
    float(problem.value) * scale**2
    / (n if formulation == "dsp-gamma" else 1)
)
```

Diagnostics retain enough information to inspect the numerical fit and compare the independent risk evaluations:

- `model_value` -- $V_\delta(\widehat{\beta})$ returned by `DSP`.
- `scalar_value` -- $V_\delta(\widehat{\beta})$ calculated by evaluating the `RobustRisk` object at $\widehat{\beta}$. This can be used as an independent check that the result of the optimization was sound when running solver diagnostics later on.
- `response_scale` -- records the response scale used in the solve.
- `formulation`, `solver`, `status` -- what special case it is (e.g. `"dsp-gamma"` or `"p=infinity"`), what numerical solver was used and reported convergence status

The code accepts `optimal` and `optimal_inaccurate` statuses when coefficients are returned. It exposes the two risk values for comparison, and does not automatically reject a discrepancy. The [tests](../tests/cvx_solver_test.py) check these values and compare fitted minima with known solutions and independent optimizers.

`CLARABEL` is the default solver. Other options are forwarded to the selected solver, for example:

```python
second = CvxOptimizer(risk).minimize(
    solver="SCS", eps_abs=1e-8, eps_rel=1e-8
)
print(fit.value, second.value)
```

For the `DSP` branch, `eps` (default `1e-3`) controls the absolute part of `DSP`'s agreement check between its two convex-problem values. The code multiplies it by $n$ to convert from scaled mean-loss units to scaled summed-loss units. It is separate from the underlying solver's convergence tolerances.
