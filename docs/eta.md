# Minimize the robust risk with eta updates

[`EtaOptimizer`](../src/DRO/eta_solver.py) uses the eta trick and repeated weighted ridge fits, with fixed smoothing. It supports $2\leq p\leq\infty$ and $\ell_\infty$ ground norm, so the $\beta$ norm is $\ell_1$.

## Fit and inspect the result

```python
import numpy as np
from DRO.robust_risk import RobustRisk
from DRO.eta_solver import EtaOptimizer, minimize_eta

# draw some data and form risk object
rng = np.random.default_rng(7)
X = rng.normal(size=(20, 3))
y = X @ np.array([1.0, -0.5, 0.0]) + 0.2 * rng.normal(size=20)
risk = RobustRisk(X, y, delta=0.15, p=3.0, norm=np.inf)

# minimize
fit = EtaOptimizer(risk).minimize()
print(fit.beta)
print(fit.value, fit.diagnostics["model_value"])

# equivalent convenience function
fit = minimize_eta(X, y, radius=0.15, p=3.0)
```

`fit.beta` contains the fitted beta and `fit.value` is the original, unsmoothed mean robust squared loss. No intercept is added.

## Response scaling and initialization

As in the [CVX solver](cvx_solver.md#response-scaling), the code sets `scale` to $s=\sqrt{\operatorname{mean}(y^2)}$, or $1$ for a zero response, and solves using `y = risk.y / scale`. The internal `beta` is in these scaled units while $X$ and $\delta$ stay unchanged. The positive `epsilon` is fixed in scaled units throughout the fit.

The default starting point is a ridge fit. Uniform simplex weights $\eta_j^{(i)}=1/(d+1)$ give

$$
G_\delta(\beta,\eta)
=(d+1)\left[\frac1n\|X\beta-y\|_2^2+\delta^2\|\beta\|_2^2\right].
$$

The common factor $(d+1)$ does not affect the minimizer. Since sklearn uses summed loss, the corresponding ridge penalty is `alpha=len(y) * delta**2`. A supplied `beta0` replaces this start and is divided by `scale`. For `delta == 0`, the solver returns ordinary least squares directly.

## The iteration

The code's `r_smooth`, `beta_smooth`, and `beta_norm` correspond to the paper's $r_i$, $b_j$, and $B$ in the pseudo-algorithm:

```python
r_smooth = np.hypot(X @ beta - y, epsilon)
beta_smooth = np.hypot(beta, epsilon)
beta_norm = float(beta_smooth.sum())
t_star = scalar_risk.transport(r_smooth, beta_norm)

w = (r_smooth + t_star * beta_norm) / r_smooth
gamma = (
    float(np.mean(t_star * (r_smooth + t_star * beta_norm)))
    / beta_smooth
)
beta_next = _weighted_ridge(X, y, w, gamma, solver=solver, **solver_options)
```

`np.hypot(a, epsilon)` computes $\sqrt{a^2+\epsilon^2}$, keeping the denominators positive. `_ScalarRisk.transport` supplies the worst-case lengths $\boldsymbol t^\star$ for these smoothed magnitudes; see the [robust risk doc](robust_risk.md) for the scalar reduction. The weights `w` and `gamma` encode the eta update, so no full eta array is stored.

`_weighted_ridge` minimizes

$$
\frac1n\sum_i w_i(x_i^\top\beta-y_i)^2+\sum_j\gamma_j\beta_j^2,
$$

where the weights are fixed at the current iterate:

$$
w_i=\frac{r_i+t_i^\star B}{r_i},
\qquad
\gamma_j=\frac{\frac1n\sum_i t_i^\star(r_i+t_i^\star B)}{b_j}.
$$

Here $r_i$ and $b_j$ are the smoothed magnitudes above, and $B=\sum_j b_j$. This gives the paper's normal equations $(X^\top W X+n\Gamma)\beta^+=X^\top W y$. The smoothing terms are constant in this solve and can be omitted.

To use sklearn's single ridge penalty, write $\beta=D\theta$ with $D_{jj}=1/\sqrt{n\gamma_j}$. Then $X\beta=(XD)\theta$ and $n\sum_j\gamma_j\beta_j^2=\|\theta\|_2^2$. The helper fits the scaled columns with `alpha=1` and `sample_weight=w`, then recovers $\beta=D\theta$. The default solver is `cholesky`; additional options are passed to `Ridge`.

## Stopping and diagnostics

After each fit, the code records

$$
\text{relative\_step}=\frac{\|\beta^+ - \beta\|_2}{1+ \|\beta\|_2 },
$$

updates `beta`, and stops when `relative_step <= tol` or `maxiter` is reached. This checks the change in beta, not an optimality gap.

The internal `beta` is multiplied by `scale` before returning, and objective values are restored by `scale**2`. Diagnostics include:

- `status`: `step_tolerance`, `iteration_limit`, or `least-squares`.
- `iterations`, `relative_step`: the iteration count and final stopping quantity.
- `model_value`: the smoothed mean objective at the returned beta.
- `scalar_value`: the unsmoothed risk evaluated independently by `RobustRisk.primal`, also returned as `fit.value`.
- `smoothing`, `response_scale`, `solver`, `formulation`: the smoothing in original units and the numerical settings used.

The difference between `model_value` and `scalar_value` reflects smoothing and numerical evaluation, not an optimization gap. The [solver tests](../tests/eta_solver_test.py) and [joint CVX tests](../tests/eta_joint_cvx_test.py) check the implementation against independent convex solves.
