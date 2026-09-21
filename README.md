# DRO

Evaluate and minimize the Wasserstein-$p$ robust squared loss for linear regression.

- [Evaluate the risk at fixed coefficients](docs/robust_risk.md) with `RobustRisk`.
- [Fit coefficients with CVXPY](docs/cvx_solver.md) with `CvxOptimizer(risk)` or the standalone `minimize_cvx` function.

The docs explain the mathematical formulations, numerical calculations, and input conventions.

## Installation

Use Python 3.13 or later. From the repository root:

```sh
python -m pip install -e .
```

The runtime dependencies are NumPy, SciPy, `CVXPY` 1.6.7, and `dsp-cvxpy` 0.4.2. The default solver is `CLARABEL`; `SCS` is also supported. Both are included in a standard `CVXPY` installation.

## Fit and evaluate

```python
import numpy as np
from DRO.robust_risk import RobustRisk
from DRO.cvx_solver import CvxOptimizer

# draw some data
rng = np.random.default_rng(7)
X = rng.normal(size=(20, 3))
beta_star = np.array([1.0, -0.5, 0.0])
y = X @ beta_star + 0.2 * rng.normal(size=20)
delta, p = 0.15, 3

# evaluate a supplied coefficient vector
risk = RobustRisk(X, y, delta, p)
beta = np.array([0.8, -0.4, 0.1])
print(risk.primal(beta))

# find coefficients minimizing the same mean robust loss
fit = CvxOptimizer(risk).minimize()
print(fit.beta)
print(fit.value)
print(fit.diagnostics["model_value"])  # CVXPY value checked against fit.value
```

`CvxOptimizer` takes the data, radius, and norms from `risk`. To fit directly from data without constructing a risk object yourself, use the standalone function.

```python
from DRO.cvx_solver import minimize_cvx

fit = minimize_cvx(X, y, radius=delta, p=p)
```

`X` has shape `(n, d)` and `y` has shape `(n,)`. No intercept is added automatically. `delta` is the Wasserstein radius. The default ground norm is $\ell_\infty$, whose dual coefficient norm is $\ell_1$. Both the scalar evaluator and the `CVXPY` solver accept real exponents $2\leq p<\infty$ and `p=np.inf`.

Both the evaluator and solver return the mean robust squared loss. Use `per_sample=False` with the evaluator only when you need summed loss. The fit includes the solver status and an independent risk evaluation. Missing coefficients or a status other than `optimal` or `optimal_inaccurate` raise an exception.
