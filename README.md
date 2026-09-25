# DRO

Evaluate and minimize the Wasserstein-$p$ robust squared loss for linear regression.

- [Evaluate the risk at a fixed beta](docs/robust_risk.md) with `RobustRisk`.
- [Fit beta with `CVXPY`](docs/cvx_solver.md) with `CvxOptimizer(risk)` or the standalone `minimize_cvx` function.
- [Fit beta with eta updates](docs/eta.md) using `EtaOptimizer(risk)` or `minimize_eta` ($\ell_\infty$ ground norm only).

The docs explain the mathematical formulations, numerical calculations, and input conventions.

## Installation

Use Python 3.13 or later. Install the package from the repository root.

```sh
python -m pip install -e .
```

The runtime dependencies are `NumPy`, `SciPy`, `CVXPY` 1.6.7, `dsp-cvxpy` 0.4.2, `h5py`, `scikit-learn`, and `Matplotlib`. The installation also includes `pytest` for running tests. The default CVX solver is `CLARABEL`. `SCS` is also supported. Both are included in a standard `CVXPY` installation.

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

# evaluate a fixed beta
risk = RobustRisk(X, y, delta, p)
beta = np.array([0.8, -0.4, 0.1])
print(risk.primal(beta))

# find beta minimizing the mean robust loss using cvx
fit = CvxOptimizer(risk).minimize()
print(fit.beta)
print(fit.value)

# find beta minimizing the mean robust loss using eta trick solver
fit = EtaOptimizer(risk).minimize()
print(fit.beta)
print(fit.value)
```

To fit directly from data without constructing a risk object yourself, use the standalone function.

```python
from DRO.cvx_solver import minimize_cvx

fit = minimize_cvx(X, y, radius=delta, p=p)
```

For the default $\ell_\infty$ ground norm, the [eta solver](docs/eta.md) is also available:

```python
from DRO.eta_solver import EtaOptimizer

fit = EtaOptimizer(risk).minimize()
```

`X` has shape `(n, d)` and `y` has shape `(n,)`. No intercept is added automatically. `delta` is the Wasserstein radius. The default ground norm is $\ell_\infty$, so the dual norm on $\beta$ is $\ell_1$. The scalar evaluator and both optimizers accept real exponents $2\leq p<\infty$ and `p=np.inf`.

The evaluator and both optimizers return the mean robust squared loss. Use `per_sample=False` with the evaluator only when you need summed loss. The fit includes the solver status and an independent risk evaluation. For `CVX`, a missing `beta` or a status other than `optimal` or `optimal_inaccurate` raises an exception. Eta uses fixed smoothing and reports `step_tolerance` or `iteration_limit` (`least-squares` at zero radius).

## Fast-rate experiment

Run or resume the default experiment with ten repetitions per $(n,p)$, then generate its plots from the saved fits. Run these commands from the repository root with your Python environment active.

```sh
PYTHONPATH=src python -m DRO.experiments.fast.rate
PYTHONPATH=src python -m DRO.experiments.fast.plot
```

For custom runs you can e.g. do:
```sh
PYTHONPATH=src python -m DRO.experiments.fast.rate \
    --n 24 48 \
    --p 2 3 6 inf \
    --k 1 \
    --fit-min-n 24 \
    --directory /tmp/fast-pilot
```
Both rate experiments use $\ell_\infty$ ground norm and default to `--solver cvx`. With default output paths, `--solver eta` saves fits in `data/fast_eta_<data config hash>/` while CVX uses `data/fast_<data config hash>/`. The default folders reuse saved seeds across solvers.

See [the experiment docs](docs/fast.md) for dependencies, settings, more about custom runs, and saved-data details.

## Slow-rate experiment

Run the default experiment and generate its rate plots in one command. Run from the repository root with your Python environment active.

```sh
PYTHONPATH=src python -m DRO.experiments.slow.rate
```

For a shorter run with separate outputs:

```sh
PYTHONPATH=src python -m DRO.experiments.slow.rate \
    --n 4096 8192 \
    --p 2 3 6 inf \
    --k 1 \
    --output-directory plots/slow-pilot/rate
```

To include smaller sizes while fitting the slope on the original range:

```sh
PYTHONPATH=src python -m DRO.experiments.slow.rate \
    --c 3 \
    --n 256 512 1024 2048 4096 6144 8192 10240 \
    --fit-n-min 4096 \
    --output-directory plots/slow/c3-extended/rate
```

Add `--solver eta` to use eta. Its default output folder is `plots/slow_eta/`. The default `CVX` output remains `plots/slow/`.

See [the experiment docs](docs/slow.md) for the data construction, slow-rate mechanism, settings, and saved results.


## Regime experiment

Run the experiment and save plots to `plots/regimes/` and data to `data/regimes/`:

```sh
PYTHONPATH=src python -m DRO.experiments.regimes.regime
```

Add `--plot-only` to redraw saved results.
