# Fast-rate experiment

The code lives in [src/DRO/experiments/fast/](../src/DRO/experiments/fast/). `rate.py` runs the experiment, `readwrite.py` handles saved fits, and `plot.py` generates figures. Install all dependencies with `python -m pip install -e .` from the repository root. The [README](../README.md#fast-rate-experiment) gives the default run commands.

## Settings

The model is $y=X\beta^*+\varepsilon$. Covariates are independent $\mathrm{Uniform}[-M,M]$, and Gaussian noise has variance $\sigma^2$. The first $s$ entries of $\beta^*$ equal $B/s$, so $\|\beta^*\|_1=B$. The radius is $\delta=KM\sqrt{\log(d/\gamma)/n}$ with $\ell_\infty$ ground norm. The default config is specified below.

| Flag | Meaning | Default |
| --- | --- | --- |
| `--solver` | optimizer (`cvx` or `eta`) | `cvx` |
| `--n` | sample-size grid | 9 sizes from 2560 to 163840 |
| `--p` | Wasserstein exponents | 2, 3, 6, inf |
| `--k` | repetitions per $(n, p)$ | 10 |
| `--d` | dimension | 10 |
| `--s` | nonzero coefficients | 5 |
| `--B` | signal norm | 15 |
| `--sigma-sq` | noise variance | 0.25 |
| `--gamma` | failure probability in the radius | 0.01 |
| `--K` | radius multiplier | 3.6 |
| `--M` | covariate bound | 1 |
| `--fit-min-n` | minimum sample size for fitted slopes | 40960 |

Use `--help` for all flags. Assume $1\leq s\leq d$, $p\geq2$, $M\geq1$, $0<\gamma<1$, and positive sample sizes, repetitions, signal norm, noise variance, and radius multiplier.

## Fast-rate bound

Let $q=p/(p-1)$ (with $q=1$ for $p=\infty$), $e_q=\|\varepsilon\|_q/n^{1/q}$, and $B=\|\beta^*\|_1$.

Define the following constants:

$$
\begin{aligned}
C_1 &= 3+\frac{\delta B}{e_q},\\
C_2 &= 3e_q+(2MC_1+\delta C_1+\delta)B,\\
C_3 &= 4C_1^2B^2,\\
C_4 &= \frac{16C_2^2s^2}{\kappa^2}=96C_2^2s^2,
\qquad \kappa=\frac{1}{\sqrt{6}}.
\end{aligned}
$$

The fast-rate bound then is

$$
\frac{\|X(\widehat{\beta}-\beta^*)\|_2^2}{n}
\leq \delta^2\max\{C_3,C_4\}
$$

under the theorem's assumptions. We track all constants as $n$ grows, as well as $e_q$.

## RE constant

The bound uses the fixed constant $\kappa=1/\sqrt{6}$ for every sample size and repetition. For independent $\mathrm{Uniform}[-1,1]$ entries and fixed $d$, $G=X^\top X/n$ converges to $I/3$, so $\lambda_{\min}(G)\geq1/6$ with probability tending to one as $n\to\infty$. On this event, Rayleigh–Ritz gives

$$
\frac{\|Xv\|_2^2}{n}=v^\top Gv
\geq\lambda_{\min}(G)\|v\|_2^2
\geq\kappa^2\|v\|_2^2
\quad\text{for every }v\in\mathbb R^d.
$$

This also holds on every restricted cone, certifying $\mathrm{RE}(s,\ell)$ for every admissible $s,\ell$. The same choice of $\kappa$ applies to custom runs with $M\geq1$, whose population Gram matrix is $M^2I/3$.

The RE plot shows $\sqrt{\max\{\lambda_{\min}(G),0\}}$ against $\kappa=1/\sqrt{6}$ and a check passes when the measured quantity is at least $\kappa$. A failed check does not prove that the restricted condition fails.

## Saved fits

Each `data/fast_<data config hash>/` (`CVX`) or `data/fast_eta_<data config hash>/` (`eta`) folder holds one `n=<n>_p=<p>_rep=<rep>.h5` per fit, including its seed, settings, and theorem quantities. `config.json` records the latest requested grid. `summary.json` gives means, slopes, and theorem checks for the last completed run.

Existing fits are reused, and their derived constants, bounds, and checks are recomputed and saved using the current formulas.

## Custom runs and plots

This small example saves its fits and plots separately from the default experiment. Run it from the repository root.

```sh
PYTHONPATH=src python -m DRO.experiments.fast.rate \
    --n 24 48 \
    --p 2 3 6 inf \
    --k 1 \
    --fit-min-n 24 \
    --directory /tmp/fast-pilot
PYTHONPATH=src python -m DRO.experiments.fast.plot \
    --directory /tmp/fast-pilot \
    --output-directory plots/fast-pilot
```

## Solver selection

Both solvers fit the same statistical problem. The experiment defaults to [CVX](cvx_solver.md) and [eta](eta.md) uses fixed smoothing and stops based on the change in beta. Solver choice does not change the data construction or theorem quantities.

```sh
PYTHONPATH=src python -m DRO.experiments.fast.rate --solver cvx
PYTHONPATH=src python -m DRO.experiments.fast.rate --solver eta
PYTHONPATH=src python -m DRO.experiments.fast.plot \
    --directory data/fast_eta_9a0f5a9f0d3f \
    --output-directory plots/fast_eta
```

For paired custom runs, pass the CVX folder as `--seed-directory` when running eta (or vice versa).

## Paired diagnostics from saved fits

No refitting is needed. The two fit folders, including their `config.json` files, contain the inputs for the accuracy-agreement and runtime diagnostics:

```sh
PYTHONPATH=src python -m DRO.experiments.fast.eta_cvx_diag \
    --cvx-directory data/fast_9a0f5a9f0d3f \
    --eta-directory data/fast_eta_9a0f5a9f0d3f \
    --output-directory plots/eta_cvx
```
