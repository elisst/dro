# Slow-rate experiment

[src/DRO/experiments/slow/rate.py](../src/DRO/experiments/slow/rate.py) constructs the data, fits robust regression, and generates the rate plot. See the [README](../README.md#slow-rate-experiment-d2) for installation and the default run command.

## Settings

The model is $y=X\beta^*+\varepsilon$, with $d=2$, $\beta^*=(2,-1)^\top$, and independent Gaussian noise of variance $\sigma^2$. The ground norm is $\ell_\infty$. The radius is

$$
\delta=KM\sqrt{\frac{\log(2/\gamma)}{n}},\qquad M=\sqrt{2}.
$$

| Flag | Meaning | Default |
| --- | --- | --- |
| `--n` | sample-size grid | 4096, 6144, 8192, 10240 |
| `--p` | Wasserstein exponents | 2, 3, 6, inf |
| `--k` | repetitions per $(n,p)$ | 10 |
| `--c` | correlation constant $c_0$ in $\rho_n=1-c_0/\sqrt n$ | 3 |
| `--sigma-sq` | noise variance | 0.25 |
| `--K` | radius multiplier | 3.6 |
| `--gamma` | failure-probability parameter in the radius rule | 0.01 |
| `--seed` | base random seed | 20260921 |
| `--fit-n-min` | smallest sample size included in slope fits | all plotted sizes |
| `--output-directory` | folder for both PDFs and results | `plots/slow/rate` |

Use `--help` for all flags. Assume at least two distinct sample sizes $n$ in increasing order, each divisible by four and satisfying $0<c_0/\sqrt n<1$. Assume $p\geq2$, $0<\gamma<1$, and positive repetitions, noise variance, and radius multiplier. If `--fit-n-min` is supplied, at least two sample sizes must remain in the fit range. The flag `--c` and saved setting `c` represent the paper's $c_0$.

We note that the construction does not require a particular value of $c_0$ as its choice affects the magnitude of the prediction error and the finite-sample behaviour.

## Correlated columns

Construct two length-$n$ column vectors $u,w$ by stacking $n/4$ copies of $(1,1,-1,-1)^\top$ and $(1,-1,1,-1)^\top$, respectively. For example, when $n=8$,

$$
u=(1,1,-1,-1,\;1,1,-1,-1)^\top,
\qquad
w=(1,-1,1,-1,\;1,-1,1,-1)^\top.
$$

Each vector has mean zero, and $u^\top w=0$. These are the building blocks for the two feature columns. With $\rho_n=1-c_0/\sqrt n$ and $c_0=3$, set

$$
X_1=\sqrt{\frac{1+\rho_n}{2}}u+\sqrt{\frac{1-\rho_n}{2}}w,
\qquad
X_2=\sqrt{\frac{1+\rho_n}{2}}u-\sqrt{\frac{1-\rho_n}{2}}w.
$$

Place these columns side by side to obtain $X=[X_1\;X_2]\in\mathbb R^{n\times2}$, i.e., each row is one observation with two features. The columns share the same $u$ contribution and have opposite $w$ contributions. They have unit empirical variance, correlation $\rho_n$, and entries bounded by $M=\sqrt2$. As $n$ grows, their correlation approaches one, making the separate effects of the two coefficients harder to distinguish. This creates a harder example in which robust regression exhibits prediction error close to the slow $n^{-1/2}$ rate. There is no positive RE constant uniform in $n$ (see paper for more rigorous argument)

## Saved results and plots

`plots/slow/rate` contains `paper.pdf`, `with_config.pdf`, and `results.json`. The JSON records the configuration, fitted slopes, and individual fits.

Points show mean prediction error $\|X(\hat\beta-\beta^*)\|_2^2/n$, with one standard error of the mean as error bars. Smaller sizes show finite-sample curvature.

## Run the experiment

For a shorter run with separate outputs:

```sh
PYTHONPATH=src python -m DRO.experiments.slow.rate \
    --n 4096 8192 \
    --k 1 \
    --output-directory plots/slow-pilot/rate
```

To reproduce the configuration of the current saved figure, including the smaller sizes:

```sh
PYTHONPATH=src python -m DRO.experiments.slow.rate \
    --c 3 \
    --n 256 512 1024 2048 4096 6144 8192 10240 \
    --fit-n-min 4096
```

Rerunning fits every requested dataset and replaces the results in the chosen output folder. Use `--output-directory` to save a separate run.
