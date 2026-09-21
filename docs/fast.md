# Fast-rate experiment

The code lives in [src/DRO/experiments/fast/](../src/DRO/experiments/fast/). `rate.py` runs the experiment, `readwrite.py` handles saved fits, and `plot.py` generates figures. Install all dependencies with `python -m pip install -e .` from the repository root. The [README](../README.md#fast-rate-experiment) gives the default run commands.

## Settings

The model is $y=X\beta^*+\varepsilon$. Covariates are independent $\mathrm{Uniform}[-M,M]$, and Gaussian noise has variance $\sigma^2$. The first $s$ coefficients equal $B/s$, so $\|\beta^*\|_1=B$. The radius is $\delta=KM\sqrt{\log(d/\gamma)/n}$ with $\ell_\infty$ ground norm. The default config is specified below.

| Flag | Meaning | Default |
| --- | --- | --- |
| `--n` | sample-size grid | 9 sizes from 2560 to 163840 |
| `--p` | Wasserstein exponents | 2, 3, 6, inf |
| `--k` | repetitions per (n, p) | 3 |
| `--d` | dimension | 10 |
| `--s` | nonzero coefficients | 5 |
| `--B` | signal norm | 15 |
| `--sigma-sq` | noise variance | 0.25 |
| `--gamma` | failure probability in the radius | 0.01 |
| `--K` | radius multiplier | 3.6 |
| `--M` | covariate bound | 1 |
| `--fit-min-n` | minimum sample size for fitted slopes | 40960 |

Use `--help` for all flags. `--K` and `--k` are distinct. Assume $1\leq s\leq d$, $p\geq2$, $M\geq1$, $0<\gamma<1$, and positive sample sizes, repetitions, signal norm, noise variance, and radius multiplier.

## Saved fits

Each `data/fast_<config hash>/` folder holds one `n=<n>_p=<p>_rep=<rep>.h5` per fit, including its seed, settings, and theorem quantities. `config.json` records the latest requested grid. `summary.json` gives means, slopes, and theorem checks for the last completed run.

Existing fits are reused. Add `--read-only` to update the summary without fitting, with an error if a requested fit is missing. Changing `n`, `p`, `k`, or `fit_min_n` keeps the same folder. Other settings change the hash. Extra fits remain available even when the latest requested grid is smaller.

Stored seeds are preserved. New seeds depend on `(n, p, repetition)` and are shared across configurations. Use a separate `--directory` when comparing solvers or numerical environments, since existing fits are reused after software changes.

## Custom runs and plots

This small example saves its fits and plots separately from the default experiment. Run it from the repository root.

```sh
PYTHONPATH=src python -m DRO.experiments.fast.rate --n 24 48 --p 2 3 6 inf --k 1 --fit-min-n 24 --directory /tmp/fast-pilot
PYTHONPATH=src python -m DRO.experiments.fast.plot --directory /tmp/fast-pilot --output-directory plots/fast-pilot
```

Plotting reads the grid in `config.json` without fitting. Each of the 12 quantities gets a subfolder with `paper.pdf` and `with_config.pdf`. Rerunning replaces the PDFs in the chosen output folder.
