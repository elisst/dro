"""
d=2 slow-rate experiment

fit and plot in one file

assume n is a multiple of four and 0 < c0/sqrt(n) < 1.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from DRO.cvx_solver import CvxOptimizer
from DRO.robust_risk import RobustRisk
from DRO.experiments.fast.plot import (
    COLORS,
    RATE_FIGSIZE,
    MARKERS,
    exponent_label,
    exponent_legend,
    new_figure,
    plt,
    style_axis,
)


def default_config():
    """
    fixed dimension and signal

    only correlation and radius change with n
    """
    return argparse.Namespace(
        n=(4096, 6144, 8192, 10240),
        p=(2.0, 3.0, 6.0, np.inf),
        k=10,
        c=3.0,  # c_0 in the paper; set with --c
        sigma_sq=0.25,
        K=3.6,
        gamma=0.01,
        seed=20260921,
        fit_n_min=None,
    )


def make_design(n, c0):
    """centered columns with correlation rho_n and entries bounded by sqrt(2)"""
    rho_n = 1 - c0 / np.sqrt(n)
    u = np.tile([1.0, 1.0, -1.0, -1.0], n // 4)
    w = np.tile([1.0, -1.0, 1.0, -1.0], n // 4)
    X1 = np.sqrt((1 + rho_n) / 2) * u + np.sqrt((1 - rho_n) / 2) * w
    X2 = np.sqrt((1 + rho_n) / 2) * u - np.sqrt((1 - rho_n) / 2) * w
    return np.column_stack((X1, X2))


def run(config):
    """run one config"""
    beta_star = np.array([2.0, -1.0])
    rows = []
    for n in config.n:
        X = make_design(n, config.c)
        delta = config.K * np.sqrt(2) * np.sqrt(np.log(2 / config.gamma) / n)
        for repetition in range(config.k):
            rng = np.random.default_rng(config.seed + n + repetition)
            noise = np.sqrt(config.sigma_sq) * rng.standard_normal(n)
            y = X @ beta_star + noise
            for p in config.p:
                print(f"fitting p={p:g} n={n} repetition={repetition + 1}", flush=True)
                risk = RobustRisk(X, y, delta, p, norm=np.inf)
                started = perf_counter()
                fit = CvxOptimizer(risk).minimize()
                seconds = perf_counter() - started
                error = np.mean((X @ (fit.beta - beta_star)) ** 2)
                rows.append(
                    {
                        "n": n,
                        "p": f"{p:g}",
                        "repetition": repetition + 1,
                        "beta_hat": fit.beta.tolist(),
                        "prediction_error": float(error),
                        "fit_seconds": seconds,
                        "status": fit.diagnostics["status"],
                        "robust_risk": fit.value,
                        "model_value": fit.diagnostics["model_value"],
                    }
                )
    return rows


def plot_rates(config, rows, output):
    """mean prediction error, SEM, selected-range slopes, and an n^-1/2 reference"""
    n = np.asarray(config.n, dtype=float)
    fit_mask = n >= (config.fit_n_min or n[0])
    fit_n = n[fit_mask]
    figure, axis = new_figure(RATE_FIGSIZE)
    summary = {}
    final_errors = []
    for index, p in enumerate(config.p):
        cells = [
            [row for row in rows if row["p"] == f"{p:g}" and row["n"] == size]
            for size in config.n
        ]
        errors = [np.array([row["prediction_error"] for row in cell]) for cell in cells]
        means = np.array([values.mean() for values in errors])
        sem = [
            values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
            for values in errors
        ]
        slope, intercept = np.polyfit(np.log(fit_n), np.log(means[fit_mask]), 1)
        color = COLORS.get(p, f"C{index}")
        axis.plot(fit_n, np.exp(intercept) * fit_n**slope, "--", color=color, lw=1)
        axis.errorbar(
            n,
            means,
            yerr=sem,
            marker=MARKERS.get(p, "o"),
            color=color,
            capsize=2,
            elinewidth=0.8,
            capthick=0.8,
            label=exponent_label(p),
        )
        final_errors.append(means[-1])
        summary[f"{p:g}"] = {
            "mean_prediction_errors": means.tolist(),
            "sem": sem,
            "slope": float(slope),
            "fit_n": fit_n.astype(int).tolist(),
        }
        print(f"p={p:g}: slope={slope:.4f}", flush=True)

    # anchor the reference to the lowest empirical curve at the largest n
    reference = min(final_errors) * (n / n[-1]) ** -0.5
    axis.plot(n, reference, "--", color="black", linewidth=1.2)

    # position consistently across short and full grids on the logarithmic axis
    label_n = n[0] * (n[-1] / n[0]) ** 0.09
    label_error = min(final_errors) * (label_n / n[-1]) ** -0.5
    axis.annotate(
        r"$n^{-1/2}$",
        (label_n, label_error),
        xytext=(0, -6),
        textcoords="offset points",
        ha="center",
        va="top",
    )
    axis.set_ylabel(r"$\|X(\widehat{\boldsymbol{\beta}}-\boldsymbol{\beta}^*)\|_2^2/n$")
    style_axis(axis, log_y=True)
    axis.set_ylim(top=axis.get_ylim()[1] * 1.2)
    exponent_legend(axis).set_loc("upper right")
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "paper.pdf", bbox_inches="tight", pad_inches=0.015)

    figure.set_size_inches(RATE_FIGSIZE[0], RATE_FIGSIZE[1] + 0.95)
    axis.set_title(
        r"Prediction error: mean $\pm$ SEM"
        + "\n"
        + rf"$d=2,\ \boldsymbol{{\beta}}^*=(2,-1),\ \sigma^2={config.sigma_sq:g},\ k={config.k}$"
        + "\n"
        + rf"$1-\rho_n={config.c:g}/\sqrt{{n}},\ K={config.K:g},\ \gamma={config.gamma:g}$"
        + "\n"
        + r"$M=\sqrt{2},\ \mathrm{ground}=\ell_\infty$",
        fontsize=11,
        pad=5,
        linespacing=1.3,
    )
    figure.savefig(output / "with_config.pdf", bbox_inches="tight", pad_inches=0.015)
    plt.close(figure)
    return summary


def main(argv=None):
    """run the experiment and save its two plots and a small results file"""
    config = default_config()
    root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description="d=2 slow-rate experiment and plot")
    parser.add_argument("--n", type=int, nargs="+", default=config.n)
    parser.add_argument("--p", type=float, nargs="+", default=config.p)
    parser.add_argument("--k", type=int, default=config.k, help="repetitions")
    parser.add_argument(
        "--c",
        type=float,
        default=config.c,
        help="paper constant c0 in rho_n = 1-c0/sqrt(n)",
    )
    parser.add_argument("--sigma-sq", type=float, default=config.sigma_sq)
    parser.add_argument("--K", type=float, default=config.K)
    parser.add_argument("--gamma", type=float, default=config.gamma)
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument(
        "--fit-n-min",
        type=int,
        default=config.fit_n_min,
        help="fit slopes using only sample sizes at least this large",
    )
    parser.add_argument(
        "--output-directory", type=Path, default=root / "plots" / "slow"
    )
    args = vars(parser.parse_args(argv))
    output = args.pop("output_directory")
    vars(config).update(args)
    rows = run(config)
    summary = plot_rates(config, rows, output)
    settings = {**vars(config), "p": [f"{p:g}" for p in config.p]}
    (output / "results.json").write_text(
        json.dumps({"config": settings, "summary": summary, "fits": rows}, indent=2)
        + "\n"
    )
    print(f"saved paper.pdf, with_config.pdf, and results.json to {output}")


if __name__ == "__main__":
    main()
