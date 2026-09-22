"""
compact paper figures from the saved fast-rate experiment

read config.json and HDF5 fits from data/fast_<config hash>
write paper and titled PDFs for each quantity under plots/fast
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import (
    LogFormatterSciNotation,
    LogLocator,
    MaxNLocator,
    NullFormatter,
)
import numpy as np

from DRO.experiments.fast.rate import KAPPA, compute_bounds, default_config
from DRO.experiments.fast.readwrite import config_fingerprint, load_run, run_metadata

# sizes are in inches and points, intended for one paper column
FIGSIZE = (3.5, 2.35)
COLORS = {2: "#CC79A7", 3: "#D55E00", 6: "#F0E442", np.inf: "#009E73"}
MARKERS = {2: "o", 3: "s", 6: "^", np.inf: "D"}
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.usetex": True,
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{amsmath}",
        "font.size": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.linewidth": 0.7,
        "axes.labelpad": 2,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.5,
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def exponent_label(p):
    return r"$p=\infty$" if np.isinf(p) else rf"$p={p:g}$"


def new_figure():
    """keep dimensions and margins consistent across the figures"""
    figure, axis = plt.subplots(figsize=FIGSIZE, layout="constrained")
    figure.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0, hspace=0)
    return figure, axis


def style_axis(axis, log_y=False):
    """use sparse log ticks and a light grid without rotated labels"""
    axis.set_xscale("log")
    axis.xaxis.set_major_locator(LogLocator(base=10))
    axis.xaxis.set_major_formatter(LogFormatterSciNotation())
    axis.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
    axis.xaxis.set_minor_formatter(NullFormatter())
    if log_y:
        axis.set_yscale("log")
        lower, upper = axis.get_ylim()
        decades = np.log10(upper / lower)
        if decades < 1.5:
            axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            axis.yaxis.set_major_formatter(
                LogFormatterSciNotation(minor_thresholds=(np.inf, np.inf))
            )
        else:
            axis.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
            axis.yaxis.set_major_formatter(LogFormatterSciNotation())
        axis.yaxis.set_minor_formatter(NullFormatter())
    else:
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=(1, 2, 2.5, 5, 10)))
    axis.set_xlabel(r"$n$")
    axis.margins(x=0.035)
    axis.grid(True, which="major", linewidth=0.45, alpha=0.22)
    axis.set_axisbelow(True)
    axis.tick_params(which="major", width=0.7, length=3)
    axis.tick_params(which="minor", width=0.5, length=1.8)
    axis.spines[["top", "right"]].set_visible(True)


def exponent_legend(axis, key=None):
    """keep the boxed legend inside the axes, clear of the curves"""
    options = {"loc": "best"}
    if key == "epsilon_term":
        options = {"loc": "center right", "bbox_to_anchor": (0.99, 0.70)}
    elif key == "radius_ratio":
        options = {"loc": "upper left"}
    return axis.legend(
        ncol=2,
        frameon=True,
        framealpha=0.95,
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=0.65,
        borderpad=0.35,
        labelspacing=0.25,
        **options,
    )


def config_title(config):
    """show the data, radius, and repetition settings on the titled copy"""
    ground = r"\infty" if np.isinf(config.ground_norm) else f"{config.ground_norm:g}"
    return (
        rf"$d={config.d},\ s={config.s},\ B={config.B:g},\ \sigma^2={config.sigma_sq:g},\ k={config.k}$"
        + "\n"
        + rf"$K={config.K:g},\ M={config.M:g},\ \gamma={config.gamma:g},\ \mathrm{{ground}}=\ell_{{{ground}}}$"
        + "\n"
        + rf"$n\in[{min(config.n)},\,{max(config.n)}]$ ({len(config.n)} sizes)"
    )


def save_figure(figure, axis, output, name, config, title):
    """save one paper copy and one copy with the configuration in its title"""
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / "paper.pdf", bbox_inches="tight", pad_inches=0.015)

    # extra height is only for the title; the paper copy stays compact
    figure.set_size_inches(FIGSIZE[0], FIGSIZE[1] + 0.95)
    axis.set_title(
        title + "\n" + config_title(config), fontsize=11, pad=5, linespacing=1.3
    )
    figure.savefig(directory / "with_config.pdf", bbox_inches="tight", pad_inches=0.015)
    plt.close(figure)


def plot_quantity(axis, config, results, key, *, full_range=False, dashed=False):
    """plot medians and empirical bands across repetitions"""
    quantiles = (0, 50, 100) if full_range else (10, 50, 90)
    for index, (p, cells) in enumerate(results.items()):
        if key == "radius" and index > 0:
            continue
        color = "black" if key == "radius" else COLORS.get(p, f"C{index}")
        intervals = [
            np.percentile([row[key] for row in cell], quantiles) for cell in cells
        ]
        lower, median, upper = np.asarray(intervals).T
        axis.plot(
            config.n,
            median,
            linestyle="--" if dashed else "-",
            marker=(
                None if dashed else ("o" if key == "radius" else MARKERS.get(p, "o"))
            ),
            color=color,
            label=None if dashed else exponent_label(p),
        )
        if not dashed:
            axis.fill_between(
                config.n, lower, upper, color=color, alpha=0.13, linewidth=0
            )


def plot_prediction_rates(config, results, output):
    """plot mean prediction error, SEM, fitted slopes, and an n^-1 reference"""
    n = np.asarray(config.n, dtype=float)
    tail = n >= config.fit_min_n
    figure, axis = new_figure()
    final_errors = []
    for index, (p, cells) in enumerate(results.items()):
        errors = [np.array([row["prediction_error"] for row in cell]) for cell in cells]
        means = np.array([values.mean() for values in errors])
        sem = [
            values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
            for values in errors
        ]
        color = COLORS.get(p, f"C{index}")
        if tail.sum() >= 2 and np.all(means[tail] > 0):
            slope, intercept = np.polyfit(np.log(n[tail]), np.log(means[tail]), 1)
            axis.plot(
                n[tail], np.exp(intercept) * n[tail] ** slope, "--", color=color, lw=1
            )
        axis.errorbar(
            n,
            means,
            yerr=sem,
            marker=MARKERS.get(p, "o"),
            capsize=2,
            elinewidth=0.8,
            capthick=0.8,
            color=color,
            label=exponent_label(p),
        )
        final_errors.append(means[-1])

    reference = np.mean(final_errors) * (n / n[-1]) ** -1
    axis.plot(n, reference, "--", color="black", linewidth=1.2)
    axis.text(n[0] * 1.10, reference[0] * 0.5, r"$n^{-1}$", ha="left", va="top")
    axis.set_ylabel(r"$\|X(\widehat{\boldsymbol{\beta}}-\boldsymbol{\beta}^*)\|_2^2/n$")
    style_axis(axis, log_y=True)
    exponent_legend(axis)
    title = rf"Prediction error: mean $\pm$ SEM; fit $n\geq {config.fit_min_n}$"
    save_figure(figure, axis, output, "rate", config, title)


def plot_diagnostic(config, results, output, filename, key, label, log_y, title):
    """plot one quantity and save paper and configuration copies"""
    figure, axis = new_figure()
    full_range = key in ("radius_ratio", "kappa_certificate")
    plot_quantity(axis, config, results, key, full_range=full_range)
    if key == "fast_rate_bound":
        plot_quantity(axis, config, results, "prediction_error", dashed=True)
    if key == "radius_ratio":
        axis.axhline(1, color="black", linestyle="--", linewidth=1)
    if key == "kappa_certificate":
        limit = KAPPA
        axis.axhline(limit, color="black", linestyle="--", linewidth=1)
        axis.annotate(
            r"$\kappa=1/\sqrt{6}$",
            (config.n[0], limit),
            xytext=(4, 6),
            va="bottom",
            textcoords="offset points",
            fontsize=11,
        )
    axis.set_ylabel(label)
    style_axis(axis, log_y)
    if key == "radius_ratio":
        lower, upper = axis.get_ylim()
        ticks = [tick for tick in axis.get_yticks() if lower <= tick <= upper]
        axis.set_yticks(sorted(set(ticks + [1])))
    if key != "radius":
        if key == "fast_rate_bound":
            styles = axis.legend(
                handles=[
                    Line2D([], [], color=".3", label="Bound"),
                    Line2D([], [], color=".3", linestyle="--", label="Error"),
                ],
                loc="center left",
                frameon=True,
                framealpha=0.95,
                handlelength=1.7,
                borderpad=0.3,
            )
            axis.add_artist(styles)
        exponent_legend(axis, key)
    if key != "radius":
        interval = "range" if full_range else r"10--90\%"
        title += f"\nmedian; shading: {interval}"
    save_figure(figure, axis, output, filename, config, title)


def plot_results(config, results, output):
    """write two PDFs per quantity in separate subfolders"""
    output.mkdir(parents=True, exist_ok=True)
    plot_prediction_rates(config, results, output)

    # folder, saved quantity, axis label, log scale, titled-copy heading
    plots = [
        ("eq", "epsilon_term", r"$e_q$", False, r"$e_q=\|\varepsilon\|_q/n^{1/q}$"),
        ("delta", "radius", r"$\delta$", False, r"$\delta=KM\sqrt{\log(d/\gamma)/n}$"),
        (
            "beta_hat_l1",
            "beta_hat_l1",
            r"$\|\widehat{\boldsymbol{\beta}}\|_1$",
            False,
            r"Fitted coefficient norm $\|\widehat{\boldsymbol{\beta}}\|_1$",
        ),
        (
            "radius_ratio",
            "radius_ratio",
            r"$\delta/\bar\delta$",
            False,
            r"Radius assumption: $\delta/\bar\delta>1$",
        ),
        ("C1", "C1", r"$C_1$", False, r"$C_1=3+\delta\|\boldsymbol{\beta}^*\|_1/e_q$"),
        (
            "C2",
            "C2",
            r"$C_2$",
            False,
            r"$C_2=3e_q+(2MC_1+\delta C_1+\delta)\|\boldsymbol{\beta}^*\|_1$",
        ),
        ("C3", "C3", r"$C_3$", True, r"$C_3=4C_1^2\|\boldsymbol{\beta}^*\|_1^2$"),
        (
            "C4",
            "C4",
            r"$C_4$",
            True,
            r"$C_4=16C_2^2s^2/\kappa^2,\quad\kappa=1/\sqrt{6}$",
        ),
        ("max", "maximum_component", r"$\max\{C_3,C_4\}$", True, r"$\max\{C_3,C_4\}$"),
        (
            "bound",
            "fast_rate_bound",
            r"$R_{\mathrm{fast}},\ E_n$",
            True,
            r"$R_{\mathrm{fast}}=\delta^2\max\{C_3,C_4\}$; dashed: prediction error",
        ),
        (
            "re_certificate",
            "kappa_certificate",
            r"$\sqrt{\max\{\lambda_{\min}(G),0\}}$",
            False,
            r"RE certificate: $G=X^\top X/n$",
        ),
    ]
    for filename, key, label, log_y, title in plots:
        plot_diagnostic(config, results, output, filename, key, label, log_y, title)


def load_results(directory):
    """read the saved grid and recompute its bounds without running the optimizer"""
    settings = json.loads((directory / "config.json").read_text())["config"]
    config = argparse.Namespace(**settings)
    config.p = [float(p) for p in config.p]
    config.ground_norm = float(config.ground_norm)

    results = {}
    for p in config.p:
        results[p] = []
        for n in config.n:
            rows = []
            for repetition in range(1, config.k + 1):
                path = directory / f"n={n}_p={p:g}_rep={repetition}.h5"
                row = load_run(path, run_metadata(config, n, p, repetition))
                rows.append(compute_bounds(row, config, n, p))
            results[p].append(rows)
    return config, results


def main(argv=None):
    """choose the saved experiment and output folder"""
    root = Path(__file__).resolve().parents[4]
    directory = root / "data" / f"fast_{config_fingerprint(default_config())}"
    parser = argparse.ArgumentParser(description="plot saved fast-rate fits")
    parser.add_argument(
        "--directory", type=Path, default=directory, help="folder of saved fits"
    )
    parser.add_argument(
        "--output-directory", type=Path, default=root / "plots" / "fast"
    )
    args = parser.parse_args(argv)
    config, results = load_results(args.directory)
    plot_results(config, results, args.output_directory)
    print(f"saved 24 PDFs (paper and with_config) to {args.output_directory}")


if __name__ == "__main__":
    main()
