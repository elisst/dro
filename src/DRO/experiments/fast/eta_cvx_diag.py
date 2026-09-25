"""
compare saved eta and saddle (CVX) fits for the fast-rate experiment

check that the fits use the same datasets, then summarize prediction error,
runtime, and robust-risk agreement without rerunning either solver
write four PDF figures and summary.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from DRO.experiments.fast.plot import (
    COLORS,
    MARKERS,
    exponent_label,
    exponent_legend,
    load_results,
    new_figure,
    plt,
    style_axis,
)
from DRO.experiments.fast.rate import default_config, summarize
from DRO.experiments.fast.readwrite import data_fingerprint, default_directory


def load_pair(cvx_directory, eta_directory):
    """load paired fits with the same settings, grid, and dataset seeds"""
    config, cvx = load_results(cvx_directory)
    eta_config, eta = load_results(eta_directory)
    if (
        getattr(config, "solver", "cvx") != "cvx"
        or getattr(eta_config, "solver", None) != "eta"
    ):
        raise ValueError("expected a CVX folder and an eta folder")
    if data_fingerprint(config) != data_fingerprint(eta_config):
        raise ValueError("data or radius settings differ")
    for key in ("n", "p", "k", "fit_min_n"):
        if getattr(config, key) != getattr(eta_config, key):
            raise ValueError(f"experiment settings differ: {key}")
    # pair repetitions by their saved seeds before comparing the solvers
    for p in config.p:
        for cvx_cell, eta_cell in zip(cvx[p], eta[p]):
            for cvx_fit, eta_fit in zip(cvx_cell, eta_cell):
                if cvx_fit["seed"] != eta_fit["seed"]:
                    raise ValueError(f"paired seed mismatch for p={p:g}")
    return config, cvx, eta


def values(results, p, key):
    """extract a saved quantity: rows are sample sizes, columns are repetitions"""
    return np.array([[fit[key] for fit in cell] for cell in results[p]])


def comparison_summary(config, cvx, eta):
    """summarize timing totals, risk differences, prediction slopes, and fit status"""
    cvx_summary = summarize(config, cvx)
    eta_summary = summarize(config, eta)
    summary = {}
    for p in config.p:
        # speedup is the ratio of total times, not the mean of individual ratios
        cvx_seconds = float(values(cvx, p, "fit_seconds").sum())
        eta_seconds = float(values(eta, p, "fit_seconds").sum())
        cvx_risk = values(cvx, p, "robust_risk")
        eta_risk = values(eta, p, "robust_risk")
        risk_difference = np.abs(eta_risk - cvx_risk)
        relative_difference = None
        if np.all(cvx_risk > 0):
            relative_difference = float(np.max(risk_difference / cvx_risk))
        summary[f"{p:g}"] = {
            "cvx_seconds": cvx_seconds,
            "eta_seconds": eta_seconds,
            "speedup": cvx_seconds / eta_seconds,
            "max_abs_risk_difference": float(risk_difference.max()),
            "max_abs_relative_risk_difference": relative_difference,
            "cvx_slope": cvx_summary[f"{p:g}"]["slope"],
            "eta_slope": eta_summary[f"{p:g}"]["slope"],
            "cvx_statuses": cvx_summary[f"{p:g}"]["solver_status_counts"],
            "eta_statuses": eta_summary[f"{p:g}"]["solver_status_counts"],
        }
    return summary


def save_plot(figure, axis, ylabel, path, *, compare_solvers=False):
    """apply the shared figure style and save a PDF"""
    axis.set_ylabel(ylabel)
    style_axis(axis, log_y=True)
    if compare_solvers:
        # one legend shows both the exponent colours and the solver line styles
        handles, _ = axis.get_legend_handles_labels()
        handles.extend([
            Line2D([], [], color=".3", linestyle="-", label="Saddle"),
            Line2D([], [], color=".3", linestyle="--", label="Eta"),
        ])
        axis.legend(
            handles=handles, ncol=2, loc="best", framealpha=0.95,
            fontsize=8.5, handlelength=1.2, columnspacing=0.8,
            labelspacing=0.2, borderpad=0.25,
        )
    else:
        exponent_legend(axis)
    figure.savefig(path, bbox_inches="tight", pad_inches=0.015)
    plt.close(figure)


def plot_overlay(config, cvx, eta, key, ylabel, path):
    """plot means over repetitions; colour gives p, line style gives the solver"""
    figure, axis = new_figure()
    for index, p in enumerate(config.p):
        for results, linestyle in ((cvx, "-"), (eta, "--")):
            observations = values(results, p, key)
            errors = None
            # prediction error bars show one standard error of the mean
            if key == "prediction_error" and config.k > 1:
                errors = observations.std(axis=1, ddof=1) / np.sqrt(config.k)
            axis.errorbar(
                config.n, observations.mean(axis=1), yerr=errors,
                color=COLORS.get(p, f"C{index}"), marker=MARKERS.get(p, "o"),
                linestyle=linestyle, capsize=2,
                elinewidth=0.8, capthick=0.8,
                label=exponent_label(p) if results is cvx else None,
            )
    save_plot(figure, axis, ylabel, path, compare_solvers=True)


def plot_speedup(config, cvx, eta, path):
    """plot total saddle time / total eta time at each n and p"""
    figure, axis = new_figure()
    for index, p in enumerate(config.p):
        cvx_times = values(cvx, p, "fit_seconds").sum(axis=1)
        eta_times = values(eta, p, "fit_seconds").sum(axis=1)
        axis.plot(config.n, cvx_times / eta_times,
                  color=COLORS.get(p, f"C{index}"), marker=MARKERS.get(p, "o"),
                  label=exponent_label(p))
    axis.axhline(1, color="black", linestyle="--", linewidth=1)
    save_plot(figure, axis,
              r"$\frac{\mathrm{Saddle\ time}\ (s)}{\mathrm{Eta\ time}\ (s)}$", path)


def plot_risk_difference(config, cvx, eta, path):
    """plot median |V_delta(beta_eta) - V_delta(beta_cvx)| and its 10--90% band"""
    figure, axis = new_figure()
    for index, p in enumerate(config.p):
        # take absolute differences for each paired repetition first
        difference = np.abs(
            values(eta, p, "robust_risk") - values(cvx, p, "robust_risk")
        )
        color = COLORS.get(p, f"C{index}")
        lower, median, upper = np.percentile(difference, (10, 50, 90), axis=1)
        # omit exact zeros on log axes rather than adding an arbitrary floor
        axis.plot(config.n, np.ma.masked_equal(median, 0),
                  color=color, marker=MARKERS.get(p, "o"),
                  label=exponent_label(p))
        axis.fill_between(config.n, lower, upper, where=lower > 0,
                          color=color, alpha=0.13, linewidth=0)
    save_plot(figure, axis,
              r"$|V_\delta^{\mathrm{eta}}-V_\delta^{\mathrm{saddle}}|$",
              path)


def plot_comparison(config, cvx, eta, output):
    """write prediction error, runtime, speedup, and risk-agreement figures"""
    output.mkdir(parents=True, exist_ok=True)
    plot_overlay(config, cvx, eta, "prediction_error",
                 r"$\|X(\widehat{\boldsymbol{\beta}}-\boldsymbol{\beta}^*)\|_2^2/n$",
                 output / "prediction_overlay.pdf")
    plot_overlay(config, cvx, eta, "fit_seconds", r"Mean fit time ($s$)",
                 output / "runtime.pdf")
    plot_speedup(config, cvx, eta, output / "speedup.pdf")
    plot_risk_difference(config, cvx, eta, output / "risk_difference.pdf")


def main(argv=None):
    """choose the saved fit folders and write the comparison"""
    config = default_config()
    root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cvx-directory", type=Path, default=default_directory(config, "cvx"))
    parser.add_argument("--eta-directory", type=Path, default=default_directory(config, "eta"))
    parser.add_argument("--output-directory", type=Path, default=root / "plots/eta_cvx")
    args = parser.parse_args(argv)

    config, cvx, eta = load_pair(args.cvx_directory, args.eta_directory)
    summary = comparison_summary(config, cvx, eta)
    plot_comparison(config, cvx, eta, args.output_directory)
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    cvx_seconds = sum(row["cvx_seconds"] for row in summary.values())
    eta_seconds = sum(row["eta_seconds"] for row in summary.values())
    print(f"CVX: {cvx_seconds:.2f}s; eta: {eta_seconds:.2f}s; speedup: {cvx_seconds / eta_seconds:.2f}x")
    print("saved prediction_overlay.pdf, runtime.pdf, speedup.pdf, "
          "risk_difference.pdf, and summary.json")


if __name__ == "__main__":
    main()
