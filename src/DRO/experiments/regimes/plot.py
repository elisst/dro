"""
plot the small- and large-delta sweeps separately and together
"""

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np

from DRO.experiments.fast.plot import (
    COLORS,
    MARKERS,
    exponent_label,
    plt,
    style_axis,
)

SINGLE_FIGSIZE = (3.5, 1.8)
COMBINED_FIGSIZE = (3.5, 3.25)


def plot_panel(axis, config, result, regime, show_exponents=True):
    """show the raw solver measurements on a common delta axis"""
    quantity = "training_mse" if regime == "S" else "beta_inf"
    for index, p in enumerate(config.p):
        rows = [
            row
            for row in result["fits"]
            if row["regime"] == regime and float(row["p"]) == p
        ]
        delta = [row["delta"] for row in rows]
        values = np.array([row[quantity] for row in rows])
        if regime == "S":
            # exact zeros cannot be shown on a log axis. do not add a positive floor
            values = np.where(values > 0, values, np.nan)
        axis.plot(
            delta,
            values,
            color=COLORS.get(p, f"C{index}"),
            marker=MARKERS.get(p, "o"),
            markevery=5,
            label=exponent_label(p),
        )
    style_axis(axis, log_y=regime == "S")
    axis.set_xlabel(r"$\delta$")
    if regime == "L":
        axis.set_xscale("linear")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.set_ylim(bottom=0)

    if regime == "S":
        axis.set_ylabel(r"$\|X\widehat{\boldsymbol{\beta}}-\boldsymbol{y}\|_2^2/n$")
        axis.legend(
            ncol=2,
            loc="upper left",
            framealpha=0.95,
            handlelength=0.8,
            handletextpad=0.35,
            columnspacing=0.4,
            borderpad=0.3,
            labelspacing=0.25,
        )
    else:
        axis.set_ylabel(r"$\|\widehat{\boldsymbol{\beta}}\|_\infty$")

    # p>2 shares one small cutoff while each p has its own large cutoff
    drawn_small = set()
    for index, row in enumerate(result["thresholds"]):
        p = float(row["p"])
        color = COLORS.get(p, f"C{index}")
        if regime == "S":
            cutoff = row["delta_S"]
            if cutoff in drawn_small:
                continue
            axis.axvline(
                cutoff, color=color if p == 2 else "0.35", ls="--", lw=1, zorder=1
            )
            label = r"$\delta_S(p = 2)$" if p == 2 else r"$\delta_S(p > 2)$"
            axis.annotate(
                label,
                (cutoff, 0.62),
                xycoords=("data", "axes fraction"),
                xytext=(-4, 0),
                textcoords="offset points",
                rotation=90,
                ha="right",
                va="center",
                fontsize=10,
            )
            drawn_small.add(cutoff)
        else:
            axis.axvline(row["delta_L"], color=color, ls="--", lw=1, zorder=1)

    if regime == "L":
        handles = axis.get_legend_handles_labels()[0] if show_exponents else []
        handles.append(Line2D([], [], color="0.35", ls="--", label=r"$\delta_L(p)$"))
        axis.legend(
            handles=handles,
            ncol=2 if show_exponents else 1,
            loc="upper right",
            framealpha=0.95,
            handlelength=1.2,
            handletextpad=0.35,
            columnspacing=0.65,
            borderpad=0.35,
            labelspacing=0.25,
        )


def save_figure(figure, config, output):
    """save the paper copy and a copy with the configuration above it"""
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "paper.pdf", bbox_inches="tight", pad_inches=0.015)
    width, height = figure.get_size_inches()
    figure.set_size_inches(width, height + 0.65)
    ground = r"\infty" if np.isinf(config.ground_norm) else f"{config.ground_norm:g}"
    figure.suptitle(
        rf"$n={config.n},\ d={config.d},\ \sigma^2={config.sigma_sq:g},\ \mathrm{{ground}}=\ell_{{{ground}}}$"
        + "\n"
        + rf"$\beta_j^*\overset{{\mathrm{{iid}}}}{{\sim}}\mathcal{{N}}(0,1),\ \mathrm{{seeds}}=({config.seed},{config.signal_seed})$",
        fontsize=11,
    )
    figure.savefig(output / "with_config.pdf", bbox_inches="tight", pad_inches=0.015)
    plt.close(figure)


def plot_regimes(config, result, output):
    """write six PDFs: small, large, and combined, each with and without settings"""
    for regime, name in (("S", "small"), ("L", "large")):
        figure, axis = plt.subplots(figsize=SINGLE_FIGSIZE, layout="constrained")
        figure.get_layout_engine().set(w_pad=0.02, h_pad=0.01)
        plot_panel(axis, config, result, regime)
        save_figure(figure, config, output / name)

    figure, (small, large) = plt.subplots(
        2, 1, figsize=COMBINED_FIGSIZE, layout="constrained"
    )
    figure.get_layout_engine().set(w_pad=0.02, h_pad=0.01, hspace=0)
    plot_panel(small, config, result, "S")
    plot_panel(large, config, result, "L", show_exponents=False)
    small.set_xlabel("")
    save_figure(figure, config, output / "both")
