"""
small- and large-delta regimes on one fixed overparametrized dataset

compute the theorem thresholds with CVXPY, then fit around each threshold
save the data, dual solutions, and fits so the figure can be redrawn without fitting

assume n < d, sigma_sq >= 0, p >= 2, ground_norm >= 1, and a nonzero response

see README.md for commands
"""

import argparse
import json
from pathlib import Path

import cvxpy as cp
import numpy as np

from DRO.cvx_solver import CvxOptimizer
from DRO.robust_risk import RobustRisk
from DRO.experiments.regimes.plot import plot_regimes


def default_config():
    """Gaussian design and a dense standard normal signal"""
    return argparse.Namespace(
        n=30,
        d=60,
        sigma_sq=0.25,
        seed=7,
        signal_seed=8,
        p=(2.0, 3.0, 6.0, np.inf),
        ground_norm=np.inf,
    )


def interpolation_alphas(X, y, ground_norm=np.inf):
    """compute the two minimum-norm dual solutions for the small thresholds"""

    # first solve max alpha.T y subject to ||X.T alpha||_ground <= 1
    alpha = cp.Variable(len(y))
    dual_constraint = cp.norm(X.T @ alpha, ground_norm) <= 1
    dual = cp.Problem(cp.Maximize(y @ alpha), [dual_constraint])
    dual.solve(solver="CLARABEL")
    if dual.status != cp.OPTIMAL:
        raise RuntimeError(f"Interpolation dual solve: {dual.status}")
    alpha_star = float(dual.value)

    # for 1 < ground_norm < infinity, the dual solution is unique (remark 4)
    if 1 < ground_norm < np.inf:
        return alpha.value.copy(), alpha.value.copy()

    # then minimize ||alpha||_infinity on that optimal face, as in remark 4
    face = cp.Problem(
        cp.Minimize(cp.norm_inf(alpha)), [dual_constraint, y @ alpha == alpha_star]
    )
    face.solve(solver="CLARABEL")
    if face.status != cp.OPTIMAL:
        raise RuntimeError(f"Minimum infinity-norm dual solve: {face.status}")
    alpha_inf = alpha.value.copy()

    # p=2 needs the minimum euclidean norm on the same optimal face
    # a separate solve is needed since the two minimizing dual vectors may differ
    face_l2 = cp.Problem(
        cp.Minimize(cp.norm(alpha, 2)), [dual_constraint, y @ alpha == alpha_star]
    )
    face_l2.solve(solver="CLARABEL")
    if face_l2.status != cp.OPTIMAL:
        raise RuntimeError(f"Minimum euclidean-norm dual solve: {face_l2.status}")
    alpha_l2 = alpha.value.copy()

    return alpha_inf, alpha_l2


def theorem_thresholds(X, y, alpha_inf, alpha_l2, p, ground_norm=np.inf):
    """return delta_S and delta_L"""
    n = len(y)
    q = 1.0 if np.isinf(p) else p / (p - 1)
    if p == 2:
        delta_small = 1 / (np.sqrt(n) * np.linalg.norm(alpha_l2, 2))
    else:
        delta_small = 1 / (n * np.linalg.norm(alpha_inf, np.inf))
    delta_large = np.linalg.norm(X.T @ y, ground_norm) / (
        n ** (1 / p) * np.linalg.norm(y, q)
    )
    return float(delta_small), float(delta_large)


def delta_grid(thresholds, regime):
    """use one absolute delta grid per panel, dense around every cutoff"""
    cutoffs = np.unique([row[f"delta_{regime}"] for row in thresholds])
    if regime == "S":
        broad = np.geomspace(0.1 * cutoffs.min(), 2 * cutoffs.max(), 20)
    else:
        broad = np.linspace(0.5 * cutoffs.min(), 1.3 * cutoffs.max(), 21)
    nearby = [cutoff * np.linspace(0.9, 1.1, 9) for cutoff in cutoffs]
    return np.unique(np.concatenate([broad, *nearby]))


def fit_one(X, y, p, delta, ground_norm=np.inf):
    """solve without using the thresholds and retain the raw measurements"""
    risk = RobustRisk(X, y, delta, p, norm=ground_norm)
    try:
        fit = CvxOptimizer(risk).minimize()
    except cp.error.SolverError:
        # retry numerical failures without Clarabel's automatic equilibration
        fit = CvxOptimizer(risk).minimize(equilibrate_enable=False)

    return {
        "p": f"{p:g}",
        "delta": float(delta),
        "beta_hat": fit.beta.tolist(),
        "training_mse": float(np.mean((X @ fit.beta - y) ** 2)),
        "beta_inf": float(np.linalg.norm(fit.beta, np.inf)),
        "risk": fit.value,
        **fit.diagnostics,
    }


def run(config, output, data_directory):
    """draw one dataset, compute the thresholds, sweep delta, and save the results"""

    # draw data
    rng = np.random.default_rng(config.seed)
    X = rng.standard_normal((config.n, config.d))
    # since d > n and Gaussian i.i.d. X should be full row rank a.s. but check we anyways
    assert np.linalg.matrix_rank(X) == config.n, "X must have full row rank"
    noise = np.sqrt(config.sigma_sq) * rng.standard_normal(config.n)
    signal_rng = np.random.default_rng(config.signal_seed)
    beta_star = signal_rng.standard_normal(config.d)
    y = X @ beta_star + noise

    # compute alphas
    alpha_inf, alpha_l2 = interpolation_alphas(X, y, config.ground_norm)

    # compute delta thresholds
    thresholds = []
    for p in config.p:
        small, large = theorem_thresholds(
            X, y, alpha_inf, alpha_l2, p, config.ground_norm
        )
        thresholds.append({"p": f"{p:g}", "delta_S": small, "delta_L": large})
        print(f"p={p:g}: delta_S={small:.8g}, delta_L={large:.8g}", flush=True)

    settings = {
        **vars(config),
        "p": [f"{p:g}" for p in config.p],
        "ground_norm": f"{config.ground_norm:g}",
    }
    result = {
        "config": settings,
        "thresholds": thresholds,
        "fits": [],
    }

    # fit risk over range of deltas for both S and L regime
    for regime in ("S", "L"):
        grid = delta_grid(thresholds, regime)
        for p in config.p:
            for delta in grid:
                row = fit_one(X, y, p, delta, config.ground_norm)
                result["fits"].append({"regime": regime, **row})
            print(f"p={p:g}, regime {regime}: {len(grid)} fits", flush=True)

    # save the data and completed sweep for plotting without refitting
    output.mkdir(parents=True, exist_ok=True)
    data_directory.mkdir(parents=True, exist_ok=True)
    np.savez(
        data_directory / "data.npz",
        X=X,
        y=y,
        beta_star=beta_star,
        alpha_inf=alpha_inf,
        alpha_l2=alpha_l2,
    )
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    """run and plot, or redraw a completed experiment from saved fits"""
    config = default_config()
    root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description="small- and large-delta regimes")
    parser.add_argument("--n", type=int, default=config.n)
    parser.add_argument("--d", type=int, default=config.d)
    parser.add_argument("--p", type=float, nargs="+", default=config.p)
    parser.add_argument("--ground-norm", type=float, default=config.ground_norm)
    parser.add_argument("--sigma-sq", type=float, default=config.sigma_sq)
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--signal-seed", type=int, default=config.signal_seed)
    parser.add_argument(
        "--output-directory", type=Path, default=root / "plots" / "regimes"
    )
    parser.add_argument(
        "--data-directory", type=Path, default=root / "data" / "regimes"
    )
    parser.add_argument("--plot-only", action="store_true")
    args = vars(parser.parse_args(argv))
    output = args.pop("output_directory")
    data_directory = args.pop("data_directory")
    plot_only = args.pop("plot_only")
    vars(config).update(args)
    if plot_only:
        result = json.loads((output / "results.json").read_text())
        config = default_config()
        vars(config).update(result["config"])
        config.p = [float(p) for p in config.p]
        config.ground_norm = float(config.ground_norm)
    else:
        result = run(config, output, data_directory)
    plot_regimes(config, result, output)
    print(f"saved six PDFs (small, large, and combined) to {output}")


if __name__ == "__main__":
    main()
