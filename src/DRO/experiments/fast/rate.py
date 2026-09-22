"""
fast-rate experiment: sparse linear regression and the bound in theorem 3

loop over n, p, and repetitions, reusing saved fits when available
save each fit, config.json, and summary.json in the same experiment folder

see README.md for commands
"""

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from DRO.cvx_solver import CvxOptimizer
from DRO.robust_risk import RobustRisk

from DRO.experiments.fast.readwrite import (
    json_value,
    config_fingerprint,
    run_metadata,
    save_run,
    load_run,
)


# Fixed RE constant used in the fast-rate theorem (also when M > 1).
KAPPA = 1.0 / np.sqrt(6.0)


def default_config():
    """
    experiment settings. sample sizes and repetitions can be extended later

    assume positive n, k (nr repeats), B, sigma_sq, and K

    assume 2 <= p <= infinity, 0 < gamma < 1, and M >= 1
    """

    # sample sizes n, exponents p, and repetitions k
    n = (2560, 5120, 10240, 20480, 40960, 61440, 81920, 122880, 163840)
    p = (2.0, 3.0, 6.0, np.inf)
    k = 10  # nr repeats
    fit_min_n = 40960  # fit slopes after this n to avoid small n influence

    # data and sparse signal
    d = 10
    sigma_sq = 0.25  # noise variance
    s = 5
    B = 15.0  # B = ||beta_star||_1 -- each active entry is B/s

    # delta = K * M * sqrt(log(d / gamma) / n)
    gamma = 0.01
    K = 3.6
    M = 1.0
    ground_norm = np.inf

    return argparse.Namespace(
        n=n,
        p=p,
        k=k,
        fit_min_n=fit_min_n,
        d=d,
        sigma_sq=sigma_sq,
        s=s,
        B=B,
        gamma=gamma,
        K=K,
        M=M,
        ground_norm=ground_norm,
    )


def run_one(config, n, p, seed):
    """draw data, fit robust regression, and compute the fast-rate quantities"""

    # define the sparse signal
    beta_star = np.zeros(config.d)
    beta_star[: config.s] = config.B / config.s

    # draw data (keep this order so saved seeds reproduce the same datasets)
    rng = np.random.default_rng(seed)
    X = rng.uniform(-config.M, config.M, size=(n, config.d))
    noise = np.sqrt(config.sigma_sq) * rng.standard_normal(n)
    y = X @ beta_star + noise

    # choose the Wasserstein radius
    delta = config.K * config.M * np.sqrt(np.log(config.d / config.gamma) / n)

    # fit robust regression
    risk = RobustRisk(X, y, delta, p, norm=config.ground_norm)
    started = perf_counter()
    fit = CvxOptimizer(risk).minimize()
    fit_seconds = perf_counter() - started
    beta_hat = fit.beta

    # compute quantities needed for the theorem
    q = 1.0 if np.isinf(p) else p / (p - 1)
    prediction_error = float(np.mean((X @ (beta_hat - beta_star)) ** 2))
    epsilon_q_norm = float(np.linalg.norm(noise, q))
    x_transpose_noise_inf = float(np.linalg.norm(X.T @ noise, np.inf))

    # the smallest Gram eigenvalue certifies the RE inequality for every vector
    kappa_certificate = 0.0 if n < config.d else np.sqrt(
        max(np.linalg.eigvalsh(X.T @ X / n).min(), 0)
    )

    # retain the fit and theorem ingredients so bounds can be recomputed later
    row = {
        "beta_hat": beta_hat,
        "beta_star": beta_star,
        "beta_hat_l1": float(np.linalg.norm(beta_hat, 1)),
        "robust_risk": fit.value,
        "fit_seconds": fit_seconds,
        "prediction_error": prediction_error,
        "epsilon_q_norm": epsilon_q_norm,
        "x_transpose_noise_inf": x_transpose_noise_inf,
        "kappa_certificate": kappa_certificate,
        "seed": seed,
        **{f"solver_{key}": value for key, value in fit.diagnostics.items()},
    }
    return compute_bounds(row, config, n, p)


def compute_bounds(row, config, n, p):
    """compute theorem 3 from the saved ingredients"""

    # theorem 3 quantities
    q = 1.0 if np.isinf(p) else p / (p - 1)
    delta = config.K * config.M * np.sqrt(np.log(config.d / config.gamma) / n)
    beta_star_l1 = float(np.linalg.norm(row["beta_star"], 1))
    e_q = float(row["epsilon_q_norm"]) / n ** (1 / q)
    delta_bar = (
        2
        * float(row["x_transpose_noise_inf"])
        / (n ** (1 / p) * float(row["epsilon_q_norm"]))
    )

    # theorem 3 constants
    M = config.M
    s = config.s
    kappa = KAPPA
    C1 = 3.0 + delta * beta_star_l1 / e_q
    C2 = 3.0 * e_q + (2.0 * M * C1 + delta * C1 + delta) * beta_star_l1

    C3 = 4.0 * C1**2 * beta_star_l1**2
    C4 = 16.0 * C2**2 * s**2 / kappa**2

    # fast bound = delta^2 max{C3, C4}
    maximum_component = max(C3, C4)
    bound = delta**2 * maximum_component

    # Accept saved fits with the earlier constant names when refreshing quantities.
    row = {
        key: value
        for key, value in row.items()
        if key not in {"C", "H", "norm_component", "re_component"}
    }

    # return the fit together with the theorem quantities
    return {
        **row,
        "sample_size": n,
        "wasserstein_exponent": p,
        "epsilon_term": e_q,
        "beta_star_l1": beta_star_l1,
        "radius": delta,
        "delta_bar": delta_bar,
        "radius_ratio": delta / delta_bar,
        "kappa": kappa,
        "C1": C1,
        "C2": C2,
        "C3": C3,
        "C4": C4,
        "maximum_component": maximum_component,
        "fast_rate_bound": bound,
        "error_bound_ratio": float(row["prediction_error"]) / bound,
        "radius_assumption_holds": delta > delta_bar,
        "re_assumption_holds": float(row["kappa_certificate"]) >= kappa,
    }


def summarize(config, results):
    """report the means and slopes, plus the main numerical checks"""
    summary = {}
    n = np.asarray(config.n)
    tail = n >= config.fit_min_n
    for p, cells in results.items():
        means = np.array(
            [np.mean([row["prediction_error"] for row in rows]) for rows in cells]
        )
        slope = None
        if tail.sum() >= 2 and np.all(means[tail] > 0):
            slope = float(np.polyfit(np.log(n[tail]), np.log(means[tail]), 1)[0])
        rows = [row for cell in cells for row in cell]
        summary[f"{p:g}"] = {
            "fits": len(rows),
            "mean_prediction_errors": means.tolist(),
            "slope": slope,
            "fit_minimum_sample_size": config.fit_min_n,
            "solver_status_counts": dict(Counter(row["solver_status"] for row in rows)),
            "near_zero_fits": sum(row["beta_hat_l1"] < 1e-6 for row in rows),
            "re_certificate_failures": sum(
                not row["re_assumption_holds"] for row in rows
            ),
            "radius_assumption_failures": sum(
                not row["radius_assumption_holds"] for row in rows
            ),
            "maximum_error_bound_ratio": max(row["error_bound_ratio"] for row in rows),
        }

    return summary


def run(config, directory=None, *, read_only=False):
    """loop over the requested fits and save the experiment in one folder"""

    # choose the folder from the experiment settings unless one is supplied
    if directory is None:
        root = Path(__file__).resolve().parents[4]
        directory = root / "data" / f"fast_{config_fingerprint(config)}"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # keep one data and radius configuration per folder
    config_path = directory / "config.json"
    if config_path.exists():
        saved = argparse.Namespace(**json.loads(config_path.read_text())["config"])
        if config_fingerprint(saved) != config_fingerprint(config):
            raise ValueError(f"configuration mismatch: {directory}")
    settings = {"kind": "fast", "config": vars(config)}
    config_path.write_text(json.dumps(json_value(settings), indent=2) + "\n")

    # load completed fits and run missing ones
    results = {}
    for p in config.p:
        results[p] = []
        for n in config.n:
            rows = []
            for repetition in range(1, config.k + 1):
                path = directory / f"n={n}_p={p:g}_rep={repetition}.h5"
                metadata = run_metadata(config, n, p, repetition)
                if path.exists():
                    row = load_run(path, metadata)
                    row = compute_bounds(row, config, n, p)
                else:
                    if read_only:
                        raise FileNotFoundError(f"missing saved fit: {path}")
                    print(f"fitting p={p:g} n={n} repetition={repetition}", flush=True)
                    # seeds depend on the dataset index, not on parameter names
                    key = f"{n}:{p:g}:{repetition}"
                    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
                    row = run_one(config, n, p, seed)
                # Refresh derived quantities in cached fits without refitting.
                save_run(path, row, metadata)
                rows.append(row)
            results[p].append(rows)
            mean_error = np.mean([row["prediction_error"] for row in rows])
            print(
                f"p={p:g} n={n}: {len(rows)} fits; mean prediction error={mean_error:.5g}",
                flush=True,
            )

    # summarize the requested grid alongside the fits
    summary = json_value(summarize(config, results))
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return results


def main(argv=None):
    """read experiment settings from the command line"""
    defaults = default_config()
    parser = argparse.ArgumentParser(
        description="fast-rate experiment with incremental saving"
    )
    parser.add_argument("--n", type=int, nargs="+", default=defaults.n)
    parser.add_argument("--p", type=float, nargs="+", default=defaults.p)
    parser.add_argument("--d", type=int, default=defaults.d)
    parser.add_argument("--s", type=int, default=defaults.s)
    parser.add_argument(
        "--B", type=float, default=defaults.B, help="l1 norm of beta_star"
    )
    parser.add_argument(
        "--sigma-sq", type=float, default=defaults.sigma_sq, help="noise variance"
    )
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--K", type=float, default=defaults.K)
    parser.add_argument(
        "--M",
        type=float,
        default=defaults.M,
        help="covariates are Uniform[-M, M]",
    )
    parser.add_argument("--k", type=int, default=defaults.k, help="repetitions")
    parser.add_argument(
        "--fit-min-n",
        type=int,
        default=defaults.fit_min_n,
        help="minimum n used for the slope",
    )
    parser.add_argument(
        "--directory",
        "--cache-root",
        type=Path,
        help="folder for fits, config, and summary",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="read saved fits without running new ones",
    )
    args = vars(parser.parse_args(argv))
    directory = args.pop("directory")
    read_only = args.pop("read_only")
    config = defaults
    vars(config).update(args)

    # slicing beta_star would silently truncate an oversized support
    if not 1 <= config.s <= config.d:
        parser.error("require 1 <= s <= d")

    run(config, directory, read_only=read_only)


if __name__ == "__main__":
    main()
