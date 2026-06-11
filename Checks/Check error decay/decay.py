"""
In-sample prediction error decay for the minimax estimator

    beta_hat = argmin_beta  Vbar_delta(beta),

    n * Vbar_delta(beta) = sup_{alpha>=0} [ n^{1/p} delta ||beta||_1 ||alpha||_q
                                            + sum_i ( alpha_i |x_i^T beta - y_i| - alpha_i^2/4 ) ]

with 1/p + 1/q = 1 and  ||.|| = ||.||_inf  (so ||.||_* = ||.||_1).

err(n) = (1/n)||X(beta_hat - beta_star)||_2^2 tracked vs n, for several p>=2,
in two regimes:
  noRE : correlated design.  Expect err = O(n^{-1/2}).
  RE   : well-conditioned design + sparse beta*.  Expect err = O(n^{-1}).

Two delta choices are run, each producing its own figure:
  'simple' : delta = K M sqrt(log d / n)                          -> decay.png
  'Cn'     : delta = C(n)/sqrt(n),                                -> decay_Cn.png
             C(n) = M sqrt(log(d/gamma)) / (sqrt(1/pi) + sqrt(log(1/gamma)/n)),
             M = max|X_ij|, giving the O(n^{-1/2}) bound w.p. 1 - 4 gamma (gamma=0.01).

Solver: Vbar is convex in beta. By Danskin, grad = (1/n) sum_i alpha*_i sign(r_i) x_i
        + (1/n) n^{1/p} delta ||alpha*||_q * subgrad(||beta||_1), where alpha* is the
        inner maximizer. We minimize with L-BFGS-B using this (sub)gradient.
Single python file.
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_delta(mode, M, d, n, K=1.0, gamma=0.01):
    """Two regularization choices.
       'simple' : delta = K * M * sqrt(log d / n)
       'Cn'     : delta = C(n)/sqrt(n),
                  C(n) = M sqrt(log(d/gamma)) / ( sqrt(1/pi) + sqrt(log(1/gamma)/n) )
                  -> high-probability (1 - 4 gamma) bound."""
    if mode == "simple":
        return K * M * np.sqrt(np.log(max(d, 2)) / n)
    # mode == "Cn"
    Cn = M * np.sqrt(np.log(d / gamma)) / (np.sqrt(1.0 / np.pi)
                                           + np.sqrt(np.log(1.0 / gamma) / n))
    return Cn / np.sqrt(n)


def one_rep(args):
    """Single (regime, p, n, rep, delta_mode) trial -> in-sample error. Top-level for pickling."""
    reg, p, n, rep, sigma, K, delta_mode, gamma = args
    X, y, beta_star, M, d = make_data(n, reg, sigma, 1000 * rep + n)
    delta = compute_delta(delta_mode, M, d, n, K=K, gamma=gamma)
    beta_hat = fit(X, y, delta, p, np.zeros(d))
    diff = X @ (beta_hat - beta_star)
    return float(np.mean(diff ** 2))


def inner_argmax(r, c, delta, n, p):
    """Return (value n*Vbar, alpha*, ||alpha*||_q)."""
    A = (n ** (1.0 / p)) * delta * c if np.isfinite(p) else delta * c
    if not np.isfinite(p):                        # p=inf, q=1
        alpha = 2.0 * np.maximum(r + A, 0.0)
        val = np.sum(np.maximum(r + A, 0.0) ** 2)
        return val, alpha, np.sum(np.abs(alpha))
    q = 2.0 if abs(p - 2.0) < 1e-12 else p / (p - 1.0)

    def neg_and_grad(a):
        nrm = np.linalg.norm(a, q)
        val = A * nrm + np.sum(a * r) - 0.25 * np.sum(a ** 2)
        if nrm > 1e-15:
            g_nrm = A * (np.abs(a) ** (q - 1)) * np.sign(a) / (nrm ** (q - 1))
        else:
            g_nrm = np.zeros_like(a)
        grad = g_nrm + r - 0.5 * a
        return -val, -grad

    a0 = np.maximum(2.0 * r, 1e-3)
    res = minimize(neg_and_grad, a0, jac=True, bounds=[(0.0, None)] * len(r),
                   method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-12})
    a = res.x
    return -res.fun, a, np.linalg.norm(a, q)


def Vbar_and_grad(beta, X, y, delta, p):
    n = len(y)
    res = X @ beta - y
    r = np.abs(res)
    c = np.sum(np.abs(beta))
    val, alpha, alpha_q = inner_argmax(r, c, delta, n, p)
    # Danskin: differentiate K at fixed alpha*
    A_coef = (n ** (1.0 / p)) * delta if np.isfinite(p) else delta
    grad_res = alpha * np.sign(res)               # d/dbeta of sum alpha_i|res_i|
    grad = X.T @ grad_res + A_coef * alpha_q * np.sign(beta)
    return val / n, grad / n


def fit(X, y, delta, p, beta_init):
    res = minimize(lambda b: Vbar_and_grad(b, X, y, delta, p), beta_init,
                   jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-8})
    return res.x


def make_data(n, regime, sigma, seed):
    rng = np.random.default_rng(seed)
    if regime == "RE":
        d = 8
        X = rng.uniform(-1.0, 1.0, size=(n, d))
        beta_star = np.zeros(d)
        beta_star[:3] = np.array([2.0, -1.5, 1.0])
    else:
        d = min(max(8, n // 2), 60)   # grows with n but capped so large-n stays tractable
        factor = rng.uniform(-1.0, 1.0, size=(n, 1))
        X = np.clip(0.9 * factor + 0.1 * rng.uniform(-1.0, 1.0, size=(n, d)),
                    -1.0, 1.0)
        beta_star = rng.standard_normal(d) / np.sqrt(d)
    M = np.max(np.abs(X))
    y = X @ beta_star + sigma * rng.standard_normal(n)
    return X, y, beta_star, M, d


def run_one_mode(delta_mode, ns, ps, regimes, sigma, K, gamma, n_reps, out_png, mode_label):
    mean_res = {(reg, p): [] for reg in regimes for p in ps}
    std_res = {(reg, p): [] for reg in regimes for p in ps}

    for reg in regimes:
        for p in ps:
            for n in ns:
                errs = np.array([one_rep((reg, p, n, rep, sigma, K, delta_mode, gamma))
                                 for rep in range(n_reps)])
                mean_res[(reg, p)].append(errs.mean())
                std_res[(reg, p)].append(errs.std(ddof=1))
                print(f"[{delta_mode}] regime={reg:5s} p={str(p):>4s} n={n:4d} "
                      f"mean={errs.mean():.5f} std={errs.std(ddof=1):.5f}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    titles = {"RE": "RE / well-conditioned  (expect ~ n^-1)",
              "noRE": "no RE / correlated  (expect ~ n^-1/2)"}
    ns_arr = np.array(ns, dtype=float)
    for ax, reg in zip(axes, regimes):
        for p in ps:
            m = np.array(mean_res[(reg, p)])
            s = np.array(std_res[(reg, p)])
            ax.errorbar(ns_arr, m, yerr=s, fmt="o-", capsize=3,
                        elinewidth=1, label=f"p = {p}")
        anchor = mean_res[(reg, 2.0)][0]
        ax.loglog(ns_arr, anchor * (ns_arr[0] / ns_arr) ** 0.5, "k--", lw=1, label="n^-1/2 ref")
        ax.loglog(ns_arr, anchor * (ns_arr[0] / ns_arr) ** 1.0, "k:", lw=1, label="n^-1 ref")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(titles[reg]); ax.set_xlabel("n")
        ax.set_ylabel(r"$\frac{1}{n}\|X(\hat\beta-\beta^*)\|_2^2$")
        ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(r"Minimax $\bar V_\delta$ estimator: in-sample error decay "
                 + mode_label + r" (mean $\pm$ std over %d runs)" % n_reps, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print("saved", out_png)

    print(f"\nFitted log-log slopes [{delta_mode}] (mean err ~ n^slope):")
    for reg in regimes:
        for p in ps:
            slope = np.polyfit(np.log(ns_arr), np.log(mean_res[(reg, p)]), 1)[0]
            print(f"  regime={reg:5s} p={str(p):>4s}: slope = {slope:+.3f}")


def run():
    ns = [40, 80, 160, 320, 640]
    ps = [2.0, 3.0, 6.0, np.inf]
    regimes = ["RE", "noRE"]
    sigma = 0.5
    K = 1.0
    gamma = 0.01           # high-probability level for the C(n) delta (bound holds w.p. 1 - 4 gamma)
    n_reps = 20

    # Original choice: delta = K M sqrt(log d / n)
    run_one_mode("simple", ns, ps, regimes, sigma, K, gamma, n_reps,
                 "/home/claude/decay.png",
                 r"($\delta = K M \sqrt{\log d / n}$)")

    # New choice: delta = C(n)/sqrt(n),  C(n) = M sqrt(log(d/gamma)) / (sqrt(1/pi) + sqrt(log(1/gamma)/n))
    run_one_mode("Cn", ns, ps, regimes, sigma, K, gamma, n_reps,
                 "/home/claude/decay_Cn.png",
                 r"($\delta = C(n)/\sqrt{n}$, $\gamma=%.2f$)" % gamma)


if __name__ == "__main__":
    run()
