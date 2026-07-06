import numpy as np
from scipy.optimize import minimize

# ---- Norm setup: ||.|| = ||.||_inf, so dual ||.||_* = ||.||_1 ----
def norm_star(beta):          # dual of inf-norm is 1-norm
    return np.sum(np.abs(beta))

# ===========================================================
#  Core: n * Vbar_delta(beta) = sup_{alpha>=0} K(beta,alpha)
#  K = n^{1/p} delta ||beta||_* ||alpha||_q
#        + sum_i ( alpha_i |r_i| - alpha_i^2/4 )
#  Optimize in gamma = alpha^q (concave). For numerics we just
#  optimize directly over alpha>=0 with a concave/structured solver.
# ===========================================================

def nVbar_p2(beta, X, y, delta):
    """p=2, q=2. Optimize over alpha>=0 numerically (problem concave in gamma=alpha^2)."""
    n = len(y)
    r = np.abs(X @ beta - y)
    c = norm_star(beta)
    A = n**0.5 * delta * c                      # coefficient on ||alpha||_2

    # maximize  A*||alpha||_2 + sum(alpha_i r_i - alpha_i^2/4),  alpha>=0
    def neg(alpha):
        nrm = np.linalg.norm(alpha)
        return -(A*nrm + np.sum(alpha*r) - np.sum(alpha**2)/4)
    alpha0 = np.maximum(2*r, 1e-3) + 1.0
    res = minimize(neg, alpha0, bounds=[(0, None)]*n, method='L-BFGS-B',
                   options={'maxiter':5000, 'ftol':1e-12})
    return -res.fun

def nVbar_pinf(beta, X, y, delta):
    """p=inf, q=1. Closed form: alpha_i = 2(delta*||beta||_* + |r_i|)_+."""
    r = np.abs(X @ beta - y)
    c = norm_star(beta)
    s = np.maximum(delta*c + r, 0.0)            # always >=0 here
    return np.sum(s**2)                          # n * Vbar

# ---- Comparison objectives ----
def sqrt_lasso_obj(beta, X, y, delta):
    """sqrt-Lasso style objective: sqrt(sum r_i^2) + n^{1/2} delta ||beta||_1."""
    r = X @ beta - y
    return np.sqrt(np.sum(r**2)) + np.sqrt(len(y))*delta*norm_star(beta)

def adversarial_obj(beta, X, y, delta):
    """Adversarial linear regression, l_inf perturbations of size delta on each x_i.
       worst-case sqrt-loss per point: |r_i| + delta*||beta||_1, summed in squares."""
    r = np.abs(X @ beta - y)
    return np.sum((r + delta*norm_star(beta))**2)

# ===========================================================
#  Run experiments over several settings
# ===========================================================
def make_data(n, d, sigma, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    beta_star = rng.standard_normal(d)
    e = sigma * rng.standard_normal(n)
    y = X @ beta_star + e
    beta = beta_star + 0.3*rng.standard_normal(d)   # some non-optimal beta to evaluate at
    return X, y, beta

settings = [
    dict(n=50,  d=5,  sigma=0.5, delta=0.1, seed=0),
    dict(n=100, d=10, sigma=1.0, delta=0.2, seed=1),
    dict(n=200, d=20, sigma=0.3, delta=0.05, seed=2),
    dict(n=80,  d=8,  sigma=0.8, delta=0.15, seed=3),
]

print("="*78)
print("CASE 1:  ||.||=inf-norm,  p=2  (q=2)   ->  compare sqrt(Vbar) vs sqrt-Lasso obj")
print("="*78)
print(f"{'n':>4}{'d':>4}{'sigma':>7}{'delta':>7}{'sqrt(nVbar)':>14}{'sqrtLasso':>12}{'diff':>11}")
for s in settings:
    X, y, beta = make_data(s['n'], s['d'], s['sigma'], s['seed'])
    nV = nVbar_p2(beta, X, y, s['delta'])
    # n*Vbar -> Vbar = nV/n ; compare sqrt(n*Vbar) with sqrt-Lasso (un-normalized) objective
    lhs = np.sqrt(nV)
    rhs = sqrt_lasso_obj(beta, X, y, s['delta'])
    print(f"{s['n']:>4}{s['d']:>4}{s['sigma']:>7}{s['delta']:>7}"
          f"{lhs:>14.6f}{rhs:>12.6f}{lhs-rhs:>11.2e}")

print()
print("="*78)
print("CASE 2:  ||.||=inf-norm,  p=inf  (q=1)  ->  compare nVbar vs adversarial obj")
print("="*78)
print(f"{'n':>4}{'d':>4}{'sigma':>7}{'delta':>7}{'nVbar':>16}{'adversarial':>16}{'diff':>11}")
for s in settings:
    X, y, beta = make_data(s['n'], s['d'], s['sigma'], s['seed'])
    nV  = nVbar_pinf(beta, X, y, s['delta'])
    adv = adversarial_obj(beta, X, y, s['delta'])
    print(f"{s['n']:>4}{s['d']:>4}{s['sigma']:>7}{s['delta']:>7}"
          f"{nV:>16.6f}{adv:>16.6f}{nV-adv:>11.2e}")