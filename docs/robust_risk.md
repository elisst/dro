# Evaluate the robust risk at fixed parameter

For a given Wasserstein radius $\delta \geq 0$, `RobustRisk` evaluates the risk $V_\delta(\beta)$ as a function of $\beta$. The alternative movement-cost convention $\bar V$ is explained below. From Theorem 1, Eq. (5), we have 
$$
V_\delta(\beta)
=\max_{t_i\geq0:\,\sum_i t_i^p\leq n\delta^p}
\frac1n\sum_{i=1}^n \bigl(|r_i|+t_i\|\beta\|_*\bigr)^2,
$$
where $r_i=x_i^\top\beta-y_i$, $2\leq p<\infty$ and $\lVert \cdot \rVert_*$ denotes the dual norm of $\lVert \cdot \rVert$. Furthermore,

- A length $t_i$ specifies how far observation $i$ may move in the input space. 
- Labels $\{y_i\}_{i=1}^n$ are fixed. 
- The constraint limits the average movement cost to $\delta^p$.

At $p=\infty$, the lengths $\{t_i\}_{i=1}^n$ are bounded uniformly by $\delta$, i.e., $0\leq t_i\leq\delta \; \forall i \in [n]$.

## Evaluate and cross-check

The following example evaluates the risk.

```python
import numpy as np
from DRO.robust_risk import RobustRisk

# draw some data using beta_star
rng = np.random.default_rng(7)
X = rng.normal(size=(20, 3)) # rows are observations and columns are features
beta_star = np.array([1.0, -0.5, 0.0])
y = X @ beta_star + 0.2 * rng.normal(size=20)

# eval risk at beta
beta = np.array([0.8, -0.4, 0.1])
delta, p = 0.15, 3
risk = RobustRisk.normalized(X, y, delta, p, norm=np.inf)

value = risk.primal(beta)
check = risk.dual(beta)
print(value, check, abs(value - check))
```

`primal` uses the scalar reduction described below. `dual` uses the saddle-point formulation supplied in Theorem 8, Eq. (20). Here $q=p/(p-1)$ for finite $p$, and $q=1$ at $p=\infty$.
$$
n V_{\delta}(\beta) = \sup_{\gamma \geq 0} K(\beta,\gamma^{1/q}) =
\sup_{\gamma \geq 0} \left [ n^{1/p} \delta \|\beta\|_* \|\gamma\|_1^{1/q}+\sum_{i=1}^n \left(\gamma^{1/q}_i |x_i^T \beta - y_i | - \gamma_i^{2/q}/4\right) \right ].
$$
$K(\beta,\gamma^{1/q})$ is convex-concave in $(\beta, \gamma)$, meaning `CVXPY` can be used to maximize w.r.t. $\gamma$ for a fixed $\beta$. This serves as a baseline for the `primal`, and is used heavily in the test files, e.g., [tests/robust_risk_test.py](../tests/robust_risk_test.py). This is a numerical cross-check. `CVXPY` uses rational approximations of the power exponents, and solver accuracy warnings should be checked.

Inputs are finite arrays `X` of shape `(n, d)` and `y` of shape `(n,)`, with `n, d > 0`. The parameter vector `beta` has shape `(d,)`. Use

- $\delta\geq0$, 
- $2\leq p\leq\infty$, and 
- a ground-norm exponent $\geq 1$. 

*The evaluator assumes these input conditions*.

The argument `norm` chooses the norm $\|\cdot\|$ in the Wasserstein transport cost $c(x,x') = \|x-x'\|^p$. Note that this is separate from the Wasserstein order `p`. The coefficient norm $\|\beta\|_*$ is the dual of this norm and is determined automatically.

## Radius and returned units

Prefer `RobustRisk.normalized(X, y, delta, p)` when `delta` is the Wasserstein radius. For finite $p$, the direct constructor instead accepts average movement cost $\delta^p$.

```python
delta = 0.15
by_radius = RobustRisk.normalized(X, y, delta, p) # V
by_cost = RobustRisk(X, y, delta**p, p) # \bar{V}

print(by_radius.primal(beta))
print(by_cost.primal(beta))  # same mathematical quantity
```

By the convention of the paper, $V_\delta(\beta)=\bar V_{\delta^p}(\beta)$. Internally, both constructors store the radius in `risk.delta`. The normalized constructor avoids forming `delta**p`, which can overflow or underflow even when the risk itself is representable.

At `p=np.inf`, both constructors take the radius directly. Do not form `delta**p` in that case.

Both evaluation methods return the **mean robust squared loss** $V_\delta(\beta)$ by default, matching the CVX solver. Pass `per_sample=False` if you need the summed loss $nV_\delta(\beta)$.

```python
mean = risk.primal(beta)                       # V_delta(beta)
total = risk.primal(beta, per_sample=False)     # n * V_delta(beta)
assert np.isclose(total, len(y) * mean)
```

## Connecting risk and code

For the rest of this guide, abbreviate $B=\|\beta\|_*$. `primal` computes the residual magnitudes $|r_i|$ (stored as the array `residual_abs`, abbreviated $|r|$ below) and $B$, then passes them to `_ScalarRisk.solve`. The helper works with the summed loss $nV_\delta(\beta)$ and selects among the following cases. `primal` divides its result by $n$ to return the mean risk by default.

| Case, in code order | Summed loss $nV_\delta(\beta)$ |
| --- | --- |
| $\delta=0$ | $\sum_i r_i^2$ |
| $p=\infty$ | $\sum_i (\lvert r_i \rvert+ \delta B)^2$ (adversarial training) |
| $B=0$ | $\sum_i r_i^2$ |
| $p=2$ | $(\|r\|_2+\sqrt n\,\delta B)^2$ (square-root Lasso)|
| $2<p<\infty$ and $\delta, B>0$ | A scalar search in $\lambda$ |

The first four cases need no numerical optimization. The general case follows the call structure below.

```text
RobustRisk.primal(beta)
    _ScalarRisk.solve(|r|, B)
        _solve_finite_p(|r|, B)
            _solve_lambda(a, b)
                _maximize_t(a, b, lambda)
            _checked_value(a, b, lambda, u)
```

Here `a`, `b`, and `u` are scaled versions of the residuals, coefficient norm, and lengths. The scaling is explained below.

## Formulas for explicit subcases

### Zero radius

For finite $p$, when $\delta=0$, the budget
$$
\begin{cases}
t_i \geq 0, \\
\sum_i t_i^p \leq n \delta^p = 0
\end{cases}
$$
forces every $t_i=0$. At $p=\infty$, the constraints $0\leq t_i\leq\delta=0$ also force every $t_i=0$. Substitution gives
$$
nV_0(\beta)=\sum_{i=1}^n(|r_i|+0\cdot B)^2=\sum_{i=1}^n r_i^2.
$$
No covariate movement is allowed, so the robust loss is the ordinary squared loss.
```python
if self.delta == 0:
    # no perturbations are allowed
    return float(np.sum(residual_abs**2))
```

### Infinite exponent

At $p=\infty$, each observation has its own constraint $0\leq t_i\leq\delta$. Since $B\geq0$, $(|r_i|+Bt_i)^2$ is nondecreasing in $t_i$. Thus choosing $t_i=\delta$ for every observation attains the maximum, giving
$$
nV_\delta(\beta)=\sum_{i=1}^n(|r_i|+\delta B)^2.
$$
This is the adversarial-training endpoint in Proposition 2, expressed as a summed loss.
```python
if np.isinf(self.p):
    # each observation can use t_i = delta in adversarial training
    return float(np.sum((residual_abs + B * self.delta) ** 2))
```

### Zero coefficient norm

When $B=\|\beta\|_*=0$, the term $Bt_i$ vanishes for every allowed length. Therefore
$$
nV_\delta(\beta)=\sum_{i=1}^n(|r_i|+0\cdot t_i)^2=\sum_{i=1}^n r_i^2.
$$
Moving the covariates cannot change the prediction of a zero coefficient vector. The function simply returns the empirical squared loss.

```python
if B == 0:
    # moving covariates cannot change a zero-coefficient prediction
    return float(np.sum(residual_abs**2))
```

### Quadratic exponent

For $p=2$, the length budget is $\|t\|_2\leq\sqrt n\,\delta$. The triangle inequality gives
$$
\sqrt{\sum_{i=1}^n(|r_i|+Bt_i)^2}
\leq\sqrt{\sum_{i=1}^n r_i^2}+B\sqrt{\sum_{i=1}^n t_i^2}
\leq\|r\|_2+B\sqrt n\,\delta.
$$
To attain this upper bound, we want equality in both steps, meaning the added vectors have to point in the same direction. For $r\neq0$, we therefore choose $t_i=c|r_i|$ for every $i$, with a common $c\geq0$. To use the whole length budget, choose $c$ so that
$$
\|t\|_2=c\sqrt{\sum_{i=1}^n|r_i|^2}=c\|r\|_2=\sqrt n\,\delta.
$$
Solving for $c$ gives the individual lengths
$$
t_i=\sqrt n\,\delta\,\frac{|r_i|}{\|r\|_2},\qquad i=1,\ldots,n.
$$
Thus each length is proportional to its residual magnitude, with the common factor chosen to use the whole budget. These lengths are nonnegative and make both inequalities equalities. Squaring gives
$$
nV_\delta(\beta)=(\|r\|_2+\sqrt n\,\delta B)^2.
$$
If $r=0$, uniform lengths $t_i=\delta$ attain the same formula. `_sqrt_lasso` evaluates it directly, including at exact interpolation. This is the square-root-lasso endpoint in Proposition 2.

```python
if self.p == 2:
    return self._sqrt_lasso(residual_abs, B)
```

## The scalar search for intermediate exponents

For finite $p$, the risk also has the scalar representation
$$
nV_\delta(\beta)
=\inf_{\lambda\geq0}\left[
n\delta^p\lambda+
\sum_{i=1}^n\sup_{t_i\geq0}\{(|r_i|+Bt_i)^2-\lambda t_i^p\}
\right].
$$
This is Lemma 1, Eq. (14), with the one-dimensional inner problem studied after Eq. (35). We write an infimum because a finite minimizing multiplier need not exist at $\delta=0$. That case is handled directly above.

Write the budget constraint as $g(t)=\sum_{i=1}^n t_i^p-n\delta^p\leq0$. Since this is a maximization problem, its Lagrangian subtracts $\lambda g(t)$, with $\lambda\geq0$.
$$
L(t,\lambda)
=\sum_{i=1}^n(|r_i|+Bt_i)^2-\lambda g(t)
=n\delta^p\lambda+\sum_{i=1}^n\left[(|r_i|+Bt_i)^2-\lambda t_i^p\right].
$$
For feasible $t$, we have $-\lambda g(t)\geq0$, so the Lagrangian is an upper bound on the loss. The dual problem minimizes $\sup_{t\geq0}L(t,\lambda)$ over $\lambda\geq0$. The supremum separates across observations, giving the scalar representation above. Intuitively, $\lambda$ is the price of movement.

To establish **strong duality**, set $z_i=t_i^p$. Each objective term becomes $r_i^2+2|r_i|Bz_i^{1/p}+B^2z_i^{2/p}$, which is concave for $p\geq2$, while the constraints $z_i\geq0$ and $\sum_i z_i\leq n\delta^p$ are affine. Thus the transformed problem is a convex optimization problem. For $\delta>0$, Slater's condition holds (take $z_i=\delta^p/2$), so the primal maximum equals the dual infimum. The multiplier formulation is therefore exact, not merely an upper bound.

```python
return self._solve_finite_p(residual_abs, B)
```

### Find the maximizing length for a given price

For $2<p<\infty$, $B>0$, and $\lambda>0$, `_maximize_t` maximizes
$$
f_i(t)=(|r_i|+Bt)^2-\lambda t^p,\qquad t\geq0.
$$
Its derivative is
$$
f_i'(t)=2B(|r_i|+Bt)-p\lambda t^{p-1}.
$$
For $t>0$, the expression $f_i'(t)/t=2B|r_i|/t+2B^2-p\lambda t^{p-2}$ decreases strictly from a positive value to a negative value. Therefore there is a unique positive maximizer, at the change of sign of $f_i'$.

The code first doubles an upper endpoint until the derivative is nonpositive, then bisects the bracket. It does this for all observations at once using arrays. Failure to obtain a finite bracket raises an error.

### Choose the price that uses the budget

Let $t_i(\lambda)$ be those maximizing lengths. Differentiating the outer objective gives
$$
n\delta^p-\sum_{i=1}^n t_i(\lambda)^p.
$$
If the lengths spend too much, increase $\lambda$. If they spend too little, decrease it. `_solve_lambda` finds the price at which the budget is met using a bracketed root search (`scipy.optimize.brentq`). This is the only outer numerical search.

### Scale the calculation before searching

The original multiplier $\lambda$ can be extremely large or small when the radius or response units change. `_solve_finite_p` instead uses
$$
s=\max\{\max_i|r_i|,\delta B\},\quad
a_i=|r_i|/s,\quad b=\delta B/s,\quad u_i=t_i/\delta.
$$
In this branch $\delta,B>0$, so $s>0$. The budget becomes $\frac1n\sum_i u_i^p\leq1$, and the scaled multiplier is $\widetilde\lambda=\lambda\delta^p/s^2$. The equivalent objective is
$$
\frac{nV_\delta(\beta)}{s^2}
=\inf_{\widetilde\lambda\geq0}
\left[n\widetilde\lambda+
\sum_{i=1}^n\sup_{u_i\geq0}\{(a_i+bu_i)^2-\widetilde\lambda u_i^p\}\right].
$$
The code uses these scaled quantities throughout the search and multiplies the final value by $s^2$. It never needs to reconstruct the original multiplier.

```python
scale = max(float(np.max(residual_abs)), self.delta * B)
a, b = residual_abs / scale, self.delta * B / scale
lam, u = self._solve_lambda(a, b)
n_value = self._checked_value(a, b, lam, u)
return n_value * scale**2
```

A bracket follows by asking which price makes $u_i=1$ stationary. Substitution into the inner derivative gives $\widetilde\lambda=2b(a_i+b)/p$. Hence the endpoints
$$
\lambda_{\rm lo}=\frac{2b(\min_i a_i+b)}p,
\qquad
\lambda_{\rm hi}=\frac{2b(\max_i a_i+b)}p
$$
place all maximizing lengths at least one at the lower endpoint and at most one at the upper endpoint. They bracket the required average budget. If the endpoints coincide, every optimal length is one and the value can be evaluated immediately.

### Check the numerical value

After the root search, `_checked_value` compares two quantities in scaled units.

- The multiplier expression, which is an upper bound when the inner maxima are computed exactly.
- The loss from feasible lengths, which is a lower bound. If rounding puts the lengths just outside the budget, they are first shrunk to feasibility.

The code rejects disagreement above its relative tolerance of $10^{-8}$. It also rejects a failed root search or unusable numerical brackets. These are numerical consistency checks, not exact-arithmetic error certificates.
