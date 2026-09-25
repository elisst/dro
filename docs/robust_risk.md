# Evaluate the robust risk

[`RobustRisk`](../src/DRO/robust_risk.py) evaluates the risk at a fixed $\beta$. Write $r=X\beta-y$ and $B=\|\beta\|_*$. Theorem 1, Eq. (7) of the paper gives, for finite $p$,

$$
V_\delta(\beta)=\max_{t_i\geq0:\,\sum_i t_i^p\leq n\delta^p}
\frac{1}{n} \sum_{i=1}^n (|r_i|+Bt_i)^2.
$$

At $p=\infty$, the constraint becomes $0\leq t_i\leq\delta$ for every observation.

## Evaluate and compare

```python
import numpy as np
from DRO.robust_risk import RobustRisk

# draw some data
rng = np.random.default_rng(7)
X = rng.normal(size=(20, 3))
y = X @ np.array([1.0, -0.5, 0.0]) + 0.2 * rng.normal(size=20)
beta = np.array([0.8, -0.4, 0.1])

# eval risk
risk = RobustRisk(X, y, delta=0.15, p=3.0, norm=np.inf)
# risk at beta using primal solver (scalar reduction)
primal = risk.primal(beta)

# risk at beta using dual solver (saddle point formulation)
dual = risk.dual(beta)

print(primal, dual, abs(primal - dual))
```

`delta` is the Wasserstein radius. Both methods return mean squared risk $V_\delta$ -- `per_sample=False` returns $nV_\delta$. `primal` uses the scalar reduction, and `dual` uses the independent gamma formulation below.

Inputs are finite arrays `X` of shape `(n, d)`, `y` of shape `(n,)`, and `beta` of shape `(d,)`, with $n,d>0$, $\delta\geq0$, $2\leq p\leq\infty$, and `norm >= 1` (including `np.inf`). *The code assumes these conditions*. `norm` is the ground-norm exponent, separate from `p`. For `norm=1, 2, np.inf`, the $\beta$ dual norms $\lVert \beta \rVert_*$ are $\ell_\infty,\ell_2,\ell_1$ respectively.

## Explicit cases

`primal` passes `residual_abs = abs(X @ beta - y)` and `B` to `_ScalarRisk.solve`. The following special cases are simple to deal with and thus require no numerical search.

**Zero radius or zero beta.** If $\delta=0$, every feasible length $t_i$ is zero. If $B=0$, the perturbation term vanishes regardless of the lengths. Either way, $nV_\delta=\sum_i r_i^2$.

```python
if self.delta == 0 or B == 0:
    return float(np.sum(residual_abs**2))
```

**Infinite exponent.** Since $B>0$, Each term $(|r_i|+Bt_i)^2$ is nondecreasing in $t_i \geq 0$, so $t_i=\delta$ attains the maximum:

$$
V_\delta(\beta)=\frac1n\sum_i(|r_i|+\delta B)^2.
$$

This is the adversarial training case of Corollary 1.

```python
if np.isinf(self.p):
    return float(np.sum((residual_abs + B * self.delta) ** 2))
```

**Quadratic exponent.** For $p=2$, the budget is $\|t\|_2\leq\sqrt n\,\delta$. The triangle inequality gives

$$
\|\,|r|+Bt\|_2\leq\|r\|_2+B\|t\|_2\leq\|r\|_2+B\sqrt n\,\delta.
$$

For $r\neq0$, equality holds at $t=\sqrt n\,\delta\,|r|/\|r\|_2$ (the two vectors are aligned and the budget is used in full). If $r=0$, take $t_i=\delta$. Squaring gives Corollary 1's formula

$$
V_\delta(\beta)=\left(\frac{\|r\|_2}{\sqrt n}+\delta B\right)^2.
$$

With `norm=np.inf`, $B=\|\beta\|_1$ and thus we get the squared square-root-Lasso objective.

```python
if self.p == 2:
    return (
        np.linalg.norm(residual_abs)
        + np.sqrt(len(residual_abs)) * self.delta * B
    ) ** 2
```

## Scalar reduction for intermediate exponents

The remaining branch has $2<p<\infty$ and $\delta,B>0$. Lemma 3, Eq. (16), gives

$$
nV_\delta(\beta)=\min_{\lambda>0}
\left[n\delta^p\lambda+\sum_i\max_{t\geq0}f_i(t;\lambda)\right],
\qquad f_i(t;\lambda)=(|r_i|+Bt)^2-\lambda t^p.
$$

Thus, to evaluate the risk we need to solve the $n$ inner maximums as well as the outer minimum. Both $\mathrm{min}$ and $\mathrm{max}$ are scalar problems, which we solve using functions from `SciPy`.

### Why scalar minimizers apply

Following Appendix A.3,

$$
f_i'(t)=2B(|r_i|+Bt)-p\lambda t^{p-1},\qquad
f_i''(t)=2B^2-p(p-1)\lambda t^{p-2}.
$$

For $p>2$, $f_i''$ decreases from a positive value to $-\infty$. Thus $f_i'$ first increases and then decreases. It starts nonnegative, is positive for small $t>0$, and tends to $-\infty$, so it crosses zero exactly once on $(0,\infty)$. Hence $f_i$ is unimodal, with a unique positive maximum, including when $r_i=0$.

For fixed $t$, $f_i(t;\lambda)$ is affine in $\lambda$. Its maximum over $t$ is therefore convex in $\lambda$, and adding $n\delta^p\lambda$ preserves convexity. The code can minimize the negative inner objectives and then minimize the outer objective using `SciPy`. It does not need to solve the derivative equations itself.

### Scale the lengths and loss

The Lagrange multiplier $\lambda$ enforcing the transport budget can become extremely large or small, depending on the radius $\delta$, the exponent $p$, and the unit of the response $y$. Computing $\delta^p$ directly can also cause numerical overflow or underflow. To avoid these numerical difficulties, `_solve_finite_p` rescales the problem:

$$
u_i=\frac{t_i}{\delta},\qquad
s=\max\{\max_i|r_i|,\delta B\},\qquad
a_i=\frac{|r_i|}{s},\qquad b=\frac{\delta B}{s}.
$$

Here, $u_i$ measures movement relative to the radius $\delta$, so the budget becomes $\frac1n\sum_i u_i^p\leq1$. $s$ scales the residual and movement contributions so that $0\leq a_i\leq1$ and $0<b\leq1$. The solver works with the corresponding scaled multiplier

$$
\widetilde\lambda=\frac{\lambda\delta^p}{s^2},
$$

without needing to compute the unscaled $\lambda$ or $\delta^p$. Since this branch has $\delta,B>0$, we know that $s>0$ and are safe to divide.
```python
scale = max(float(np.max(residual_abs)), self.delta * B)
a = residual_abs / scale
b = self.delta * B / scale
```

Substituting $t=\delta u$ gives

$$
(|r_i|+Bt)^2-\lambda t^p = \left(s a_i + \frac{bs}{\delta} \delta u \right)^2 - \lambda u^p \delta^p
= s^2 \left[ \left( a_i + b u \right)^2 - \frac{u^p \lambda \delta^p}{s^2} \right]
=s^2\left[(a_i+bu)^2-\widetilde\lambda u^p\right].
$$

Consequently,

$$
\frac{nV_\delta(\beta)}{s^2}
=\min_{\widetilde\lambda>0}H(\widetilde\lambda),
\qquad
H(\widetilde\lambda)
=n\widetilde\lambda+
\sum_i\max_{u\geq0}
\left[(a_i+bu)^2-\widetilde\lambda u^p\right].
$$

The code's `lam` represents the scaled multiplier $\widetilde\lambda$. Multiplying the optimum by `scale**2` restores the summed squared loss in the original response units and dividing by $n$ gives the mean risk $V_\delta(\beta)$. This rescaling leaves the input radius and mathematical risk unchanged.

### Bound the multiplier search

Before introducing the multiplier $\widetilde{\lambda}$, the scaled problem is

$$
\max_{\substack{u_i\geq0 \\ \frac{1}{n} \sum_i u_i^p\leq 1}}
\sum_i(a_i+bu_i)^2.
$$

Since $b>0$, the maximizing lengths $\{ u_i^\star \}_{i=1}^n$ use the full budget since increasing any length increases its loss contribution $(a_i+bu_i)^2$.

Duality replaces this constrained maximization with

$$
\min_{\widetilde\lambda>0}H(\widetilde\lambda),
\qquad
H(\widetilde\lambda)
=n\widetilde\lambda+
\sum_i\max_{u\geq0}
\left[(a_i+bu)^2-\widetilde\lambda u^p\right].
$$

The inner maximizations now impose only $u\geq0$, i.e., they no longer enforce the shared budget $\frac{1}{n} \sum_i u_i^p\leq 1$. For each trial multiplier $\widetilde\lambda$, every observation independently chooses its preferred movement length $u$ at that price. *These lengths may collectively exceed or underspend the budget*. The penalized objective above does not necessarily increase with $u$, because the penalty $-\widetilde\lambda u^p$ eventually dominates the loss (as $\widetilde{\lambda}$ increases from $0$).

The outer minimization adjusts the multiplier $\widetilde{\lambda}$ until the inner maximizers use exactly the budget. Writing their lengths as $u_i(\widetilde\lambda)$, the derivative of the optimized objective is

$$
H'(\widetilde\lambda)=n-\sum_i u_i(\widetilde\lambda)^p.
$$

By the chain rule, the derivative has a direct term and a term through the maximizing lengths where the latter vanishes because each inner objective has zero derivative with respect to length at its maximum. Now:

- If the lengths exceed the budget, i.e. $\frac{1}{n} \sum_i u_i(\widetilde{\lambda})^p > 1 \Leftrightarrow \sum_i u_i(\widetilde{\lambda})^p > n$, this derivative is negative, so increasing $\widetilde{\lambda}$ reduces $H$ (movement more expensive).
- If the lengths do not exceed the budget, i.e. $\sum_i u_i(\widetilde{\lambda})^p < n$, this derivative is positive, so decreasing $\widetilde{\lambda}$ reduces $H$ (movement cheaper).

At the optimal multiplier, $H'=0$ and the budget is met exactly. Thus the outer choice of multiplier enforces the original constraint that the individual inner problems remain unconstrained apart from nonnegativity. The code minimizes $H$ directly, and this derivative explains why that minimization finds the budget balance.

To find an interval containing the optimal multiplier, we use the reference lengths $u_i=1$, which satisfy $\frac1n\sum_i u_i^p=1$. This does not assume that the optimal lengths are uniform. Instead, we ask *which multiplier makes $u=1$ maximize each observation's penalized objective*. Differentiating its inner objective with respect to $u$, and setting the derivative to zero at $u=1$, gives

$$
\left.\frac{\partial}{\partial u}
\left[(a_i+bu)^2-\widetilde\lambda u^p\right]\right|_{u=1}
=2b(a_i+b)-p\widetilde\lambda=0.
$$

Thus observation $i$ chooses length one at the threshold

$$
\widetilde\lambda_i=\frac{2b(a_i+b)}p.
$$

Each observation has its own threshold $\widetilde\lambda_i$ at which its maximizing length equals one, but all observations share a single multiplier $\widetilde\lambda$. Below an observation's threshold, its maximizing length exceeds one and above its threshold, its maximizing length is less than one.

We therefore take $\widetilde\lambda_{\rm lo}=\min_i\widetilde\lambda_i$ and $\widetilde\lambda_{\rm hi}=\max_i\widetilde\lambda_i$:

```python
lambda_lo = 2 * b * (float(np.min(a)) + b) / self.p
lambda_hi = 2 * b * (float(np.max(a)) + b) / self.p
```

At the lower bound, movement is cheap, meaning observations whose threshold equals $\widetilde\lambda_{\rm lo}$ choose length one, and every other observation chooses a length greater than one. Thus the average cost $\frac1n\sum_i u_i^p$ is at least one. At the upper bound, movement is expensive, meaning observations whose threshold equals $\widetilde\lambda_{\rm hi}$ choose length one, and every other observation chooses a length less than one. The average cost is therefore at most one. Several observations may share either endpoint threshold.

Now imagine increasing the shared multiplier from the lower bound to the upper bound. Every maximizing length decreases continuously, so the average cost also decreases continuously. When the bounds differ, it starts above one and ends below one, i.e., it cannot get from one side to the other without passing through

$$
\frac1n\sum_i u_i(\widetilde\lambda)^p=1.
$$

This crossing is the budget balance we seek. It is unique because the average cost is strictly decreasing. As shown above, this is also where $H'(\widetilde\lambda)=0$, so it gives the minimizing multiplier. The individual lengths can differ from one -- some observations can use more movement and others less -- provided their average cost equals one.

If the bounds coincide, all observations have the same threshold (equivalently, all $a_i$ are equal), so that shared multiplier makes every maximizing length equal one. The budget is already balanced and no search is needed. In $H$, the outer term $n\widetilde\lambda$ then cancels the sum of the penalties $\sum_i\widetilde\lambda u_i^p=n\widetilde\lambda$, leaving $\sum_i(a_i+b)^2$. Multiplying by `scale**2` restores the summed risk in the original response units:

```python
if lambda_lo == lambda_hi:
    return float(np.sum((a + b) ** 2) * scale**2)
```

Otherwise, the code searches using the relative multiplier

$$
\rho=\frac{\widetilde\lambda}{\widetilde\lambda_{\rm hi}}.
$$

This makes the upper search bound one and measures the absolute search tolerance relative to `lambda_hi`.

```python
def objective(relative_lambda):
    lam = relative_lambda * lambda_hi
    return len(a) * lam + np.sum(self._maximize_t(a, b, lam))

result = minimize_scalar(
    objective,
    bounds=(lambda_lo / lambda_hi, 1),
    method="bounded",
    options={"xatol": 1e-10},
)
```

`SciPy`'s [`minimize_scalar`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html), with `method="bounded"`, numerically minimizes a function of one variable within the supplied bounds. Here that function is $H(\rho\widetilde\lambda_{\rm hi})$. Since $H$ is convex, any local minimum is also global. Each evaluation calls `_maximize_t` to compute the inner maxima described next.

### Compute the inner maxima

For a fixed trial multiplier $\widetilde{\lambda}$, `_maximize_t` computes

$$
\max_{u\geq0}\left[(a_i+bu)^2-\widetilde\lambda u^p\right]
$$

for every observation. These are independent problems since each observation has its own $a_i$, while $b$ and $\widetilde\lambda$ are shared.

The code uses minimization routines, so it changes the sign of the objective:

```python
def negative_objective(u, a):
    return lam * u**self.p - (a + b * u) ** 2
```

For $p>2$, $b>0$, and $\widetilde\lambda>0$, this function decreases to a unique positive minimum and then increases. We find it in two steps.

First, `SciPy`'s [`elementwise.bracket_minimum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.elementwise.bracket_minimum.html) starts from $u=1$ and finds a bracket containing the minimum, without yet locating the minimizer precisely. `xmin=0.0` keeps lengths nonnegative, and the array `a` gives a separate problem for each observation:

```python
bracket = elementwise.bracket_minimum(
    negative_objective, 1.0, xmin=0.0, args=(a,)
)
```

After checking that bracketing succeeded, we pass the brackets to [`elementwise.find_minimum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.elementwise.find_minimum.html), which locates each minimizer within its bracket to numerical tolerance:

```python
result = elementwise.find_minimum(
    negative_objective, bracket.bracket, args=(a,)
)
```

The array `result.f_x` contains the minimum values of the negative objectives. After checking convergence for every observation, `_maximize_t` changes their signs back:

```python
return -result.f_x
```

These are the inner maximum values, rather than the maximizing lengths. The outer `objective` sums them and adds $n\widetilde\lambda$, completing one evaluation of $H$.

Once the outer minimization succeeds, `_solve_finite_p` multiplies `result.fun` by `scale**2` to return the summed risk in the original response units. A failed bracket search, inner minimization, or outer minimization raises an error.

## Return perturbation lengths for eta updates

`_ScalarRisk.solve(a, B)` returns the summed risk as before. With `return_t=True`, it returns `(value, t_star)`, including the maximizing perturbation lengths. The eta solver uses this option with its smoothed residual magnitudes and coefficient norm.

`RobustRisk.primal(beta, return_t=True)` also returns `(value, t_star)`. The `per_sample` option changes only the value's normalization, not the lengths. Both interfaces reuse the existing scalar minimization; there is no separate solver for the lengths.

## Gamma formulation

For an independent evaluation, `dual` uses Theorem 6, Eq. (22):

$$
nV_\delta(\beta)=\max_{\gamma\geq0}K(\beta,\gamma),\qquad
K(\beta,\gamma)=n^{1/p}\delta B\left(\sum_i\gamma_i\right)^{1/q}
+\sum_i |r_i|\gamma_i^{1/q}-\frac14\sum_i\gamma_i^{2/q}.
$$

Here $K(\beta,\gamma)=K(\beta,\gamma^{1/q})$ in the paper's notation. Since $1/q\leq1$ and $2/q\geq1$, the first two terms are concave in $\gamma$ and the last is negative convex. For fixed $\beta$, `CVXPY` can maximize this concave expression directly:

```python
gamma = cp.Variable(self.n, nonneg=True)
objective = (
    self.n ** (1 / self.p) * self.delta * B
    * cp.power(cp.sum(gamma), 1 / self.q, max_denom=65536)
    + residual_abs @ cp.power(gamma, 1 / self.q, max_denom=65536)
    - cp.sum(cp.power(gamma, 2 / self.q, max_denom=65536)) / 4
)
problem = cp.Problem(cp.Maximize(objective))
problem.solve()
```

The formula extends to $p=\infty$ by setting $q=1$ and $n^{1/p}=1$ -- each gamma maximum then equals $(|r_i|+\delta B)^2$. Both evaluators therefore include that endpoint. `CVXPY` approximates noninteger powers rationally (`max_denom=65536`), so primal-dual agreement is a numerical cross-check. The [tests](../tests/robust_risk_test.py) also compare both methods with the explicit formulas above.
