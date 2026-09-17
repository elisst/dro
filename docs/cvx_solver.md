# Minimize the robust risk with CVXPY

For a given Wasserstein radius $\delta\geq0$, `CvxOptimizer` finds a coefficient vector minimizing $V_\delta(\beta)$. From Theorem 1, Eq. (5), we have
$$
\min_{\beta\in\mathbb R^d} V_\delta(\beta),
\qquad
V_\delta(\beta)
=\max_{t_i\geq0:\,\sum_i t_i^p\leq n\delta^p}
\frac1n\sum_{i=1}^n\bigl(|r_i|+t_i\|\beta\|_*\bigr)^2,
$$
where $r_i=x_i^\top\beta-y_i$, $2\leq p<\infty$, and $\|\cdot\|_*$ is dual to the ground norm $\|\cdot\|$.

`RobustRisk` evaluates this worst-case loss at a supplied $\beta$. `CvxOptimizer` takes that risk object and chooses $\beta$. It uses the stored data and parameters to build a convex minimization over $\beta$ and auxiliary variables. It calls the scalar evaluator afterward to check the fitted risk.

## Fit and cross-check

The following example fits the coefficients and evaluates their risk.

```python
import numpy as np
from DRO.cvx_solver import CvxOptimizer
from DRO.robust_risk import RobustRisk

# draw some data using beta_star
rng = np.random.default_rng(7)
X = rng.normal(size=(20, 3))  # rows are observations and columns are features
beta_star = np.array([1.0, -0.5, 0.0])
y = X @ beta_star + 0.2 * rng.normal(size=20)

# minimize over beta
delta, p = 0.15, 3
risk = RobustRisk.normalized(X, y, delta, p, norm=np.inf)
optimizer = CvxOptimizer(risk)
fit = optimizer.minimize()

print(fit.beta)  # fitted coefficients, shape (d,)
print("Scalar risk", fit.value)
print("CVXPY risk", fit.diagnostics["model_value"])

# this repeats the scalar evaluation already used for fit.value
print(risk.primal(fit.beta))
```

The returned `CvxResult` contains `beta`, `value`, and `diagnostics`. After `CVXPY` finds `fit.beta`, the solver computes the risk in two ways and checks that they agree.

- `fit.value` comes from evaluating `RobustRisk.primal` at `fit.beta`. This uses the scalar reduction in the risk evaluator guide. Use this value when reporting the fitted risk or comparing fits.
- `fit.diagnostics["model_value"]` comes from `CVXPY`'s optimized objective in the convex minimization derived below, converted to mean squared loss. Keeping it lets you inspect what `CVXPY` returned and compare it with the scalar evaluation without solving again.
- `fit.diagnostics["scalar_value"]` is exactly the same number as `fit.value`. It is included so that saving the diagnostics dictionary alone preserves both sides of the comparison. It adds no new numerical information.

See below for more information on different diagnostics.

The instance holds the supplied `RobustRisk` object. Every call to `.minimize()` builds a fresh convex model and leaves that risk object unchanged. You can also pass data and parameters directly to `minimize_cvx`, which constructs the risk object and calls `CvxOptimizer(risk).minimize()`.

```python
from DRO.cvx_solver import minimize_cvx

fit = minimize_cvx(X, y, radius=delta, p=p, norm=np.inf)
# equivalent to CvxOptimizer(risk).minimize() for the risk defined above
```

Inputs are finite arrays `X` of shape `(n, d)` and `y` of shape `(n,)`, with `n, d > 0`. Use a finite radius $\delta\geq0$, $2\leq p\leq\infty$, and `norm` equal to `1`, `2`, or `np.inf`. The solver assumes these input conditions. No intercept is added automatically. The `norm` argument to `RobustRisk` or `minimize_cvx` chooses the ground norm in the transport cost $c(x,x')=\|x-x'\|^p$, separately from the Wasserstein order `p`.

## Radius and returned units

`CvxOptimizer(risk)` uses the radius stored in `risk.delta`. Prefer `RobustRisk.normalized(X, y, delta, p)` to specify the Wasserstein radius $\delta$. The standalone function `minimize_cvx(X, y, radius=delta, p=p)` takes this same radius. For finite $p$, the paper's convention is $V_\delta(\beta)=\bar V_{\delta^p}(\beta)$, and the direct constructor `RobustRisk(X, y, delta**p, p)` accepts the average movement cost $\delta^p$. Either constructor stores the radius in `risk.delta`, so both work with `CvxOptimizer`. At $p=\infty$, both take the radius directly.

Both the evaluator and the solver return the **mean robust squared loss** $V_\delta(\beta)$. For a fit, this is evaluated at the returned coefficients $\widehat{\beta}$.

## Connecting risk and code

The implementation has two levels, corresponding to `RobustRisk` and its internal `_ScalarRisk` helper.

- `CvxOptimizer` reads the supplied risk object and handles response scaling and returned units.
- `_CvxProblem` builds the mathematical model, solves it with `CVXPY`, and checks the numerical solution.

`CvxResult` is only the container for the output. The main call structure is

```text
CvxOptimizer.minimize(...)
    _response_scale(fixed_beta)
    _CvxProblem(scaled_risk, ...)    # a separate risk object in scaled units
        _build_objective()
            _finite_p_objective()   # only for delta > 0 and 2 < p < infinity
                _quadratic_upper_bound(...)
                    _square_over_linear(...) twice
    _CvxProblem.solve(...)
        _check_solution(...)
            RobustRisk.primal(...)  # independent scalar evaluation
            _coefficient_optimality(...)  # only when beta is fitted
    _restore_units(...)
```

To explain the model, first work in the original response units. `_CvxProblem` introduces residual variables $r=X\beta-y$ and a scalar $B\geq\|\beta\|_*$. Increasing $B$ cannot decrease the worst-case loss, so taking $B=\|\beta\|_*$ always attains the best value for fixed $\beta$. The norm bound therefore preserves the minimum. Keeping these constraints explicit also provides the multipliers used to check coefficient optimality.

`_build_objective` chooses among the following cases. Write $J$ for the objective supplied to `CVXPY`.

| Case, in code order | Objective $J$ | Mean risk from the model |
| --- | --- | --- |
| $\delta=0$ | $\sum_i r_i^2$ | $J/n$ |
| $p=2$ | $\|r\|_2+\sqrt n\,\delta B$ | $(J/\sqrt n)^2$ |
| $p=\infty$ | $\sum_i(\lvert r_i\rvert+\delta B)^2$ | $J/n$ |
| $2<p<\infty$ and $\delta>0$ | $\sum_i u_i+n^{2/p}\|w_{\mathrm{scaled}}\|_{p/(p-2)}$ with the constraints below | $J/n$ |

## Formulas for explicit subcases

### Zero radius

At $\delta=0$, every movement length is zero. Substituting in Eq. (5) gives
$$
nV_0(\beta)=\sum_i r_i^2.
$$
Thus the solver minimizes ordinary squared loss, for every $p$.

```python
if delta == 0:
    return cp.sum_squares(r)
```

### Quadratic exponent

For $p=2$, Proposition 2 gives
$$
V_\delta(\beta)
=\left(\frac{\|X\beta-y\|_2}{\sqrt n}+\delta\|\beta\|_*\right)^2.
$$
The [robust risk doc](robust_risk.md#quadratic-exponent) derives this formula by aligning the movement lengths with the residual magnitudes. The expression inside the square is nonnegative, so minimizing it or its square gives the same coefficients. Multiplying it by $\sqrt n$ gives the model objective. With the default ground norm, this is square-root Lasso.

```python
if p == 2:
    # minimizing a nonnegative function or its square gives the same beta
    return cp.norm(r, 2) + np.sqrt(n) * delta * B
```

To recover the mean squared risk from the optimized objective $J$, compute $(J/\sqrt n)^2$.

### Infinite exponent

At $p=\infty$, each observation can move the full distance $\delta$. The loss is nondecreasing in each length, so choosing every $t_i=\delta$ and taking $B=\|\beta\|_*$ gives
$$
nV_\delta(\beta)=\sum_i(|r_i|+\delta B)^2.
$$
This is the adversarial-training endpoint in Proposition 2. We retain the mean normalization from Eq. (5). The printed infinity expression in Proposition 2 omits $1/n$.

```python
if np.isinf(p):
    return cp.sum_squares(cp.abs(r) + delta * B)
```

## Formulating a convex objective for intermediate exponents

If we express the risk $V_\delta(\beta)$ once again using Theorem 1, Eq. (5), the minimization problem over $\beta$ becomes
$$
\widehat{\beta} = \operatorname*{arg\,min}_\beta V_\delta(\beta)
=\operatorname*{arg\,min}_\beta \left\{ \max_{t_i\geq0:\,\sum_i t_i^p\leq n\delta^p}
\frac1n\sum_{i=1}^n\bigl(|r_i|+t_i\|\beta\|_*\bigr)^2 \right\}.
$$
For $2<p<\infty$ and $\delta>0$, the substitution $t_i^p = \sigma_i$ makes the inner risk evaluation a concave maximization with an affine budget when $\beta$ is fixed. However, fitting $\beta$ still requires an outer minimization.

In the following steps, we eliminate the inner maximization to obtain one convex program that is compatible with `CVXPY`.

### The outer minimization is not directly compatible with CVXPY

The outer objective is already convex in $\beta$. For every fixed feasible $t$, each $|r_i|+t_i\|\beta\|_*$ is nonnegative and convex, so its square is convex. Taking the pointwise maximum over $t$ preserves convexity. The remaining task is to express this convex function in a form `CVXPY` can use.

This nested min–max expression is not directly supported by `CVXPY`'s standard DCP interface. We therefore eliminate the inner maximization and obtain an equivalent convex formulation that `CVXPY` can convert to a conic program.

### Quadratic upper bounds

Fix $\beta$ and introduce $B\geq\|\beta\|_*$. Increasing $B$ cannot decrease the loss, so minimizing over this bound preserves the original minimum. Writing $a_i=|r_i|$ and suppressing the index, the loss for one observation is
$$
(a+Bt)^2=a^2+2aBt+B^2t^2.
$$
If we could remove the linear term, then we would have something only dependent on squared movement $t^2$, which is easier to optimize. We therefore seek an upper bound of the form $u+wt^2$ valid for all $t$.

Our goal is to choose a valid upper bound whose worst-case value (maximum) is as small as possible. Every bound gives an upper estimate of the risk, so to recover the risk exactly we must show that at least one bound has the same maximum as the loss.

### Converting to a concave problem

For fixed $\beta,B$, the original loss is convex in $t$, so maximizing it is not a convex optimization problem in that representation. For this reformulation, use squared movement,
$$
z_i=t_i^2.
$$
The mean loss and feasible set become
$$
F(z)=\frac1n\sum_i\left(a_i^2+2a_iB\sqrt{z_i}+B^2z_i\right),
\qquad
\mathcal Z=\left\{z\geq0:\sum_i z_i^{p/2}\leq n\delta^p\right\}.
$$
Now $F$ is concave since it is a sum of constant, square-root, and linear terms with nonnegative coefficients. The set $\mathcal Z$ is convex because $p/2>1$.

From Boyd and Vandenberghe's *Convex Optimization*, §3.1.3, pp. 69–70, since $F$ is concave and differentiable when all coordinates are positive, we have that
$$
F(z)\leq F(s)+\nabla F(s)^\top(z-s)=:H_s(z)
$$
at any reference point $s$ with strictly positive coordinates. The tangent plane $H_s$ is an affine upper bound on the loss and equals it at $s$. The inequality extends to feasible $z$ with zero coordinates by continuity. Since $z_i=t_i^2$, an affine bound in $z$ is exactly the quadratic bound in movement that we wanted.

### Showing one bound has the exact worst-case value

A maximizer $z^*$ exists because $\mathcal Z$ is compact and $F$ is continuous. In addition to the concavity inequality, constrained optimality gives
$$
\nabla F(z^*)^\top(z-z^*)\leq0
\qquad\text{for every }z\in\mathcal Z.
$$
This is the maximization version of the criterion in Boyd's §4.2.3, Eq. (4.21). If the inner product were positive for some feasible $z$, moving a little from $z^*$ toward $z$ would increase the loss. The move would remain feasible because $\mathcal Z$ is convex, contradicting maximality. Note we do not require $\nabla F(z^*)=0$ (the maximizer lies on the budget boundary when $B>0$).

Combining concavity with optimality gives
$$
F(z)\leq H_{z^*}(z)\leq F(z^*)
\qquad\text{for every }z\in\mathcal Z.
$$
At $z=z^*$, both inequalities are equalities. Thus the loss and this tangent bound have the same maximum, attained where they touch. We use $z^*$ to prove that an exact bound exists. The solver will choose the bound without knowing $z^*$ beforehand.

When $B>0$, every coordinate of $z^*$ is positive, so the gradient used above exists. If $B=0$, the loss is constant in movement and a constant bound is exact.

### Write the tangent coefficients explicitly

For one observation, write $f_i(z_i)=a_i^2+2a_iB\sqrt{z_i}+B^2z_i$. Choose a contact movement $\tau_i>0$, so the contact point in squared movement is $z_i=\tau_i^2$. The tangent inequality is
$$
\begin{aligned}
f_i(z_i)
&\leq f_i(\tau_i^2)+f_i'(\tau_i^2)(z_i-\tau_i^2)\\
&=\underbrace{a_i^2+a_iB\tau_i}_{u_i}
+\underbrace{\left(B^2+\frac{a_iB}{\tau_i}\right)}_{w_i}z_i.
\end{aligned}
$$
Increasing the contact movement raises the constant $u_i$ and lowers the coefficient $w_i$ when $a_iB>0$. In the original movement variable, this gives the bound from before,
$$
(a_i+Bt_i)^2\leq u_i+w_it_i^2.
$$

### Let the solver choose the coefficients

We now describe valid bounds directly through $u,w$, without introducing the unknown contact movements as decision variables. Impose
$$
\frac{a_i^2}{u_i}+\frac{B^2}{w_i}\leq1,
\qquad u_i,w_i\geq0.
$$
Cauchy–Schwarz $(c_1d_1+c_2d_2)^2\leq(c_1^2+c_2^2)(d_1^2+d_2^2)$ proves that every permitted pair gives an upper bound.
$$
(a_i+Bt_i)^2 = \left( \frac{a_i}{\sqrt{u_i}}\sqrt{u_i} + \frac{B}{\sqrt{w_i}}\sqrt{w_i}t_i \right)^2
\leq \underbrace{\left(\frac{a_i^2}{u_i}+\frac{B^2}{w_i}\right)}_{\leq 1}(u_i+w_it_i^2)
\leq u_i+w_it_i^2.
$$
The tangent coefficients above are included, because whenever $a_i+B\tau_i>0$,
$$
\frac{a_i^2}{u_i}+\frac{B^2}{w_i}
=\frac{a_i}{a_i+B\tau_i}+\frac{B\tau_i}{a_i+B\tau_i}=1.
$$
All quotients use the closed convex convention, with $0/0=0$ and a positive numerator over zero taken as infinite. If $a_i=B=0$, take $u_i=w_i=0$.

Every allowed bound is valid, and the exact tangent bound is among them. Minimizing their worst-case values therefore recovers the risk. Moreover, $a_i^2=r_i^2$, so each constraint is a sum of two square-over-linear functions. It remains convex when $\beta,B,u,w$ are optimized jointly.

### Evaluate the remaining linear maximization

To recap, for a fixed $\beta$ and $B=\|\beta\|_*$, we started from
$$
V_\delta(\beta)
=\max_{t\geq0:\,\sum_i t_i^p\leq n\delta^p}
\frac1n\sum_i(a_i+Bt_i)^2.
$$
We then set $z_i=t_i^2$ and replaced each loss by a bound $u_i+w_i z_i$. Every allowed bound lies above the loss, and we proved that one of them has exactly the same worst-case value. Thus, choosing the best bound gives the equivalent problem
$$
V_\delta(\beta)
=\min_{\substack{u,w\geq0\\a_i^2/u_i+B^2/w_i\leq1\ \text{for all }i}}
\max_{z\in\mathcal Z}
\frac1n\left(\sum_i u_i+w^\top z\right).
$$
Here the minimization chooses a valid bound, and the maximization finds its worst-case value over the movement budget. Fitting $\beta$ adds an outer minimization, which can be combined with the minimization over $u,w$ once we evaluate the remaining maximum.

The benefit is that, for chosen $u,w$, the term $\sum_i u_i$ is constant in $z$, and all movement dependence is now linear.
$$
\max_{z\in\mathcal Z}\frac1n\left(\sum_i u_i+w^\top z\right)
=\frac1n\sum_i u_i+\frac1n\max_{z\in\mathcal Z}w^\top z.
$$
So the remaining question is how large the weighted sum $w^\top z$ can become under the budget. Maximizing a linear function over a norm ball gives the radius times the dual norm of its coefficient vector. We now apply that fact to our budget, verifying both the upper bound and that it can be attained.

Let $k=p/(p-2)$ be the Hölder conjugate of $p/2$, so that
$$
\frac{1}{p/2}+\frac{1}{k}=1.
$$
The budget is a norm ball.
$$
\sum_i z_i^{p/2}\leq n\delta^p
\quad\Longleftrightarrow\quad
\|z\|_{p/2}\leq R,\qquad R=n^{2/p}\delta^2.
$$
Hölder's inequality gives $w^\top z\leq\|w\|_k\|z\|_{p/2}\leq R\|w\|_k$. For $w\ne0$, equality is attained at
$$
z_i=R\left(\frac{w_i}{\|w\|_k}\right)^{k-1}.
$$
To verify this, we check that the vector is feasible and reaches the upper bound. Since $w_i\geq0$, it is nonnegative. Also, $(k-1)p/2=k$, so
$$
\sum_i z_i^{p/2}
=R^{p/2}\sum_i\left(\frac{w_i}{\|w\|_k}\right)^{(k-1)p/2}
=R^{p/2}\frac{\sum_i w_i^k}{\|w\|_k^k}
=R^{p/2}.
$$
Thus $\|z\|_{p/2}=R$, i.e., the vector uses exactly the available budget. Its objective value is
$$
w^\top z
=\frac{R\sum_i w_i^k}{\|w\|_k^{k-1}}
=R\|w\|_k.
$$
If $w=0$, the maximum of $w^\top z$ is zero. Hence
$$
\max_{z\in\mathcal Z}\frac1n\left(\sum_i u_i+w^\top z\right)
=\frac1n\sum_i u_i+\delta^2n^{2/p-1}\|w\|_k.
$$
There is no movement maximization left.

### The resulting convex program

Combining the objective and the constraints gives
$$
\begin{aligned}
\min_{\beta,r,B,u,w}\quad &
\frac1n\sum_i u_i+\delta^2n^{2/p-1}\|w\|_k\\
\text{subject to}\quad &r=X\beta-y,\qquad B\geq\|\beta\|_*,\\
&u,w\geq0,\qquad
\frac{r_i^2}{u_i}+\frac{B^2}{w_i}\leq1,\quad i=1,\ldots,n.
\end{aligned}
$$
The constraints select valid quadratic upper bounds and the objective evaluates their worst-case value. An exact tangent bound is feasible, so this program has the same minimum as the original robust risk. Its objective is a sum plus a norm, and its constraints are jointly convex square-over-linear inequalities.

For example, when $p=4$, we have $k=2$ and the objective is $\sum_i u_i/n+\delta^2\|w\|_2/\sqrt n$.

### Match the model to the code

The implementation uses the scaled slope $w_{\mathrm{scaled}}=\delta^2w$. This keeps the movement contribution in loss units and avoids putting $\delta^2$ in the objective coefficient. The same model is then
$$
\min\quad\frac1n\sum_i u_i+n^{2/p-1}\|w_{\mathrm{scaled}}\|_k,
\qquad
\frac{r_i^2}{u_i}+\frac{(\delta B)^2}{(w_{\mathrm{scaled}})_i}\leq1.
$$
The other constraints are unchanged. `_finite_p_objective` minimizes $n$ times this mean-loss objective.

```python
u, w_scaled = (cp.Variable(n, name=name) for name in ("u", "w_scaled"))
k = p / (p - 2)
w_norm = cp.pnorm(w_scaled, k, approx=False)
self.constraints += _quadratic_upper_bound(r, delta * B, u, w_scaled)
return cp.sum(u) + n**(2 / p) * w_norm
```

The helper `_quadratic_upper_bound` represents the sum-of-quotients constraint using second-order cones, including cases where a denominator is zero. The implementation uses `approx=False` to represent the $k$-norm without approximating its exponent.

## Scale the calculation and check the solution

### Use common response and coefficient units

`CvxOptimizer._response_scale` chooses a scale $s>0$. A separate `RobustRisk` object with responses $y/s$ is passed to `_CvxProblem`, leaving the supplied object unchanged. Its coefficient variable represents $\beta/s$. The design matrix, radius, and ground norm remain the same, because
$$
|x_i^\top(\beta/s)-y_i/s|+t_i\|\beta/s\|_*
=\frac1s\left(|r_i|+t_i\|\beta\|_*\right).
$$
Thus squared risks scale by $1/s^2$. Usually $s$ is the root mean square of `y`. With `fixed_beta`, it also accounts for the predictions and $\delta\|\beta\|_*$. If all these quantities vanish, the code uses $s=1$.

After the solve and checks, `_restore_units` multiplies coefficients by $s$ and squared risks by $s^2$. The multiplication is done as `(value * scale) * scale` to avoid overflowing `scale**2` when the final risk is representable.

### Compare the independent risk values

`_CvxProblem.solve` first requires `CVXPY` status `optimal`. `_check_solution` then compares the convex-model value with `RobustRisk.primal` at the returned coefficients. The latter uses the scalar reduction in Lemma 1, Eq. (14), with the inner problem studied after Eq. (35). `_check_solution` also checks the numerical feasibility of every model constraint.

Both values are computed in scaled units. For example, the value comparison is

```python
value_error = abs(scalar_value - model_value) / max(1.0, abs(scalar_value))
```

This mixes absolute and relative error. A small scaled risk need not have a small *relative* error, and agreement at one coefficient vector does not alone establish that those coefficients minimize the risk.

For a numerical cross-check at a supplied coefficient vector of shape `(d,)`, use `fixed_beta`. This holds the coefficients constant while solving the convex model. The risk-value and feasibility checks still run, but the coefficient optimality check is skipped.

```python
beta = np.array([0.8, -0.4, 0.1])
evaluation = optimizer.minimize(fixed_beta=beta)
print(evaluation.value, evaluation.diagnostics["model_value"])
```

### Check coefficient optimality

Agreement between the two risk evaluations checks the objective value at the returned coefficients, but it does not establish that those coefficients minimize the risk. For fitted coefficients, `_coefficient_optimality` therefore also checks a first-order optimality condition, namely that the effects of the residuals and the coefficient norm must balance.

This check uses multipliers returned by the solver and rejects solutions whose balance error exceeds the numerical tolerance. It is skipped when `fixed_beta` is supplied, because those coefficients are prescribed rather than optimized.

### Read the diagnostics and choose a solver

The solver defaults to `CLARABEL` and also supports `SCS`. The requested solver tolerance defaults to `1e-9`, and the additional checks use `check_tolerance=2e-6`. Both tolerances are assumed finite and positive.

```python
second = optimizer.minimize(solver="SCS", tolerance=1e-8)
print(fit.value, second.value)
print(np.linalg.norm(X @ (fit.beta - second.beta)))
print(second.diagnostics)
```

Comparing predictions is useful when minimizing coefficients need not be unique. The table below explains the diagnostics and their units.

| Entry | Meaning and units |
| --- | --- |
| `model_value` | `CVXPY` objective converted to mean squared loss in the original response units. Retained for comparison with `fit.value` |
| `scalar_value` | Exact duplicate of `fit.value`, so the diagnostics dictionary also contains the scalar side of the comparison |
| `response_scale` | Response/coefficient scale $s$ |
| `scaled_value_error` | Absolute difference of scaled mean risks, divided by $\max(1,\lvert\text{scaled scalar risk}\rvert)$ |
| `scaled_constraint_violation` | Largest `CVXPY` constraint violation in the scaled model |
| `scaled_stationarity_error` | Normalized excess of the residual contribution over what the coefficient-norm term can balance, in scaled units |
| `scaled_complementarity_error` | Normalized mismatch in the required balance along the fitted coefficient vector, in scaled units |

The last two entries are present only for fitted coefficients. `solver`, `status`, `iterations`, and both requested tolerances are also recorded. These are numerical consistency and optimality checks, not exact-arithmetic certificates or guarantees of relative coefficient accuracy.

Rejected statuses and failed numerical checks raise `RuntimeError`. Unsupported floating-point scales can raise `FloatingPointError`, and underlying solver errors propagate. A failed call returns no fit.
