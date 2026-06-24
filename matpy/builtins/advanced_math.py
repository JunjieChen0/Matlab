"""Advanced math built-in functions for MatPy (P0 priority)."""

import numpy as np
from scipy import optimize, integrate, signal, stats
from scipy import linalg as scipy_linalg
from matpy.builtins import register
from matpy.runtime.types import Mat


# ── Optimization ──────────────────────────────────────────────

@register("fzero")
def _fzero(func, x0, *args):
    if callable(func):
        result = optimize.fsolve(func, float(x0), args=args)
        return float(result[0])
    raise RuntimeError("fzero: first argument must be a function")

@register("fminsearch")
def _fminsearch(func, x0, *args):
    if callable(func):
        result = optimize.minimize(func, float(x0), method='Nelder-Mead', args=args)
        return float(result.x[0])
    raise RuntimeError("fminsearch: first argument must be a function")

@register("fminbnd")
def _fminbnd(func, a, b, *args):
    if callable(func):
        result = optimize.minimize_scalar(func, bounds=(float(a), float(b)), method='bounded', args=args)
        return float(result.x)
    raise RuntimeError("fminbnd: first argument must be a function")

@register("fmincon")
def _fmincon(func, x0, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None, *args):
    if callable(func):
        x0 = np.array(x0, dtype=float).flatten()
        constraints = []
        if A is not None and b is not None:
            A = np.array(A, dtype=float)
            b = np.array(b, dtype=float).flatten()
            constraints.append({'type': 'ineq', 'fun': lambda x: b - A @ x})
        if Aeq is not None and beq is not None:
            Aeq = np.array(Aeq, dtype=float)
            beq = np.array(beq, dtype=float).flatten()
            constraints.append({'type': 'eq', 'fun': lambda x: beq - Aeq @ x})
        bounds = None
        if lb is not None or ub is not None:
            lb = float(lb) if lb is not None else None
            ub = float(ub) if ub is not None else None
            bounds = [(lb, ub)] * len(x0)
        result = optimize.minimize(func, x0, method='SLSQP', bounds=bounds, constraints=constraints, args=args)
        return Mat(result.x)
    raise RuntimeError("fmincon: first argument must be a function")

@register("linprog")
def _linprog(c, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None):
    c = np.array(c, dtype=float).flatten()
    A_ub = np.array(A, dtype=float) if A is not None else None
    b_ub = np.array(b, dtype=float).flatten() if b is not None else None
    A_eq = np.array(Aeq, dtype=float) if Aeq is not None else None
    b_eq = np.array(beq, dtype=float).flatten() if beq is not None else None
    bounds = None
    if lb is not None or ub is not None:
        lb_val = float(lb) if lb is not None else None
        ub_val = float(ub) if ub is not None else None
        bounds = [(lb_val, ub_val)] * len(c)
    result = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    return Mat(result.x)


# ── ODE Solvers ───────────────────────────────────────────────

@register("ode45")
def _ode45(func, tspan, y0, *args):
    from scipy.integrate import solve_ivp
    tspan = np.array(tspan, dtype=float)
    y0 = np.array(y0, dtype=float).flatten()
    t_eval = np.linspace(tspan[0], tspan[1], 100)
    result = solve_ivp(func, tspan, y0, t_eval=t_eval, args=args, method='RK45')
    return Mat(result.t), Mat(result.y)

@register("ode23")
def _ode23(func, tspan, y0, *args):
    from scipy.integrate import solve_ivp
    tspan = np.array(tspan, dtype=float)
    y0 = np.array(y0, dtype=float).flatten()
    t_eval = np.linspace(tspan[0], tspan[1], 100)
    result = solve_ivp(func, tspan, y0, t_eval=t_eval, args=args, method='RK23')
    return Mat(result.t), Mat(result.y)

@register("ode15s")
def _ode15s(func, tspan, y0, *args):
    from scipy.integrate import solve_ivp
    tspan = np.array(tspan, dtype=float)
    y0 = np.array(y0, dtype=float).flatten()
    t_eval = np.linspace(tspan[0], tspan[1], 100)
    result = solve_ivp(func, tspan, y0, t_eval=t_eval, args=args, method='BDF')
    return Mat(result.t), Mat(result.y)


# ── Integration ───────────────────────────────────────────────

@register("integral")
def _integral(func, a, b, *args):
    if callable(func):
        result, _ = integrate.quad(func, float(a), float(b), args=args)
        return float(result)
    raise RuntimeError("integral: first argument must be a function")

@register("integral2")
def _integral2(func, a, b, c, d, *args):
    if callable(func):
        result, _ = integrate.dblquad(func, float(a), float(b), float(c), float(d), args=args)
        return float(result)
    raise RuntimeError("integral2: first argument must be a function")

@register("quad")
def _quad(func, a, b, *args):
    return _integral(func, a, b, *args)


# ── Interpolation ─────────────────────────────────────────────

@register("spline")
def _spline(x, y, xq):
    from scipy.interpolate import CubicSpline
    xd = x.data if isinstance(x, Mat) else np.array(x).flatten()
    yd = y.data if isinstance(y, Mat) else np.array(y).flatten()
    xqd = xq.data if isinstance(xq, Mat) else np.array(xq).flatten()
    cs = CubicSpline(xd, yd)
    return Mat(cs(xqd))

@register("pchip")
def _pchip(x, y, xq):
    from scipy.interpolate import PchipInterpolator
    xd = x.data if isinstance(x, Mat) else np.array(x).flatten()
    yd = y.data if isinstance(y, Mat) else np.array(y).flatten()
    xqd = xq.data if isinstance(xq, Mat) else np.array(xq).flatten()
    f = PchipInterpolator(xd, yd)
    return Mat(f(xqd))

@register("mkpp")
def _mkpp(breaks, coefs):
    b = breaks.data if isinstance(breaks, Mat) else np.array(breaks).flatten()
    c = coefs.data if isinstance(coefs, Mat) else np.array(coefs)
    return (b, c)

@register("ppval")
def _ppval(pp, x):
    from scipy.interpolate import PPoly
    breaks, coefs = pp
    xd = x.data if isinstance(x, Mat) else np.array(x).flatten()
    poly = PPoly(coefs.T, breaks)
    return Mat(poly(xd))


# ── Polynomials ───────────────────────────────────────────────

@register("polyfit")
def _polyfit(x, y, n):
    xd = x.data if isinstance(x, Mat) else np.array(x).flatten()
    yd = y.data if isinstance(y, Mat) else np.array(y).flatten()
    coeffs = np.polyfit(xd, yd, int(n))
    return Mat(coeffs)

@register("polyder")
def _polyder(p):
    pd = p.data if isinstance(p, Mat) else np.array(p).flatten()
    return Mat(np.polyder(pd))

@register("polyint")
def _polyint(p, k=0):
    pd = p.data if isinstance(p, Mat) else np.array(p).flatten()
    return Mat(np.polyint(pd, k=float(k)))

@register("residue")
def _residue(b, a):
    bd = b.data if isinstance(b, Mat) else np.array(b).flatten()
    ad = a.data if isinstance(a, Mat) else np.array(a).flatten()
    r, p, k = signal.residue(bd, ad)
    return Mat(r), Mat(p), Mat(k)


# ── Statistics ────────────────────────────────────────────────

@register("mean")
def _mean(x, dim=None):
    data = x.data if isinstance(x, Mat) else np.array(x)
    if dim is not None:
        return Mat(np.mean(data, axis=int(dim) - 1))
    return float(np.mean(data))

@register("median")
def _median(x, dim=None):
    data = x.data if isinstance(x, Mat) else np.array(x)
    if dim is not None:
        return Mat(np.median(data, axis=int(dim) - 1))
    return float(np.median(data))

@register("std")
def _std(x, flag=0, dim=None):
    data = x.data if isinstance(x, Mat) else np.array(x)
    ddof = 0 if flag else 1
    if dim is not None:
        return Mat(np.std(data, axis=int(dim) - 1, ddof=ddof))
    return float(np.std(data, ddof=ddof))

@register("var")
def _var(x, flag=0, dim=None):
    data = x.data if isinstance(x, Mat) else np.array(x)
    ddof = 0 if flag else 1
    if dim is not None:
        return Mat(np.var(data, axis=int(dim) - 1, ddof=ddof))
    return float(np.var(data, ddof=ddof))

@register("cov")
def _cov(x, y=None):
    xd = x.data if isinstance(x, Mat) else np.array(x)
    if y is not None:
        yd = y.data if isinstance(y, Mat) else np.array(y)
        return Mat(np.cov(xd.flatten(), yd.flatten()))
    return Mat(np.cov(xd))

@register("corrcoef")
def _corrcoef(x, y=None):
    xd = x.data if isinstance(x, Mat) else np.array(x)
    if y is not None:
        yd = y.data if isinstance(y, Mat) else np.array(y)
        return Mat(np.corrcoef(xd.flatten(), yd.flatten()))
    return Mat(np.corrcoef(xd))

@register("mode")
def _mode(x, dim=None):
    data = x.data if isinstance(x, Mat) else np.array(x)
    from scipy import stats as sp_stats
    if dim is not None:
        axis = int(dim) - 1
        result = sp_stats.mode(data, axis=axis, keepdims=True)
        return Mat(result.mode)
    result = sp_stats.mode(data.flatten(), keepdims=True)
    return float(result.mode[0])

@register("prctile")
def _prctile(x, p):
    data = x.data if isinstance(x, Mat) else np.array(x).flatten()
    p_val = p.data if isinstance(p, Mat) else np.array(p)
    return Mat(np.percentile(data, p_val))

@register("quantile")
def _quantile(x, q):
    data = x.data if isinstance(x, Mat) else np.array(x).flatten()
    q_val = q.data if isinstance(q, Mat) else np.array(q)
    return Mat(np.quantile(data, q_val))

@register("regress")
def _regress(y, X):
    yd = y.data if isinstance(y, Mat) else np.array(y).flatten()
    Xd = X.data if isinstance(X, Mat) else np.array(X)
    if Xd.ndim == 1:
        Xd = Xd.reshape(-1, 1)
    Xd = np.column_stack([np.ones(len(yd)), Xd])
    coeffs = np.linalg.lstsq(Xd, yd, rcond=None)[0]
    return Mat(coeffs)

@register("ttest")
def _ttest(x, mu=0):
    data = x.data if isinstance(x, Mat) else np.array(x).flatten()
    t_stat, p_value = stats.ttest_1samp(data, float(mu))
    return float(t_stat), float(p_value)

@register("ttest2")
def _ttest2(x, y):
    xd = x.data if isinstance(x, Mat) else np.array(x).flatten()
    yd = y.data if isinstance(y, Mat) else np.array(y).flatten()
    t_stat, p_value = stats.ttest_ind(xd, yd)
    return float(t_stat), float(p_value)

@register("anova1")
def _anova1(*args):
    groups = []
    for arg in args:
        data = arg.data if isinstance(arg, Mat) else np.array(arg).flatten()
        groups.append(data)
    f_stat, p_value = stats.f_oneway(*groups)
    return float(f_stat), float(p_value)

@register("chi2gof")
def _chi2gof(x, nbins=10):
    data = x.data if isinstance(x, Mat) else np.array(x).flatten()
    observed, bins = np.histogram(data, bins=int(nbins))
    expected = np.ones_like(observed) * len(data) / int(nbins)
    chi2, p_value = stats.chisquare(observed, expected)
    return float(chi2), float(p_value)


# ── Linear Algebra ────────────────────────────────────────────

@register("lu")
def _lu(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    P, L, U = scipy_linalg.lu(data)
    return Mat(P), Mat(L), Mat(U)

@register("qr")
def _qr(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    Q, R = np.linalg.qr(data)
    return Mat(Q), Mat(R)

@register("chol")
def _chol(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.linalg.cholesky(data))

@register("expm")
def _expm(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(scipy_linalg.expm(data))

@register("logm")
def _logm(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(scipy_linalg.logm(data))

@register("sqrtm")
def _sqrtm(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(scipy_linalg.sqrtm(data))

@register("kron")
def _kron(a, b):
    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    return Mat(np.kron(da, db))

@register("cond")
def _cond(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return float(np.linalg.cond(data))

@register("rref")
def _rref(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    # Row reduction using Gauss elimination
    m, n = data.shape
    A = data.copy().astype(float)
    pivot_row = 0
    for col in range(n):
        # Find pivot
        max_idx = np.argmax(np.abs(A[pivot_row:, col])) + pivot_row
        if A[max_idx, col] == 0:
            continue
        # Swap rows
        A[[pivot_row, max_idx]] = A[[max_idx, pivot_row]]
        # Scale pivot row
        A[pivot_row] = A[pivot_row] / A[pivot_row, col]
        # Eliminate column
        for i in range(m):
            if i != pivot_row:
                A[i] -= A[i, col] * A[pivot_row]
        pivot_row += 1
        if pivot_row >= m:
            break
    return Mat(A)

@register("null")
def _null(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    _, s, vh = np.linalg.svd(data)
    tol = max(data.shape) * np.max(s) * np.finfo(float).eps
    null_mask = s < tol
    null_space = vh[null_mask]
    return Mat(null_space.T)

@register("orth")
def _orth(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    u, s, _ = np.linalg.svd(data)
    tol = max(data.shape) * np.max(s) * np.finfo(float).eps
    rank = np.sum(s > tol)
    return Mat(u[:, :rank])
