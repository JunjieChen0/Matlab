"""Advanced math built-in functions for MatPy (P0 priority)."""

import numpy as np
from scipy import optimize, integrate, signal, stats
from scipy import linalg as scipy_linalg
from matpy.builtins import register
from matpy.runtime.types import Mat


# ── Optimization ──────────────────────────────────────────────

def _fzero(func, x0, *args):
    if callable(func):
        result = optimize.fsolve(func, float(x0), args=args)
        return float(result[0])
    raise RuntimeError("fzero: first argument must be a function")

def _fminsearch(func, x0, *args):
    if callable(func):
        result = optimize.minimize(func, float(x0), method='Nelder-Mead', args=args)
        return float(result.x[0])
    raise RuntimeError("fminsearch: first argument must be a function")

def _fminbnd(func, a, b, *args):
    if callable(func):
        result = optimize.minimize_scalar(func, bounds=(float(a), float(b)), method='bounded', args=args)
        return float(result.x)
    raise RuntimeError("fminbnd: first argument must be a function")

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

def _ode45(func, tspan, y0, *args):
    from scipy.integrate import solve_ivp
    tspan = np.array(tspan, dtype=float)
    y0 = np.array(y0, dtype=float).flatten()
    t_eval = np.linspace(tspan[0], tspan[1], 100)
    result = solve_ivp(func, tspan, y0, t_eval=t_eval, args=args, method='RK45')
    return Mat(result.t), Mat(result.y)

def _ode23(func, tspan, y0, *args):
    from scipy.integrate import solve_ivp
    tspan = np.array(tspan, dtype=float)
    y0 = np.array(y0, dtype=float).flatten()
    t_eval = np.linspace(tspan[0], tspan[1], 100)
    result = solve_ivp(func, tspan, y0, t_eval=t_eval, args=args, method='RK23')
    return Mat(result.t), Mat(result.y)

def _ode15s(func, tspan, y0, *args):
    from scipy.integrate import solve_ivp
    tspan = np.array(tspan, dtype=float)
    y0 = np.array(y0, dtype=float).flatten()
    t_eval = np.linspace(tspan[0], tspan[1], 100)
    result = solve_ivp(func, tspan, y0, t_eval=t_eval, args=args, method='BDF')
    return Mat(result.t), Mat(result.y)


# ── Integration ───────────────────────────────────────────────

def _integral(func, a, b, *args):
    if callable(func):
        result, _ = integrate.quad(func, float(a), float(b), args=args)
        return float(result)
    raise RuntimeError("integral: first argument must be a function")

def _integral2(func, a, b, c, d, *args):
    if callable(func):
        result, _ = integrate.dblquad(func, float(a), float(b), float(c), float(d), args=args)
        return float(result)
    raise RuntimeError("integral2: first argument must be a function")

@register("quad")
def _quad(func, a, b, *args):
    return _integral(func, a, b, *args)


# ── Interpolation ─────────────────────────────────────────────

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

def _mkpp(breaks, coefs):
    b = breaks.data if isinstance(breaks, Mat) else np.array(breaks).flatten()
    c = coefs.data if isinstance(coefs, Mat) else np.array(coefs)
    return (b, c)

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

def _mode(x, dim=None):
    data = x.data if isinstance(x, Mat) else np.array(x)
    from scipy import stats as sp_stats
    if dim is not None:
        axis = int(dim) - 1
        result = sp_stats.mode(data, axis=axis, keepdims=True)
        return Mat(result.mode)
    result = sp_stats.mode(data.flatten(), keepdims=True)
    return float(result.mode[0])

def _prctile(x, p):
    data = x.data if isinstance(x, Mat) else np.array(x).flatten()
    p_val = p.data if isinstance(p, Mat) else np.array(p)
    return Mat(np.percentile(data, p_val))

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

def _ttest(x, mu=0):
    data = x.data if isinstance(x, Mat) else np.array(x).flatten()
    t_stat, p_value = stats.ttest_1samp(data, float(mu))
    return float(t_stat), float(p_value)

def _ttest2(x, y):
    xd = x.data if isinstance(x, Mat) else np.array(x).flatten()
    yd = y.data if isinstance(y, Mat) else np.array(y).flatten()
    t_stat, p_value = stats.ttest_ind(xd, yd)
    return float(t_stat), float(p_value)

def _anova1(*args):
    groups = []
    for arg in args:
        data = arg.data if isinstance(arg, Mat) else np.array(arg).flatten()
        groups.append(data)
    f_stat, p_value = stats.f_oneway(*groups)
    return float(f_stat), float(p_value)

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
    """Kronecker tensor product."""
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


# ── Additional Advanced Math Functions ─────────────────────────

@register("schur")
def _schur(x):
    """Schur decomposition."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    T, Z = scipy_linalg.schur(data)
    return Mat(T), Mat(Z)


@register("hess")
def _hess(x):
    """Hessenberg decomposition."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    H, P = scipy_linalg.hessenberg(data, calc_q=True)
    return Mat(H), Mat(P)


@register("qz")
def _qz(a, b):
    """QZ decomposition."""
    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    AA, BB, Q, Z = scipy_linalg.qz(da, db)
    return Mat(AA), Mat(BB), Mat(Q), Mat(Z)


@register("cdf2rdf")
def _cdf2rdf(v, d):
    """Convert complex diagonal form to real block diagonal form."""
    vd = v.data if isinstance(v, Mat) else np.array(v)
    dd = d.data if isinstance(d, Mat) else np.array(d)
    return Mat(scipy_linalg.cdf2rdf(vd, dd)[0]), Mat(scipy_linalg.cdf2rdf(vd, dd)[1])


@register("rsf2csf")
def _rsf2csf(t, z):
    """Convert real Schur form to complex Schur form."""
    td = t.data if isinstance(t, Mat) else np.array(t)
    zd = z.data if isinstance(z, Mat) else np.array(z)
    return Mat(scipy_linalg.rsf2csf(td, zd)[0]), Mat(scipy_linalg.rsf2csf(td, zd)[1])


@register("funm")
def _funm(x, func):
    """Matrix function."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(scipy_linalg.funm(data, func))


@register("signm")
def _signm(x):
    """Matrix sign function."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(scipy_linalg.signm(data))


@register("cosm")
def _cosm(x):
    """Matrix cosine."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(scipy_linalg.cosm(data))


@register("sinm")
def _sinm(x):
    """Matrix sine."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(scipy_linalg.sinm(data))


@register("tanm")
def _tanm(x):
    """Matrix tangent."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(scipy_linalg.tanm(data))


@register("coshm")
def _coshm(x):
    """Matrix hyperbolic cosine."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(scipy_linalg.coshm(data))


@register("sinhm")
def _sinhm(x):
    """Matrix hyperbolic sine."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(scipy_linalg.sinhm(data))


@register("tanhm")
def _tanhm(x):
    """Matrix hyperbolic tangent."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(scipy_linalg.tanhm(data))


def _cond(x, p=None):
    """Condition number."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    if p is not None:
        return np.linalg.cond(data, p)
    return np.linalg.cond(data)


@register("rcond")
def _rcond(x):
    """Reciprocal condition number."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return 1.0 / np.linalg.cond(data)


@register("condest")
def _condest(x):
    """1-norm condition number estimate."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return np.linalg.cond(data, 1)


@register("normest")
def _normest(x, tol=None):
    """2-norm estimate."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return np.linalg.norm(data, 2)


@register("rank")
def _rank(x, tol=None):
    """Matrix rank."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    if tol is not None:
        return np.linalg.matrix_rank(data, tol=float(tol))
    return np.linalg.matrix_rank(data)


@register("det")
def _det(x):
    """Matrix determinant."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return np.linalg.det(data)


@register("inv")
def _inv(x):
    """Matrix inverse."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.linalg.inv(data))


@register("pinv")
def _pinv(x, tol=None):
    """Pseudo-inverse."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    if tol is not None:
        return Mat(np.linalg.pinv(data, rcond=float(tol)))
    return Mat(np.linalg.pinv(data))


@register("trace")
def _trace(x):
    """Matrix trace."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return np.trace(data)


@register("norm")
def _norm(x, p=None):
    """Vector or matrix norm."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    if p is not None:
        return np.linalg.norm(data, int(p))
    return np.linalg.norm(data)


@register("cross")
def _cross(a, b):
    """Cross product."""
    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    return Mat(np.cross(da.flatten(), db.flatten()))


@register("dot")
def _dot(a, b):
    """Dot product."""
    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    return np.dot(da.flatten(), db.flatten())


def _kron(a, b):
    """Kronecker tensor product."""
    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    return Mat(np.kron(da, db))


def _interp1(x, y, xq, method="linear"):
    """1-D interpolation."""
    from scipy.interpolate import interp1d
    xd = x.data if isinstance(x, Mat) else np.array(x)
    yd = y.data if isinstance(y, Mat) else np.array(y)
    xqd = xq.data if isinstance(xq, Mat) else np.array(xq)
    f = interp1d(xd.flatten(), yd.flatten(), kind=str(method))
    return Mat(f(xqd.flatten()))


def _interp2(x, y, v, xq, yq, method="linear"):
    """2-D interpolation."""
    from scipy.interpolate import RegularGridInterpolator
    xd = x.data if isinstance(x, Mat) else np.array(x)
    yd = y.data if isinstance(y, Mat) else np.array(y)
    vd = v.data if isinstance(v, Mat) else np.array(v)
    xqd = xq.data if isinstance(xq, Mat) else np.array(xq)
    yqd = yq.data if isinstance(yq, Mat) else np.array(yq)
    interp = RegularGridInterpolator((xd.flatten(), yd.flatten()), vd, method=str(method))
    points = np.column_stack([xqd.flatten(), yqd.flatten()])
    result = interp(points)
    return Mat(result.reshape(xqd.shape))


def _interp3(x, y, z, v, xq, yq, zq, method="linear"):
    """3-D interpolation."""
    from scipy.interpolate import RegularGridInterpolator
    xd = x.data if isinstance(x, Mat) else np.array(x)
    yd = y.data if isinstance(y, Mat) else np.array(y)
    zd = z.data if isinstance(z, Mat) else np.array(z)
    vd = v.data if isinstance(v, Mat) else np.array(v)
    xqd = xq.data if isinstance(xq, Mat) else np.array(xq)
    yqd = yq.data if isinstance(yq, Mat) else np.array(yq)
    zqd = zq.data if isinstance(zq, Mat) else np.array(zq)
    interp = RegularGridInterpolator((xd.flatten(), yd.flatten(), zd.flatten()), vd, method=str(method))
    points = np.column_stack([xqd.flatten(), yqd.flatten(), zqd.flatten()])
    result = interp(points)
    return Mat(result.reshape(xqd.shape))
