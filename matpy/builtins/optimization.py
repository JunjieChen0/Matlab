"""Optimization Toolbox for MatPy."""

import numpy as np
from scipy import optimize as scipy_optimize
from matpy.builtins import register
from matpy.runtime.types import Mat


@register("quadprog")
def _quadprog(H, f, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None):
    H = H.data if isinstance(H, Mat) else np.array(H)
    f = f.data if isinstance(f, Mat) else np.array(f)
    f = f.flatten()
    constraints = []
    if A is not None and b is not None:
        A_data = A.data if isinstance(A, Mat) else np.array(A)
        b_data = b.data if isinstance(b, Mat) else np.array(b)
        constraints.append({"type": "ineq", "fun": lambda x: b_data.flatten() - A_data @ x})
    if Aeq is not None and beq is not None:
        Aeq_data = Aeq.data if isinstance(Aeq, Mat) else np.array(Aeq)
        beq_data = beq.data if isinstance(beq, Mat) else np.array(beq)
        constraints.append({"type": "eq", "fun": lambda x: beq_data.flatten() - Aeq_data @ x})
    bounds = None
    if lb is not None or ub is not None:
        lb_val = float(lb) if lb is not None else None
        ub_val = float(ub) if ub is not None else None
        bounds = [(lb_val, ub_val)] * len(f)
    result = scipy_optimize.minimize(
        lambda x: 0.5 * x @ H @ x + f @ x,
        np.zeros(len(f)),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    return Mat(result.x), float(result.fun)


@register("intlinprog")
def _intlinprog(f, intcon, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None):
    f = f.data if isinstance(f, Mat) else np.array(f)
    f = f.flatten()
    constraints = []
    if A is not None and b is not None:
        A_data = A.data if isinstance(A, Mat) else np.array(A)
        b_data = b.data if isinstance(b, Mat) else np.array(b)
        from scipy.optimize import LinearConstraint
        constraints.append(LinearConstraint(A_data, -np.inf, b_data.flatten()))
    if Aeq is not None and beq is not None:
        Aeq_data = Aeq.data if isinstance(Aeq, Mat) else np.array(Aeq)
        beq_data = beq.data if isinstance(beq, Mat) else np.array(beq)
        from scipy.optimize import LinearConstraint
        constraints.append(LinearConstraint(Aeq_data, beq_data.flatten(), beq_data.flatten()))
    bounds = None
    if lb is not None or ub is not None:
        lb_val = float(lb) if lb is not None else -np.inf
        ub_val = float(ub) if ub is not None else np.inf
        bounds = scipy_optimize.Bounds(np.full(len(f), lb_val), np.full(len(f), ub_val))
    intcon_data = intcon.data if isinstance(intcon, Mat) else np.array(intcon)
    integrality = np.zeros(len(f))
    for idx in intcon_data.flatten():
        integrality[int(idx) - 1] = 1
    from scipy.optimize import milp
    result = milp(f, constraints=constraints, bounds=bounds, integrality=integrality)
    return Mat(result.x), float(result.fun)


@register("lsqlin")
def _lsqlin(C, d, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None):
    C_data = C.data if isinstance(C, Mat) else np.array(C)
    d_data = d.data if isinstance(d, Mat) else np.array(d)
    constraints = []
    if A is not None and b is not None:
        A_data = A.data if isinstance(A, Mat) else np.array(A)
        b_data = b.data if isinstance(b, Mat) else np.array(b)
        constraints.append({"type": "ineq", "fun": lambda x: b_data.flatten() - A_data @ x})
    if Aeq is not None and beq is not None:
        Aeq_data = Aeq.data if isinstance(Aeq, Mat) else np.array(Aeq)
        beq_data = beq.data if isinstance(beq, Mat) else np.array(beq)
        constraints.append({"type": "eq", "fun": lambda x: beq_data.flatten() - Aeq_data @ x})
    bounds = None
    if lb is not None or ub is not None:
        lb_val = float(lb) if lb is not None else None
        ub_val = float(ub) if ub is not None else None
        bounds = [(lb_val, ub_val)] * C_data.shape[1]
    result = scipy_optimize.minimize(
        lambda x: 0.5 * np.sum((C_data @ x - d_data.flatten()) ** 2),
        np.zeros(C_data.shape[1]),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    return Mat(result.x), float(result.fun)


@register("lsqnonneg")
def _lsqnonneg(C, d):
    C_data = C.data if isinstance(C, Mat) else np.array(C)
    d_data = d.data if isinstance(d, Mat) else np.array(d)
    x, rnorm = scipy_optimize.nnls(C_data, d_data.flatten())
    return Mat(x), float(rnorm)


@register("optimset")
def _optimset(*args):
    options = {}
    i = 0
    while i < len(args):
        if isinstance(args[i], str):
            options[args[i]] = args[i + 1]
            i += 2
        else:
            i += 1
    return options


@register("optimget")
def _optimget(options, key, default=None):
    if isinstance(options, dict):
        return options.get(str(key), default)
    return default


@register("ga")
def _ga(func, nvars, *args):
    nvars = int(nvars)
    bounds = [(-100, 100)] * nvars
    result = scipy_optimize.differential_evolution(func, bounds, args=args, maxiter=100)
    return Mat(result.x), float(result.fun)


@register("particleswarm")
def _particleswarm(func, nvars, *args):
    nvars = int(nvars)
    bounds = [(-100, 100)] * nvars
    result = scipy_optimize.differential_evolution(func, bounds, args=args, maxiter=100)
    return Mat(result.x), float(result.fun)


@register("simulannealbnd")
def _simulannealbnd(func, x0, *args):
    x0 = x0.data if isinstance(x0, Mat) else np.array(x0)
    x0 = x0.flatten()
    result = scipy_optimize.minimize(func, x0, method="Nelder-Mead", args=args)
    return Mat(result.x), float(result.fun)


@register("fminsearch")
def _fminsearch(func, x0, *args):
    x0 = x0.data if isinstance(x0, Mat) else np.array(x0)
    x0 = x0.flatten()
    result = scipy_optimize.minimize(func, x0, method="Nelder-Mead", args=args)
    return Mat(result.x), float(result.fun)


@register("fminbnd")
def _fminbnd(func, a, b, *args):
    a = float(a.data.flat[0]) if isinstance(a, Mat) else float(a)
    b = float(b.data.flat[0]) if isinstance(b, Mat) else float(b)
    result = scipy_optimize.minimize_scalar(func, bounds=(a, b), method="bounded", args=args)
    return float(result.x), float(result.fun)


@register("fzero")
def _fzero(func, x0, *args):
    x0 = float(x0.data.flat[0]) if isinstance(x0, Mat) else float(x0)
    result = scipy_optimize.fsolve(func, x0, args=args)
    return float(result[0])


@register("fsolve")
def _fsolve(func, x0, *args):
    x0 = x0.data if isinstance(x0, Mat) else np.array(x0)
    x0 = x0.flatten()
    result = scipy_optimize.fsolve(func, x0, args=args, full_output=True)
    return Mat(result[0])


@register("fmincon")
def _fmincon(func, x0, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None, *args):
    x0 = x0.data if isinstance(x0, Mat) else np.array(x0)
    x0 = x0.flatten()
    constraints = []
    if A is not None and b is not None:
        A_data = A.data if isinstance(A, Mat) else np.array(A)
        b_data = b.data if isinstance(b, Mat) else np.array(b)
        constraints.append({"type": "ineq", "fun": lambda x: b_data.flatten() - A_data @ x})
    if Aeq is not None and beq is not None:
        Aeq_data = Aeq.data if isinstance(Aeq, Mat) else np.array(Aeq)
        beq_data = beq.data if isinstance(beq, Mat) else np.array(beq)
        constraints.append({"type": "eq", "fun": lambda x: beq_data.flatten() - Aeq_data @ x})
    bounds = None
    if lb is not None or ub is not None:
        lb_val = float(lb) if lb is not None else None
        ub_val = float(ub) if ub is not None else None
        bounds = [(lb_val, ub_val)] * len(x0)
    result = scipy_optimize.minimize(func, x0, method="SLSQP", bounds=bounds, constraints=constraints, args=args)
    return Mat(result.x), float(result.fun)


@register("linprog")
def _linprog(c, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None):
    c = c.data if isinstance(c, Mat) else np.array(c)
    c = c.flatten()
    A_ub = A.data if isinstance(A, Mat) else np.array(A) if A is not None else None
    b_ub = b.data if isinstance(b, Mat) else np.array(b) if b is not None else None
    A_eq = Aeq.data if isinstance(Aeq, Mat) else np.array(Aeq) if Aeq is not None else None
    b_eq = beq.data if isinstance(beq, Mat) else np.array(beq) if beq is not None else None
    bounds = None
    if lb is not None or ub is not None:
        lb_val = float(lb) if lb is not None else None
        ub_val = float(ub) if ub is not None else None
        bounds = [(lb_val, ub_val)] * len(c)
    if A_ub is not None:
        A_ub = A_ub.reshape(-1, len(c))
    if b_ub is not None:
        b_ub = b_ub.flatten()
    if A_eq is not None:
        A_eq = A_eq.reshape(-1, len(c))
    if b_eq is not None:
        b_eq = b_eq.flatten()
    result = scipy_optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    return Mat(result.x), float(result.fun)


@register("fminunc")
def _fminunc(func, x0, *args):
    x0 = x0.data if isinstance(x0, Mat) else np.array(x0)
    x0 = x0.flatten()
    result = scipy_optimize.minimize(func, x0, method="BFGS", args=args)
    return Mat(result.x), float(result.fun)


@register("lsqnonlin")
def _lsqnonlin(func, x0, *args):
    x0 = x0.data if isinstance(x0, Mat) else np.array(x0)
    x0 = x0.flatten()
    result = scipy_optimize.least_squares(func, x0, args=args)
    return Mat(result.x), float(result.cost)


@register("lsqcurvefit")
def _lsqcurvefit(func, x0, xdata, ydata, *args):
    x0 = x0.data if isinstance(x0, Mat) else np.array(x0)
    x0 = x0.flatten()
    xd = xdata.data if isinstance(xdata, Mat) else np.array(xdata)
    yd = ydata.data if isinstance(ydata, Mat) else np.array(ydata)
    popt, pcov = scipy_optimize.curve_fit(func, xd.flatten(), yd.flatten(), p0=x0)
    return Mat(popt), Mat(pcov)


@register("fminimax")
def _fminimax(func, x0, *args):
    x0 = x0.data if isinstance(x0, Mat) else np.array(x0)
    x0 = x0.flatten()
    result = scipy_optimize.minimize(
        lambda x: np.max(np.abs(func(x, *args))),
        x0,
        method="Nelder-Mead"
    )
    return Mat(result.x), float(result.fun)


@register("fgoalattain")
def _fgoalattain(func, x0, goal, weight, *args):
    x0 = x0.data if isinstance(x0, Mat) else np.array(x0)
    x0 = x0.flatten()
    goal_data = goal.data if isinstance(goal, Mat) else np.array(goal)
    weight_data = weight.data if isinstance(weight, Mat) else np.array(weight)
    def objective(x):
        fvals = func(x, *args)
        return np.max(weight_data.flatten() * (fvals - goal_data.flatten()))
    result = scipy_optimize.minimize(objective, x0, method="Nelder-Mead")
    return Mat(result.x), float(result.fun)


@register("psearch")
def _psearch(func, x0, *args):
    x0 = x0.data if isinstance(x0, Mat) else np.array(x0)
    x0 = x0.flatten()
    result = scipy_optimize.minimize(func, x0, method="Nelder-Mead", args=args)
    return Mat(result.x), float(result.fun)


@register("integral")
def _integral(func, a, b, *args):
    from scipy import integrate
    a = float(a.data.flat[0]) if isinstance(a, Mat) else float(a)
    b = float(b.data.flat[0]) if isinstance(b, Mat) else float(b)
    result, _ = integrate.quad(func, a, b, args=args)
    return float(result)


@register("integral2")
def _integral2(func, a, b, c, d, *args):
    from scipy import integrate
    a = float(a.data.flat[0]) if isinstance(a, Mat) else float(a)
    b = float(b.data.flat[0]) if isinstance(b, Mat) else float(b)
    c_val = float(c.data.flat[0]) if isinstance(c, Mat) else float(c)
    d_val = float(d.data.flat[0]) if isinstance(d, Mat) else float(d)
    result, _ = integrate.dblquad(func, a, b, c_val, d_val, args=args)
    return float(result)
