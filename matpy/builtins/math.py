"""Math built-in functions for MatPy."""

import math
import numpy as np
from matpy.builtins import register
from matpy.runtime.types import Mat
from matpy.runtime.matrix import to_mat


def _wrap_math(func):
    def wrapper(*args):
        unwrapped = []
        for a in args:
            if isinstance(a, Mat):
                unwrapped.append(a.data)
            else:
                unwrapped.append(a)
        result = func(*unwrapped)
        return to_mat(result)
    return wrapper


register("sin", _wrap_math(np.sin))
register("cos", _wrap_math(np.cos))
register("tan", _wrap_math(np.tan))
register("asin", _wrap_math(np.arcsin))
register("acos", _wrap_math(np.arccos))
register("atan", _wrap_math(np.arctan))
register("atan2", _wrap_math(np.arctan2))
register("sinh", _wrap_math(np.sinh))
register("cosh", _wrap_math(np.cosh))
register("tanh", _wrap_math(np.tanh))
register("asinh", _wrap_math(np.arcsinh))
register("acosh", _wrap_math(np.arccosh))
register("atanh", _wrap_math(np.arctanh))

register("exp", _wrap_math(np.exp))
register("log", _wrap_math(np.log))
register("log2", _wrap_math(np.log2))
register("log10", _wrap_math(np.log10))
register("sqrt", _wrap_math(np.sqrt))

register("ceil", _wrap_math(np.ceil))
register("floor", _wrap_math(np.floor))
register("fix", _wrap_math(np.fix))
register("round", _wrap_math(np.round))

register("abs", _wrap_math(np.abs))
register("angle", _wrap_math(np.angle))
register("sign", _wrap_math(np.sign))
register("conj", _wrap_math(np.conj))

@register("min")
def _min(*args):
    arrs = [a.data if isinstance(a, Mat) else np.array(a) for a in args]
    if len(arrs) == 1:
        return to_mat(np.min(arrs[0]))
    elif len(arrs) == 2:
        return to_mat(np.minimum(arrs[0], arrs[1]))
    return to_mat(np.min(arrs))

@register("max")
def _max(*args):
    arrs = [a.data if isinstance(a, Mat) else np.array(a) for a in args]
    if len(arrs) == 1:
        return to_mat(np.max(arrs[0]))
    elif len(arrs) == 2:
        return to_mat(np.maximum(arrs[0], arrs[1]))
    return to_mat(np.max(arrs))

@register("sum")
def _sum(x, dim=None):
    data = x.data if isinstance(x, Mat) else np.array(x)
    if dim is not None:
        return to_mat(np.sum(data, axis=int(dim) - 1))
    return to_mat(np.sum(data))

@register("prod")
def _prod(x, dim=None):
    data = x.data if isinstance(x, Mat) else np.array(x)
    if dim is not None:
        return to_mat(np.prod(data, axis=int(dim) - 1))
    return to_mat(np.prod(data))

@register("cumsum")
def _cumsum(x, dim=None):
    data = x.data if isinstance(x, Mat) else np.array(x)
    if dim is not None:
        return to_mat(np.cumsum(data, axis=int(dim) - 1))
    return to_mat(np.cumsum(data))

@register("cumprod")
def _cumprod(x, dim=None):
    data = x.data if isinstance(x, Mat) else np.array(x)
    if dim is not None:
        return to_mat(np.cumprod(data, axis=int(dim) - 1))
    return to_mat(np.cumprod(data))

@register("diff")
def _diff(x, n=1):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return to_mat(np.diff(data, n=int(n)))

register("mod", _wrap_math(np.mod))
register("rem", _wrap_math(np.remainder))

@register("pi")
def _pi():
    return math.pi

@register("inf")
def _inf():
    return float("inf")

@register("nan")
def _nan():
    return float("nan")

@register("eps")
def _eps():
    return np.finfo(float).eps
