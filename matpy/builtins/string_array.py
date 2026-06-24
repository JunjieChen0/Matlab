"""String array built-in functions for MatPy."""

import numpy as np
from matpy.builtins import register
from matpy.runtime.types import Mat, StringArray, Missing


@register("string")
def _string(*args):
    if len(args) == 0:
        return StringArray()
    elif len(args) == 1:
        arg = args[0]
        if isinstance(arg, (int, float)):
            n = int(arg)
            return StringArray(np.array([Missing()] * n, dtype=object))
        elif isinstance(arg, Mat):
            data = arg.data
            return StringArray(np.array([str(x) for x in data.flat], dtype=object).reshape(data.shape))
        elif isinstance(arg, StringArray):
            return arg
        elif isinstance(arg, str):
            return StringArray(arg)
        else:
            return StringArray(str(arg))
    else:
        return StringArray([str(a) for a in args])


@register("isstring")
def _isstring(x):
    return isinstance(x, StringArray)


@register("char")
def _char(x):
    if isinstance(x, StringArray):
        data = x.data
        if data.ndim == 0:
            return str(data.item())
        return str(data)
    return str(x)


@register("cellstr")
def _cellstr(x):
    if isinstance(x, StringArray):
        data = x.data
        if data.ndim == 0:
            return [str(data.item())]
        return [str(x) for x in data.flat]
    return [str(x)]


@register("strlength")
def _strlength(x):
    if isinstance(x, StringArray):
        data = x.data
        if data.ndim == 0:
            return len(str(data.item()))
        return Mat(np.array([len(str(s)) for s in data.flat]).reshape(data.shape))
    return len(str(x))


@register("append")
def _append(*args):
    parts = []
    for arg in args:
        if isinstance(arg, StringArray):
            parts.append(arg.data.flat[0] if arg.data.ndim == 0 else arg.data)
        else:
            parts.append(str(arg))
    return StringArray(np.concatenate(parts))
