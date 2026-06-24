"""I/O built-in functions for MatPy."""

import sys
import numpy as np
from matpy.builtins import register
from matpy.runtime.types import Mat
from matpy.runtime.matrix import mat_str


@register("disp")
def _disp(*args):
    for arg in args:
        print(mat_str(arg))


@register("fprintf")
def _fprintf(fmt, *args):
    if isinstance(fmt, str):
        result = fmt
        for a in args:
            if isinstance(a, Mat):
                a = a.to_python()
            result = result.replace("%d", str(int(a)), 1) if "%d" in result else result
            result = result.replace("%f", str(float(a)), 1) if "%f" in result else result
            result = result.replace("%s", str(a), 1) if "%s" in result else result
            result = result.replace("%g", str(float(a)), 1) if "%g" in result else result
            result = result.replace("%e", f"{float(a):e}", 1) if "%e" in result else result
        print(result, end="")
    return len(args)


@register("sprintf")
def _sprintf(fmt, *args):
    result = fmt
    for a in args:
        if isinstance(a, Mat):
            a = a.to_python()
        result = result.replace("%d", str(int(a)), 1) if "%d" in result else result
        result = result.replace("%f", str(float(a)), 1) if "%f" in result else result
        result = result.replace("%s", str(a), 1) if "%s" in result else result
        result = result.replace("%g", str(float(a)), 1) if "%g" in result else result
    return result


@register("num2str")
def _num2str(x, fmt=None):
    data = x.data if isinstance(x, Mat) else np.array(x)
    if fmt:
        return " ".join(fmt % v for v in data.flat)
    return " ".join(str(v) for v in data.flat)


@register("int2str")
def _int2str(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return " ".join(str(int(v)) for v in data.flat)


@register("str2num")
def _str2num(s):
    try:
        parts = s.strip().split()
        return Mat(np.array([float(p) for p in parts]))
    except ValueError:
        return Mat(np.array([]))


@register("str2double")
def _str2double(s):
    try:
        return float(s.strip())
    except ValueError:
        return float("nan")


@register("class")
def _class(x):
    if isinstance(x, Mat):
        return "double"
    if isinstance(x, str):
        return "char"
    if isinstance(x, bool):
        return "logical"
    return type(x).__name__


@register("isa")
def _isa(x, class_name):
    if class_name == "double":
        return isinstance(x, (int, float, Mat))
    if class_name == "char":
        return isinstance(x, str)
    if class_name == "logical":
        return isinstance(x, bool)
    return False


@register("input")
def _input(prompt=""):
    try:
        return input(prompt)
    except EOFError:
        return ""


@register("feval")
def _feval(func_handle, *args):
    """Evaluate a function handle."""
    if callable(func_handle):
        return func_handle(*args)
    raise RuntimeError("First argument must be a function handle")


@register("nargin")
def _nargin(func_name=None):
    """Number of input arguments."""
    return 0


@register("nargout")
def _nargout(func_name=None):
    """Number of output arguments."""
    return 0


@register("mfilename")
def _mfilename():
    """Return name of current script."""
    return "<script>"


@register("addpath")
def _addpath(*args):
    """Add directory to MATLAB path."""
    import sys
    for arg in args:
        path = str(arg)
        if path not in sys.path:
            sys.path.insert(0, path)


@register("rmpath")
def _rmpath(*args):
    """Remove directory from MATLAB path."""
    import sys
    for arg in args:
        path = str(arg)
        if path in sys.path:
            sys.path.remove(path)


@register("path")
def _path():
    """Display MATLAB path."""
    import sys
    for p in sys.path:
        print(p)


@register("genpath")
def _genpath(root=""):
    """Generate path string."""
    import os
    paths = []
    root = str(root)
    if os.path.isdir(root):
        for dirpath, dirnames, filenames in os.walk(root):
            paths.append(dirpath)
    return ";".join(paths)
