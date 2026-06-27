"""Built-in function registry for MatPy."""

from __future__ import annotations
from typing import Any, Callable

_BUILTINS: dict[str, Callable] = {}


def register(name: str, func: Callable | None = None):
    def decorator(f):
        _BUILTINS[name] = f
        return f

    if func is not None:
        _BUILTINS[name] = func
        return func
    return decorator


def get_builtin(name: str) -> Callable | None:
    return _BUILTINS.get(name)


def all_builtins() -> dict[str, Callable]:
    return dict(_BUILTINS)


# Import sub-modules to trigger registration
from matpy.builtins import math as _math
from matpy.builtins import matrix_ops as _matrix_ops
from matpy.builtins import io as _io
from matpy.builtins import plotting as _plotting
from matpy.builtins import data_struct as _data_struct
from matpy.builtins import string as _string
from matpy.builtins import file_io as _file_io
from matpy.builtins import advanced_math as _advanced_math
from matpy.builtins import sparse as _sparse
from matpy.builtins import string_array as _string_array
from matpy.builtins import common as _common
from matpy.builtins import control as _control
from matpy.builtins import signal as _signal
from matpy.builtins import optimization as _optimization
from matpy.builtins import statistics as _statistics
from matpy.builtins import image as _image
