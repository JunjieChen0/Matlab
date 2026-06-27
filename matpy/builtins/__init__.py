"""Built-in function registry for MatPy."""

from __future__ import annotations
from typing import Callable

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


# Import sub-modules to trigger registration (must be after register function)
from matpy.builtins import math as _math  # noqa: E402, F401
from matpy.builtins import matrix_ops as _matrix_ops  # noqa: E402, F401
from matpy.builtins import io as _io  # noqa: E402, F401
from matpy.builtins import plotting as _plotting  # noqa: E402, F401
from matpy.builtins import data_struct as _data_struct  # noqa: E402, F401
from matpy.builtins import string as _string  # noqa: E402, F401
from matpy.builtins import file_io as _file_io  # noqa: E402, F401
from matpy.builtins import advanced_math as _advanced_math  # noqa: E402, F401
from matpy.builtins import sparse as _sparse  # noqa: E402, F401
from matpy.builtins import string_array as _string_array  # noqa: E402, F401
from matpy.builtins import common as _common  # noqa: E402, F401
from matpy.builtins import control as _control  # noqa: E402, F401
from matpy.builtins import signal as _signal  # noqa: E402, F401
from matpy.builtins import optimization as _optimization  # noqa: E402, F401
from matpy.builtins import statistics as _statistics  # noqa: E402, F401
from matpy.builtins import image as _image  # noqa: E402, F401
