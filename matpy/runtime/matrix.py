"""Matrix operations and indexing for MatPy runtime."""

from __future__ import annotations
from typing import Any
import numpy as np
from matpy.runtime.types import Mat


def matlab_index(data: np.ndarray, indices: tuple) -> np.ndarray:
    """Apply 1-based MATLAB indexing to a numpy array."""
    converted = []
    for idx in indices:
        if isinstance(idx, int):
            if idx == 0:
                converted.append(-1)
            else:
                converted.append(idx - 1)
        elif isinstance(idx, slice):
            start = idx.start
            stop = idx.stop
            step = idx.step
            if start is not None:
                start = start - 1
            converted.append(slice(start, stop, step))
        elif isinstance(idx, np.ndarray):
            if idx.dtype == bool:
                converted.append(idx)
            else:
                converted.append(idx - 1)
        else:
            converted.append(idx)
    return data[tuple(converted)]


def resolve_end(data: np.ndarray, dim: int) -> int:
    if dim < data.ndim:
        return data.shape[dim]
    return 1


def make_range(start: float, step_or_stop: float, stop: float | None = None) -> np.ndarray:
    if stop is None:
        return np.arange(start, step_or_stop + 1)
    else:
        return np.arange(start, stop + step_or_stop, step_or_stop)


def to_mat(value: Any) -> Any:
    if isinstance(value, Mat):
        return value
    if isinstance(value, np.ndarray):
        return Mat(value)
    if isinstance(value, (int, float, complex, bool, np.integer, np.floating, np.complexfloating)):
        return Mat(np.array(value))
    return value


def from_mat(value: Any) -> Any:
    if isinstance(value, Mat):
        return value.to_python()
    return value


def mat_str(value: Any) -> str:
    if isinstance(value, Mat):
        arr = value.data
        if arr.ndim == 0:
            return str(arr.item())
        elif arr.ndim == 1:
            return "  ".join(str(x) for x in arr)
        else:
            lines = []
            for row in arr:
                lines.append("  ".join(str(x) for x in row))
            return "\n".join(lines)
    if isinstance(value, str):
        return value
    return str(value)
