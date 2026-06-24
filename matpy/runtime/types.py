"""Matlab-compatible data types for MatPy runtime."""

from __future__ import annotations
from typing import Any
import numpy as np
from scipy import sparse as sp


class Mat:
    """MATLAB-style matrix wrapper around numpy.ndarray or sparse matrix.

    - 1-based indexing (Matlab convention)
    - Supports 'end' keyword in indexing
    - Supports colon indexing A(1,:)
    - Supports logical indexing A(A>5)
    - Supports sparse matrices
    """

    def __init__(self, data):
        if isinstance(data, Mat):
            self._data = data._data.copy() if sp.issparse(data._data) else data._data.copy()
        elif sp.issparse(data):
            self._data = data
        elif isinstance(data, np.ndarray):
            self._data = data
        elif isinstance(data, (list, tuple)):
            self._data = np.array(data)
        else:
            self._data = np.array(data)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def shape(self) -> tuple:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def dtype(self):
        return self._data.dtype

    def __repr__(self) -> str:
        return f"Mat({self._data!r})"

    def __str__(self) -> str:
        return str(self._data)

    def __len__(self) -> int:
        return self._data.shape[0] if self._data.ndim > 0 else 1

    def to_python(self) -> Any:
        if self._data.ndim == 0:
            return self._data.item()
        return self._data

    def scalar(self) -> Any:
        if self._data.size == 1:
            return self._data.flat[0]
        return self._data

    def is_scalar(self) -> bool:
        return self._data.ndim == 0 or self._data.size == 1


class CellArray:
    """MATLAB cell array — heterogeneous container."""

    def __init__(self, data: list[list[Any]] | None = None):
        self._data: list[list[Any]] = data or [[]]

    @property
    def shape(self) -> tuple:
        rows = len(self._data)
        cols = max((len(r) for r in self._data), default=0)
        return (rows, cols)

    def get(self, row: int, col: int = 0) -> Any:
        r = row - 1
        c = col - 1
        if 0 <= r < len(self._data) and 0 <= c < len(self._data[r]):
            return self._data[r][c]
        raise IndexError(f"Cell index ({row},{col}) out of bounds")

    def get_linear(self, idx: int) -> Any:
        """1-based linear indexing."""
        flat = [item for row in self._data for item in row]
        i = idx - 1
        if 0 <= i < len(flat):
            return flat[i]
        raise IndexError(f"Cell index {idx} out of bounds")

    def set(self, row: int, col: int, value: Any):
        r = row - 1
        c = col - 1
        while len(self._data) <= r:
            self._data.append([])
        while len(self._data[r]) <= c:
            self._data[r].append(None)
        self._data[r][c] = value

    def numel(self) -> int:
        return sum(len(row) for row in self._data)

    def __repr__(self) -> str:
        return f"CellArray({self._data!r})"

    def __str__(self) -> str:
        lines = []
        for row in self._data:
            lines.append("  " + "  ".join(repr(x) for x in row))
        return "{" + "\n".join(lines) + "}"


class Struct:
    """MATLAB struct — dynamic field container."""

    def __init__(self, fields: dict[str, Any] | None = None):
        self._fields: dict[str, Any] = fields or {}

    def get_field(self, name: str) -> Any:
        if name not in self._fields:
            raise AttributeError(f"Struct has no field '{name}'")
        return self._fields[name]

    def set_field(self, name: str, value: Any):
        self._fields[name] = value

    def has_field(self, name: str) -> bool:
        return name in self._fields

    def field_names(self) -> list[str]:
        return list(self._fields.keys())

    def numel(self) -> int:
        return 1

    def __repr__(self) -> str:
        return f"Struct({self._fields!r})"

    def __str__(self) -> str:
        if not self._fields:
            return "struct with no fields"
        parts = [f"  {k}: {self._format_value(v)}" for k, v in self._fields.items()]
        return "struct with fields:\n" + "\n".join(parts)

    def _format_value(self, val: Any) -> str:
        if isinstance(val, Mat):
            data = val.data
            if data.ndim == 0:
                return str(data.item())
            elif data.size <= 10:
                return str(data)
            else:
                return f"[{data.shape[0]}x{data.shape[1]} {data.dtype}]"
        elif isinstance(val, str):
            return f"'{val}'"
        elif isinstance(val, Struct):
            return f"[1x1 struct]"
        elif isinstance(val, CellArray):
            return f"{{{val.shape[0]}x{val.shape[1]} cell}}"
        else:
            return repr(val)


class FuncHandle:
    """MATLAB function handle."""

    def __init__(self, func: callable, name: str = "<anonymous>"):
        self.func = func
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:
        return f"@{self.name}"


class StringArray:
    """MATLAB string array (double-quoted strings).

    In MATLAB:
    - "hello" creates a string scalar
    - ["hello", "world"] creates a string array
    - string(3) creates a 3x1 missing string array
    """

    def __init__(self, data=None):
        if data is None:
            self._data = np.array([], dtype=object)
        elif isinstance(data, str):
            self._data = np.array(data, dtype=object)
        elif isinstance(data, (list, tuple)):
            self._data = np.array(data, dtype=object)
        elif isinstance(data, np.ndarray):
            self._data = data.astype(object)
        else:
            self._data = np.array(str(data), dtype=object)

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._data.shape

    def __str__(self):
        if self._data.ndim == 0:
            return str(self._data.item())
        return str(self._data)

    def __repr__(self):
        if self._data.ndim == 0:
            return f'"{self._data.item()}"'
        return f"string({self._data})"

    def __len__(self):
        return self._data.size

    def __eq__(self, other):
        if isinstance(other, StringArray):
            return self._data == other._data
        return self._data == other

    def __add__(self, other):
        if isinstance(other, StringArray):
            return StringArray(self._data + other._data)
        return StringArray(self._data + str(other))

    def __getitem__(self, key):
        result = self._data[key]
        if isinstance(result, np.ndarray):
            return StringArray(result)
        return result


class Missing:
    """MATLAB missing value."""
    def __repr__(self):
        return "missing"

    def __str__(self):
        return "missing"

    def __eq__(self, other):
        return isinstance(other, Missing)


class ClassDefRuntime:
    """Runtime representation of a MATLAB classdef."""

    def __init__(self, name: str, superclass: str | None = None):
        self.name = name
        self.superclass = superclass
        self.properties: dict[str, Any] = {}  # name -> default value
        self.property_attrs: dict[str, dict[str, Any]] = {}
        self.methods: dict[str, Any] = {}  # name -> FuncDef or callable
        self.method_attrs: dict[str, dict[str, Any]] = {}

    def __repr__(self) -> str:
        return f"<class '{self.name}'>"


class ClassInstance:
    """Instance of a MATLAB classdef."""

    def __init__(self, class_def: ClassDefRuntime):
        self._class_def = class_def
        self._properties: dict[str, Any] = {}
        # Initialize properties with defaults
        for name, default in class_def.properties.items():
            self._properties[name] = default

    def get_property(self, name: str) -> Any:
        if name in self._properties:
            return self._properties[name]
        raise AttributeError(f"'{self._class_def.name}' has no property '{name}'")

    def set_property(self, name: str, value: Any):
        self._properties[name] = value

    def has_property(self, name: str) -> bool:
        return name in self._properties

    def has_method(self, name: str) -> bool:
        return name in self._class_def.methods

    def get_method(self, name: str) -> Any:
        if name in self._class_def.methods:
            return self._class_def.methods[name]
        raise AttributeError(f"'{self._class_def.name}' has no method '{name}'")

    def property_names(self) -> list[str]:
        return list(self._properties.keys())

    def __repr__(self) -> str:
        return f"<{self._class_def.name} instance>"

    def __str__(self) -> str:
        parts = []
        for name, val in self._properties.items():
            parts.append(f"  {name}: {val}")
        return f"{self._class_def.name} with properties:\n" + "\n".join(parts)
