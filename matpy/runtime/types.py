"""Matlab-compatible data types for MatPy runtime."""

from __future__ import annotations
from typing import Any, Callable
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

    def __init__(self, func: Callable, name: str = "<anonymous>"):
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

    def __init__(self, name: str, superclass: str | None = None, class_attrs: dict | None = None):
        self.name = name
        self.superclass = superclass
        self.class_attrs = class_attrs or {}
        self.properties: dict[str, Any] = {}  # name -> default value
        self.property_attrs: dict[str, dict[str, Any]] = {}
        self.methods: dict[str, Any] = {}  # name -> FuncDef or callable
        self.method_attrs: dict[str, dict[str, Any]] = {}

    @property
    def is_abstract(self) -> bool:
        return bool(self.class_attrs.get('Abstract', False))

    @property
    def is_sealed(self) -> bool:
        return bool(self.class_attrs.get('Sealed', False))

    @property
    def access(self) -> str:
        return self.class_attrs.get('Access', 'public')

    def is_constant_property(self, prop_name: str) -> bool:
        """Check if a property is Constant."""
        prop_attr = self.property_attrs.get(prop_name, {})
        return bool(prop_attr.get('Constant', False))

    def is_static_method(self, meth_name: str) -> bool:
        """Check if a method is Static."""
        meth_attr = self.method_attrs.get(meth_name, {})
        return bool(meth_attr.get('Static', False))

    def get_property_access(self, prop_name: str) -> str:
        """Get the Access level for a property."""
        prop_attr = self.property_attrs.get(prop_name, {})
        return prop_attr.get('Access', 'public')

    def get_method_access(self, meth_name: str) -> str:
        """Get the Access level for a method."""
        meth_attr = self.method_attrs.get(meth_name, {})
        return meth_attr.get('Access', 'public')

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

    def set_property(self, name: str, value: Any, caller_class: str | None = None):
        """Set a property value with access control."""
        # Check if property is Constant
        if self._class_def.is_constant_property(name):
            raise AttributeError(f"Cannot modify Constant property '{name}' of '{self._class_def.name}'")
        # Check Access level
        access = self._class_def.get_property_access(name)
        if access == 'private' and caller_class != self._class_def.name:
            raise AttributeError(f"Property '{name}' of '{self._class_def.name}' is private")
        self._properties[name] = value

    def has_property(self, name: str) -> bool:
        return name in self._properties

    def has_method(self, name: str) -> bool:
        return name in self._class_def.methods

    def get_method(self, name: str, caller_class: str | None = None) -> Any:
        """Get a method with access control."""
        if name not in self._class_def.methods:
            raise AttributeError(f"'{self._class_def.name}' has no method '{name}'")
        # Check Access level
        access = self._class_def.get_method_access(name)
        if access == 'private' and caller_class != self._class_def.name:
            raise AttributeError(f"Method '{name}' of '{self._class_def.name}' is private")
        return self._class_def.methods[name]

    def property_names(self) -> list[str]:
        return list(self._properties.keys())

    def __repr__(self) -> str:
        return f"<{self._class_def.name} instance>"

    def __str__(self) -> str:
        parts = []
        for name, val in self._properties.items():
            parts.append(f"  {name}: {val}")
        return f"{self._class_def.name} with properties:\n" + "\n".join(parts)


class MException(Exception):
    """MATLAB MException object for try/catch error handling."""

    def __init__(self, message: str, identifier: str = "", original: Exception | None = None, stack: list | None = None):
        self.message = message
        self.identifier = identifier
        self._original = original
        self.stack = stack or []  # Call stack for debugging

    def get_field(self, name: str) -> Any:
        if name == "message":
            return self.message
        if name == "identifier":
            return self.identifier
        if name == "stack":
            return self.stack
        raise AttributeError(f"MException has no field '{name}'")

    def set_field(self, name: str, value: Any):
        if name == "message":
            self.message = value
        elif name == "identifier":
            self.identifier = value
        else:
            raise AttributeError(f"MException has no field '{name}'")

    def has_field(self, name: str) -> bool:
        return name in ("message", "identifier", "stack")

    def field_names(self) -> list[str]:
        return ["message", "identifier", "stack"]

    def throw(self):
        """Throw this exception (for rethrow support)."""
        raise MatPyRethrowError(self)

    def __repr__(self) -> str:
        return f"MException(message='{self.message}', identifier='{self.identifier}')"

    def __str__(self) -> str:
        return self.message


class MatPyRethrowError(Exception):
    """Exception wrapper for rethrow support."""
    def __init__(self, mexc: MException):
        self.mexception = mexc
        super().__init__(mexc.message)


class Table:
    """MATLAB table data type for tabular data."""

    def __init__(self, data: dict[str, np.ndarray], row_names: list[str] | None = None):
        self._data = {k: np.array(v) for k, v in data.items()}
        self._row_names = row_names
        # Validate all columns have the same length
        if self._data:
            lengths = {k: len(v) for k, v in self._data.items()}
            unique_lengths = set(lengths.values())
            if len(unique_lengths) > 1:
                raise ValueError(f"Table columns must have the same length, got: {lengths}")

    @property
    def Variables(self) -> list[str]:
        return list(self._data.keys())

    @property
    def Height(self) -> int:
        return len(next(iter(self._data.values()))) if self._data else 0

    @property
    def Width(self) -> int:
        return len(self._data)

    def get_field(self, name: str) -> Any:
        if name in self._data:
            return Mat(self._data[name])
        if name == "Variables":
            return self.Variables
        if name == "Height":
            return self.Height
        if name == "Width":
            return self.Width
        if name == "RowNames":
            return self._row_names
        raise AttributeError(f"Table has no field '{name}'")

    def set_field(self, name: str, value: Any):
        self._data[name] = np.array(value) if not isinstance(value, Mat) else value.data

    def has_field(self, name: str) -> bool:
        return name in self._data or name in ("Variables", "Height", "Width", "RowNames")

    def field_names(self) -> list[str]:
        return list(self._data.keys())

    def __getitem__(self, key):
        """Support T(row_indices) and T(row_indices, col_indices) indexing."""
        if isinstance(key, tuple):
            row_idx, col_idx = key
        else:
            row_idx = key
            col_idx = slice(None)

        # Handle row indexing
        if isinstance(row_idx, str) and row_idx == ":":
            row_idx = list(range(self.Height))
        elif isinstance(row_idx, int):
            row_idx = [row_idx - 1]  # 1-based to 0-based
        elif isinstance(row_idx, slice):
            start = (row_idx.start - 1) if row_idx.start is not None else 0
            stop = row_idx.stop if row_idx.stop is not None else self.Height
            step = row_idx.step or 1
            row_idx = list(range(start, stop, step))
        elif isinstance(row_idx, Mat):
            row_idx = (row_idx.data.astype(int) - 1).flatten().tolist()
        elif isinstance(row_idx, list):
            row_idx = [r - 1 for r in row_idx]

        # Handle column indexing
        if isinstance(col_idx, str) and col_idx == ":":
            col_names = list(self._data.keys())
        elif isinstance(col_idx, str):
            col_names = [col_idx]
        elif isinstance(col_idx, slice) and col_idx == slice(None):
            col_names = list(self._data.keys())
        elif isinstance(col_idx, list):
            col_names = col_idx
        else:
            col_names = list(self._data.keys())

        # Extract rows
        new_data = {}
        for col in col_names:
            if col in self._data:
                new_data[col] = self._data[col][row_idx]
        return Table(new_data)

    def __repr__(self) -> str:
        return f"Table({self.Height}x{self.Width})"

    def __str__(self) -> str:
        lines = [f"  {self.Height}x{self.Width} table"]
        if self._data:
            header = "  " + "  ".join(f"{k:>10}" for k in self._data.keys())
            lines.append(header)
            for i in range(min(5, self.Height)):
                row = "  " + "  ".join(f"{self._data[k][i]:>10}" for k in self._data.keys())
                lines.append(row)
            if self.Height > 5:
                lines.append("  ...")
        return "\n".join(lines)


class Map:
    """MATLAB containers.Map type for key-value storage."""

    def __init__(self, keys=None, values=None, key_type='any', value_type='any'):
        self._data: dict = {}
        self._key_type = key_type
        self._value_type = value_type
        if keys is not None and values is not None:
            kd = keys.data if isinstance(keys, Mat) else np.array(keys)
            vd = values.data if isinstance(values, Mat) else np.array(values)
            for k, v in zip(kd.flat, vd.flat):
                self._data[k] = v

    def get_field(self, name: str) -> Any:
        if name == "KeyType":
            return self._key_type
        if name == "ValueType":
            return self._value_type
        if name == "Count":
            return len(self._data)
        raise AttributeError(f"Map has no field '{name}'")

    def has_field(self, name: str) -> bool:
        return name in ("KeyType", "ValueType", "Count")

    def field_names(self) -> list[str]:
        return ["KeyType", "ValueType", "Count"]

    def isKey(self, key) -> bool:
        return key in self._data

    def keys(self) -> list:
        return list(self._data.keys())

    def values(self) -> list:
        return list(self._data.values())

    def remove(self, key):
        if key in self._data:
            del self._data[key]

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __repr__(self) -> str:
        return f"Map({len(self._data)} entries)"

    def __str__(self) -> str:
        lines = [f"  Map with {len(self._data)} entries"]
        for i, (k, v) in enumerate(self._data.items()):
            if i >= 5:
                lines.append("  ...")
                break
            lines.append(f"    {k}: {v}")
        return "\n".join(lines)


# ── Datetime Types ─────────────────────────────────────────────

class Datetime:
    """MATLAB datetime type."""

    def __init__(self, *args, **kwargs):
        import datetime as dt
        if len(args) == 0:
            self._dt = dt.datetime.now()
        elif len(args) == 1 and isinstance(args[0], str):
            # Parse string
            try:
                self._dt = dt.datetime.fromisoformat(args[0])
            except ValueError:
                self._dt = dt.datetime.strptime(args[0], "%Y-%m-%d %H:%M:%S")
        elif len(args) >= 3:
            year = int(args[0])
            month = int(args[1])
            day = int(args[2])
            hour = int(args[3]) if len(args) > 3 else 0
            minute = int(args[4]) if len(args) > 4 else 0
            second = int(args[5]) if len(args) > 5 else 0
            self._dt = dt.datetime(year, month, day, hour, minute, second)
        else:
            self._dt = dt.datetime.now()

    @property
    def Year(self) -> int:
        return self._dt.year

    @property
    def Month(self) -> int:
        return self._dt.month

    @property
    def Day(self) -> int:
        return self._dt.day

    @property
    def Hour(self) -> int:
        return self._dt.hour

    @property
    def Minute(self) -> int:
        return self._dt.minute

    @property
    def Second(self) -> int:
        return self._dt.second

    def get_field(self, name: str) -> Any:
        fields = {
            "Year": self.Year,
            "Month": self.Month,
            "Day": self.Day,
            "Hour": self.Hour,
            "Minute": self.Minute,
            "Second": self.Second,
        }
        if name in fields:
            return fields[name]
        raise AttributeError(f"Datetime has no field '{name}'")

    def __repr__(self) -> str:
        return f"Datetime({self._dt.isoformat()})"

    def __str__(self) -> str:
        return self._dt.strftime("%Y-%m-%d %H:%M:%S")

    def __add__(self, other):
        if isinstance(other, Duration):
            import datetime as dt
            return Datetime.from_dt(self._dt + dt.timedelta(days=other.days, seconds=other.seconds))
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Datetime):
            diff = self._dt - other._dt
            return Duration(days=diff.days, seconds=diff.seconds)
        if isinstance(other, Duration):
            import datetime as dt
            return Datetime.from_dt(self._dt - dt.timedelta(days=other.days, seconds=other.seconds))
        return NotImplemented

    @classmethod
    def from_dt(cls, dt_obj):
        obj = cls.__new__(cls)
        obj._dt = dt_obj
        return obj


class Duration:
    """MATLAB duration type."""

    def __init__(self, hours=0, minutes=0, seconds=0, days=0):
        total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
        self._total_seconds = total_seconds
        self.days = int(total_seconds // 86400)
        self.seconds = total_seconds % 86400

    @property
    def Hours(self) -> float:
        return self._total_seconds / 3600

    @property
    def Minutes(self) -> float:
        return self._total_seconds / 60

    @property
    def Seconds(self) -> float:
        return self._total_seconds

    def get_field(self, name: str) -> Any:
        fields = {
            "Hours": self.Hours,
            "Minutes": self.Minutes,
            "Seconds": self.Seconds,
        }
        if name in fields:
            return fields[name]
        raise AttributeError(f"Duration has no field '{name}'")

    def __repr__(self) -> str:
        return f"Duration(hours={self.Hours})"

    def __str__(self) -> str:
        h = int(self._total_seconds // 3600)
        m = int((self._total_seconds % 3600) // 60)
        s = self._total_seconds % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    def __add__(self, other):
        if isinstance(other, Duration):
            return Duration(seconds=self._total_seconds + other._total_seconds)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Duration):
            return Duration(seconds=self._total_seconds - other._total_seconds)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Duration(seconds=self._total_seconds * other)
        return NotImplemented


class CalendarDuration:
    """MATLAB calendarDuration type."""

    def __init__(self, years=0, months=0, days=0):
        self.years = years
        self.months = months
        self.days = days

    def get_field(self, name: str) -> Any:
        fields = {
            "Years": self.years,
            "Months": self.months,
            "Days": self.days,
        }
        if name in fields:
            return fields[name]
        raise AttributeError(f"CalendarDuration has no field '{name}'")

    def __repr__(self) -> str:
        return f"CalendarDuration(years={self.years}, months={self.months}, days={self.days})"

    def __str__(self) -> str:
        parts = []
        if self.years:
            parts.append(f"{self.years}y")
        if self.months:
            parts.append(f"{self.months}m")
        if self.days:
            parts.append(f"{self.days}d")
        return "".join(parts) if parts else "0d"


# ── Categorical Type ───────────────────────────────────────────

class Categorical:
    """MATLAB categorical array type."""

    def __init__(self, data, categories=None):
        if isinstance(data, Mat):
            data = data.data
        data = np.array(data).flatten()
        if categories is None:
            categories = sorted(set(data))
        self._categories = list(categories)
        self._codes = np.array([self._categories.index(x) if x in self._categories else -1 for x in data])
        self._data = data

    @property
    def categories(self) -> list:
        return self._categories

    @property
    def codes(self) -> np.ndarray:
        return self._codes

    def isundefined(self) -> np.ndarray:
        return self._codes == -1

    def get_field(self, name: str) -> Any:
        if name == "categories":
            return self.categories
        if name == "Codes":
            return Mat(self._codes)
        raise AttributeError(f"Categorical has no field '{name}'")

    def __repr__(self) -> str:
        return f"Categorical({len(self._data)} elements, {len(self._categories)} categories)"

    def __str__(self) -> str:
        return " ".join(str(x) for x in self._data[:10]) + (" ..." if len(self._data) > 10 else "")

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return self._data[idx]
        return Categorical(self._data[idx], self._categories)
