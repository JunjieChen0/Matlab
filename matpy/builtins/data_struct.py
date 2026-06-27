"""Cell array and struct built-in functions for MatPy."""

import numpy as np
from matpy.builtins import register
from matpy.runtime.types import Mat, CellArray, Struct


@register("cell")
def _cell(*args):
    if len(args) == 1:
        n = int(args[0])
        return CellArray([[None] * n])
    elif len(args) == 2:
        rows = int(args[0])
        cols = int(args[1])
        return CellArray([[None] * cols for _ in range(rows)])
    return CellArray()


@register("struct")
def _struct(*args):
    s = Struct()
    i = 0
    while i < len(args):
        if isinstance(args[i], str):
            name = args[i]
            val = args[i + 1] if i + 1 < len(args) else None
            s.set_field(name, val)
            i += 2
        else:
            i += 1
    return s


@register("fieldnames")
def _fieldnames(s):
    if isinstance(s, Struct):
        return s.field_names()
    return []


@register("isfield")
def _isfield(s, name):
    if isinstance(s, Struct):
        return s.has_field(str(name))
    return False


@register("rmfield")
def _rmfield(s, name):
    if isinstance(s, Struct):
        new_s = Struct(dict(s._fields))
        field = str(name)
        if field in new_s._fields:
            del new_s._fields[field]
        return new_s
    return s


def _numel(x):
    if isinstance(x, CellArray):
        return x.numel()
    if isinstance(x, Struct):
        return 1
    if isinstance(x, Mat):
        return x.data.size
    return 1


def _iscell(x):
    return isinstance(x, CellArray)


def _isstruct(x):
    return isinstance(x, Struct)


@register("celldisp")
def _celldisp(c):
    if isinstance(c, CellArray):
        for i, row in enumerate(c._data):
            for j, val in enumerate(row):
                print(f"{{{i + 1},{j + 1}}}:")
                if isinstance(val, Mat):
                    print(f"  {val.data}")
                else:
                    print(f"  {val}")


@register("cellfun")
def _cellfun(func, c):
    if isinstance(c, CellArray):
        results = []
        for row in c._data:
            for val in row:
                if callable(func):
                    results.append(func(val))
        return Mat(np.array(results))
    return c


@register("deal")
def _deal(*args):
    if len(args) == 1:
        return args[0]
    return list(args)


@register("table")
def _table(*args, **kwargs):
    """Create a table from variables.
    Usage: table(col1, col2, ..., 'VariableNames', {'name1', 'name2', ...})
    """
    from matpy.runtime.types import Table, CellArray

    data = {}

    # Extract VariableNames if provided
    var_names = None
    if "VariableNames" in kwargs:
        vn = kwargs.pop("VariableNames")
        if isinstance(vn, CellArray):
            flat = [item for row in vn._data for item in row]
            var_names = [str(v) for v in flat]
        elif isinstance(vn, (list, np.ndarray)):
            var_names = [str(v) for v in vn]
        else:
            var_names = [str(vn)]

    # Check for VariableNames in positional args
    clean_args = []
    i = 0
    while i < len(args):
        if (
            isinstance(args[i], str)
            and args[i] == "VariableNames"
            and i + 1 < len(args)
        ):
            vn = args[i + 1]
            if isinstance(vn, CellArray):
                flat = [item for row in vn._data for item in row]
                var_names = [str(v) for v in flat]
            elif isinstance(vn, (list, np.ndarray)):
                var_names = [str(v) for v in vn]
            i += 2
        else:
            clean_args.append(args[i])
            i += 1

    # Create table from positional args (data columns)
    for idx, val in enumerate(clean_args):
        name = var_names[idx] if var_names and idx < len(var_names) else f"Var{idx + 1}"
        if isinstance(val, Mat):
            data[name] = val.data
        elif isinstance(val, CellArray):
            flat = [item for row in val._data for item in row]
            data[name] = np.array(flat)
        else:
            data[name] = np.array(val)

    return Table(data)


@register("array2table")
def _array2table(A, *args, **kwargs):
    """Convert array to table."""
    from matpy.runtime.types import Table

    data = A.data if isinstance(A, Mat) else np.array(A)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    # Generate column names
    if "VariableNames" in kwargs:
        names = kwargs["VariableNames"]
    elif args:
        names = [str(a) for a in args]
    else:
        names = [f"Var{i + 1}" for i in range(data.shape[1])]
    table_data = {names[i]: data[:, i] for i in range(min(len(names), data.shape[1]))}
    return Table(table_data)


@register("table2array")
def _table2array(T):
    """Convert table to array."""
    from matpy.runtime.types import Table

    if isinstance(T, Table):
        arrays = list(T._data.values())
        if arrays:
            return Mat(np.column_stack(arrays))
    return Mat(np.array([]))


@register("containers_Map")
def _containers_map(*args, **kwargs):
    """Create a containers.Map object."""
    from matpy.runtime.types import Map

    if len(args) >= 2:
        return Map(args[0], args[1])
    return Map(**kwargs)


@register("isKey")
def _iskey(m, key):
    """Check if key exists in Map."""
    from matpy.runtime.types import Map

    if isinstance(m, Map):
        return m.isKey(key)
    return False


@register("keys")
def _keys(m):
    """Get keys from Map."""
    from matpy.runtime.types import Map

    if isinstance(m, Map):
        return m.keys()
    return []


@register("values")
def _values(m):
    """Get values from Map."""
    from matpy.runtime.types import Map

    if isinstance(m, Map):
        return m.values()
    return []


# ── Additional Data Structure Functions ────────────────────────


def _cell2mat(c):
    """Convert cell array to matrix."""
    if isinstance(c, CellArray):
        data = []
        for row in c._data:
            for item in row:
                if isinstance(item, Mat):
                    data.append(item.data)
                else:
                    data.append(item)
        if data:
            return Mat(np.array(data))
    return Mat(np.array([]))


def _mat2cell(x, *args):
    """Convert matrix to cell array."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    if len(args) == 0:
        # Each element becomes a cell
        cells = [[item] for item in data.flat]
        return CellArray(cells)
    elif len(args) == 1:
        # Split into rows
        n = int(args[0])
        cells = [data[i : i + n, :].tolist() for i in range(0, data.shape[0], n)]
        return CellArray(cells)
    return CellArray()


def _num2cell(x, dim=None):
    """Convert array to cell array."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    if dim is not None:
        dim = int(dim) - 1
        cells = []
        for i in range(data.shape[dim]):
            cells.append([np.take(data, i, axis=dim)])
        return CellArray(cells)
    else:
        cells = [[item] for item in data.flat]
        return CellArray(cells)


def _cellfun(func, c):
    """Apply function to each cell."""
    if isinstance(c, CellArray):
        results = []
        for row in c._data:
            for item in row:
                results.append(func(item))
        return Mat(np.array(results))
    return Mat(np.array([]))


@register("struct2cell")
def _struct2cell(s):
    """Convert struct to cell array."""
    if isinstance(s, Struct):
        values = []
        for name in s.field_names():
            values.append([s.get_field(name)])
        return CellArray(values)
    return CellArray()


@register("cell2struct")
def _cell2struct(c, names):
    """Convert cell array to struct."""
    if isinstance(c, CellArray) and isinstance(names, (list, Mat)):
        if isinstance(names, Mat):
            names = names.data.flatten().tolist()
        s = Struct()
        flat = [item for row in c._data for item in row]
        for i, name in enumerate(names):
            if i < len(flat):
                s.set_field(str(name), flat[i])
        return s
    return Struct()


def _table(*args, **kwargs):
    """Create a table."""
    from matpy.runtime.types import Table

    if kwargs:
        return Table(kwargs)
    elif len(args) >= 2:
        # table(data, 'VariableNames', names)
        data = args[0]
        if isinstance(data, Mat):
            data = data.data
        names = args[1] if len(args) > 1 else None
        if isinstance(names, (list, Mat)):
            if isinstance(names, Mat):
                names = names.data.flatten().tolist()
            table_data = {
                str(names[i]): data[:, i] for i in range(min(len(names), data.shape[1]))
            }
            return Table(table_data)
    return Table({})


def _array2table(x, **kwargs):
    """Convert array to table."""
    from matpy.runtime.types import Table

    data = x.data if isinstance(x, Mat) else np.array(x)
    names = kwargs.get("VariableNames", [f"Var{i + 1}" for i in range(data.shape[1])])
    if isinstance(names, Mat):
        names = names.data.flatten().tolist()
    table_data = {
        str(names[i]): data[:, i] for i in range(min(len(names), data.shape[1]))
    }
    return Table(table_data)


def _table2array(T):
    """Convert table to array."""
    from matpy.runtime.types import Table

    if isinstance(T, Table):
        arrays = list(T._data.values())
        if arrays:
            return Mat(np.column_stack(arrays))
    return Mat(np.array([]))


def _containers_map(*args, **kwargs):
    """Create a containers.Map object."""
    from matpy.runtime.types import Map

    if len(args) >= 2:
        return Map(args[0], args[1])
    return Map(**kwargs)


@register("remove")
def _remove(m, key):
    """Remove key from Map."""
    from matpy.runtime.types import Map

    if isinstance(m, Map):
        m.remove(key)
    return m
