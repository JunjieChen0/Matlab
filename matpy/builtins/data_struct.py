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


@register("numel")
def _numel(x):
    if isinstance(x, CellArray):
        return x.numel()
    if isinstance(x, Struct):
        return 1
    if isinstance(x, Mat):
        return x.data.size
    return 1


@register("iscell")
def _iscell(x):
    return isinstance(x, CellArray)


@register("isstruct")
def _isstruct(x):
    return isinstance(x, Struct)


@register("celldisp")
def _celldisp(c):
    if isinstance(c, CellArray):
        for i, row in enumerate(c._data):
            for j, val in enumerate(row):
                print(f"{{{i+1},{j+1}}}:")
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
