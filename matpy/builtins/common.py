"""Common built-in functions for MatPy."""

import numpy as np
from matpy.builtins import register
from matpy.runtime.types import Mat, CellArray, Struct, StringArray


# ── Array Creation ────────────────────────────────────────────

@register("meshgrid")
def _meshgrid(*args):
    arrays = [a.data if isinstance(a, Mat) else np.array(a).flatten() for a in args]
    result = np.meshgrid(*arrays)
    return tuple(Mat(r) for r in result)


@register("ndgrid")
def _ndgrid(*args):
    arrays = [a.data if isinstance(a, Mat) else np.array(a).flatten() for a in args]
    result = np.meshgrid(*arrays, indexing='ij')
    return tuple(Mat(r) for r in result)


@register("blkdiag")
def _blkdiag(*args):
    from scipy.linalg import block_diag
    arrays = [a.data if isinstance(a, Mat) else np.array(a) for a in args]
    return Mat(block_diag(*arrays))


@register("toeplitz")
def _toeplitz(c, r=None):
    from scipy.linalg import toeplitz
    cd = c.data if isinstance(c, Mat) else np.array(c).flatten()
    if r is not None:
        rd = r.data if isinstance(r, Mat) else np.array(r).flatten()
        return Mat(toeplitz(cd, rd))
    return Mat(toeplitz(cd))


@register("hankel")
def _hankel(c, r=None):
    from scipy.linalg import hankel
    cd = c.data if isinstance(c, Mat) else np.array(c).flatten()
    if r is not None:
        rd = r.data if isinstance(r, Mat) else np.array(r).flatten()
        return Mat(hankel(cd, rd))
    return Mat(hankel(cd))


@register("vander")
def _vander(x, n=None):
    xd = x.data if isinstance(x, Mat) else np.array(x).flatten()
    if n is None:
        n = len(xd)
    return Mat(np.vander(xd, int(n)))


@register("pascal")
def _pascal(n, kind=1):
    from scipy.linalg import pascal
    return Mat(pascal(int(n), kind=int(kind)))


# ── Array Manipulation ────────────────────────────────────────

@register("cell2mat")
def _cell2mat(c):
    if isinstance(c, CellArray):
        rows = []
        for row in c._data:
            row_vals = []
            for val in row:
                if isinstance(val, Mat):
                    row_vals.append(val.data)
                else:
                    row_vals.append(np.array(val))
            rows.append(np.concatenate(row_vals))
        return Mat(np.vstack(rows))
    return c


@register("mat2cell")
def _mat2cell(x, *args):
    data = x.data if isinstance(x, Mat) else np.array(x)
    if len(args) == 0:
        return CellArray([[data]])
    row_sizes = [int(a) for a in args[0].flat] if isinstance(args[0], Mat) else [int(a) for a in np.array(args[0]).flat]
    result = []
    start = 0
    for sz in row_sizes:
        result.append([data[start:start+sz]])
        start += sz
    return CellArray(result)


@register("num2cell")
def _num2cell(x, dim=None):
    data = x.data if isinstance(x, Mat) else np.array(x)
    if dim is not None:
        d = int(dim) - 1
        result = []
        for i in range(data.shape[d]):
            result.append([np.take(data, i, axis=d)])
        return CellArray(result)
    result = []
    for val in data.flat:
        result.append([val])
    return CellArray(result)


@register("arrayfun")
def _arrayfun(func, *args):
    if len(args) == 1:
        data = args[0].data if isinstance(args[0], Mat) else np.array(args[0])
        result = np.array([func(x) for x in data.flat]).reshape(data.shape)
        return Mat(result)
    return func(*args)


# ── Type Checking ─────────────────────────────────────────────

@register("isnumeric")
def _isnumeric_enhanced(x):
    if isinstance(x, Mat):
        return np.issubdtype(x.dtype, np.number)
    return isinstance(x, (int, float, complex, np.integer, np.floating, np.complexfloating))


@register("isfloat")
def _isfloat(x):
    if isinstance(x, Mat):
        return np.issubdtype(x.dtype, np.floating)
    return isinstance(x, (float, np.floating))


@register("isinteger")
def _isinteger(x):
    if isinstance(x, Mat):
        return np.issubdtype(x.dtype, np.integer)
    return isinstance(x, (int, np.integer))


@register("isa")
def _isa_enhanced(x, class_name):
    class_name = str(class_name)
    if class_name == 'double':
        return isinstance(x, (int, float, Mat))
    elif class_name == 'char':
        return isinstance(x, str)
    elif class_name == 'logical':
        return isinstance(x, bool)
    elif class_name == 'cell':
        return isinstance(x, CellArray)
    elif class_name == 'struct':
        return isinstance(x, Struct)
    elif class_name == 'string':
        return isinstance(x, StringArray)
    return False


@register("class")
def _class_enhanced(x):
    if isinstance(x, Mat):
        return 'double'
    elif isinstance(x, str):
        return 'char'
    elif isinstance(x, bool):
        return 'logical'
    elif isinstance(x, CellArray):
        return 'cell'
    elif isinstance(x, Struct):
        return 'struct'
    elif isinstance(x, StringArray):
        return 'string'
    return type(x).__name__


@register("cast")
def _cast(x, class_name):
    data = x.data if isinstance(x, Mat) else np.array(x)
    class_name = str(class_name)
    dtype_map = {
        'double': np.float64, 'single': np.float32,
        'int32': np.int32, 'int64': np.int64,
        'uint8': np.uint8, 'uint16': np.uint16, 'uint32': np.uint32,
    }
    dtype = dtype_map.get(class_name, np.float64)
    return Mat(data.astype(dtype))


@register("intmax")
def _intmax(class_name='int32'):
    class_name = str(class_name)
    dtype_map = {'int32': np.int32, 'int64': np.int64, 'uint8': np.uint8, 'uint16': np.uint16, 'uint32': np.uint32}
    return np.iinfo(dtype_map.get(class_name, np.int32)).max


@register("intmin")
def _intmin(class_name='int32'):
    class_name = str(class_name)
    dtype_map = {'int32': np.int32, 'int64': np.int64, 'uint8': np.uint8}
    return np.iinfo(dtype_map.get(class_name, np.int32)).min


@register("realmax")
def _realmax(class_name='double'):
    class_name = str(class_name)
    if class_name == 'single':
        return np.finfo(np.float32).max
    return np.finfo(np.float64).max


@register("realmin")
def _realmin(class_name='double'):
    class_name = str(class_name)
    if class_name == 'single':
        return np.finfo(np.float32).tiny
    return np.finfo(np.float64).tiny


@register("flintmax")
def _flintmax(class_name='double'):
    if str(class_name) == 'single':
        return 2**24
    return 2**53


@register("eps")
def _eps(x=None):
    if x is not None:
        data = x.data if isinstance(x, Mat) else np.array(x)
        return np.finfo(data.dtype).eps
    return np.finfo(np.float64).eps


# ── Timing ────────────────────────────────────────────────────

@register("tic")
def _tic():
    import time
    return time.time()


@register("toc")
def _toc(start=None):
    import time
    if start is not None:
        return time.time() - start
    return time.time()


@register("cputime")
def _cputime():
    import time
    return time.process_time()


@register("clock")
def _clock():
    import datetime
    now = datetime.datetime.now()
    return Mat(np.array([now.year, now.month, now.day, now.hour, now.minute, now.second]))


@register("now")
def _now():
    import datetime
    epoch = datetime.datetime(1900, 1, 1)
    now = datetime.datetime.now()
    delta = now - epoch
    return delta.days + delta.seconds / 86400 + 693960


@register("date")
def _date():
    import datetime
    return datetime.datetime.now().strftime('%d-%b-%Y')


@register("datetime")
def _datetime():
    import datetime
    return str(datetime.datetime.now())


@register("pause")
def _pause(seconds=None):
    import time
    if seconds is not None:
        time.sleep(float(seconds))
    else:
        time.sleep(0.01)


# ── Display ────────────────────────────────────────────────────

@register("disp")
def _disp_enhanced(*args):
    for arg in args:
        if isinstance(arg, Mat):
            data = arg.data
            if data.ndim == 0:
                print(data.item())
            elif data.ndim == 1:
                print('  '.join(str(x) for x in data))
            else:
                for row in data:
                    print('  '.join(str(x) for x in row))
        elif isinstance(arg, Struct):
            print(arg)
        elif isinstance(arg, CellArray):
            print(arg)
        else:
            print(arg)


@register("display")
def _display(x, name=None):
    if name:
        print(f"{name} =")
    _disp_enhanced(x)


@register("format")
def _format(fmt=None):
    if fmt is None:
        return
    import numpy as np
    fmt = str(fmt).lower()
    if fmt == 'short':
        np.set_printoptions(precision=4)
    elif fmt == 'long':
        np.set_printoptions(precision=15)
    elif fmt == 'shorte':
        np.set_printoptions(formatter={'float_kind': lambda x: f'{x:.4e}'})
    elif fmt == 'longe':
        np.set_printoptions(formatter={'float_kind': lambda x: f'{x:.15e}'})
    elif fmt == 'compact':
        np.set_printoptions(linewidth=80)
    elif fmt == 'loose':
        np.set_printoptions(linewidth=75)


@register("fprintf")
def _fprintf_enhanced(fmt, *args):
    result = str(fmt)
    for a in args:
        if isinstance(a, Mat):
            a = a.to_python()
        result = result.replace('%d', str(int(a)), 1) if '%d' in result else result
        result = result.replace('%f', str(float(a)), 1) if '%f' in result else result
        result = result.replace('%s', str(a), 1) if '%s' in result else result
        result = result.replace('%g', str(float(a)), 1) if '%g' in result else result
        result = result.replace('%e', f'{float(a):e}', 1) if '%e' in result else result
    print(result, end='')
    return len(result)


@register("sprintf")
def _sprintf_enhanced(fmt, *args):
    result = str(fmt)
    for a in args:
        if isinstance(a, Mat):
            a = a.to_python()
        result = result.replace('%d', str(int(a)), 1) if '%d' in result else result
        result = result.replace('%f', str(float(a)), 1) if '%f' in result else result
        result = result.replace('%s', str(a), 1) if '%s' in result else result
        result = result.replace('%g', str(float(a)), 1) if '%g' in result else result
        result = result.replace('%e', f'{float(a):e}', 1) if '%e' in result else result
    return result


@register("num2str")
def _num2str_enhanced(x, fmt=None):
    data = x.data if isinstance(x, Mat) else np.array(x)
    if fmt:
        return ' '.join(fmt % v for v in data.flat)
    return ' '.join(str(v) for v in data.flat)


@register("int2str")
def _int2str_enhanced(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return ' '.join(str(int(v)) for v in data.flat)


@register("str2double")
def _str2double_enhanced(s):
    try:
        return float(str(s).strip())
    except ValueError:
        return float('nan')


@register("str2num")
def _str2num_enhanced(s):
    try:
        parts = str(s).strip().split()
        return Mat(np.array([float(p) for p in parts]))
    except ValueError:
        return Mat(np.array([]))
