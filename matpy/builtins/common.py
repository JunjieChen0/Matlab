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
    result = np.meshgrid(*arrays, indexing="ij")
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
    row_sizes = (
        [int(a) for a in args[0].flat]
        if isinstance(args[0], Mat)
        else [int(a) for a in np.array(args[0]).flat]
    )
    result = []
    start = 0
    for sz in row_sizes:
        result.append([data[start : start + sz]])
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
    return isinstance(
        x, (int, float, complex, np.integer, np.floating, np.complexfloating)
    )


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
    if class_name == "double":
        return isinstance(x, (int, float, Mat))
    elif class_name == "char":
        return isinstance(x, str)
    elif class_name == "logical":
        return isinstance(x, bool)
    elif class_name == "cell":
        return isinstance(x, CellArray)
    elif class_name == "struct":
        return isinstance(x, Struct)
    elif class_name == "string":
        return isinstance(x, StringArray)
    return False


@register("class")
def _class_enhanced(x):
    if isinstance(x, Mat):
        return "double"
    elif isinstance(x, str):
        return "char"
    elif isinstance(x, bool):
        return "logical"
    elif isinstance(x, CellArray):
        return "cell"
    elif isinstance(x, Struct):
        return "struct"
    elif isinstance(x, StringArray):
        return "string"
    return type(x).__name__


@register("cast")
def _cast(x, class_name):
    data = x.data if isinstance(x, Mat) else np.array(x)
    class_name = str(class_name)
    dtype_map = {
        "double": np.float64,
        "single": np.float32,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
    }
    dtype = dtype_map.get(class_name, np.float64)
    return Mat(data.astype(dtype))


@register("intmax")
def _intmax(class_name="int32"):
    class_name = str(class_name)
    dtype_map = {
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
    }
    return np.iinfo(dtype_map.get(class_name, np.int32)).max


@register("intmin")
def _intmin(class_name="int32"):
    class_name = str(class_name)
    dtype_map = {"int32": np.int32, "int64": np.int64, "uint8": np.uint8}
    return np.iinfo(dtype_map.get(class_name, np.int32)).min


@register("realmax")
def _realmax(class_name="double"):
    class_name = str(class_name)
    if class_name == "single":
        return np.finfo(np.float32).max
    return np.finfo(np.float64).max


@register("realmin")
def _realmin(class_name="double"):
    class_name = str(class_name)
    if class_name == "single":
        return np.finfo(np.float32).tiny
    return np.finfo(np.float64).tiny


@register("flintmax")
def _flintmax(class_name="double"):
    if str(class_name) == "single":
        return 2**24
    return 2**53


# ── Explicit Type Conversion Functions ─────────────────────────


@register("int8")
def _int8(x):
    """Convert to 8-bit signed integer."""
    if isinstance(x, Mat):
        return Mat(x.data.astype(np.int8))
    return np.int8(x)


@register("int16")
def _int16(x):
    """Convert to 16-bit signed integer."""
    if isinstance(x, Mat):
        return Mat(x.data.astype(np.int16))
    return np.int16(x)


@register("int32")
def _int32(x):
    """Convert to 32-bit signed integer."""
    if isinstance(x, Mat):
        return Mat(x.data.astype(np.int32))
    return np.int32(x)


@register("int64")
def _int64(x):
    """Convert to 64-bit signed integer."""
    if isinstance(x, Mat):
        return Mat(x.data.astype(np.int64))
    return np.int64(x)


@register("uint8")
def _uint8(x):
    """Convert to 8-bit unsigned integer."""
    if isinstance(x, Mat):
        return Mat(x.data.astype(np.uint8))
    return np.uint8(x)


@register("uint16")
def _uint16(x):
    """Convert to 16-bit unsigned integer."""
    if isinstance(x, Mat):
        return Mat(x.data.astype(np.uint16))
    return np.uint16(x)


@register("uint32")
def _uint32(x):
    """Convert to 32-bit unsigned integer."""
    if isinstance(x, Mat):
        return Mat(x.data.astype(np.uint32))
    return np.uint32(x)


@register("uint64")
def _uint64(x):
    """Convert to 64-bit unsigned integer."""
    if isinstance(x, Mat):
        return Mat(x.data.astype(np.uint64))
    return np.uint64(x)


@register("single")
def _single(x):
    """Convert to single precision."""
    if isinstance(x, Mat):
        return Mat(x.data.astype(np.float32))
    return np.float32(x)


@register("logical")
def _logical(x):
    """Convert to logical (boolean)."""
    if isinstance(x, Mat):
        return Mat(x.data.astype(bool))
    return bool(x)


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
    return Mat(
        np.array([now.year, now.month, now.day, now.hour, now.minute, now.second])
    )


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

    return datetime.datetime.now().strftime("%d-%b-%Y")


@register("datetime")
def _datetime(*args, **kwargs):
    """Create datetime object."""
    from matpy.runtime.types import Datetime

    if len(args) == 0:
        return Datetime()
    elif len(args) == 1 and isinstance(args[0], str):
        return Datetime(args[0])
    elif len(args) >= 3:
        return Datetime(*args)
    return Datetime()


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
                print("  ".join(str(x) for x in data))
            else:
                for row in data:
                    print("  ".join(str(x) for x in row))
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
    if fmt == "short":
        np.set_printoptions(precision=4)
    elif fmt == "long":
        np.set_printoptions(precision=15)
    elif fmt == "shorte":
        np.set_printoptions(formatter={"float_kind": lambda x: f"{x:.4e}"})
    elif fmt == "longe":
        np.set_printoptions(formatter={"float_kind": lambda x: f"{x:.15e}"})
    elif fmt == "compact":
        np.set_printoptions(linewidth=80)
    elif fmt == "loose":
        np.set_printoptions(linewidth=75)


@register("fprintf")
def _fprintf_enhanced(fmt, *args):
    result = str(fmt)
    for a in args:
        if isinstance(a, Mat):
            a = a.to_python()
        result = result.replace("%d", str(int(a)), 1) if "%d" in result else result
        result = result.replace("%f", str(float(a)), 1) if "%f" in result else result
        result = result.replace("%s", str(a), 1) if "%s" in result else result
        result = result.replace("%g", str(float(a)), 1) if "%g" in result else result
        result = result.replace("%e", f"{float(a):e}", 1) if "%e" in result else result
    print(result, end="")
    return len(result)


@register("sprintf")
def _sprintf_enhanced(fmt, *args):
    result = str(fmt)
    for a in args:
        if isinstance(a, Mat):
            a = a.to_python()
        result = result.replace("%d", str(int(a)), 1) if "%d" in result else result
        result = result.replace("%f", str(float(a)), 1) if "%f" in result else result
        result = result.replace("%s", str(a), 1) if "%s" in result else result
        result = result.replace("%g", str(float(a)), 1) if "%g" in result else result
        result = result.replace("%e", f"{float(a):e}", 1) if "%e" in result else result
    return result


@register("num2str")
def _num2str_enhanced(x, fmt=None):
    data = x.data if isinstance(x, Mat) else np.array(x)
    if fmt:
        return " ".join(fmt % v for v in data.flat)
    return " ".join(str(v) for v in data.flat)


@register("int2str")
def _int2str_enhanced(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return " ".join(str(int(v)) for v in data.flat)


@register("str2double")
def _str2double_enhanced(s):
    try:
        return float(str(s).strip())
    except ValueError:
        return float("nan")


@register("str2num")
def _str2num_enhanced(s):
    try:
        parts = str(s).strip().split()
        return Mat(np.array([float(p) for p in parts]))
    except ValueError:
        return Mat(np.array([]))


# ── Array Query Functions ──────────────────────────────────────


@register("isvector")
def _isvector(x):
    """Check if input is a vector."""
    if isinstance(x, Mat):
        data = x.data
    else:
        data = np.array(x)
    return data.ndim == 1 or (
        data.ndim == 2 and (data.shape[0] == 1 or data.shape[1] == 1)
    )


@register("isscalar")
def _isscalar(x):
    """Check if input is scalar."""
    if isinstance(x, Mat):
        data = x.data
    else:
        data = np.array(x)
    return data.ndim == 0 or (data.ndim == 2 and data.shape == (1, 1))


@register("isrow")
def _isrow(x):
    """Check if input is a row vector."""
    if isinstance(x, Mat):
        data = x.data
    else:
        data = np.array(x)
    return data.ndim == 2 and data.shape[0] == 1


@register("iscolumn")
def _iscolumn(x):
    """Check if input is a column vector."""
    if isinstance(x, Mat):
        data = x.data
    else:
        data = np.array(x)
    return data.ndim == 2 and data.shape[1] == 1


@register("ismatrix")
def _ismatrix(x):
    """Check if input is a matrix."""
    if isinstance(x, Mat):
        data = x.data
    else:
        data = np.array(x)
    return data.ndim == 2


@register("isempty")
def _isempty(x):
    """Check if input is empty."""
    if isinstance(x, Mat):
        data = x.data
    else:
        data = np.array(x)
    return data.size == 0


@register("isequal")
def _isequal(*args):
    """Check if arrays are equal."""
    if len(args) < 2:
        return True
    first = args[0].data if isinstance(args[0], Mat) else np.array(args[0])
    for a in args[1:]:
        other = a.data if isinstance(a, Mat) else np.array(a)
        if not np.array_equal(first, other):
            return False
    return True


@register("isequaln")
def _isequaln(*args):
    """Check if arrays are equal (NaN treated as equal)."""
    if len(args) < 2:
        return True
    first = args[0].data if isinstance(args[0], Mat) else np.array(args[0])
    for a in args[1:]:
        other = a.data if isinstance(a, Mat) else np.array(a)
        if not np.array_equal(first, other, equal_nan=True):
            return False
    return True


@register("isfinite")
def _isfinite(x):
    """Check if elements are finite."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.isfinite(data))


@register("isinf")
def _isinf(x):
    """Check if elements are infinite."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.isinf(data))


@register("isnan")
def _isnan(x):
    """Check if elements are NaN."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.isnan(data))


@register("isreal")
def _isreal(x):
    """Check if input is real."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return np.isrealobj(data)


def _isinteger(x):
    """Check if input is integer type."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return np.issubdtype(data.dtype, np.integer)


def _isfloat(x):
    """Check if input is floating point."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return np.issubdtype(data.dtype, np.floating)


@register("islogical")
def _islogical(x):
    """Check if input is logical."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return data.dtype == bool


@register("ischar")
def _ischar(x):
    """Check if input is character array."""
    return isinstance(x, str)


@register("isstring")
def _isstring(x):
    """Check if input is string."""
    return isinstance(x, (str, StringArray))


@register("iscell")
def _iscell(x):
    """Check if input is cell array."""
    return isinstance(x, CellArray)


@register("isstruct")
def _isstruct(x):
    """Check if input is struct."""
    return isinstance(x, Struct)


# ── Set Operations ─────────────────────────────────────────────


@register("unique")
def _unique(x):
    """Unique values."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.unique(data))


@register("union")
def _union(a, b):
    """Set union."""
    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    return Mat(np.union1d(da.flatten(), db.flatten()))


@register("intersect")
def _intersect(a, b):
    """Set intersection."""
    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    return Mat(np.intersect1d(da.flatten(), db.flatten()))


@register("setdiff")
def _setdiff(a, b):
    """Set difference."""
    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    return Mat(np.setdiff1d(da.flatten(), db.flatten()))


@register("setxor")
def _setxor(a, b):
    """Set exclusive or."""
    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    return Mat(np.setxor1d(da.flatten(), db.flatten()))


@register("ismember")
def _ismember(a, b):
    """Check if elements are members of set."""
    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    return Mat(np.isin(da, db))


# ── Sorting ────────────────────────────────────────────────────


@register("sort")
def _sort(x, dim=None, mode="ascend"):
    """Sort array."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    if dim is not None:
        axis = int(dim) - 1
    else:
        axis = None
    if mode == "descend":
        if axis is None:
            return Mat(-np.sort(-data.flatten()))
        return Mat(-np.sort(-data, axis=axis))
    if axis is None:
        return Mat(np.sort(data.flatten()))
    return Mat(np.sort(data, axis=axis))


@register("sortrows")
def _sortrows(x, col=None):
    """Sort rows of matrix."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    if col is not None:
        col = int(col) - 1
        idx = np.argsort(data[:, col])
    else:
        idx = np.lexsort(data.T[::-1])
    return Mat(data[idx])


@register("find")
def _find(x):
    """Find nonzero elements."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    indices = np.nonzero(data.flatten())[0]
    return Mat(indices + 1)  # 1-based indexing


# ── Datetime Functions ─────────────────────────────────────────


def _datetime(*args, **kwargs):
    """Create datetime object."""
    from matpy.runtime.types import Datetime

    if len(args) == 0:
        return Datetime()
    elif len(args) == 1 and isinstance(args[0], str):
        return Datetime(args[0])
    elif len(args) >= 3:
        return Datetime(*args)
    return Datetime()


@register("duration")
def _duration(hours=0, minutes=0, seconds=0):
    """Create duration object."""
    from matpy.runtime.types import Duration

    return Duration(hours=float(hours), minutes=float(minutes), seconds=float(seconds))


@register("years")
def _years(x):
    """Convert to years duration."""
    from matpy.runtime.types import CalendarDuration

    return CalendarDuration(years=int(x))


@register("months")
def _months(x):
    """Convert to months duration."""
    from matpy.runtime.types import CalendarDuration

    return CalendarDuration(months=int(x))


@register("days")
def _days(x):
    """Convert to days duration."""
    from matpy.runtime.types import CalendarDuration

    return CalendarDuration(days=int(x))


@register("hours")
def _hours(x):
    """Convert to hours duration."""
    from matpy.runtime.types import Duration

    return Duration(hours=float(x))


@register("minutes")
def _minutes(x):
    """Convert to minutes duration."""
    from matpy.runtime.types import Duration

    return Duration(minutes=float(x))


@register("seconds")
def _seconds(x):
    """Convert to seconds duration."""
    from matpy.runtime.types import Duration

    return Duration(seconds=float(x))


@register("between")
def _between(t1, t2):
    """Time between two datetimes."""
    if isinstance(t1, Datetime) and isinstance(t2, Datetime):
        return t2 - t1
    return None


@register("year")
def _year(dt):
    """Extract year from datetime."""
    if isinstance(dt, Datetime):
        return dt.Year
    import datetime

    return datetime.datetime.now().year


@register("month")
def _month(dt):
    """Extract month from datetime."""
    if isinstance(dt, Datetime):
        return dt.Month
    import datetime

    return datetime.datetime.now().month


@register("day")
def _day(dt):
    """Extract day from datetime."""
    if isinstance(dt, Datetime):
        return dt.Day
    import datetime

    return datetime.datetime.now().day


@register("hour")
def _hour(dt):
    """Extract hour from datetime."""
    if isinstance(dt, Datetime):
        return dt.Hour
    return 0


@register("minute")
def _minute(dt):
    """Extract minute from datetime."""
    if isinstance(dt, Datetime):
        return dt.Minute
    return 0


@register("second")
def _second(dt):
    """Extract second from datetime."""
    if isinstance(dt, Datetime):
        return dt.Second
    return 0


@register("weekday")
def _weekday(dt):
    """Day of week."""
    if isinstance(dt, Datetime):
        return dt._dt.weekday() + 1
    return 1


@register("isdatetime")
def _isdatetime(x):
    """Check if input is datetime."""
    from matpy.runtime.types import Datetime

    return isinstance(x, Datetime)


@register("isduration")
def _isduration(x):
    """Check if input is duration."""
    from matpy.runtime.types import Duration

    return isinstance(x, Duration)


# ── Categorical Functions ──────────────────────────────────────


@register("categorical")
def _categorical(x, categories=None):
    """Create categorical array."""
    from matpy.runtime.types import Categorical

    return Categorical(x, categories)


@register("categories")
def _categories(x):
    """Get categories."""
    if isinstance(x, Categorical):
        return x.categories
    return []


@register("isundefined")
def _isundefined(x):
    """Check for undefined categorical elements."""
    if isinstance(x, Categorical):
        return Mat(x.isundefined())
    return Mat(np.array([]))


@register("iscategorical")
def _iscategorical(x):
    """Check if input is categorical."""
    from matpy.runtime.types import Categorical

    return isinstance(x, Categorical)


@register("addcats")
def _addcats(x, newcats):
    """Add categories to categorical array."""
    if isinstance(x, Categorical):
        cats = x.categories.copy()
        if isinstance(newcats, (list, Mat)):
            if isinstance(newcats, Mat):
                newcats = newcats.data.flatten().tolist()
            cats.extend(newcats)
        else:
            cats.append(newcats)
        return Categorical(x._data, cats)
    return x


@register("removecats")
def _removecats(x, rmcats):
    """Remove categories from categorical array."""
    if isinstance(x, Categorical):
        cats = x.categories.copy()
        if isinstance(rmcats, (list, Mat)):
            if isinstance(rmcats, Mat):
                rmcats = rmcats.data.flatten().tolist()
            for c in rmcats:
                if c in cats:
                    cats.remove(c)
        else:
            if rmcats in cats:
                cats.remove(rmcats)
        return Categorical(x._data, cats)
    return x


@register("mergecats")
def _mergecats(x, cats, newname):
    """Merge categories."""
    if isinstance(x, Categorical):
        cats_list = x.categories.copy()
        if isinstance(cats, (list, Mat)):
            if isinstance(cats, Mat):
                cats = cats.data.flatten().tolist()
            # Remove old categories and add new one
            for c in cats:
                if c in cats_list:
                    cats_list.remove(c)
            cats_list.append(str(newname))
        return Categorical(x._data, cats_list)
    return x


@register("renamecats")
def _renamecats(x, oldnames, newnames):
    """Rename categories."""
    if isinstance(x, Categorical):
        cats = x.categories.copy()
        if isinstance(oldnames, (list, Mat)):
            if isinstance(oldnames, Mat):
                oldnames = oldnames.data.flatten().tolist()
            if isinstance(newnames, Mat):
                newnames = newnames.data.flatten().tolist()
            for old, new in zip(oldnames, newnames):
                if old in cats:
                    idx = cats.index(old)
                    cats[idx] = new
        return Categorical(x._data, cats)
    return x
