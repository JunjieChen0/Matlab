"""I/O built-in functions for MatPy."""

import sys
import numpy as np
from matpy.builtins import register
from matpy.runtime.types import Mat


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


# ── Additional I/O Functions ───────────────────────────────────


@register("inputname")
def _inputname(n):
    """Get name of input argument."""
    return f"argin{n}"


def _nargout(func=None):
    """Number of function output arguments."""
    if func is None:
        return 0
    return 1  # Simplified


@register("varargin")
def _varargin():
    """Variable length input argument list."""
    return []


@register("varargout")
def _varargout():
    """Variable length output argument list."""
    return []


@register("validatestring")
def _validatestring(s, valid_strings):
    """Validate string value."""
    s = str(s).lower()
    if isinstance(valid_strings, (list, Mat)):
        if isinstance(valid_strings, Mat):
            valid_strings = valid_strings.data.flatten().tolist()
        for vs in valid_strings:
            if str(vs).lower().startswith(s):
                return str(vs)
    return str(s)


@register("validateattributes")
def _validateattributes(x, classes, attributes=None):
    """Validate array attributes."""
    data = x.data if isinstance(x, Mat) else np.array(x)

    # Validate class type
    if isinstance(classes, (list, tuple)):
        class_strs = [str(c).lower() for c in classes]
    else:
        class_strs = [str(classes).lower()]

    # Basic type checking
    dtype_str = str(data.dtype).lower()
    type_ok = False
    for cls in class_strs:
        if cls in ("double", "float64", "float") and np.issubdtype(
            data.dtype, np.floating
        ):
            type_ok = True
        elif cls in ("single", "float32") and data.dtype == np.float32:
            type_ok = True
        elif cls in ("int32", "int") and np.issubdtype(data.dtype, np.integer):
            type_ok = True
        elif cls in ("logical", "bool") and data.dtype == bool:
            type_ok = True
        elif cls in ("char", "string") and isinstance(x, str):
            type_ok = True
        elif cls == "numeric" and np.issubdtype(data.dtype, np.number):
            type_ok = True

    if not type_ok:
        raise ValueError(f"Invalid type. Expected one of {class_strs}, got {dtype_str}")

    # Basic attribute validation
    if attributes:
        if isinstance(attributes, (list, tuple)):
            for attr in attributes:
                attr_str = str(attr).lower()
                if attr_str == "nonempty" and data.size == 0:
                    raise ValueError("Array must be nonempty")
                elif attr_str == "positive" and np.any(data <= 0):
                    raise ValueError("All elements must be positive")
                elif attr_str == "nonnegative" and np.any(data < 0):
                    raise ValueError("All elements must be nonnegative")
                elif attr_str == "finite" and not np.all(np.isfinite(data)):
                    raise ValueError("All elements must be finite")
                elif attr_str == "integer" and not np.all(data == np.floor(data)):
                    raise ValueError("All elements must be integers")

    return True


@register("inputParser")
class _InputParser:
    """Input parser for function arguments."""

    def __init__(self):
        self._params = {}
        self._results = {}

    def addParameter(self, name, default=None):
        self._params[name] = default

    def addRequired(self, name, validator=None):
        self._params[name] = None

    def parse(self, *args, **kwargs):
        # Set defaults
        for name, default in self._params.items():
            self._results[name] = default
        # Parse positional arguments
        param_names = list(self._params.keys())
        for i, arg in enumerate(args):
            if i < len(param_names):
                self._results[param_names[i]] = arg
        # Parse keyword arguments
        for name, value in kwargs.items():
            if name in self._results:
                self._results[name] = value
        return self._results

    def Results(self):
        return self._results


@register("onCleanup")
class _OnCleanup:
    """Cleanup object."""

    def __init__(self, func):
        self._func = func

    def __del__(self):
        self._func()


@register("timer")
class _Timer:
    """Timer object."""

    def __init__(self, *args, **kwargs):
        self._start = None

    def start(self):
        import time

        self._start = time.time()

    def stop(self):
        pass

    def tic(self):
        import time

        self._start = time.time()

    def toc(self):
        import time

        if self._start is not None:
            return time.time() - self._start
        return 0.0


# ── MEX Interface Functions ────────────────────────────────────


@register("py.numpy.array")
def _py_numpy_array(*args):
    """Create numpy array."""
    import numpy as np

    return Mat(np.array(*args))


@register("py.numpy.zeros")
def _py_numpy_zeros(*args):
    """Create numpy zeros array."""
    import numpy as np

    return Mat(np.zeros(*args))


@register("py.numpy.ones")
def _py_numpy_ones(*args):
    """Create numpy ones array."""
    import numpy as np

    return Mat(np.ones(*args))


@register("py.numpy.linspace")
def _py_numpy_linspace(*args):
    """Create numpy linspace array."""
    import numpy as np

    return Mat(np.linspace(*args))


@register("py.numpy.reshape")
def _py_numpy_reshape(arr, shape):
    """Reshape numpy array."""
    import numpy as np

    if isinstance(arr, Mat):
        return Mat(arr.data.reshape(*shape))
    return Mat(np.array(arr).reshape(*shape))


@register("py.numpy.dot")
def _py_numpy_dot(a, b):
    """Compute dot product."""
    import numpy as np

    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    return Mat(np.dot(da, db))


@register("py.numpy.cross")
def _py_numpy_cross(a, b):
    """Compute cross product."""
    import numpy as np

    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    return Mat(np.cross(da, db))


@register("py.numpy.linalg.norm")
def _py_numpy_linalg_norm(x, ord=None):
    """Compute matrix or vector norm."""
    import numpy as np

    data = x.data if isinstance(x, Mat) else np.array(x)
    return np.linalg.norm(data, ord)


@register("py.numpy.linalg.inv")
def _py_numpy_linalg_inv(x):
    """Compute matrix inverse."""
    import numpy as np

    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.linalg.inv(data))


@register("py.numpy.linalg.det")
def _py_numpy_linalg_det(x):
    """Compute matrix determinant."""
    import numpy as np

    data = x.data if isinstance(x, Mat) else np.array(x)
    return np.linalg.det(data)


@register("py.numpy.linalg.eig")
def _py_numpy_linalg_eig(x):
    """Compute eigenvalues and eigenvectors."""
    import numpy as np

    data = x.data if isinstance(x, Mat) else np.array(x)
    eigenvalues, eigenvectors = np.linalg.eig(data)
    return Mat(eigenvalues), Mat(eigenvectors)


@register("py.numpy.linalg.svd")
def _py_numpy_linalg_svd(x):
    """Compute singular value decomposition."""
    import numpy as np

    data = x.data if isinstance(x, Mat) else np.array(x)
    U, S, Vt = np.linalg.svd(data)
    return Mat(U), Mat(S), Mat(Vt)


@register("py.scipy.linalg.expm")
def _py_scipy_linalg_expm(x):
    """Compute matrix exponential."""
    from scipy.linalg import expm
    import numpy as np

    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(expm(data))


@register("py.scipy.linalg.logm")
def _py_scipy_linalg_logm(x):
    """Compute matrix logarithm."""
    from scipy.linalg import logm
    import numpy as np

    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(logm(data))


@register("py.scipy.linalg.sqrtm")
def _py_scipy_linalg_sqrtm(x):
    """Compute matrix square root."""
    from scipy.linalg import sqrtm
    import numpy as np

    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(sqrtm(data))


@register("py.scipy.optimize.minimize")
def _py_scipy_optimize_minimize(func, x0, *args, **kwargs):
    """Minimize a function."""
    from scipy.optimize import minimize
    import numpy as np

    x0_data = x0.data if isinstance(x0, Mat) else np.array(x0)
    result = minimize(func, x0_data.flatten(), *args, **kwargs)
    return Mat(result.x), float(result.fun)


@register("py.scipy.integrate.quad")
def _py_scipy_integrate_quad(func, a, b, *args, **kwargs):
    """Compute a definite integral."""
    from scipy.integrate import quad

    result, error = quad(func, float(a), float(b), *args, **kwargs)
    return float(result), float(error)


@register("py.scipy.interpolate.interp1d")
def _py_scipy_interpolate_interp1d(x, y, kind="linear"):
    """Create a 1-D interpolation function."""
    from scipy.interpolate import interp1d
    import numpy as np

    xd = x.data if isinstance(x, Mat) else np.array(x)
    yd = y.data if isinstance(y, Mat) else np.array(y)
    return interp1d(xd.flatten(), yd.flatten(), kind=str(kind))


@register("py.matplotlib.pyplot.plot")
def _py_matplotlib_pyplot_plot(*args):
    """Create a plot."""
    import matplotlib.pyplot as plt
    import numpy as np

    processed_args = []
    for arg in args:
        if isinstance(arg, Mat):
            processed_args.append(arg.data.flatten())
        else:
            processed_args.append(np.array(arg).flatten())
    plt.plot(*processed_args)
    return None


@register("py.matplotlib.pyplot.show")
def _py_matplotlib_pyplot_show():
    """Show the current plot."""
    import matplotlib.pyplot as plt

    plt.show()
    return None


@register("py.matplotlib.pyplot.savefig")
def _py_matplotlib_pyplot_savefig(filename, **kwargs):
    """Save the current plot to a file."""
    import matplotlib.pyplot as plt

    plt.savefig(str(filename), **kwargs)
    return None


@register("rethrow")
def _rethrow(exc):
    """Rethrow a caught MException.
    Usage: try ... catch e; rethrow(e); end
    """
    from matpy.runtime.types import MException

    if isinstance(exc, MException):
        exc.throw()
    raise RuntimeError(str(exc))


# Global variable to store last error
_last_error = None


@register("lasterror")
def _lasterror():
    """Return last error as MException.

    Usage: e = lasterror()
    """
    return _last_error


@register("warning")
def _warning(*args):
    """Display warning message.

    Usage:
        warning(msg) - Display warning message
        warning('off', id) - Turn off warning
        warning('on', id) - Turn on warning
        warning('error', id) - Turn warning into error
    """
    if len(args) == 0:
        return

    if len(args) == 1:
        # Simple warning message
        msg = str(args[0])
        print(f"Warning: {msg}", file=sys.stderr)
        return

    if len(args) >= 2:
        action = str(args[0]).lower()
        warning_id = str(args[1])  # noqa: F841

        if action == "off":
            # Suppress warning (store in global state)
            pass
        elif action == "on":
            # Enable warning
            pass
        elif action == "error":
            # Convert warning to error
            pass
        return


@register("error")
def _error(*args):
    """Throw error with message and optional identifier.

    Usage:
        error(msg)
        error(id, msg)
    """
    from matpy.runtime.types import MException

    global _last_error

    if len(args) == 1:
        msg = str(args[0])
        exc = MException(message=msg, identifier="MATLAB:error")
    elif len(args) >= 2:
        identifier = str(args[0])
        msg = str(args[1])
        exc = MException(message=msg, identifier=identifier)
    else:
        exc = MException(message="Unknown error", identifier="MATLAB:error")

    _last_error = exc
    raise exc


@register("exception")
def _exception(identifier, message):
    """Create MException object.

    Usage: exc = MException(id, msg)
    """
    from matpy.runtime.types import MException

    return MException(message=str(message), identifier=str(identifier))
