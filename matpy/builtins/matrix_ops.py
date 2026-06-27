"""Matrix operation built-in functions for MatPy."""

import numpy as np
from matpy.builtins import register
from matpy.runtime.types import Mat, CellArray
from matpy.runtime.matrix import to_mat


@register("zeros")
def _zeros(*args):
    """ZEROS Zeros array.
    ZEROS(N) is an N-by-N matrix of zeros.
    ZEROS(M,N) is an M-by-N matrix of zeros.
    ZEROS(M,N,P,...) is an M-by-N-by-P-by-... array of zeros.
    """
    if len(args) == 1:
        n = int(args[0])
        return Mat(np.zeros((n, n)))
    elif len(args) == 2:
        return Mat(np.zeros((int(args[0]), int(args[1]))))
    else:
        shape = tuple(int(a) for a in args)
        return Mat(np.zeros(shape))


@register("ones")
def _ones(*args):
    """ONES Ones array.
    ONES(N) is an N-by-N matrix of ones.
    ONES(M,N) is an M-by-N matrix of ones.
    ONES(M,N,P,...) is an M-by-N-by-P-by-... array of ones.
    """
    if len(args) == 1:
        n = int(args[0])
        return Mat(np.ones((n, n)))
    elif len(args) == 2:
        return Mat(np.ones((int(args[0]), int(args[1]))))
    else:
        shape = tuple(int(a) for a in args)
        return Mat(np.ones(shape))


@register("eye")
def _eye(*args):
    if len(args) == 1:
        n = int(args[0])
        return Mat(np.eye(n))
    elif len(args) == 2:
        return Mat(np.eye(int(args[0]), int(args[1])))
    else:
        return Mat(np.eye(1))


@register("linspace")
def _linspace(start, stop, n=100):
    return Mat(np.linspace(float(start), float(stop), int(n)))


@register("logspace")
def _logspace(start, stop, n=50):
    return Mat(np.logspace(float(start), float(stop), int(n)))


@register("size")
def _size(x, dim=None):
    data = x.data if isinstance(x, Mat) else np.array(x)
    if dim is not None:
        d = int(dim) - 1
        if d < data.ndim:
            return data.shape[d]
        return 1
    return Mat(np.array(data.shape))


@register("length")
def _length(x):
    if isinstance(x, CellArray):
        return max(x.shape)
    data = x.data if isinstance(x, Mat) else np.array(x)
    if data.ndim == 0:
        return 1
    return max(data.shape)


@register("numel")
def _numel(x):
    if isinstance(x, CellArray):
        return len([item for row in x._data for item in row])
    data = x.data if isinstance(x, Mat) else np.array(x)
    return data.size


@register("ndims")
def _ndims(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return max(data.ndim, 2)


@register("reshape")
def _reshape(x, *args):
    data = x.data if isinstance(x, Mat) else np.array(x)
    shape = tuple(int(a) for a in args)
    return Mat(data.reshape(shape))


@register("transpose")
def _transpose(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(data.T)


@register("ctranspose")
def _ctranspose(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(data.conj().T)


@register("diag")
def _diag(x, k=0):
    """Diagonal matrices and diagonals of matrix."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    k = int(k)
    if data.ndim == 1 or (data.ndim == 2 and min(data.shape) == 1):
        # Vector input: create diagonal matrix
        return Mat(np.diag(data.flatten(), k))
    else:
        # Matrix input: extract diagonal
        return Mat(np.diag(data, k))


@register("triu")
def _triu(x, k=0):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.triu(data, k=int(k)))


@register("tril")
def _tril(x, k=0):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.tril(data, k=int(k)))


@register("flipud")
def _flipud(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.flipud(data))


@register("fliplr")
def _fliplr(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.fliplr(data))


def _sort(x, dim=1, direction="ascend"):
    data = x.data if isinstance(x, Mat) else np.array(x)
    axis = int(dim) - 1
    return Mat(np.sort(data, axis=axis))


def _unique(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.unique(data))


def _find(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    indices = np.nonzero(data)[0]
    return Mat(indices + 1)


@register("repmat")
def _repmat(x, m, n=1):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.tile(data, (int(m), int(n))))


@register("cat")
def _cat(dim, *args):
    arrays = [a.data if isinstance(a, Mat) else np.array(a) for a in args]
    return Mat(np.concatenate(arrays, axis=int(dim) - 1))


@register("horzcat")
def _horzcat(*args):
    arrays = [a.data if isinstance(a, Mat) else np.array(a) for a in args]
    return Mat(np.concatenate(arrays, axis=1))


@register("vertcat")
def _vertcat(*args):
    arrays = [a.data if isinstance(a, Mat) else np.array(a) for a in args]
    return Mat(np.concatenate(arrays, axis=0))


@register("rand")
def _rand(*args):
    if len(args) == 0:
        return np.random.rand()
    elif len(args) == 1:
        n = int(args[0])
        return Mat(np.random.rand(n, n))
    elif len(args) == 2:
        return Mat(np.random.rand(int(args[0]), int(args[1])))
    else:
        shape = tuple(int(a) for a in args)
        return Mat(np.random.rand(*shape))


@register("randn")
def _randn(*args):
    if len(args) == 0:
        return np.random.randn()
    elif len(args) == 1:
        n = int(args[0])
        return Mat(np.random.randn(n, n))
    elif len(args) == 2:
        return Mat(np.random.randn(int(args[0]), int(args[1])))
    else:
        shape = tuple(int(a) for a in args)
        return Mat(np.random.randn(*shape))


@register("randi")
def _randi(imax, *args):
    if len(args) == 0:
        return np.random.randint(1, int(imax) + 1)
    elif len(args) == 1:
        n = int(args[0])
        return Mat(np.random.randint(1, int(imax) + 1, (n, n)))
    elif len(args) == 2:
        return Mat(np.random.randint(1, int(imax) + 1, (int(args[0]), int(args[1]))))
    else:
        shape = tuple(int(a) for a in args)
        return Mat(np.random.randint(1, int(imax) + 1, shape))


def _norm(x, p=2):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return np.linalg.norm(data, ord=int(p))


def _inv(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.linalg.inv(data))


def _det(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return np.linalg.det(data)


@register("eig")
def _eig(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    vals, vecs = np.linalg.eig(data)
    return Mat(vals), Mat(vecs)


@register("svd")
def _svd(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    u, s, vh = np.linalg.svd(data)
    return Mat(u), Mat(s), Mat(vh)


def _pinv(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.linalg.pinv(data))


def _schur(x):
    from scipy.linalg import schur as scipy_schur

    data = x.data if isinstance(x, Mat) else np.array(x)
    T, Z = scipy_schur(data)
    return Mat(T), Mat(Z)


def _hess(x):
    from scipy.linalg import hessenberg

    data = x.data if isinstance(x, Mat) else np.array(x)
    H, Q = hessenberg(data, calc_q=True)
    return Mat(H), Mat(Q)


def _rank(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return np.linalg.matrix_rank(data)


def _cross(a, b):
    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    return Mat(np.cross(da, db))


def _dot(a, b):
    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    return np.dot(da, db)


def _isequal(a, b):
    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    return np.array_equal(da, db)


def _isempty(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return data.size == 0


def _isnumeric(x):
    if isinstance(x, Mat):
        return np.issubdtype(x.dtype, np.number)
    return isinstance(x, (int, float, complex))


def _ischar(x):
    return isinstance(x, str)


def _islogical(x):
    if isinstance(x, Mat):
        return x.dtype == bool
    return isinstance(x, bool)


def _intersect(a, b):
    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    return Mat(np.intersect1d(da, db))


def _union(a, b):
    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    return Mat(np.union1d(da, db))


def _setdiff(a, b):
    da = a.data if isinstance(a, Mat) else np.array(a).flatten()
    db = b.data if isinstance(b, Mat) else np.array(b).flatten()
    result = np.setdiff1d(da, db)
    return Mat(result)


def _setxor(a, b):
    da = a.data if isinstance(a, Mat) else np.array(a).flatten()
    db = b.data if isinstance(b, Mat) else np.array(b).flatten()
    return Mat(np.setxor1d(da, db))


def _ismember(a, b):
    da = a.data if isinstance(a, Mat) else np.array(a).flatten()
    db = b.data if isinstance(b, Mat) else np.array(b).flatten()
    result = np.isin(da, db)
    return Mat(result)


@register("gradient")
def _gradient(x, *args):
    data = x.data if isinstance(x, Mat) else np.array(x)
    if len(args) == 0:
        return Mat(np.gradient(data))
    elif len(args) == 1:
        return Mat(np.gradient(data, float(args[0])))
    else:
        return Mat(np.gradient(data, *[float(a) for a in args]))


@register("trapz")
def _trapz(x, y=None):
    if y is None:
        data = x.data if isinstance(x, Mat) else np.array(x)
        return np.trapz(data)
    else:
        xd = x.data if isinstance(x, Mat) else np.array(x)
        yd = y.data if isinstance(y, Mat) else np.array(y)
        return np.trapz(yd, xd)


def _conv(a, b):
    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    return Mat(np.convolve(da, db))


def _deconv(a, b):
    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    q, r = np.polydiv(da, db)
    return Mat(q), Mat(r)


def _fft(x, n=None):
    data = x.data if isinstance(x, Mat) else np.array(x)
    if n is not None:
        return Mat(np.fft.fft(data, n=int(n)))
    return Mat(np.fft.fft(data))


def _ifft(x, n=None):
    data = x.data if isinstance(x, Mat) else np.array(x)
    if n is not None:
        return Mat(np.fft.ifft(data, n=int(n)))
    return Mat(np.fft.ifft(data))


def _fftshift(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.fft.fftshift(data))


def _ifftshift(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.fft.ifftshift(data))


@register("real")
def _real(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.real(data))


@register("imag")
def _imag(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.imag(data))


@register("unwrap")
def _unwrap(x, *args):
    data = x.data if isinstance(x, Mat) else np.array(x)
    if len(args) > 0:
        return Mat(np.unwrap(data, discont=float(args[0])))
    return Mat(np.unwrap(data))


@register("roots")
def _roots(p):
    data = p.data if isinstance(p, Mat) else np.array(p)
    return Mat(np.roots(data))


@register("poly")
def _poly(r):
    data = r.data if isinstance(r, Mat) else np.array(r)
    return Mat(np.poly(data))


@register("polyval")
def _polyval(p, x):
    pd = p.data if isinstance(p, Mat) else np.array(p)
    xd = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.polyval(pd, xd))


@register("accumarray")
def _accumarray(subs, val, func=None):
    subs_data = subs.data if isinstance(subs, Mat) else np.array(subs)
    val_data = val.data if isinstance(val, Mat) else np.array(val)
    if func is None:
        func = np.sum
    max_idx = int(subs_data.max())
    result = np.zeros(max_idx)
    for i, s in enumerate(subs_data.flat):
        result[int(s) - 1] += val_data.flat[i]
    return Mat(result)


# ── N-D Array Operations ──────────────────────────────────────


@register("permute")
def _permute(x, order):
    """Permute array dimensions."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    order_data = order.data if isinstance(order, Mat) else np.array(order)
    # MATLAB is 1-based, numpy is 0-based
    axes = [int(o) - 1 for o in order_data.flat]
    return Mat(np.transpose(data, axes))


@register("ipermute")
def _ipermute(x, order):
    """Inverse permute array dimensions."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    order_data = order.data if isinstance(order, Mat) else np.array(order)
    # Compute inverse permutation
    axes = [int(o) - 1 for o in order_data.flat]
    inv_axes = [0] * len(axes)
    for i, a in enumerate(axes):
        inv_axes[a] = i
    return Mat(np.transpose(data, inv_axes))


@register("shiftdim")
def _shiftdim(x, n=None):
    """Shift array dimensions."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    if n is not None:
        n = int(n)
        return Mat(np.moveaxis(data, 0, n))
    # Remove leading singleton dimensions
    while data.ndim > 0 and data.shape[0] == 1:
        data = data.squeeze(axis=0)
    return Mat(data)


@register("squeeze")
def _squeeze(x):
    """Remove singleton dimensions."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.squeeze(data))


def _reshape_nd(x, *args):
    """Reshape array."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    if len(args) == 1 and isinstance(args[0], (list, tuple, Mat)):
        shape_data = args[0].data if isinstance(args[0], Mat) else np.array(args[0])
        shape = tuple(int(s) for s in shape_data.flat)
    else:
        shape = tuple(int(a) for a in args)
    return Mat(data.reshape(shape))


def _ndims(x):
    """Number of array dimensions."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return max(data.ndim, 2)


def _size_nd(x, dim=None):
    """Array size."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    if dim is not None:
        d = int(dim) - 1
        if d < data.ndim:
            return data.shape[d]
        return 1
    return Mat(np.array(data.shape))


def _length_nd(x):
    """Length of largest dimension."""
    if isinstance(x, CellArray):
        return len(x._data)
    data = x.data if isinstance(x, Mat) else np.array(x)
    if data.ndim == 0:
        return 1
    return max(data.shape)


def _numel_nd(x):
    """Number of elements."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return data.size


@register("circshift")
def _circshift(x, k):
    """Circular shift."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    if isinstance(k, Mat):
        k = k.data
    if isinstance(k, np.ndarray):
        shifts = [int(s) for s in k.flat]
    else:
        shifts = [int(k)]
    # Pad shifts for missing dimensions
    while len(shifts) < data.ndim:
        shifts.append(0)
    return Mat(np.roll(data, shifts, axis=range(len(shifts))))


@register("flip")
def _flip(x, dim=1):
    """Flip array along dimension."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.flip(data, axis=int(dim) - 1))


def _fliplr_nd(x):
    """Flip matrix left to right."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.fliplr(data))


def _flipud_nd(x):
    """Flip matrix up to down."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.flipud(data))


@register("rot90")
def _rot90(x, k=1):
    """Rotate matrix 90 degrees."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.rot90(data, k=int(k)))


@register("padarray")
def _padarray(x, pad_size, val=0, direction="both"):
    """Pad array."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    if isinstance(pad_size, Mat):
        pad_size = pad_size.data
    pad = [int(p) for p in np.array(pad_size).flat]
    while len(pad) < data.ndim:
        pad.append(0)
    if direction == "pre":
        pad_width = [(p, 0) for p in pad]
    elif direction == "post":
        pad_width = [(0, p) for p in pad]
    else:
        pad_width = [(p, p) for p in pad]
    return Mat(np.pad(data, pad_width, constant_values=val))


@register("ind2sub")
def _ind2sub(size, idx):
    """Linear index to subscript."""
    size_data = size.data if isinstance(size, Mat) else np.array(size)
    shape = tuple(int(s) for s in size_data.flat)
    idx_data = idx.data if isinstance(idx, Mat) else np.array(idx)
    subs = np.unravel_index(idx_data.astype(int) - 1, shape)
    return tuple(Mat(np.array(s) + 1) for s in subs)


@register("sub2ind")
def _sub2ind(size, *args):
    """Subscript to linear index."""
    size_data = size.data if isinstance(size, Mat) else np.array(size)
    shape = tuple(int(s) for s in size_data.flat)
    subs = tuple(
        (a.data if isinstance(a, Mat) else np.array(a)).astype(int) - 1 for a in args
    )
    return Mat(np.ravel_multi_index(subs, shape) + 1)


@register("gallery")
def _gallery(name, *args):
    """Test matrices."""
    name = str(name).lower()
    if name == "hadamard":
        from scipy.linalg import hadamard

        return Mat(hadamard(int(args[0])))
    elif name == "hilb":
        n = int(args[0])
        return Mat(np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)]))
    elif name == "invhilb":
        n = int(args[0])
        hilb = np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)])
        return Mat(np.linalg.inv(hilb))
    elif name == "magic":
        n = int(args[0])
        if n % 2 == 1:
            p = np.arange(1, n + 1)
            M = (p[:, None] + p[None, :] - (n + 3) // 2) % n
            M = M * n + ((p[:, None] + 2 * p[None, :] - 2) % n) + 1
            return Mat(M)
        return Mat(np.eye(n))
    elif name == "pascal":
        n = int(args[0])
        M = np.zeros((n, n), dtype=int)
        for i in range(n):
            M[i, 0] = 1
            M[0, i] = 1
        for i in range(1, n):
            for j in range(1, n):
                M[i, j] = M[i - 1, j] + M[i, j - 1]
        return Mat(M)
    elif name == "rosser":
        return Mat(
            np.array(
                [
                    [611, 196, -192, 407, -8, -52, -49, 29],
                    [196, 899, 113, -192, -71, -43, -8, -44],
                    [-192, 113, 899, 196, 61, 49, 8, 52],
                    [407, -192, 196, 611, 8, 44, 59, -23],
                    [-8, -71, 61, 8, 411, -599, 208, 208],
                    [-52, -43, 49, 44, -599, 411, 208, 208],
                    [-49, -8, 8, 59, 208, 208, 99, -911],
                    [29, -44, 52, -23, 208, 208, -911, 99],
                ]
            )
        )
    elif name == "wilkinson":
        n = int(args[0])
        M = (
            np.diag(np.arange(n))
            + np.diag(np.ones(n - 1), 1)
            + np.diag(np.ones(n - 1), -1)
        )
        return Mat(M)
    elif name == "cauchy":
        n = int(args[0])
        x = np.arange(1, n + 1)
        return Mat(1.0 / (x[:, None] + x[None, :]))
    elif name == "fiedler":
        n = int(args[0])
        x = np.arange(1, n + 1)
        return Mat(np.abs(x[:, None] - x[None, :]))
    elif name == "minij":
        n = int(args[0])
        x = np.arange(1, n + 1)
        return Mat(np.minimum(x[:, None], x[None, :]))
    elif name == "moler":
        n = int(args[0])
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    M[i, j] = i + 1
                else:
                    M[i, j] = min(i, j) - 1
        return Mat(M)
    elif name == "pei":
        n = int(args[0])
        alpha = args[1] if len(args) > 1 else 1
        return Mat(alpha * np.eye(n) + np.ones((n, n)))
    return Mat(np.eye(int(args[0]) if args else 3))


@register("compan")
def _compan(p):
    """Companion matrix."""
    pd = p.data if isinstance(p, Mat) else np.array(p).flatten()
    n = len(pd) - 1
    if n <= 0:
        return Mat(np.array([]))
    M = np.zeros((n, n))
    M[0, :] = -pd[1:] / pd[0]
    for i in range(n - 1):
        M[i + 1, i] = 1
    return Mat(M)


def _cat(dim, *args):
    """Concatenate arrays along dimension."""
    arrays = []
    for a in args:
        if isinstance(a, Mat):
            arrays.append(a.data)
        else:
            arrays.append(np.array(a))
    axis = int(dim) - 1
    return Mat(np.concatenate(arrays, axis=axis))


def _vertcat(*args):
    """Vertically concatenate arrays."""
    arrays = []
    for a in args:
        if isinstance(a, Mat):
            arrays.append(a.data)
        else:
            arrays.append(np.array(a))
    return Mat(np.vstack(arrays))


def _horzcat(*args):
    """Horizontally concatenate arrays."""
    arrays = []
    for a in args:
        if isinstance(a, Mat):
            arrays.append(a.data)
        else:
            arrays.append(np.array(a))
    return Mat(np.hstack(arrays))


def _ndgrid(*args):
    """Generate N-D grids."""
    if len(args) == 1:
        # ndgrid(x) = meshgrid(x)
        x = args[0].data if isinstance(args[0], Mat) else np.array(args[0])
        return Mat(x)
    elif len(args) == 2:
        x = args[0].data if isinstance(args[0], Mat) else np.array(args[0])
        y = args[1].data if isinstance(args[1], Mat) else np.array(args[1])
        X, Y = np.meshgrid(x.flatten(), y.flatten(), indexing="ij")
        return Mat(X), Mat(Y)
    else:
        arrays = []
        for a in args:
            if isinstance(a, Mat):
                arrays.append(a.data.flatten())
            else:
                arrays.append(np.array(a).flatten())
        grids = np.meshgrid(*arrays, indexing="ij")
        return tuple(Mat(g) for g in grids)


def _kron(a, b):
    """Kronecker tensor product."""
    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    return Mat(np.kron(da, db))


def _diag(x, k=0):
    """Diagonal matrices and diagonals of matrix."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    k = int(k)
    if data.ndim == 1 or (data.ndim == 2 and min(data.shape) == 1):
        # Vector input: create diagonal matrix
        return Mat(np.diag(data.flatten(), k))
    else:
        # Matrix input: extract diagonal
        return Mat(np.diag(data, k))


def _triu(x, k=0):
    """Upper triangular matrix."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.triu(data, int(k)))


def _tril(x, k=0):
    """Lower triangular matrix."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.tril(data, int(k)))


def _repmat(x, *args):
    """Replicate and tile array."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    if len(args) == 1:
        n = int(args[0])
        reps = (n, n)
    elif len(args) == 2:
        reps = (int(args[0]), int(args[1]))
    else:
        reps = tuple(int(a) for a in args)
    return Mat(np.tile(data, reps))


@register("repelem")
def _repelem(x, *args):
    """Replicate elements of array."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    reps = [int(a) for a in args]
    return Mat(np.repeat(data, reps, axis=range(len(reps))))


def _meshgrid(*args):
    """Generate meshgrid."""
    arrays = []
    for a in args:
        if isinstance(a, Mat):
            arrays.append(a.data.flatten())
        else:
            arrays.append(np.array(a).flatten())
    grids = np.meshgrid(*arrays)
    return tuple(Mat(g) for g in grids)


def _linspace(start, stop, n=100):
    """Linearly spaced vector."""
    return Mat(np.linspace(float(start), float(stop), int(n)))


def _logspace(start, stop, n=50):
    """Logarithmically spaced vector."""
    return Mat(np.logspace(float(start), float(stop), int(n)))


def _blkdiag(*args):
    """Block diagonal matrix."""
    matrices = []
    for a in args:
        if isinstance(a, Mat):
            matrices.append(a.data)
        else:
            matrices.append(np.array(a))
    from scipy.linalg import block_diag

    return Mat(block_diag(*matrices))


def _hankel(c, r=None):
    """Hankel matrix."""
    cd = c.data if isinstance(c, Mat) else np.array(c)
    if r is None:
        rd = np.zeros_like(cd)
    else:
        rd = r.data if isinstance(r, Mat) else np.array(r)
    n = len(cd)
    m = len(rd)
    H = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if i + j < n:
                H[i, j] = cd[i + j]
            else:
                H[i, j] = rd[i + j - n + 1]
    return Mat(H)


def _toeplitz(c, r=None):
    """Toeplitz matrix."""
    cd = c.data if isinstance(c, Mat) else np.array(c)
    if r is None:
        rd = cd
    else:
        rd = r.data if isinstance(r, Mat) else np.array(r)
    n = len(cd)
    m = len(rd)
    T = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if i >= j:
                T[i, j] = cd[i - j]
            else:
                T[i, j] = rd[j - i]
    return Mat(T)


# ── Additional Linear Algebra Functions ───────────────────────


@register("gsvd")
def _gsvd(A, B):
    """Generalized singular value decomposition."""
    from scipy.linalg import gsvd

    A_data = A.data if isinstance(A, Mat) else np.array(A)
    B_data = B.data if isinstance(B, Mat) else np.array(B)
    U, V, C, S, Q, X = gsvd(A_data, B_data)
    return Mat(U), Mat(V), Mat(C), Mat(S), Mat(Q)


@register("balance")
def _balance(A):
    """Diagonal scaling to improve eigenvalue accuracy."""
    from scipy.linalg import balance

    A_data = A.data if isinstance(A, Mat) else np.array(A)
    T, B = balance(A_data)
    return Mat(T), Mat(B)


@register("schur")
def _schur(A):
    """Schur decomposition."""
    from scipy.linalg import schur

    A_data = A.data if isinstance(A, Mat) else np.array(A)
    T, Z = schur(A_data)
    return Mat(T), Mat(Z)


@register("hess")
def _hess(A):
    """Hessenberg decomposition."""
    from scipy.linalg import hessenberg

    A_data = A.data if isinstance(A, Mat) else np.array(A)
    H, Q = hessenberg(A_data, calc_q=True)
    return Mat(H), Mat(Q)


@register("expm")
def _expm(A):
    """Matrix exponential."""
    from scipy.linalg import expm as scipy_expm

    A_data = A.data if isinstance(A, Mat) else np.array(A)
    return Mat(scipy_expm(A_data))


@register("logm")
def _logm(A):
    """Matrix logarithm."""
    from scipy.linalg import logm as scipy_logm

    A_data = A.data if isinstance(A, Mat) else np.array(A)
    return Mat(scipy_logm(A_data))


@register("sqrtm")
def _sqrtm(A):
    """Matrix square root."""
    from scipy.linalg import sqrtm as scipy_sqrtm

    A_data = A.data if isinstance(A, Mat) else np.array(A)
    return Mat(scipy_sqrtm(A_data))


@register("linsolve")
def _linsolve(A, B):
    """Solve linear system Ax = B."""
    A_data = A.data if isinstance(A, Mat) else np.array(A)
    B_data = B.data if isinstance(B, Mat) else np.array(B)
    x = np.linalg.solve(A_data, B_data)
    return Mat(x)


@register("kron")
def _kron(A, B):
    """Kronecker tensor product."""
    A_data = A.data if isinstance(A, Mat) else np.array(A)
    B_data = B.data if isinstance(B, Mat) else np.array(B)
    return Mat(np.kron(A_data, B_data))


@register("null")
def _null(A):
    """Null space of matrix."""
    A_data = A.data if isinstance(A, Mat) else np.array(A)
    _, s, Vh = np.linalg.svd(A_data)
    tol = max(A_data.shape) * np.max(s) * np.finfo(float).eps
    mask = s > tol
    null_space = Vh[mask == False].T.conj()
    return Mat(null_space)


@register("orth")
def _orth(A):
    """Range space of matrix (orthogonal basis)."""
    A_data = A.data if isinstance(A, Mat) else np.array(A)
    U, s, _ = np.linalg.svd(A_data)
    tol = max(A_data.shape) * np.max(s) * np.finfo(float).eps
    mask = s > tol
    return Mat(U[:, mask])


@register("subspace")
def _subspace(A, B):
    """Angle between two subspaces."""
    A_data = A.data if isinstance(A, Mat) else np.array(A)
    B_data = B.data if isinstance(B, Mat) else np.array(B)
    _, s, _ = np.linalg.svd(A_data.T.conj() @ B_data)
    s = np.clip(s, 0, 1)
    angles = np.arccos(s)
    return float(np.min(angles))


@register("cond")
def _cond(A, p=None):
    """Condition number."""
    A_data = A.data if isinstance(A, Mat) else np.array(A)
    return float(np.linalg.cond(A_data, p))


@register("condest")
def _condest(A):
    """1-norm condition number estimate."""
    A_data = A.data if isinstance(A, Mat) else np.array(A)
    return float(np.linalg.cond(A_data, 1))


@register("funm")
def _funm(A, func):
    """Evaluate general matrix function."""
    from scipy.linalg import funm as scipy_funm

    A_data = A.data if isinstance(A, Mat) else np.array(A)
    return Mat(scipy_funm(A_data, func))
