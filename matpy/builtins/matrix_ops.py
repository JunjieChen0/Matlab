"""Matrix operation built-in functions for MatPy."""

import numpy as np
from matpy.builtins import register
from matpy.runtime.types import Mat
from matpy.runtime.matrix import to_mat


@register("zeros")
def _zeros(*args):
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
    data = x.data if isinstance(x, Mat) else np.array(x)
    if data.ndim == 0:
        return 1
    return max(data.shape)

@register("numel")
def _numel(x):
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
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.diag(data, k=int(k)))

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

@register("sort")
def _sort(x, dim=1, direction="ascend"):
    data = x.data if isinstance(x, Mat) else np.array(x)
    axis = int(dim) - 1
    return Mat(np.sort(data, axis=axis))

@register("unique")
def _unique(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.unique(data))

@register("find")
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

@register("norm")
def _norm(x, p=2):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return np.linalg.norm(data, ord=int(p))

@register("inv")
def _inv(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.linalg.inv(data))

@register("det")
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

@register("pinv")
def _pinv(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.linalg.pinv(data))

@register("rank")
def _rank(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return np.linalg.matrix_rank(data)

@register("cross")
def _cross(a, b):
    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    return Mat(np.cross(da, db))

@register("dot")
def _dot(a, b):
    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    return np.dot(da, db)

@register("isequal")
def _isequal(a, b):
    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    return np.array_equal(da, db)

@register("isempty")
def _isempty(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return data.size == 0

@register("isnumeric")
def _isnumeric(x):
    if isinstance(x, Mat):
        return np.issubdtype(x.dtype, np.number)
    return isinstance(x, (int, float, complex))

@register("ischar")
def _ischar(x):
    return isinstance(x, str)

@register("islogical")
def _islogical(x):
    if isinstance(x, Mat):
        return x.dtype == bool
    return isinstance(x, bool)


@register("intersect")
def _intersect(a, b):
    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    return Mat(np.intersect1d(da, db))


@register("union")
def _union(a, b):
    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    return Mat(np.union1d(da, db))


@register("setdiff")
def _setdiff(a, b):
    da = a.data if isinstance(a, Mat) else np.array(a).flatten()
    db = b.data if isinstance(b, Mat) else np.array(b).flatten()
    result = np.setdiff1d(da, db)
    return Mat(result)


@register("setxor")
def _setxor(a, b):
    da = a.data if isinstance(a, Mat) else np.array(a).flatten()
    db = b.data if isKindOfClass(b, Mat) else np.array(b).flatten()
    return Mat(np.setxor1d(da, db))


@register("ismember")
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


@register("conv")
def _conv(a, b):
    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    return Mat(np.convolve(da, db))


@register("deconv")
def _deconv(a, b):
    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    q, r = np.polydiv(da, db)
    return Mat(q), Mat(r)


@register("fft")
def _fft(x, n=None):
    data = x.data if isinstance(x, Mat) else np.array(x)
    if n is not None:
        return Mat(np.fft.fft(data, n=int(n)))
    return Mat(np.fft.fft(data))


@register("ifft")
def _ifft(x, n=None):
    data = x.data if isinstance(x, Mat) else np.array(x)
    if n is not None:
        return Mat(np.fft.ifft(data, n=int(n)))
    return Mat(np.fft.ifft(data))


@register("fftshift")
def _fftshift(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.fft.fftshift(data))


@register("ifftshift")
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


@register("interp1")
def _interp1(x, v, xq, method="linear"):
    xd = x.data if isinstance(x, Mat) else np.array(x)
    vd = v.data if isinstance(v, Mat) else np.array(v)
    xqd = xq.data if isinstance(xq, Mat) else np.array(xq)
    return Mat(np.interp(xqd, xd, vd))


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


@register("reshape")
def _reshape_nd(x, *args):
    """Reshape array."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    if len(args) == 1 and isinstance(args[0], (list, tuple, Mat)):
        shape_data = args[0].data if isinstance(args[0], Mat) else np.array(args[0])
        shape = tuple(int(s) for s in shape_data.flat)
    else:
        shape = tuple(int(a) for a in args)
    return Mat(data.reshape(shape))


@register("ndims")
def _ndims(x):
    """Number of array dimensions."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return max(data.ndim, 2)


@register("size")
def _size_nd(x, dim=None):
    """Array size."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    if dim is not None:
        d = int(dim) - 1
        if d < data.ndim:
            return data.shape[d]
        return 1
    return Mat(np.array(data.shape))


@register("length")
def _length_nd(x):
    """Length of largest dimension."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    if data.ndim == 0:
        return 1
    return max(data.shape)


@register("numel")
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


@register("fliplr")
def _fliplr_nd(x):
    """Flip matrix left to right."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.fliplr(data))


@register("flipud")
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
def _padarray(x, pad_size, val=0, direction='both'):
    """Pad array."""
    data = x.data if isinstance(x, Mat) else np.array(x)
    if isinstance(pad_size, Mat):
        pad_size = pad_size.data
    pad = [int(p) for p in np.array(pad_size).flat]
    while len(pad) < data.ndim:
        pad.append(0)
    if direction == 'pre':
        pad_width = [(p, 0) for p in pad]
    elif direction == 'post':
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
    subs = tuple((a.data if isinstance(a, Mat) else np.array(a)).astype(int) - 1 for a in args)
    return Mat(np.ravel_multi_index(subs, shape) + 1)
