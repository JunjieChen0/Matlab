"""Sparse matrix built-in functions for MatPy."""

import numpy as np
from scipy import sparse as sp
from matpy.builtins import register
from matpy.runtime.types import Mat


@register("sparse")
def _sparse(*args):
    def to_numpy(x):
        if isinstance(x, Mat):
            return x.data
        return np.array(x)
    if len(args) == 2:
        return Mat(sp.csr_matrix(to_numpy(args[0])))
    elif len(args) == 3:
        i = to_numpy(args[0]).flatten().astype(int) - 1
        j = to_numpy(args[1]).flatten().astype(int) - 1
        v = to_numpy(args[2]).flatten()
        return Mat(sp.csr_matrix((v, (i, j))))
    elif len(args) == 5:
        i = to_numpy(args[0]).flatten().astype(int) - 1
        j = to_numpy(args[1]).flatten().astype(int) - 1
        v = to_numpy(args[2]).flatten()
        return Mat(sp.csr_matrix((v, (i, j)), shape=(int(args[3]), int(args[4]))))
    return Mat(sp.csr_matrix((1, 1)))

@register("full")
def _full(x):
    if isinstance(x, Mat):
        if sp.issparse(x.data):
            return Mat(x.data.toarray())
        return x
    return Mat(np.array(x))

@register("spdiags")
def _spdiags(B, d, m, n):
    B_data = B.data if isinstance(B, Mat) else np.array(B)
    d_data = np.array(d).flatten()
    return Mat(sp.diags(B_data, d_data, shape=(int(m), int(n)), format='csr'))

@register("sprand")
def _sprand(m, n, density=0.1):
    return Mat(sp.rand(int(m), int(n), density=float(density), format='csr'))

@register("sprandn")
def _sprandn(m, n, density=0.1):
    return Mat(sp.random(int(m), int(n), density=float(density), format='csr', data_rvs=np.random.randn))

@register("nnz")
def _nnz(x):
    if isinstance(x, Mat):
        data = x.data
        if sp.issparse(data):
            return int(data.nnz)
        return int(np.count_nonzero(data))
    return int(np.count_nonzero(np.array(x)))

@register("nonzeros")
def _nonzeros(x):
    if isinstance(x, Mat):
        data = x.data
        if sp.issparse(data):
            return Mat(data.data)
        return Mat(data[data != 0])
    return Mat(np.array(x)[np.array(x) != 0])

@register("issparse")
def _issparse(x):
    if isinstance(x, Mat):
        return sp.issparse(x.data)
    return False

@register("spconvert")
def _spconvert(x):
    if isinstance(x, Mat):
        data = x.data
        if data.shape[1] >= 3:
            i = data[:, 0].astype(int) - 1
            j = data[:, 1].astype(int) - 1
            v = data[:, 2]
            if data.shape[1] == 3:
                return Mat(sp.csr_matrix((v, (i, j))))
            else:
                m = int(data[0, 3])
                n = int(data[0, 4]) if data.shape[1] > 4 else int(data[:, 1].max())
                return Mat(sp.csr_matrix((v, (i, j)), shape=(m, n)))
    return Mat(sp.csr_matrix((1, 1)))

@register("spones")
def _spones(x):
    if isinstance(x, Mat):
        data = x.data
        if sp.issparse(data):
            return Mat(sp.csr_matrix(np.ones_like(data.data), data.indices, data.indptr, shape=data.shape))
    return Mat(sp.csr_matrix(np.ones_like(np.array(x))))

@register("spalloc")
def _spalloc(m, n, nzmax=0):
    return Mat(sp.csr_matrix((int(m), int(n)), dtype=float))

@register("speye")
def _speye(m, n=None):
    return Mat(sp.eye(int(m), int(n) if n else int(m), format='csr'))

@register("spzeros")
def _spzeros(m, n=None):
    return Mat(sp.csr_matrix((int(m), int(n) if n else int(m))))

@register("eigs")
def _eigs(A, k=6, which='LM'):
    from scipy.sparse.linalg import eigs as scipy_eigs
    data = A.data if isinstance(A, Mat) else np.array(A)
    if sp.issparse(data):
        vals, vecs = scipy_eigs(data, k=int(k), which=which)
    else:
        vals, vecs = scipy_eigs(data, k=int(k), which=which)
    return Mat(vals), Mat(vecs)

@register("svds")
def _svds(A, k=6):
    from scipy.sparse.linalg import svds as scipy_svds
    data = A.data if isinstance(A, Mat) else np.array(A)
    if sp.issparse(data):
        u, s, vt = scipy_svds(data, k=int(k))
    else:
        u, s, vt = scipy_svds(data, k=int(k))
    return Mat(u), Mat(s), Mat(vt)
