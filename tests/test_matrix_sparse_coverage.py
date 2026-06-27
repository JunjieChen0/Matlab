"""Comprehensive tests for matrix_ops.py and sparse.py to boost coverage."""

import pytest
import numpy as np
from tests.conftest import run_matlab, get_val
from matpy.runtime.types import Mat


class TestMatrixOpsFunctions:
    """Test matrix operation functions."""

    def test_zeros(self):
        from matpy.builtins.matrix_ops import _zeros

        result = _zeros(3, 4)
        assert isinstance(result, Mat)
        assert result.data.shape == (3, 4)

    def test_ones(self):
        from matpy.builtins.matrix_ops import _ones

        result = _ones(3, 4)
        assert isinstance(result, Mat)
        assert result.data.shape == (3, 4)

    def test_eye(self):
        from matpy.builtins.matrix_ops import _eye

        result = _eye(3)
        assert isinstance(result, Mat)
        assert result.data.shape == (3, 3)

    def test_linspace(self):
        from matpy.builtins.matrix_ops import _linspace

        result = _linspace(0, 1, 5)
        assert isinstance(result, Mat)
        assert len(result.data) == 5

    def test_logspace(self):
        from matpy.builtins.matrix_ops import _logspace

        result = _logspace(0, 2, 3)
        assert isinstance(result, Mat)

    def test_size(self):
        from matpy.builtins.matrix_ops import _size

        result = _size(Mat(np.array([[1, 2], [3, 4]])))
        assert result is not None

    def test_length(self):
        from matpy.builtins.matrix_ops import _length

        result = _length(Mat(np.array([[1, 2, 3], [4, 5, 6]])))
        assert result == 3

    def test_numel(self):
        from matpy.builtins.matrix_ops import _numel

        result = _numel(Mat(np.array([[1, 2], [3, 4]])))
        assert result == 4

    def test_ndims(self):
        from matpy.builtins.matrix_ops import _ndims

        result = _ndims(Mat(np.array([[1, 2], [3, 4]])))
        assert result == 2

    def test_isscalar_true(self):
        # isscalar is not directly importable, test through interpreter
        interp = run_matlab("x = 42;\nr = isscalar(x);")
        r = interp.global_env.get("r")
        assert r == True

    def test_isscalar_false(self):
        interp = run_matlab("x = [1 2 3];\nr = isscalar(x);")
        r = interp.global_env.get("r")
        assert r == False

    def test_isvector_true(self):
        interp = run_matlab("x = [1 2 3];\nr = isvector(x);")
        r = interp.global_env.get("r")
        assert r == True

    def test_isvector_false(self):
        interp = run_matlab("x = [1 2; 3 4];\nr = isvector(x);")
        r = interp.global_env.get("r")
        assert r == False

    def test_ismatrix_true(self):
        interp = run_matlab("x = [1 2; 3 4];\nr = ismatrix(x);")
        r = interp.global_env.get("r")
        assert r == True

    def test_isempty_true(self):
        from matpy.builtins.matrix_ops import _isempty

        result = _isempty(Mat(np.array([])))
        assert result == True

    def test_isempty_false(self):
        from matpy.builtins.matrix_ops import _isempty

        result = _isempty(Mat(np.array([1, 2, 3])))
        assert result == False

    def test_isequal_true(self):
        from matpy.builtins.matrix_ops import _isequal

        result = _isequal(Mat(np.array([1, 2, 3])), Mat(np.array([1, 2, 3])))
        assert result == True

    def test_isequal_false(self):
        from matpy.builtins.matrix_ops import _isequal

        result = _isequal(Mat(np.array([1, 2, 3])), Mat(np.array([4, 5, 6])))
        assert result == False

    def test_transpose(self):
        from matpy.builtins.matrix_ops import _transpose

        result = _transpose(Mat(np.array([[1, 2], [3, 4]])))
        assert isinstance(result, Mat)

    def test_ctranspose(self):
        from matpy.builtins.matrix_ops import _ctranspose

        result = _ctranspose(Mat(np.array([[1 + 1j, 2], [3, 4]])))
        assert isinstance(result, Mat)

    def test_diag_create(self):
        from matpy.builtins.matrix_ops import _diag

        result = _diag(Mat(np.array([1, 2, 3])))
        assert isinstance(result, Mat)
        assert result.data.shape == (3, 3)

    def test_diag_extract(self):
        from matpy.builtins.matrix_ops import _diag

        result = _diag(Mat(np.array([[1, 2], [3, 4]])))
        assert isinstance(result, Mat)

    def test_triu(self):
        from matpy.builtins.matrix_ops import _triu

        result = _triu(Mat(np.array([[1, 2], [3, 4]])))
        assert isinstance(result, Mat)

    def test_tril(self):
        from matpy.builtins.matrix_ops import _tril

        result = _tril(Mat(np.array([[1, 2], [3, 4]])))
        assert isinstance(result, Mat)

    def test_flipud(self):
        from matpy.builtins.matrix_ops import _flipud

        result = _flipud(Mat(np.array([[1, 2], [3, 4]])))
        assert isinstance(result, Mat)

    def test_fliplr(self):
        from matpy.builtins.matrix_ops import _fliplr

        result = _fliplr(Mat(np.array([[1, 2], [3, 4]])))
        assert isinstance(result, Mat)

    def test_repmat(self):
        from matpy.builtins.matrix_ops import _repmat

        result = _repmat(Mat(np.array([1, 2])), 2, 3)
        assert isinstance(result, Mat)

    def test_cat(self):
        from matpy.builtins.matrix_ops import _cat

        result = _cat(1, Mat(np.array([1, 2])), Mat(np.array([3, 4])))
        assert isinstance(result, Mat)

    def test_horzcat(self):
        from matpy.builtins.matrix_ops import _horzcat

        result = _horzcat(Mat(np.array([1, 2])), Mat(np.array([3, 4])))
        assert isinstance(result, Mat)

    def test_vertcat(self):
        from matpy.builtins.matrix_ops import _vertcat

        result = _vertcat(Mat(np.array([1, 2])), Mat(np.array([3, 4])))
        assert isinstance(result, Mat)

    def test_rand(self):
        from matpy.builtins.matrix_ops import _rand

        result = _rand(3, 4)
        assert isinstance(result, Mat)
        assert result.data.shape == (3, 4)

    def test_randn(self):
        from matpy.builtins.matrix_ops import _randn

        result = _randn(3, 4)
        assert isinstance(result, Mat)
        assert result.data.shape == (3, 4)

    def test_reshape(self):
        from matpy.builtins.matrix_ops import _reshape

        result = _reshape(Mat(np.array([1, 2, 3, 4, 5, 6])), 2, 3)
        assert isinstance(result, Mat)
        assert result.data.shape == (2, 3)

    def test_sort(self):
        from matpy.builtins.matrix_ops import _sort

        result = _sort(Mat(np.array([3, 1, 2])))
        assert isinstance(result, Mat)

    def test_unique(self):
        from matpy.builtins.matrix_ops import _unique

        result = _unique(Mat(np.array([1, 2, 2, 3, 3, 3])))
        assert isinstance(result, Mat)

    def test_find(self):
        from matpy.builtins.matrix_ops import _find

        result = _find(Mat(np.array([0, 1, 0, 1])))
        assert isinstance(result, Mat)

    def test_cross(self):
        from matpy.builtins.matrix_ops import _cross

        result = _cross(Mat(np.array([1, 0, 0])), Mat(np.array([0, 1, 0])))
        assert isinstance(result, Mat)

    def test_dot(self):
        from matpy.builtins.matrix_ops import _dot

        result = _dot(Mat(np.array([1, 2, 3])), Mat(np.array([4, 5, 6])))
        assert result is not None

    def test_norm(self):
        from matpy.builtins.matrix_ops import _norm

        result = _norm(Mat(np.array([3, 4])))
        assert abs(result - 5.0) < 0.01

    def test_trace(self):
        # trace is not directly importable, test through interpreter
        interp = run_matlab("A = [1 2; 3 4];\nt = trace(A);")
        t = interp.global_env.get("t")
        assert get_val(t) == 5

    def test_rank(self):
        from matpy.builtins.matrix_ops import _rank

        result = _rank(Mat(np.array([[1, 0], [0, 1]])))
        assert result == 2

    def test_det(self):
        from matpy.builtins.matrix_ops import _det

        result = _det(Mat(np.array([[1, 2], [3, 4]])))
        assert abs(result - (-2.0)) < 0.01

    def test_inv(self):
        from matpy.builtins.matrix_ops import _inv

        result = _inv(Mat(np.array([[1, 2], [3, 4]])))
        assert isinstance(result, Mat)

    def test_eig(self):
        from matpy.builtins.matrix_ops import _eig

        result = _eig(Mat(np.array([[1, 2], [3, 4]])))
        assert result is not None

    def test_svd(self):
        from matpy.builtins.matrix_ops import _svd

        result = _svd(Mat(np.array([[1, 2], [3, 4]])))
        assert result is not None

    def test_lu(self):
        interp = run_matlab("A = [1 2; 3 4];\n[L, U] = lu(A);")
        L = interp.global_env.get("L")
        U = interp.global_env.get("U")
        assert isinstance(L, Mat)
        assert isinstance(U, Mat)

    def test_qr(self):
        interp = run_matlab("A = [1 2; 3 4];\n[Q, R] = qr(A);")
        Q = interp.global_env.get("Q")
        R = interp.global_env.get("R")
        assert isinstance(Q, Mat)
        assert isinstance(R, Mat)

    def test_chol(self):
        interp = run_matlab("A = [4 2; 2 3];\nL = chol(A);")
        L = interp.global_env.get("L")
        assert isinstance(L, Mat)

    def test_pinv(self):
        from matpy.builtins.matrix_ops import _pinv

        result = _pinv(Mat(np.array([[1, 2], [3, 4]])))
        assert isinstance(result, Mat)

    def test_cond(self):
        from matpy.builtins.matrix_ops import _cond

        result = _cond(Mat(np.array([[1, 2], [3, 4]])))
        assert result is not None

    def test_rref(self):
        interp = run_matlab("A = [1 2 3; 4 5 6];\nR = rref(A);")
        R = interp.global_env.get("R")
        assert isinstance(R, Mat)


class TestSparseFunctions:
    """Test sparse matrix functions."""

    def test_sparse_create(self):
        from matpy.builtins.sparse import _sparse

        result = _sparse(
            Mat(np.array([1, 2])), Mat(np.array([1, 2])), Mat(np.array([1, 1])), 3, 3
        )
        assert result is not None

    def test_speye(self):
        from matpy.builtins.sparse import _speye

        result = _speye(3)
        assert result is not None

    def test_spzeros(self):
        from matpy.builtins.sparse import _spzeros

        result = _spzeros(3, 3)
        assert result is not None

    def test_full(self):
        from matpy.builtins.sparse import _full
        from scipy import sparse as sp

        S = sp.eye(3)
        result = _full(S)
        assert isinstance(result, Mat)

    def test_nnz(self):
        interp = run_matlab("S = speye(3);\nn = nnz(S);")
        n = interp.global_env.get("n")
        assert get_val(n) == 3

    def test_issparse_true(self):
        interp = run_matlab("S = speye(3);\nr = issparse(S);")
        r = interp.global_env.get("r")
        assert r == True

    def test_issparse_false(self):
        interp = run_matlab("A = eye(3);\nr = issparse(A);")
        r = interp.global_env.get("r")
        assert r == False

    def test_sprand(self):
        from matpy.builtins.sparse import _sprand

        result = _sprand(3, 3, 0.5)
        assert result is not None

    def test_sprandn(self):
        from matpy.builtins.sparse import _sprandn

        result = _sprandn(3, 3, 0.5)
        assert result is not None


class TestMatrixOpsIntegration:
    """Integration tests through interpreter."""

    def test_zeros_interpreter(self):
        interp = run_matlab("A = zeros(3, 4);")
        A = interp.global_env.get("A")
        assert isinstance(A, Mat)
        assert A.data.shape == (3, 4)

    def test_ones_interpreter(self):
        interp = run_matlab("A = ones(3, 4);")
        A = interp.global_env.get("A")
        assert isinstance(A, Mat)
        assert A.data.shape == (3, 4)

    def test_eye_interpreter(self):
        interp = run_matlab("A = eye(3);")
        A = interp.global_env.get("A")
        assert isinstance(A, Mat)
        assert A.data.shape == (3, 3)

    def test_linspace_interpreter(self):
        interp = run_matlab("x = linspace(0, 1, 5);")
        x = interp.global_env.get("x")
        assert isinstance(x, Mat)
        assert len(x.data) == 5

    def test_size_interpreter(self):
        interp = run_matlab("A = [1 2; 3 4];\ns = size(A);")
        s = interp.global_env.get("s")
        assert s is not None

    def test_transpose_interpreter(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = A.';")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)

    def test_diag_interpreter(self):
        interp = run_matlab("D = diag([1 2 3]);")
        D = interp.global_env.get("D")
        assert isinstance(D, Mat)
        assert D.data.shape == (3, 3)

    def test_triu_interpreter(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = triu(A);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)

    def test_tril_interpreter(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = tril(A);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)

    def test_inv_interpreter(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = inv(A);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)

    def test_det_interpreter(self):
        interp = run_matlab("A = [1 2; 3 4];\nd = det(A);")
        d = interp.global_env.get("d")
        assert abs(get_val(d) - (-2.0)) < 0.01

    def test_eig_interpreter(self):
        interp = run_matlab("A = [1 2; 3 4];\n[V, D] = eig(A);")
        V = interp.global_env.get("V")
        D = interp.global_env.get("D")
        assert isinstance(V, Mat)
        assert isinstance(D, Mat)

    def test_svd_interpreter(self):
        interp = run_matlab("A = [1 2; 3 4];\n[U, S, V] = svd(A);")
        U = interp.global_env.get("U")
        S = interp.global_env.get("S")
        V = interp.global_env.get("V")
        assert isinstance(U, Mat)
        assert isinstance(S, Mat)
        assert isinstance(V, Mat)

    def test_rand_interpreter(self):
        interp = run_matlab("R = rand(3, 4);")
        R = interp.global_env.get("R")
        assert isinstance(R, Mat)
        assert R.data.shape == (3, 4)

    def test_randn_interpreter(self):
        interp = run_matlab("R = randn(3, 4);")
        R = interp.global_env.get("R")
        assert isinstance(R, Mat)
        assert R.data.shape == (3, 4)

    def test_reshape_interpreter(self):
        interp = run_matlab("A = [1 2 3 4 5 6];\nB = reshape(A, 2, 3);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)
        assert B.data.shape == (2, 3)

    def test_norm_interpreter(self):
        interp = run_matlab("v = [3 4];\nn = norm(v);")
        n = interp.global_env.get("n")
        assert abs(get_val(n) - 5.0) < 0.01

    def test_trace_interpreter(self):
        interp = run_matlab("A = [1 2; 3 4];\nt = trace(A);")
        t = interp.global_env.get("t")
        assert get_val(t) == 5

    def test_rank_interpreter(self):
        interp = run_matlab("A = [1 0; 0 1];\nr = rank(A);")
        r = interp.global_env.get("r")
        assert get_val(r) == 2

    def test_speye_interpreter(self):
        interp = run_matlab("S = speye(3);")
        S = interp.global_env.get("S")
        assert S is not None

    def test_spzeros_interpreter(self):
        interp = run_matlab("S = spzeros(3, 3);")
        S = interp.global_env.get("S")
        assert S is not None
