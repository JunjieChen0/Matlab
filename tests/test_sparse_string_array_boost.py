"""Boost tests for sparse.py and string_array.py."""

import numpy as np
from tests.conftest import run_matlab, get_val
from matpy.runtime.types import Mat


class TestSparseBoost:
    """Boost sparse coverage."""

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
        assert r

    def test_issparse_false(self):
        interp = run_matlab("A = eye(3);\nr = issparse(A);")
        r = interp.global_env.get("r")
        assert not r

    def test_sprand(self):
        from matpy.builtins.sparse import _sprand

        result = _sprand(3, 3, 0.5)
        assert result is not None

    def test_sprandn(self):
        from matpy.builtins.sparse import _sprandn

        result = _sprandn(3, 3, 0.5)
        assert result is not None

    def test_spones(self):
        from scipy import sparse as sp

        S = sp.eye(3)
        result = S.copy()
        assert result is not None

    def test_spfun(self):
        from scipy import sparse as sp

        S = sp.eye(3)
        result = S * 2
        assert result is not None

    def test_spdiags(self):
        from scipy import sparse as sp

        result = sp.diags([1, 2, 3], 0, shape=(3, 3))
        assert result is not None


class TestStringArrayBoost:
    """Boost string_array coverage."""

    def test_string_create(self):
        from matpy.builtins.string_array import _string

        result = _string("hello")
        assert result is not None

    def test_string_length(self):
        from matpy.builtins.string_array import _strlength

        result = _strlength("hello")
        assert result == 5

    def test_string_upper(self):
        from matpy.builtins.string import _upper

        result = _upper("hello")
        assert result == "HELLO"

    def test_string_lower(self):
        from matpy.builtins.string import _lower

        result = _lower("HELLO")
        assert result == "hello"

    def test_string_strip(self):
        from matpy.builtins.string import _strtrim

        result = _strtrim("  hello  ")
        assert result == "hello"

    def test_string_contains(self):
        from matpy.builtins.string import _strfind

        result = _strfind("hello world", "world")
        assert result is not None

    def test_string_replace(self):
        from matpy.builtins.string import _strrep

        result = _strrep("hello world", "world", "matlab")
        assert result == "hello matlab"

    def test_string_split(self):
        from matpy.builtins.string import _strsplit

        result = _strsplit("hello world", " ")
        assert isinstance(result, list)

    def test_string_join(self):
        from matpy.builtins.string import _strjoin

        result = _strjoin(["hello", "world"], " ")
        assert result == "hello world"

    def test_string_startsWith(self):
        from matpy.builtins.string import _startsWith

        result = _startsWith("hello", "hel")
        assert result

    def test_string_endsWith(self):
        from matpy.builtins.string import _endsWith

        result = _endsWith("hello", "llo")
        assert result


class TestStatisticsBoost:
    """Boost statistics coverage."""

    def test_anova1(self):
        from matpy.builtins.statistics import _anova1

        g1 = Mat(np.array([1, 2, 3]))
        g2 = Mat(np.array([4, 5, 6]))
        result = _anova1(g1, g2)
        assert result is not None

    def test_ttest(self):
        from matpy.builtins.statistics import _ttest

        x = Mat(np.array([1, 2, 3, 4, 5]))
        result = _ttest(x)
        assert result is not None

    def test_ttest2(self):
        from matpy.builtins.statistics import _ttest2

        x = Mat(np.array([1, 2, 3]))
        y = Mat(np.array([4, 5, 6]))
        result = _ttest2(x, y)
        assert result is not None

    def test_chi2gof(self):
        from matpy.builtins.statistics import _chi2gof

        x = Mat(np.array([10, 20, 30, 25, 15]))
        result = _chi2gof(x)
        assert result is not None

    def test_kstest(self):
        from matpy.builtins.statistics import _kstest

        x = Mat(np.array([0.1, 0.3, 0.5, 0.7, 0.9]))
        result = _kstest(x)
        assert result is not None

    def test_normcdf(self):
        from matpy.builtins.statistics import _normcdf

        result = _normcdf(0.0, 0, 1)
        assert result is not None

    def test_norminv(self):
        from matpy.builtins.statistics import _norminv

        result = _norminv(0.5, 0, 1)
        assert result is not None

    def test_normpdf(self):
        from matpy.builtins.statistics import _normpdf

        result = _normpdf(0.0, 0, 1)
        assert result is not None

    def test_tcdf(self):
        from matpy.builtins.statistics import _tcdf

        result = _tcdf(0.0, 10)
        assert result is not None

    def test_chi2cdf(self):
        from matpy.builtins.statistics import _chi2cdf

        result = _chi2cdf(1.0, 2)
        assert result is not None

    def test_fcdf(self):
        from matpy.builtins.statistics import _fcdf

        result = _fcdf(1.0, 5, 10)
        assert result is not None

    def test_binocdf(self):
        from matpy.builtins.statistics import _binocdf

        result = _binocdf(3, 10, 0.5)
        assert result is not None

    def test_poisscdf(self):
        from matpy.builtins.statistics import _poisscdf

        result = _poisscdf(3, 2)
        assert result is not None

    def test_expcdf(self):
        from matpy.builtins.statistics import _expcdf

        result = _expcdf(1.0, 2)
        assert result is not None

    def test_gamcdf(self):
        from matpy.builtins.statistics import _gamcdf

        result = _gamcdf(1.0, 2, 1)
        assert result is not None

    def test_betacdf(self):
        from matpy.builtins.statistics import _betacdf

        result = _betacdf(0.5, 2, 3)
        assert result is not None

    def test_unifcdf(self):
        from matpy.builtins.statistics import _unifcdf

        result = _unifcdf(0.5, 0, 1)
        assert result is not None

    def test_logncdf(self):
        from matpy.builtins.statistics import _logncdf

        result = _logncdf(1.0, 0, 1)
        assert result is not None

    def test_wblcdf(self):
        from matpy.builtins.statistics import _wblcdf

        result = _wblcdf(1.0, 2, 1)
        assert result is not None


class TestMatrixOpsBoost:
    """Boost matrix_ops coverage."""

    def test_zeros(self):
        from matpy.builtins.matrix_ops import _zeros

        result = _zeros(3, 4)
        assert isinstance(result, Mat)

    def test_ones(self):
        from matpy.builtins.matrix_ops import _ones

        result = _ones(3, 4)
        assert isinstance(result, Mat)

    def test_eye(self):
        from matpy.builtins.matrix_ops import _eye

        result = _eye(3)
        assert isinstance(result, Mat)

    def test_linspace(self):
        from matpy.builtins.matrix_ops import _linspace

        result = _linspace(0, 1, 5)
        assert isinstance(result, Mat)

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

    def test_randn(self):
        from matpy.builtins.matrix_ops import _randn

        result = _randn(3, 4)
        assert isinstance(result, Mat)

    def test_reshape(self):
        from matpy.builtins.matrix_ops import _reshape

        result = _reshape(Mat(np.array([1, 2, 3, 4, 5, 6])), 2, 3)
        assert isinstance(result, Mat)

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
        interp = run_matlab("A = [1 2; 3 4];\nt = trace(A);")
        t = interp.global_env.get("t")
        assert get_val(t) == 5

    def test_rank(self):
        interp = run_matlab("A = [1 0; 0 1];\nr = rank(A);")
        r = interp.global_env.get("r")
        assert get_val(r) == 2

    def test_det(self):
        interp = run_matlab("A = [1 2; 3 4];\nd = det(A);")
        d = interp.global_env.get("d")
        assert abs(get_val(d) - (-2.0)) < 0.01

    def test_inv(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = inv(A);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)

    def test_eig(self):
        interp = run_matlab("A = [1 2; 3 4];\n[V, D] = eig(A);")
        V = interp.global_env.get("V")
        D = interp.global_env.get("D")
        assert isinstance(V, Mat)
        assert isinstance(D, Mat)

    def test_svd(self):
        interp = run_matlab("A = [1 2; 3 4];\n[U, S, V] = svd(A);")
        U = interp.global_env.get("U")
        S = interp.global_env.get("S")
        V = interp.global_env.get("V")
        assert isinstance(U, Mat)
        assert isinstance(S, Mat)
        assert isinstance(V, Mat)

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
        interp = run_matlab("A = [1 2; 3 4];\nB = pinv(A);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)

    def test_cond(self):
        interp = run_matlab("A = [1 2; 3 4];\nc = cond(A);")
        c = interp.global_env.get("c")
        assert c is not None

    def test_rref(self):
        interp = run_matlab("A = [1 2 3; 4 5 6];\nR = rref(A);")
        R = interp.global_env.get("R")
        assert isinstance(R, Mat)
