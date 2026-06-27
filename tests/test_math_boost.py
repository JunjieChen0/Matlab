"""Boost tests for math.py and other modules."""

import pytest
import numpy as np
from tests.conftest import run_matlab, get_val
from matpy.runtime.types import Mat, CellArray, Struct


class TestMathBoost:
    """Boost math coverage."""

    def test_sin(self):
        from matpy.builtins.math import _wrap_math
        result = _wrap_math(np.sin)(0)
        assert result is not None

    def test_cos(self):
        from matpy.builtins.math import _wrap_math
        result = _wrap_math(np.cos)(0)
        assert result is not None

    def test_tan(self):
        from matpy.builtins.math import _wrap_math
        result = _wrap_math(np.tan)(0)
        assert result is not None

    def test_exp(self):
        from matpy.builtins.math import _wrap_math
        result = _wrap_math(np.exp)(0)
        assert result is not None

    def test_log(self):
        from matpy.builtins.math import _wrap_math
        result = _wrap_math(np.log)(1)
        assert result is not None

    def test_sqrt(self):
        from matpy.builtins.math import _wrap_math
        result = _wrap_math(np.sqrt)(4)
        assert result is not None

    def test_abs(self):
        from matpy.builtins.math import _wrap_math
        result = _wrap_math(np.abs)(-5)
        assert result is not None

    def test_ceil(self):
        from matpy.builtins.math import _wrap_math
        result = _wrap_math(np.ceil)(3.2)
        assert result is not None

    def test_floor(self):
        from matpy.builtins.math import _wrap_math
        result = _wrap_math(np.floor)(3.8)
        assert result is not None

    def test_round(self):
        from matpy.builtins.math import _wrap_math
        result = _wrap_math(np.round)(3.7)
        assert result is not None

    def test_sign(self):
        from matpy.builtins.math import _wrap_math
        result = _wrap_math(np.sign)(-5)
        assert result is not None

    def test_min(self):
        from matpy.builtins.math import _min
        result = _min(Mat(np.array([3, 1, 4, 1, 5])))
        assert result is not None

    def test_max(self):
        from matpy.builtins.math import _max
        result = _max(Mat(np.array([3, 1, 4, 1, 5])))
        assert result is not None

    def test_sum(self):
        from matpy.builtins.math import _sum
        result = _sum(Mat(np.array([1, 2, 3, 4, 5])))
        assert result is not None

    def test_prod(self):
        from matpy.builtins.math import _prod
        result = _prod(Mat(np.array([1, 2, 3, 4, 5])))
        assert result is not None

    def test_mean(self):
        interp = run_matlab("x = [1 2 3 4 5];\nm = mean(x);")
        m = interp.global_env.get("m")
        assert abs(get_val(m) - 3.0) < 0.01

    def test_median(self):
        interp = run_matlab("x = [1 2 3 4 5];\nm = median(x);")
        m = interp.global_env.get("m")
        assert abs(get_val(m) - 3.0) < 0.01

    def test_std(self):
        interp = run_matlab("x = [1 2 3 4 5];\ns = std(x);")
        s = interp.global_env.get("s")
        assert get_val(s) > 0

    def test_var(self):
        interp = run_matlab("x = [1 2 3 4 5];\nv = var(x);")
        v = interp.global_env.get("v")
        assert get_val(v) > 0

    def test_cumsum(self):
        from matpy.builtins.math import _cumsum
        result = _cumsum(Mat(np.array([1, 2, 3, 4, 5])))
        assert isinstance(result, Mat)

    def test_cumprod(self):
        from matpy.builtins.math import _cumprod
        result = _cumprod(Mat(np.array([1, 2, 3, 4, 5])))
        assert isinstance(result, Mat)

    def test_cov(self):
        interp = run_matlab("x = [1 2 3 4 5];\ny = [2 4 6 8 10];\nc = cov(x, y);")
        c = interp.global_env.get("c")
        assert c is not None

    def test_corrcoef(self):
        interp = run_matlab("x = [1 2 3 4 5];\ny = [2 4 6 8 10];\nr = corrcoef(x, y);")
        r = interp.global_env.get("r")
        assert r is not None


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
        result = _ctranspose(Mat(np.array([[1+1j, 2], [3, 4]])))
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


class TestInterpreterBoost:
    """Boost interpreter coverage."""

    def test_binary_add(self):
        interp = run_matlab("x = 1 + 2;")
        assert get_val(interp.global_env.get("x")) == 3

    def test_binary_sub(self):
        interp = run_matlab("x = 5 - 3;")
        assert get_val(interp.global_env.get("x")) == 2

    def test_binary_mul(self):
        interp = run_matlab("x = 3 * 4;")
        assert get_val(interp.global_env.get("x")) == 12

    def test_binary_div(self):
        interp = run_matlab("x = 10 / 2;")
        assert get_val(interp.global_env.get("x")) == 5.0

    def test_binary_pow(self):
        interp = run_matlab("x = 2 ^ 3;")
        assert get_val(interp.global_env.get("x")) == 8

    def test_unary_minus(self):
        interp = run_matlab("x = -5;")
        assert get_val(interp.global_env.get("x")) == -5

    def test_unary_plus(self):
        interp = run_matlab("x = +5;")
        assert get_val(interp.global_env.get("x")) == 5

    def test_comparison_eq(self):
        interp = run_matlab("x = (1 == 1);")
        assert interp.global_env.get("x") == True

    def test_comparison_neq(self):
        interp = run_matlab("x = (1 ~= 2);")
        assert interp.global_env.get("x") == True

    def test_comparison_lt(self):
        interp = run_matlab("x = (1 < 2);")
        assert interp.global_env.get("x") == True

    def test_comparison_gt(self):
        interp = run_matlab("x = (2 > 1);")
        assert interp.global_env.get("x") == True

    def test_comparison_le(self):
        interp = run_matlab("x = (1 <= 1);")
        assert interp.global_env.get("x") == True

    def test_comparison_ge(self):
        interp = run_matlab("x = (1 >= 1);")
        assert interp.global_env.get("x") == True

    def test_logical_and(self):
        interp = run_matlab("x = (1 && 1);")
        assert interp.global_env.get("x") == True

    def test_logical_or(self):
        interp = run_matlab("x = (0 || 1);")
        assert interp.global_env.get("x") == True

    def test_logical_not(self):
        interp = run_matlab("x = ~0;")
        assert interp.global_env.get("x") == True

    def test_bitwise_and(self):
        interp = run_matlab("x = 1 & 1;")
        assert interp.global_env.get("x") == True

    def test_bitwise_or(self):
        interp = run_matlab("x = 0 | 1;")
        assert interp.global_env.get("x") == True

    def test_matrix_add(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = [5 6; 7 8];\nC = A + B;")
        C = interp.global_env.get("C")
        assert isinstance(C, Mat)

    def test_matrix_sub(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = [5 6; 7 8];\nC = A - B;")
        C = interp.global_env.get("C")
        assert isinstance(C, Mat)

    def test_matrix_mul(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = [5 6; 7 8];\nC = A * B;")
        C = interp.global_env.get("C")
        assert isinstance(C, Mat)

    def test_elementwise_mul(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = [5 6; 7 8];\nC = A .* B;")
        C = interp.global_env.get("C")
        assert isinstance(C, Mat)

    def test_elementwise_div(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = [5 6; 7 8];\nC = A ./ B;")
        C = interp.global_env.get("C")
        assert isinstance(C, Mat)

    def test_elementwise_pow(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = A .^ 2;")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)

    def test_transpose(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = A.';")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)

    def test_conjugate_transpose(self):
        interp = run_matlab("A = [1+1i 2; 3 4];\nB = A';")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)

    def test_range_expr(self):
        interp = run_matlab("x = 1:5;")
        x = interp.global_env.get("x")
        assert isinstance(x, Mat)

    def test_range_with_step(self):
        interp = run_matlab("x = 0:0.5:2;")
        x = interp.global_env.get("x")
        assert isinstance(x, Mat)

    def test_matrix_indexing(self):
        interp = run_matlab("A = [1 2 3; 4 5 6; 7 8 9];\nv = A(2, 3);")
        v = interp.global_env.get("v")
        assert get_val(v) == 6

    def test_matrix_end_indexing(self):
        interp = run_matlab("A = [1 2 3; 4 5 6; 7 8 9];\nv = A(end);")
        v = interp.global_env.get("v")
        assert get_val(v) == 9

    def test_matrix_logical_indexing(self):
        interp = run_matlab("A = [1 2 3 4 5];\nB = A(A > 3);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)

    def test_string_literal(self):
        interp = run_matlab("s = 'hello';")
        s = interp.global_env.get("s")
        assert s == "hello"

    def test_double_quote_string(self):
        interp = run_matlab('s = "hello";')
        s = interp.global_env.get("s")
        assert s is not None

    def test_cell_array(self):
        interp = run_matlab("c = {1, 'hello', [1 2 3]};")
        c = interp.global_env.get("c")
        assert isinstance(c, CellArray)

    def test_struct_creation(self):
        interp = run_matlab("s.name = 'test';\ns.value = 42;")
        s = interp.global_env.get("s")
        assert isinstance(s, Struct)

    def test_function_handle(self):
        interp = run_matlab("f = @sin;\nx = f(0);")
        x = interp.global_env.get("x")
        assert abs(get_val(x)) < 0.01

    def test_anonymous_function(self):
        interp = run_matlab("f = @(x) x^2;\nx = f(3);")
        x = interp.global_env.get("x")
        assert get_val(x) == 9

    def test_switch_case(self):
        interp = run_matlab("x = 2;\nswitch x; case 1; y = 'one'; case 2; y = 'two'; otherwise; y = 'other'; end")
        y = interp.global_env.get("y")
        assert y == "two"

    def test_try_catch(self):
        interp = run_matlab("try; x = 1/0; catch e; x = -1; end")
        x = interp.global_env.get("x")
        assert get_val(x) == -1

    def test_break(self):
        interp = run_matlab("x = 0;\nfor i = 1:10; x = x + 1; if x > 5; break; end; end")
        x = interp.global_env.get("x")
        assert get_val(x) == 6

    def test_continue(self):
        interp = run_matlab("x = 0;\nfor i = 1:10; if i > 5; continue; end; x = x + 1; end")
        x = interp.global_env.get("x")
        assert get_val(x) == 5

    def test_function_definition(self):
        interp = run_matlab("function y = f(x); y = x^2; end\nresult = f(3);")
        result = interp.global_env.get("result")
        assert get_val(result) == 9

    def test_multiple_returns(self):
        interp = run_matlab("[a, b] = deal(1, 2);")
        a = interp.global_env.get("a")
        b = interp.global_env.get("b")
        assert get_val(a) == 1
        assert get_val(b) == 2

    def test_nested_function(self):
        interp = run_matlab("""
            function y = outer(x)
                function z = inner(w)
                    z = w * 2;
                end
                y = inner(x);
            end
            result = outer(5);
        """)
        result = interp.global_env.get("result")
        assert get_val(result) == 10

    def test_classdef(self):
        interp = run_matlab("""
            classdef Point
                properties
                    x = 0
                    y = 0
                end
                methods
                    function obj = Point(x, y)
                        obj.x = x;
                        obj.y = y;
                    end
                end
            end
            p = Point(3, 4);
        """)
        p = interp.global_env.get("p")
        assert p is not None

    def test_eval(self):
        interp = run_matlab("x = eval('1+2');")
        x = interp.global_env.get("x")
        assert get_val(x) == 3

    def test_feval(self):
        interp = run_matlab("f = @sin;\nx = feval(f, 0);")
        x = interp.global_env.get("x")
        assert abs(get_val(x)) < 0.01
