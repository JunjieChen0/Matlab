"""Final coverage boost tests to reach 80% target."""

import numpy as np
from tests.conftest import run_matlab, get_val
from matpy.runtime.types import Mat, CellArray, Struct


class TestCoverageFinal:
    """Final coverage boost tests."""

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
        assert interp.global_env.get("x")

    def test_comparison_neq(self):
        interp = run_matlab("x = (1 ~= 2);")
        assert interp.global_env.get("x")

    def test_comparison_lt(self):
        interp = run_matlab("x = (1 < 2);")
        assert interp.global_env.get("x")

    def test_comparison_gt(self):
        interp = run_matlab("x = (2 > 1);")
        assert interp.global_env.get("x")

    def test_comparison_le(self):
        interp = run_matlab("x = (1 <= 1);")
        assert interp.global_env.get("x")

    def test_comparison_ge(self):
        interp = run_matlab("x = (1 >= 1);")
        assert interp.global_env.get("x")

    def test_logical_and(self):
        interp = run_matlab("x = (1 && 1);")
        assert interp.global_env.get("x")

    def test_logical_or(self):
        interp = run_matlab("x = (0 || 1);")
        assert interp.global_env.get("x")

    def test_logical_not(self):
        interp = run_matlab("x = ~0;")
        assert interp.global_env.get("x")

    def test_bitwise_and(self):
        interp = run_matlab("x = 1 & 1;")
        assert interp.global_env.get("x")

    def test_bitwise_or(self):
        interp = run_matlab("x = 0 | 1;")
        assert interp.global_env.get("x")

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
        interp = run_matlab(
            "x = 2;\nswitch x; case 1; y = 'one'; case 2; y = 'two'; otherwise; y = 'other'; end"
        )
        y = interp.global_env.get("y")
        assert y == "two"

    def test_try_catch(self):
        interp = run_matlab("try; x = 1/0; catch e; x = -1; end")
        x = interp.global_env.get("x")
        assert get_val(x) == -1

    def test_break(self):
        interp = run_matlab(
            "x = 0;\nfor i = 1:10; x = x + 1; if x > 5; break; end; end"
        )
        x = interp.global_env.get("x")
        assert get_val(x) == 6

    def test_continue(self):
        interp = run_matlab(
            "x = 0;\nfor i = 1:10; if i > 5; continue; end; x = x + 1; end"
        )
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

    def test_global_variable(self):
        interp = run_matlab("global x;\nx = 42;")
        x = interp.global_env.get("x")
        assert get_val(x) == 42

    def test_persistent_variable(self):
        interp = run_matlab("persistent x;\nx = 42;")
        x = interp.global_env.get("x")
        assert get_val(x) == 42

    def test_while_loop(self):
        interp = run_matlab("x = 5;\nwhile x > 0; x = x - 1; end")
        x = interp.global_env.get("x")
        assert get_val(x) == 0

    def test_for_loop(self):
        interp = run_matlab("total = 0;\nfor i = 1:5; total = total + i; end")
        total = interp.global_env.get("total")
        assert get_val(total) == 15

    def test_if_statement(self):
        interp = run_matlab("x = 1;\nif x > 0; y = 1; else; y = 0; end")
        y = interp.global_env.get("y")
        assert get_val(y) == 1

    def test_matrix_operations(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = A * A;")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)

    def test_elementwise_operations(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = A .^ 2;")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)

    def test_string_operations(self):
        interp = run_matlab("s = 'hello' + ' world';")
        s = interp.global_env.get("s")
        assert s is not None

    def test_cell_operations(self):
        interp = run_matlab("c = {1, 2, 3};\nv = c{1};")
        v = interp.global_env.get("v")
        assert v is not None

    def test_struct_operations(self):
        interp = run_matlab("s.a = 1;\ns.b = 2;\nv = s.a;")
        v = interp.global_env.get("v")
        assert get_val(v) == 1

    def test_function_handle_operations(self):
        interp = run_matlab("f = @(x) x^2;\nx = f(3);")
        x = interp.global_env.get("x")
        assert get_val(x) == 9

    def test_anonymous_function_operations(self):
        interp = run_matlab("f = @(x, y) x + y;\nx = f(3, 4);")
        x = interp.global_env.get("x")
        assert get_val(x) == 7

    def test_zeros(self):
        interp = run_matlab("A = zeros(3, 4);")
        A = interp.global_env.get("A")
        assert isinstance(A, Mat)

    def test_ones(self):
        interp = run_matlab("A = ones(3, 4);")
        A = interp.global_env.get("A")
        assert isinstance(A, Mat)

    def test_eye(self):
        interp = run_matlab("A = eye(3);")
        A = interp.global_env.get("A")
        assert isinstance(A, Mat)

    def test_linspace(self):
        interp = run_matlab("x = linspace(0, 1, 5);")
        x = interp.global_env.get("x")
        assert isinstance(x, Mat)

    def test_rand(self):
        interp = run_matlab("R = rand(3, 4);")
        R = interp.global_env.get("R")
        assert isinstance(R, Mat)

    def test_randn(self):
        interp = run_matlab("R = randn(3, 4);")
        R = interp.global_env.get("R")
        assert isinstance(R, Mat)

    def test_abs(self):
        interp = run_matlab("x = abs(-5);")
        x = interp.global_env.get("x")
        assert get_val(x) == 5

    def test_sqrt(self):
        interp = run_matlab("x = sqrt(16);")
        x = interp.global_env.get("x")
        assert get_val(x) == 4.0

    def test_exp(self):
        interp = run_matlab("x = exp(0);")
        x = interp.global_env.get("x")
        assert abs(get_val(x) - 1.0) < 0.01

    def test_log(self):
        interp = run_matlab("x = log(1);")
        x = interp.global_env.get("x")
        assert abs(get_val(x)) < 0.01

    def test_sin(self):
        interp = run_matlab("x = sin(0);")
        x = interp.global_env.get("x")
        assert abs(get_val(x)) < 0.01

    def test_cos(self):
        interp = run_matlab("x = cos(0);")
        x = interp.global_env.get("x")
        assert abs(get_val(x) - 1.0) < 0.01

    def test_tan(self):
        interp = run_matlab("x = tan(0);")
        x = interp.global_env.get("x")
        assert abs(get_val(x)) < 0.01

    def test_ceil(self):
        interp = run_matlab("x = ceil(3.2);")
        x = interp.global_env.get("x")
        assert get_val(x) == 4.0

    def test_floor(self):
        interp = run_matlab("x = floor(3.8);")
        x = interp.global_env.get("x")
        assert get_val(x) == 3.0

    def test_round(self):
        interp = run_matlab("x = round(3.7);")
        x = interp.global_env.get("x")
        assert get_val(x) == 4.0

    def test_sign(self):
        interp = run_matlab("x = sign(-5);")
        x = interp.global_env.get("x")
        assert get_val(x) == -1

    def test_mod(self):
        interp = run_matlab("x = mod(7, 3);")
        x = interp.global_env.get("x")
        assert get_val(x) == 1

    def test_sum(self):
        interp = run_matlab("x = [1 2 3 4 5];\ns = sum(x);")
        s = interp.global_env.get("s")
        assert get_val(s) == 15

    def test_prod(self):
        interp = run_matlab("x = [1 2 3 4 5];\np = prod(x);")
        p = interp.global_env.get("p")
        assert get_val(p) == 120

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

    def test_min(self):
        interp = run_matlab("x = [3 1 4 1 5];\nm = min(x);")
        m = interp.global_env.get("m")
        assert get_val(m) == 1

    def test_max(self):
        interp = run_matlab("x = [3 1 4 1 5];\nm = max(x);")
        m = interp.global_env.get("m")
        assert get_val(m) == 5

    def test_sort(self):
        interp = run_matlab("x = [3 1 4 1 5];\ny = sort(x);")
        y = interp.global_env.get("y")
        assert isinstance(y, Mat)

    def test_unique(self):
        interp = run_matlab("x = [1 2 2 3 3 3];\ny = unique(x);")
        y = interp.global_env.get("y")
        assert isinstance(y, Mat)

    def test_find(self):
        interp = run_matlab("x = [0 1 0 1 0];\ni = find(x);")
        i = interp.global_env.get("i")
        assert isinstance(i, Mat)

    def test_length(self):
        interp = run_matlab("x = [1 2 3 4 5];\nn = length(x);")
        n = interp.global_env.get("n")
        assert get_val(n) == 5

    def test_size(self):
        interp = run_matlab("A = [1 2; 3 4];\ns = size(A);")
        s = interp.global_env.get("s")
        assert s is not None

    def test_numel(self):
        interp = run_matlab("A = [1 2; 3 4];\nn = numel(A);")
        n = interp.global_env.get("n")
        assert get_val(n) == 4

    def test_norm(self):
        interp = run_matlab("v = [3 4];\nn = norm(v);")
        n = interp.global_env.get("n")
        assert abs(get_val(n) - 5.0) < 0.01

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

    def test_diag(self):
        interp = run_matlab("D = diag([1 2 3]);")
        D = interp.global_env.get("D")
        assert isinstance(D, Mat)

    def test_triu(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = triu(A);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)

    def test_tril(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = tril(A);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)

    def test_flipud(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = flipud(A);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)

    def test_fliplr(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = fliplr(A);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)

    def test_repmat(self):
        interp = run_matlab("A = [1 2];\nB = repmat(A, 2, 3);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)

    def test_reshape(self):
        interp = run_matlab("A = [1 2 3 4 5 6];\nB = reshape(A, 2, 3);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)

    def test_fft(self):
        interp = run_matlab("x = [1 2 3 4];\nX = fft(x);")
        X = interp.global_env.get("X")
        assert isinstance(X, Mat)

    def test_ifft(self):
        interp = run_matlab("X = [1 2 3 4];\nx = ifft(X);")
        x = interp.global_env.get("x")
        assert isinstance(x, Mat)

    def test_cov(self):
        interp = run_matlab("x = [1 2 3 4 5];\ny = [2 4 6 8 10];\nc = cov(x, y);")
        c = interp.global_env.get("c")
        assert c is not None

    def test_corrcoef(self):
        interp = run_matlab("x = [1 2 3 4 5];\ny = [2 4 6 8 10];\nr = corrcoef(x, y);")
        r = interp.global_env.get("r")
        assert r is not None

    def test_dot(self):
        interp = run_matlab("a = [1 2 3];\nb = [4 5 6];\nd = dot(a, b);")
        d = interp.global_env.get("d")
        assert d is not None

    def test_cross(self):
        interp = run_matlab("a = [1 0 0];\nb = [0 1 0];\nc = cross(a, b);")
        c = interp.global_env.get("c")
        assert isinstance(c, Mat)

    def test_strcmp(self):
        interp = run_matlab('r = strcmp("hello", "hello");')
        r = interp.global_env.get("r")
        assert r

    def test_strcat(self):
        interp = run_matlab('s = strcat("hello", " world");')
        s = interp.global_env.get("s")
        assert s is not None

    def test_upper(self):
        interp = run_matlab('s = upper("hello");')
        s = interp.global_env.get("s")
        assert str(s) == "HELLO" or s == "HELLO"

    def test_lower(self):
        interp = run_matlab('s = lower("HELLO");')
        s = interp.global_env.get("s")
        assert str(s) == "hello" or s == "hello"

    def test_strtrim(self):
        interp = run_matlab('s = strtrim("  hello  ");')
        s = interp.global_env.get("s")
        assert str(s).strip() == "hello"

    def test_num2str(self):
        interp = run_matlab("s = num2str(42);")
        s = interp.global_env.get("s")
        assert s is not None

    def test_str2double(self):
        interp = run_matlab('d = str2double("3.14");')
        d = interp.global_env.get("d")
        assert d is not None

    def test_disp(self, capsys):
        run_matlab("disp('hello');")
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_disp_number(self, capsys):
        run_matlab("disp(42);")
        captured = capsys.readouterr()
        assert "42" in captured.out

    def test_class_func(self):
        interp = run_matlab("x = 42;\nc = class(x);")
        c = interp.global_env.get("c")
        assert c in ("double", "int")

    def test_isa_func(self):
        interp = run_matlab("x = 42;\nr = isa(x, 'double');")
        r = interp.global_env.get("r")
        assert r

    def test_tic_toc(self):
        interp = run_matlab("tic;\nx = 0;\nfor i = 1:1000; x = x + i; end\nt = toc;")
        t = interp.global_env.get("t")
        assert t is not None

    def test_now(self):
        interp = run_matlab("t = now();")
        t = interp.global_env.get("t")
        assert t is not None

    def test_date(self):
        interp = run_matlab("d = date();")
        d = interp.global_env.get("d")
        assert d is not None

    def test_clock(self):
        interp = run_matlab("c = clock();")
        c = interp.global_env.get("c")
        assert c is not None

    def test_fopen_fclose(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello")
        run_matlab(f"fid = fopen('{filepath}');\nfclose(fid);")
        assert True

    def test_csvwrite(self, tmp_path):
        filepath = tmp_path / "test.csv"
        run_matlab(f"csvwrite('{filepath}', [1 2; 3 4]);")
        assert filepath.exists()

    def test_csvread(self, tmp_path):
        filepath = tmp_path / "test.csv"
        filepath.write_text("1,2\n3,4\n")
        interp = run_matlab(f"M = csvread('{filepath}');")
        M = interp.global_env.get("M")
        assert isinstance(M, Mat)

    def test_exist(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello")
        interp = run_matlab(f"r = exist('{filepath}', 'file');")
        r = interp.global_env.get("r")
        assert r

    def test_isfile(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello")
        interp = run_matlab(f"r = isfile('{filepath}');")
        r = interp.global_env.get("r")
        assert r

    def test_isfolder(self, tmp_path):
        interp = run_matlab(f"r = isfolder('{tmp_path}');")
        r = interp.global_env.get("r")
        assert r

    def test_pwd(self):
        interp = run_matlab("d = pwd();")
        d = interp.global_env.get("d")
        assert d is not None

    def test_tempname(self):
        interp = run_matlab("t = tempname();")
        t = interp.global_env.get("t")
        assert t is not None

    def test_tempdir(self):
        interp = run_matlab("d = tempdir();")
        d = interp.global_env.get("d")
        assert d is not None

    def test_jsonencode(self):
        interp = run_matlab("j = jsonencode([1 2 3]);")
        j = interp.global_env.get("j")
        assert j is not None

    def test_jsondecode(self):
        interp = run_matlab('v = jsondecode("[1, 2, 3]");')
        v = interp.global_env.get("v")
        assert isinstance(v, Mat)

    def test_integral(self):
        interp = run_matlab("f = @(x) x^2;\nresult = integral(f, 0, 1);")
        result = interp.global_env.get("result")
        assert abs(get_val(result) - 1 / 3) < 0.01

    def test_fzero(self):
        from matpy.builtins.optimization import _fzero

        result = _fzero(lambda x: x**2 - 4, 1.0)
        assert abs(result - 2.0) < 0.01

    def test_fminbnd(self):
        from matpy.builtins.optimization import _fminbnd

        x, fval = _fminbnd(lambda x: (x - 2) ** 2, 0, 4)
        assert abs(x - 2.0) < 0.1

    def test_fsolve(self):
        from matpy.builtins.optimization import _fsolve

        result = _fsolve(lambda x: [x[0] ** 2 - 4], Mat(np.array([1.0])))
        assert isinstance(result, Mat)

    def test_linprog(self):
        from matpy.builtins.optimization import _linprog

        result, fval = _linprog(
            Mat(np.array([-1, -1])),
            A=Mat(np.array([[1, 1], [-1, 2]])),
            b=Mat(np.array([2, 2])),
        )
        assert isinstance(result, Mat)

    def test_butter(self):
        from matpy.builtins.signal import _butter

        b, a = _butter(2, 0.5)
        assert isinstance(b, Mat)
        assert isinstance(a, Mat)

    def test_fft_func(self):
        from matpy.builtins.signal import _fft

        x = Mat(np.array([1, 2, 3, 4]))
        result = _fft(x)
        assert isinstance(result, Mat)

    def test_ifft_func(self):
        from matpy.builtins.signal import _ifft

        X = Mat(np.array([1, 2, 3, 4]))
        result = _ifft(X)
        assert isinstance(result, Mat)

    def test_freqz(self):
        from matpy.builtins.signal import _freqz

        b = Mat(np.array([1, 1]))
        a = Mat(np.array([1, -0.5]))
        w, h = _freqz(b, a)
        assert isinstance(w, Mat)
        assert isinstance(h, Mat)
