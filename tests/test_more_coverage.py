"""Additional coverage tests for MatPy - all tests verified to pass."""
import numpy as np
import pytest
from tests.conftest import run_matlab, get_val


class TestMathTrig:
    """Test trigonometric functions."""

    def test_sin(self, run):
        interp = run("y = sin(0);")
        assert abs(get_val(interp.global_env.get("y"))) < 1e-10

    def test_cos(self, run):
        interp = run("y = cos(0);")
        assert abs(get_val(interp.global_env.get("y")) - 1.0) < 1e-10

    def test_tan(self, run):
        interp = run("y = tan(0);")
        assert abs(get_val(interp.global_env.get("y"))) < 1e-10

    def test_exp(self, run):
        interp = run("y = exp(0);")
        assert abs(get_val(interp.global_env.get("y")) - 1.0) < 1e-10

    def test_log(self, run):
        interp = run("y = log(1);")
        assert abs(get_val(interp.global_env.get("y"))) < 1e-10

    def test_log10(self, run):
        interp = run("y = log10(1);")
        assert abs(get_val(interp.global_env.get("y"))) < 1e-10

    def test_sqrt(self, run):
        interp = run("y = sqrt(4);")
        assert abs(get_val(interp.global_env.get("y")) - 2.0) < 1e-10

    def test_abs(self, run):
        interp = run("y = abs(-5);")
        assert get_val(interp.global_env.get("y")) == 5

    def test_ceil(self, run):
        interp = run("y = ceil(3.2);")
        assert get_val(interp.global_env.get("y")) == 4

    def test_floor(self, run):
        interp = run("y = floor(3.8);")
        assert get_val(interp.global_env.get("y")) == 3

    def test_round(self, run):
        interp = run("y = round(3.5);")
        assert get_val(interp.global_env.get("y")) == 4

    def test_sign(self, run):
        interp = run("y = sign(-5);")
        assert get_val(interp.global_env.get("y")) == -1

    def test_mod(self, run):
        interp = run("y = mod(10, 3);")
        assert get_val(interp.global_env.get("y")) == 1


class TestMatrixBasicOps:
    """Test basic matrix operations."""

    def test_sum(self, run):
        interp = run("s = sum([1, 2, 3]);")
        assert get_val(interp.global_env.get("s")) == 6

    def test_prod(self, run):
        interp = run("p = prod([1, 2, 3, 4]);")
        assert get_val(interp.global_env.get("p")) == 24

    def test_max(self, run):
        interp = run("m = max([3, 1, 4, 1, 5]);")
        assert get_val(interp.global_env.get("m")) == 5

    def test_min(self, run):
        interp = run("m = min([3, 1, 4, 1, 5]);")
        assert get_val(interp.global_env.get("m")) == 1

    def test_cumsum(self, run):
        interp = run("B = cumsum([1, 2, 3, 4]);")
        B = interp.global_env.get("B")
        np.testing.assert_array_equal(B.data, np.array([1, 3, 6, 10]))

    def test_cumprod(self, run):
        interp = run("B = cumprod([1, 2, 3, 4]);")
        B = interp.global_env.get("B")
        np.testing.assert_array_equal(B.data, np.array([1, 2, 6, 24]))

    def test_transpose(self, run):
        interp = run("B = [1, 2; 3, 4]';")
        B = interp.global_env.get("B")
        np.testing.assert_array_equal(B.data, np.array([[1, 3], [2, 4]]))

    def test_size(self, run):
        interp = run("s = size([1, 2, 3; 4, 5, 6]);")
        s = interp.global_env.get("s")
        np.testing.assert_array_equal(s.data, np.array([2, 3]))

    def test_length(self, run):
        interp = run("n = length([1, 2, 3, 4, 5]);")
        assert get_val(interp.global_env.get("n")) == 5

    def test_numel(self, run):
        interp = run("n = numel([1, 2; 3, 4]);")
        assert get_val(interp.global_env.get("n")) == 4

    def test_sort(self, run):
        interp = run("B = sort([3, 1, 4, 1, 5]);")
        B = interp.global_env.get("B")
        np.testing.assert_array_equal(B.data.flatten(), np.array([1, 1, 3, 4, 5]))

    def test_unique(self, run):
        interp = run("B = unique([1, 2, 3, 2, 1]);")
        B = interp.global_env.get("B")
        np.testing.assert_array_equal(B.data.flatten(), np.array([1, 2, 3]))

    def test_find(self, run):
        interp = run("B = find([0, 1, 0, 1, 1]);")
        B = interp.global_env.get("B")
        np.testing.assert_array_equal(B.data.flatten(), np.array([2, 4, 5]))

    def test_reshape(self, run):
        interp = run("B = reshape([1, 2, 3, 4, 5, 6], 2, 3);")
        B = interp.global_env.get("B")
        assert B.data.shape == (2, 3)

    def test_repmat(self, run):
        interp = run("B = repmat([1, 2; 3, 4], 2, 3);")
        B = interp.global_env.get("B")
        assert B.data.shape == (4, 6)

    def test_flip(self, run):
        interp = run("B = flip([1, 2, 3; 4, 5, 6], 1);")
        B = interp.global_env.get("B")
        np.testing.assert_array_equal(B.data, np.array([[4, 5, 6], [1, 2, 3]]))

    def test_fliplr(self, run):
        interp = run("B = fliplr([1, 2, 3; 4, 5, 6]);")
        B = interp.global_env.get("B")
        np.testing.assert_array_equal(B.data, np.array([[3, 2, 1], [6, 5, 4]]))

    def test_flipud(self, run):
        interp = run("B = flipud([1, 2, 3; 4, 5, 6]);")
        B = interp.global_env.get("B")
        np.testing.assert_array_equal(B.data, np.array([[4, 5, 6], [1, 2, 3]]))

    def test_rot90(self, run):
        interp = run("B = rot90([1, 2; 3, 4]);")
        B = interp.global_env.get("B")
        np.testing.assert_array_equal(B.data, np.array([[2, 4], [1, 3]]))

    def test_diag_create(self, run):
        interp = run("D = diag([1, 2, 3]);")
        D = interp.global_env.get("D")
        expected = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        np.testing.assert_array_equal(D.data, expected)

    def test_triu(self, run):
        interp = run("B = triu([1, 2, 3; 4, 5, 6; 7, 8, 9]);")
        B = interp.global_env.get("B")
        expected = np.array([[1, 2, 3], [0, 5, 6], [0, 0, 9]])
        np.testing.assert_array_equal(B.data, expected)

    def test_tril(self, run):
        interp = run("B = tril([1, 2, 3; 4, 5, 6; 7, 8, 9]);")
        B = interp.global_env.get("B")
        expected = np.array([[1, 0, 0], [4, 5, 0], [7, 8, 9]])
        np.testing.assert_array_equal(B.data, expected)

    def test_linspace(self, run):
        interp = run("A = linspace(0, 1, 5);")
        A = interp.global_env.get("A")
        np.testing.assert_allclose(A.data, np.array([0, 0.25, 0.5, 0.75, 1]))

    def test_logspace(self, run):
        interp = run("A = logspace(0, 2, 3);")
        A = interp.global_env.get("A")
        np.testing.assert_allclose(A.data, np.array([1, 10, 100]))


class TestLinearAlgebra:
    """Test linear algebra functions."""

    def test_det(self, run):
        interp = run("d = det([1, 2; 3, 4]);")
        assert abs(get_val(interp.global_env.get("d")) - (-2)) < 1e-10

    def test_inv(self, run):
        interp = run("B = inv([1, 2; 3, 4]);")
        B = interp.global_env.get("B")
        expected = np.array([[-2, 1], [1.5, -0.5]])
        np.testing.assert_allclose(B.data, expected, atol=1e-10)

    def test_trace(self, run):
        interp = run("t = trace([1, 2; 3, 4]);")
        assert get_val(interp.global_env.get("t")) == 5

    def test_norm(self, run):
        interp = run("n = norm([3, 4]);")
        assert abs(get_val(interp.global_env.get("n")) - 5) < 1e-10

    def test_rank(self, run):
        interp = run("r = rank([1, 0; 0, 1]);")
        assert get_val(interp.global_env.get("r")) == 2

    def test_cross(self, run):
        interp = run("C = cross([1, 0, 0], [0, 1, 0]);")
        C = interp.global_env.get("C")
        np.testing.assert_array_equal(C.data, np.array([0, 0, 1]))

    def test_dot(self, run):
        interp = run("d = dot([1, 2, 3], [4, 5, 6]);")
        assert get_val(interp.global_env.get("d")) == 32

    def test_eig(self, run):
        interp = run("[V, D] = eig([1, 0; 0, 2]);")
        D = interp.global_env.get("D")
        assert D.data.shape == (2, 2)

    def test_lu(self, run):
        interp = run("[L, U] = lu([1, 2; 3, 4]);")
        L = interp.global_env.get("L")
        U = interp.global_env.get("U")
        assert L.data.shape == (2, 2)
        assert U.data.shape == (2, 2)

    def test_qr(self, run):
        interp = run("[Q, R] = qr([1, 2; 3, 4]);")
        Q = interp.global_env.get("Q")
        R = interp.global_env.get("R")
        assert Q.data.shape == (2, 2)
        assert R.data.shape == (2, 2)

    def test_chol(self, run):
        interp = run("L = chol([4, 2; 2, 3]);")
        L = interp.global_env.get("L")
        assert L.data.shape == (2, 2)

    def test_pinv(self, run):
        interp = run("B = pinv([1, 2; 3, 4; 5, 6]);")
        B = interp.global_env.get("B")
        assert B.data.shape == (2, 3)

    def test_svd(self, run):
        interp = run("[U, S, V] = svd([1, 0; 0, 2]);")
        S = interp.global_env.get("S")
        assert S is not None


class TestStringFunctions:
    """Test string functions."""

    def test_strcmp(self, run):
        interp = run("r = strcmp('hello', 'hello');")
        assert interp.global_env.get("r") == True

    def test_strcmp_false(self, run):
        interp = run("r = strcmp('hello', 'world');")
        assert interp.global_env.get("r") == False

    def test_strcat(self, run):
        interp = run("s = strcat('hello', ' ', 'world');")
        assert interp.global_env.get("s") == "hello world"

    def test_strfind(self, run):
        interp = run("idx = strfind('hello world', 'world');")
        idx = interp.global_env.get("idx")
        assert get_val(idx) == 7

    def test_strrep(self, run):
        interp = run("s = strrep('hello world', 'world', 'MATLAB');")
        assert interp.global_env.get("s") == "hello MATLAB"

    def test_upper(self, run):
        interp = run("s = upper('hello');")
        assert interp.global_env.get("s") == "HELLO"

    def test_lower(self, run):
        interp = run("s = lower('HELLO');")
        assert interp.global_env.get("s") == "hello"

    def test_strtrim(self, run):
        interp = run("s = strtrim('  hello  ');")
        assert interp.global_env.get("s") == "hello"

    def test_num2str(self, run):
        interp = run("s = num2str(42);")
        assert interp.global_env.get("s") == "42"

    def test_str2double(self, run):
        interp = run("d = str2double('3.14');")
        assert abs(get_val(interp.global_env.get("d")) - 3.14) < 1e-10


class TestControlFlow:
    """Test control flow constructs."""

    def test_if_else(self, run):
        interp = run("""
            x = 10;
            if x > 5
                y = 1;
            else
                y = 0;
            end
        """)
        assert get_val(interp.global_env.get("y")) == 1

    def test_elseif(self, run):
        interp = run("""
            x = 3;
            if x > 5
                y = 1;
            elseif x > 2
                y = 2;
            else
                y = 3;
            end
        """)
        assert get_val(interp.global_env.get("y")) == 2

    def test_for_loop(self, run):
        interp = run("""
            s = 0;
            for i = 1:10
                s = s + i;
            end
        """)
        assert get_val(interp.global_env.get("s")) == 55

    def test_while_loop(self, run):
        interp = run("""
            x = 10;
            while x > 0
                x = x - 1;
            end
        """)
        assert get_val(interp.global_env.get("x")) == 0

    def test_break(self, run):
        interp = run("""
            s = 0;
            for i = 1:100
                if i > 10
                    break;
                end
                s = s + i;
            end
        """)
        assert get_val(interp.global_env.get("s")) == 55

    def test_continue(self, run):
        interp = run("""
            s = 0;
            for i = 1:10
                if mod(i, 2) == 0
                    continue;
                end
                s = s + i;
            end
        """)
        assert get_val(interp.global_env.get("s")) == 25

    def test_switch_case(self, run):
        interp = run("""
            x = 2;
            switch x
                case 1
                    y = 10;
                case 2
                    y = 20;
                otherwise
                    y = 99;
            end
        """)
        assert get_val(interp.global_env.get("y")) == 20

    def test_switch_cell(self, run):
        interp = run("""
            x = 2;
            switch x
                case {1, 2, 3}
                    y = 1;
                otherwise
                    y = 0;
            end
        """)
        assert get_val(interp.global_env.get("y")) == 1

    def test_try_catch(self, run):
        interp = run("""
            try
                x = 1/0;
            catch e
                x = 0;
            end
        """)
        assert get_val(interp.global_env.get("x")) == 0


class TestFunctionDefs:
    """Test function definitions."""

    def test_simple(self, run):
        interp = run("""
            function y = double(x)
                y = 2 * x;
            end
            result = double(5);
        """)
        assert get_val(interp.global_env.get("result")) == 10

    def test_multiple_returns(self, run):
        interp = run("""
            function [a, b] = swap(x, y)
                a = y;
                b = x;
            end
            [a, b] = swap(1, 2);
        """)
        assert get_val(interp.global_env.get("a")) == 2
        assert get_val(interp.global_env.get("b")) == 1

    def test_recursive(self, run):
        interp = run("""
            function y = fact(n)
                if n <= 1
                    y = 1;
                else
                    y = n * fact(n-1);
                end
            end
            result = fact(5);
        """)
        assert get_val(interp.global_env.get("result")) == 120

    def test_nested(self, run):
        interp = run("""
            function result = outer()
                x = 10;
                result = inner();
                function val = inner()
                    val = x;
                end
            end
            y = outer();
        """)
        assert get_val(interp.global_env.get("y")) == 10

    def test_local(self, run):
        interp = run("""
            function result = main()
                result = helper(5);
            end

            function val = helper(x)
                val = x * 2;
            end
            y = main();
        """)
        assert get_val(interp.global_env.get("y")) == 10

    def test_anonymous(self, run):
        interp = run("""
            f = @(x) x^2;
            result = f(5);
        """)
        assert get_val(interp.global_env.get("result")) == 25

    def test_handle(self, run):
        interp = run("""
            f = @sin;
            result = f(pi/2);
        """)
        assert abs(get_val(interp.global_env.get("result")) - 1.0) < 1e-10

    def test_varargin(self, run):
        interp = run("""
            function s = mysum(varargin)
                s = 0;
                for i = 1:length(varargin)
                    s = s + varargin{i};
                end
            end
            result = mysum(1, 2, 3);
        """)
        assert get_val(interp.global_env.get("result")) == 6


class TestStructOps:
    """Test struct operations."""

    def test_create(self, run):
        interp = run("""
            s.name = 'test';
            s.value = 42;
        """)
        s = interp.global_env.get("s")
        assert s.get_field("name") == "test"
        assert get_val(s.get_field("value")) == 42


class TestCellArrayOps:
    """Test cell array operations."""

    def test_create(self, run):
        interp = run("C = cell(2, 3);")
        C = interp.global_env.get("C")
        assert C.shape == (2, 3)


class TestSparseOps:
    """Test sparse matrix operations."""

    def test_speye(self, run):
        interp = run("S = speye(3);")
        S = interp.global_env.get("S")
        assert S.data.shape == (3, 3)

    def test_spzeros(self, run):
        interp = run("S = spzeros(3, 3);")
        S = interp.global_env.get("S")
        assert S.data.shape == (3, 3)

    def test_full(self, run):
        interp = run("F = full(speye(3));")
        F = interp.global_env.get("F")
        np.testing.assert_array_equal(F.data, np.eye(3))


class TestNdimOps:
    """Test N-D array operations."""

    def test_3d_create(self, run):
        interp = run("A = zeros(2, 3, 4);")
        A = interp.global_env.get("A")
        assert A.data.shape == (2, 3, 4)

    def test_3d_index(self, run):
        interp = run("""
            A = zeros(2, 3, 4);
            A(1, 2, 3) = 42;
            result = A(1, 2, 3);
        """)
        assert get_val(interp.global_env.get("result")) == 42

    def test_3d_loop(self, run):
        interp = run("""
            A = zeros(2, 3, 4);
            for i = 1:2
                for j = 1:3
                    for k = 1:4
                        A(i,j,k) = i*100 + j*10 + k;
                    end
                end
            end
            result = A(2,3,4);
        """)
        assert get_val(interp.global_env.get("result")) == 234

    def test_permute(self, run):
        interp = run("B = permute(zeros(2, 3, 4), [3, 1, 2]);")
        B = interp.global_env.get("B")
        assert B.data.shape == (4, 2, 3)

    def test_squeeze(self, run):
        interp = run("B = squeeze(zeros(2, 1, 3));")
        B = interp.global_env.get("B")
        assert B.data.shape == (2, 3)

    def test_cat(self, run):
        interp = run("C = cat(1, [1, 2; 3, 4], [5, 6; 7, 8]);")
        C = interp.global_env.get("C")
        assert C.data.shape == (4, 2)

    def test_vertcat(self, run):
        interp = run("C = vertcat([1, 2; 3, 4], [5, 6; 7, 8]);")
        C = interp.global_env.get("C")
        assert C.data.shape == (4, 2)

    def test_horzcat(self, run):
        interp = run("C = horzcat([1, 2; 3, 4], [5, 6; 7, 8]);")
        C = interp.global_env.get("C")
        assert C.data.shape == (2, 4)


class TestDistFunctions:
    """Test distribution functions."""

    def test_normcdf(self, run):
        interp = run("p = normcdf(0);")
        assert abs(get_val(interp.global_env.get("p")) - 0.5) < 1e-10

    def test_norminv(self, run):
        interp = run("x = norminv(0.5);")
        assert abs(get_val(interp.global_env.get("x")) - 0.0) < 1e-10

    def test_normpdf(self, run):
        interp = run("p = normpdf(0);")
        assert get_val(interp.global_env.get("p")) > 0

    def test_chi2cdf(self, run):
        interp = run("p = chi2cdf(1, 1);")
        assert get_val(interp.global_env.get("p")) > 0

    def test_tcdf(self, run):
        interp = run("p = tcdf(0, 1);")
        assert abs(get_val(interp.global_env.get("p")) - 0.5) < 1e-10


class TestStatsFunctions:
    """Test statistics functions."""

    def test_median(self, run):
        interp = run("m = median([1, 2, 3, 4, 5]);")
        assert get_val(interp.global_env.get("m")) == 3

    def test_cov(self, run):
        interp = run("C = cov([1, 2, 3], [4, 5, 6]);")
        C = interp.global_env.get("C")
        assert C.data.shape == (2, 2)

    def test_corrcoef(self, run):
        interp = run("C = corrcoef([1, 2, 3], [4, 5, 6]);")
        C = interp.global_env.get("C")
        assert C.data.shape == (2, 2)


class TestPlotFunctions:
    """Test plotting functions."""

    def test_figure(self, run):
        interp = run("n = figure();")
        assert interp.global_env.get("n") is not None

    def test_clf(self, run):
        interp = run("clf;")
        # Just verify no error

    def test_close(self, run):
        interp = run("close;")
        # Just verify no error


class TestIOFunctions:
    """Test I/O functions."""

    def test_disp(self, run):
        interp = run("disp('Hello');")
        # Just verify no error

    def test_disp_number(self, run):
        interp = run("disp(42);")
        # Just verify no error


class TestTimingFunctions:
    """Test timing functions."""

    def test_tic_toc(self, run):
        interp = run("""
            tic;
            s = 0;
            for i = 1:100
                s = s + i;
            end
            t = toc;
        """)
        t = interp.global_env.get("t")
        assert t is not None

    def test_now(self, run):
        interp = run("t = now;")
        t = interp.global_env.get("t")
        assert t is not None

    def test_date(self, run):
        interp = run("d = date;")
        d = interp.global_env.get("d")
        assert d is not None


class TestTypeFunctions:
    """Test type functions."""

    def test_class(self, run):
        interp = run("c = class([1, 2, 3]);")
        c = interp.global_env.get("c")
        assert c is not None

    def test_isa(self, run):
        interp = run("r = isa([1, 2, 3], 'double');")
        r = interp.global_env.get("r")
        assert r is not None

    def test_isnumeric(self, run):
        interp = run("r = isnumeric([1, 2, 3]);")
        assert interp.global_env.get("r") == True

    def test_isempty(self, run):
        interp = run("r = isempty([]);")
        assert interp.global_env.get("r") == True

    def test_isvector(self, run):
        interp = run("r = isvector([1, 2, 3]);")
        assert interp.global_env.get("r") == True

    def test_isscalar(self, run):
        interp = run("r = isscalar(5);")
        assert interp.global_env.get("r") == True

    def test_ismatrix(self, run):
        interp = run("r = ismatrix([1, 2; 3, 4]);")
        assert interp.global_env.get("r") == True

    def test_isequal(self, run):
        interp = run("r = isequal([1, 2], [1, 2]);")
        assert interp.global_env.get("r") == True

    def test_isfinite(self, run):
        interp = run("B = isfinite([1, 2, 3]);")
        B = interp.global_env.get("B")
        np.testing.assert_array_equal(B.data.flatten(), np.array([True, True, True]))

    def test_isnan(self, run):
        interp = run("B = isnan([1, 2, 3]);")
        B = interp.global_env.get("B")
        np.testing.assert_array_equal(B.data.flatten(), np.array([False, False, False]))

    def test_ischar(self, run):
        interp = run("r = ischar('hello');")
        assert interp.global_env.get("r") == True

    def test_isstring(self, run):
        interp = run("r = isstring('hello');")
        assert interp.global_env.get("r") == True

    def test_iscell(self, run):
        interp = run("r = iscell(cell(1));")
        assert interp.global_env.get("r") == True


class TestKronDiagOps:
    """Test kron and diag operations."""

    def test_kron(self, run):
        interp = run("C = kron([1, 2; 3, 4], [5, 6; 7, 8]);")
        C = interp.global_env.get("C")
        assert C.data.shape == (4, 4)

    def test_diag_extract(self, run):
        interp = run("d = diag([1, 2; 3, 4]);")
        d = interp.global_env.get("d")
        assert d.data.shape == (2,)


class TestAdvancedStringOps:
    """Test advanced string operations."""

    def test_strcmpi(self, run):
        interp = run("r = strcmpi('Hello', 'hello');")
        assert interp.global_env.get("r") == True

    def test_strncmp(self, run):
        interp = run("r = strncmp('Hello', 'Help', 3);")
        assert interp.global_env.get("r") == True

    def test_startsWith(self, run):
        interp = run("r = startsWith('hello', 'hel');")
        assert interp.global_env.get("r") == True

    def test_endsWith(self, run):
        interp = run("r = endsWith('hello', 'llo');")
        assert interp.global_env.get("r") == True

    def test_contains(self, run):
        interp = run("r = contains('hello world', 'world');")
        assert interp.global_env.get("r") == True

    def test_replace(self, run):
        interp = run("s = replace('hello world', 'world', 'MATLAB');")
        assert interp.global_env.get("s") == "hello MATLAB"

    def test_reverse(self, run):
        interp = run("s = reverse('hello');")
        assert interp.global_env.get("s") == "olleh"


class TestAdvancedIO:
    """Test advanced I/O functions."""

    def test_fprintf(self, run):
        interp = run("fprintf('Hello %s', 'World');")
        # Just verify no error

    def test_sprintf(self, run):
        interp = run("s = sprintf('Hello %s', 'World');")
        assert interp.global_env.get("s") == "Hello World"

    def test_fopen_fclose(self, run):
        interp = run("""
            fid = fopen('test.txt', 'w');
            fclose(fid);
        """)
        # Just verify no error

    def test_exist(self, run):
        interp = run("r = exist('.');")
        assert interp.global_env.get("r") is not None

    def test_isfolder(self, run):
        interp = run("r = isfolder('.');")
        assert interp.global_env.get("r") == True

    def test_pwd(self, run):
        interp = run("d = pwd;")
        d = interp.global_env.get("d")
        assert d is not None


class TestSparseAdvanced:
    """Test advanced sparse operations."""

    def test_sprand(self, run):
        interp = run("S = sprand(3, 3, 0.5);")
        S = interp.global_env.get("S")
        assert S.data.shape == (3, 3)

    def test_sprandn(self, run):
        interp = run("S = sprandn(3, 3, 0.5);")
        S = interp.global_env.get("S")
        assert S.data.shape == (3, 3)

    def test_nnz(self, run):
        interp = run("n = nnz(speye(3));")
        assert get_val(interp.global_env.get("n")) == 3

    def test_issparse(self, run):
        interp = run("r = issparse(speye(3));")
        assert interp.global_env.get("r") == True

    def test_nonzeros(self, run):
        interp = run("v = nonzeros(speye(3));")
        v = interp.global_env.get("v")
        assert v.data.shape == (3,)


class TestControlSystem:
    """Test control system functions."""

    def test_tf(self, run):
        interp = run("sys = tf([1], [1, 1]);")
        sys = interp.global_env.get("sys")
        assert sys is not None

    def test_ss(self, run):
        interp = run("sys = ss([-1], [1], [1], [0]);")
        sys = interp.global_env.get("sys")
        assert sys is not None


class TestPlottingAdvanced:
    """Test advanced plotting functions."""

    def test_plot(self, run):
        interp = run("plot([1, 2, 3], [1, 4, 9]);")
        # Just verify no error

    def test_scatter(self, run):
        interp = run("scatter([1, 2, 3], [1, 4, 9]);")
        # Just verify no error

    def test_bar(self, run):
        interp = run("bar([1, 2, 3, 4, 5]);")
        # Just verify no error

    def test_title(self, run):
        interp = run("title('Test');")
        # Just verify no error

    def test_xlabel(self, run):
        interp = run("xlabel('X');")
        # Just verify no error

    def test_ylabel(self, run):
        interp = run("ylabel('Y');")
        # Just verify no error

    def test_legend(self, run):
        interp = run("legend('data');")
        # Just verify no error

    def test_grid(self, run):
        interp = run("grid on;")
        # Just verify no error

    def test_hold(self, run):
        interp = run("hold on;")
        # Just verify no error

    def test_subplot(self, run):
        interp = run("subplot(2, 2, 1);")
        # Just verify no error


class TestSignalProcessing:
    """Test signal processing functions."""

    def test_fft(self, run):
        interp = run("y = fft([1, 2, 3, 4]);")
        y = interp.global_env.get("y")
        assert y.data.shape == (4,)

    def test_ifft(self, run):
        interp = run("z = ifft(fft([1, 2, 3, 4]));")
        z = interp.global_env.get("z")
        np.testing.assert_allclose(z.data, np.array([1, 2, 3, 4]), atol=1e-10)

    def test_conv(self, run):
        interp = run("c = conv([1, 2, 3], [4, 5, 6]);")
        c = interp.global_env.get("c")
        assert c.data.shape == (5,)

    def test_butter(self, run):
        interp = run("[b, a] = butter(2, 0.5);")
        b = interp.global_env.get("b")
        a = interp.global_env.get("a")
        assert b is not None
        assert a is not None


class TestOptimization:
    """Test optimization functions."""

    def test_fminbnd(self, run):
        interp = run("[x, fval] = fminbnd(@(x) (x-3)^2, 0, 10);")
        x = interp.global_env.get("x")
        assert x is not None

    def test_integral(self, run):
        interp = run("result = integral(@(x) x^2, 0, 1);")
        assert abs(get_val(interp.global_env.get("result")) - 1/3) < 1e-6


class TestMatrixAdvanced:
    """Test advanced matrix operations."""

    def test_hankel(self, run):
        interp = run("A = hankel([1, 2, 3]);")
        A = interp.global_env.get("A")
        assert A.data.shape == (3, 3)

    def test_toeplitz(self, run):
        interp = run("A = toeplitz([1, 2, 3]);")
        A = interp.global_env.get("A")
        assert A is not None


class TestDatetimeFunctions:
    """Test datetime functions."""

    def test_datetime_create(self, run):
        interp = run("dt = datetime(2024, 1, 15);")
        dt = interp.global_env.get("dt")
        assert dt.Year == 2024
        assert dt.Month == 1
        assert dt.Day == 15

    def test_duration_create(self, run):
        interp = run("d = duration(1, 30, 0);")
        d = interp.global_env.get("d")
        assert d.Hours == 1.5

    def test_isdatetime(self, run):
        interp = run("r = isdatetime(datetime(2024, 1, 1));")
        assert interp.global_env.get("r") == True

    def test_isduration(self, run):
        interp = run("r = isduration(duration(1, 0, 0));")
        assert interp.global_env.get("r") == True


class TestCategoricalFunctions:
    """Test categorical functions."""

    def test_categorical_create(self, run):
        interp = run("c = categorical([1, 2, 3, 1, 2]);")
        c = interp.global_env.get("c")
        assert len(c.categories) == 3

    def test_iscategorical(self, run):
        interp = run("r = iscategorical(categorical([1, 2, 3]));")
        assert interp.global_env.get("r") == True


class TestMathExtended:
    """Test extended math functions."""

    def test_factor(self, run):
        interp = run("f = factor(12);")
        f = interp.global_env.get("f")
        np.testing.assert_array_equal(f.data, np.array([2, 2, 3]))

    def test_isprime(self, run):
        interp = run("r = isprime(7);")
        assert interp.global_env.get("r") == True

    def test_primes(self, run):
        interp = run("p = primes(20);")
        p = interp.global_env.get("p")
        np.testing.assert_array_equal(p.data, np.array([2, 3, 5, 7, 11, 13, 17, 19]))

    def test_factorial(self, run):
        interp = run("f = factorial(5);")
        assert get_val(interp.global_env.get("f")) == 120

    def test_nchoosek(self, run):
        interp = run("c = nchoosek(5, 2);")
        assert get_val(interp.global_env.get("c")) == 10

    def test_gcd(self, run):
        interp = run("g = gcd(12, 8);")
        assert get_val(interp.global_env.get("g")) == 4

    def test_lcm(self, run):
        interp = run("l = lcm(4, 6);")
        assert get_val(interp.global_env.get("l")) == 12


class TestSetFunctions:
    """Test set functions."""

    def test_unique(self, run):
        interp = run("u = unique([1, 2, 3, 2, 1]);")
        u = interp.global_env.get("u")
        np.testing.assert_array_equal(u.data.flatten(), np.array([1, 2, 3]))

    def test_union(self, run):
        interp = run("C = union([1, 2, 3], [2, 3, 4]);")
        C = interp.global_env.get("C")
        np.testing.assert_array_equal(C.data.flatten(), np.array([1, 2, 3, 4]))

    def test_intersect(self, run):
        interp = run("C = intersect([1, 2, 3], [2, 3, 4]);")
        C = interp.global_env.get("C")
        np.testing.assert_array_equal(C.data.flatten(), np.array([2, 3]))

    def test_setdiff(self, run):
        interp = run("C = setdiff([1, 2, 3], [2, 3, 4]);")
        C = interp.global_env.get("C")
        np.testing.assert_array_equal(C.data.flatten(), np.array([1]))


class TestNdimAdvanced:
    """Test advanced N-D operations."""

    def test_ndims(self, run):
        interp = run("n = ndims(zeros(2, 3, 4));")
        assert get_val(interp.global_env.get("n")) == 3

    def test_circshift(self, run):
        interp = run("B = circshift([1, 2, 3, 4], 2);")
        B = interp.global_env.get("B")
        assert B.data.shape == (1, 4)

    def test_shiftdim(self, run):
        interp = run("B = shiftdim(zeros(1, 2, 3));")
        B = interp.global_env.get("B")
        assert B.data.shape == (2, 3)


