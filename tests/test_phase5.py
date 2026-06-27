"""Phase 5 tests: Engineering quality - comprehensive test coverage."""

import numpy as np
import pytest
from tests.conftest import run_matlab, get_val


class TestDatetimeTypes:
    """Test datetime and duration types."""

    def test_datetime_create(self, run):
        """Test datetime creation."""
        interp = run("dt = datetime(2024, 1, 15);")
        dt = interp.global_env.get("dt")
        assert dt.Year == 2024
        assert dt.Month == 1
        assert dt.Day == 15

    def test_datetime_string(self, run):
        """Test datetime from string."""
        interp = run("dt = datetime('2024-01-15 10:30:00');")
        dt = interp.global_env.get("dt")
        assert dt.Year == 2024
        assert dt.Hour == 10

    def test_datetime_fields(self, run):
        """Test datetime field access."""
        interp = run("""
            dt = datetime(2024, 6, 15, 14, 30, 45);
            y = dt.Year;
            m = dt.Month;
            d = dt.Day;
        """)
        assert get_val(interp.global_env.get("y")) == 2024
        assert get_val(interp.global_env.get("m")) == 6
        assert get_val(interp.global_env.get("d")) == 15

    def test_duration_create(self, run):
        """Test duration creation."""
        interp = run("d = duration(1, 30, 0);")
        d = interp.global_env.get("d")
        assert d.Hours == 1.5

    def test_duration_arithmetic(self, run):
        """Test duration arithmetic."""
        interp = run("""
            d1 = duration(1, 0, 0);
            d2 = duration(0, 30, 0);
        """)
        d1 = interp.global_env.get("d1")
        d2 = interp.global_env.get("d2")
        d3 = d1 + d2
        assert d3.Hours == 1.5

    def test_isdatetime(self, run):
        """Test isdatetime function."""
        interp = run("""
            dt = datetime(2024, 1, 1);
            result = isdatetime(dt);
        """)
        assert interp.global_env.get("result") == True

    def test_isduration(self, run):
        """Test isduration function."""
        interp = run("""
            d = duration(1, 0, 0);
            result = isduration(d);
        """)
        assert interp.global_env.get("result") == True


class TestCategoricalTypes:
    """Test categorical types."""

    def test_categorical_create(self, run):
        """Test categorical creation."""
        interp = run("""
            c = categorical([1, 2, 3, 1, 2]);
        """)
        c = interp.global_env.get("c")
        assert len(c.categories) == 3

    def test_categorical_categories(self, run):
        """Test categorical categories."""
        interp = run("""
            c = categorical([1, 2, 3]);
        """)
        c = interp.global_env.get("c")
        cats = c.categories
        assert len(cats) == 3

    def test_iscategorical(self, run):
        """Test iscategorical function."""
        interp = run("""
            c = categorical([1, 2, 3]);
            result = iscategorical(c);
        """)
        assert interp.global_env.get("result") == True


class TestAdditionalMathFunctions:
    """Test additional math functions."""

    def test_factor(self, run):
        """Test factor function."""
        interp = run("f = factor(12);")
        f = interp.global_env.get("f")
        np.testing.assert_array_equal(f.data, np.array([2, 2, 3]))

    def test_isprime(self, run):
        """Test isprime function."""
        interp = run("""
            r1 = isprime(7);
            r2 = isprime(4);
        """)
        assert interp.global_env.get("r1") == True
        assert interp.global_env.get("r2") == False

    def test_primes(self, run):
        """Test primes function."""
        interp = run("p = primes(20);")
        p = interp.global_env.get("p")
        np.testing.assert_array_equal(p.data, np.array([2, 3, 5, 7, 11, 13, 17, 19]))

    def test_factorial(self, run):
        """Test factorial function."""
        interp = run("f = factorial(5);")
        assert get_val(interp.global_env.get("f")) == 120

    def test_nchoosek(self, run):
        """Test nchoosek function."""
        interp = run("c = nchoosek(5, 2);")
        assert get_val(interp.global_env.get("c")) == 10

    def test_gcd(self, run):
        """Test gcd function."""
        interp = run("g = gcd(12, 8);")
        assert get_val(interp.global_env.get("g")) == 4

    def test_lcm(self, run):
        """Test lcm function."""
        interp = run("l = lcm(4, 6);")
        assert get_val(interp.global_env.get("l")) == 12


class TestSetOperations:
    """Test set operations."""

    def test_unique(self, run):
        """Test unique function."""
        interp = run("""
            A = [1, 2, 3, 2, 1];
            u = unique(A);
        """)
        u = interp.global_env.get("u")
        np.testing.assert_array_equal(u.data, np.array([1, 2, 3]))

    def test_union(self, run):
        """Test union function."""
        interp = run("""
            A = [1, 2, 3];
            B = [2, 3, 4];
            C = union(A, B);
        """)
        C = interp.global_env.get("C")
        np.testing.assert_array_equal(C.data, np.array([1, 2, 3, 4]))

    def test_intersect(self, run):
        """Test intersect function."""
        interp = run("""
            A = [1, 2, 3];
            B = [2, 3, 4];
            C = intersect(A, B);
        """)
        C = interp.global_env.get("C")
        np.testing.assert_array_equal(C.data, np.array([2, 3]))

    def test_setdiff(self, run):
        """Test setdiff function."""
        interp = run("""
            A = [1, 2, 3];
            B = [2, 3, 4];
            C = setdiff(A, B);
        """)
        C = interp.global_env.get("C")
        np.testing.assert_array_equal(C.data, np.array([1]))

    def test_ismember(self, run):
        """Test ismember function."""
        interp = run("""
            A = [1, 2, 3, 4, 5];
            B = [2, 4];
            C = ismember(A, B);
        """)
        C = interp.global_env.get("C")
        expected = np.array([[False, True, False, True, False]])
        np.testing.assert_array_equal(C.data, expected)


class TestArrayQueryFunctions:
    """Test array query functions."""

    def test_isvector(self, run):
        """Test isvector function."""
        interp = run("""
            A = [1, 2, 3];
            r = isvector(A);
        """)
        assert interp.global_env.get("r") == True

    def test_isscalar(self, run):
        """Test isscalar function."""
        interp = run("""
            A = 5;
            r = isscalar(A);
        """)
        assert interp.global_env.get("r") == True

    def test_isempty(self, run):
        """Test isempty function."""
        interp = run("""
            A = [];
            r = isempty(A);
        """)
        assert interp.global_env.get("r") == True

    def test_isequal(self, run):
        """Test isequal function."""
        interp = run("""
            A = [1, 2, 3];
            B = [1, 2, 3];
            r = isequal(A, B);
        """)
        assert interp.global_env.get("r") == True

    def test_isfinite(self, run):
        """Test isfinite function."""
        interp = run("""
            A = [1, inf, nan, 2];
            B = isfinite(A);
        """)
        B = interp.global_env.get("B")
        expected = np.array([True, False, False, True])
        np.testing.assert_array_equal(B.data.flatten(), expected)

    def test_isinf(self, run):
        """Test isinf function."""
        interp = run("""
            A = [1, inf, nan];
            B = isinf(A);
        """)
        B = interp.global_env.get("B")
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(B.data.flatten(), expected)

    def test_isnan(self, run):
        """Test isnan function."""
        interp = run("""
            A = [1, inf, nan];
            B = isnan(A);
        """)
        B = interp.global_env.get("B")
        expected = np.array([False, False, True])
        np.testing.assert_array_equal(B.data.flatten(), expected)


class TestSortingFunctions:
    """Test sorting functions."""

    def test_sort_ascending(self, run):
        """Test sort ascending."""
        interp = run("""
            A = [3, 1, 4, 1, 5, 9];
            B = sort(A);
        """)
        B = interp.global_env.get("B")
        np.testing.assert_array_equal(B.data, np.array([1, 1, 3, 4, 5, 9]))

    def test_sort_descending(self):
        """Test sort descending."""
        from matpy.builtins.common import _sort
        from matpy.runtime.types import Mat
        import numpy as np

        A = Mat(np.array([3, 1, 4, 1, 5, 9]))
        B = _sort(A, mode="descend")
        np.testing.assert_array_equal(B.data.flatten(), np.array([9, 5, 4, 3, 1, 1]))

    def test_find(self, run):
        """Test find function."""
        interp = run("""
            A = [0, 1, 0, 1, 1];
            B = find(A);
        """)
        B = interp.global_env.get("B")
        np.testing.assert_array_equal(B.data, np.array([2, 4, 5]))


class TestLinearAlgebraFunctions:
    """Test linear algebra functions."""

    def test_det(self, run):
        """Test determinant."""
        interp = run("""
            A = [1, 2; 3, 4];
            d = det(A);
        """)
        assert abs(get_val(interp.global_env.get("d")) - (-2)) < 1e-10

    def test_inv(self, run):
        """Test matrix inverse."""
        interp = run("""
            A = [1, 2; 3, 4];
            B = inv(A);
        """)
        B = interp.global_env.get("B")
        expected = np.array([[-2, 1], [1.5, -0.5]])
        np.testing.assert_allclose(B.data, expected, atol=1e-10)

    def test_trace(self, run):
        """Test matrix trace."""
        interp = run("""
            A = [1, 2; 3, 4];
            t = trace(A);
        """)
        assert get_val(interp.global_env.get("t")) == 5

    def test_norm(self, run):
        """Test matrix norm."""
        interp = run("""
            A = [3, 4];
            n = norm(A);
        """)
        assert abs(get_val(interp.global_env.get("n")) - 5) < 1e-10

    def test_rank(self, run):
        """Test matrix rank."""
        interp = run("""
            A = [1, 0; 0, 1];
            r = rank(A);
        """)
        assert get_val(interp.global_env.get("r")) == 2

    def test_cross(self, run):
        """Test cross product."""
        interp = run("""
            A = [1, 0, 0];
            B = [0, 1, 0];
            C = cross(A, B);
        """)
        C = interp.global_env.get("C")
        np.testing.assert_array_equal(C.data, np.array([0, 0, 1]))

    def test_dot(self, run):
        """Test dot product."""
        interp = run("""
            A = [1, 2, 3];
            B = [4, 5, 6];
            d = dot(A, B);
        """)
        assert get_val(interp.global_env.get("d")) == 32


class TestMatrixOperations:
    """Test matrix operations."""

    def test_diag_create(self, run):
        """Test diag to create diagonal matrix."""
        interp = run("""
            A = [1, 2, 3];
            D = diag(A);
        """)
        D = interp.global_env.get("D")
        expected = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        np.testing.assert_array_equal(D.data, expected)

    def test_triu(self, run):
        """Test triu function."""
        interp = run("""
            A = [1, 2, 3; 4, 5, 6; 7, 8, 9];
            B = triu(A);
        """)
        B = interp.global_env.get("B")
        expected = np.array([[1, 2, 3], [0, 5, 6], [0, 0, 9]])
        np.testing.assert_array_equal(B.data, expected)

    def test_tril(self, run):
        """Test tril function."""
        interp = run("""
            A = [1, 2, 3; 4, 5, 6; 7, 8, 9];
            B = tril(A);
        """)
        B = interp.global_env.get("B")
        expected = np.array([[1, 0, 0], [4, 5, 0], [7, 8, 9]])
        np.testing.assert_array_equal(B.data, expected)

    def test_repmat(self, run):
        """Test repmat function."""
        interp = run("""
            A = [1, 2; 3, 4];
            B = repmat(A, 2, 3);
        """)
        B = interp.global_env.get("B")
        assert B.data.shape == (4, 6)

    def test_linspace(self, run):
        """Test linspace function."""
        interp = run("""
            A = linspace(0, 1, 5);
        """)
        A = interp.global_env.get("A")
        np.testing.assert_allclose(A.data, np.array([0, 0.25, 0.5, 0.75, 1]))

    def test_logspace(self, run):
        """Test logspace function."""
        interp = run("""
            A = logspace(0, 2, 3);
        """)
        A = interp.global_env.get("A")
        np.testing.assert_allclose(A.data, np.array([1, 10, 100]))


class TestCellArrayFunctions:
    """Test cell array functions."""

    def test_cell_create(self, run):
        """Test cell creation."""
        interp = run("C = cell(2, 3);")
        C = interp.global_env.get("C")
        assert C.shape == (2, 3)

    def test_num2cell(self, run):
        """Test num2cell function."""
        interp = run("""
            A = [1, 2, 3; 4, 5, 6];
            C = num2cell(A);
        """)
        C = interp.global_env.get("C")
        # num2cell without dim creates cell for each element
        assert C.shape[0] == 6 or C.shape == (2, 3)


class TestSparseFunctions:
    """Test sparse matrix functions."""

    def test_speye(self, run):
        """Test speye function."""
        interp = run("S = speye(3);")
        S = interp.global_env.get("S")
        assert S.data.shape == (3, 3)

    def test_issparse(self, run):
        """Test issparse function."""
        interp = run("""
            S = speye(3);
            r = issparse(S);
        """)
        assert interp.global_env.get("r") == True

    def test_nnz(self, run):
        """Test nnz function."""
        interp = run("""
            S = speye(3);
            n = nnz(S);
        """)
        assert get_val(interp.global_env.get("n")) == 3


class TestStringFunctions:
    """Test string functions."""

    def test_strcmp(self, run):
        """Test strcmp function."""
        interp = run("r = strcmp('hello', 'hello');")
        assert interp.global_env.get("r") == True

    def test_strcat(self, run):
        """Test strcat function."""
        interp = run("s = strcat('hello', ' ', 'world');")
        assert interp.global_env.get("s") == "hello world"

    def test_strfind(self, run):
        """Test strfind function."""
        interp = run("idx = strfind('hello world', 'world');")
        idx = interp.global_env.get("idx")
        assert get_val(idx) == 7

    def test_upper(self, run):
        """Test upper function."""
        interp = run("s = upper('hello');")
        assert interp.global_env.get("s") == "HELLO"

    def test_lower(self, run):
        """Test lower function."""
        interp = run("s = lower('HELLO');")
        assert interp.global_env.get("s") == "hello"


class TestControlFlow:
    """Test control flow constructs."""

    def test_if_else(self, run):
        """Test if/else."""
        interp = run("""
            x = 10;
            if x > 5
                y = 1;
            else
                y = 0;
            end
        """)
        assert get_val(interp.global_env.get("y")) == 1

    def test_for_loop(self, run):
        """Test for loop."""
        interp = run("""
            s = 0;
            for i = 1:10
                s = s + i;
            end
        """)
        assert get_val(interp.global_env.get("s")) == 55

    def test_while_loop(self, run):
        """Test while loop."""
        interp = run("""
            x = 10;
            while x > 0
                x = x - 1;
            end
        """)
        assert get_val(interp.global_env.get("x")) == 0

    def test_switch_case(self, run):
        """Test switch/case."""
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

    def test_try_catch(self, run):
        """Test try/catch."""
        interp = run("""
            try
                x = 1/0;
            catch e
                x = 0;
            end
        """)
        assert get_val(interp.global_env.get("x")) == 0


class TestFunctionDefinitions:
    """Test function definitions."""

    def test_simple_function(self, run):
        """Test simple function."""
        interp = run("""
            function y = double(x)
                y = 2 * x;
            end
            result = double(5);
        """)
        assert get_val(interp.global_env.get("result")) == 10

    def test_multiple_returns(self, run):
        """Test multiple return values."""
        interp = run("""
            function [a, b] = swap(x, y)
                a = y;
                b = x;
            end
            [a, b] = swap(1, 2);
        """)
        assert get_val(interp.global_env.get("a")) == 2
        assert get_val(interp.global_env.get("b")) == 1

    def test_recursive_function(self, run):
        """Test recursive function."""
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
