"""Built-in function tests for MatPy."""

import pytest
import numpy as np
from tests.conftest import run_matlab, get_val
from matpy.runtime.types import Mat, CellArray


class TestMathFunctions:
    def test_abs(self):
        interp = run_matlab("a = abs(-5);")
        assert get_val(interp.global_env.get("a")) == 5

    def test_sqrt(self):
        interp = run_matlab("b = sqrt(16);")
        assert get_val(interp.global_env.get("b")) == 4.0

    def test_round(self):
        interp = run_matlab("c = round(3.7);")
        assert get_val(interp.global_env.get("c")) == 4.0

    def test_ceil(self):
        interp = run_matlab("d = ceil(3.2);")
        assert get_val(interp.global_env.get("d")) == 4.0

    def test_floor(self):
        interp = run_matlab("e = floor(3.8);")
        assert get_val(interp.global_env.get("e")) == 3.0


class TestMatrixFunctions:
    def test_transpose(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = transpose(A);")
        B = interp.global_env.get("B")
        assert np.allclose(B.data, np.array([[1, 3], [2, 4]]))

    def test_linspace(self):
        interp = run_matlab("x = linspace(0, 1, 5);")
        x = interp.global_env.get("x")
        assert isinstance(x, Mat)
        assert len(x.data) == 5
        assert np.isclose(x.data[0], 0)
        assert np.isclose(x.data[-1], 1)

    def test_inv(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = inv(A);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)
        assert abs(B.data[0, 0] - (-2.0)) < 0.01

    def test_det(self):
        interp = run_matlab("A = [1 2; 3 4];\nd = det(A);")
        assert abs(get_val(interp.global_env.get("d")) - (-2.0)) < 0.01

    def test_rank(self):
        interp = run_matlab("A = [1 2; 3 4];\nr = rank(A);")
        assert get_val(interp.global_env.get("r")) == 2

    def test_norm(self):
        interp = run_matlab("A = [1 2; 3 4];\nn = norm(A);")
        assert get_val(interp.global_env.get("n")) > 0

    def test_cond(self):
        interp = run_matlab("A = [1 2; 3 4];\nc = cond(A);")
        assert get_val(interp.global_env.get("c")) > 1


class TestMatrixDecomposition:
    def test_lu(self):
        interp = run_matlab("A = [1 2; 3 4];\n[L, U] = lu(A);")
        assert isinstance(interp.global_env.get("L"), Mat)
        assert isinstance(interp.global_env.get("U"), Mat)

    def test_qr(self):
        interp = run_matlab("A = [1 2; 3 4];\n[Q, R] = qr(A);")
        assert isinstance(interp.global_env.get("Q"), Mat)
        assert isinstance(interp.global_env.get("R"), Mat)

    def test_eig(self):
        interp = run_matlab("A = [1 0; 0 2];\n[V, D] = eig(A);")
        assert isinstance(interp.global_env.get("V"), Mat)
        assert isinstance(interp.global_env.get("D"), Mat)

    def test_svd(self):
        interp = run_matlab("A = [1 2; 3 4];\n[U, S, V] = svd(A);")
        assert isinstance(interp.global_env.get("U"), Mat)
        assert isinstance(interp.global_env.get("S"), Mat)
        assert isinstance(interp.global_env.get("V"), Mat)


class TestAdvancedMath:
    def test_integral(self):
        interp = run_matlab("result = integral(@(x) x^2, 0, 1);")
        assert abs(get_val(interp.global_env.get("result")) - 1 / 3) < 0.01

    def test_mean(self):
        interp = run_matlab("x = [1 2 3 4 5];\nm = mean(x);")
        assert get_val(interp.global_env.get("m")) == 3.0

    def test_std(self):
        interp = run_matlab("x = [1 2 3 4 5];\ns = std(x);")
        assert abs(get_val(interp.global_env.get("s")) - 1.5811) < 0.01


class TestDistributionFunctions:
    def test_normcdf(self):
        interp = run_matlab("p = normcdf(0);")
        assert abs(get_val(interp.global_env.get("p")) - 0.5) < 0.01

    def test_norminv(self):
        interp = run_matlab("x = norminv(0.975);")
        assert abs(get_val(interp.global_env.get("x")) - 1.96) < 0.01

    def test_chi2cdf(self):
        interp = run_matlab("p = chi2cdf(3.84, 1);")
        assert abs(get_val(interp.global_env.get("p")) - 0.95) < 0.01

    def test_tcdf(self):
        interp = run_matlab("p = tcdf(1.96, 100);")
        assert abs(get_val(interp.global_env.get("p")) - 0.975) < 0.01

    def test_normpdf(self):
        interp = run_matlab("p = normpdf(0);")
        assert abs(get_val(interp.global_env.get("p")) - 0.3989) < 0.01


class TestStatisticalFunctions:
    def test_geomean(self):
        interp = run_matlab("x = [1 2 4 8];\nm = geomean(x);")
        assert abs(get_val(interp.global_env.get("m")) - 2.828) < 0.01

    def test_harmmean(self):
        interp = run_matlab("x = [1 2 4];\nm = harmmean(x);")
        assert abs(get_val(interp.global_env.get("m")) - 1.714) < 0.01

    def test_iqr(self):
        interp = run_matlab("x = [1 2 3 4 5 6 7 8 9 10];\nr = iqr(x);")
        assert abs(get_val(interp.global_env.get("r")) - 4.5) < 0.01

    def test_cov(self):
        interp = run_matlab("x = [1 2 3];\ny = [4 5 6];\nc = cov(x, y);")
        c = interp.global_env.get("c")
        assert isinstance(c, Mat)
        assert c.shape == (2, 2)

    def test_corrcoef(self):
        interp = run_matlab("x = [1 2 3];\ny = [4 5 6];\nr = corrcoef(x, y);")
        r = interp.global_env.get("r")
        assert isinstance(r, Mat)
        assert abs(r.data[0, 1] - 1.0) < 0.01

    def test_zscore(self):
        interp = run_matlab("x = [1 2 3 4 5];\nz = zscore(x);")
        assert isinstance(interp.global_env.get("z"), Mat)


class TestStringFunctions:
    def test_strcmp(self):
        interp = run_matlab("r = strcmp('hello', 'hello');")
        assert interp.global_env.get("r") == True

    def test_strcat(self):
        interp = run_matlab("s = strcat('hello', ' ', 'world');")
        assert interp.global_env.get("s") == "hello world"

    def test_upper(self):
        interp = run_matlab("s = upper('hello');")
        assert interp.global_env.get("s") == "HELLO"

    def test_lower(self):
        interp = run_matlab("s = lower('HELLO');")
        assert interp.global_env.get("s") == "hello"

    def test_strtrim(self):
        interp = run_matlab("s = strtrim('  hello  ');")
        assert interp.global_env.get("s") == "hello"

    def test_num2str(self):
        interp = run_matlab("s = num2str(42);")
        assert "42" in str(interp.global_env.get("s"))

    def test_str2double(self):
        interp = run_matlab("x = str2double('3.14');")
        assert abs(get_val(interp.global_env.get("x")) - 3.14) < 0.01


class TestDataStructures:
    def test_cell_create(self):
        interp = run_matlab("c = cell(2, 3);")
        assert isinstance(interp.global_env.get("c"), CellArray)

    def test_struct_create(self):
        interp = run_matlab("s.name = 'test';\ns.value = 42;")
        s = interp.global_env.get("s")
        assert s.get_field("name") == "test"
        assert s.get_field("value") == 42

    def test_fieldnames(self):
        interp = run_matlab("s.name = 'test';\ns.value = 42;")
        s = interp.global_env.get("s")
        assert "name" in s.field_names()
        assert "value" in s.field_names()


class TestSparseMatrix:
    def test_speye(self):
        interp = run_matlab("A = speye(3);")
        A = interp.global_env.get("A")
        assert isinstance(A, Mat)
        assert A.shape == (3, 3)

    def test_spzeros(self):
        interp = run_matlab("A = spzeros(3, 3);")
        A = interp.global_env.get("A")
        assert isinstance(A, Mat)
        assert A.shape == (3, 3)

    def test_sparse_full(self):
        interp = run_matlab("A = speye(3);\nB = full(A);\nC = sparse(B);")
        assert isinstance(interp.global_env.get("B"), Mat)
        assert isinstance(interp.global_env.get("C"), Mat)


class TestStringArray:
    def test_string_create(self):
        from matpy.runtime.types import StringArray

        interp = run_matlab('s = string("hello");')
        assert isinstance(interp.global_env.get("s"), StringArray)


class TestNdimArray:
    def test_zeros_3d(self):
        interp = run_matlab("A = zeros(2, 3, 4);")
        A = interp.global_env.get("A")
        assert isinstance(A, Mat)
        assert A.shape == (2, 3, 4)

    def test_squeeze(self):
        interp = run_matlab("A = zeros(1, 3, 1);\nB = squeeze(A);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)
        assert B.shape == (3,)


class TestTimingFunctions:
    def test_tic_toc(self):
        interp = run_matlab("t = tic;\npause(0.01);\nelapsed = toc(t);")
        assert get_val(interp.global_env.get("elapsed")) > 0

    def test_now(self):
        interp = run_matlab("t = now;")
        assert get_val(interp.global_env.get("t")) > 0

    def test_date(self):
        interp = run_matlab("d = date;")
        d = interp.global_env.get("d")
        assert isinstance(d, str)
        assert len(d) > 0


class TestTypeFunctions:
    def test_class(self):
        interp = run_matlab("x = 42;")
        assert isinstance(interp.global_env.get("x"), (int, float))

    def test_isa(self):
        interp = run_matlab("x = 42;\nr = isa(x, 'double');")
        assert interp.global_env.get("r") == True

    def test_cast(self):
        interp = run_matlab("x = [1.5 2.5 3.5];\ny = cast(x, 'int32');")
        y = interp.global_env.get("y")
        assert isinstance(y, Mat)
        assert y.dtype == np.int32


class TestDisplayFunctions:
    def test_disp(self):
        interp = run_matlab("disp(42);")

    def test_disp_string(self):
        interp = run_matlab("disp('hello');")
