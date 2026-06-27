"""Interpreter tests for MatPy."""

import pytest
import numpy as np
from tests.conftest import run_matlab, get_val
from matpy.runtime.types import Mat, Struct, ClassInstance


class TestArithmetic:
    def test_basic(self):
        interp = run_matlab("x = 1 + 2 * 3;")
        assert get_val(interp.global_env.get("x")) == 7

    def test_precedence(self):
        interp = run_matlab("x = (1 + 2) * 3;")
        assert get_val(interp.global_env.get("x")) == 9

    def test_negative(self):
        interp = run_matlab("x = -5;")
        assert get_val(interp.global_env.get("x")) == -5


class TestMatrixCreation:
    def test_2d(self):
        interp = run_matlab("x = [1 2; 3 4];")
        x = interp.global_env.get("x")
        assert isinstance(x, Mat)
        assert x.shape == (2, 2)

    def test_zeros(self):
        interp = run_matlab("x = zeros(3, 4);")
        x = interp.global_env.get("x")
        assert isinstance(x, Mat)
        assert x.shape == (3, 4)

    def test_ones(self):
        interp = run_matlab("x = ones(2, 3);")
        x = interp.global_env.get("x")
        assert isinstance(x, Mat)
        assert x.shape == (2, 3)

    def test_eye(self):
        interp = run_matlab("x = eye(3);")
        x = interp.global_env.get("x")
        assert isinstance(x, Mat)
        assert x.shape == (3, 3)

    def test_range(self):
        interp = run_matlab("x = 1:10;")
        x = interp.global_env.get("x")
        assert isinstance(x, Mat)
        assert len(x.data) == 10


class TestMatrixIndexing:
    def test_basic(self):
        interp = run_matlab("x = [1 2 3; 4 5 6];\na = x(1, 2);\nb = x(2, 3);")
        assert get_val(interp.global_env.get("a")) == 2
        assert get_val(interp.global_env.get("b")) == 6

    def test_logical(self):
        interp = run_matlab("A = [1 2 3 4 5];\nB = A(A > 3);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)
        assert np.allclose(B.data, np.array([4, 5]))

    def test_logical_write(self):
        interp = run_matlab("A = [1 2 3 4 5];\nA(A > 3) = 0;")
        A = interp.global_env.get("A")
        assert isinstance(A, Mat)
        assert np.allclose(A.data, np.array([1, 2, 3, 0, 0]))

    def test_compound_logical(self):
        interp = run_matlab("A = [1 2 3 4 5 6];\nB = A(A > 2 & A < 5);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)
        assert np.allclose(B.data, np.array([3, 4]))


class TestControlFlow:
    def test_for_loop(self):
        interp = run_matlab("s = 0;\nfor i = 1:10\n  s = s + i;\nend")
        assert get_val(interp.global_env.get("s")) == 55

    def test_while_loop(self):
        interp = run_matlab(
            "i = 0; s = 0;\nwhile i < 10\n  i = i + 1;\n  s = s + i;\nend"
        )
        assert get_val(interp.global_env.get("s")) == 55

    def test_if_else(self):
        interp = run_matlab("x = 10;\nif x > 5\n  y = 1;\nelse\n  y = 0;\nend")
        assert get_val(interp.global_env.get("y")) == 1

    def test_switch_case(self):
        interp = run_matlab(
            "x = 2;\nswitch x\n  case 1\n    y = 'one';\n  case 2\n    y = 'two';\n  otherwise\n    y = 'other';\nend"
        )
        assert interp.global_env.get("y") == "two"


class TestFunctions:
    def test_basic(self):
        interp = run_matlab(
            "function y = square(x)\n  y = x^2;\nend\nresult = square(5);"
        )
        assert get_val(interp.global_env.get("result")) == 25

    def test_multiple_returns(self):
        interp = run_matlab(
            "function [q, r] = mydiv(a, b)\n  q = floor(a / b);\n  r = a - q * b;\nend\n[q, r] = mydiv(17, 5);"
        )
        assert get_val(interp.global_env.get("q")) == 3
        assert get_val(interp.global_env.get("r")) == 2

    def test_recursive(self):
        interp = run_matlab(
            "function r = fact(n)\nif n <= 1\n  r = 1;\nelse\n  r = n * fact(n-1);\nend\nend\nx = fact(5);"
        )
        assert get_val(interp.global_env.get("x")) == 120

    def test_handle(self):
        interp = run_matlab("f = @sin;\nresult = f(0);")
        assert get_val(interp.global_env.get("result")) == 0.0

    def test_anonymous(self):
        interp = run_matlab("g = @(x, y) x.^2 + y.^2;\nresult = g(3, 4);")
        assert get_val(interp.global_env.get("result")) == 25

    def test_eval(self):
        interp = run_matlab("x = 10;\neval('y = x * 2;');")
        assert get_val(interp.global_env.get("y")) == 20

    def test_feval_handle(self):
        interp = run_matlab("f = @sin;\nresult = feval(f, 0);")
        assert get_val(interp.global_env.get("result")) == 0.0

    def test_feval_name(self):
        interp = run_matlab("result = feval('cos', 0);")
        assert get_val(interp.global_env.get("result")) == 1.0


class TestStruct:
    def test_field_access(self):
        interp = run_matlab("s.name = 'test';\ns.value = 42;")
        s = interp.global_env.get("s")
        assert isinstance(s, Struct)
        assert s.get_field("name") == "test"
        assert s.get_field("value") == 42


class TestClassInheritance:
    def test_basic(self):
        source = """
        classdef Base
            properties
                x = 0
            end
            methods
                function obj = Base(v)
                    obj.x = v;
                end
            end
        end

        classdef Child < Base
            properties
                y = 0
            end
            methods
                function obj = Child(a, b)
                    obj.x = a;
                    obj.y = b;
                end
            end
        end

        c = Child(3, 4);
        """
        interp = run_matlab(source)
        c = interp.global_env.get("c")
        assert isinstance(c, ClassInstance)
        assert c.get_property("x") == 3
        assert c.get_property("y") == 4

    def test_inherited_method(self):
        source = """
        classdef Base
            properties
                x = 0
            end
            methods
                function obj = Base(v)
                    obj.x = v;
                end
                function d = get_x(obj)
                    d = obj.x;
                end
            end
        end

        classdef Child < Base
            properties
                y = 0
            end
            methods
                function obj = Child(a, b)
                    obj.x = a;
                    obj.y = b;
                end
            end
        end

        c = Child(3, 4);
        result = c.get_x();
        """
        interp = run_matlab(source)
        assert get_val(interp.global_env.get("result")) == 3


class TestEdgeCases:
    def test_empty_matrix(self):
        interp = run_matlab("x = [];")
        x = interp.global_env.get("x")
        assert isinstance(x, Mat)

    def test_scalar(self):
        interp = run_matlab("x = 42;")
        assert interp.global_env.get("x") == 42

    def test_complex(self):
        interp = run_matlab("x = 1 + 2i;")
        assert isinstance(interp.global_env.get("x"), complex)

    def test_string(self):
        interp = run_matlab("s = 'hello';")
        assert interp.global_env.get("s") == "hello"

    def test_nested_expressions(self):
        interp = run_matlab("x = (1 + 2) * (3 + 4);")
        assert interp.global_env.get("x") == 21

    def test_multiple_statements(self):
        interp = run_matlab("x = 1; y = 2; z = x + y;")
        assert interp.global_env.get("z") == 3


class TestNdimArray:
    def test_3d_creation(self):
        interp = run_matlab("A = zeros(2, 3, 4);")
        A = interp.global_env.get("A")
        assert isinstance(A, Mat)
        assert A.shape == (2, 3, 4)

    def test_3d_indexing(self):
        interp = run_matlab("A = zeros(2, 3, 4);\nA(1, 2, 3) = 42;\nv = A(1, 2, 3);")
        assert get_val(interp.global_env.get("v")) == 42

    def test_3d_for_loop(self):
        """Test that for loop over 3D array iterates along first dimension."""
        interp = run_matlab("""
            A = zeros(2, 3, 4);
            for i = 1:2
                for j = 1:3
                    for k = 1:4
                        A(i, j, k) = i * 100 + j * 10 + k;
                    end
                end
            end
            s = 0;
            for col = A
                s = s + sum(sum(col));
            end
        """)
        # Sum of all elements should be correct
        s = interp.global_env.get("s")
        assert get_val(s) > 0

    def test_3d_permute(self):
        interp = run_matlab("A = zeros(2, 3, 4);\nB = permute(A, [3 1 2]);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)
        assert B.shape == (4, 2, 3)

    def test_3d_squeeze(self):
        interp = run_matlab("A = zeros(1, 3, 1);\nB = squeeze(A);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)
        assert B.shape == (3,)

    def test_4d_creation(self):
        interp = run_matlab("A = zeros(2, 3, 4, 5);")
        A = interp.global_env.get("A")
        assert isinstance(A, Mat)
        assert A.shape == (2, 3, 4, 5)


class TestTryCatch:
    def test_basic(self):
        interp = run_matlab("""
            try
                x = 1/0;
            catch e
                msg = e.message;
            end
        """)
        msg = interp.global_env.get("msg")
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_identifier(self):
        interp = run_matlab("""
            try
                x = 1/0;
            catch e
                id = e.identifier;
            end
        """)
        id_val = interp.global_env.get("id")
        assert isinstance(id_val, str)

    def test_no_catch_var(self):
        interp = run_matlab("""
            x = 0;
            try
                x = 1/0;
            catch
                x = 42;
            end
        """)
        assert get_val(interp.global_env.get("x")) == 42

    def test_no_error(self):
        interp = run_matlab("""
            x = 0;
            try
                x = 42;
            catch e
                x = 0;
            end
        """)
        assert get_val(interp.global_env.get("x")) == 42
