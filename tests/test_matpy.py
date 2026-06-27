"""Unit tests for MatPy."""

import pytest
import numpy as np
from matpy.lexer import Lexer
from matpy.parser import Parser
from matpy.interpreter import Interpreter
from matpy.runtime.types import Mat, CellArray, Struct, ClassInstance


def run_matlab(source: str):
    interpreter = Interpreter()
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    program = parser.parse()
    interpreter.run(program)
    return interpreter


class TestLexer:
    def test_basic_tokens(self):
        lexer = Lexer("x = 1 + 2;")
        tokens = lexer.tokenize()
        assert len(tokens) > 0

    def test_numbers(self):
        lexer = Lexer("42 3.14 1e-3")
        tokens = lexer.tokenize()
        nums = [t for t in tokens if t.type.name == "NUMBER"]
        assert len(nums) == 3

    def test_strings(self):
        lexer = Lexer("'hello' \"world\"")
        tokens = lexer.tokenize()
        strings = [t for t in tokens if t.type.name == "STRING"]
        assert len(strings) == 2

    def test_comments(self):
        lexer = Lexer("x = 1; % this is a comment")
        tokens = lexer.tokenize()
        assert all(t.type.name != "COMMENT" for t in tokens)


class TestParser:
    def test_assignment(self):
        tokens = Lexer("x = 42;").tokenize()
        program = Parser(tokens).parse()
        assert len(program.statements) == 1

    def test_if_statement(self):
        source = "if x > 0\n  disp(x);\nelse\n  disp(-x);\nend"
        tokens = Lexer(source).tokenize()
        program = Parser(tokens).parse()
        assert len(program.statements) == 1

    def test_for_loop(self):
        source = "for i = 1:10\n  disp(i);\nend"
        tokens = Lexer(source).tokenize()
        program = Parser(tokens).parse()
        assert len(program.statements) == 1

    def test_function_def(self):
        source = "function y = square(x)\n  y = x^2;\nend"
        tokens = Lexer(source).tokenize()
        program = Parser(tokens).parse()
        assert len(program.statements) == 1

    def test_matrix_literal(self):
        tokens = Lexer("[1 2; 3 4]").tokenize()
        program = Parser(tokens).parse()
        assert len(program.statements) == 1


class TestInterpreter:
    def test_arithmetic(self):
        interp = run_matlab("x = 1 + 2 * 3;")
        x = interp.global_env.get("x")
        assert x == 7 or (hasattr(x, "data") and x.data.flat[0] == 7)

    def test_matrix_creation(self):
        interp = run_matlab("x = [1 2; 3 4];")
        x = interp.global_env.get("x")
        assert isinstance(x, Mat)
        assert x.shape == (2, 2)

    def test_matrix_indexing(self):
        interp = run_matlab("x = [1 2 3; 4 5 6];\na = x(1, 2);\nb = x(2, 3);")
        a = interp.global_env.get("a")
        b = interp.global_env.get("b")
        assert a == 2 or (hasattr(a, "data") and a.data.flat[0] == 2)
        assert b == 6 or (hasattr(b, "data") and b.data.flat[0] == 6)

    def test_for_loop(self):
        interp = run_matlab("s = 0;\nfor i = 1:10\n  s = s + i;\nend")
        s = interp.global_env.get("s")
        assert s == 55 or (hasattr(s, "data") and s.data.flat[0] == 55)

    def test_while_loop(self):
        interp = run_matlab(
            "i = 0; s = 0;\nwhile i < 10\n  i = i + 1;\n  s = s + i;\nend"
        )
        s = interp.global_env.get("s")
        assert s == 55 or (hasattr(s, "data") and s.data.flat[0] == 55)

    def test_if_else(self):
        interp = run_matlab("x = 10;\nif x > 5\n  y = 1;\nelse\n  y = 0;\nend")
        y = interp.global_env.get("y")
        assert y == 1 or (hasattr(y, "data") and y.data.flat[0] == 1)

    def test_function_call(self):
        interp = run_matlab(
            "function y = square(x)\n  y = x^2;\nend\nresult = square(5);"
        )
        result = interp.global_env.get("result")
        assert result == 25 or (hasattr(result, "data") and result.data.flat[0] == 25)

    def test_multiple_returns(self):
        interp = run_matlab(
            "function [q, r] = mydiv(a, b)\n  q = floor(a / b);\n  r = a - q * b;\nend\n[q, r] = mydiv(17, 5);"
        )
        q = interp.global_env.get("q")
        r = interp.global_env.get("r")
        assert q == 3 or (hasattr(q, "data") and q.data.flat[0] == 3)
        assert r == 2 or (hasattr(r, "data") and r.data.flat[0] == 2)

    def test_builtin_zeros(self):
        interp = run_matlab("x = zeros(3, 4);")
        x = interp.global_env.get("x")
        assert isinstance(x, Mat)
        assert x.shape == (3, 4)

    def test_matrix_operations(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = [5 6; 7 8];\nC = A + B;\nD = A * B;")
        C = interp.global_env.get("C")
        D = interp.global_env.get("D")
        assert np.allclose(C.data, np.array([[6, 8], [10, 12]]))
        assert np.allclose(D.data, np.array([[19, 22], [43, 50]]))

    def test_struct(self):
        interp = run_matlab("s.name = 'test';\ns.value = 42;")
        s = interp.global_env.get("s")
        assert isinstance(s, Struct)
        assert s.get_field("name") == "test"
        assert s.get_field("value") == 42

    def test_function_handle(self):
        interp = run_matlab("f = @sin;\nresult = f(0);")
        result = interp.global_env.get("result")
        assert result == 0.0 or (hasattr(result, "data") and result.data.flat[0] == 0.0)

    def test_anonymous_function(self):
        interp = run_matlab("g = @(x, y) x.^2 + y.^2;\nresult = g(3, 4);")
        result = interp.global_env.get("result")
        assert result == 25 or (hasattr(result, "data") and result.data.flat[0] == 25)

    def test_range_expression(self):
        interp = run_matlab("x = 1:10;")
        x = interp.global_env.get("x")
        assert isinstance(x, Mat)
        assert len(x.data) == 10

    def test_switch_case(self):
        interp = run_matlab(
            "x = 2;\nswitch x\n  case 1\n    y = 'one';\n  case 2\n    y = 'two';\n  otherwise\n    y = 'other';\nend"
        )
        y = interp.global_env.get("y")
        assert y == "two"


class TestBuiltinFunctions:
    def test_math_functions(self):
        interp = run_matlab(
            "a = abs(-5);\nb = sqrt(16);\nc = round(3.7);\nd = ceil(3.2);\ne = floor(3.8);"
        )

        def get_val(x):
            if hasattr(x, "data"):
                return x.data.flat[0]
            return x

        assert get_val(interp.global_env.get("a")) == 5
        assert get_val(interp.global_env.get("b")) == 4.0
        assert get_val(interp.global_env.get("c")) == 4.0
        assert get_val(interp.global_env.get("d")) == 4.0
        assert get_val(interp.global_env.get("e")) == 3.0

    def test_matrix_functions(self):
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


class TestLogicalIndexing:
    def test_logical_indexing_read(self):
        interp = run_matlab("A = [1 2 3 4 5];\nB = A(A > 3);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)
        assert np.allclose(B.data, np.array([4, 5]))

    def test_logical_indexing_write(self):
        interp = run_matlab("A = [1 2 3 4 5];\nA(A > 3) = 0;")
        A = interp.global_env.get("A")
        assert isinstance(A, Mat)
        assert np.allclose(A.data, np.array([1, 2, 3, 0, 0]))

    def test_compound_logical(self):
        interp = run_matlab("A = [1 2 3 4 5 6];\nB = A(A > 2 & A < 5);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)
        assert np.allclose(B.data, np.array([3, 4]))


class TestClassInheritance:
    def test_basic_inheritance(self):
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
        result = interp.global_env.get("result")
        assert result == 3 or (hasattr(result, "data") and result.data.flat[0] == 3)


class TestEvalFeval:
    def test_eval(self):
        interp = run_matlab("x = 10;\neval('y = x * 2;');")
        y = interp.global_env.get("y")
        assert y == 20 or (hasattr(y, "data") and y.data.flat[0] == 20)

    def test_feval_handle(self):
        interp = run_matlab("f = @sin;\nresult = feval(f, 0);")
        result = interp.global_env.get("result")
        assert result == 0.0 or (hasattr(result, "data") and result.data.flat[0] == 0.0)

    def test_feval_name(self):
        interp = run_matlab("result = feval('cos', 0);")
        result = interp.global_env.get("result")
        assert result == 1.0 or (hasattr(result, "data") and result.data.flat[0] == 1.0)


class TestAdvancedMath:
    def test_integral(self):
        interp = run_matlab("result = integral(@(x) x^2, 0, 1);")
        result = interp.global_env.get("result")
        assert abs(result - 1 / 3) < 0.01

    def test_mean(self):
        interp = run_matlab("x = [1 2 3 4 5];\nm = mean(x);")
        m = interp.global_env.get("m")
        assert m == 3.0 or (hasattr(m, "data") and m.data.flat[0] == 3.0)

    def test_std(self):
        interp = run_matlab("x = [1 2 3 4 5];\ns = std(x);")
        s = interp.global_env.get("s")
        assert abs(s - 1.5811) < 0.01


class TestSparseMatrix:
    def test_sparse_create(self):
        interp = run_matlab("A = speye(3);")
        A = interp.global_env.get("A")
        assert isinstance(A, Mat)
        assert A.shape == (3, 3)

    def test_sparse_full(self):
        interp = run_matlab("A = speye(3);\nB = full(A);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)
        assert np.allclose(B.data, np.eye(3))


class TestStringArray:
    def test_string_create(self):
        interp = run_matlab('s = string("hello");')
        s = interp.global_env.get("s")
        from matpy.runtime.types import StringArray

        assert isinstance(s, StringArray)


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


class TestDistributionFunctions:
    def test_normcdf(self):
        interp = run_matlab("p = normcdf(0);")
        p = interp.global_env.get("p")
        val = p.data.flat[0] if isinstance(p, Mat) else p
        assert abs(val - 0.5) < 0.01

    def test_norminv(self):
        interp = run_matlab("x = norminv(0.975);")
        x = interp.global_env.get("x")
        val = x.data.flat[0] if isinstance(x, Mat) else x
        assert abs(val - 1.96) < 0.01

    def test_chi2cdf(self):
        interp = run_matlab("p = chi2cdf(3.84, 1);")
        p = interp.global_env.get("p")
        val = p.data.flat[0] if isinstance(p, Mat) else p
        assert abs(val - 0.95) < 0.01

    def test_tcdf(self):
        interp = run_matlab("p = tcdf(1.96, 100);")
        p = interp.global_env.get("p")
        val = p.data.flat[0] if isinstance(p, Mat) else p
        assert abs(val - 0.975) < 0.01

    def test_normpdf(self):
        interp = run_matlab("p = normpdf(0);")
        p = interp.global_env.get("p")
        val = p.data.flat[0] if isinstance(p, Mat) else p
        assert abs(val - 0.3989) < 0.01


class TestStatisticalFunctions:
    def test_geomean(self):
        interp = run_matlab("x = [1 2 4 8];\nm = geomean(x);")
        m = interp.global_env.get("m")
        val = m.data.flat[0] if isinstance(m, Mat) else m
        assert abs(val - 2.828) < 0.01

    def test_harmmean(self):
        interp = run_matlab("x = [1 2 4];\nm = harmmean(x);")
        m = interp.global_env.get("m")
        val = m.data.flat[0] if isinstance(m, Mat) else m
        assert abs(val - 1.714) < 0.01

    def test_iqr(self):
        interp = run_matlab("x = [1 2 3 4 5 6 7 8 9 10];\nr = iqr(x);")
        r = interp.global_env.get("r")
        val = r.data.flat[0] if isinstance(r, Mat) else r
        assert abs(val - 4.5) < 0.01

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
        z = interp.global_env.get("z")
        assert isinstance(z, Mat)


class TestMatrixOperations:
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

    def test_eig(self):
        interp = run_matlab("A = [1 0; 0 2];\n[V, D] = eig(A);")
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

    def test_inv(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = inv(A);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)
        assert abs(B.data[0, 0] - (-2.0)) < 0.01

    def test_det(self):
        interp = run_matlab("A = [1 2; 3 4];\nd = det(A);")
        d = interp.global_env.get("d")
        assert abs(d - (-2.0)) < 0.01

    def test_rank(self):
        interp = run_matlab("A = [1 2; 3 4];\nr = rank(A);")
        r = interp.global_env.get("r")
        assert r == 2

    def test_norm(self):
        interp = run_matlab("A = [1 2; 3 4];\nn = norm(A);")
        n = interp.global_env.get("n")
        assert n > 0

    def test_cond(self):
        interp = run_matlab("A = [1 2; 3 4];\nc = cond(A);")
        c = interp.global_env.get("c")
        assert c > 1

    def test_trace(self):
        interp = run_matlab("A = [1 2; 3 4];\nt = sum(diag(A));")
        t = interp.global_env.get("t")
        val = t.data.flat[0] if isinstance(t, Mat) else t
        assert val == 5


class TestStringOperations:
    def test_strcmp(self):
        interp = run_matlab("r = strcmp('hello', 'hello');")
        r = interp.global_env.get("r")
        assert r

    def test_strcat(self):
        interp = run_matlab("s = strcat('hello', ' ', 'world');")
        s = interp.global_env.get("s")
        assert s == "hello world"

    def test_upper(self):
        interp = run_matlab("s = upper('hello');")
        s = interp.global_env.get("s")
        assert s == "HELLO"

    def test_lower(self):
        interp = run_matlab("s = lower('HELLO');")
        s = interp.global_env.get("s")
        assert s == "hello"

    def test_strtrim(self):
        interp = run_matlab("s = strtrim('  hello  ');")
        s = interp.global_env.get("s")
        assert s == "hello"

    def test_num2str(self):
        interp = run_matlab("s = num2str(42);")
        s = interp.global_env.get("s")
        assert "42" in str(s)

    def test_str2double(self):
        interp = run_matlab("x = str2double('3.14');")
        x = interp.global_env.get("x")
        assert abs(x - 3.14) < 0.01


class TestCellOperations:
    def test_cell_create(self):
        interp = run_matlab("c = cell(2, 3);")
        c = interp.global_env.get("c")
        assert isinstance(c, CellArray)

    def test_struct_create(self):
        interp = run_matlab("s.name = 'test';\ns.value = 42;")
        s = interp.global_env.get("s")
        assert isinstance(s, Struct)
        assert s.get_field("name") == "test"
        assert s.get_field("value") == 42

    def test_fieldnames(self):
        interp = run_matlab("s.name = 'test';\ns.value = 42;")
        s = interp.global_env.get("s")
        assert isinstance(s, Struct)
        assert "name" in s.field_names()
        assert "value" in s.field_names()


class TestSparseOperations:
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
        B = interp.global_env.get("B")
        C = interp.global_env.get("C")
        assert isinstance(B, Mat)
        assert isinstance(C, Mat)


class TestTimingFunctions:
    def test_tic_toc(self):
        interp = run_matlab("t = tic;\npause(0.01);\nelapsed = toc(t);")
        elapsed = interp.global_env.get("elapsed")
        assert elapsed > 0

    def test_now(self):
        interp = run_matlab("t = now;")
        t = interp.global_env.get("t")
        assert t > 0

    def test_date(self):
        interp = run_matlab("d = date;")
        d = interp.global_env.get("d")
        assert isinstance(d, str)
        assert len(d) > 0


class TestTypeFunctions:
    def test_class(self):
        interp = run_matlab("x = 42;")
        x = interp.global_env.get("x")
        assert isinstance(x, (int, float))

    def test_isa(self):
        interp = run_matlab("x = 42;\nr = isa(x, 'double');")
        r = interp.global_env.get("r")
        assert r

    def test_cast(self):
        interp = run_matlab("x = [1.5 2.5 3.5];\ny = cast(x, 'int32');")
        y = interp.global_env.get("y")
        assert isinstance(y, Mat)
        assert y.dtype == np.int32


class TestDisplayFunctions:
    def test_disp(self):
        run_matlab("disp(42);")
        # Just check it doesn't error

    def test_disp_string(self):
        run_matlab("disp('hello');")
        # Just check it doesn't error


class TestEdgeCases:
    def test_empty_matrix(self):
        interp = run_matlab("x = [];")
        x = interp.global_env.get("x")
        assert isinstance(x, Mat)

    def test_scalar(self):
        interp = run_matlab("x = 42;")
        x = interp.global_env.get("x")
        assert x == 42

    def test_complex(self):
        interp = run_matlab("x = 1 + 2i;")
        x = interp.global_env.get("x")
        assert isinstance(x, complex)

    def test_string(self):
        interp = run_matlab("s = 'hello';")
        s = interp.global_env.get("s")
        assert s == "hello"

    def test_nested_expressions(self):
        interp = run_matlab("x = (1 + 2) * (3 + 4);")
        x = interp.global_env.get("x")
        assert x == 21

    def test_multiple_statements(self):
        interp = run_matlab("x = 1; y = 2; z = x + y;")
        z = interp.global_env.get("z")
        assert z == 3

    def test_function_recursive(self):
        interp = run_matlab(
            "function r = fact(n)\nif n <= 1\n  r = 1;\nelse\n  r = n * fact(n-1);\nend\nend\nx = fact(5);"
        )
        x = interp.global_env.get("x")
        assert x == 120


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
