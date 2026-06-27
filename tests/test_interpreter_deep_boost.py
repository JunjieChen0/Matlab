"""Deep boost tests for interpreter.py and other core modules."""

from tests.conftest import run_matlab, get_val
from matpy.runtime.types import Mat, CellArray, Struct


class TestInterpreterDeepBoost:
    """Deep boost interpreter coverage."""

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
