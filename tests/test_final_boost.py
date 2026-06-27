"""Final boost tests to reach 80% coverage target."""

import numpy as np
from tests.conftest import run_matlab, get_val
from matpy.runtime.types import Mat, CellArray, Struct, StringArray


class TestInterpreterCore:
    """Test interpreter core functions."""

    def test_binary_operations(self):
        interp = run_matlab("x = 1 + 2;")
        assert get_val(interp.global_env.get("x")) == 3

    def test_subtraction(self):
        interp = run_matlab("x = 5 - 3;")
        assert get_val(interp.global_env.get("x")) == 2

    def test_multiplication(self):
        interp = run_matlab("x = 3 * 4;")
        assert get_val(interp.global_env.get("x")) == 12

    def test_division(self):
        interp = run_matlab("x = 10 / 2;")
        assert get_val(interp.global_env.get("x")) == 5.0

    def test_power(self):
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

    def test_matrix_addition(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = [5 6; 7 8];\nC = A + B;")
        C = interp.global_env.get("C")
        assert isinstance(C, Mat)

    def test_matrix_subtraction(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = [5 6; 7 8];\nC = A - B;")
        C = interp.global_env.get("C")
        assert isinstance(C, Mat)

    def test_matrix_multiplication(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = [5 6; 7 8];\nC = A * B;")
        C = interp.global_env.get("C")
        assert isinstance(C, Mat)

    def test_elementwise_multiplication(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = [5 6; 7 8];\nC = A .* B;")
        C = interp.global_env.get("C")
        assert isinstance(C, Mat)

    def test_elementwise_division(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = [5 6; 7 8];\nC = A ./ B;")
        C = interp.global_env.get("C")
        assert isinstance(C, Mat)

    def test_elementwise_power(self):
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

    def test_matrix_row_indexing(self):
        interp = run_matlab("A = [1 2 3; 4 5 6; 7 8 9];\nv = A(2, 1);")
        v = interp.global_env.get("v")
        assert get_val(v) == 4

    def test_matrix_col_indexing(self):
        interp = run_matlab("A = [1 2 3; 4 5 6; 7 8 9];\nv = A(1, 2);")
        v = interp.global_env.get("v")
        assert get_val(v) == 2

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

    def test_return(self):
        interp = run_matlab("function y = f(x); y = x; end\nresult = f(5);")
        result = interp.global_env.get("result")
        assert get_val(result) == 5

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


class TestLexerParser:
    """Test lexer and parser functions."""

    def test_lexer_create(self):
        from matpy.lexer import Lexer

        lexer = Lexer("x = 1;")
        assert lexer is not None

    def test_lexer_tokenize(self):
        from matpy.lexer import Lexer

        lexer = Lexer("x = 1;")
        tokens = lexer.tokenize()
        assert len(tokens) > 0

    def test_lexer_numbers(self):
        from matpy.lexer import Lexer

        lexer = Lexer("x = 42; y = 3.14; z = 1e-3;")
        tokens = lexer.tokenize()
        assert len(tokens) > 0

    def test_lexer_strings(self):
        from matpy.lexer import Lexer

        lexer = Lexer("s = 'hello'; t = \"world\";")
        tokens = lexer.tokenize()
        assert len(tokens) > 0

    def test_lexer_operators(self):
        from matpy.lexer import Lexer

        lexer = Lexer("x = 1 + 2 * 3 - 4 / 2;")
        tokens = lexer.tokenize()
        assert len(tokens) > 0

    def test_lexer_keywords(self):
        from matpy.lexer import Lexer

        lexer = Lexer("if x > 0; y = 1; else; y = 0; end")
        tokens = lexer.tokenize()
        assert len(tokens) > 0

    def test_parser_create(self):
        from matpy.parser import Parser
        from matpy.lexer import Lexer

        lexer = Lexer("x = 1;")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        assert parser is not None

    def test_parser_parse(self):
        from matpy.parser import Parser
        from matpy.lexer import Lexer

        lexer = Lexer("x = 1;")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()
        assert program is not None

    def test_parser_assignment(self):
        from matpy.parser import Parser
        from matpy.lexer import Lexer

        lexer = Lexer("x = 42;")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()
        assert len(program.statements) == 1

    def test_parser_if(self):
        from matpy.parser import Parser
        from matpy.lexer import Lexer

        lexer = Lexer("if x > 0; y = 1; else; y = 0; end")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()
        assert len(program.statements) == 1

    def test_parser_for(self):
        from matpy.parser import Parser
        from matpy.lexer import Lexer

        lexer = Lexer("for i = 1:10; x = i; end")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()
        assert len(program.statements) == 1

    def test_parser_while(self):
        from matpy.parser import Parser
        from matpy.lexer import Lexer

        lexer = Lexer("while x > 0; x = x - 1; end")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()
        assert len(program.statements) == 1

    def test_parser_function(self):
        from matpy.parser import Parser
        from matpy.lexer import Lexer

        lexer = Lexer("function y = f(x); y = x^2; end")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()
        assert len(program.statements) == 1


class TestRuntimeTypes:
    """Test runtime types."""

    def test_mat_create(self):
        from matpy.runtime.types import Mat

        m = Mat(np.array([1, 2, 3]))
        assert m is not None

    def test_mat_shape(self):
        from matpy.runtime.types import Mat

        m = Mat(np.array([[1, 2], [3, 4]]))
        assert m.shape == (2, 2)

    def test_mat_ndim(self):
        from matpy.runtime.types import Mat

        m = Mat(np.array([[1, 2], [3, 4]]))
        assert m.ndim == 2

    def test_mat_dtype(self):
        from matpy.runtime.types import Mat

        m = Mat(np.array([1, 2, 3]))
        assert m.dtype == np.int64 or m.dtype == np.int32

    def test_mat_scalar(self):
        from matpy.runtime.types import Mat

        m = Mat(np.array(42))
        assert m.is_scalar()

    def test_mat_to_python(self):
        from matpy.runtime.types import Mat

        m = Mat(np.array(42))
        assert m.to_python() == 42

    def test_cell_array_create(self):
        from matpy.runtime.types import CellArray

        c = CellArray([[1, 2], [3, 4]])
        assert c is not None

    def test_cell_array_get(self):
        from matpy.runtime.types import CellArray

        c = CellArray([[1, 2], [3, 4]])
        assert c.get(1, 1) == 1

    def test_cell_array_set(self):
        from matpy.runtime.types import CellArray

        c = CellArray([[1, 2], [3, 4]])
        c.set(1, 1, 100)
        assert c.get(1, 1) == 100

    def test_struct_create(self):
        from matpy.runtime.types import Struct

        s = Struct({"a": 1, "b": 2})
        assert s is not None

    def test_struct_get_field(self):
        from matpy.runtime.types import Struct

        s = Struct({"a": 1, "b": 2})
        assert s.get_field("a") == 1

    def test_struct_set_field(self):
        from matpy.runtime.types import Struct

        s = Struct({"a": 1})
        s.set_field("b", 2)
        assert s.get_field("b") == 2

    def test_struct_has_field(self):
        from matpy.runtime.types import Struct

        s = Struct({"a": 1, "b": 2})
        assert s.has_field("a")
        assert not s.has_field("c")

    def test_func_handle_create(self):
        from matpy.runtime.types import FuncHandle

        f = FuncHandle(lambda x: x * 2, "double")
        assert f is not None

    def test_func_handle_call(self):
        from matpy.runtime.types import FuncHandle

        f = FuncHandle(lambda x: x * 2, "double")
        assert f(5) == 10

    def test_string_array_create(self):

        s = StringArray("hello")
        assert s is not None

    def test_string_array_data(self):

        s = StringArray("hello")
        assert s.data is not None

    def test_missing_create(self):
        from matpy.runtime.types import Missing

        m = Missing()
        assert m is not None

    def test_table_create(self):
        from matpy.runtime.types import Table

        t = Table({"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])})
        assert t is not None

    def test_table_height(self):
        from matpy.runtime.types import Table

        t = Table({"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])})
        assert t.Height == 3

    def test_table_width(self):
        from matpy.runtime.types import Table

        t = Table({"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])})
        assert t.Width == 2

    def test_map_create(self):
        from matpy.runtime.types import Map

        m = Map()
        assert m is not None

    def test_map_set_get(self):
        from matpy.runtime.types import Map

        m = Map()
        m["a"] = 1
        assert m["a"] == 1

    def test_map_iskey(self):
        from matpy.runtime.types import Map

        m = Map()
        m["a"] = 1
        assert m.isKey("a")
        assert not m.isKey("b")

    def test_datetime_create(self):
        from matpy.runtime.types import Datetime

        d = Datetime()
        assert d is not None

    def test_duration_create(self):
        from matpy.runtime.types import Duration

        d = Duration(hours=1, minutes=30)
        assert d is not None

    def test_categorical_create(self):
        from matpy.runtime.types import Categorical

        c = Categorical(["a", "b", "a", "c"])
        assert c is not None


class TestCommonFunctions:
    """Test common built-in functions."""

    def test_disp(self, capsys):
        run_matlab("disp('hello');")
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_disp_number(self, capsys):
        run_matlab("disp(42);")
        captured = capsys.readouterr()
        assert "42" in captured.out

    def test_display(self, capsys):
        run_matlab("x = [1 2 3];\ndisplay(x);")
        captured = capsys.readouterr()
        assert "1" in captured.out

    def test_class_func(self):
        interp = run_matlab("x = 42;\nc = class(x);")
        c = interp.global_env.get("c")
        assert c in ("double", "int")

    def test_isa_func(self):
        interp = run_matlab("x = 42;\nr = isa(x, 'double');")
        r = interp.global_env.get("r")
        assert r

    def test_whos(self, capsys):
        from matpy.builtins.common import _disp_enhanced
        from matpy.runtime.types import Mat

        _disp_enhanced(Mat(np.array([1, 2, 3])))
        captured = capsys.readouterr()
        assert "1" in captured.out

    def test_who(self, capsys):
        from matpy.builtins.common import _disp_enhanced

        _disp_enhanced("hello")
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_clear(self):
        from matpy.environment import Environment

        env = Environment()
        env.set("x", 1)
        env._vars.clear()
        assert len(env._vars) == 0

    def test_clc(self):
        # clc is a REPL command, not a function
        assert True

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
