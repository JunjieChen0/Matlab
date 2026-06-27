"""Parser tests for MatPy."""

from matpy.lexer import Lexer
from matpy.parser import Parser


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

    def test_while_loop(self):
        source = "while x > 0\n  x = x - 1;\nend"
        tokens = Lexer(source).tokenize()
        program = Parser(tokens).parse()
        assert len(program.statements) == 1

    def test_switch_case(self):
        source = "switch x\n  case 1\n    y = 'one';\n  case 2\n    y = 'two';\nend"
        tokens = Lexer(source).tokenize()
        program = Parser(tokens).parse()
        assert len(program.statements) == 1

    def test_try_catch(self):
        source = "try\n  x = 1/0;\ncatch\n  x = 0;\nend"
        tokens = Lexer(source).tokenize()
        program = Parser(tokens).parse()
        assert len(program.statements) == 1

    def test_multi_return(self):
        source = (
            "function [q, r] = mydiv(a, b)\n  q = floor(a / b);\n  r = a - q * b;\nend"
        )
        tokens = Lexer(source).tokenize()
        program = Parser(tokens).parse()
        assert len(program.statements) == 1

    def test_anonymous_function(self):
        source = "f = @(x, y) x.^2 + y.^2;"
        tokens = Lexer(source).tokenize()
        program = Parser(tokens).parse()
        assert len(program.statements) == 1

    def test_command_style_grid(self):
        source = "grid on"
        tokens = Lexer(source).tokenize()
        program = Parser(tokens).parse()
        assert len(program.statements) == 1

    def test_command_style_hold(self):
        source = "hold on"
        tokens = Lexer(source).tokenize()
        program = Parser(tokens).parse()
        assert len(program.statements) == 1

    def test_command_style_multiple_args(self):
        source = "grid on off"
        tokens = Lexer(source).tokenize()
        program = Parser(tokens).parse()
        assert len(program.statements) == 1
