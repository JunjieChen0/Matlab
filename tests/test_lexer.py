"""Lexer tests for MatPy."""

from matpy.lexer import Lexer


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

    def test_operators(self):
        lexer = Lexer("x = a + b * c - d / e;")
        tokens = lexer.tokenize()
        assert len(tokens) > 0

    def test_brackets(self):
        lexer = Lexer("x = [1 2; 3 4];")
        tokens = lexer.tokenize()
        assert len(tokens) > 0

    def test_ellipsis(self):
        lexer = Lexer("x = 1 + 2 + ...\n3 + 4;")
        tokens = lexer.tokenize()
        assert len(tokens) > 0

    def test_colon_operator(self):
        lexer = Lexer("x = 1:10;")
        tokens = lexer.tokenize()
        assert len(tokens) > 0
