"""Shared fixtures for MatPy tests."""

import pytest
from matpy.lexer import Lexer
from matpy.parser import Parser
from matpy.interpreter import Interpreter


def run_matlab(source: str) -> Interpreter:
    """Execute MATLAB source code and return the interpreter."""
    interpreter = Interpreter()
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    program = parser.parse()
    interpreter.run(program)
    return interpreter


@pytest.fixture
def interp():
    """Create a clean Interpreter instance."""
    return Interpreter()


@pytest.fixture
def run():
    """Execute MATLAB code and return the interpreter."""

    def _run(source: str) -> Interpreter:
        return run_matlab(source)

    return _run


def get_val(x):
    """Extract scalar value from Mat or raw number."""
    if hasattr(x, "data"):
        data = x.data
        if hasattr(data, "flat"):
            return data.flat[0]
        # memoryview or other buffer types
        import numpy as np

        return np.array(data).flat[0]
    return x
