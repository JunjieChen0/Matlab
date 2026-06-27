"""Bytecode compiler and VM tests for MatPy."""

from matpy.lexer import Lexer
from matpy.parser import Parser
from matpy.bytecode import BytecodeCompiler, BytecodeVM
from matpy.runtime.types import Mat


def run_bytecode(source: str):
    """Compile and execute MATLAB source via bytecode VM."""
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    compiler = BytecodeCompiler()
    instructions = compiler.compile(program)
    vm = BytecodeVM()
    vm.functions = compiler.functions
    return vm.run(instructions, compiler.constants), vm


class TestBasicArithmetic:
    def test_add(self):
        result, _ = run_bytecode("x = 1 + 2;")
        assert result == 3 or (hasattr(result, "data") and result.data.flat[0] == 3)

    def test_sub(self):
        result, _ = run_bytecode("x = 5 - 3;")
        assert result == 2 or (hasattr(result, "data") and result.data.flat[0] == 2)

    def test_mul(self):
        result, _ = run_bytecode("x = 3 * 4;")
        assert result == 12 or (hasattr(result, "data") and result.data.flat[0] == 12)

    def test_div(self):
        result, _ = run_bytecode("x = 10 / 2;")
        assert result == 5.0 or (hasattr(result, "data") and result.data.flat[0] == 5.0)

    def test_precedence(self):
        result, _ = run_bytecode("x = 1 + 2 * 3;")
        assert result == 7 or (hasattr(result, "data") and result.data.flat[0] == 7)


class TestVariables:
    def test_store_load(self):
        result, vm = run_bytecode("x = 42;")
        assert vm.variables.get("x") == 42

    def test_variable_use(self):
        result, vm = run_bytecode("x = 10; y = x + 5;")
        assert vm.variables.get("y") == 15


class TestControlFlow:
    def test_if_true(self):
        result, vm = run_bytecode("x = 1; if x > 0; y = 1; else; y = 0; end")
        assert vm.variables.get("y") == 1

    def test_if_false(self):
        result, vm = run_bytecode("x = -1; if x > 0; y = 1; else; y = 0; end")
        assert vm.variables.get("y") == 0

    def test_for_loop(self):
        result, vm = run_bytecode("s = 0; for i = 1:10; s = s + i; end")
        assert vm.variables.get("s") == 55

    def test_while_loop(self):
        result, vm = run_bytecode(
            "i = 0; s = 0; while i < 10; i = i + 1; s = s + i; end"
        )
        assert vm.variables.get("s") == 55


class TestMatrix:
    def test_matrix_creation(self):
        result, vm = run_bytecode("A = [1 2; 3 4];")
        A = vm.variables.get("A")
        assert isinstance(A, Mat)
        assert A.shape == (2, 2)

    def test_range(self):
        result, vm = run_bytecode("x = 1:5;")
        x = vm.variables.get("x")
        assert isinstance(x, Mat)


class TestBuiltins:
    def test_zeros(self):
        result, vm = run_bytecode("A = zeros(3, 3);")
        A = vm.variables.get("A")
        assert isinstance(A, Mat)
        assert A.shape == (3, 3)

    def test_disp(self):
        result, _ = run_bytecode("disp(42);")
        # Just verify no error


class TestUserFunctions:
    def test_simple_function(self):
        source = """
        function y = square(x)
            y = x^2;
        end
        result = square(5);
        """
        result, vm = run_bytecode(source)
        assert vm.variables.get("result") == 25 or (
            hasattr(vm.variables.get("result"), "data")
            and vm.variables.get("result").data.flat[0] == 25
        )

    def test_function_with_multiple_ops(self):
        source = """
        function y = add_mul(a, b, c)
            y = (a + b) * c;
        end
        result = add_mul(2, 3, 4);
        """
        result, vm = run_bytecode(source)
        assert vm.variables.get("result") == 20 or (
            hasattr(vm.variables.get("result"), "data")
            and vm.variables.get("result").data.flat[0] == 20
        )
