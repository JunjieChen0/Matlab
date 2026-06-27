"""Performance benchmark for MatPy: tree-walking vs bytecode VM."""

import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matpy.lexer import Lexer
from matpy.parser import Parser
from matpy.interpreter import Interpreter
from matpy.bytecode import BytecodeCompiler, BytecodeVM


BENCHMARKS = {
    "fibonacci_100": """
        function f = fib(n)
            f = zeros(1, n);
            f(1) = 1;
            f(2) = 1;
            for i = 3:n
                f(i) = f(i-1) + f(i-2);
            end
        end
        result = fib(100);
    """,
    "loop_sum": """
        s = 0;
        for i = 1:1000
            s = s + i;
        end
    """,
    "matrix_ops": """
        A = [1 2; 3 4];
        B = [5 6; 7 8];
        for i = 1:100
            C = A * B;
            D = A + B;
        end
    """,
    "nested_loop": """
        s = 0;
        for i = 1:100
            for j = 1:100
                s = s + i * j;
            end
        end
    """,
}


def benchmark_tree(source: str, iterations: int = 5) -> float:
    """Benchmark tree-walking interpreter."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        interpreter = Interpreter()
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()
        interpreter.run(program)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return min(times)


def benchmark_bytecode(source: str, iterations: int = 5) -> float:
    """Benchmark bytecode VM."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()
        compiler = BytecodeCompiler()
        instructions = compiler.compile(program)
        vm = BytecodeVM()
        vm.functions = compiler.functions
        vm.run(instructions, compiler.constants)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return min(times)


def main():
    print("=" * 60)
    print("MatPy Performance Benchmark")
    print("=" * 60)
    print(f"{'Benchmark':<20} {'Tree (ms)':<12} {'Bytecode (ms)':<14} {'Speedup':<10}")
    print("-" * 60)

    for name, source in BENCHMARKS.items():
        tree_time = benchmark_tree(source) * 1000
        bytecode_time = benchmark_bytecode(source) * 1000
        speedup = tree_time / bytecode_time if bytecode_time > 0 else float("inf")
        print(f"{name:<20} {tree_time:<12.2f} {bytecode_time:<14.2f} {speedup:<10.2f}x")

    print("=" * 60)


if __name__ == "__main__":
    main()
