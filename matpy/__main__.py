"""CLI entry point for MatPy."""

import sys
import os

from matpy.lexer import Lexer, LexError
from matpy.parser import Parser, ParseError
from matpy.interpreter import Interpreter, InterpreterError, MatPyError


def run_source(source: str, filename: str = "<stdin>", interpreter: Interpreter | None = None, exit_on_error: bool = True) -> Interpreter:
    if interpreter is None:
        interpreter = Interpreter()

    try:
        lexer = Lexer(source, filename)
        tokens = lexer.tokenize()
        parser = Parser(tokens, filename)
        program = parser.parse()
        interpreter.run(program)
    except LexError as e:
        print(f"\n  Lexical error in {filename}: {e}", file=sys.stderr)
        _show_source_context(source, e.line, e.col)
        if exit_on_error:
            sys.exit(1)
        raise
    except ParseError as e:
        print(f"\n  Parse error in {filename}: {e}", file=sys.stderr)
        _show_source_context(source, e.token.line, e.token.col)
        if exit_on_error:
            sys.exit(1)
        raise
    except MatPyError as e:
        print(f"\n  Error in {filename}: {e}", file=sys.stderr)
        if exit_on_error:
            sys.exit(1)
        raise
    except Exception as e:
        print(f"\n  Unexpected error in {filename}: {e}", file=sys.stderr)
        if exit_on_error:
            sys.exit(1)
        raise

    return interpreter


def _show_source_context(source: str, line: int, col: int, context: int = 2):
    """Show source code context around an error."""
    lines = source.split("\n")
    start = max(0, line - context - 1)
    end = min(len(lines), line + context)

    print(f"\n  Context:", file=sys.stderr)
    for i in range(start, end):
        marker = ">>>" if i == line - 1 else "   "
        print(f"  {marker} {i+1:4d} | {lines[i]}", file=sys.stderr)
        if i == line - 1 and col > 0:
            print(f"       | {' ' * (col - 1)}^", file=sys.stderr)
    print(file=sys.stderr)


def run_file(filepath: str):
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    filename = os.path.basename(filepath)
    run_source(source, filename)


def repl():
    """Interactive REPL with history and special commands."""
    print("MatPy v0.3.0 — MATLAB interpreter in Python")
    print("Type 'quit' or 'exit' to exit, 'help' for help.\n")

    # Try to enable readline for history
    try:
        import readline
        histfile = os.path.expanduser("~/.matpy_history")
        try:
            readline.read_history_file(histfile)
        except FileNotFoundError:
            pass
        readline.set_history_length(1000)
    except ImportError:
        readline = None

    interpreter = Interpreter()
    block_buffer = []  # For multi-line input
    in_block = False

    while True:
        try:
            if in_block:
                line = input("... ")
            else:
                line = input(">> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        # Save history
        if readline:
            try:
                readline.write_history_file(os.path.expanduser("~/.matpy_history"))
            except:
                pass

        # Special commands
        stripped = line.strip()
        if not in_block:
            if stripped in ("quit", "exit"):
                print("Bye!")
                break
            elif stripped == "help":
                print("MatPy v0.3.0")
                print("Commands:")
                print("  quit/exit    — Exit REPL")
                print("  help         — Show this help")
                print("  who          — List variables")
                print("  whos         — List variables with details")
                print("  clear        — Clear all variables")
                print("  clc          — Clear screen")
                print("  demo         — Run demo script")
                print("")
                print("MATLAB syntax supported:")
                print("  x = [1 2 3; 4 5 6]  — Matrix")
                print("  for i = 1:10 ... end — Loop")
                print("  function y = f(x) ... end — Function")
                print("  plot(x, y)           — Plot")
                continue
            elif stripped == "who":
                vars_dict = interpreter.global_env.all_vars()
                names = [n for n in vars_dict if not n.startswith("_")]
                if names:
                    print("  " + "  ".join(names))
                else:
                    print("No variables defined.")
                continue
            elif stripped == "whos":
                vars_dict = interpreter.global_env.all_vars()
                if not vars_dict:
                    print("No variables defined.")
                    continue
                print(f"  {'Name':15s} {'Size':15s} {'Class':10s}")
                print(f"  {'─' * 45}")
                for name, val in vars_dict.items():
                    if name.startswith("_"):
                        continue
                    from matpy.runtime.types import Mat, Struct, CellArray
                    if isinstance(val, Mat):
                        sz = "x".join(str(d) for d in val.shape)
                        cls = "double"
                    elif isinstance(val, str):
                        sz = f"1x{len(val)}"
                        cls = "char"
                    elif isinstance(val, Struct):
                        sz = "1x1"
                        cls = "struct"
                    elif isinstance(val, CellArray):
                        sz = f"{val.shape[0]}x{val.shape[1]}"
                        cls = "cell"
                    else:
                        sz = "1x1"
                        cls = type(val).__name__
                    print(f"  {name:15s} {sz:15s} {cls:10s}")
                continue
            elif stripped == "clear":
                interpreter.global_env = __import__('matpy.environment', fromlist=['Environment']).Environment(name="global")
                print("Workspace cleared.")
                continue
            elif stripped == "clc":
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            elif stripped == "demo":
                demo_source = """
x = 0:0.1:2*pi;
y = sin(x);
figure;
plot(x, y);
xlabel('x');
ylabel('sin(x)');
title('Sine Wave');
grid('on');
saveas(gcf, 'demo_sine.png');
disp('Demo plot saved to demo_sine.png');
"""
                try:
                    lexer = Lexer(demo_source, "<demo>")
                    tokens = lexer.tokenize()
                    parser = Parser(tokens, "<demo>")
                    program = parser.parse()
                    interpreter.run(program)
                except Exception as e:
                    print(f"Demo error: {e}", file=sys.stderr)
                continue

        # Multi-line block detection
        if not in_block:
            # Check if this starts a block
            block_starters = ["function", "if", "for", "while", "switch", "try", "classdef"]
            first_word = stripped.split()[0] if stripped.split() else ""
            if first_word in block_starters:
                in_block = True
                block_buffer = [line]
                continue
        else:
            block_buffer.append(line)
            # Check if block is complete (has 'end')
            if stripped == "end":
                source = "\n".join(block_buffer)
                in_block = False
                block_buffer = []
            else:
                continue

        source = line
        while source.rstrip().endswith("..."):
            try:
                continuation = input("   ")
                source = source.rstrip()[:-3] + continuation
            except (EOFError, KeyboardInterrupt):
                break

        try:
            lexer = Lexer(source, "<repl>")
            tokens = lexer.tokenize()
            parser = Parser(tokens, "<repl>")
            program = parser.parse()
            interpreter.run(program)
        except LexError as e:
            print(f"Lexical error: {e}", file=sys.stderr)
        except ParseError as e:
            print(f"Parse error: {e}", file=sys.stderr)
        except RuntimeError as e:
            print(f"Runtime error: {e}", file=sys.stderr)
        except KeyboardInterrupt:
            print("\nInterrupted.")


def main():
    if len(sys.argv) < 2:
        repl()
        return

    arg = sys.argv[1]

    if arg in ("-h", "--help"):
        print("MatPy v0.3.0 — MATLAB interpreter in Python")
        print("")
        print("Usage:")
        print("  matpy              Start interactive REPL")
        print("  matpy file.m       Execute MATLAB file")
        print("  matpy -h/--help    Show this help")
        print("  matpy -v/--version Show version")
        return

    if arg in ("-v", "--version"):
        print("MatPy v0.3.0")
        return

    run_file(arg)


if __name__ == "__main__":
    main()
