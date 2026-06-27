"""CLI entry point for MatPy."""

import sys
import os

from matpy.lexer import Lexer, LexError
from matpy.parser import Parser, ParseError
from matpy.interpreter import Interpreter, InterpreterError, MatPyError


def run_source(
    source: str,
    filename: str = "<stdin>",
    interpreter: Interpreter | None = None,
    exit_on_error: bool = True,
    engine: str = "tree",
) -> Interpreter:
    if interpreter is None:
        interpreter = Interpreter()

    try:
        lexer = Lexer(source, filename)
        tokens = lexer.tokenize()
        parser = Parser(tokens, filename)
        program = parser.parse()

        if engine == "bytecode":
            from matpy.bytecode import BytecodeCompiler, BytecodeVM

            compiler = BytecodeCompiler()
            instructions = compiler.compile(program)
            vm = BytecodeVM()
            vm.functions = compiler.functions
            vm.func_bytecode = compiler.func_bytecode
            vm.func_params = compiler.func_params
            vm.run(instructions, compiler.constants)
            return interpreter
        else:
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
        print(f"  {marker} {i + 1:4d} | {lines[i]}", file=sys.stderr)
        if i == line - 1 and col > 0:
            print(f"       | {' ' * (col - 1)}^", file=sys.stderr)
    print(file=sys.stderr)


def run_file(filepath: str, engine: str = "tree"):
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    filename = os.path.basename(filepath)
    run_source(source, filename, engine=engine)


def repl():
    """Interactive REPL with history and special commands."""
    print("MatPy v0.5.0 — MATLAB interpreter in Python")
    print("Type 'quit' or 'exit' to exit, 'help' for help.\n")

    # Try to enable readline for history and tab completion
    try:
        import readline

        histfile = os.path.expanduser("~/.matpy_history")
        try:
            readline.read_history_file(histfile)
        except FileNotFoundError:
            pass
        readline.set_history_length(1000)

        # Tab completion for variables and functions
        def completer(text, state):
            """Tab completer for MatPy REPL."""
            options = []
            # Add variable names
            try:
                vars_dict = interpreter.global_env.all_vars()
                options.extend(
                    [
                        n
                        for n in vars_dict
                        if n.startswith(text) and not n.startswith("_")
                    ]
                )
            except Exception:
                pass
            # Add built-in function names
            try:
                from matpy.builtins import all_builtins

                builtins_dict = all_builtins()
                options.extend([n for n in builtins_dict if n.startswith(text)])
            except Exception:
                pass
            # Add keywords
            keywords = [
                "function",
                "end",
                "if",
                "else",
                "elseif",
                "for",
                "while",
                "switch",
                "case",
                "otherwise",
                "try",
                "catch",
                "return",
                "break",
                "continue",
                "global",
                "persistent",
                "classdef",
                "properties",
                "methods",
                "events",
                "arguments",
                "true",
                "false",
                "pi",
                "inf",
                "nan",
                "eps",
            ]
            options.extend([k for k in keywords if k.startswith(text)])
            # Deduplicate while preserving order
            seen = set()
            unique = []
            for o in options:
                if o not in seen:
                    seen.add(o)
                    unique.append(o)
            return unique[state] if state < len(unique) else None

        readline.set_completer(completer)
        readline.parse_and_bind("tab: complete")
    except ImportError:
        readline = None

    interpreter = Interpreter()
    block_buffer = []  # For multi-line input
    in_block = False

    # Debugging state
    breakpoints = {}  # {filename: set(line_numbers)}
    debug_mode = False
    debug_step = False

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
            # %time magic command
            elif stripped.startswith("%time "):
                code = stripped[6:]
                import time

                start = time.perf_counter()
                try:
                    lexer = Lexer(code, "<repl>")
                    tokens = lexer.tokenize()
                    parser = Parser(tokens, "<repl>")
                    program = parser.parse()
                    interpreter.run(program)
                except Exception as e:
                    print(f"Error: {e}", file=sys.stderr)
                elapsed = time.perf_counter() - start
                print(f"Elapsed time: {elapsed:.6f} seconds")
                continue
            # %who magic command
            elif stripped == "%who":
                vars_dict = interpreter.global_env.all_vars()
                names = [n for n in vars_dict if not n.startswith("_")]
                if names:
                    print("  " + "  ".join(names))
                continue
            # %whos magic command
            elif stripped == "%whos":
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
            # %clear magic command
            elif stripped == "%clear":
                interpreter.global_env = __import__(
                    "matpy.environment", fromlist=["Environment"]
                ).Environment(name="global")
                print("Workspace cleared.")
                continue
            # %doc magic command — show function documentation
            elif stripped.startswith("%doc "):
                func_name = stripped[5:].strip()
                from matpy.builtins import get_builtin

                builtin = get_builtin(func_name)
                if builtin:
                    if builtin.__doc__:
                        print(builtin.__doc__)
                    else:
                        print(f"'{func_name}' has no documentation.")
                else:
                    # Check user-defined functions
                    if func_name in interpreter.functions:
                        fdef = interpreter.functions[func_name]
                        params = ", ".join(fdef.params)
                        rets = ", ".join(fdef.returns) if fdef.returns else ""
                        sig = (
                            f"function {rets} = {func_name}({params})"
                            if rets
                            else f"function {func_name}({params})"
                        )
                        print(sig)
                        print(f"  Defined at line {fdef.line}")
                    else:
                        print(f"'{func_name}' not found.")
                continue
            # %lookfor magic command — search functions by keyword
            elif stripped.startswith("%lookfor "):
                keyword = stripped[9:].strip().lower()
                from matpy.builtins import all_builtins

                matches = []
                for name, func in all_builtins().items():
                    doc = func.__doc__ or ""
                    if keyword in name.lower() or keyword in doc.lower():
                        summary = doc.strip().split("\n")[0][:60] if doc else ""
                        matches.append((name, summary))
                if matches:
                    print(f"  {'Function':20s} {'Description'}")
                    print(f"  {'─' * 60}")
                    for name, summary in sorted(matches):
                        print(f"  {name:20s} {summary}")
                else:
                    print(f"No functions found matching '{keyword}'.")
                continue
            elif stripped == "help" or stripped.startswith("help "):
                parts = stripped.split()
                if len(parts) > 1:
                    # Help for a specific function
                    func_name = parts[1]
                    from matpy.builtins import get_builtin

                    builtin = get_builtin(func_name)
                    if builtin and builtin.__doc__:
                        print(builtin.__doc__)
                    else:
                        print(f"No help found for '{func_name}'")
                else:
                    print("MatPy v0.5.0")
                    print("Commands:")
                    print("  quit/exit    — Exit REPL")
                    print("  help         — Show this help")
                    print("  help func    — Show help for function")
                    print("  who          — List variables")
                    print("  whos         — List variables with details")
                    print("  clear        — Clear all variables")
                    print("  clc          — Clear screen")
                    print("  demo         — Run demo script")
                    print("")
                    print("Magic commands:")
                    print("  %time code    — Time code execution")
                    print("  %who          — List variables")
                    print("  %whos         — List variables with details")
                    print("  %clear        — Clear workspace")
                    print("  %doc func     — Show function documentation")
                    print("  %lookfor kw   — Search functions by keyword")
                    print("")
                    print("Debugging commands:")
                    print("  dbstop func   — Set breakpoint in function")
                    print("  dbstop in func at N — Set breakpoint at line N")
                    print("  dbstep        — Step to next line")
                    print("  dbcont        — Continue execution")
                    print("  dbquit        — Exit debug mode")
                    print("  dbstatus      — Show breakpoints")
                    print("  dbclear       — Clear all breakpoints")
                    print("")
                    print("MATLAB syntax supported:")
                    print("  x = [1 2 3; 4 5 6]  — Matrix")
                    print("  for i = 1:10 ... end — Loop")
                    print("  function y = f(x) ... end — Function")
                    print("  plot(x, y)           — Plot")
                    print("")
                    print("Tab completion: Press Tab for variable/function names")
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
                interpreter.global_env = __import__(
                    "matpy.environment", fromlist=["Environment"]
                ).Environment(name="global")
                print("Workspace cleared.")
                continue
            elif stripped == "clc":
                os.system("cls" if os.name == "nt" else "clear")
                continue
            # Debugging commands
            elif stripped.startswith("dbstop "):
                # Set breakpoint: dbstop in funcname at lineno
                parts = stripped.split()
                if len(parts) >= 4 and parts[1] == "in" and parts[3] == "at":
                    func_name = parts[2]
                    line_no = int(parts[4])
                    if func_name not in breakpoints:
                        breakpoints[func_name] = set()
                    breakpoints[func_name].add(line_no)
                    print(f"Breakpoint set in '{func_name}' at line {line_no}")
                elif len(parts) >= 2:
                    # Simple form: dbstop funcname
                    func_name = parts[1]
                    if func_name not in breakpoints:
                        breakpoints[func_name] = set()
                    breakpoints[func_name].add(1)
                    print(f"Breakpoint set in '{func_name}' at line 1")
                continue
            elif stripped == "dbstep":
                debug_step = True
                print("Stepping...")
                continue
            elif stripped == "dbcont":
                debug_mode = False
                debug_step = False
                print("Continuing...")
                continue
            elif stripped == "dbquit":
                debug_mode = False
                debug_step = False
                breakpoints.clear()
                print("Debug mode exited.")
                continue
            elif stripped == "dbstatus":
                if breakpoints:
                    print("Breakpoints:")
                    for func, lines in breakpoints.items():
                        print(f"  {func}: lines {sorted(lines)}")
                else:
                    print("No breakpoints set.")
                continue
            elif stripped == "dbclear":
                breakpoints.clear()
                print("All breakpoints cleared.")
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
            block_starters = [
                "function",
                "if",
                "for",
                "while",
                "switch",
                "try",
                "classdef",
            ]
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
        print("MatPy v0.5.0 — MATLAB interpreter in Python")
        print("")
        print("Usage:")
        print("  matpy                   Start interactive REPL")
        print("  matpy file.m            Execute MATLAB file")
        print("  matpy --bytecode file.m Execute with bytecode VM")
        print("  matpy -h/--help         Show this help")
        print("  matpy -v/--version      Show version")
        return

    if arg in ("-v", "--version"):
        print("MatPy v0.5.0")
        return

    if arg == "--bytecode":
        if len(sys.argv) < 3:
            print("Error: --bytecode requires a file argument", file=sys.stderr)
            sys.exit(1)
        run_file(sys.argv[2], engine="bytecode")
        return

    run_file(arg)


if __name__ == "__main__":
    main()
