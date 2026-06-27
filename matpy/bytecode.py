"""Bytecode compiler and virtual machine for MatPy.

Compiles AST to linear bytecode instructions for faster execution.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Any
import numpy as np

from matpy.ast_nodes import (
    Program, NumberLiteral, StringLiteral, Identifier, BinaryOp, UnaryOp,
    RangeExpr, MatrixLiteral, CellLiteral, IndexExpr, FieldAccess,
    FuncCallExpr, ParenExpr, ExprStmt, Assignment, IfStmt, ForStmt,
    WhileStmt, SwitchStmt, ReturnStmt, BreakStmt, ContinueStmt, FuncDef,
    GlobalStmt, PersistentStmt, Expr, Stmt,
)
from matpy.runtime.types import Mat


class OpCode(Enum):
    PUSH_CONST = auto()
    PUSH_VAR = auto()
    POP = auto()
    DUP = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    POW = auto()
    NEG = auto()
    DOT_MUL = auto()
    DOT_DIV = auto()
    DOT_POW = auto()
    EQ = auto()
    NEQ = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    LAND = auto()
    LOR = auto()
    STORE_VAR = auto()
    LOAD_VAR = auto()
    INDEX = auto()
    INDEX_ASSIGN = auto()
    FIELD_ACCESS = auto()
    JUMP = auto()
    JUMP_IF_FALSE = auto()
    JUMP_IF_TRUE = auto()
    CALL = auto()
    CALL_USER_FUNC = auto()
    RETURN = auto()
    BUILD_MATRIX = auto()
    BUILD_RANGE = auto()
    FOR_INIT = auto()
    FOR_NEXT = auto()
    FOR_END = auto()
    TRY_START = auto()
    TRY_END = auto()
    CATCH_START = auto()
    NOP = auto()
    HALT = auto()


@dataclass
class Instruction:
    op: OpCode
    arg: Any = None
    line: int = 0

    def __repr__(self):
        return f"{self.op.name} {self.arg!r}" if self.arg is not None else self.op.name


class BytecodeCompiler:
    def __init__(self):
        self.instructions: list[Instruction] = []
        self.constants: list[Any] = []
        self.constant_map: dict = {}
        self.label_counter = 0
        self.labels: dict[int, int] = {}
        self.functions: dict[str, Any] = {}
        self.func_bytecode: dict[str, list[Instruction]] = {}
        self.func_params: dict[str, list[str]] = {}

    def compile(self, program: Program) -> list[Instruction]:
        for stmt in program.statements:
            if isinstance(stmt, FuncDef):
                self.functions[stmt.name] = stmt
                self._compile_function(stmt)
            else:
                self._compile_stmt(stmt)
        self.instructions.append(Instruction(OpCode.HALT))
        self._resolve_labels(self.instructions)
        return self.instructions

    def _compile_function(self, func_def: FuncDef):
        """Compile a function body to bytecode."""
        old_instructions = self.instructions
        old_labels = self.labels
        self.instructions = []
        self.labels = {}

        self.func_params[func_def.name] = func_def.params

        for stmt in func_def.body:
            self._compile_stmt(stmt)

        # Add implicit return
        if func_def.returns:
            ret_name = func_def.returns[0]
            idx = self._add_constant(ret_name)
            self._emit(OpCode.LOAD_VAR, idx)
        self._emit(OpCode.RETURN)

        self._resolve_labels(self.instructions)
        self.func_bytecode[func_def.name] = self.instructions

        self.instructions = old_instructions
        self.labels = old_labels

    def _resolve_labels(self, instructions: list[Instruction]):
        """Resolve label references to instruction indices."""
        for i, instr in enumerate(instructions):
            if instr.op in (OpCode.JUMP, OpCode.JUMP_IF_FALSE, OpCode.JUMP_IF_TRUE):
                if instr.arg in self.labels:
                    instructions[i] = Instruction(instr.op, self.labels[instr.arg], instr.line)

    def _emit(self, op: OpCode, arg: Any = None):
        self.instructions.append(Instruction(op, arg))

    def _add_constant(self, value: Any) -> int:
        key = id(value) if not isinstance(value, (int, float, str, bool)) else value
        if key in self.constant_map:
            return self.constant_map[key]
        idx = len(self.constants)
        self.constants.append(value)
        self.constant_map[key] = idx
        return idx

    def _new_label(self) -> int:
        self.label_counter += 1
        return self.label_counter

    def _emit_label(self, label: int):
        self.labels[label] = len(self.instructions)

    def _compile_stmt(self, stmt: Stmt):
        match stmt:
            case ExprStmt(expr=expr):
                self._compile_expr(expr)
                self._emit(OpCode.POP)
            case Assignment(targets=targets, value=value):
                self._compile_expr(value)
                for target in targets:
                    if isinstance(target, Identifier):
                        self._emit(OpCode.STORE_VAR, self._add_constant(target.name))
                    elif isinstance(target, IndexExpr):
                        # Compile indexed assignment: A(i,j) = value
                        self._compile_expr(target.base)
                        for idx in target.indices:
                            self._compile_expr(idx)
                        self._emit(OpCode.INDEX_ASSIGN, len(target.indices))
            case IfStmt(condition=condition, body=body, elif_branches=elif_branches, else_body=else_body):
                end_label = self._new_label()
                self._compile_expr(condition)
                false_label = self._new_label()
                self._emit(OpCode.JUMP_IF_FALSE, false_label)
                for s in body:
                    self._compile_stmt(s)
                self._emit(OpCode.JUMP, end_label)
                self._emit_label(false_label)
                for cond, elif_body in elif_branches:
                    self._compile_expr(cond)
                    next_label = self._new_label()
                    self._emit(OpCode.JUMP_IF_FALSE, next_label)
                    for s in elif_body:
                        self._compile_stmt(s)
                    self._emit(OpCode.JUMP, end_label)
                    self._emit_label(next_label)
                for s in else_body:
                    self._compile_stmt(s)
                self._emit_label(end_label)
            case ForStmt(var=var, iter_expr=iter_expr, body=body):
                # Compile the iterator expression
                self._compile_expr(iter_expr)
                # Initialize for loop (convert to list, set index to 0)
                iter_idx = self._add_constant(f"__iter_{var}__")
                index_idx = self._add_constant(f"__index_{var}__")
                var_idx = self._add_constant(var)
                self._emit(OpCode.FOR_INIT, iter_idx)

                loop_start = self._new_label()
                loop_end = self._new_label()

                self._emit_label(loop_start)
                # Check if iterator is exhausted
                self._emit(OpCode.FOR_END, (iter_idx, index_idx))
                self._emit(OpCode.JUMP_IF_TRUE, loop_end)

                # Get next value and store in loop variable
                self._emit(OpCode.FOR_NEXT, (iter_idx, index_idx, var_idx))

                # Compile loop body
                for s in body:
                    self._compile_stmt(s)

                # Jump back to loop start
                self._emit(OpCode.JUMP, loop_start)
                self._emit_label(loop_end)
            case WhileStmt(condition=condition, body=body):
                loop_start = self._new_label()
                loop_end = self._new_label()
                self._emit_label(loop_start)
                self._compile_expr(condition)
                self._emit(OpCode.JUMP_IF_FALSE, loop_end)
                for s in body:
                    self._compile_stmt(s)
                self._emit(OpCode.JUMP, loop_start)
                self._emit_label(loop_end)
            case ReturnStmt():
                self._emit(OpCode.RETURN)
            case FuncDef():
                self.functions[stmt.name] = stmt
            case SwitchStmt():
                # Simplified switch: compile as if/elseif chain
                self._compile_expr(stmt.expr)
                val_idx = self._add_constant("__switch_val__")
                self._emit(OpCode.STORE_VAR, val_idx)
                end_label = self._new_label()
                for case_expr, body in stmt.cases:
                    self._emit(OpCode.LOAD_VAR, val_idx)
                    self._compile_expr(case_expr)
                    self._emit(OpCode.EQ)
                    next_label = self._new_label()
                    self._emit(OpCode.JUMP_IF_FALSE, next_label)
                    for s in body:
                        self._compile_stmt(s)
                    self._emit(OpCode.JUMP, end_label)
                    self._emit_label(next_label)
                if stmt.otherwise:
                    for s in stmt.otherwise:
                        self._compile_stmt(s)
                self._emit_label(end_label)
            case TryCatchStmt(try_body=try_body, catch_var=catch_var, catch_body=catch_body):
                # Compile try/catch as try-start/try-end/catch-start markers
                catch_label = self._new_label()
                end_label = self._new_label()
                self._emit(OpCode.TRY_START, catch_label)
                for s in try_body:
                    self._compile_stmt(s)
                self._emit(OpCode.TRY_END)
                self._emit(OpCode.JUMP, end_label)
                self._emit_label(catch_label)
                self._emit(OpCode.CATCH_START, self._add_constant(catch_var) if catch_var else None)
                for s in catch_body:
                    self._compile_stmt(s)
                self._emit_label(end_label)
            case GlobalStmt(names=names):
                for name in names:
                    self._emit(OpCode.PUSH_CONST, self._add_constant(name))
            case PersistentStmt(names=names):
                for name in names:
                    self._emit(OpCode.PUSH_CONST, self._add_constant(name))
            case _:
                pass

    def _compile_expr(self, expr: Expr):
        match expr:
            case NumberLiteral(value=value):
                self._emit(OpCode.PUSH_CONST, self._add_constant(value))
            case StringLiteral(value=value):
                self._emit(OpCode.PUSH_CONST, self._add_constant(value))
            case Identifier(name=name):
                self._emit(OpCode.LOAD_VAR, self._add_constant(name))
            case BinaryOp(op=op, left=left, right=right):
                self._compile_expr(left)
                self._compile_expr(right)
                op_map = {
                    "+": OpCode.ADD, "-": OpCode.SUB, "*": OpCode.MUL,
                    "/": OpCode.DIV, "^": OpCode.POW,
                    ".*": OpCode.DOT_MUL, "./": OpCode.DOT_DIV, ".^": OpCode.DOT_POW,
                    "==": OpCode.EQ, "~=": OpCode.NEQ,
                    "<": OpCode.LT, ">": OpCode.GT, "<=": OpCode.LE, ">=": OpCode.GE,
                    "&": OpCode.AND, "|": OpCode.OR, "&&": OpCode.LAND, "||": OpCode.LOR,
                }
                self._emit(op_map.get(op, OpCode.NOP))
            case UnaryOp(op=op, operand=operand):
                self._compile_expr(operand)
                if op == "-":
                    self._emit(OpCode.NEG)
                elif op == "~":
                    self._emit(OpCode.NOT)
            case FuncCallExpr(name=name, args=args):
                for arg in args:
                    self._compile_expr(arg)
                # Use CALL_USER_FUNC if it's a user-defined function
                if name in self.functions:
                    self._emit(OpCode.CALL_USER_FUNC, (name, len(args)))
                else:
                    self._emit(OpCode.CALL, (name, len(args)))
            case MatrixLiteral(rows=rows):
                for row in rows:
                    for elem in row:
                        self._compile_expr(elem)
                self._emit(OpCode.BUILD_MATRIX, (len(rows), len(rows[0]) if rows else 0))
            case RangeExpr(start=start, stop=stop, step=step):
                self._compile_expr(start)
                self._compile_expr(stop)
                if step:
                    self._compile_expr(step)
                self._emit(OpCode.BUILD_RANGE, 3 if step else 2)
            case ParenExpr(expr=inner):
                self._compile_expr(inner)
            case IndexExpr(base=base, indices=indices):
                self._compile_expr(base)
                for idx in indices:
                    self._compile_expr(idx)
                self._emit(OpCode.INDEX, len(indices))
            case FieldAccess(base=base, field_name=field_name):
                self._compile_expr(base)
                self._emit(OpCode.FIELD_ACCESS, self._add_constant(field_name))
            case _:
                self._emit(OpCode.PUSH_CONST, self._add_constant(None))


class BytecodeVM:
    def __init__(self):
        self.stack: list[Any] = []
        self.variables: dict[str, Any] = {}
        self.call_stack: list[tuple] = []
        self.functions: dict[str, Any] = {}
        self.func_bytecode: dict[str, list[Instruction]] = {}
        self.func_params: dict[str, list[str]] = {}

    def run(self, instructions: list[Instruction], constants: list[Any]) -> Any:
        self.constants = constants
        self.stack = []
        self.variables = {}
        return self._execute_bytecode(instructions)

    def _run_main_loop(self, instructions: list[Instruction], constants: list[Any]) -> Any:
        """Legacy main loop - kept for compatibility."""
        self.constants = constants
        self.ip = 0
        self.instructions = instructions

        while self.ip < len(instructions):
            instr = instructions[self.ip]
            self.ip += 1

            match instr.op:
                case OpCode.PUSH_CONST:
                    self.stack.append(constants[instr.arg])
                case OpCode.POP:
                    self.stack.pop()
                case OpCode.DUP:
                    self.stack.append(self.stack[-1])
                case OpCode.ADD:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._add(a, b))
                case OpCode.SUB:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._sub(a, b))
                case OpCode.MUL:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._mul(a, b))
                case OpCode.DIV:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._div(a, b))
                case OpCode.POW:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._pow(a, b))
                case OpCode.NEG:
                    self.stack.append(self._neg(self.stack.pop()))
                case OpCode.DOT_MUL:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._dot_mul(a, b))
                case OpCode.DOT_DIV:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._dot_div(a, b))
                case OpCode.DOT_POW:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._dot_pow(a, b))
                case OpCode.EQ:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._eq(a, b))
                case OpCode.NEQ:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._neq(a, b))
                case OpCode.LT:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._lt(a, b))
                case OpCode.GT:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._gt(a, b))
                case OpCode.LE:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._le(a, b))
                case OpCode.GE:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._ge(a, b))
                case OpCode.AND:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._and(a, b))
                case OpCode.OR:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._or(a, b))
                case OpCode.NOT:
                    self.stack.append(self._not(self.stack.pop()))
                case OpCode.LAND:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._is_truthy(a) and self._is_truthy(b))
                case OpCode.LOR:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._is_truthy(a) or self._is_truthy(b))
                case OpCode.STORE_VAR:
                    name = constants[instr.arg]
                    self.variables[name] = self.stack[-1]
                case OpCode.LOAD_VAR:
                    name = constants[instr.arg]
                    self.stack.append(self.variables.get(name))
                case OpCode.INDEX:
                    # Index into array: base(idx1, idx2, ...)
                    argc = instr.arg
                    indices = [self.stack.pop() for _ in range(argc)]
                    indices.reverse()
                    base = self.stack.pop()
                    if isinstance(base, Mat):
                        # Convert 1-based indexing
                        converted = []
                        for idx in indices:
                            if isinstance(idx, (int, float)):
                                converted.append(int(idx) - 1)
                            elif isinstance(idx, Mat):
                                converted.append(idx.data - 1)
                            else:
                                converted.append(idx)
                        try:
                            result = base.data[tuple(converted)]
                            if isinstance(result, np.ndarray) and result.ndim == 0:
                                result = result.item()
                            self.stack.append(Mat(result) if isinstance(result, np.ndarray) else result)
                        except (IndexError, ValueError):
                            self.stack.append(None)
                    else:
                        self.stack.append(None)
                case OpCode.INDEX_ASSIGN:
                    # Assign to indexed array: base(idx1, idx2, ...) = value
                    argc = instr.arg
                    indices = [self.stack.pop() for _ in range(argc)]
                    indices.reverse()
                    base = self.stack.pop()
                    value = self.stack.pop()
                    if isinstance(base, Mat):
                        # Convert 1-based indexing
                        converted = []
                        for idx in indices:
                            if isinstance(idx, (int, float)):
                                converted.append(int(idx) - 1)
                            elif isinstance(idx, Mat):
                                converted.append(idx.data - 1)
                            else:
                                converted.append(idx)
                        try:
                            base.data[tuple(converted)] = value
                        except (IndexError, ValueError):
                            pass
                    self.stack.append(value)
                case OpCode.JUMP:
                    self.ip = instr.arg
                case OpCode.JUMP_IF_FALSE:
                    if not self._is_truthy(self.stack.pop()):
                        self.ip = instr.arg
                case OpCode.JUMP_IF_TRUE:
                    if self._is_truthy(self.stack.pop()):
                        self.ip = instr.arg
                case OpCode.CALL:
                    name, argc = instr.arg
                    args = [self.stack.pop() for _ in range(argc)]
                    args.reverse()
                    # Call built-in function
                    from matpy.builtins import get_builtin
                    builtin = get_builtin(name)
                    if builtin:
                        result = builtin(*args)
                        self.stack.append(result)
                    else:
                        self.stack.append(None)
                case OpCode.CALL_USER_FUNC:
                    name, argc = instr.arg
                    args = [self.stack.pop() for _ in range(argc)]
                    args.reverse()
                    if name in self.func_bytecode:
                        # Native bytecode execution
                        result = self._call_user_func_native(name, args)
                    else:
                        result = self._call_user_func(name, args)
                    self.stack.append(result)
                case OpCode.RETURN:
                    if not self.call_stack:
                        return self.stack[-1] if self.stack else None
                    # Restore caller's state
                    saved_ip, saved_instructions, saved_variables, saved_constants = self.call_stack.pop()
                    ret_val = self.stack.pop() if self.stack else None
                    self.ip = saved_ip
                    self.instructions = saved_instructions
                    self.variables = saved_variables
                    self.constants = saved_constants
                    self.stack.append(ret_val)
                case OpCode.BUILD_MATRIX:
                    rows, cols = instr.arg
                    elems = [self.stack.pop() for _ in range(rows * cols)]
                    elems.reverse()
                    self.stack.append(Mat(np.array(elems).reshape(rows, cols) if rows > 1 else np.array(elems)))
                case OpCode.BUILD_RANGE:
                    count = instr.arg
                    if count == 3:
                        step, stop, start = self.stack.pop(), self.stack.pop(), self.stack.pop()
                    else:
                        stop, start = self.stack.pop(), self.stack.pop()
                        step = 1
                    self.stack.append(Mat(np.arange(start, stop + step, step)))
                case OpCode.FOR_INIT:
                    # Initialize for loop iterator
                    iter_name = self.constants[instr.arg]
                    val = self.stack.pop()
                    if isinstance(val, Mat):
                        data = val.data
                        if data.ndim == 1:
                            items = list(data)
                        else:
                            items = list(data.flat)
                    elif isinstance(val, np.ndarray):
                        if val.ndim == 1:
                            items = list(val)
                        else:
                            items = list(val.flat)
                    elif isinstance(val, (list, tuple)):
                        items = list(val)
                    else:
                        items = [val]
                    self.variables[iter_name] = items
                case OpCode.FOR_NEXT:
                    # Get next value from iterator
                    iter_idx, index_idx, var_idx = instr.arg
                    iter_name = self.constants[iter_idx]
                    index_name = self.constants[index_idx]
                    var_name = self.constants[var_idx]
                    items = self.variables.get(iter_name, [])
                    idx = self.variables.get(index_name, 0)
                    if idx < len(items):
                        self.variables[var_name] = items[idx]
                        self.variables[index_name] = idx + 1
                case OpCode.FOR_END:
                    # Check if iterator is exhausted (push True if done)
                    iter_idx, index_idx = instr.arg
                    iter_name = self.constants[iter_idx]
                    index_name = self.constants[index_idx]
                    items = self.variables.get(iter_name, [])
                    idx = self.variables.get(index_name, 0)
                    self.stack.append(idx >= len(items))
                case OpCode.TRY_START:
                    # Store catch label for exception handling
                    self._try_catch_stack = getattr(self, '_try_catch_stack', [])
                    self._try_catch_stack.append(instr.arg)
                case OpCode.TRY_END:
                    # Pop try context if no exception
                    if hasattr(self, '_try_catch_stack') and self._try_catch_stack:
                        self._try_catch_stack.pop()
                case OpCode.CATCH_START:
                    # Exception was caught, store it in variable if specified
                    if instr.arg is not None:
                        exc = self.stack.pop() if self.stack else None
                        var_name = constants[instr.arg]
                        self.variables[var_name] = exc
                case OpCode.HALT:
                    break
                case _:
                    pass

        return self.stack[-1] if self.stack else None

    def _to_numpy(self, val):
        return val.data if isinstance(val, Mat) else val

    def _add(self, a, b):
        a, b = self._to_numpy(a), self._to_numpy(b)
        r = a + b
        return Mat(r) if isinstance(r, np.ndarray) else r

    def _sub(self, a, b):
        a, b = self._to_numpy(a), self._to_numpy(b)
        r = a - b
        return Mat(r) if isinstance(r, np.ndarray) else r

    def _mul(self, a, b):
        a, b = self._to_numpy(a), self._to_numpy(b)
        r = a @ b if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) else a * b
        return Mat(r) if isinstance(r, np.ndarray) else r

    def _div(self, a, b):
        a, b = self._to_numpy(a), self._to_numpy(b)
        r = a / b
        return Mat(r) if isinstance(r, np.ndarray) else r

    def _pow(self, a, b):
        a, b = self._to_numpy(a), self._to_numpy(b)
        r = np.linalg.matrix_power(a, int(b)) if isinstance(a, np.ndarray) else a ** b
        return Mat(r) if isinstance(r, np.ndarray) else r

    def _neg(self, a):
        a = self._to_numpy(a)
        r = -a
        return Mat(r) if isinstance(r, np.ndarray) else r

    def _dot_mul(self, a, b):
        a, b = self._to_numpy(a), self._to_numpy(b)
        return Mat(a * b)

    def _dot_div(self, a, b):
        a, b = self._to_numpy(a), self._to_numpy(b)
        return Mat(a / b)

    def _dot_pow(self, a, b):
        a, b = self._to_numpy(a), self._to_numpy(b)
        return Mat(a ** b)

    def _eq(self, a, b):
        a, b = self._to_numpy(a), self._to_numpy(b)
        r = a == b
        return Mat(r) if isinstance(r, np.ndarray) else r

    def _neq(self, a, b):
        a, b = self._to_numpy(a), self._to_numpy(b)
        r = a != b
        return Mat(r) if isinstance(r, np.ndarray) else r

    def _lt(self, a, b):
        a, b = self._to_numpy(a), self._to_numpy(b)
        r = a < b
        return Mat(r) if isinstance(r, np.ndarray) else r

    def _gt(self, a, b):
        a, b = self._to_numpy(a), self._to_numpy(b)
        r = a > b
        return Mat(r) if isinstance(r, np.ndarray) else r

    def _le(self, a, b):
        a, b = self._to_numpy(a), self._to_numpy(b)
        r = a <= b
        return Mat(r) if isinstance(r, np.ndarray) else r

    def _ge(self, a, b):
        a, b = self._to_numpy(a), self._to_numpy(b)
        r = a >= b
        return Mat(r) if isinstance(r, np.ndarray) else r

    def _and(self, a, b):
        a, b = self._to_numpy(a), self._to_numpy(b)
        return Mat(np.logical_and(a, b)) if isinstance(a, np.ndarray) else a and b

    def _or(self, a, b):
        a, b = self._to_numpy(a), self._to_numpy(b)
        return Mat(np.logical_or(a, b)) if isinstance(a, np.ndarray) else a or b

    def _not(self, a):
        a = self._to_numpy(a)
        return Mat(np.logical_not(a)) if isinstance(a, np.ndarray) else not a

    def _is_truthy(self, val):
        if isinstance(val, Mat):
            return bool(np.all(val.data))
        if isinstance(val, np.ndarray):
            return bool(np.all(val))
        return bool(val)

    def _call_user_func(self, name: str, args: list) -> Any:
        """Call a user-defined function using the tree-walking interpreter."""
        from matpy.interpreter import Interpreter
        from matpy.environment import Environment

        func_def = self.functions[name]

        # Create interpreter with all functions registered
        interp = Interpreter()
        interp.functions = self.functions

        # Call the function directly
        result = interp._call_user_func(func_def, args, interp.global_env)
        return result

    def _call_user_func_native(self, name: str, args: list) -> Any:
        """Call a user-defined function using native bytecode execution."""
        # Save current state
        saved_stack = list(self.stack)
        saved_variables = dict(self.variables)
        saved_constants = self.constants

        # Set up new frame
        self.stack = []
        self.constants = saved_constants

        # Bind parameters
        params = self.func_params.get(name, [])
        for i, param in enumerate(params):
            self.variables[param] = args[i] if i < len(args) else None

        # Execute function bytecode
        func_instructions = self.func_bytecode[name]
        result = self._execute_bytecode(func_instructions)

        # Restore caller's state
        self.stack = saved_stack
        self.variables = saved_variables
        self.constants = saved_constants

        return result

    def _execute_bytecode(self, instructions: list[Instruction]) -> Any:
        """Execute a list of bytecode instructions until RETURN or HALT."""
        ip = 0
        while ip < len(instructions):
            instr = instructions[ip]
            ip += 1

            match instr.op:
                case OpCode.PUSH_CONST:
                    self.stack.append(self.constants[instr.arg])
                case OpCode.POP:
                    self.stack.pop()
                case OpCode.DUP:
                    self.stack.append(self.stack[-1])
                case OpCode.ADD:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._add(a, b))
                case OpCode.SUB:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._sub(a, b))
                case OpCode.MUL:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._mul(a, b))
                case OpCode.DIV:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._div(a, b))
                case OpCode.POW:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._pow(a, b))
                case OpCode.NEG:
                    a = self.stack.pop()
                    self.stack.append(self._neg(a))
                case OpCode.DOT_MUL:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._dot_mul(a, b))
                case OpCode.DOT_DIV:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._dot_div(a, b))
                case OpCode.DOT_POW:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._dot_pow(a, b))
                case OpCode.EQ:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._eq(a, b))
                case OpCode.NEQ:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._neq(a, b))
                case OpCode.LT:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._lt(a, b))
                case OpCode.GT:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._gt(a, b))
                case OpCode.LE:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._le(a, b))
                case OpCode.GE:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._ge(a, b))
                case OpCode.AND:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._and(a, b))
                case OpCode.OR:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._or(a, b))
                case OpCode.NOT:
                    a = self.stack.pop()
                    self.stack.append(self._not(a))
                case OpCode.LAND:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._is_truthy(a) and self._is_truthy(b))
                case OpCode.LOR:
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(self._is_truthy(a) or self._is_truthy(b))
                case OpCode.STORE_VAR:
                    name = self.constants[instr.arg]
                    self.variables[name] = self.stack[-1]
                case OpCode.LOAD_VAR:
                    name = self.constants[instr.arg]
                    self.stack.append(self.variables.get(name))
                case OpCode.INDEX:
                    argc = instr.arg
                    indices = [self.stack.pop() for _ in range(argc)]
                    indices.reverse()
                    base = self.stack.pop()
                    if isinstance(base, Mat):
                        converted = []
                        for idx in indices:
                            if isinstance(idx, (int, float)):
                                converted.append(int(idx) - 1)
                            elif isinstance(idx, Mat):
                                converted.append(idx.data - 1)
                            else:
                                converted.append(idx)
                        try:
                            result = base.data[tuple(converted)]
                            if isinstance(result, np.ndarray) and result.ndim == 0:
                                result = result.item()
                            self.stack.append(Mat(result) if isinstance(result, np.ndarray) else result)
                        except (IndexError, ValueError):
                            self.stack.append(None)
                    else:
                        self.stack.append(None)
                case OpCode.INDEX_ASSIGN:
                    argc = instr.arg
                    indices = [self.stack.pop() for _ in range(argc)]
                    indices.reverse()
                    base = self.stack.pop()
                    value = self.stack.pop()
                    if isinstance(base, Mat):
                        converted = []
                        for idx in indices:
                            if isinstance(idx, (int, float)):
                                converted.append(int(idx) - 1)
                            elif isinstance(idx, Mat):
                                converted.append(idx.data - 1)
                            else:
                                converted.append(idx)
                        try:
                            base.data[tuple(converted)] = value
                        except (IndexError, ValueError):
                            pass
                    self.stack.append(value)
                case OpCode.JUMP:
                    ip = instr.arg
                case OpCode.JUMP_IF_FALSE:
                    if not self._is_truthy(self.stack.pop()):
                        ip = instr.arg
                case OpCode.JUMP_IF_TRUE:
                    if self._is_truthy(self.stack.pop()):
                        ip = instr.arg
                case OpCode.CALL:
                    name, argc = instr.arg
                    args = [self.stack.pop() for _ in range(argc)]
                    args.reverse()
                    from matpy.builtins import get_builtin
                    builtin = get_builtin(name)
                    if builtin:
                        result = builtin(*args)
                        self.stack.append(result)
                    else:
                        self.stack.append(None)
                case OpCode.CALL_USER_FUNC:
                    name, argc = instr.arg
                    args = [self.stack.pop() for _ in range(argc)]
                    args.reverse()
                    if name in self.func_bytecode:
                        result = self._call_user_func_native(name, args)
                    else:
                        result = self._call_user_func(name, args)
                    self.stack.append(result)
                case OpCode.BUILD_MATRIX:
                    rows, cols = instr.arg
                    elems = [self.stack.pop() for _ in range(rows * cols)]
                    elems.reverse()
                    self.stack.append(Mat(np.array(elems).reshape(rows, cols) if rows > 1 else np.array(elems)))
                case OpCode.BUILD_RANGE:
                    count = instr.arg
                    if count == 3:
                        step, stop, start = self.stack.pop(), self.stack.pop(), self.stack.pop()
                    else:
                        stop, start = self.stack.pop(), self.stack.pop()
                        step = 1
                    self.stack.append(Mat(np.arange(start, stop + step, step)))
                case OpCode.FOR_INIT:
                    iter_name = self.constants[instr.arg]
                    val = self.stack.pop()
                    if isinstance(val, Mat):
                        data = val.data
                        items = list(data) if data.ndim == 1 else list(data.flat)
                    elif isinstance(val, np.ndarray):
                        items = list(val) if val.ndim == 1 else list(val.flat)
                    elif isinstance(val, (list, tuple)):
                        items = list(val)
                    else:
                        items = [val]
                    self.variables[iter_name] = items
                case OpCode.FOR_NEXT:
                    iter_idx, index_idx, var_idx = instr.arg
                    iter_name = self.constants[iter_idx]
                    index_name = self.constants[index_idx]
                    var_name = self.constants[var_idx]
                    items = self.variables.get(iter_name, [])
                    idx = self.variables.get(index_name, 0)
                    if idx < len(items):
                        self.variables[var_name] = items[idx]
                        self.variables[index_name] = idx + 1
                case OpCode.FOR_END:
                    iter_idx, index_idx = instr.arg
                    iter_name = self.constants[iter_idx]
                    index_name = self.constants[index_idx]
                    items = self.variables.get(iter_name, [])
                    idx = self.variables.get(index_name, 0)
                    self.stack.append(idx >= len(items))
                case OpCode.TRY_START:
                    self._try_catch_stack = getattr(self, '_try_catch_stack', [])
                    self._try_catch_stack.append(instr.arg)
                case OpCode.TRY_END:
                    if hasattr(self, '_try_catch_stack') and self._try_catch_stack:
                        self._try_catch_stack.pop()
                case OpCode.CATCH_START:
                    if instr.arg is not None:
                        exc = self.stack.pop() if self.stack else None
                        var_name = self.constants[instr.arg]
                        self.variables[var_name] = exc
                case OpCode.RETURN:
                    return self.stack.pop() if self.stack else None
                case OpCode.HALT:
                    return self.stack[-1] if self.stack else None
                case _:
                    pass

        return self.stack[-1] if self.stack else None
