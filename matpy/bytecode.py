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
    WhileStmt, ReturnStmt, BreakStmt, ContinueStmt, FuncDef,
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
    RETURN = auto()
    BUILD_MATRIX = auto()
    BUILD_RANGE = auto()
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

    def compile(self, program: Program) -> list[Instruction]:
        for stmt in program.statements:
            if isinstance(stmt, FuncDef):
                self.functions[stmt.name] = stmt
            else:
                self._compile_stmt(stmt)
        self.instructions.append(Instruction(OpCode.HALT))
        # Resolve labels
        for i, instr in enumerate(self.instructions):
            if instr.op in (OpCode.JUMP, OpCode.JUMP_IF_FALSE, OpCode.JUMP_IF_TRUE):
                if instr.arg in self.labels:
                    self.instructions[i] = Instruction(instr.op, self.labels[instr.arg], instr.line)
        return self.instructions

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
                self._compile_expr(iter_expr)
                iter_idx = self._add_constant(f"__iter_{var}__")
                self._emit(OpCode.STORE_VAR, iter_idx)
                var_idx = self._add_constant(var)
                loop_start = self._new_label()
                loop_end = self._new_label()
                self._emit_label(loop_start)
                self._emit(OpCode.LOAD_VAR, iter_idx)
                self._emit(OpCode.PUSH_CONST, self._add_constant(None))
                self._emit(OpCode.EQ)
                self._emit(OpCode.JUMP_IF_TRUE, loop_end)
                self._emit(OpCode.LOAD_VAR, iter_idx)
                self._emit(OpCode.STORE_VAR, var_idx)
                for s in body:
                    self._compile_stmt(s)
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

    def run(self, instructions: list[Instruction], constants: list[Any]) -> Any:
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
