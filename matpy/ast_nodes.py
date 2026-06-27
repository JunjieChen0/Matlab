"""AST node definitions for MatPy."""

from dataclasses import dataclass, field
from typing import Any


# ── Base ──────────────────────────────────────────────────────


class Node:
    pass


class Expr(Node):
    pass


class Stmt(Node):
    pass


# ── Expressions ───────────────────────────────────────────────


@dataclass
class NumberLiteral(Expr):
    value: int | float | complex


@dataclass
class StringLiteral(Expr):
    value: str


@dataclass
class Identifier(Expr):
    name: str


@dataclass
class BinaryOp(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass
class UnaryOp(Expr):
    op: str
    operand: Expr


@dataclass
class RangeExpr(Expr):
    start: Expr
    stop: Expr
    step: Expr | None = None


@dataclass
class MatrixLiteral(Expr):
    rows: list[list[Expr]] = field(default_factory=list)


@dataclass
class CellLiteral(Expr):
    rows: list[list[Expr]] = field(default_factory=list)


@dataclass
class IndexExpr(Expr):
    base: Expr
    indices: list[Expr] = field(default_factory=list)
    is_cell: bool = False


@dataclass
class FieldAccess(Expr):
    base: Expr
    field_name: str


@dataclass
class DynamicFieldAccess(Expr):
    base: Expr
    field_expr: Expr


@dataclass
class FuncCallExpr(Expr):
    name: str
    args: list[Expr] = field(default_factory=list)


@dataclass
class ParenExpr(Expr):
    expr: Expr


@dataclass
class FuncHandle(Expr):
    name: str | None = None
    params: list[str] = field(default_factory=list)
    body: Expr | None = None


@dataclass
class AnonFuncExpr(Expr):
    params: list[str]
    body: Expr


# ── Statements ────────────────────────────────────────────────


@dataclass
class ExprStmt(Stmt):
    expr: Expr


@dataclass
class Assignment(Stmt):
    targets: list[Expr]
    value: Expr


@dataclass
class IfStmt(Stmt):
    condition: Expr
    body: list[Stmt] = field(default_factory=list)
    elif_branches: list[tuple[Expr, list[Stmt]]] = field(default_factory=list)
    else_body: list[Stmt] = field(default_factory=list)


@dataclass
class ForStmt(Stmt):
    var: str
    iter_expr: Expr
    body: list[Stmt] = field(default_factory=list)


@dataclass
class WhileStmt(Stmt):
    condition: Expr
    body: list[Stmt] = field(default_factory=list)


@dataclass
class SwitchStmt(Stmt):
    expr: Expr
    cases: list[tuple[Expr, list[Stmt]]] = field(default_factory=list)
    otherwise: list[Stmt] = field(default_factory=list)


@dataclass
class TryCatchStmt(Stmt):
    try_body: list[Stmt] = field(default_factory=list)
    catch_var: str | None = None
    catch_body: list[Stmt] = field(default_factory=list)


@dataclass
class ReturnStmt(Stmt):
    pass


@dataclass
class BreakStmt(Stmt):
    pass


@dataclass
class ContinueStmt(Stmt):
    pass


@dataclass
class FuncDef(Stmt):
    name: str
    params: list[str] = field(default_factory=list)
    returns: list[str] = field(default_factory=list)
    body: list[Stmt] = field(default_factory=list)
    arg_specs: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class GlobalStmt(Stmt):
    names: list[str] = field(default_factory=list)


@dataclass
class PersistentStmt(Stmt):
    names: list[str] = field(default_factory=list)


@dataclass
class ClassDef(Stmt):
    name: str
    superclass: str | None = None
    class_attrs: dict[str, Any] = field(default_factory=dict)
    properties: dict[str, Expr | None] = field(default_factory=dict)
    property_attrs: dict[str, dict[str, Any]] = field(default_factory=dict)
    methods: dict[str, FuncDef] = field(default_factory=dict)
    method_attrs: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class Program(Node):
    statements: list[Stmt] = field(default_factory=list)


class NodeVisitor:
    def visit(self, node: Node) -> Any:
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: Node) -> Any:
        raise NotImplementedError(f"No visit_{type(node).__name__} method")
