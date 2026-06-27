"""Tree-walking interpreter for MatPy."""

from __future__ import annotations
from typing import Any
import numpy as np

from matpy.ast_nodes import (
    Program,
    NumberLiteral,
    StringLiteral,
    Identifier,
    BinaryOp,
    UnaryOp,
    RangeExpr,
    MatrixLiteral,
    CellLiteral,
    IndexExpr,
    FieldAccess,
    DynamicFieldAccess,
    FuncCallExpr,
    ParenExpr,
    FuncHandle,
    AnonFuncExpr,
    ExprStmt,
    Assignment,
    IfStmt,
    ForStmt,
    WhileStmt,
    SwitchStmt,
    TryCatchStmt,
    ReturnStmt,
    BreakStmt,
    ContinueStmt,
    FuncDef,
    GlobalStmt,
    PersistentStmt,
    ClassDef,
    Expr,
    Stmt,
    NodeVisitor,
)
from matpy.environment import Environment
from matpy.runtime.types import (
    Mat,
    CellArray,
    Struct,
    FuncHandle as RTFuncHandle,
    ClassInstance,
    MException,
    Table,
)
from matpy.builtins import get_builtin


class ReturnSignal(Exception):
    def __init__(self, value=None):
        self.value = value


class BreakSignal(Exception):
    pass


class ContinueSignal(Exception):
    pass


class MatPyError(Exception):
    """Base exception for MatPy errors."""

    def __init__(
        self,
        message: str,
        line: int = 0,
        col: int = 0,
        filename: str = "<unknown>",
        identifier: str = "",
    ):
        self.line = line
        self.col = col
        self.filename = filename
        self.identifier = identifier
        if line > 0:
            super().__init__(f"{filename}:{line}:{col}: {message}")
        else:
            super().__init__(message)


class MatPySyntaxError(MatPyError):
    """Syntax error in MatPy."""

    pass


class MatPyTypeError(MatPyError):
    """Type error in MatPy."""

    pass


class MatPyIndexError(MatPyError):
    """Index error in MatPy."""

    pass


class MatPyNameError(MatPyError):
    """Name error in MatPy."""

    pass


class MatPyValueError(MatPyError):
    """Value error in MatPy."""

    pass


class MatPyDimensionError(MatPyError):
    """Dimension mismatch error in MatPy."""

    pass


class MatPyMemoryError(MatPyError):
    """Memory error in MatPy."""

    pass


class MatPyNotImplementedError(MatPyError):
    """Feature not implemented in MatPy."""

    pass


class MatPyFileNotFoundError(MatPyError):
    """File not found error in MatPy."""

    pass


class InterpreterError(MatPyError):
    """Runtime error in MatPy interpreter."""

    def __init__(self, message: str, node=None, line: int = 0, col: int = 0):
        self.node = node
        super().__init__(message, line, col)


class Interpreter(NodeVisitor):
    def __init__(self):
        self.global_env = Environment(name="global")
        self.functions: dict[str, FuncDef] = {}
        self.classes: dict[str, Any] = {}
        self.current_file: str = "<unknown>"
        self.current_line: int = 0

    def run(self, program: Program):
        # Support local functions: first function is main, rest are local
        func_defs = [s for s in program.statements if isinstance(s, FuncDef)]
        non_func_stmts = [s for s in program.statements if not isinstance(s, FuncDef)]

        if func_defs:
            # Register all functions (first is main, rest are local)
            for func_def in func_defs:
                self.functions[func_def.name] = func_def

            # Execute non-function statements
            self._exec_block(non_func_stmts, self.global_env)

            # If there's a main function and no other statements, call it
            if not non_func_stmts and func_defs:
                main_func = func_defs[0]
                try:
                    self._call_user_func(main_func, [], self.global_env)
                except ReturnSignal:
                    pass
        else:
            self._exec_block(program.statements, self.global_env)

    def _exec_block(self, stmts: list[Stmt], env: Environment):
        for stmt in stmts:
            self._exec_stmt(stmt, env)

    def _exec_stmt(self, stmt: Stmt, env: Environment):
        match stmt:
            case ExprStmt(expr=expr):
                self._eval(expr, env)
            case Assignment(targets=targets, value=value):
                self._exec_assign(targets, value, env)
            case IfStmt():
                self._exec_if(stmt, env)
            case ForStmt():
                self._exec_for(stmt, env)
            case WhileStmt():
                self._exec_while(stmt, env)
            case SwitchStmt():
                self._exec_switch(stmt, env)
            case TryCatchStmt():
                self._exec_try(stmt, env)
            case ReturnStmt():
                raise ReturnSignal()
            case BreakStmt():
                raise BreakSignal()
            case ContinueStmt():
                raise ContinueSignal()
            case FuncDef():
                # Check if this is a nested function (inside another function)
                if env.name.startswith("func:"):
                    # Nested function: store with closure environment
                    stmt._closure_env = env
                    # Store in parent function's scope
                    env.set(stmt.name, stmt)
                else:
                    # Top-level function
                    self.functions[stmt.name] = stmt
            case GlobalStmt(names=names):
                for name in names:
                    env.define_global(name)
            case PersistentStmt(names=names):
                for name in names:
                    env.define_persistent(name)
            case ClassDef():
                self._exec_classdef(stmt, env)
            case _:
                raise InterpreterError(f"Unknown statement type: {type(stmt).__name__}")

    def _exec_assign(self, targets: list[Expr], value: Expr, env: Environment):
        val = self._eval(value, env)

        if len(targets) > 1 and isinstance(value, FuncCallExpr):
            results = val
            if not isinstance(results, (list, tuple)):
                results = [results]
            while len(results) < len(targets):
                results.append(None)
            for target, result in zip(targets, results):
                if isinstance(target, Identifier) and target.name == "~":
                    continue
                self._assign_target(target, result, env)
        elif len(targets) == 1:
            self._assign_target(targets[0], val, env)
        else:
            for target in targets:
                if isinstance(target, Identifier) and target.name == "~":
                    continue
                self._assign_target(target, val, env)

    def _assign_target(self, target: Expr, value: Any, env: Environment):
        if isinstance(target, Identifier):
            env.set(target.name, value)
        elif isinstance(target, IndexExpr):
            base = self._eval(target.base, env)
            if isinstance(base, Mat):
                indices = tuple(self._eval(idx, env) for idx in target.indices)
                data = base.data.copy()

                # Handle logical indexing assignment: A(A > 5) = 0
                if len(indices) == 1:
                    idx = indices[0]
                    if isinstance(idx, Mat):
                        idx = idx.data
                    if isinstance(idx, np.ndarray) and idx.dtype == bool:
                        data[idx] = value
                        env.set(target.base.name, Mat(data))
                        return

                converted = []
                for idx in indices:
                    if isinstance(idx, (int, float, np.integer, np.floating)):
                        converted.append(int(idx) - 1)
                    elif isinstance(idx, Mat):
                        arr = idx.data
                        if arr.dtype == bool:
                            converted.append(arr)
                        else:
                            converted.append(arr - 1)
                    elif isinstance(idx, np.ndarray):
                        if idx.dtype == bool:
                            converted.append(idx)
                        else:
                            converted.append(idx - 1)
                    else:
                        converted.append(idx)
                if len(converted) == 1:
                    if (
                        isinstance(converted[0], np.ndarray)
                        and converted[0].dtype == bool
                    ):
                        data[converted[0]] = value
                    else:
                        data.flat[converted[0]] = value
                else:
                    data[tuple(converted)] = value
                env.set(target.base.name, Mat(data))
            elif isinstance(base, Struct):
                if isinstance(target.indices[0], StringLiteral):
                    base.set_field(target.indices[0].value, value)
        elif isinstance(target, FieldAccess):
            try:
                base = self._eval(target.base, env)
            except NameError:
                # Variable doesn't exist — create a new Struct
                base = Struct()
                if isinstance(target.base, Identifier):
                    env.set(target.base.name, base)
            if isinstance(base, Struct):
                base.set_field(target.field_name, value)
            elif isinstance(base, ClassInstance):
                base.set_property(target.field_name, value)

    def _exec_if(self, stmt: IfStmt, env: Environment):
        cond = self._eval(stmt.condition, env)
        if self._is_truthy(cond):
            self._exec_block(stmt.body, env)  # Use same scope
            return
        for cond_expr, body in stmt.elif_branches:
            if self._is_truthy(self._eval(cond_expr, env)):
                self._exec_block(body, env)
                return
        if stmt.else_body:
            self._exec_block(stmt.else_body, env)

    def _exec_for(self, stmt: ForStmt, env: Environment):
        iter_val = self._eval(stmt.iter_expr, env)
        if isinstance(iter_val, Mat):
            data = iter_val.data
            if data.ndim == 0:
                # MATLAB: for i = scalar iterates once
                items = [iter_val]
            elif data.ndim == 1:
                items = list(data)
            elif data.ndim == 2:
                # MATLAB: for i = A iterates over columns
                items = [Mat(data[:, i]) for i in range(data.shape[1])]
            else:
                # MATLAB: for i = A iterates along first dimension
                # Each item is a (n-1)D slice
                items = [Mat(data[i, ...]) for i in range(data.shape[0])]
        elif isinstance(iter_val, np.ndarray):
            if iter_val.ndim <= 2:
                items = list(iter_val.flat)
            else:
                items = [Mat(iter_val[i, ...]) for i in range(iter_val.shape[0])]
        elif isinstance(iter_val, (list, tuple)):
            items = list(iter_val)
        else:
            items = [iter_val]

        for item in items:
            env.set(stmt.var, item)  # Set in parent scope
            try:
                self._exec_block(stmt.body, env)
            except BreakSignal:
                break
            except ContinueSignal:
                continue

    def _exec_while(self, stmt: WhileStmt, env: Environment):
        while self._is_truthy(self._eval(stmt.condition, env)):
            try:
                self._exec_block(stmt.body, env)  # Use same scope
            except BreakSignal:
                break
            except ContinueSignal:
                continue

    def _exec_switch(self, stmt: SwitchStmt, env: Environment):
        val = self._eval(stmt.expr, env)
        for case_expr, body in stmt.cases:
            case_val = self._eval(case_expr, env)
            # Support cell array in case: case {1,2,3}
            if isinstance(case_val, CellArray):
                # Iterate through all elements in the cell array
                flat = [item for row in case_val._data for item in row]
                for item in flat:
                    if self._is_equal(val, item):
                        self._exec_block(body, env)
                        return
            elif self._is_equal(val, case_val):
                self._exec_block(body, env)  # Use same scope
                return
        if stmt.otherwise:
            self._exec_block(stmt.otherwise, env)

    def _exec_try(self, stmt: TryCatchStmt, env: Environment):
        try:
            self._exec_block(stmt.try_body, env)
        except Exception as e:
            # Re-raise control flow signals — MATLAB try/catch does NOT intercept return/break/continue
            if isinstance(e, (ReturnSignal, BreakSignal, ContinueSignal)):
                raise
            # Set lasterror
            import matpy.builtins.io as io_module

            if isinstance(e, MException):
                io_module._last_error = e
            else:
                io_module._last_error = MException(
                    message=str(e), identifier=type(e).__name__, original=e
                )

            if stmt.catch_var:
                # Create MException object with message and identifier
                if isinstance(e, MException):
                    exc = e
                else:
                    exc = MException(
                        message=str(e), identifier=type(e).__name__, original=e
                    )
                env.set(stmt.catch_var, exc)
            self._exec_block(stmt.catch_body, env)

    def _exec_classdef(self, stmt: ClassDef, env: Environment):
        """Execute a classdef block — register the class and its constructor."""
        from matpy.runtime.types import ClassDefRuntime, ClassInstance

        class_runtime = ClassDefRuntime(stmt.name, stmt.superclass, stmt.class_attrs)

        # Inherit from superclass if specified
        if stmt.superclass and stmt.superclass in self.classes:
            super_cls = self.classes[stmt.superclass]
            # Prevent subclassing sealed classes
            if super_cls.is_sealed:
                raise InterpreterError(
                    f"Cannot subclass sealed class '{stmt.superclass}'"
                )
            # Inherit properties
            for prop_name, default_val in super_cls.properties.items():
                if prop_name not in stmt.properties:
                    class_runtime.properties[prop_name] = default_val
                    class_runtime.property_attrs[prop_name] = (
                        super_cls.property_attrs.get(prop_name, {})
                    )
            # Inherit methods
            for meth_name, meth_def in super_cls.methods.items():
                if meth_name not in stmt.methods:
                    class_runtime.methods[meth_name] = meth_def
                    class_runtime.method_attrs[meth_name] = super_cls.method_attrs.get(
                        meth_name, {}
                    )

        # Evaluate property defaults
        for prop_name, default_expr in stmt.properties.items():
            prop_attr = stmt.property_attrs.get(prop_name, {})
            if default_expr is not None:
                class_runtime.properties[prop_name] = self._eval(default_expr, env)
            else:
                class_runtime.properties[prop_name] = None
            class_runtime.property_attrs[prop_name] = prop_attr
            # Store Constant properties in the class runtime for class-level access
            if prop_attr.get("Constant"):
                class_runtime.properties[prop_name] = (
                    self._eval(default_expr, env) if default_expr is not None else None
                )

        # Register methods
        for meth_name, meth_def in stmt.methods.items():
            class_runtime.methods[meth_name] = meth_def
            class_runtime.method_attrs[meth_name] = stmt.method_attrs.get(meth_name, {})

        # Register Static methods as standalone builtin functions
        for meth_name, meth_def in stmt.methods.items():
            meth_attr = stmt.method_attrs.get(meth_name, {})
            if meth_attr.get("Static"):

                def make_static_call(md):
                    def static_call(*args):
                        return self._call_user_func(md, list(args), env)

                    return static_call

                from matpy.builtins import register

                register(f"{stmt.name}.{meth_name}", make_static_call(meth_def))

        # Register the class name as a constructor function
        def constructor(*args):
            # Prevent instantiation of abstract classes
            if class_runtime.is_abstract:
                raise InterpreterError(
                    f"Cannot instantiate abstract class '{stmt.name}'"
                )
            instance = ClassInstance(class_runtime)
            # Call constructor method if it exists
            if stmt.name in class_runtime.methods:
                result = self._call_user_func(
                    class_runtime.methods[stmt.name], list(args), env
                )
                if result is not None and isinstance(result, ClassInstance):
                    return result
            return instance

        # Store class definition
        self.classes[stmt.name] = class_runtime

        # Register constructor as a built-in function
        constructor.__name__ = stmt.name
        from matpy.builtins import register

        register(stmt.name, constructor)

    def _eval(self, expr: Expr, env: Environment) -> Any:
        match expr:
            case NumberLiteral(value=v):
                return v
            case StringLiteral(value=v):
                return v
            case Identifier(name=name):
                return self._eval_identifier(name, env)
            case BinaryOp(op=op, left=left, right=right):
                return self._eval_binop(op, left, right, env)
            case UnaryOp(op=op, operand=operand):
                return self._eval_unaryop(op, operand, env)
            case RangeExpr():
                return self._eval_range(expr, env)
            case MatrixLiteral(rows=rows):
                return self._eval_matrix(rows, env)
            case CellLiteral(rows=rows):
                return self._eval_cell(rows, env)
            case FuncCallExpr(name=name, args=args):
                return self._eval_func_call(name, args, env)
            case ParenExpr(expr=inner):
                return self._eval(inner, env)
            case IndexExpr():
                return self._eval_index(expr, env)
            case FieldAccess(base=base, field_name=field):
                return self._eval_field_access(base, field, env)
            case DynamicFieldAccess(base=base, field_expr=fexpr):
                field_name = self._eval(fexpr, env)
                return self._eval_field_access(base, str(field_name), env)
            case FuncHandle(name=name):
                return self._eval_func_handle(name, env)
            case AnonFuncExpr(params=params, body=body):
                return self._eval_anon_func(params, body, env)
            case _:
                raise InterpreterError(
                    f"Unknown expression type: {type(expr).__name__}"
                )

    def _eval_identifier(self, name: str, env: Environment) -> Any:
        if name == "end":
            return "end_marker"
        builtin = get_builtin(name)
        if builtin is not None and name in ("pi", "inf", "nan", "eps"):
            return builtin()
        try:
            return env.get(name)
        except NameError:
            builtin = get_builtin(name)
            if builtin is not None:
                return builtin()
            raise

    def _eval_binop(self, op: str, left: Expr, right: Expr, env: Environment) -> Any:
        # Short-circuit operators: only evaluate right operand if needed
        if op == "&&":
            left_val = self._eval(left, env)
            if not self._is_truthy(left_val):
                return False
            r = self._eval(right, env)
            return self._is_truthy(r)
        elif op == "||":
            left_val = self._eval(left, env)
            if self._is_truthy(left_val):
                return True
            r = self._eval(right, env)
            return self._is_truthy(r)

        left_val = self._eval(left, env)
        right_val = self._eval(right, env)

        if isinstance(left_val, Mat):
            left_val = left_val.data
        if isinstance(right_val, Mat):
            right_val = right_val.data

        match op:
            case "+":
                result = left_val + right_val
            case "-":
                result = left_val - right_val
            case "*":
                if isinstance(left_val, np.ndarray) and isinstance(right_val, np.ndarray):
                    result = left_val @ right_val
                else:
                    result = left_val * right_val
            case "/":
                if isinstance(left_val, np.ndarray) and isinstance(right_val, np.ndarray):
                    try:
                        result = np.linalg.solve(right_val.T, left_val.T).T
                    except np.linalg.LinAlgError:
                        # Use least-squares for non-square or singular matrices
                        result, _, _, _ = np.linalg.lstsq(right_val.T, left_val.T, rcond=None)
                        result = result.T
                else:
                    result = left_val / right_val
            case "\\":
                if isinstance(left_val, np.ndarray):
                    try:
                        result = np.linalg.solve(left_val, right_val)
                    except np.linalg.LinAlgError:
                        # Use least-squares for non-square or singular matrices
                        result, _, _, _ = np.linalg.lstsq(left_val, right_val, rcond=None)
                else:
                    result = right_val / left_val
            case "^":
                if isinstance(left_val, np.ndarray):
                    # Check if exponent is effectively an integer
                    if isinstance(right_val, (int, np.integer)) or (
                        isinstance(right_val, float) and right_val == int(right_val)
                    ):
                        result = np.linalg.matrix_power(left_val, int(right_val))
                    elif left_val.shape[0] == left_val.shape[1]:
                        # Fractional matrix power via eigendecomposition for square matrices
                        eigvals, eigvecs = np.linalg.eig(left_val)
                        result = eigvecs @ np.diag(eigvals**right_val) @ np.linalg.inv(eigvecs)
                        result = np.real(result)
                    else:
                        raise InterpreterError(
                            "Matrix power with non-integer exponent requires a square matrix"
                        )
                else:
                    result = left_val**right_val
            case ".*":
                result = left_val * right_val
            case "./":
                result = left_val / right_val
            case ".\\":
                result = right_val / left_val
            case ".^":
                result = left_val**right_val
            case "==":
                result = left_val == right_val
            case "~=":
                result = left_val != right_val
            case "<":
                result = left_val < right_val
            case ">":
                result = left_val > right_val
            case "<=":
                result = left_val <= right_val
            case ">=":
                result = left_val >= right_val
            case "&":
                result = (
                    np.logical_and(left_val, right_val)
                    if isinstance(left_val, np.ndarray)
                    else bool(bool(left_val) and bool(right_val))
                )
            case "|":
                result = (
                    np.logical_or(left_val, right_val)
                    if isinstance(left_val, np.ndarray)
                    else bool(bool(left_val) or bool(right_val))
                )
            case _:
                raise InterpreterError(f"Unknown operator: {op}")

        if isinstance(result, np.ndarray):
            return Mat(result)
        return result

    def _eval_unaryop(self, op: str, operand: Expr, env: Environment) -> Any:
        val = self._eval(operand, env)
        if isinstance(val, Mat):
            data = val.data
            if op == "+":
                return val
            elif op == "-":
                return Mat(-data)
            elif op == "~":
                return Mat(np.logical_not(data))
        else:
            if op == "+":
                return val
            elif op == "-":
                return -val
            elif op == "~":
                return not val
            else:
                raise InterpreterError(
                    f"Unary operator '{op}' not supported for {type(val).__name__}"
                )

    def _eval_range(self, expr: RangeExpr, env: Environment) -> Any:
        start = self._eval(expr.start, env)
        stop = self._eval(expr.stop, env)
        if isinstance(start, Mat):
            start = start.to_python()
        if isinstance(stop, Mat):
            stop = stop.to_python()

        if expr.step is not None:
            step = self._eval(expr.step, env)
            if isinstance(step, Mat):
                step = step.to_python()
            return Mat(np.arange(start, stop + step, step))
        else:
            return Mat(np.arange(start, stop + 1))

    def _eval_matrix(self, rows: list[list[Expr]], env: Environment) -> Mat:
        if not rows:
            return Mat(np.array([]))

        evaluated_rows = []
        for row in rows:
            vals = []
            for elem in row:
                v = self._eval(elem, env)
                if isinstance(v, Mat):
                    data = v.data
                    if data.ndim == 0:
                        vals.append(data.item())
                    elif data.ndim == 1:
                        vals.extend(data.tolist())
                    else:
                        vals.extend(data.flatten().tolist())
                elif isinstance(v, np.ndarray):
                    vals.extend(v.flatten().tolist())
                else:
                    vals.append(v)
            evaluated_rows.append(vals)

        max_cols = max(len(r) for r in evaluated_rows)
        for row in evaluated_rows:
            while len(row) < max_cols:
                row.append(0)

        return Mat(np.array(evaluated_rows))

    def _eval_cell(self, rows: list[list[Expr]], env: Environment) -> CellArray:
        data = []
        for row in rows:
            data.append([self._eval(elem, env) for elem in row])
        return CellArray(data)

    def _eval_index(self, expr: IndexExpr, env: Environment) -> Any:
        base = self._eval(expr.base, env)
        indices = [self._eval(idx, env) for idx in expr.indices]

        # Table indexing: T(rows, cols) or T{:, 'col'}
        if isinstance(base, Table):
            # Convert indices for table
            if len(indices) == 1:
                return base[indices[0]]
            elif len(indices) >= 2:
                return base[(indices[0], indices[1])]
            return base

        # Handle ":" as "all elements" in Mat indexing
        if isinstance(base, Mat):
            indices = [slice(None) if idx == ":" else idx for idx in indices]

        if isinstance(base, Mat):
            data = base.data
            resolved = []
            for i, idx in enumerate(indices):
                if isinstance(idx, str) and idx == "end_marker":
                    # MATLAB: A(end) with single index → numel(A) for multi-dim arrays
                    if len(indices) == 1 and data.ndim > 1:
                        resolved.append(data.size)
                    else:
                        dim = i if i < data.ndim else 0
                        resolved.append(data.shape[dim])
                else:
                    resolved.append(idx)

            converted = []
            for idx in resolved:
                if isinstance(idx, (int, float, np.integer, np.floating)):
                    converted.append(int(idx) - 1)
                elif isinstance(idx, Mat):
                    converted.append(idx.data - 1)
                elif isinstance(idx, np.ndarray):
                    if idx.dtype == bool:
                        converted.append(idx)
                    else:
                        converted.append(idx - 1)
                elif isinstance(idx, slice):
                    converted.append(idx)
                else:
                    converted.append(idx)

            try:
                result = data[tuple(converted)]
                if isinstance(result, np.ndarray):
                    if result.ndim == 0:
                        return result.item()
                    return Mat(result)
                return result
            except (IndexError, ValueError) as e:
                raise InterpreterError(f"Index error: {e}")

        elif isinstance(base, CellArray):
            if len(indices) >= 2:
                return base.get(int(indices[0]), int(indices[1]))
            elif len(indices) == 1:
                # Use linear indexing for single index
                return base.get_linear(int(indices[0]))
        elif isinstance(base, str):
            idx = int(indices[0]) - 1
            return base[idx]

        raise InterpreterError(f"Cannot index into {type(base).__name__}")

    def _eval_field_access(self, base: Expr, field: str, env: Environment) -> Any:
        obj = self._eval(base, env)
        if isinstance(obj, Struct):
            return obj.get_field(field)
        if isinstance(obj, MException):
            return obj.get_field(field)
        if isinstance(obj, Table):
            return obj.get_field(field)
        if isinstance(obj, ClassInstance):
            if obj.has_property(field):
                return obj.get_property(field)
            if obj.has_method(field):
                return obj.get_method(field)
            raise AttributeError(
                f"'{obj._class_def.name}' has no property or method '{field}'"
            )
        # Support Datetime, Duration, CalendarDuration, Categorical field access
        if hasattr(obj, "get_field"):
            try:
                return obj.get_field(field)
            except AttributeError:
                pass
        raise InterpreterError(f"Cannot access field '{field}' on {type(obj).__name__}")

    def _eval_func_call(self, name: str, args: list[Expr], env: Environment) -> Any:
        evaled_args = [self._eval(arg, env) for arg in args]

        # Special handling for eval — execute string as MATLAB code
        if name == "eval" and len(evaled_args) >= 1:
            code = str(evaled_args[0])
            return self._eval_string(code, env)

        # Special handling for feval — call function by name or handle
        if name == "feval" and len(evaled_args) >= 1:
            func_or_name = evaled_args[0]
            func_args = evaled_args[1:]
            # If it's a function handle, call it directly
            if isinstance(func_or_name, RTFuncHandle):
                return func_or_name(*func_args)
            # If it's a string, look up the function
            func_name = str(func_or_name)
            if func_name in self.functions:
                return self._call_user_func(self.functions[func_name], func_args, env)
            builtin = get_builtin(func_name)
            if builtin:
                return builtin(*func_args)
            raise InterpreterError(f"Undefined function '{func_name}'")

        # Priority 1: Check if this is a method call on a ClassInstance
        if len(args) > 0:
            first_arg = evaled_args[0]
            if isinstance(first_arg, ClassInstance):
                # Check for Static method first: ClassName.method(args)
                # Parser: A.b(c) → FuncCallExpr("b", [A, c])
                # A evaluates to ClassInstance (constructor called); look up "A.b"
                class_name = first_arg._class_def.name
                static_name = f"{class_name}.{name}"
                static_builtin = get_builtin(static_name)
                if static_builtin is not None:
                    return static_builtin(*evaled_args[1:])
                if first_arg.has_method(name):
                    method = first_arg.get_method(name)
                    if isinstance(method, FuncDef):
                        return self._call_user_func(
                            method, [first_arg] + evaled_args[1:], env
                        )
                if first_arg.has_property(name):
                    return first_arg.get_property(name)

        # Priority 2: Check if this is a method call on a Struct
        if len(args) > 0:
            first_arg = evaled_args[0]
            if isinstance(first_arg, Struct):
                field_val = first_arg.get_field(name)
                if len(args) > 1:
                    indices = evaled_args[1:]
                    if isinstance(field_val, Mat):
                        return self._eval_index_for_mat(field_val, indices)
                return field_val

        # Priority 3: Check user-defined functions
        if name in self.functions:
            func_def = self.functions[name]
            return self._call_user_func(func_def, evaled_args, env)

        # Priority 4: Check environment — if it's a variable, treat as indexing
        try:
            val = env.get(name)
            if isinstance(val, Mat):
                return self._eval_index_for_mat(val, evaled_args)
            if isinstance(val, Table):
                if len(evaled_args) == 1:
                    return val[evaled_args[0]]
                elif len(evaled_args) >= 2:
                    return val[(evaled_args[0], evaled_args[1])]
                return val
            if isinstance(val, RTFuncHandle):
                return val(*evaled_args)
            if isinstance(val, FuncDef):
                return self._call_user_func(val, evaled_args, env)
        except NameError:
            pass

        # Priority 5: Check built-in functions
        builtin = get_builtin(name)
        if builtin is not None:
            if name in ("whos", "who"):
                return builtin(env)
            return builtin(*evaled_args)

        # Priority 6: Check Python module functions (MEX interface)
        if "." in name:
            parts = name.split(".")
            if len(parts) == 2:
                module_name, func_name = parts
                # Whitelist of allowed modules for MEX interface
                _ALLOWED_MEX_MODULES = {
                    "numpy",
                    "np",
                    "scipy",
                    "matplotlib",
                    "math",
                    "random",
                    "statistics",
                    "itertools",
                    "functools",
                    "collections",
                    "re",
                    "json",
                    "csv",
                    "datetime",
                    "os.path",
                    "pathlib",
                }
                if module_name in _ALLOWED_MEX_MODULES:
                    try:
                        import importlib

                        module = importlib.import_module(module_name)
                        func = getattr(module, func_name)
                        if callable(func):
                            return func(*evaled_args)
                    except (ImportError, AttributeError):
                        pass

        raise InterpreterError(f"Undefined function '{name}'")

    def _eval_index_for_mat(self, mat: Mat, indices: list) -> Any:
        """Evaluate indexing on a Mat object with pre-evaluated indices."""
        data = mat.data
        resolved = []
        for i, idx in enumerate(indices):
            if isinstance(idx, str) and idx == "end_marker":
                # MATLAB: A(end) with single index → numel(A) for multi-dim arrays
                if len(indices) == 1 and data.ndim > 1:
                    resolved.append(data.size)
                else:
                    dim = i if i < data.ndim else 0
                    resolved.append(data.shape[dim])
            else:
                resolved.append(idx)

        # Handle logical indexing: A(boolean_mask)
        if len(resolved) == 1:
            idx = resolved[0]
            # Convert Mat to numpy array for indexing
            if isinstance(idx, Mat):
                idx = idx.data
            if isinstance(idx, np.ndarray) and idx.dtype == bool:
                # Logical indexing — return flattened result
                result = data[idx]
                if result.size == 0:
                    return Mat(np.array([]))
                return Mat(result)

        # Single index on multi-dim array → linear indexing (MATLAB convention)
        if len(resolved) == 1 and data.ndim > 1:
            idx = resolved[0]
            if isinstance(idx, (int, float, np.integer, np.floating)):
                linear_idx = int(idx) - 1
                try:
                    return data.flat[linear_idx]
                except IndexError as e:
                    raise InterpreterError(f"Index error: {e}")
            elif isinstance(idx, Mat):
                arr = idx.data - 1
                return Mat(data.flat[arr])
            elif isinstance(idx, np.ndarray):
                if idx.dtype == bool:
                    return Mat(data[idx])
                return Mat(data.flat[idx - 1])

        converted = []
        for idx in resolved:
            if isinstance(idx, (int, float, np.integer, np.floating)):
                converted.append(int(idx) - 1)
            elif isinstance(idx, Mat):
                arr = idx.data
                if arr.dtype == bool:
                    converted.append(arr)
                else:
                    converted.append(arr - 1)
            elif isinstance(idx, np.ndarray):
                if idx.dtype == bool:
                    converted.append(idx)
                else:
                    converted.append(idx - 1)
            elif isinstance(idx, slice):
                converted.append(idx)
            else:
                converted.append(idx)

        try:
            result = data[tuple(converted)]
            if isinstance(result, np.ndarray):
                if result.ndim == 0:
                    return result.item()
                return Mat(result)
            return result
        except (IndexError, ValueError) as e:
            raise InterpreterError(f"Index error: {e}")

    def _call_user_func(self, func_def: FuncDef, args: list, env: Environment) -> Any:
        from matpy.runtime.types import CellArray

        # Support nested functions with closure environment
        if hasattr(func_def, "_closure_env") and func_def._closure_env is not None:
            # Nested function: use closure environment as parent
            func_env = func_def._closure_env.child(f"func:{func_def.name}")
        else:
            # Regular function: use calling environment as parent
            func_env = env.child(f"func:{func_def.name}")

        # Pre-register nested functions (MATLAB behavior: nested functions are visible throughout)
        for stmt in func_def.body:
            if isinstance(stmt, FuncDef):
                stmt._closure_env = func_env
                func_env.set(stmt.name, stmt)

        # Handle varargin
        has_varargin = "varargin" in func_def.params
        if has_varargin:
            regular_params = [p for p in func_def.params if p != "varargin"]
            for i, param in enumerate(regular_params):
                if i < len(args):
                    func_env.set(param, args[i])
                else:
                    func_env.set(param, None)
            varargin_args = args[len(regular_params) :]
            varargin_cell = CellArray([[arg] for arg in varargin_args])
            func_env.set("varargin", varargin_cell)
        else:
            for i, param in enumerate(func_def.params):
                if i < len(args):
                    func_env.set(param, args[i])
                else:
                    func_env.set(param, None)

        # Set nargin and nargout
        func_env.set("nargin", len(args))
        func_env.set("nargout", len(func_def.returns))

        # Validate arguments if specs exist
        if func_def.arg_specs:
            self._validate_arguments(func_def.arg_specs, func_env)

        # If this is a class constructor, pre-initialize the output variable
        if func_def.returns and len(func_def.returns) == 1:
            ret_name = func_def.returns[0]
            if ret_name not in func_def.params:
                class_name = func_def.name
                if class_name in self.classes:
                    from matpy.runtime.types import ClassInstance

                    func_env.set(ret_name, ClassInstance(self.classes[class_name]))

        try:
            self._exec_block(func_def.body, func_env)
            # MATLAB: return the output variable(s) implicitly
            if func_def.returns:
                # Handle varargout
                if "varargout" in func_def.returns:
                    regular_returns = [r for r in func_def.returns if r != "varargout"]
                    results = []
                    for r in regular_returns:
                        try:
                            results.append(func_env.get(r))
                        except NameError:
                            results.append(None)
                    # Get varargout cell array
                    try:
                        varargout = func_env.get("varargout")
                        if isinstance(varargout, CellArray):
                            for row in varargout._data:
                                for val in row:
                                    results.append(val)
                    except NameError:
                        pass
                    return (
                        results if len(results) > 1 else results[0] if results else None
                    )
                elif len(func_def.returns) == 1:
                    return func_env.get(func_def.returns[0])
                else:
                    return [func_env.get(r) for r in func_def.returns]
            return None
        except ReturnSignal as ret:
            return ret.value

    def _eval_func_handle(self, name: str | None, env: Environment) -> Any:
        if name is None:
            return RTFuncHandle(lambda: None)
        if name in self.functions:
            func_def = self.functions[name]

            def caller(*args):
                return self._call_user_func(func_def, list(args), env)

            return RTFuncHandle(caller, name)
        builtin = get_builtin(name)
        if builtin is not None:
            return RTFuncHandle(builtin, name)
        raise InterpreterError(f"Cannot create handle for undefined function '{name}'")

    def _eval_anon_func(self, params: list[str], body: Expr, env: Environment) -> Any:
        closure_env = env

        def caller(*args):
            child = closure_env.child("anon")
            for i, param in enumerate(params):
                child.set(param, args[i] if i < len(args) else None)
            return self._eval(body, child)

        return RTFuncHandle(caller, "<anonymous>")

    def _eval_string(self, code: str, env: Environment) -> Any:
        """Execute MATLAB code from a string in the given environment."""
        try:
            from matpy.lexer import Lexer
            from matpy.parser import Parser

            lexer = Lexer(code, "<eval>")
            tokens = lexer.tokenize()
            parser = Parser(tokens, "<eval>")
            program = parser.parse()
            result = None
            for stmt in program.statements:
                self._exec_stmt(stmt, env)
                if isinstance(stmt, ExprStmt):
                    result = self._eval(stmt.expr, env)
            return result
        except Exception as e:
            raise InterpreterError(f"eval error: {e}")

    def _load_package(self, package_name: str, env: Environment):
        """Load a MATLAB package from +package directory."""
        import os

        package_dir = os.path.join(os.getcwd(), f"+{package_name}")
        if not os.path.isdir(package_dir):
            raise InterpreterError(f"Package '{package_name}' not found")

        # Load all .m files in the package directory
        for filename in os.listdir(package_dir):
            if filename.endswith(".m"):
                filepath = os.path.join(package_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    code = f.read()
                try:
                    from matpy.lexer import Lexer
                    from matpy.parser import Parser

                    lexer = Lexer(code, filepath)
                    tokens = lexer.tokenize()
                    parser = Parser(tokens, filepath)
                    program = parser.parse()
                    # Register functions with package prefix
                    for stmt in program.statements:
                        if isinstance(stmt, FuncDef):
                            qualified_name = f"{package_name}.{stmt.name}"
                            self.functions[qualified_name] = stmt
                except Exception as e:
                    print(f"Warning: Could not load {filepath}: {e}")

    def _exec_parfor(self, stmt: ForStmt, env: Environment):
        """Execute parfor loop using multiprocessing."""
        import multiprocessing as mp

        iter_val = self._eval(stmt.iter_expr, env)
        if isinstance(iter_val, Mat):
            items = list(iter_val.data.flat)
        elif isinstance(iter_val, np.ndarray):
            items = list(iter_val.flat)
        elif isinstance(iter_val, (list, tuple)):
            items = list(iter_val)
        else:
            items = [iter_val]

        def execute_iteration(item):
            """Execute a single parfor iteration."""
            child_env = env.child(f"parfor:{stmt.var}")
            child_env.set(stmt.var, item)
            try:
                self._exec_block(stmt.body, child_env)
            except (BreakSignal, ContinueSignal, ReturnSignal):
                pass
            return child_env

        # Use multiprocessing Pool for parallel execution
        with mp.Pool() as pool:
            results = pool.map(execute_iteration, items)

        # Merge results back to parent environment
        for result_env in results:
            for key, value in result_env._variables.items():
                if key != stmt.var:
                    env.set(key, value)

    def _is_truthy(self, val: Any) -> bool:
        if isinstance(val, Mat):
            data = val.data
            if data.size == 1:
                return bool(data.flat[0])
            return bool(np.all(data))
        if isinstance(val, np.ndarray):
            if val.size == 1:
                return bool(val.flat[0])
            return bool(np.all(val))
        return bool(val)

    def _check_jit_feasibility(self, func_def: FuncDef) -> dict:
        """Check if a function is suitable for JIT compilation."""
        analysis = {
            "suitable": True,
            "reasons": [],
            "warnings": [],
        }

        # Check for unsupported features
        for stmt in func_def.body:
            if isinstance(stmt, TryCatchStmt):
                analysis["suitable"] = False
                analysis["reasons"].append("try/catch not supported in JIT")
            if isinstance(stmt, ClassDef):
                analysis["suitable"] = False
                analysis["reasons"].append("classdef not supported in JIT")

        return analysis

    def _is_equal(self, a: Any, b: Any) -> bool:
        if isinstance(a, Mat):
            a = a.data
        if isinstance(b, Mat):
            b = b.data
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return np.array_equal(a, b)
        result = a == b
        # Handle case where comparison returns numpy array instead of scalar bool
        if isinstance(result, np.ndarray):
            return bool(np.all(result))
        return bool(result)

    def _validate_arguments(
        self, arg_specs: dict[str, dict[str, Any]], env: Environment
    ):
        """Validate function arguments against specs from arguments block."""
        BUILT_IN_VALIDATORS = {
            "mustBeNumeric": lambda v: (
                isinstance(v, (int, float, complex, Mat))
                or (isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number))
            ),
            "mustBePositive": lambda v: isinstance(v, (int, float)) and v > 0,
            "mustBeNonnegative": lambda v: isinstance(v, (int, float)) and v >= 0,
            "mustBeNonempty": lambda v: v is not None,
            "mustBeNonNan": lambda v: not (isinstance(v, float) and np.isnan(v)),
            "mustBeFinite": lambda v: (
                not (isinstance(v, float) and (np.isinf(v) or np.isnan(v)))
            ),
            "mustBeInteger": lambda v: isinstance(v, (int, np.integer)),
            "mustBeText": lambda v: isinstance(v, str),
            "mustBeMember": lambda v: True,  # placeholder
        }

        for param_name, spec in arg_specs.items():
            try:
                val = env.get(param_name)
            except NameError:
                val = None

            # Apply default if value is None and default exists
            if val is None and "default" in spec:
                default_val = spec["default"]
                # Convert string/number literals
                if isinstance(default_val, str):
                    try:
                        default_val = float(default_val)
                    except ValueError:
                        pass
                env.set(param_name, default_val)
                val = default_val

            if val is None and "default" not in spec:
                continue

            # Type check
            if "type" in spec:
                expected_type = spec["type"]
                type_map = {
                    "double": (int, float, Mat),
                    "char": str,
                    "string": str,
                    "logical": bool,
                    "int32": int,
                    "int64": int,
                }
                if expected_type in type_map:
                    if not isinstance(val, type_map[expected_type]):
                        try:
                            if expected_type == "double":
                                val = float(val) if not isinstance(val, Mat) else val
                            elif expected_type == "char" or expected_type == "string":
                                val = str(val)
                        except (ValueError, TypeError):
                            raise MatPyTypeError(
                                f"Argument '{param_name}' must be of type '{expected_type}'"
                            )
                        env.set(param_name, val)

            # Validation functions
            if "validation" in spec:
                for validator_name in spec["validation"]:
                    if validator_name in BUILT_IN_VALIDATORS:
                        validator = BUILT_IN_VALIDATORS[validator_name]
                        if not validator(val):
                            raise MatPyValueError(
                                f"Argument '{param_name}' failed validation '{validator_name}'"
                            )
