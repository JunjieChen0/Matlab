"""Tree-walking interpreter for MatPy."""

from __future__ import annotations
from typing import Any
import numpy as np

from matpy.ast_nodes import (
    Program, NumberLiteral, StringLiteral, Identifier, BinaryOp, UnaryOp,
    RangeExpr, MatrixLiteral, CellLiteral, IndexExpr, FieldAccess,
    DynamicFieldAccess, FuncCallExpr, ParenExpr, FuncHandle, AnonFuncExpr,
    ExprStmt, Assignment, IfStmt, ForStmt, WhileStmt, SwitchStmt,
    TryCatchStmt, ReturnStmt, BreakStmt, ContinueStmt, FuncDef,
    GlobalStmt, PersistentStmt, ClassDef, Expr, Stmt, NodeVisitor,
)
from matpy.environment import Environment
from matpy.runtime.types import Mat, CellArray, Struct, FuncHandle as RTFuncHandle, ClassInstance
from matpy.runtime.matrix import to_mat, from_mat, mat_str
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
    def __init__(self, message: str, line: int = 0, col: int = 0, filename: str = "<unknown>"):
        self.line = line
        self.col = col
        self.filename = filename
        if line > 0:
            super().__init__(f"{filename}:{line}:{col}: {message}")
        else:
            super().__init__(message)


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
                    if isinstance(converted[0], np.ndarray) and converted[0].dtype == bool:
                        data[converted[0]] = value
                    else:
                        data.flat[converted[0]] = value
                elif len(converted) == 2:
                    data[converted[0], converted[1]] = value
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
            if data.ndim == 1:
                items = list(data)
            elif data.ndim == 2:
                items = [Mat(data[:, i]) for i in range(data.shape[1])]
            else:
                items = list(data.flat)
        elif isinstance(iter_val, np.ndarray):
            items = list(iter_val.flat)
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
            if self._is_equal(val, case_val):
                self._exec_block(body, env)  # Use same scope
                return
        if stmt.otherwise:
            self._exec_block(stmt.otherwise, env)

    def _exec_try(self, stmt: TryCatchStmt, env: Environment):
        try:
            self._exec_block(stmt.try_body, env.child("try"))
        except Exception as e:
            child = env.child("catch")
            if stmt.catch_var:
                child.set(stmt.catch_var, str(e))
            self._exec_block(stmt.catch_body, child)

    def _exec_classdef(self, stmt: ClassDef, env: Environment):
        """Execute a classdef block — register the class and its constructor."""
        from matpy.runtime.types import ClassDefRuntime, ClassInstance

        class_runtime = ClassDefRuntime(stmt.name, stmt.superclass)

        # Inherit from superclass if specified
        if stmt.superclass and stmt.superclass in self.classes:
            super_cls = self.classes[stmt.superclass]
            # Inherit properties
            for prop_name, default_val in super_cls.properties.items():
                if prop_name not in stmt.properties:
                    class_runtime.properties[prop_name] = default_val
                    class_runtime.property_attrs[prop_name] = super_cls.property_attrs.get(prop_name, {})
            # Inherit methods
            for meth_name, meth_def in super_cls.methods.items():
                if meth_name not in stmt.methods:
                    class_runtime.methods[meth_name] = meth_def
                    class_runtime.method_attrs[meth_name] = super_cls.method_attrs.get(meth_name, {})

        # Evaluate property defaults
        for prop_name, default_expr in stmt.properties.items():
            if default_expr is not None:
                class_runtime.properties[prop_name] = self._eval(default_expr, env)
            else:
                class_runtime.properties[prop_name] = None
            class_runtime.property_attrs[prop_name] = stmt.property_attrs.get(prop_name, {})

        # Register methods
        for meth_name, meth_def in stmt.methods.items():
            class_runtime.methods[meth_name] = meth_def
            class_runtime.method_attrs[meth_name] = stmt.method_attrs.get(meth_name, {})

        # Register the class name as a constructor function
        def constructor(*args):
            instance = ClassInstance(class_runtime)
            # Call constructor method if it exists
            if stmt.name in class_runtime.methods:
                result = self._call_user_func(class_runtime.methods[stmt.name], list(args), env)
                if result is not None and isinstance(result, ClassInstance):
                    return result
            return instance

        # Store class definition
        self.classes[stmt.name] = class_runtime

        # Register constructor as a built-in function
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
                raise InterpreterError(f"Unknown expression type: {type(expr).__name__}")

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
        l = self._eval(left, env)
        r = self._eval(right, env)

        if isinstance(l, Mat):
            l = l.data
        if isinstance(r, Mat):
            r = r.data

        match op:
            case "+":
                result = l + r
            case "-":
                result = l - r
            case "*":
                if isinstance(l, np.ndarray) and isinstance(r, np.ndarray):
                    result = l @ r
                else:
                    result = l * r
            case "/":
                if isinstance(l, np.ndarray) and isinstance(r, np.ndarray):
                    result = np.linalg.solve(r.T, l.T).T
                else:
                    result = l / r
            case "\\":
                result = np.linalg.solve(l, r) if isinstance(l, np.ndarray) else r / l
            case "^":
                if isinstance(l, np.ndarray):
                    result = np.linalg.matrix_power(l, int(r))
                else:
                    result = l ** r
            case ".*":
                result = l * r
            case "./":
                result = l / r
            case ".\\":
                result = r / l
            case ".^":
                result = l ** r
            case "==":
                result = l == r
            case "~=":
                result = l != r
            case "<":
                result = l < r
            case ">":
                result = l > r
            case "<=":
                result = l <= r
            case ">=":
                result = l >= r
            case "&":
                result = np.logical_and(l, r) if isinstance(l, np.ndarray) else (l and r)
            case "|":
                result = np.logical_or(l, r) if isinstance(l, np.ndarray) else (l or r)
            case "&&":
                result = self._is_truthy(l) and self._is_truthy(r)
            case "||":
                result = self._is_truthy(l) or self._is_truthy(r)
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

        if isinstance(base, Mat):
            data = base.data
            resolved = []
            for i, idx in enumerate(indices):
                if isinstance(idx, str) and idx == "end_marker":
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
                return base.get(int(indices[0]))
        elif isinstance(base, str):
            idx = int(indices[0]) - 1
            return base[idx]

        raise InterpreterError(f"Cannot index into {type(base).__name__}")

    def _eval_field_access(self, base: Expr, field: str, env: Environment) -> Any:
        obj = self._eval(base, env)
        if isinstance(obj, Struct):
            return obj.get_field(field)
        if isinstance(obj, ClassInstance):
            if obj.has_property(field):
                return obj.get_property(field)
            if obj.has_method(field):
                return obj.get_method(field)
            raise AttributeError(f"'{obj._class_def.name}' has no property or method '{field}'")
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
                if first_arg.has_method(name):
                    method = first_arg.get_method(name)
                    if isinstance(method, FuncDef):
                        return self._call_user_func(method, [first_arg] + evaled_args[1:], env)
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

        raise InterpreterError(f"Undefined function '{name}'")

    def _eval_index_for_mat(self, mat: Mat, indices: list) -> Any:
        """Evaluate indexing on a Mat object with pre-evaluated indices."""
        data = mat.data
        resolved = []
        for i, idx in enumerate(indices):
            if isinstance(idx, str) and idx == "end_marker":
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
        func_env = env.child(f"func:{func_def.name}")
        for i, param in enumerate(func_def.params):
            if i < len(args):
                func_env.set(param, args[i])
            else:
                func_env.set(param, None)

        # If this is a class constructor, pre-initialize the output variable
        if func_def.returns and len(func_def.returns) == 1:
            ret_name = func_def.returns[0]
            # Check if the return variable is used in the body (constructor pattern)
            if ret_name not in func_def.params:
                # Pre-initialize as ClassInstance if it's a constructor
                class_name = func_def.name
                if class_name in self.classes:
                    from matpy.runtime.types import ClassInstance
                    func_env.set(ret_name, ClassInstance(self.classes[class_name]))

        try:
            self._exec_block(func_def.body, func_env)
            # MATLAB: return the output variable(s) implicitly
            if func_def.returns:
                if len(func_def.returns) == 1:
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
            return result
        except Exception as e:
            raise InterpreterError(f"eval error: {e}")

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

    def _is_equal(self, a: Any, b: Any) -> bool:
        if isinstance(a, Mat):
            a = a.data
        if isinstance(b, Mat):
            b = b.data
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return np.array_equal(a, b)
        return a == b
