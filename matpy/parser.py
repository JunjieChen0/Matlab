"""Recursive descent parser for MatPy."""

from typing import Any
from matpy.tokens import Token, TokenType
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
)


class ParseError(Exception):
    def __init__(self, message: str, token: Token):
        super().__init__(f"Line {token.line}, Col {token.col}: {message}")
        self.token = token


class Parser:
    def __init__(self, tokens: list[Token], filename: str = "<stdin>"):
        self.tokens = tokens
        self.filename = filename
        self.pos = 0

    # ── helpers ──────────────────────────────────────────────────

    def _peek(self) -> Token:
        if self.pos >= len(self.tokens):
            return Token(TokenType.EOF, None, 0, 0)
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        if self.pos >= len(self.tokens):
            return Token(TokenType.EOF, None, 0, 0)
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _check(self, ttype: TokenType) -> bool:
        return self._peek().type == ttype

    def _match(self, ttype: TokenType) -> Token | None:
        if self._check(ttype):
            return self._advance()
        return None

    def _expect(self, ttype: TokenType) -> Token:
        tok = self._peek()
        if tok.type != ttype:
            raise ParseError(
                f"Expected {ttype.name}, got {tok.type.name} ({tok.value!r})",
                tok,
            )
        return self._advance()

    def _skip_newlines(self):
        while self._check(TokenType.NEWLINE):
            self._advance()

    def _consume_stmt_end(self):
        self._match(TokenType.SEMICOLON)
        self._skip_newlines()

    # ── main entry ───────────────────────────────────────────────

    def parse(self) -> Program:
        stmts: list[Stmt] = []
        self._skip_newlines()
        while not self._check(TokenType.EOF):
            stmts.extend(self._parse_statement())
            self._skip_newlines()
        return Program(stmts)

    # ── statement parsing ────────────────────────────────────────

    def _parse_statement(self) -> list[Stmt]:
        tok = self._peek()
        match tok.type:
            case TokenType.FUNCTION:
                return [self._parse_func_def()]
            case TokenType.IF:
                return [self._parse_if()]
            case TokenType.FOR:
                return [self._parse_for()]
            case TokenType.WHILE:
                return [self._parse_while()]
            case TokenType.SWITCH:
                return [self._parse_switch()]
            case TokenType.TRY:
                return [self._parse_try()]
            case TokenType.RETURN:
                self._advance()
                self._consume_stmt_end()
                return [ReturnStmt()]
            case TokenType.BREAK:
                self._advance()
                self._consume_stmt_end()
                return [BreakStmt()]
            case TokenType.CONTINUE:
                self._advance()
                self._consume_stmt_end()
                return [ContinueStmt()]
            case TokenType.GLOBAL:
                return [self._parse_global()]
            case TokenType.PERSISTENT:
                return [self._parse_persistent()]
            case TokenType.CLASSDEF:
                return [self._parse_classdef()]
            case _:
                return self._parse_assign_or_expr()

    def _parse_func_def(self) -> FuncDef:
        self._expect(TokenType.FUNCTION)
        self._skip_newlines()

        returns: list[str] = []
        if self._match(TokenType.LBRACKET):
            while not self._check(TokenType.RBRACKET):
                returns.append(self._expect(TokenType.IDENTIFIER).value)
                self._match(TokenType.COMMA)
            self._expect(TokenType.RBRACKET)
            self._expect(TokenType.ASSIGN)
        elif (
            self._check(TokenType.IDENTIFIER)
            and self.pos + 1 < len(self.tokens)
            and self.tokens[self.pos + 1].type == TokenType.ASSIGN
        ):
            returns.append(self._expect(TokenType.IDENTIFIER).value)
            self._expect(TokenType.ASSIGN)

        name = self._expect(TokenType.IDENTIFIER).value

        params: list[str] = []
        self._expect(TokenType.LPAREN)
        while not self._check(TokenType.RPAREN):
            params.append(self._expect(TokenType.IDENTIFIER).value)
            self._match(TokenType.COMMA)
        self._expect(TokenType.RPAREN)
        self._consume_stmt_end()

        # Parse optional arguments block
        arg_specs: dict[str, dict[str, Any]] = {}
        if self._check(TokenType.ARGUMENTS):
            arg_specs = self._parse_arguments_block()

        body = self._parse_block_until_end()
        self._expect(TokenType.END)
        self._consume_stmt_end()
        return FuncDef(
            name=name, params=params, returns=returns, body=body, arg_specs=arg_specs
        )

    def _parse_if(self) -> IfStmt:
        self._expect(TokenType.IF)
        condition = self._parse_expr()
        self._consume_stmt_end()

        body = self._parse_block_until_branch()
        elif_branches: list[tuple[Expr, list[Stmt]]] = []
        else_body: list[Stmt] = []

        while self._check(TokenType.ELSEIF):
            self._advance()
            cond = self._parse_expr()
            self._consume_stmt_end()
            b = self._parse_block_until_branch()
            elif_branches.append((cond, b))

        if self._match(TokenType.ELSE):
            self._consume_stmt_end()
            else_body = self._parse_block_until_end()

        self._expect(TokenType.END)
        self._consume_stmt_end()
        return IfStmt(
            condition=condition,
            body=body,
            elif_branches=elif_branches,
            else_body=else_body,
        )

    def _parse_for(self) -> ForStmt:
        self._expect(TokenType.FOR)
        var = self._expect(TokenType.IDENTIFIER).value
        self._expect(TokenType.ASSIGN)
        iter_expr = self._parse_expr()
        self._consume_stmt_end()

        body = self._parse_block_until_end()
        self._expect(TokenType.END)
        self._consume_stmt_end()
        return ForStmt(var=var, iter_expr=iter_expr, body=body)

    def _parse_while(self) -> WhileStmt:
        self._expect(TokenType.WHILE)
        condition = self._parse_expr()
        self._consume_stmt_end()

        body = self._parse_block_until_end()
        self._expect(TokenType.END)
        self._consume_stmt_end()
        return WhileStmt(condition=condition, body=body)

    def _parse_switch(self) -> SwitchStmt:
        self._expect(TokenType.SWITCH)
        expr = self._parse_expr()
        self._consume_stmt_end()

        cases: list[tuple[Expr, list[Stmt]]] = []
        otherwise: list[Stmt] = []

        while not self._check(TokenType.END):
            if self._check(TokenType.CASE):
                self._advance()
                case_expr = self._parse_expr()
                self._consume_stmt_end()
                case_body = self._parse_block_until_branch()
                cases.append((case_expr, case_body))
            elif self._check(TokenType.OTHERWISE):
                self._advance()
                self._consume_stmt_end()
                otherwise = self._parse_block_until_end()
            else:
                break

        self._expect(TokenType.END)
        self._consume_stmt_end()
        return SwitchStmt(expr=expr, cases=cases, otherwise=otherwise)

    def _parse_try(self) -> TryCatchStmt:
        self._expect(TokenType.TRY)
        self._consume_stmt_end()

        try_body = self._parse_block_until_branch()
        catch_var: str | None = None
        catch_body: list[Stmt] = []

        if self._match(TokenType.CATCH):
            if self._check(TokenType.IDENTIFIER):
                catch_var = self._advance().value
            self._consume_stmt_end()
            catch_body = self._parse_block_until_end()

        self._expect(TokenType.END)
        self._consume_stmt_end()
        return TryCatchStmt(
            try_body=try_body, catch_var=catch_var, catch_body=catch_body
        )

    def _parse_global(self) -> GlobalStmt:
        self._expect(TokenType.GLOBAL)
        names: list[str] = []
        names.append(self._expect(TokenType.IDENTIFIER).value)
        while self._match(TokenType.COMMA):
            names.append(self._expect(TokenType.IDENTIFIER).value)
        self._consume_stmt_end()
        return GlobalStmt(names=names)

    def _parse_persistent(self) -> PersistentStmt:
        self._expect(TokenType.PERSISTENT)
        names: list[str] = []
        names.append(self._expect(TokenType.IDENTIFIER).value)
        while self._match(TokenType.COMMA):
            names.append(self._expect(TokenType.IDENTIFIER).value)
        self._consume_stmt_end()
        return PersistentStmt(names=names)

    def _parse_classdef(self) -> ClassDef:
        """Parse: classdef (Attrs) ClassName < SuperClass ... end"""
        self._expect(TokenType.CLASSDEF)

        # Optional class attributes: (Abstract, Sealed, Access=private)
        class_attrs = self._parse_attr_list()

        name = self._expect(TokenType.IDENTIFIER).value

        # Optional superclass
        superclass = None
        if self._match(TokenType.LT):
            superclass = self._expect(TokenType.IDENTIFIER).value

        self._consume_stmt_end()

        # Parse class body
        properties: dict[str, Expr | None] = {}
        property_attrs: dict[str, dict[str, Any]] = {}
        methods: dict[str, FuncDef] = {}
        method_attrs: dict[str, dict[str, Any]] = {}

        self._skip_newlines()
        while not self._check(TokenType.END) and not self._check(TokenType.EOF):
            if self._check(TokenType.PROPERTIES):
                self._advance()
                block_attrs = self._parse_attr_list()
                self._consume_stmt_end()
                self._parse_properties_block(properties, property_attrs, block_attrs)
            elif self._check(TokenType.METHODS):
                self._advance()
                block_attrs = self._parse_attr_list()
                self._consume_stmt_end()
                self._parse_methods_block(methods, method_attrs, block_attrs)
            elif self._check(TokenType.FUNCTION):
                # Top-level function in classdef
                func = self._parse_func_def()
                methods[func.name] = func
            else:
                self._advance()  # skip unknown tokens

        self._expect(TokenType.END)
        self._consume_stmt_end()
        return ClassDef(
            name=name,
            superclass=superclass,
            class_attrs=class_attrs,
            properties=properties,
            property_attrs=property_attrs,
            methods=methods,
            method_attrs=method_attrs,
        )

    def _parse_attr_list(self) -> dict[str, Any]:
        """Parse optional attribute list: (Key1, Key2=val, ...).
        Returns dict of attributes. Empty dict if no parenthesized list."""
        attrs: dict[str, Any] = {}
        if not self._check(TokenType.LPAREN):
            return attrs
        self._advance()  # consume '('
        while not self._check(TokenType.RPAREN) and not self._check(TokenType.EOF):
            if self._check(TokenType.IDENTIFIER):
                key = self._advance().value
                if self._match(TokenType.ASSIGN):
                    # Parse value: could be identifier, string, or number
                    if self._check(TokenType.IDENTIFIER):
                        val = self._advance().value
                    elif self._check(TokenType.STRING):
                        val = self._advance().value
                    elif self._check(TokenType.NUMBER):
                        val = self._advance().value
                    else:
                        val = self._advance().value
                    attrs[key] = val
                else:
                    attrs[key] = True
                self._match(TokenType.COMMA)  # optional comma separator
            else:
                self._advance()
        self._expect(TokenType.RPAREN)
        return attrs

    def _parse_arguments_block(self) -> dict[str, dict[str, Any]]:
        """Parse arguments block:
            arguments
                x double {mustBeNumeric}
                y (1,1) double = 0
                opts.Method char = 'default'
            end
        Returns dict of param_name -> {type, size, validation, default}.
        """
        specs: dict[str, dict[str, Any]] = {}
        self._expect(TokenType.ARGUMENTS)
        self._consume_stmt_end()
        self._skip_newlines()

        while not self._check(TokenType.END) and not self._check(TokenType.EOF):
            if self._check(TokenType.IDENTIFIER):
                param_name = self._advance().value
                spec: dict[str, Any] = {}

                # Optional size spec: (1,1), (:,1), etc.
                if self._check(TokenType.LPAREN):
                    self._advance()  # consume '('
                    size_parts = []
                    while not self._check(TokenType.RPAREN) and not self._check(
                        TokenType.EOF
                    ):
                        size_parts.append(self._advance().value)
                        self._match(TokenType.COMMA)
                    self._expect(TokenType.RPAREN)
                    spec["size"] = size_parts

                # Optional type annotation
                if self._check(TokenType.IDENTIFIER):
                    spec["type"] = self._advance().value

                # Optional validation: {mustBeNumeric, mustBePositive}
                if self._check(TokenType.LBRACE):
                    self._advance()  # consume '{'
                    validators = []
                    while not self._check(TokenType.RBRACE) and not self._check(
                        TokenType.EOF
                    ):
                        validators.append(self._advance().value)
                        self._match(TokenType.COMMA)
                    self._expect(TokenType.RBRACE)
                    spec["validation"] = validators

                # Optional default value
                if self._match(TokenType.ASSIGN):
                    if self._check(TokenType.NUMBER):
                        spec["default"] = self._advance().value
                    elif self._check(TokenType.STRING):
                        spec["default"] = self._advance().value
                    elif self._check(TokenType.IDENTIFIER):
                        spec["default"] = self._advance().value
                    else:
                        spec["default"] = None

                self._consume_stmt_end()
                specs[param_name] = spec
            else:
                self._advance()

        self._expect(TokenType.END)
        self._consume_stmt_end()
        return specs

    def _parse_properties_block(
        self, props: dict, attrs: dict, block_attrs: dict | None = None
    ):
        """Parse properties block."""
        self._skip_newlines()
        while not self._check(TokenType.END) and not self._check(TokenType.EOF):
            if self._check(TokenType.IDENTIFIER):
                prop_name = self._advance().value
                default_val = None
                if self._match(TokenType.ASSIGN):
                    default_val = self._parse_expr()
                self._consume_stmt_end()
                props[prop_name] = default_val
                # Merge block-level attrs with per-property attrs
                prop_attr = dict(block_attrs) if block_attrs else {}
                attrs[prop_name] = prop_attr
            else:
                self._advance()
        self._expect(TokenType.END)
        self._consume_stmt_end()

    def _parse_methods_block(
        self, meths: dict, attrs: dict, block_attrs: dict | None = None
    ):
        """Parse methods block."""
        self._skip_newlines()
        while not self._check(TokenType.END) and not self._check(TokenType.EOF):
            if self._check(TokenType.FUNCTION):
                func = self._parse_func_def()
                meths[func.name] = func
                # Merge block-level attrs with per-method attrs
                meth_attr = dict(block_attrs) if block_attrs else {}
                attrs[func.name] = meth_attr
            else:
                self._advance()
        self._expect(TokenType.END)
        self._consume_stmt_end()

    def _parse_block_until_end(self) -> list[Stmt]:
        stmts: list[Stmt] = []
        self._skip_newlines()
        while not self._check(TokenType.END) and not self._check(TokenType.EOF):
            stmts.extend(self._parse_statement())
            self._skip_newlines()
        return stmts

    def _parse_block_until_branch(self) -> list[Stmt]:
        stop = {
            TokenType.END,
            TokenType.ELSEIF,
            TokenType.ELSE,
            TokenType.OTHERWISE,
            TokenType.CATCH,
            TokenType.CASE,
        }
        stmts: list[Stmt] = []
        self._skip_newlines()
        while self._peek().type not in stop and not self._check(TokenType.EOF):
            stmts.extend(self._parse_statement())
            self._skip_newlines()
        return stmts

    def _parse_assign_or_expr(self) -> list[Stmt]:
        # Check for multi-return assignment: [a, b] = expr
        if self._check(TokenType.LBRACKET):
            save = self.pos
            self._advance()

            # Check if this is actually a matrix literal (starts with number)
            if self._check(TokenType.NUMBER):
                self.pos = save
                expr = self._parse_expr()
                self._consume_stmt_end()
                return [ExprStmt(expr=expr)]

            targets: list[Expr] = []
            while not self._check(TokenType.RBRACKET) and not self._check(
                TokenType.EOF
            ):
                if self._check(TokenType.TILDE):
                    self._advance()
                    targets.append(Identifier("~"))
                else:
                    targets.append(Identifier(self._expect(TokenType.IDENTIFIER).value))
                if not self._check(TokenType.RBRACKET):
                    self._match(TokenType.COMMA)
            self._expect(TokenType.RBRACKET)

            if self._match(TokenType.ASSIGN):
                value = self._parse_expr()
                self._consume_stmt_end()
                return [Assignment(targets=targets, value=value)]

            self.pos = save

        # Check for simple assignment: x = expr
        if (
            self._check(TokenType.IDENTIFIER)
            and self.pos + 1 < len(self.tokens)
            and self.tokens[self.pos + 1].type == TokenType.ASSIGN
        ):
            target = Identifier(self._advance().value)
            self._advance()  # consume =
            value = self._parse_expr()
            self._consume_stmt_end()
            return [Assignment(targets=[target], value=value)]

        # Check for command-style syntax: func arg1 arg2
        # In MATLAB, ANY function can use command syntax: `func arg1 arg2` ≡ `func('arg1', 'arg2')`
        # Triggered when an identifier is followed by another identifier/number/string/keyword
        # (but NOT by '=' or '(' which indicate assignment or function call)
        _CMD_TRIGGER_TYPES = (
            {
                TokenType.IDENTIFIER,
                TokenType.STRING,
                TokenType.NUMBER,
                TokenType.FUNCTION,
                TokenType.END,
                TokenType.IF,
                TokenType.ELSEIF,
                TokenType.ELSE,
                TokenType.FOR,
                TokenType.WHILE,
                TokenType.SWITCH,
                TokenType.CASE,
                TokenType.OTHERWISE,
                TokenType.TRY,
                TokenType.CATCH,
                TokenType.RETURN,
                TokenType.BREAK,
                TokenType.CONTINUE,
                TokenType.TRUE,
                TokenType.FALSE,
                TokenType.GLOBAL,
                TokenType.PERSISTENT,
            }
            if hasattr(TokenType, "TRUE")
            else {
                TokenType.IDENTIFIER,
                TokenType.STRING,
                TokenType.NUMBER,
                TokenType.FUNCTION,
                TokenType.END,
                TokenType.IF,
                TokenType.ELSEIF,
                TokenType.ELSE,
                TokenType.FOR,
                TokenType.WHILE,
                TokenType.SWITCH,
                TokenType.CASE,
                TokenType.OTHERWISE,
                TokenType.TRY,
                TokenType.CATCH,
                TokenType.RETURN,
                TokenType.BREAK,
                TokenType.CONTINUE,
                TokenType.GLOBAL,
                TokenType.PERSISTENT,
            }
        )
        if (
            self._check(TokenType.IDENTIFIER)
            and self.pos + 1 < len(self.tokens)
            and self.tokens[self.pos + 1].type in _CMD_TRIGGER_TYPES
        ):
            name = self._peek().value
            self._advance()  # consume function name
            args = []
            # Collect all arguments until end of line
            # In MATLAB command syntax, everything after the function name is treated as string arguments
            while not self._check(TokenType.NEWLINE) and not self._check(TokenType.EOF):
                if self._check(TokenType.SEMICOLON):
                    self._advance()
                    break
                # Collect tokens for this argument - collect ALL non-delimiter tokens
                arg_parts = []
                while (
                    not self._check(TokenType.NEWLINE)
                    and not self._check(TokenType.EOF)
                    and not self._check(TokenType.SEMICOLON)
                    and not self._check(TokenType.COMMA)
                ):
                    tok = self._advance()
                    arg_parts.append(str(tok.value))
                if arg_parts:
                    args.append(StringLiteral(value=" ".join(arg_parts)))
            self._skip_newlines()
            return [ExprStmt(expr=FuncCallExpr(name=name, args=args))]

        # Parse as expression first, then check for assignment
        save = self.pos
        expr = self._parse_expr()

        # Check if this is an indexed assignment: f(1) = expr or A(i,j) = expr
        if self._check(TokenType.ASSIGN):
            # Verify the expression is a valid assignment target
            if isinstance(expr, (FuncCallExpr, IndexExpr, FieldAccess)):
                self._advance()  # consume =
                value = self._parse_expr()
                self._consume_stmt_end()
                # Convert FuncCallExpr to IndexExpr for assignment
                if isinstance(expr, FuncCallExpr):
                    target = IndexExpr(base=Identifier(expr.name), indices=expr.args)
                else:
                    target = expr
                return [Assignment(targets=[target], value=value)]

        self._consume_stmt_end()
        return [ExprStmt(expr=expr)]

    # ── expression parsing ───────────────────────────────────────

    def _parse_expr(self) -> Expr:
        return self._parse_or_expr()

    def _parse_or_expr(self) -> Expr:
        left = self._parse_and_expr()
        while self._match(TokenType.PIPE_PIPE):
            right = self._parse_and_expr()
            left = BinaryOp(op="||", left=left, right=right)
        return left

    def _parse_and_expr(self) -> Expr:
        left = self._parse_bitor_expr()
        while self._match(TokenType.AMP_AMP):
            right = self._parse_bitor_expr()
            left = BinaryOp(op="&&", left=left, right=right)
        return left

    def _parse_bitor_expr(self) -> Expr:
        left = self._parse_bitand_expr()
        while self._match(TokenType.PIPE):
            right = self._parse_bitand_expr()
            left = BinaryOp(op="|", left=left, right=right)
        return left

    def _parse_bitand_expr(self) -> Expr:
        left = self._parse_comparison()
        while self._match(TokenType.AMP):
            right = self._parse_comparison()
            left = BinaryOp(op="&", left=left, right=right)
        return left

    def _parse_comparison(self) -> Expr:
        left = self._parse_colon_expr()
        for tt in (
            TokenType.EQ,
            TokenType.NEQ,
            TokenType.LT,
            TokenType.GT,
            TokenType.LE,
            TokenType.GE,
        ):
            if self._match(tt):
                right = self._parse_colon_expr()
                op_map = {
                    TokenType.EQ: "==",
                    TokenType.NEQ: "~=",
                    TokenType.LT: "<",
                    TokenType.GT: ">",
                    TokenType.LE: "<=",
                    TokenType.GE: ">=",
                }
                return BinaryOp(op=op_map[tt], left=left, right=right)
        return left

    def _parse_colon_expr(self) -> Expr:
        left = self._parse_add_expr()
        if self._match(TokenType.COLON):
            mid = self._parse_add_expr()
            if self._match(TokenType.COLON):
                stop = self._parse_add_expr()
                return RangeExpr(start=left, step=mid, stop=stop)
            return RangeExpr(start=left, stop=mid)
        return left

    def _parse_add_expr(self) -> Expr:
        left = self._parse_mul_expr()
        while self._peek().type in (TokenType.PLUS, TokenType.MINUS):
            op_tok = self._advance()
            right = self._parse_mul_expr()
            left = BinaryOp(op=op_tok.value, left=left, right=right)
        return left

    def _parse_mul_expr(self) -> Expr:
        left = self._parse_unary()
        while self._peek().type in (
            TokenType.STAR,
            TokenType.SLASH,
            TokenType.BACKSLASH,
            TokenType.DOT_STAR,
            TokenType.DOT_SLASH,
            TokenType.DOT_BACKSLASH,
            TokenType.CARET,
            TokenType.DOT_CARET,
        ):
            op_tok = self._advance()
            right = self._parse_unary()
            left = BinaryOp(op=op_tok.value, left=left, right=right)
        return left

    def _parse_unary(self) -> Expr:
        if self._peek().type in (TokenType.PLUS, TokenType.MINUS, TokenType.TILDE):
            op_tok = self._advance()
            operand = self._parse_unary()
            return UnaryOp(op=op_tok.value, operand=operand)
        return self._parse_postfix()

    def _parse_postfix(self) -> Expr:
        expr = self._parse_primary()

        while True:
            if self._check(TokenType.LPAREN):
                self._advance()
                args: list[Expr] = []
                if not self._check(TokenType.RPAREN):
                    # Handle ':' as "all" in indexing context
                    if self._check(TokenType.COLON):
                        args.append(StringLiteral(value=":"))
                        self._advance()
                    else:
                        args.append(self._parse_expr())
                    while self._match(TokenType.COMMA):
                        if self._check(TokenType.COLON):
                            args.append(StringLiteral(value=":"))
                            self._advance()
                        else:
                            args.append(self._parse_expr())
                self._expect(TokenType.RPAREN)
                if isinstance(expr, Identifier):
                    expr = FuncCallExpr(name=expr.name, args=args)
                else:
                    expr = IndexExpr(base=expr, indices=args)

            elif self._check(TokenType.LBRACE):
                self._advance()
                args = []
                if not self._check(TokenType.RBRACE):
                    args.append(self._parse_expr())
                    while self._match(TokenType.COMMA):
                        args.append(self._parse_expr())
                self._expect(TokenType.RBRACE)
                expr = IndexExpr(base=expr, indices=args, is_cell=True)

            elif self._check(TokenType.DOT):
                self._advance()
                if self._check(TokenType.LPAREN):
                    self._advance()
                    field_expr = self._parse_expr()
                    self._expect(TokenType.RPAREN)
                    expr = DynamicFieldAccess(base=expr, field_expr=field_expr)
                elif self._check(TokenType.IDENTIFIER):
                    field_name = self._advance().value
                    if self._check(TokenType.LPAREN):
                        self._advance()
                        args = []
                        if not self._check(TokenType.RPAREN):
                            args.append(self._parse_expr())
                            while self._match(TokenType.COMMA):
                                args.append(self._parse_expr())
                        self._expect(TokenType.RPAREN)
                        expr = FuncCallExpr(name=field_name, args=[expr] + args)
                    else:
                        expr = FieldAccess(base=expr, field_name=field_name)
                else:
                    raise ParseError("Expected identifier after '.'", self._peek())

            elif self._check(TokenType.TRANSPOSE):
                self._advance()
                expr = FuncCallExpr(name="ctranspose", args=[expr])
            elif self._check(TokenType.DOT_TRANSPOSE):
                self._advance()
                expr = FuncCallExpr(name="transpose", args=[expr])
            else:
                break

        return expr

    def _parse_primary(self) -> Expr:
        tok = self._peek()
        match tok.type:
            case TokenType.NUMBER:
                self._advance()
                return NumberLiteral(value=tok.value)
            case TokenType.STRING:
                self._advance()
                return StringLiteral(value=tok.value)
            case TokenType.IDENTIFIER:
                self._advance()
                return Identifier(name=tok.value)
            case TokenType.LPAREN:
                self._advance()
                expr = self._parse_expr()
                self._expect(TokenType.RPAREN)
                return ParenExpr(expr=expr)
            case TokenType.LBRACKET:
                return self._parse_matrix_literal()
            case TokenType.LBRACE:
                return self._parse_cell_literal()
            case TokenType.AT:
                return self._parse_func_handle()
            case TokenType.END:
                self._advance()
                return Identifier(name="end")
            case TokenType.TILDE:
                self._advance()
                return Identifier(name="~")
            case _:
                raise ParseError(
                    f"Unexpected token: {tok.type.name} ({tok.value!r})",
                    tok,
                )

    def _parse_matrix_literal(self) -> MatrixLiteral:
        self._expect(TokenType.LBRACKET)
        rows: list[list[Expr]] = []
        current_row: list[Expr] = []

        while not self._check(TokenType.RBRACKET) and not self._check(TokenType.EOF):
            if self._check(TokenType.SEMICOLON):
                self._advance()
                if current_row:
                    rows.append(current_row)
                    current_row = []
            elif self._check(TokenType.NEWLINE):
                self._advance()
                if current_row:
                    rows.append(current_row)
                    current_row = []
            else:
                current_row.append(self._parse_expr())
                self._match(TokenType.COMMA)

        if current_row:
            rows.append(current_row)
        self._expect(TokenType.RBRACKET)
        return MatrixLiteral(rows=rows)

    def _parse_cell_literal(self) -> CellLiteral:
        self._expect(TokenType.LBRACE)
        rows: list[list[Expr]] = []
        current_row: list[Expr] = []

        while not self._check(TokenType.RBRACE) and not self._check(TokenType.EOF):
            if self._check(TokenType.SEMICOLON):
                self._advance()
                if current_row:
                    rows.append(current_row)
                    current_row = []
            elif self._check(TokenType.NEWLINE):
                self._advance()
                if current_row:
                    rows.append(current_row)
                    current_row = []
            else:
                current_row.append(self._parse_expr())
                self._match(TokenType.COMMA)

        if current_row:
            rows.append(current_row)
        self._expect(TokenType.RBRACE)
        return CellLiteral(rows=rows)

    def _parse_func_handle(self) -> Expr:
        self._expect(TokenType.AT)
        if self._check(TokenType.LPAREN):
            self._advance()
            params: list[str] = []
            if not self._check(TokenType.RPAREN):
                params.append(self._expect(TokenType.IDENTIFIER).value)
                while self._match(TokenType.COMMA):
                    params.append(self._expect(TokenType.IDENTIFIER).value)
            self._expect(TokenType.RPAREN)
            body = self._parse_expr()
            return AnonFuncExpr(params=params, body=body)
        else:
            name = self._expect(TokenType.IDENTIFIER).value
            return FuncHandle(name=name)
